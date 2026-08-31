"""M0/M1 execution primitives. There is deliberately no main-run command.

The journal makes a completed operation reusable, and an interrupted operation
ambiguous. Ambiguous paid work is never retried automatically. A completed update
contains its optimizer checkpoint before the next operation can begin.
"""
import datetime
import hashlib
import json
import math
import os
import time
from pathlib import Path

from protocol import ACQUISITION_UPDATES, LEARNING_RATES, REFERENCE_EVAL_STEPS


class AmbiguousOperation(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class Journal:
    """One append-only local execution record, not a scheduling framework."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        if self.path.exists():
            # A partial last line is an error, not permission to replay work.
            with self.path.open(encoding="utf-8") as handle:
                self.rows = [json.loads(line) for line in handle]
        self.pending = {}
        self.completed = {}
        for row in self.rows:
            key = row["operation"]
            if row["type"] == "inflight":
                if key in self.pending or key in self.completed:
                    raise AmbiguousOperation(f"duplicate operation: {key}")
                self.pending[key] = row
            elif row["type"] == "complete":
                if key not in self.pending:
                    raise AmbiguousOperation(f"completion has no start: {key}")
                self.pending.pop(key)
                self.completed[key] = row
            else:
                raise ValueError(f"unknown journal event: {row['type']}")

    def _append(self, row):
        row = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), **row}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows.append(row)
        return row

    def call(self, operation, inputs, function):
        digest = hashlib.sha256(canonical(inputs).encode()).hexdigest()
        if operation in self.completed:
            row = self.completed[operation]
            if row["inputs_sha256"] != digest:
                raise RuntimeError(f"resume contract changed: {operation}")
            return row["result"]
        if self.pending:
            raise AmbiguousOperation("unresolved operation(s): " + ", ".join(sorted(self.pending)))
        begun = self._append({"type": "inflight", "operation": operation, "inputs_sha256": digest})
        self.pending[operation] = begun
        started = time.monotonic()
        result = function()
        # Serializability/finite-value checks happen before acknowledging work.
        canonical(result)
        row = self._append({"type": "complete", "operation": operation,
                            "inputs_sha256": digest, "elapsed_seconds": time.monotonic() - started,
                            "result": result})
        self.completed[operation] = row
        self.pending.pop(operation)
        return result


def batch_at(rows, batch_size, step):
    if batch_size <= 0 or step < 1:
        raise ValueError("positive batch size and one-based update required")
    start = (step - 1) * batch_size
    batch = rows[start:start + batch_size]
    if len(batch) != batch_size:
        raise ValueError("frozen corpus is too short; cycling examples is not allowed")
    return batch


def reference_sweep(backend, origin, task, journal, evaluate):
    """Registered acquisition reference: 3 independent 120-update trajectories.

    `task` contains candidate, batch_size, reference token rows, and data_sha256.
    `evaluate(client, sampler_path, step)` returns gate, heldout and accounting.
    It must use frozen evaluation items/seeds, and does not evaluate protected IF.
    Backend numerical failures must be returned as valid=False with accounting;
    transport errors remain journal ambiguities, never scientific divergence.
    """
    from rules import acquisition_references

    candidate, batch_size = task["candidate"], task["batch_size"]
    if batch_size not in (16, 32, 64):
        raise ValueError("acquisition batch must be fixed at 16, 32 or 64")
    if len(task["reference"]) != ACQUISITION_UPDATES * batch_size:
        raise ValueError("reference corpus must contain exactly 120 frozen batches")
    trajectories = []
    for learning_rate in LEARNING_RATES:
        branch = f"reference/{candidate}/{learning_rate:g}"
        identity = {"branch": branch, "origin": origin, "learning_rate": learning_rate,
                    "batch_size": batch_size, "warmup_steps": 10, "updates": 120,
                    "eval_steps": list(REFERENCE_EVAL_STEPS), "data_sha256": task["data_sha256"]}
        trajectory = {"learning_rate": learning_rate, "points": []}
        state, sampler_path = origin["state_path"], origin["sampler_path"]
        client = None
        for step in range(ACQUISITION_UPDATES + 1):
            if step:
                operation = f"{branch}/update/{step:03d}"

                def update():
                    nonlocal client
                    if client is None:
                        client = backend.branch(state, resume=step > 1)
                    result = backend.train_step(client, batch_at(task["reference"], batch_size, step),
                                                learning_rate=learning_rate, step=step, warmup_steps=10)
                    if result.get("valid", True):
                        result = {**result, "checkpoint": backend.save(client, branch.replace("/", "-"), step=step, resume=True)}
                    return result

                result = journal.call(operation, {**identity, "step": step, "from": state}, update)
                if not result.get("valid", True):
                    trajectory["failure_step"] = step
                    trajectory["failure"] = result.get("failure", "numerically invalid")
                    break
                state = result["checkpoint"]["state_path"]
                sampler_path = result["checkpoint"]["sampler_path"]
                # If this update was loaded from the journal, no live client was
                # created. The next unevaluated/update operation restores its state.
            if step in REFERENCE_EVAL_STEPS:
                operation = f"{branch}/evaluate/{step:03d}"

                def measure():
                    nonlocal client
                    if client is None:
                        client = backend.branch(state, resume=step > 0)
                    return evaluate(client, sampler_path, step)

                result = journal.call(operation, {**identity, "step": step, "state": state,
                                                  "sampler_path": sampler_path}, measure)
                valid = result.get("valid", True) and all(
                    isinstance(result.get(key), (int, float)) and math.isfinite(result[key])
                    for key in ("gate", "heldout"))
                if not valid:
                    trajectory["failure_step"] = step
                    trajectory["failure"] = result.get("failure", "numerically invalid evaluation")
                    break
                trajectory["points"].append({"step": step, "gate": result["gate"],
                                              "heldout": result["heldout"], "valid": True})
        trajectories.append(trajectory)
    references = acquisition_references(trajectories)
    # This freeze is durable before a screening function can consume references.
    return journal.call(f"reference/{candidate}/freeze", {"trajectories": trajectories},
                        lambda: {"candidate": candidate, "trajectories": trajectories, "references": references})


def learning_phase(backend, origin, task, rows, arm, thresholds, journal, branch, evaluate):
    """One M1 learning phase; evaluate(kind, client, sampler, step) is explicit.

    kind=start: gate+heldout+IF; kind=update: gate+IF; kind=heldout: heldout.
    No IF score truncates a reference sweep; this separate phase uses §6/§7.
    """
    from rules import learning_decision
    identity = {"branch": branch, "origin": origin, "candidate": task["candidate"], "arm": arm,
                "learning_rate": task["learning_rate"], "batch_size": task["batch_size"],
                "data_sha256": task["data_sha256"], "thresholds": thresholds}
    state, sampler = origin["state_path"], origin["sampler_path"]
    client = None

    def measure(kind, step):
        nonlocal client
        if client is None:
            client = backend.branch(state, resume=step > 0)
        result = evaluate(kind, client, sampler, step)
        if result.get("valid", True) is False:
            raise RuntimeError("invalid learning measurement; no automatic retuning or replay")
        return result

    start = journal.call(f"{branch}/start", identity, lambda: measure("start", 0))
    points, target_tokens = [], 0
    for step in range(1, ACQUISITION_UPDATES + 1):
        def update():
            nonlocal client
            if client is None:
                client = backend.branch(state, resume=step > 1)
            result = backend.train_step(client, batch_at(rows, task["batch_size"], step),
                                        learning_rate=task["learning_rate"], step=step, warmup_steps=10)
            if result.get("valid", True):
                result = {**result, "checkpoint": backend.save(client, branch.replace("/", "-"), step=step, resume=True)}
            return result
        result = journal.call(f"{branch}/update/{step:03d}", {**identity, "step": step, "from": state}, update)
        if not result.get("valid", True):
            return {"status": "numerical_failure", "step": step, "result": result, "primary_eligible": False}
        target_tokens += result["accounting"]["gradient_target_tokens"]
        state, sampler = result["checkpoint"]["state_path"], result["checkpoint"]["sampler_path"]
        point = journal.call(f"{branch}/evaluate/{step:03d}", {**identity, "state": state, "step": step},
                             lambda: measure("update", step))
        points.append({**point, "step": step, "target_tokens": target_tokens})
        decision = learning_decision(arm, start, points, thresholds)
        if decision["stop"]:
            heldout = journal.call(f"{branch}/heldout/{step:03d}", {**identity, "state": state, "step": step},
                                   lambda: measure("heldout", step))
            points[-1]["heldout"] = heldout["heldout"]
            decision = learning_decision(arm, start, points, thresholds)

            def save_a():
                nonlocal client
                if client is None:
                    client = backend.branch(state, resume=True)
                return backend.save(client, branch.replace("/", "-") + "-A", step=step)

            checkpoint = journal.call(f"{branch}/A", {**identity, "state": state, "step": step}, save_a)
            return journal.call(f"{branch}/complete", {**identity, "checkpoint": checkpoint}, lambda: {
                "status": "complete", "A": checkpoint, "start": start, "points": points,
                "decision": decision, "primary_eligible": decision["primary_eligible"],
                "target_tokens": target_tokens})
    raise AssertionError("learning cap must terminate")


def repair_phase(backend, a, a_score, origin, previous_b, arm, cycle, task_slot,
                 prompt_tokens, thresholds, journal, branch, evaluate_if):
    """One repair or physical A/B alias. Prompt and rollout schedules are task-keyed."""
    import random
    from protocol import REPAIR_CAP, REPAIR_CHECK_EVERY, schedule_seed
    from rules import repair_decision
    teacher_checkpoint = origin if arm == "fixed" or cycle == 1 else previous_b
    if arm not in ("fixed", "rolling", "learn-only") or cycle < 1:
        raise ValueError("invalid repair arm or cycle")
    if arm != "learn-only" and teacher_checkpoint is None:
        raise ValueError("rolling repair requires the preceding B")
    identity = {"branch": branch, "A": a, "A_score": a_score, "teacher": teacher_checkpoint,
                "arm": arm, "cycle": cycle, "task": task_slot, "thresholds": thresholds,
                "pool_sha256": hashlib.sha256(canonical(prompt_tokens).encode()).hexdigest()}
    decision = repair_decision(arm, a_score, [], thresholds)
    if decision["stop"]:
        return journal.call(f"{branch}/complete", identity, lambda: {
            "B": {**a, "alias_of": "A"}, "decision": decision, "checks": [], "target_tokens": 0})
    if len(prompt_tokens) != 2000:
        raise ValueError("repair requires the frozen 2,000-prompt pool")
    state, sampler_path = a["state_path"], a["sampler_path"]
    client = teacher = None
    checks, target_tokens = [], 0
    for step in range(1, REPAIR_CAP + 1):
        def update():
            nonlocal client, teacher
            if client is None:
                client = backend.branch(state, resume=step > 1)
            if teacher is None:
                teacher = backend.sampler(teacher_checkpoint["sampler_path"])
            indices = random.Random(schedule_seed(task_slot, "repair-prompts", step)).sample(range(2000), 64)
            result = backend.repair_step(client, teacher, [prompt_tokens[i] for i in indices], step=step,
                                         seed=schedule_seed(task_slot, "repair-rollouts", step))
            return {**result, "checkpoint": backend.save(client, branch.replace("/", "-"), step=step, resume=True)}
        result = journal.call(f"{branch}/update/{step:03d}", {**identity, "step": step, "from": state}, update)
        target_tokens += result["accounting"]["gradient_target_tokens"]
        state, sampler_path = result["checkpoint"]["state_path"], result["checkpoint"]["sampler_path"]
        if step % REPAIR_CHECK_EVERY:
            continue
        checked = journal.call(f"{branch}/criterion/{step:03d}", {**identity, "step": step, "sampler": sampler_path},
                               lambda: evaluate_if(sampler_path, step))
        checks.append({**checked, "step": step})
        decision = repair_decision(arm, a_score, checks, thresholds)
        if decision["stop"]:
            def save_b():
                nonlocal client
                if client is None:
                    client = backend.branch(state, resume=True)
                return backend.save(client, branch.replace("/", "-") + "-B", step=step)
            checkpoint = journal.call(f"{branch}/B", {**identity, "step": step, "from": state}, save_b)
            return journal.call(f"{branch}/complete", {**identity, "B": checkpoint}, lambda: {
                "B": checkpoint, "decision": decision, "checks": checks, "target_tokens": target_tokens})
    raise AssertionError("repair cap must terminate")


def probe_sweep(backend, origin, probe, learning_rate, journal, branch, evaluate_loss, *, extended=False):
    """Fixed-budget fresh-optimizer probe, with explicit endpoint evaluation."""
    from protocol import PROBE_BUDGETS
    budget, cadence = PROBE_BUDGETS[probe["class"]]
    budget *= 2 if extended else 1
    eval_steps = sorted(set(range(0, budget + 1, cadence)) | {budget})
    if len(probe["train"]) < budget * probe["batch_size"]:
        raise ValueError("frozen probe corpus is too short")
    identity = {"branch": branch, "origin": origin, "probe": probe["candidate"], "learning_rate": learning_rate,
                "budget": budget, "eval_steps": eval_steps, "batch_size": probe["batch_size"],
                "data_sha256": probe["data_sha256"], "warmup_steps": 0}
    state, client = origin["state_path"], None
    curve = []
    for step in range(budget + 1):
        if step:
            def update():
                nonlocal client
                if client is None:
                    client = backend.branch(state, resume=step > 1)
                result = backend.train_step(client, batch_at(probe["train"], probe["batch_size"], step),
                                            learning_rate=learning_rate, step=step, warmup_steps=0)
                if result.get("valid", True):
                    result = {**result, "checkpoint": backend.save(client, branch.replace("/", "-"), step=step, resume=True)}
                return result
            result = journal.call(f"{branch}/update/{step:03d}", {**identity, "step": step, "from": state}, update)
            if not result.get("valid", True):
                return {"status": "numerical_failure", "failure_step": step, "curve": curve}
            state = result["checkpoint"]["state_path"]
        if step in eval_steps:
            def measure():
                nonlocal client
                if client is None:
                    client = backend.branch(state, resume=step > 0)
                return evaluate_loss(client, step)
            result = journal.call(f"{branch}/evaluate/{step:03d}", {**identity, "state": state, "step": step}, measure)
            if not result.get("valid", True):
                return {"status": "numerical_failure", "failure_step": step, "curve": curve}
            curve.append({"step": step, "loss": result["nll"]})
    return journal.call(f"{branch}/complete", identity, lambda: {"status": "complete", "curve": curve,
                        "steps": [point["step"] for point in curve], "losses": [point["loss"] for point in curve]})
