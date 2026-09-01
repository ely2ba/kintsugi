"""M0/M1 execution primitives. There is deliberately no main-run command.

The journal reuses completed operations. Only explicitly marked, durably cached
sampling evaluations can resume; interrupted updates remain ambiguous. A
completed update contains its optimizer checkpoint before the next operation.
"""
import datetime
import hashlib
import json
import math
import os
import time
from pathlib import Path

import eval_cache
from protocol import ACQUISITION_UPDATES, LEARNING_RATES, REFERENCE_EVAL_STEPS


class AmbiguousOperation(RuntimeError):
    pass


class InvalidMeasurement(RuntimeError):
    """A scientific evaluation failure, not a transport retry opportunity."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _sampling_evaluation(operation):
    """Constrain recovery markers to the existing read-only sampling call sites."""
    parts = operation.split("/")
    if parts[0] == "probe" or any(part in ("update", "origin", "download", "A", "B") for part in parts):
        return False
    return (operation in ("m1/cycle0/criterion", "closeout/cycle0/if-heldout", "closeout/kl-reference")
            or parts[-1] == "start"
            or (len(parts) > 1 and parts[-2] in ("evaluate", "heldout", "criterion", "realization", "if-heldout", "diversity"))
            or operation.startswith("closeout/native/"))


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
        self.attempts = {}
        for row in self.rows:
            self._accept(row)

    def _accept(self, row):
        canonical(row)
        key, event = row["operation"], row["type"]
        if event == "inflight":
            if self.pending or key in self.completed:
                raise AmbiguousOperation(f"duplicate or overlapping operation: {key}")
            if "recoverable" in row and (row["recoverable"] is not True or not _sampling_evaluation(key)):
                raise AmbiguousOperation(f"invalid evaluation recovery marker: {key}")
            self.pending[key] = row
            self.attempts[key] = {"attempt": 1, "active": True, "elapsed_seconds": 0.0,
                                  "timing_complete": True, "recoverable": row.get("recoverable", False),
                                  "authorized": False, "blocked": False, "last_failure_type": None}
            return
        if event == "implementation_revision":
            revision = row["result"]
            expected = {"test_commit", "freeze_sha256", "identity_freeze_sha256",
                        "resume_from_freeze_commit", "original_test_commit"}
            identity = self.completed.get("m1/identity", {}).get("result", {})
            pending = self.pending.get(row.get("pending_operation"))
            if (set(revision) != expected or key != f"m1/implementation/{revision['freeze_sha256']}"
                    or key in self.completed or pending is None
                    or row.get("pending_inputs_sha256") != pending["inputs_sha256"]
                    or row["inputs_sha256"] != hashlib.sha256(canonical(revision).encode()).hexdigest()
                    or revision["identity_freeze_sha256"] != identity.get("freeze_sha256")
                    or "m1/origin" not in self.completed or not self.recoverable_pending()):
                raise AmbiguousOperation(f"invalid local implementation revision: {key}")
            self.completed[key] = row
            return
        if event not in ("complete", "failed", "resume", "recovery_authorized"):
            raise ValueError(f"unknown journal event: {event}")
        if key not in self.pending:
            raise AmbiguousOperation(f"{event} has no start: {key}")
        if row["inputs_sha256"] != self.pending[key]["inputs_sha256"]:
            raise AmbiguousOperation(f"{event} input hash mismatch: {key}")
        attempt = self.attempts[key]
        if event == "recovery_authorized":
            blocked_setup_recovery = (attempt["blocked"] and attempt["recoverable"]
                                      and not attempt["active"]
                                      and row.get("failed_attempt") == attempt["attempt"]
                                      and row.get("failed_error_type") == attempt["last_failure_type"]
                                      and attempt["last_failure_type"] == "APIConnectionError")
            legacy_interruption_recovery = not attempt["blocked"]
            if (not _sampling_evaluation(key) or row.get("recoverable") is not True
                    or not isinstance(row.get("reason"), str) or not row["reason"].strip()
                    or attempt["authorized"]
                    or not (legacy_interruption_recovery or blocked_setup_recovery)):
                raise AmbiguousOperation(f"invalid evaluation recovery authorization: {key}")
            attempt.update(recoverable=True, authorized=True, blocked=False)
        elif event == "resume":
            previous_status = "interrupted" if attempt["active"] else "failed"
            if (not attempt["recoverable"] or attempt["blocked"]
                    or row.get("attempt") != attempt["attempt"] + 1
                    or row.get("previous_attempt_status") != previous_status):
                raise AmbiguousOperation(f"invalid evaluation resume: {key}")
            if attempt["active"]:
                attempt["timing_complete"] = False
            attempt.update(attempt=row["attempt"], active=True)
        elif event == "failed":
            elapsed = row.get("elapsed_seconds")
            if (not attempt["active"] or row.get("attempt") != attempt["attempt"]
                    or row.get("attempt_status") != "failed" or not isinstance(row.get("retryable"), bool)
                    or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0
                    or (row["retryable"] and not attempt["recoverable"])):
                raise AmbiguousOperation(f"invalid failed attempt: {key}")
            attempt["elapsed_seconds"] += elapsed
            attempt.update(active=False, blocked=not row["retryable"],
                           last_failure_type=row.get("error_type"))
        else:
            if "attempt" in row and row["attempt"] != attempt["attempt"]:
                raise AmbiguousOperation(f"completion attempt mismatch: {key}")
            self.pending.pop(key)
            self.attempts.pop(key)
            self.completed[key] = row

    def recoverable_pending(self):
        """Only a single prospectively marked or explicitly authorized evaluation."""
        if len(self.pending) != 1:
            return False
        key = next(iter(self.pending))
        attempt = self.attempts[key]
        return _sampling_evaluation(key) and attempt["recoverable"] and not attempt["blocked"]

    def _append(self, row):
        row = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), **row}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows.append(row)
        return row

    def _record(self, row):
        self._accept(self._append(row))

    def call(self, operation, inputs, function, *, recoverable=False):
        digest = hashlib.sha256(canonical(inputs).encode()).hexdigest()
        if operation in self.completed:
            row = self.completed[operation]
            if row["inputs_sha256"] != digest:
                raise RuntimeError(f"resume contract changed: {operation}")
            return row["result"]
        if recoverable and not _sampling_evaluation(operation):
            raise AmbiguousOperation(f"not a recoverable sampling evaluation: {operation}")
        if self.pending:
            if operation in self.pending and self.pending[operation]["inputs_sha256"] != digest:
                raise RuntimeError(f"resume contract changed: {operation}")
            if operation not in self.pending or not recoverable or not self.recoverable_pending():
                raise AmbiguousOperation("unresolved operation(s): " + ", ".join(sorted(self.pending)))
            attempt = self.attempts[operation]
            self._record({"type": "resume", "operation": operation, "inputs_sha256": digest,
                          "attempt": attempt["attempt"] + 1, "attempt_status": "inflight",
                          "previous_attempt_status": "interrupted" if attempt["active"] else "failed"})
        else:
            marker = {"recoverable": True} if recoverable else {}
            self._record({"type": "inflight", "operation": operation, "inputs_sha256": digest, **marker})
        attempt = self.attempts[operation]
        started = time.monotonic()
        validating = False
        try:
            with eval_cache.evaluation_scope(self.path.parent / "evaluations", operation, digest):
                result = function()
            # Invalid scientific values are never made retryable by the cache.
            validating = True
            canonical(result)
        except Exception as error:
            retryable = recoverable and not validating and isinstance(error, eval_cache.SamplingTransportError)
            self._record({"type": "failed", "operation": operation, "inputs_sha256": digest,
                          "attempt": attempt["attempt"], "attempt_status": "failed",
                          "elapsed_seconds": time.monotonic() - started,
                          "retryable": retryable, "error_type": type(error).__name__})
            raise
        elapsed = time.monotonic() - started
        self._record({"type": "complete", "operation": operation, "inputs_sha256": digest,
                      "attempt": attempt["attempt"], "attempt_status": "complete",
                      "attempt_elapsed_seconds": elapsed, "elapsed_seconds": attempt["elapsed_seconds"] + elapsed,
                      "timing_complete": attempt["timing_complete"], "result": result})
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
                        name = branch.replace("/", "-")
                        checkpoint = (backend.save(client, name, step=step, resume=True)
                                      if step in REFERENCE_EVAL_STEPS
                                      else backend.save_state(client, name, step=step))
                        result = {**result, "checkpoint": checkpoint}
                    return result

                result = journal.call(operation, {**identity, "step": step, "from": state}, update)
                if not result.get("valid", True):
                    trajectory["failure_step"] = step
                    trajectory["failure"] = result.get("failure", "numerically invalid")
                    break
                state = result["checkpoint"]["state_path"]
                sampler_path = result["checkpoint"].get("sampler_path", sampler_path)
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
                                                  "sampler_path": sampler_path}, measure,
                                      recoverable=task.get("metric") == "verifier_success")
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
            raise InvalidMeasurement("invalid learning measurement; no automatic retuning or replay")
        return result

    recoverable = task.get("metric") == "verifier_success"
    start = journal.call(f"{branch}/start", identity, lambda: measure("start", 0), recoverable=recoverable)
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
                             lambda: measure("update", step), recoverable=recoverable)
        points.append({**point, "step": step, "target_tokens": target_tokens})
        decision = learning_decision(arm, start, points, thresholds)
        if decision["stop"]:
            heldout = journal.call(f"{branch}/heldout/{step:03d}", {**identity, "state": state, "step": step},
                                   lambda: measure("heldout", step), recoverable=recoverable)
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
                               lambda: evaluate_if(sampler_path, step), recoverable=True)
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
                    result = {**result, "checkpoint": backend.save_state(
                        client, branch.replace("/", "-"), step=step)}
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
