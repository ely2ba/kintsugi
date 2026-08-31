"""Explicit M1 calibration driver. There is no M2 command.

Includes screening, persistence, probe selection/stability, diversity, physical
checkpoint measurements, and a measured twelve-lineage projection. Successful
execution stops with a launch packet requiring publication and M2 authorization.
Every remote operation is enclosed by the existing append-only M1 journal.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from backend import Backend, ACCOUNTING_KEYS
import calibrate
import data
import if_suite
import measure
from protocol import (CHECKPOINT_TTL, IF_HASHES, LANGUAGE_PROBES, LEARNING_RATES,
                      LORA_SEED, MODEL, PERSISTENCE_TASKS, STRUCTURED_PROBES,
                      PROBE_BUDGETS, REPAIR_POOL_HASH, TASK_SLOTS, TOKENIZER_REVISION, schedule_seed)
import rules


REMAINING_M1 = ["execute the registered calibration and publish its frozen launch packet"]
M1_RUNNER_COMPLETE = True


def _read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _local(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("freeze paths must be relative and contained in this repository")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("freeze path escapes the repository")
    return resolved


def validate_measurement_manifest(manifest):
    """No default noise construction: absent/pending methods forbid paid work."""
    if type(manifest.get("noise_repeats")) is not int or manifest["noise_repeats"] != 3:
        raise ValueError("measurement manifest must register exactly three independent complete realizations")
    bound = manifest.get("paired_noise_bound", {})
    if not isinstance(bound, dict) or bound.get("method") != "v1_operational_2_5_sd":
        raise ValueError("paired-noise construction is absent, pending, or not implemented; no paid launch")
    if "multiplier" in bound and bound["multiplier"] != 2.5:
        raise ValueError("the registered operational noise multiplier is 2.5")
    registered = {"single_checkpoint_sd_conversion": "sqrt(2)", "direct_null_difference_sd_conversion": 1.0,
                  "sample_sd_ddof": 1, "all_three_probe_clocks_must_be_defined_and_bracketed": True}
    if any(key in bound and bound[key] != value for key, value in registered.items()):
        raise ValueError("noise construction differs from the user-registered operational rule")
    deterministic = manifest.get("deterministic_bounds", {})
    if any(value != 0.0 for value in deterministic.values()):
        raise ValueError("registered deterministic fixed-checkpoint measurement bounds are zero")
    expected_kl = {"prompt_count": 32, "samples": 1, "max_tokens": 512,
                   "temperature": 1.0, "seed": 20260831}
    if not isinstance(manifest.get("kl"), dict) or any(manifest["kl"].get(key) != value for key, value in expected_kl.items()):
        raise ValueError("frozen cycle-0 trajectory recipe is absent or different")
    return manifest


def preflight(root):
    """Read-only local freeze/data/test-commit verification; never connects."""
    root = Path(root).resolve()
    freeze_path = root / "manifests/freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("completed manifests/freeze.json is required before any backend connection")
    frozen = _read(freeze_path)
    commit = frozen.get("test_commit", "")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("freeze must name the full tested code commit")
    code = frozen.get("code_hashes", {})
    required_code = {"SPEC.md", "protocol.py", "backend.py", "calibrate.py", "data.py",
                     "measure.py", "rules.py", "tasks.py", "probes.py", "if_suite.py", "m1.py",
                     "requirements.txt", "tests/test_m1.py"}
    required_code.update(str(path.relative_to(root)) for path in root.glob("*.py"))
    required_code.update(str(path.relative_to(root)) for path in (root / "tests").rglob("*.py"))
    if not isinstance(code, dict) or not required_code <= code.keys():
        raise RuntimeError("freeze lacks required executable code/test commitments")
    for relative, digest in code.items():
        if data.sha256_file(_local(root, relative)) != digest:
            raise RuntimeError(f"code differs from completed freeze: {relative}")
    tracked = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit], cwd=root,
                             capture_output=True, text=True, check=False)
    if tracked.returncode or not set(code) <= set(tracked.stdout.splitlines()):
        raise RuntimeError("tested commit does not contain all committed code files")
    unchanged = subprocess.run(["git", "diff", "--quiet", commit, "--", *sorted(code)],
                               cwd=root, capture_output=True, check=False)
    if unchanged.returncode:
        raise RuntimeError("code differs from the tested commit")
    manifests = frozen.get("manifests", {})
    actual = {str(path.relative_to(root)) for path in (root / "manifests").rglob("*.json")
              if path != freeze_path}
    required_manifests = {"manifests/shared.json", "manifests/measurement.json", "manifests/sources.json"}
    required_manifests.update(f"manifests/tasks/{candidate}.json" for pair in TASK_SLOTS.values() for candidate in pair)
    required_manifests.update(f"manifests/probes/{candidate}.json" for candidate in STRUCTURED_PROBES + LANGUAGE_PROBES)
    required_manifests.update(f"manifests/diversity/{candidate}.json" for candidate in ("graph_coloring", "set_partition"))
    required_manifests.update(f"manifests/orders/O{order}.json" for order in range(1, 5))
    if not isinstance(manifests, dict) or set(manifests) != actual or not required_manifests <= actual:
        raise RuntimeError("freeze must commit the complete actual acquisition/probe/diversity/measurement manifest set")
    for relative, digest in manifests.items():
        path = _local(root, relative)
        if data.sha256_file(path) != digest:
            raise RuntimeError(f"manifest differs from completed freeze: {relative}")
        manifest = _read(path)
        if "tokenizer" in manifest and (manifest["tokenizer"] != MODEL
                                        or manifest.get("tokenizer_revision") != TOKENIZER_REVISION):
            raise RuntimeError("manifest tokenizer differs from the registered model/revision")
        if relative.startswith("manifests/tasks/"):
            needed = {"reference", "screen1", "screen2", "main", "persistence", "gate", "heldout"}
            if (not needed <= manifest.get("splits", {}).keys() or manifest.get("batch_size") not in (16, 32, 64)
                    or manifest.get("candidate") != path.stem
                    or manifest.get("metric") not in ("negative_nll", "verifier_success")):
                raise RuntimeError("incomplete acquisition candidate manifest")
            if any(manifest["splits"][name]["examples"] != 120 * manifest["batch_size"]
                   for name in ("reference", "screen1", "screen2", "main", "persistence")):
                raise RuntimeError("acquisition corpus does not contain the registered 120 fixed batches")
        if relative.startswith("manifests/probes/"):
            if not {"train", "val"} <= manifest.get("splits", {}).keys() or manifest.get("candidate") != path.stem:
                raise RuntimeError("incomplete probe manifest")
        records = list(manifest.get("splits", {}).values())
        if "path" in manifest and "sha256" in manifest:
            records.append(manifest)
        for record in records:
            row_path = _local(root, record["path"])
            if data.sha256_file(row_path) != record["sha256"]:
                raise RuntimeError(f"frozen token corpus differs: {record['path']}")
            with row_path.open(encoding="utf-8") as handle:
                count = sum(1 for _ in handle)
            if count != record["examples"]:
                raise RuntimeError("frozen corpus row count differs")
    shared = _read(root / "manifests/shared.json")
    if shared.get("if_hashes") != IF_HASHES or if_suite.manifest_hashes() != IF_HASHES:
        raise RuntimeError("protected IF suite differs from its frozen v1 hash")
    if shared.get("repair_pool", {}).get("pool_sha256") != REPAIR_POOL_HASH:
        raise RuntimeError("repair-pool identity differs from the registered v1 pool")
    if data.sha256_file(root / "data/repair_pool.jsonl") != shared.get("repair_pool_file_sha256"):
        raise RuntimeError("repair-pool file differs from frozen commitment")
    measurement = validate_measurement_manifest(_read(root / "manifests/measurement.json"))
    allowed = M1_RUNNER_COMPLETE and frozen.get("m1_runner_complete") is True
    return {"status": "ready_for_m1" if allowed else "blocked_incomplete_m1_runner",
            "paid_launch_allowed": allowed, "freeze_sha256": data.sha256_file(freeze_path),
            "test_commit": commit, "measurement": measurement, "m1_complete": False,
            "remaining_m1": REMAINING_M1}


def _usage(*results):
    return {key: sum(result.get("accounting", {}).get(key, 0) for result in results)
            for key in ACCOUNTING_KEYS}


def load_task(root, candidate):
    manifest = _read(root / f"manifests/tasks/{candidate}.json")
    splits = {name: data.load_rows(_local(root, record["path"]))
              for name, record in manifest["splits"].items()}
    return {**manifest, **splits, "manifest": manifest,
            "data_sha256": hashlib.sha256(calibrate.canonical(manifest["splits"]).encode()).hexdigest()}


def _task_metric(api, client, sampler_path, task, split, repeat=0):
    return measure.evaluate_task(api, client, sampler_path, task[split], task["manifest"],
                                 schedule_seed(task["slot"], f"evaluate-{split}", repeat))


def _reference_evaluator(api, task):
    def evaluate(client, sampler_path, step):
        gate = _task_metric(api, client, sampler_path, task, "gate")
        if not gate["valid"]:
            return {"valid": False, "gate": None, "heldout": None, "accounting": gate["accounting"]}
        heldout = _task_metric(api, client, sampler_path, task, "heldout")
        return {"valid": heldout["valid"], "gate": gate["q"], "heldout": heldout["q"],
                "accounting": _usage(gate, heldout)}
    return evaluate


def _learning_evaluator(api, task):
    def evaluate(kind, client, sampler_path, step):
        if kind == "heldout":
            heldout = _task_metric(api, client, sampler_path, task, "heldout")
            return {"valid": heldout["valid"], "heldout": heldout["q"], "accounting": heldout["accounting"]}
        gate = _task_metric(api, client, sampler_path, task, "gate")
        protected = measure.evaluate_if(api, sampler_path, "criterion",
                                        schedule_seed(task["slot"], "evaluate-if"))
        result = {"valid": gate["valid"], "gate": gate["q"], "if_score": protected["if_score"],
                  "accounting": _usage(gate, protected)}
        if kind == "start":
            heldout = _task_metric(api, client, sampler_path, task, "heldout")
            result.update(valid=result["valid"] and heldout["valid"], heldout=heldout["q"],
                          accounting=_usage(result, heldout))
        return result
    return evaluate


def task_noise(api, origin, task, journal, repeats):
    """User-registered deterministic fixed-checkpoint score noise: exactly zero."""
    return journal.call(f"noise/task/{task['candidate']}/deterministic", {"origin": origin,
                        "task": task["data_sha256"], "method": "v1_operational_2_5_sd"},
                        lambda: {"valid": True, "sample_sd": 0.0, "bound": 0.0,
                                 "source": "deterministic_fixed_checkpoint", "paid_repetitions": 0})


def run_event(api, origin, start, previous_b, task, split, learning_rate, arm,
              cycle, thresholds, repair_prompts, journal, branch):
    recipe = {**task, "learning_rate": learning_rate}
    learned = calibrate.learning_phase(api, start, recipe, task[split], arm, thresholds,
                                       journal, branch + "/learn", _learning_evaluator(api, task))
    if learned.get("status") == "numerical_failure":
        return {"branch": branch, "primary_eligible": False, "repair_success": False,
                "learning": learned, "target_tokens": None, "start_checkpoint": start}
    a_score = learned["decision"]["checkpoint"]["if_score"]
    repaired = calibrate.repair_phase(
        api, learned["A"], a_score, origin, previous_b, arm, cycle, task["slot"],
        repair_prompts, thresholds, journal, branch + "/repair",
        lambda path, step: measure.evaluate_if(api, path, "criterion",
                                               schedule_seed(task["slot"], "evaluate-if")),
    )
    result = {"branch": branch, "candidate": task["candidate"], "slot": task["slot"],
              "arm": arm, "cycle": cycle, "learning_rate": learning_rate,
              "start_checkpoint": start,
              "primary_eligible": learned["primary_eligible"],
              "repair_success": repaired["decision"]["repair_success"] is True,
              "target_tokens": learned["target_tokens"], "A": learned["A"], "B": repaired["B"],
              "learning": learned, "repair": repaired}
    return journal.call(branch + "/event", {"learning": learned, "repair": repaired}, lambda: result)


def screen_candidate(api, origin, task, if_limits, repair_prompts, journal, noise_repeats):
    reference = calibrate.reference_sweep(api, origin, task, journal, _reference_evaluator(api, task))
    refs = reference["references"]
    if refs["status"] != "defined":
        return {"candidate": task["candidate"], "selected": None, "failure": refs["status"], "reference": reference}
    noise = task_noise(api, origin, task, journal, noise_repeats)
    if not noise["valid"]:
        return {"candidate": task["candidate"], "selected": None, "failure": noise["failure"], "reference": reference}
    if refs["gate_ref"] == refs["gate0"] and noise["sample_sd"] == 0:
        result = {"status": "contract_attention_needed", "candidate": task["candidate"], "selected": None,
                  "issue": "Zero reference gain and zero measured noise give M=0 under §5, while §13.1 requires meaningful headroom; no failure classification or backup is authorized.",
                  "reference": reference, "noise": noise, "m1_complete": False}
        return journal.call(f"screen/{task['candidate']}/contract-attention", {"reference": reference, "noise": noise}, lambda: result)
    thresholds = {**if_limits, **rules.acquisition_thresholds(
        refs["gate0"], refs["gate_ref"], refs["heldout0"], refs["heldout_ref"], noise["sample_sd"])}
    thresholds = journal.call(f"screen/{task['candidate']}/thresholds", {"reference": reference, "noise": noise,
                              "if_limits": if_limits}, lambda: thresholds)
    recipes = []
    for learning_rate in LEARNING_RATES:
        realizations = []
        for realization in (1, 2):
            event = run_event(api, origin, origin, None, task, f"screen{realization}", learning_rate,
                              "fixed", 1, thresholds, repair_prompts, journal,
                              f"screen/{task['candidate']}/{learning_rate:g}/{realization}")
            realizations.append({"realization": realization, **event})
        recipes.append({"learning_rate": learning_rate, "batch_size": task["batch_size"],
                        "realizations": realizations})
    selected = rules.select_task_recipe(recipes)
    result = {"candidate": task["candidate"], "slot": task["slot"], "selected": selected,
              "thresholds": thresholds, "reference": reference, "noise": noise, "recipes": recipes}
    return journal.call(f"screen/{task['candidate']}/selection", {"recipes": recipes, "thresholds": thresholds}, lambda: result)


def screen_tasks(api, origin, root, if_limits, repair_prompts, journal, noise_repeats):
    selected, attempted = {}, []
    for slot, candidates in TASK_SLOTS.items():
        for candidate in candidates:
            result = screen_candidate(api, origin, load_task(root, candidate), if_limits,
                                      repair_prompts, journal, noise_repeats)
            attempted.append(result)
            if result.get("status") == "contract_attention_needed":
                return {**result, "slot": slot, "selected": selected, "attempted": attempted}
            if result["selected"] is not None:
                selected[slot] = result
                break
        if slot not in selected:
            return {"status": "m1_failed", "failure": "both_task_candidates_failed", "slot": slot,
                    "selected": selected, "attempted": attempted, "m1_complete": False}
    return {"status": "task_screening_complete", "selected": selected, "attempted": attempted}


def persistence(api, origin, root, selections, repair_prompts, journal):
    events = []

    def event(slot, cycle, arm, start, previous_b):
        selected = selections[slot]
        result = run_event(api, origin, start, previous_b, load_task(root, selected["candidate"]),
                           "persistence", selected["selected"]["learning_rate"], arm, cycle,
                           selected["thresholds"], repair_prompts, journal,
                           f"persistence/{'common' if cycle == 1 else arm}/{cycle}/{slot}")
        events.append(result)
        return result

    common = event(PERSISTENCE_TASKS[0], 1, "fixed", origin, None)
    if not (common["primary_eligible"] and common["repair_success"]):
        return {"status": "m1_failed", "failure": "persistence_event_failed", "events": events, "m1_complete": False}
    terminals = {}
    for arm in ("fixed", "rolling"):
        current = common["B"]
        for cycle, slot in enumerate(PERSISTENCE_TASKS[1:], 2):
            result = event(slot, cycle, arm, current, current)
            if not (result["primary_eligible"] and result["repair_success"]):
                return {"status": "m1_failed", "failure": "persistence_event_failed", "events": events, "m1_complete": False}
            current = result["B"]
        terminals[arm] = current
    return {"status": "persistence_manipulation_complete", "events": events, "terminals": terminals,
            "probe_gate_pending": True}


def state_panel(origin, selections, persistence_result):
    """Complete prospective selected-recipe panel; physical aliases deduplicate."""
    entries = [{"state": "cycle0", "checkpoint": origin}]
    events = [event for selected in selections.values() for event in selected["selected"]["realizations"]]
    events += persistence_result["events"]
    for event in events:
        for label in ("A", "B"):
            entries.append({"state": event["branch"] + "/" + label, "checkpoint": event[label]})
    physical = {}
    for entry in entries:
        path = entry["checkpoint"]["sampler_path"]
        if path not in physical:
            physical[path] = {**entry, "aliases": []}
        else:
            physical[path]["aliases"].append(entry["state"])
    return list(physical.values())


def load_probe(root, candidate):
    manifest = _read(root / f"manifests/probes/{candidate}.json")
    return {**manifest, "manifest": manifest,
            "train": data.load_rows(_local(root, manifest["splits"]["train"]["path"])),
            "val": data.load_rows(_local(root, manifest["splits"]["val"]["path"])),
            "data_sha256": hashlib.sha256(calibrate.canonical(manifest["splits"]).encode()).hexdigest()}


def calibrate_probes(api, origin, root, panel, journal, measurement):
    """All candidate/LR references and full prospective state-panel trajectories.

    Standard cycle-0 execution is separate from the extended reference, notably
    because language update 245 is not an extended-reference cadence point.
    The completed standard cycle-0 trace supplies noise replicate 1; two fresh
    disposable executions supply replicates 2/3, with identical data/recipe.
    """
    if measurement["noise_repeats"] != 3:
        raise ValueError("probe stability requires exactly three complete executions")
    names = [entry["state"] for entry in panel]
    physical = [entry["checkpoint"]["sampler_path"] for entry in panel]
    if (not panel or len(set(names)) != len(names) or len(set(physical)) != len(physical)
            or names.count("cycle0") != 1
            or next(entry["checkpoint"] for entry in panel if entry["state"] == "cycle0") != origin):
        raise ValueError("complete unique state panel must contain the physical cycle-0 origin")
    candidates, selections = [], {}
    for probe_class, order in (("structured", STRUCTURED_PROBES), ("language", LANGUAGE_PROBES)):
        class_results = []
        for candidate in order:
            probe = load_probe(root, candidate)
            if probe["class"] != probe_class:
                raise ValueError("probe candidate is assigned to the wrong class")
            budget, cadence = PROBE_BUDGETS[probe_class]
            for learning_rate in LEARNING_RATES:
                branch = f"probe/{candidate}/{learning_rate:g}"
                reference = calibrate.probe_sweep(
                    api, origin, probe, learning_rate, journal, branch + "/reference",
                    lambda client, step: measure.evaluate_probe_loss(api, client, probe["val"]), extended=True)
                if reference["status"] != "complete":
                    result = {"candidate": candidate, "learning_rate": learning_rate,
                              "passes": False, "failure": "invalid_reference_trajectory", "reference": reference}
                    candidates.append(result)
                    class_results.append(result)
                    continue
                target = rules.probe_reference_target(reference["steps"], reference["losses"], budget, cadence)["reference_target"]
                states, failed = [], []
                for entry in panel:
                    state_branch = branch + "/state/" + entry["state"]
                    curve = calibrate.probe_sweep(
                        api, entry["checkpoint"], probe, learning_rate, journal, state_branch,
                        lambda client, step: measure.evaluate_probe_loss(api, client, probe["val"]), extended=False)
                    if curve["status"] != "complete":
                        failed.append({"state": entry["state"], "curve": curve})
                    else:
                        states.append({"state": entry["state"], "branch": state_branch,
                                       "steps": curve["steps"], "losses": curve["losses"]})
                if failed:
                    result = {"candidate": candidate, "learning_rate": learning_rate, "passes": False,
                              "failure": "invalid_state_trajectory", "failed_states": failed,
                              "reference_target": target, "completed_states": states}
                else:
                    result = rules.m1_probe_candidate(candidate, learning_rate, states, target,
                                                       0.0, budget, cadence)
                    result.update(reference_target=target, eval_noise_sd=0.0, probe_class=probe_class,
                                  state_curves=states, standard_branches=[state["branch"] for state in states])
                    if result["passes"]:
                        cycle0 = next(state for state in states if state["state"] == "cycle0")
                        replicates = [rules.trainability_clocks(cycle0["steps"], cycle0["losses"], target,
                                                                 result["delta_l"], 0.0)]
                        replicate_branches = [branch + "/state/cycle0"]
                        for repeat in (1, 2):
                            repeat_branch = branch + f"/noise/{repeat}"
                            curve = calibrate.probe_sweep(
                                api, origin, probe, learning_rate, journal, repeat_branch,
                                lambda client, step: measure.evaluate_probe_loss(api, client, probe["val"]), extended=False)
                            replicate_branches.append(repeat_branch)
                            replicates.append(rules.trainability_clocks(
                                curve["steps"], curve["losses"], target, result["delta_l"], 0.0)
                                if curve["status"] == "complete" else
                                {"t50": None, "tdelta": None, "t50_status": "numerical_failure",
                                 "tdelta_status": "numerical_failure", "censor_step": budget})
                        noise = rules.probe_clock_noise(replicates)
                        result.update(clock_noise=noise, noise_replicate_branches=replicate_branches,
                                      noise_replicates=replicates, noise_bounds=noise["bounds"], passes=noise["passes"])
                        if not noise["passes"]:
                            result["failure"] = "unstable_cycle0_probe_clocks"
                result = journal.call(branch + "/calibrated", {"result": result}, lambda: result)
                candidates.append(result)
                class_results.append(result)
        selected = rules.select_probe(class_results, order)
        if selected is not None:
            selections[probe_class] = selected
    if set(selections) != {"structured", "language"}:
        return {"status": "m1_failed", "failure": "no_valid_probe_in_each_class", "m1_complete": False,
                "selected": selections, "candidates": candidates}
    result = {"status": "probe_calibration_complete", "selected": selections, "candidates": candidates,
              "deterministic_eval_noise_sd": 0.0, "retention_noise_bound": 0.0}
    return journal.call("m1/probes/freeze", {"candidates": candidates}, lambda: result)


def diversity_qualification(examples, result, manifest):
    """First frozen realization only; exact rational coverage boundary checks."""
    if not examples or len(result["results"]) != len(examples):
        raise ValueError("diversity qualification requires the complete frozen panel")
    count = len(examples)
    valid = sum(row["valid_outputs"] for row in result["results"])
    solved = sum(row["valid_outputs"] > 0 for row in result["results"])
    lengths = sorted(length for row in result["results"] for length in row["lengths"])
    if len(lengths) != 8 * count:
        raise ValueError("diversity qualification requires eight draws on every item")
    family_items = sum(example["family_count"] >= 4 for example in examples)
    rule = manifest["safe_length_rule"]
    p95 = lengths[(95 * len(lengths) + 99) // 100 - 1]
    checks = {"family_coverage": 5 * family_items >= 4 * count,
              "pass1_above_floor": 20 * valid > 8 * count,
              "pass1_below_ceiling": 5 * valid < 4 * 8 * count,
              "coverage_gap": 20 * (8 * solved - valid) >= 8 * count,
              "safe_length": p95 <= rule["p95_tokens_at_most"],
              "no_truncation": result["truncation_rate"] <= rule["max_truncation_rate"]}
    return {"passes": all(checks.values()), "checks": checks, "p95_completion_tokens": p95,
            "family_coverage": family_items / count, "qualification_realization": 0}


def calibrate_diversity(api, origin, root, journal, measurement):
    """First qualifying registered panel; three complete draws for its noise."""
    if measurement["noise_repeats"] != 3:
        raise ValueError("diversity noise requires three full-panel realizations")
    attempts = []
    for candidate in ("graph_coloring", "set_partition"):
        manifest = _read(root / f"manifests/diversity/{candidate}.json")
        examples = data.load_rows(_local(root, manifest["path"]))
        recipe = manifest["sampling"]
        if recipe["samples"] != 8 or recipe["temperature"] != 1.0:
            raise ValueError("diversity sampling differs from the frozen recipe")
        sampler = None

        def evaluate(repeat):
            nonlocal sampler
            if sampler is None:
                sampler = api.sampler(origin["sampler_path"])
            sampled = api.sample(sampler, [row["prompt_tokens"] for row in examples], samples=8,
                                  max_tokens=recipe["max_tokens"], temperature=recipe["temperature"],
                                  seed=recipe["seed"] + repeat * len(examples))
            return {**measure.diversity_summary(examples, sampled["groups"]),
                    "accounting": sampled["accounting"]}

        identity = {"origin": origin, "candidate": candidate, "manifest": manifest}
        first = journal.call(f"diversity/{candidate}/realization/0", {**identity, "repeat": 0}, lambda: evaluate(0))
        qualified = diversity_qualification(examples, first, manifest)
        if not qualified["passes"]:
            attempts.append({"candidate": candidate, "qualification": qualified, "first_realization": first})
            continue
        realizations = [first]
        for repeat in (1, 2):
            realizations.append(journal.call(f"diversity/{candidate}/realization/{repeat}",
                                             {**identity, "repeat": repeat}, lambda: evaluate(repeat)))
        noise = {}
        for metric in ("pass1", "pass8", "coverage_gap", "unique_valid_outputs", "strategy_families",
                       "strategy_family_concentration", "sampled_token_surprisal", "mean_completion_tokens", "truncation_rate"):
            values = [result[metric] for result in realizations]
            noise[metric] = (rules.measurement_noise_bound(values) if all(value is not None for value in values)
                             else {"status": "metric_unavailable", "bound": None, "count": 3})
        selected = {"candidate": candidate, "manifest": manifest, "qualification": qualified,
                    "realizations": realizations, "noise": noise,
                    "realization_branches": [f"diversity/{candidate}/realization/{repeat}" for repeat in (0, 1, 2)]}
        attempts.append(selected)
        result = {"status": "diversity_calibration_complete", "selected": selected, "attempts": attempts}
        return journal.call("m1/diversity/freeze", {"attempts": attempts}, lambda: result)
    return {"status": "m1_failed", "failure": "no_valid_diversity_panel", "attempts": attempts, "m1_complete": False}


def run(root, *, project_id, keychain_service):
    root = Path(root).resolve()
    checked = preflight(root)  # Must precede credentials, service creation and all paid work.
    journal = calibrate.Journal(root / "runs/m1/journal.jsonl")
    if journal.pending:
        raise calibrate.AmbiguousOperation("unresolved M1 operation; no paid call will be retried")
    identity = {"freeze_sha256": checked["freeze_sha256"],
                "project_sha256": hashlib.sha256(project_id.encode()).hexdigest(),
                "model": MODEL, "lora_seed": LORA_SEED}
    # Freeze identity locally before connecting; changed project/freeze cannot
    # attach to the same execution journal on a later invocation.
    journal.call("m1/identity", identity, lambda: identity)
    previous = journal.completed.get("m1/measurement-closeout")
    if previous:
        data.write_once(root / "runs/m1/launch_packet.json", previous["result"]["launch_packet"])
        return previous["result"]
    previous_failure = journal.completed.get("m1/stopped")
    if previous_failure:
        return previous_failure["result"]
    if not checked.get("paid_launch_allowed", False):
        raise RuntimeError("full M1 runner and its completed public freeze are required before paid launch; staged screening/persistence is not enough")
    api = Backend.connect(project_id, keychain_service, seed=LORA_SEED, checkpoint_ttl=CHECKPOINT_TTL)

    def create_origin():
        client = api.origin()
        saved = api.save(client, "m1-cycle0", step=0)
        downloaded = api.download_sampler(saved["sampler_path"], root / "runs/m1/adapters/cycle0")
        # download_sampler validates full layout/rank/alpha before this operation
        # can complete and before the first learning update can be reached.
        return {**saved, "download": downloaded}

    origin = journal.call("m1/origin", identity, create_origin)
    criterion = journal.call("m1/cycle0/criterion", {"origin": origin},
                              lambda: measure.evaluate_if(api, origin["sampler_path"], "criterion",
                                                          schedule_seed("T1", "evaluate-if")))
    if_limits = rules.if_thresholds(criterion["if_score"])
    if_limits = journal.call("m1/if-thresholds", {"criterion": criterion}, lambda: if_limits)
    pool = data.load_rows(root / "data/repair_pool.jsonl")
    if len(pool) != 2001:
        raise RuntimeError("frozen repair pool must contain its header and 2,000 prompts")
    repair_prompts = [data.render_repair_prompt(api.tokenizer, row["prompt"]) for row in pool[1:]]
    screening = screen_tasks(api, origin, root, if_limits, repair_prompts, journal,
                             checked["measurement"]["noise_repeats"])
    if screening["status"] != "task_screening_complete":
        return journal.call("m1/stopped", {"screening": screening}, lambda: screening)
    persisted = persistence(api, origin, root, screening["selected"], repair_prompts, journal)
    if persisted["status"] == "m1_failed":
        return journal.call("m1/stopped", {"persistence": persisted}, lambda: persisted)
    panel = state_panel(origin, screening["selected"], persisted)
    probes = calibrate_probes(api, origin, root, panel, journal, checked["measurement"])
    if probes["status"] == "m1_failed":
        return journal.call("m1/stopped", {"probes": probes}, lambda: probes)
    diversity = calibrate_diversity(api, origin, root, journal, checked["measurement"])
    if diversity["status"] == "m1_failed":
        return journal.call("m1/stopped", {"diversity": diversity}, lambda: diversity)
    result = {"status": "awaiting_drift_retention_cost_closeout", "m1_complete": False,
              "origin": origin, "if_thresholds": if_limits, "screening": screening,
              "persistence": persisted, "state_panel": panel, "probes": probes, "diversity": diversity,
              "remaining_m1": REMAINING_M1, "main_run_authorized": False}
    stage = journal.call("m1/probes-diversity-stage", {"screening": screening, "persistence": persisted,
                                                     "probes": probes, "diversity": diversity}, lambda: result)
    from closeout import finish_m1
    return finish_m1(api, root, journal, checked, stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="local-only freeze, corpus and tested-code checks")
    live = subparsers.add_parser("run", help="explicit registered M1 calibration; never M2")
    live.add_argument("--project-id", required=True)
    live.add_argument("--keychain-service", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        result = preflight(root) if args.command == "preflight" else run(
            root, project_id=args.project_id, keychain_service=args.keychain_service)
    except (RuntimeError, ValueError, OSError) as error:
        print(calibrate.canonical({"status": "blocked", "reason": str(error),
                                   "m1_complete": False, "main_run_authorized": False}))
        return 2
    summary = {key: result[key] for key in ("status", "m1_complete", "paid_launch_allowed", "failure",
                                           "issue", "slot", "freeze_sha256", "test_commit", "remaining_m1")
               if key in result}
    if args.command == "run":
        summary["journal"] = str(root / "runs/m1/journal.jsonl")
    print(calibrate.canonical(summary))
    return 2 if result.get("status") in ("m1_failed", "contract_attention_needed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
