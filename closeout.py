"""Registered M1 physical-checkpoint measurements and measured-cost handoff.

No new intervention or M2 execution is defined here. Physical aliases share
measurements, while each lifecycle context keeps its own acquisition denominator.
"""
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

from backend import ACCOUNTING_KEYS, adapter_geometry
from data import load_rows
from measure import diversity_summary, evaluate_if, evaluate_task, forward_kl
from protocol import ORDERS, schedule_seed
from rules import retention_summary


def contexts(stage):
    """List selected screening and persistence contexts, never unused LR trials."""
    result = []
    for selected in stage["screening"]["selected"].values():
        for event in selected["selected"]["realizations"]:
            slot = event["slot"]
            result.append({"event": event,
                           "acquired": {slot: event["learning"]["decision"]["checkpoint"]["heldout"]}})
    common_acquired, histories = {}, {}
    for event in stage["persistence"]["events"]:
        slot = event["slot"]
        if event["cycle"] == 1:
            acquired = {slot: event["learning"]["decision"]["checkpoint"]["heldout"]}
            common_acquired = dict(acquired)
        else:
            acquired = dict(histories.get(event["arm"], common_acquired))
            acquired[slot] = event["learning"]["decision"]["checkpoint"]["heldout"]
            histories[event["arm"]] = dict(acquired)
        result.append({"event": event, "acquired": acquired})
    return result


def checkpoint_key(checkpoint):
    return hashlib.sha256(checkpoint["sampler_path"].encode()).hexdigest()[:20]


def _read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def measure_m1_checkpoints(api, root, journal, stage, selected_diversity, measurement):
    """Download and measure every selected physical A/B checkpoint once."""
    root = Path(root)
    selections = stage["screening"]["selected"]
    origin = stage["origin"]
    pool = load_rows(root / "data/repair_pool.jsonl")[1:]
    from data import render_repair_prompt
    kl_recipe = measurement["kl"]
    kl_prompts = [render_repair_prompt(api.tokenizer, row["prompt"])
                  for row in pool[:kl_recipe["prompt_count"]]]
    reference = journal.call("closeout/kl-reference", {"origin": origin, "recipe": kl_recipe,
                             "prompt_tokens": kl_prompts},
        lambda: api.sample(api.sampler(origin["sampler_path"]), kl_prompts,
                           samples=kl_recipe["samples"], max_tokens=kl_recipe["max_tokens"],
                           temperature=kl_recipe["temperature"], seed=kl_recipe["seed"]), recoverable=True)
    panel_manifest = _read(root / f"manifests/diversity/{selected_diversity['candidate']}.json")
    panel = load_rows(root / panel_manifest["path"])
    native = {}
    for slot, selection in selections.items():
        manifest = _read(root / f"manifests/tasks/{selection['candidate']}.json")
        native[slot] = (manifest, load_rows(root / manifest["splits"]["heldout"]["path"]))
    downloaded, physical, retention = {}, {}, []

    def download(checkpoint):
        key = checkpoint_key(checkpoint)
        if key not in downloaded:
            if checkpoint["sampler_path"] == origin["sampler_path"]:
                downloaded[key] = origin["download"]
            else:
                downloaded[key] = journal.call(f"closeout/download/{key}", {"checkpoint": checkpoint},
                    lambda: api.download_sampler(checkpoint["sampler_path"], root / "runs/m1/adapters" / key))
        return Path(downloaded[key]["directory"]) / "adapter_model.safetensors"

    download(origin)
    origin_heldout = journal.call("closeout/cycle0/if-heldout", {"origin": origin},
        lambda: evaluate_if(api, origin["sampler_path"], "heldout", schedule_seed("T1", "if-heldout")),
        recoverable=True)
    for context in contexts(stage):
        event = context["event"]
        predecessor = event["start_checkpoint"]
        for label in ("A", "B"):
            checkpoint = event[label]
            key = checkpoint_key(checkpoint)
            if key not in physical:
                current_path, previous_path = download(checkpoint), download(predecessor)
                geometry = journal.call(f"closeout/geometry/{key}",
                    {"checkpoint": checkpoint, "predecessor": predecessor},
                    lambda: {"layers": adapter_geometry(current_path, previous_path=previous_path)})
                kl = journal.call(f"closeout/kl/{key}", {"checkpoint": checkpoint, "reference": "closeout/kl-reference"},
                    lambda: forward_kl(api, checkpoint["sampler_path"], kl_prompts, reference["groups"]))
                protected = journal.call(f"closeout/if-heldout/{key}", {"checkpoint": checkpoint},
                    lambda: evaluate_if(api, checkpoint["sampler_path"], "heldout", schedule_seed("T1", "if-heldout")),
                    recoverable=True)

                def diversity():
                    recipe = panel_manifest["sampling"]
                    sampled = api.sample(api.sampler(checkpoint["sampler_path"]),
                        [row["prompt_tokens"] for row in panel], samples=recipe["samples"],
                        max_tokens=recipe["max_tokens"], temperature=recipe["temperature"], seed=recipe["seed"])
                    return {**diversity_summary(panel, sampled["groups"]), "accounting": sampled["accounting"]}

                diverse = journal.call(f"closeout/diversity/{key}",
                    {"checkpoint": checkpoint, "panel_sha256": panel_manifest["sha256"]}, diversity,
                    recoverable=True)
                physical[key] = {"checkpoint": checkpoint, "predecessor": predecessor,
                                  "geometry": geometry, "kl": kl, "if_heldout": protected,
                                  "diversity": diverse, "native": {}}
            rows, client = [], None
            for slot, acquired in context["acquired"].items():
                manifest, heldout = native[slot]
                if slot not in physical[key]["native"]:
                    # Current-task A was already measured after its gate crossing.
                    if label == "A" and slot == event["slot"]:
                        metric = {"q": acquired, "reused": True, "source": event["branch"] + "/learn/heldout"}
                    else:
                        def task_metric():
                            nonlocal client
                            if client is None and manifest["metric"] == "negative_nll":
                                client = api.branch(checkpoint["state_path"], resume=False)
                            return evaluate_task(api, client, checkpoint["sampler_path"], heldout,
                                                 manifest, schedule_seed(slot, "evaluate-heldout"))
                        metric = journal.call(f"closeout/native/{key}/{slot}",
                            {"checkpoint": checkpoint, "manifest": manifest}, task_metric,
                            recoverable=manifest["metric"] == "verifier_success")
                    if metric.get("valid", True) is False:
                        raise RuntimeError("invalid native task measurement; M1 closeout is incomplete")
                    physical[key]["native"][slot] = metric
                rows.append({"task": slot, "baseline": selections[slot]["reference"]["references"]["heldout0"],
                             "acquired": acquired, "current": physical[key]["native"][slot]["q"],
                             "noise_sd": 0.0, "minimum_movement": selections[slot]["thresholds"]["minimum_movement"]})
            retention.append({"event": event["branch"], "checkpoint": label, "physical_key": key,
                              **retention_summary(rows, event["slot"])})
            predecessor = checkpoint
    return {"physical_checkpoints": physical, "retention": retention, "cycle0_if_heldout": origin_heldout}


def cost_of_prefix(journal, prefix, prices):
    from costs import token_cost
    return sum(token_cost(row["result"]["accounting"], prices)
               for name, row in journal.completed.items()
               if name.startswith(prefix) and "accounting" in row["result"])


def measured_task_units(journal, stage, prices):
    """Use selected recipes only; control prefixes retain their own stop dose."""
    from costs import token_cost
    units = {}
    for slot, selected in stage["screening"]["selected"].items():
        by_recipe = []
        for event in selected["selected"]["realizations"]:
            branch = event["branch"]
            learn = event["learning"]
            start = learn["start"]["gate"]
            threshold = selected["thresholds"]
            competent = next(point["step"] for point in learn["points"]
                if point["gate"] >= threshold["gate_competence"]
                and point["gate"] - start >= threshold["minimum_movement"])
            prefix = branch + "/learn/"
            selected_heldout = cost_of_prefix(journal, prefix + "heldout/", prices)
            control = selected_heldout
            for name, row in journal.completed.items():
                if not name.startswith(prefix) or "accounting" not in row["result"]:
                    continue
                suffix = name[len(prefix):]
                if suffix == "start" or (suffix.startswith(("update/", "evaluate/"))
                                         and int(suffix.rsplit("/", 1)[1]) <= competent):
                    control += token_cost(row["result"]["accounting"], prices)
            by_recipe.append({"learn_to_competence": control,
                              "learn_to_damage": cost_of_prefix(journal, prefix, prices),
                              "repair": cost_of_prefix(journal, branch + "/repair/", prices),
                              "native_heldout": selected_heldout})
        units[slot] = {key: statistics.mean(row[key] for row in by_recipe) for key in by_recipe[0]}
    return units


def measured_projection(root, journal, stage, measured):
    """Price the full registered design using selected-recipe measured units."""
    from costs import project_m2, token_cost
    snapshot = _read(Path(root) / "manifests/prices.json")
    prices = {"train": snapshot["train_and_forward"], "prefill": snapshot["prefill"],
              "cached": snapshot["cached_prefill"], "sample": snapshot["sample"]}
    units = {}
    for probe_class, selected in stage["probes"]["selected"].items():
        units[probe_class + "_probe"] = statistics.mean(
            cost_of_prefix(journal, branch + "/", prices) for branch in selected["standard_branches"])
    units["diversity"] = statistics.mean(
        token_cost(realization["accounting"], prices)
        for realization in stage["diversity"]["selected"]["realizations"])
    physical = list(measured["physical_checkpoints"].values())
    for endpoint in ("if_heldout", "kl"):
        units[endpoint] = statistics.mean(token_cost(row[endpoint]["accounting"], prices) for row in physical)
    task_units = measured_task_units(journal, stage, prices)
    unreconciled = [row["operation"] for row in journal.rows
                    if row["type"] == "recovery_authorized" and row.get("prior_accounting") == "unavailable"]
    return {**project_m2(task_units, units, prices), "task_units_usd": task_units,
            "measurement_units_usd": units, "price_source": snapshot,
            "m1_estimated_usd": cost_of_prefix(journal, "", prices),
            "unreconciled_sampling_operations": unreconciled}


def lifecycle_exposures(journal, stage, prices):
    """Registered competence/band exposures, from actual journal operations."""
    from costs import token_cost

    def timing(names):
        starts = [row["timestamp"] for row in journal.rows
                  if row["type"] == "inflight" and row["operation"] in names]
        ends = [journal.completed[name]["timestamp"] for name in names]
        durations = [journal.completed[name].get("elapsed_seconds") for name in names]
        timing_complete = (all(journal.completed[name].get("timing_complete", True) for name in names)
                           and all(value is not None for value in durations))
        return {"started_at": min(starts) if starts else None,
                "finished_at": max(ends) if ends else None,
                "wall_seconds": (datetime.fromisoformat(max(ends)) - datetime.fromisoformat(min(starts))).total_seconds()
                                if starts and ends else None,
                "active_wall_seconds": sum(durations) if timing_complete else None,
                "timing_complete": timing_complete}

    rows = []
    for context in contexts(stage):
        event = context["event"]
        learn, branch = event["learning"], event["branch"]
        history = [{**learn["start"], "step": 0, "target_tokens": 0}, *learn["points"]]
        thresholds = stage["screening"]["selected"][event["slot"]]["thresholds"]

        def dose(predicate):
            point = next((point for point in history if predicate(point)), None)
            if point is None:
                return None  # Censored/nonattainment is never a constructed zero.
            step, tokens, forward, usd, operations = point["step"], 0, 0, 0.0, []
            prefix = branch + "/learn/"
            for name, operation in journal.completed.items():
                if not name.startswith(prefix):
                    continue
                suffix = name[len(prefix):]
                if suffix == "start" or (suffix.startswith(("update/", "evaluate/"))
                                         and int(suffix.rsplit("/", 1)[1]) <= step):
                    usage = operation["result"].get("accounting", {})
                    tokens += usage.get("train_tokens", 0)
                    forward += usage.get("forward_tokens", 0)
                    usd += token_cost(usage, prices)
                    operations.append(name)
            return {"updates": step, "gradient_target_tokens": point["target_tokens"],
                    "train_tokens": tokens, "train_priced_tokens": tokens + forward,
                    "estimated_usd": usd, **timing(operations)}

        repair_ops = [record for name, record in journal.completed.items()
                      if name.startswith(branch + "/repair/")]
        rows.append({"event": branch, "task": event["slot"], "arm": event["arm"], "cycle": event["cycle"],
                     "competence": dose(lambda point: point["gate"] >= thresholds["gate_competence"]),
                     "damage_band_entry": dose(lambda point: thresholds["damage_low"] <= point["if_score"] <= thresholds["damage_high"]),
                     "valid_damage": dose(lambda point: point["gate"] >= thresholds["gate_competence"]
                                          and thresholds["damage_low"] <= point["if_score"] <= thresholds["damage_high"]),
                     "repair_updates": event["repair"]["decision"]["repair_steps"],
                     "repair_gradient_target_tokens": event["repair"]["target_tokens"],
                     "repair_accounting": {key: sum(op["result"].get("accounting", {}).get(key, 0) for op in repair_ops)
                                           for key in ACCOUNTING_KEYS},
                     "repair_estimated_usd": cost_of_prefix(journal, branch + "/repair/", prices),
                     **timing([name for name in journal.completed if name.startswith(branch + "/")])})
    return rows


def launch_packet(root, checked, stage, measured, projection, exposures):
    """Only registered launch fields; private checkpoint identifiers stay local."""
    root = Path(root)
    tasks = {}
    for slot, selected in stage["screening"]["selected"].items():
        manifest = _read(root / f"manifests/tasks/{selected['candidate']}.json")
        tasks[slot] = {"candidate": selected["candidate"],
                       "learning_rate": selected["selected"]["learning_rate"],
                       "batch_size": selected["selected"]["batch_size"],
                       "thresholds": selected["thresholds"], "metric": manifest["metric"],
                       "manifest": f"manifests/tasks/{selected['candidate']}.json",
                       "main_corpus_sha256": manifest["splits"]["main"]["sha256"]}
    probes = {}
    for kind, selected in stage["probes"]["selected"].items():
        probes[kind] = {key: selected[key] for key in
                       ("candidate", "learning_rate", "standard_budget", "reference_target", "delta_l",
                        "minimum_headroom", "noise_bounds", "clock_noise", "states")}
    orders = {name: {**_read(root / f"manifests/orders/{name}.json"),
                     "selected_tasks": [tasks[slot]["candidate"] for slot in slots]}
              for name, slots in ORDERS.items()}
    diversity = stage["diversity"]["selected"]
    return {"status": "M1_measurements_complete_pending_publication", "main_run_authorized": False,
            "tasks": tasks, "orders": orders, "if_thresholds": stage["if_thresholds"], "probes": probes,
            "diversity": {key: diversity[key] for key in ("candidate", "qualification", "noise")},
            "persistence": {"passes": True, "required_events": 5, "observed_events": len(stage["persistence"]["events"]),
                            "both_probe_classes_pass_all_states": all(value["passes"] for value in stage["probes"]["selected"].values())},
            "deterministic_noise_bounds": {"protected_if": 0, "task_metrics": 0, "normalized_retention": 0},
            "projection": projection, "lifecycle_exposures": exposures,
            "physical_checkpoints_measured": len(measured["physical_checkpoints"]),
            "tested_code_commit": checked["test_commit"], "input_freeze_sha256": checked["freeze_sha256"],
            "publication_freeze_commit": None,
            "caveats": ["The publication/freeze commit must be recorded after publishing the completed calibration package, before requesting M2 authorization.",
                        "Costs are measured projections, not guarantees; scorer token estimates await posted billing reconciliation.",
                        "The full design remains twelve order-by-policy lineages. Measurement triplicates are not extra lineages.",
                        "No deviations from the scientific contract were introduced by this runner."]}


def finish_m1(api, root, journal, checked, stage):
    """Complete measurements and prepare a handoff; never authorize or run M2."""
    from data import write_once
    measured = measure_m1_checkpoints(api, root, journal, stage, stage["diversity"]["selected"], checked["measurement"])
    projection = measured_projection(root, journal, stage, measured)
    exposures = lifecycle_exposures(journal, stage, projection["prices_usd_per_million_tokens"])
    packet = launch_packet(root, checked, stage, measured, projection, exposures)
    if projection["unreconciled_sampling_operations"]:
        packet["caveats"].append(
            "An interrupted pre-cache sampling evaluation was repeated with explicit approval; "
            "its original request outcomes and usage were unavailable. The M1 ledger estimate "
            "excludes that unknown usage pending billing reconciliation; no training update was repeated.")
    result = journal.call("m1/measurement-closeout", {"packet": packet}, lambda: {
        "status": packet["status"], "m1_complete": False, "main_run_authorized": False,
        "measurements": measured, "launch_packet": packet,
        "remaining_m1": ["publish CALIBRATION and final frozen selections; record the publication commit"]})
    write_once(Path(root) / "runs/m1/launch_packet.json", packet)
    return result
