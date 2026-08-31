"""Pure decision rules for SPEC.md, v2, including the 2026-08-31 amendment.

Task scores supplied here are already oriented Q: accuracy for exact tasks and
negative NLL for loss-based tasks. No function infers a score's direction, noise
bound, checkpoint, missing measurement, or successful intervention.
"""

import math
import statistics


def _finite(name, *values):
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must be finite")


def _nonnegative(name, value):
    _finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _integer(name, value, minimum, maximum):
    _finite(name, value)
    if isinstance(value, bool) or value != int(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _arm(arm):
    if arm not in {"fixed", "rolling", "learn-only"}:
        raise ValueError("arm must be fixed, rolling, or learn-only")


def if_thresholds(cycle0_score):
    """SPEC §2.2: exact integer ceil/floor arithmetic, on the 60-item criterion."""
    _integer("cycle-0 criterion score", cycle0_score, 0, 60)
    score = int(cycle0_score)
    return {
        "damage_low": (3 * score + 4) // 5,
        "damage_high": (17 * score) // 20,
        "recovery_target": (19 * score + 19) // 20,
    }


def acquisition_references(trajectories, cycle0_scores=None):
    """SPEC §5.1: independent maxima at 0,5,...,120, with a complete valid LR.

Each trajectory has ``learning_rate``, ``points`` (step/gate/heldout), and an
optional ``failure_step`` for a numerical failure between evaluations. A point
with ``valid=False`` or nonfinite metrics terminates that LR's valid prefix.
Later points never contribute. Incomplete prefixes cannot alone define a
reference, even if their early scores are high. All three registered LRs must
be represented; a failed attempt may have no points. Supply frozen cycle0_scores
(gate/heldout) when available. Otherwise use update zero from the lowest
registered LR with a valid zero measurement, without averaging noisy repeats.
"""
    registered_lrs = {1e-5, 3e-5, 1e-4}
    rates = [trajectory["learning_rate"] for trajectory in trajectories]
    if len(rates) != 3 or set(rates) != registered_lrs:
        raise ValueError("reference sweep must contain each registered learning rate once")
    registered_steps = list(range(0, 121, 5))
    valid_points = []
    statuses = []
    baselines = []
    complete_rates = []
    for trajectory in trajectories:
        learning_rate = trajectory["learning_rate"]
        failure_step = trajectory.get("failure_step")
        if failure_step is not None:
            _integer("reference failure step", failure_step, 0, 120)
        prefix = []
        failed = False
        for point in trajectory["points"]:
            step = point["step"]
            if failure_step is not None and step >= failure_step:
                failed = True
                break
            _integer("reference step", step, 0, 120)
            if len(prefix) >= len(registered_steps) or step != registered_steps[len(prefix)]:
                raise ValueError("reference evaluations must form the registered 0,5,...,120 prefix")
            gate, heldout = point.get("gate"), point.get("heldout")
            if (point.get("valid", True) is not True or gate is None or heldout is None
                    or not math.isfinite(gate) or not math.isfinite(heldout)):
                failure_step = step
                failed = True
                break
            record = {"learning_rate": learning_rate, "step": step, "gate": gate, "heldout": heldout}
            prefix.append(record)
            valid_points.append(record)
        if failure_step is not None:
            failed = True
        if prefix:
            baselines.append({"learning_rate": learning_rate,
                              "gate": prefix[0]["gate"], "heldout": prefix[0]["heldout"]})
        complete = len(prefix) == len(registered_steps) and not failed
        if complete:
            complete_rates.append(learning_rate)
        statuses.append({
            "learning_rate": learning_rate,
            "status": "complete" if complete else "numerical_failure" if failed else "incomplete",
            "last_valid_step": prefix[-1]["step"] if prefix else None,
            "failure_step": failure_step,
        })
    if cycle0_scores is not None:
        _finite("frozen cycle-0 scores", cycle0_scores["gate"], cycle0_scores["heldout"])
        baseline = cycle0_scores
    else:
        baseline = min(baselines, key=lambda row: row["learning_rate"]) if baselines else None
    sweep_complete = all(row["status"] != "incomplete" for row in statuses)
    defined = bool(complete_rates) and sweep_complete
    return {
        "status": "defined" if defined else "incomplete_sweep" if not sweep_complete else "no_complete_valid_trajectory",
        "gate0": baseline["gate"] if baseline else None,
        "heldout0": baseline["heldout"] if baseline else None,
        "gate_ref": max(point["gate"] for point in valid_points) if defined else None,
        "heldout_ref": max(point["heldout"] for point in valid_points) if defined else None,
        "complete_learning_rates": sorted(complete_rates),
        "sweep_complete": sweep_complete,
        "trajectory_status": statuses,
        "cycle0_observations": sorted(baselines, key=lambda row: row["learning_rate"]),
        "valid_points": valid_points,
        "registered_steps": registered_steps,
    }


def acquisition_thresholds(gate0, gate_ref, heldout0, heldout_ref, gate_noise_sd):
    """SPEC §5: caller supplies oriented scores, including negative NLL."""
    _finite("acquisition scores", gate0, gate_ref, heldout0, heldout_ref)
    _nonnegative("gate noise SD", gate_noise_sd)
    if gate_ref < gate0 or heldout_ref < heldout0:
        raise ValueError("reference maxima cannot be below their included cycle-0 points")
    return {
        "gate0": gate0,
        "gate_ref": gate_ref,
        "heldout0": heldout0,
        "heldout_ref": heldout_ref,
        "gate_noise_sd": gate_noise_sd,
        "gate_competence": gate0 + 0.70 * (gate_ref - gate0),
        "heldout_competence": heldout0 + 0.50 * (heldout_ref - heldout0),
        "minimum_movement": max(5 * gate_noise_sd, 0.15 * (gate_ref - gate0)),
    }


def acquisition_status(gate_start, gate_end, thresholds):
    _finite("gate measurements", gate_start, gate_end, thresholds["gate_competence"])
    minimum = thresholds["minimum_movement"]
    _nonnegative("minimum movement", minimum)
    headroom = thresholds["gate_competence"] - gate_start
    movement = gate_end - gate_start
    return {
        "headroom": headroom,
        "movement": movement,
        "sufficient_headroom": headroom >= minimum,
        "sufficient_movement": movement >= minimum,
        "gate_competent": gate_end >= thresholds["gate_competence"],
    }


def primary_eligibility(arm, start, a, thresholds):
    """SPEC §14: eligibility fixed before repair, independent of its outcome."""
    _arm(arm)
    _integer("start criterion score", start["if_score"], 0, 60)
    _integer("A criterion score", a["if_score"], 0, 60)
    acquisition = acquisition_status(start["gate"], a["gate"], thresholds)
    heldout = a.get("heldout")
    if heldout is not None:
        _finite("held-out score", heldout)
    checks = {
        "repair_arm": arm != "learn-only",
        "restored_start": start["if_score"] >= thresholds["recovery_target"],
        "sufficient_headroom": acquisition["sufficient_headroom"],
        "gate_competence": acquisition["gate_competent"],
        "minimum_movement": acquisition["sufficient_movement"],
        "heldout_competence": heldout is not None and heldout >= thresholds["heldout_competence"],
        "in_damage_band": thresholds["damage_low"] <= a["if_score"] <= thresholds["damage_high"],
        "actual_repair_required": a["if_score"] < thresholds["recovery_target"],
    }
    return {"eligible": all(checks.values()), "checks": checks,
            "failures": [name for name, passed in checks.items() if not passed]}


def learning_decision(arm, start, points, thresholds):
    """SPEC §§6–7: first real stopping checkpoint, without interpolated damage.

``start`` has if_score/gate; contiguous post-update points have step/if_score/
gate and optionally heldout. A missing selected-A held-out measurement is
pending, not a measured failure. Supplied points after the first stop are not
used to replace A or classify its learning history.
"""
    _arm(arm)
    _integer("start criterion score", start["if_score"], 0, 60)
    _finite("start gate score", start["gate"])
    for expected_step, point in enumerate(points, 1):
        _integer("learning step", point["step"], 1, 120)
        if point["step"] != expected_step:
            raise ValueError("learning evaluations must be contiguous after every update")
        _integer("criterion score", point["if_score"], 0, 60)
        _finite("task gate score", point["gate"])
        if point.get("heldout") is not None:
            _finite("held-out measurement", point["heldout"])
    low, high = thresholds["damage_low"], thresholds["damage_high"]
    selected = None
    used_points = []
    for point in points:
        used_points.append(point)
        acquisition = acquisition_status(start["gate"], point["gate"], thresholds)
        stopping_condition = (
            acquisition["sufficient_movement"] if arm == "learn-only"
            else low <= point["if_score"] <= high
        )
        if acquisition["gate_competent"] and stopping_condition:
            selected = point
            break
    terminal = used_points[-1] if used_points else {**start, "step": 0}
    history = [{**start, "step": 0}, *used_points]
    stop = selected is not None or terminal["step"] == 120
    first_competence = next((point["step"] for point in history
                             if point["gate"] >= thresholds["gate_competence"]), None)
    first_band = next((point["step"] for point in history if low <= point["if_score"] <= high), None)
    first_below = next((point["step"] for point in history if point["if_score"] < low), None)
    flags = []
    acquisition = acquisition_status(start["gate"], terminal["gate"], thresholds)
    if start["if_score"] < thresholds["recovery_target"]:
        flags.append("unrestored_start")
    if not acquisition["sufficient_headroom"]:
        flags.append("already_competent")
    if stop and selected is None:
        if first_competence is None:
            flags.append("competence_unmet")
        else:
            if all(point["if_score"] > high for point in history if point["step"] >= first_competence):
                flags.append("undamageable")
            if any(left["if_score"] > high and right["if_score"] < low
                   for left, right in zip(history, history[1:])):
                flags.append("band_overshoot")
        if any(point["if_score"] <= high and
               (first_competence is None or point["step"] < first_competence) for point in history):
            flags.append("damage_before_competence")
    awaiting_heldout = selected is not None and selected.get("heldout") is None
    if selected is not None and not awaiting_heldout and selected["heldout"] < thresholds["heldout_competence"]:
        flags.append("heldout_competence_fail")
    if not stop:
        classification = "learning"
    elif awaiting_heldout:
        classification = "heldout_pending"
    elif len(flags) == 1:
        classification = flags[0]
    elif flags or selected is None:
        classification = "mixed_gate_failure"
    else:
        classification = "valid_acquisition"
    eligibility = primary_eligibility(arm, start, terminal, thresholds)
    eligible = selected is not None and eligibility["eligible"]
    return {
        "stop": stop,
        "stop_step": terminal["step"] if stop else None,
        "checkpoint": dict(terminal),
        "classification": classification,
        "flags": sorted(flags),
        "awaiting_heldout": awaiting_heldout,
        "primary_eligible": eligible,
        "eligibility": eligibility,
        "first_competence_step": first_competence,
        "first_band_step": first_band,
        "first_below_band_step": first_below,
        "competence_status": "observed" if first_competence is not None else "right_censored" if stop else "incomplete",
        "valid_damage_step": selected["step"] if eligible else None,
        "valid_damage_status": "observed" if eligible else "right_censored" if terminal["step"] == 120 else "unavailable",
    }


def repair_decision(arm, a_score, checks, thresholds):
    """SPEC §8: check at 5,10,...,150; a success at step 150 is a success.

The cap is an observed expenditure, never an imputed time to successful repair.
No-op and control A/B aliases have no repair-effect observation.
"""
    _arm(arm)
    _integer("A criterion score", a_score, 0, 60)
    target = thresholds["recovery_target"]
    if arm == "learn-only" or a_score >= target:
        if checks:
            raise ValueError("a control or no-repair-required cycle cannot have repair checks")
        return {
            "stop": True, "status": "learn_only_control" if arm == "learn-only" else "no_repair_required",
            "repair_steps": 0, "recovered": a_score >= target, "repair_success": None,
            "time_to_success": None, "time_to_success_status": "not_applicable",
            "censor_step": None, "next_check_step": None, "criterion_score": a_score,
            "repair_effect_observed": False, "identity_difference": 0.0,
        }
    for index, check in enumerate(checks, 1):
        _integer("repair check step", check["step"], 5, 150)
        if check["step"] != 5 * index:
            raise ValueError("repair checks must follow the registered 5,10,...,150 prefix")
        _integer("repair criterion score", check["if_score"], 0, 60)
    selected = next((check for check in checks if check["if_score"] >= target), None)
    terminal = selected or (checks[-1] if checks else {"step": 0, "if_score": a_score})
    success = selected is not None
    failed = not success and terminal["step"] == 150
    return {
        "stop": success or failed,
        "status": "repaired" if success else "repair_failure" if failed else "repairing",
        "repair_steps": terminal["step"],
        "recovered": success,
        "repair_success": success if success or failed else None,
        "time_to_success": terminal["step"] if success else None,
        "time_to_success_status": "observed" if success else "right_censored" if failed else "incomplete",
        "censor_step": 150 if failed else None,
        "next_check_step": None if success or failed else terminal["step"] + 5,
        "criterion_score": terminal["if_score"],
        "repair_effect_observed": terminal["step"] > 0,
        "identity_difference": None,
    }


def repair_effect(a_value, b_value, arm, repair_steps):
    """Separate numerical A/B identity from evidence of an actual intervention."""
    _arm(arm)
    _integer("repair steps", repair_steps, 0, 150)
    if arm == "learn-only" and repair_steps:
        raise ValueError("learn-only cannot receive repair")
    available = a_value is not None and b_value is not None
    for value in (a_value, b_value):
        if value is not None:
            _finite("endpoint measurement", value)
    intervention = arm != "learn-only" and repair_steps > 0
    if not intervention and available and a_value != b_value:
        raise ValueError("a physical A/B alias must reuse the same endpoint measurement")
    difference = b_value - a_value if available else None
    return {
        "identity_difference": difference,
        "repair_effect": difference if intervention else None,
        "repair_effect_observed": intervention and available,
        "status": ("learn_only_control" if arm == "learn-only" else "no_repair_required")
        if not intervention else "observed" if available else "measurement_unavailable",
    }


def normalized_retention(baseline, acquired, current, noise_sd, minimum_movement):
    """SPEC §10: strict denominator guards and unclipped oriented retention."""
    _nonnegative("task noise SD", noise_sd)
    _nonnegative("minimum movement", minimum_movement)
    for value in (baseline, acquired, current):
        if value is not None:
            _finite("retention score", value)
    if baseline is None or acquired is None:
        return {"value": None, "denominator": None, "status": "acquisition_measurement_unavailable"}
    denominator = acquired - baseline
    if denominator <= 5 * noise_sd or denominator <= minimum_movement:
        return {"value": None, "denominator": denominator, "status": "undefined_denominator"}
    return {
        "value": (current - baseline) / denominator if current is not None else None,
        "denominator": denominator,
        "status": "defined" if current is not None else "measurement_unavailable",
    }


def retention_summary(tasks, current_task):
    """Summarize acquired tasks only; undefined values are excluded, not zeroed.

Rows have task/baseline/acquired/current/noise_sd/minimum_movement. The current
task is excluded from the primary prior-task mean, including in cycle one.
"""
    names = [row["task"] for row in tasks]
    if len(names) != len(set(names)) or current_task not in names:
        raise ValueError("retention rows must uniquely include the current acquired task")
    results = {
        row["task"]: normalized_retention(row["baseline"], row["acquired"], row.get("current"),
                                          row["noise_sd"], row["minimum_movement"])
        for row in tasks
    }
    prior = [result["value"] for task, result in results.items() if task != current_task]
    all_values = [result["value"] for result in results.values()]
    out = {"tasks": results, "current": results[current_task]["value"]}
    for label, values in (("prior", prior), ("all", all_values)):
        defined = [value for value in values if value is not None]
        out[f"{label}_mean"] = statistics.mean(defined) if defined else None
        out[f"{label}_coverage"] = {
            "defined": len(defined), "total": len(values),
            "fraction": len(defined) / len(values) if values else None,
        }
    return out


def _curve(steps, losses):
    if len(steps) != len(losses) or len(steps) == 0:
        raise ValueError("curve needs aligned, nonempty steps and losses")
    _finite("curve", *steps, *losses)
    if steps[0] < 0 or any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("curve steps must be nonnegative and strictly increasing")


# Adapted from kintsugi-v1/analyze.py:18-34 at
# d2aa7cd94a0a618169496f0235fa021ea46c0372. Retain first-bracket interpolation
# and unavailable/censored clocks. V2 additionally rejects a zero-headroom
# normalized clock and an unbracketed initial crossing; neither becomes zero.
def interpolated_crossing_time(steps, losses, target):
    """First downward crossing bracketed by two supplied registered points."""
    _curve(steps, losses)
    _finite("clock target", target)
    if losses[0] <= target:
        return None
    for left_step, right_step, left_loss, right_loss in zip(steps, steps[1:], losses, losses[1:]):
        if left_loss > target >= right_loss:
            fraction = (target - left_loss) / (right_loss - left_loss)
            return float(left_step + fraction * (right_step - left_step))
    return None


def validation_loss_auc(steps, losses):
    """Time-normalized trapezoidal AUC over the observed registered span.

Arithmetic adapted from kintsugi-v1/analyze.py:5-13 at
d2aa7cd94a0a618169496f0235fa021ea46c0372; no NumPy-version fallback is needed.
"""
    _curve(steps, losses)
    if len(steps) == 1:
        return float(losses[0])
    area = sum((right_step - left_step) * (left_loss + right_loss) / 2
               for left_step, right_step, left_loss, right_loss
               in zip(steps, steps[1:], losses, losses[1:]))
    return area / (steps[-1] - steps[0])


def trainability_clocks(steps, losses, reference_target, delta_l, eval_noise_sd):
    """SPEC §9: fixed-reference clocks, with explicit headroom/censor statuses.

The caller supplies registered evaluations, never smoothed or interpolated
input points. Returned clock values are None unless an actual first crossing
is bracketed. ``censor_step`` records exposure, not an imputed clock value.
"""
    _curve(steps, losses)
    if steps[0] != 0:
        raise ValueError("a probe curve must include its measured update-0 loss")
    _finite("probe reference and displacement", reference_target, delta_l)
    _nonnegative("evaluation noise SD", eval_noise_sd)
    if delta_l <= 0:
        raise ValueError("the registered absolute loss reduction must be positive")
    headroom = losses[0] - reference_target
    t50_target = (losses[0] + reference_target) / 2
    tdelta_target = losses[0] - delta_l
    t50_headroom = headroom > 0 and headroom >= 5 * eval_noise_sd
    tdelta_headroom = headroom >= delta_l + 5 * eval_noise_sd
    t50 = interpolated_crossing_time(steps, losses, t50_target) if t50_headroom else None
    tdelta = interpolated_crossing_time(steps, losses, tdelta_target) if tdelta_headroom else None
    return {
        "headroom": headroom,
        "t50_target": t50_target,
        "t50": t50,
        "t50_status": "undefined_headroom" if not t50_headroom else "observed" if t50 is not None else "right_censored",
        "tdelta_target": tdelta_target,
        "tdelta": tdelta,
        "tdelta_status": "undefined_headroom" if not tdelta_headroom else "observed" if tdelta is not None else "right_censored",
        "censor_step": steps[-1],
        "initial_loss": losses[0],
        "best_loss_reduction": losses[0] - min(losses),
        "final_loss": losses[-1],
        "auc": validation_loss_auc(steps, losses),
        "progress": [(losses[0] - loss) / headroom for loss in losses] if headroom > 0 else None,
    }


def registered_probe_steps(budget, eval_every):
    """Include update zero and the final budget even when cadence does not divide it."""
    _finite("probe budget", budget)
    if isinstance(budget, bool) or budget < 1 or budget != int(budget):
        raise ValueError("probe budget must be a positive integer")
    _integer("probe evaluation cadence", eval_every, 1, budget)
    steps = list(range(0, int(budget) + 1, int(eval_every)))
    if steps[-1] != budget:
        steps.append(int(budget))
    return steps


def probe_reference_target(steps, losses, standard_budget, eval_every):
    """SPEC §9.3: probe references, unlike acquisition references, use 2× budget."""
    _curve(steps, losses)
    if list(steps) != registered_probe_steps(2 * standard_budget, eval_every):
        raise ValueError("probe reference needs every registered twice-budget evaluation")
    target = min(losses)
    return {"reference_target": target, "reference_step": steps[list(losses).index(target)]}


def m1_probe_candidate(candidate, learning_rate, states, reference_target,
                       eval_noise_sd, standard_budget, eval_every):
    """SPEC §§9.4–9.5: apply every rule to every supplied frozen M1 state.

Each state has state/steps/losses. The caller supplies the complete prospective
state panel; this function never substitutes a smaller panel after failures.
"""
    if learning_rate not in {1e-5, 3e-5, 1e-4}:
        raise ValueError("probe learning rate is outside the registered grid")
    _finite("reference target", reference_target)
    _nonnegative("evaluation noise SD", eval_noise_sd)
    expected_steps = registered_probe_steps(standard_budget, eval_every)
    names = [state["state"] for state in states]
    if not names or len(names) != len(set(names)):
        raise ValueError("M1 needs a nonempty, uniquely identified state panel")
    for state in states:
        _curve(state["steps"], state["losses"])
        if list(state["steps"]) != expected_steps:
            raise ValueError("each M1 state needs the full registered standard-budget curve")
    minimum_headroom = min(state["losses"][0] - reference_target for state in states)
    delta_l = 0.25 * minimum_headroom
    results = []
    for state in states:
        headroom = state["losses"][0] - reference_target
        clocks = trainability_clocks(state["steps"], state["losses"], reference_target,
                                     delta_l, eval_noise_sd) if delta_l > 0 else None
        final_progress = clocks["progress"][-1] if clocks and clocks["progress"] is not None else None
        t50 = clocks["t50"] if clocks else None
        checks = {
            "normalized_headroom": headroom > 0 and headroom >= 5 * eval_noise_sd,
            "final_progress": final_progress is not None and final_progress >= 0.60,
            "bracketed_t50": clocks is not None and clocks["t50_status"] == "observed",
            "dynamic_coverage": t50 is not None and 0.20 * standard_budget <= t50 <= 0.80 * standard_budget,
            "absolute_headroom": delta_l > 0 and headroom >= delta_l + 5 * eval_noise_sd,
            "absolute_crossing": clocks is not None and clocks["tdelta_status"] == "observed",
        }
        results.append({"state": state["state"], "passes": all(checks.values()),
                        "checks": checks, "final_progress": final_progress, "clocks": clocks})
    observed_t50 = [result["clocks"]["t50"] for result in results
                    if result["clocks"] is not None and result["clocks"]["t50"] is not None]
    return {
        "candidate": candidate, "learning_rate": learning_rate,
        "passes": all(result["passes"] for result in results),
        "minimum_headroom": minimum_headroom, "delta_l": delta_l,
        "standard_budget": standard_budget,
        "median_t50": statistics.median(observed_t50) if len(observed_t50) == len(states) else None,
        "states": results,
    }


def select_probe(candidates, candidate_order):
    """SPEC §9.6: lowest passing LR first, then headroom/centrality/listed order."""
    if len(candidate_order) != len(set(candidate_order)):
        raise ValueError("registered candidate order must be unique")
    seen = set()
    lowest_by_candidate = {}
    for result in candidates:
        name, rate = result["candidate"], result["learning_rate"]
        if name not in candidate_order or rate not in {1e-5, 3e-5, 1e-4} or (name, rate) in seen:
            raise ValueError("unregistered or duplicate candidate/LR pair")
        seen.add((name, rate))
        if result["passes"]:
            _finite("passing probe summaries", result["minimum_headroom"], result["median_t50"])
            if name not in lowest_by_candidate or rate < lowest_by_candidate[name]["learning_rate"]:
                lowest_by_candidate[name] = result
    if not lowest_by_candidate:
        return None
    selected = min(lowest_by_candidate.values(), key=lambda result: (
        -result["minimum_headroom"],
        abs(result["median_t50"] - result["standard_budget"] / 2),
        candidate_order.index(result["candidate"]),
    ))
    return dict(selected)


def select_task_recipe(recipes):
    """SPEC §6.1: two eligible, repairable realizations, then median target dose.

A recipe has learning_rate/batch_size/realizations. Each realization has a
distinct realization ID, primary_eligible, repair_success, and target_tokens.
The batch size is fixed across this task's recipe grid before outcomes.
"""
    batches = {recipe["batch_size"] for recipe in recipes}
    if not batches <= {16, 32, 64} or len(batches) > 1:
        raise ValueError("a task's batch size must be fixed before comparing recipes")
    rates = [recipe["learning_rate"] for recipe in recipes]
    if len(rates) != len(set(rates)) or not set(rates) <= {1e-5, 3e-5, 1e-4}:
        raise ValueError("recipe learning rates must be unique registered values")
    qualified = []
    for recipe in recipes:
        realizations = recipe["realizations"]
        if len(realizations) != 2 or len({row["realization"] for row in realizations}) != 2:
            raise ValueError("recipe screening requires two distinct realizations")
        if all(row["primary_eligible"] is True and row["repair_success"] is True for row in realizations):
            for row in realizations:
                _nonnegative("target-token dose", row["target_tokens"])
            dose = statistics.median(row["target_tokens"] for row in realizations)
            qualified.append({**recipe, "median_target_tokens": dose})
    return min(qualified, key=lambda recipe: (recipe["median_target_tokens"], recipe["learning_rate"])) if qualified else None


def cohort_summary(cycles):
    """SPEC §§14–15: a failed repair stays primary; one invalid cycle only loses strictness."""
    groups = {}
    for row in cycles:
        _arm(row["arm"])
        _integer("lifecycle cycle", row["cycle"], 1, 7)
        group = groups.setdefault(row["lineage"], [])
        if any(previous["cycle"] == row["cycle"] or previous["arm"] != row["arm"] for previous in group):
            raise ValueError("lineage cycles must be unique and use one arm")
        group.append(row)
    primary = [row for row in cycles if row["arm"] != "learn-only" and row["primary_eligible"] is True]
    strict_lineages = sorted(name for name, rows in groups.items()
                             if len(rows) == 7 and rows[0]["arm"] != "learn-only"
                             and all(row["primary_eligible"] is True for row in rows))
    return {
        "lifecycle": list(cycles),
        "primary": primary,
        "criterion_matched": [row for row in primary if row.get("repair_success") is True],
        "strict_lineages": strict_lineages,
        "strict": [row for row in primary if row["lineage"] in strict_lineages],
    }


def paired_clocks_available(probe):
    """Both registered clocks must be observed at both physical endpoints."""
    for checkpoint in ("A", "B"):
        clocks = probe.get(checkpoint, {})
        for clock in ("t50", "tdelta"):
            value = clocks.get(clock)
            if (clocks.get(f"{clock}_status") != "observed" or value is None
                    or not math.isfinite(value) or value <= 0):
                return False
    return True


def coverage_requirements(cycles, claimed_probes=()):
    """SPEC §16: fixed denominators; paired clock coverage never changes eligibility.

Rows include arm/order/task/cycle/primary_eligible. Claimed probe data have
``probes[name][A or B]`` trainability-clock results, including observed statuses.
"""
    if len(claimed_probes) != len(set(claimed_probes)) or not set(claimed_probes) <= {"structured", "language"}:
        raise ValueError("claimed probes must be unique registered probe classes")
    tasks = [f"T{index}" for index in range(1, 8)]
    orders = {f"O{index}" for index in range(1, 5)}
    seen_cycles, seen_tasks = set(), set()
    eligible = []
    for row in cycles:
        _arm(row["arm"])
        _integer("lifecycle cycle", row["cycle"], 1, 7)
        if row["order"] not in orders or row["task"] not in tasks:
            raise ValueError("coverage row is outside the registered task/order design")
        cycle_key = (row["arm"], row["order"], row["cycle"])
        task_key = (row["arm"], row["order"], row["task"])
        if cycle_key in seen_cycles or task_key in seen_tasks:
            raise ValueError("duplicate scheduled observation cannot increase coverage")
        seen_cycles.add(cycle_key)
        seen_tasks.add(task_key)
        if row["arm"] != "learn-only" and row["primary_eligible"] is True:
            eligible.append(row)
    task_coverage = {}
    for task in tasks:
        count = sum(row["task"] == task for row in eligible)
        task_coverage[task] = {"eligible": count, "scheduled": 8, "passes": count >= 6}
    arm_orders = {arm: sorted({row["order"] for row in eligible if row["arm"] == arm})
                  for arm in ("fixed", "rolling")}
    manipulation_passes = (len(eligible) >= 42 and all(result["passes"] for result in task_coverage.values())
                           and all(set(represented) == orders for represented in arm_orders.values()))
    probe_coverage = {}
    for probe in claimed_probes:
        count = sum(paired_clocks_available(row.get("probes", {}).get(probe, {})) for row in eligible)
        probe_coverage[probe] = {
            "available": count, "eligible": len(eligible),
            "fraction": count / len(eligible) if eligible else None,
            "passes": bool(eligible) and 5 * count >= 4 * len(eligible),
        }
    return {
        "eligible": len(eligible), "scheduled": 56,
        "per_task": task_coverage, "orders_per_arm": arm_orders,
        "probe_coverage": probe_coverage, "manipulation_passes": manipulation_passes,
        "passes": manipulation_passes and all(result["passes"] for result in probe_coverage.values()),
    }


def measurement_noise_bound(values, direct_null_differences=None, *, kind="stochastic_endpoint"):
    """SPEC §17.0: the operational bound from exactly three complete repeats.

For independent single-checkpoint repeats, paired-difference SD is sqrt(2)
times sample SD (ddof=1). Direct null differences already measure the paired
quantity, so their sample SD receives only the 2.5 multiplier. The caller must
establish independence and the same frozen recipe; equal values are permitted.

Use kind='deterministic' for registered deterministic protected IF, exact task
gate/held-out metrics, validation NLL, and inherited normalized retention.
Use 'process' for steps, doses, tokens, and cost: the bound is not applicable.
Stochastic normalized retention requires kind='stochastic_retention' and a
prospective propagated bound, never an automatic zero. No kind is inferred
from observed variation. These are not CIs, p-values, or seed-level inference.
"""
    if kind not in {"stochastic_endpoint", "deterministic", "process", "stochastic_retention"}:
        raise ValueError("unknown registered measurement-noise kind")
    direct = direct_null_differences is not None
    supplied = direct_null_differences if direct else values
    repeats = list(supplied) if supplied is not None else []
    result = {
        "status": "defined", "count": len(repeats),
        "sample_sd": None, "paired_difference_sd": None, "bound": None,
        "source": "direct_null_paired_differences" if direct else "single_checkpoint_repeats",
    }
    if kind == "deterministic":
        result.update(sample_sd=0.0, paired_difference_sd=0.0, bound=0.0,
                      source="registered_deterministic_evaluation")
        return result
    if kind == "process":
        result.update(status="not_applicable", source="observed_process_quantity")
        return result
    if kind == "stochastic_retention":
        result.update(status="contract_attention_required", source="unregistered_stochastic_retention")
        return result
    if len(repeats) != 3:
        result["status"] = "invalid_replicate_count"
        return result
    try:
        finite = all(value is not None and math.isfinite(value) for value in repeats)
    except (TypeError, ValueError, OverflowError):
        finite = False
    if not finite:
        result["status"] = "nonfinite_replicate"
        return result
    try:
        sample_sd = statistics.stdev(repeats)
    except (OverflowError, statistics.StatisticsError):
        result["status"] = "nonfinite_noise_statistic"
        return result
    paired_sd = sample_sd if direct else math.sqrt(2) * sample_sd
    bound = 2.5 * paired_sd
    if not all(math.isfinite(value) for value in (sample_sd, paired_sd, bound)):
        result["status"] = "nonfinite_noise_statistic"
        return result
    result.update(sample_sd=sample_sd, paired_difference_sd=paired_sd, bound=bound)
    return result


def probe_clock_noise(replicates):
    """SPEC §17.0: all three complete cycle-0 trajectories must supply both clocks.

Inputs are trainability_clocks results from the same candidate/LR, reference,
displacement, and frozen standard budget. Their 'observed' statuses certify
first bracketed crossings. The caller establishes those shared settings and
independent trajectory execution. No surviving-subset SD is calculated if any
repeat is unavailable, censored, unbracketed, or incomplete.
"""
    replicates = list(replicates) if replicates is not None else []
    failures = []
    if len(replicates) != 3:
        failures.append({"reason": "requires_three_complete_trajectories"})
    budgets = []
    for index, replicate in enumerate(replicates, 1):
        if not isinstance(replicate, dict):
            failures.append({"replicate": index, "reason": "unavailable_trajectory"})
            continue
        budget = replicate.get("censor_step")
        if budget not in {32, 245}:
            failures.append({"replicate": index, "reason": "incomplete_standard_budget"})
        else:
            budgets.append(budget)
        for clock in ("t50", "tdelta"):
            value = replicate.get(clock)
            if (replicate.get(f"{clock}_status") != "observed" or value is None
                    or not math.isfinite(value) or value <= 0
                    or (budget in {32, 245} and value > budget)):
                failures.append({"replicate": index, "clock": clock,
                                 "reason": "unavailable_or_unbracketed_clock"})
    if len(set(budgets)) > 1:
        failures.append({"reason": "inconsistent_standard_budgets"})
    output = {
        "status": "candidate_invalid" if failures else "defined",
        "passes": not failures, "count": len(replicates),
        "source": "three_complete_cycle0_probe_trajectories",
        "standard_budget": budgets[0] if len(budgets) == len(replicates) and len(set(budgets)) == 1 else None,
        "bounds": {"t50": None, "tdelta": None}, "failures": failures,
    }
    for clock in ("t50", "tdelta"):
        if failures:
            result = {"status": "candidate_invalid", "count": len(replicates),
                      "sample_sd": None, "paired_difference_sd": None, "bound": None,
                      "source": "single_checkpoint_repeats"}
        else:
            result = measurement_noise_bound([replicate[clock] for replicate in replicates])
        output[clock] = result
        output["bounds"][clock] = result["bound"]
    return output


def _direction(value):
    if value is None:
        return None
    _finite("effect", value)
    return 1 if value > 0 else -1 if value < 0 else 0


def _order_agreement(order_effects, direction):
    if not set(order_effects) <= {"O1", "O2", "O3", "O4"}:
        raise ValueError("order effects must use the four registered order IDs")
    directions = [_direction(value) for value in order_effects.values()]
    return {
        "valid": sum(value is not None for value in directions),
        "consistent": sum(value == direction for value in directions) if direction in {-1, 1} else 0,
    }


def trainability_claim(t50_effect, tdelta_effect, relative_t50_effect,
                       t50_noise_bound, tdelta_noise_bound, order_effects,
                       task_adjusted_effect, coverage_met):
    """SPEC §17.2: a domain claim, using supplied M1 paired-noise bounds.

Effects and order summaries are B minus A; relative_t50_effect is a fraction,
not percent. The caller supplies the registered order-level and task-adjusted
summaries. This predicate neither fits them nor treats cycles as random seeds.
"""
    _nonnegative("t50 paired-noise bound", t50_noise_bound)
    _nonnegative("absolute-clock paired-noise bound", tdelta_noise_bound)
    direction = _direction(t50_effect)
    absolute_direction = _direction(tdelta_effect)
    relative_direction = _direction(relative_t50_effect)
    adjusted_direction = _direction(task_adjusted_effect)
    orders = _order_agreement(order_effects, direction)
    available = all(value is not None for value in (t50_effect, tdelta_effect, relative_t50_effect))
    checks = {
        "clock_direction_agreement": direction in {-1, 1} and absolute_direction == direction,
        "t50_above_noise": t50_effect is not None and abs(t50_effect) > t50_noise_bound,
        "tdelta_above_noise": tdelta_effect is not None and abs(tdelta_effect) > tdelta_noise_bound,
        "relative_practical_margin": relative_t50_effect is not None and abs(relative_t50_effect) > 0.10,
        "relative_direction_agreement": direction in {-1, 1} and relative_direction == direction,
        "order_agreement": orders["consistent"] >= 3,
        "task_adjusted_direction": direction in {-1, 1} and adjusted_direction == direction,
        "coverage": coverage_met is True,
    }
    passes = all(checks.values())
    return {
        "passes": passes,
        "claim": ("trainability_debt" if direction == 1 else "plasticity_restoration") if passes else None,
        "direction": direction, "tdelta_direction": absolute_direction,
        "practically_meaningful_direction": relative_direction if checks["relative_practical_margin"] else 0 if available else None,
        "measurement_available": available,
        "checks": checks, "orders": orders,
    }


def global_trainability_claim(structured, language):
    """SPEC §17.2: retain domain disagreement, not a single plasticity number.

One passing domain can support a global direction only when the other domain
is measured with adequate coverage and has no opposite practically meaningful
primary effect or opposite above-noise companion-clock effect.
"""
    probes = (structured, language)
    if not all(probe["measurement_available"] and probe["checks"]["coverage"] for probe in probes):
        return {"passes": False, "claim": None, "status": "insufficient_probe_coverage"}
    passing = [probe for probe in probes if probe["passes"]]
    if not passing:
        return {"passes": False, "claim": None, "status": "no_domain_claim"}
    direction = passing[0]["direction"]
    opposite = any(
        probe["practically_meaningful_direction"] == -direction
        or (probe["tdelta_direction"] == -direction and probe["checks"]["tdelta_above_noise"])
        for probe in probes
    )
    if opposite:
        return {"passes": False, "claim": None, "status": "domain_dependent"}
    return {"passes": True, "claim": passing[0]["claim"], "status": "global", "direction": direction}


def retention_claim(effect, noise_bound, order_effects):
    """SPEC §17.3 only; a pooled headline also needs §16 manipulation coverage."""
    _nonnegative("retention M1 noise bound", noise_bound)
    direction = _direction(effect)
    orders = _order_agreement(order_effects, direction)
    checks = {
        "practical_margin": effect is not None and abs(effect) > 0.05,
        "above_noise": effect is not None and abs(effect) > noise_bound,
        "order_agreement": orders["consistent"] >= 3,
    }
    passes = all(checks.values())
    return {"passes": passes, "claim": ("retention_increase" if direction == 1 else "retention_loss") if passes else None,
            "direction": direction, "checks": checks, "orders": orders}


def diversity_claim(gap_effect, noise_bound, order_effects, strategy_family_direction):
    """SPEC §17.4; strategy direction must come from coverage or concentration.

Supply +1 for greater family diversity, -1 for less, 0 for no directional
evidence, or None for unavailable. Surprisal alone is not strategy evidence.
A pooled headline separately requires §16 manipulation coverage.
"""
    _nonnegative("diversity M1 noise bound", noise_bound)
    if strategy_family_direction not in {-1, 0, 1, None}:
        raise ValueError("strategy-family evidence must have an oriented direction")
    direction = _direction(gap_effect)
    orders = _order_agreement(order_effects, direction)
    checks = {
        "practical_margin": gap_effect is not None and abs(gap_effect) > 0.03,
        "above_noise": gap_effect is not None and abs(gap_effect) > noise_bound,
        "order_agreement": orders["consistent"] >= 3,
        "strategy_family_evidence": direction in {-1, 1} and strategy_family_direction == direction,
    }
    passes = all(checks.values())
    return {"passes": passes, "claim": ("diversity_increase" if direction == 1 else "diversity_loss") if passes else None,
            "direction": direction, "checks": checks, "orders": orders}
