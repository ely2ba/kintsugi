"""Pure USD arithmetic for measured costs of the complete registered M2 design.

No price lookup, API calls, files, new funding tiers, or inferred spending caps.
Caller-supplied task/measurement unit costs are already USD. Token prices are
explicit USD per million tokens and must come from the caller's frozen source.
"""

import math

from backend import ACCOUNTING_KEYS
from protocol import ARMS, ORDERS, TASK_SLOTS


PRICE_KEYS = ("train", "prefill", "cached", "sample")
TASK_UNIT_KEYS = ("learn_to_competence", "learn_to_damage", "repair", "native_heldout")
MEASUREMENT_UNIT_KEYS = ("structured_probe", "language_probe", "if_heldout", "diversity", "kl")


def _amount(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is outside finite numeric range") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _units(values, keys, name):
    if type(values) is not dict or set(values) != set(keys):
        raise ValueError(f"{name} requires exactly: {', '.join(keys)}")
    return {key: _amount(values[key], f"{name}.{key}") for key in keys}


def _sum(values, name):
    try:
        total = math.fsum(values)
    except OverflowError as error:
        raise ValueError(f"{name} exceeds finite numeric range") from error
    return _amount(total, name)


def token_cost(accounting, prices):
    """Estimated USD, using the backend's explicit non-overlapping token fields.

    Prices have exactly train/prefill/cached/sample keys, in USD per 1M tokens.
    Absent known accounting fields mean zero; unknown fields are rejected.
    Forward tokens use the training price. Scoring-prefill and the scorer's
    discarded one-token sample estimate retain their respective billing prices.
    Gradient target tokens measure exposure and are never billed a second time.
    """
    prices = _units(prices, PRICE_KEYS, "prices_usd_per_million_tokens")
    if type(accounting) is not dict or set(accounting) - set(ACCOUNTING_KEYS):
        raise ValueError("unknown token-accounting fields")
    tokens = {key: accounting.get(key, 0) for key in ACCOUNTING_KEYS}
    if any(type(value) is not int or value < 0 for value in tokens.values()):
        raise ValueError("token counts must be nonnegative integers")
    billable = {
        "train": tokens["train_tokens"] + tokens["forward_tokens"],
        "prefill": tokens["prefill_tokens"] + tokens["scoring_prefill_tokens"],
        "cached": tokens["cached_tokens"],
        "sample": tokens["sample_tokens"] + tokens["scoring_discarded_sample_tokens_estimate"],
    }
    try:
        return _sum((count / 1_000_000 * prices[key] for key, count in billable.items()), "token cost")
    except OverflowError as error:
        raise ValueError("token cost exceeds finite numeric range") from error


def _design_counts():
    """Enumerate physical A/B states, including each acquired task at each state."""
    per_task = {slot: {"learn_to_competence": 0, "learn_to_damage": 0, "repair": 0,
                       "native_heldout_total": 0, "native_heldout_included": 0}
                for slot in TASK_SLOTS}
    counts = {"lineages": 0, "lineage_cycles": 0, "repair_opportunities": 0,
              "learn_only_physical_checkpoints": 0, "repair_arm_physical_checkpoints": 0}
    for sequence in ORDERS.values():
        for arm in ARMS:
            counts["lineages"] += 1
            acquired = []
            for slot in sequence:
                acquired.append(slot)
                counts["lineage_cycles"] += 1
                learning_unit = "learn_to_competence" if arm == "learn-only" else "learn_to_damage"
                per_task[slot][learning_unit] += 1
                # The selected-current-task A held-out check is already charged
                # inside this lineage-cycle's learning-phase unit cost.
                per_task[slot]["native_heldout_included"] += 1
                if arm == "learn-only":
                    physical_states = ("A",)
                    counts["learn_only_physical_checkpoints"] += 1
                else:
                    physical_states = ("A", "B")
                    counts["repair_arm_physical_checkpoints"] += 2
                    counts["repair_opportunities"] += 1
                    per_task[slot]["repair"] += 1
                for _ in physical_states:
                    for acquired_slot in acquired:
                        per_task[acquired_slot]["native_heldout_total"] += 1
    counts["physical_checkpoints"] = counts["learn_only_physical_checkpoints"] + counts["repair_arm_physical_checkpoints"]
    for record in per_task.values():
        record["native_heldout_additional"] = record["native_heldout_total"] - record["native_heldout_included"]
    counts["task_heldout_evaluations"] = sum(record["native_heldout_total"] for record in per_task.values())
    counts["current_A_heldout_in_learning"] = sum(record["native_heldout_included"] for record in per_task.values())
    counts["additional_task_heldout"] = sum(record["native_heldout_additional"] for record in per_task.values())
    counts["per_task"] = per_task
    return counts


def project_m2(task_units, measurement_units, prices=None):
    """Project all twelve lineages from measured USD units, without rounding up.

    task_units[T1..T7] must provide learn_to_competence, learn_to_damage, repair,
    and native_heldout. Learning units include start measurements, optimizer
    updates and evaluations, and the selected A current-task held-out check.
    Repair units include all scheduled five-update protected-criterion checks.

    measurement_units provides structured_probe, language_probe, if_heldout,
    diversity, and kl costs per physical checkpoint. Optional prices are a
    validated provenance snapshot only: these already-USD units are not repriced.
    """
    if type(task_units) is not dict or set(task_units) != set(TASK_SLOTS):
        raise ValueError("task_units must contain exactly the seven registered task slots")
    task_units = {slot: _units(task_units[slot], TASK_UNIT_KEYS, f"task_units.{slot}") for slot in TASK_SLOTS}
    measurement_units = _units(measurement_units, MEASUREMENT_UNIT_KEYS, "measurement_units")
    prices = None if prices is None else _units(prices, PRICE_KEYS, "prices_usd_per_million_tokens")
    counts = _design_counts()
    lines = []

    def add(category, item, unit, count, unit_usd, task=None):
        lines.append({"category": category, "item": item, "task": task, "unit": unit,
                      "count": count, "unit_usd": unit_usd,
                      "subtotal_usd": _amount(count * unit_usd, f"{item} subtotal")})

    for slot, units in task_units.items():
        count = counts["per_task"][slot]
        add("acquisition", f"{slot}: learn-only to competence", "learning phase", count["learn_to_competence"], units["learn_to_competence"], slot)
        add("acquisition", f"{slot}: repair-arm learning to damage", "learning phase", count["learn_to_damage"], units["learn_to_damage"], slot)
        add("repair", f"{slot}: scheduled repair", "repair opportunity", count["repair"], units["repair"], slot)
        add("retention", f"{slot}: additional native held-out evaluation", "task evaluation", count["native_heldout_additional"], units["native_heldout"], slot)
    for name, unit_usd in measurement_units.items():
        add("checkpoint_measurements", name.replace("_", " "), "physical checkpoint",
            counts["physical_checkpoints"], unit_usd)
    subtotals = {category: _sum((line["subtotal_usd"] for line in lines if line["category"] == category), category)
                 for category in ("acquisition", "repair", "retention", "checkpoint_measurements")}
    return {
        "currency": "USD",
        "line_items": lines,
        "subtotals_usd": subtotals,
        "total_usd": _sum((line["subtotal_usd"] for line in lines), "M2 total"),
        "counts": counts,
        "prices_usd_per_million_tokens": prices,
        "assumptions": [
            "Complete registered M2 design: four fixed orders, three arms per order, seven tasks and cycles; no cases removed for budget.",
            "All 56 scheduled repair opportunities are projected as actual repairs: 28 learn-only A states plus 112 repair-arm A/B states, or 140 physical checkpoints.",
            "No-op repairs would reduce actual cost and physical checkpoint measurements; they do not reduce this full-design projection.",
            "Learning units include start measurements, per-update evaluations, and the current-task held-out check at A; repair units include scheduled criterion checks.",
            "Native acquired-task held-out evaluations total 560; 84 current-task A checks are included in learning units, leaving 476 additional evaluations.",
            "Cycle-0 measurements are reused from M1 and are not charged again; each probe unit is the complete registered probe run at one physical checkpoint.",
            "Unit costs are measured estimates, not spending guarantees or new budget caps; lifecycle costs and scoring-token estimates may vary.",
            "Optional token prices document provenance only; already-USD unit costs are not multiplied by token prices again.",
        ],
    }
