"""Registered constants and fixed data schedules; no outcome-dependent defaults."""
import hashlib

MODEL = "Qwen/Qwen3.5-4B"
TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RENDERER = "qwen3_5_disable_thinking"
LORA_RANK = LORA_ALPHA = 32
LORA_SEED = 1337  # One initialization, not an independent replication seed.
MASTER_SEED = 20260831
LEARNING_RATES = (1e-5, 3e-5, 1e-4)
ACQUISITION_UPDATES = 120
REFERENCE_EVAL_STEPS = tuple(range(0, 121, 5))
WARMUP_UPDATES = 10
IF_MAX_TOKENS, IF_RECOVERY_FRACTION = 96, 0.95
REPAIR_GROUPS, REPAIR_ROLLOUTS, REPAIR_MAX_TOKENS = 64, 4, 4096
REPAIR_LR, REPAIR_CAP, REPAIR_CHECK_EVERY = 1e-4, 150, 5
CHECKPOINT_TTL = 90 * 24 * 60 * 60

TASK_SLOTS = {
    "T1": ("arithmetic_derivations", "equation_derivations"),
    "T2": ("wikipedia_is", "wikipedia_eu"),
    "T3": ("nl_sql", "spreadsheet_formulas"),
    "T4": ("json_extraction", "xml_extraction"),
    "T5": ("base_conversion", "modular_numerals"),
    "T6": ("string_programs", "finite_state_rewrite"),
    "T7": ("legal_text", "biomedical_abstracts"),
}
ORDERS = {
    "O1": ("T1", "T2", "T3", "T4", "T5", "T6", "T7"),
    "O2": ("T2", "T4", "T1", "T6", "T3", "T7", "T5"),
    "O3": ("T5", "T3", "T6", "T1", "T7", "T4", "T2"),
    "O4": ("T7", "T6", "T2", "T5", "T4", "T3", "T1"),
}
ARMS = ("learn-only", "fixed", "rolling")
PERSISTENCE_TASKS = ("T3", "T2", "T6")
STRUCTURED_PROBES = ("graph_path", "calendar_arithmetic", "unit_conversion")
LANGUAGE_PROBES = ("wikipedia_vi", "wikipedia_id", "wikipedia_fi")
PROBE_BUDGETS = {"structured": (32, 4), "language": (245, 25)}
IF_HASHES = {
    "all": "a51b4a12dab1103d135ac9fad931b3f0dfab3071dcf119067e93d45d6f2728e1",
    "criterion": "5defb4057d3fbc38ae8317f1704e69a6e735f84bd234e8d17b33aca10919c157",
    "heldout": "46ffb14c2fec5670ec4342dc3e883cd63b4e69b515b77fbe4bea717967b376c2",
}
REPAIR_POOL_HASH = "5b7fa61b96f0db6ff7163578a11d8a009024da6a28ba13345cc02ce9a0e3012d"


def schedule_seed(task, purpose, index=0):
    """Task-specific common randomness. Order/arm are deliberately not inputs."""
    if task not in TASK_SLOTS and task not in STRUCTURED_PROBES + LANGUAGE_PROBES:
        raise ValueError("schedule requires a registered task slot or fixed probe")
    if not isinstance(index, int) or index < 0:
        raise ValueError("schedule index must be a nonnegative integer")
    label = f"{MASTER_SEED}:{task}:{purpose}:{index}"
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big") & 0x7fffffff


def order_manifests():
    return {
        order: {
            "order": order,
            "slots": list(sequence),
            "lineages": [{"id": f"{arm}-{order}", "arm": arm} for arm in ARMS],
            "randomness": "task-keyed shared data, example order, evaluation and repair schedule",
            "cycles": [
                {"cycle": cycle, "slot": slot,
                 "task_candidates": list(TASK_SLOTS[slot]),
                 "task_schedule_key": slot, "repair_schedule_key": slot}
                for cycle, slot in enumerate(sequence, 1)
            ],
        }
        for order, sequence in ORDERS.items()
    }
