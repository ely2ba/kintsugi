"""Fixed synthetic probe and diversity candidates for SPEC §§9 and 11.

Probe training uses the single deterministic gold completion and native loss.
Exact checkers are diagnostics, not alternative learning metrics. Graph paths
accept any cheapest valid path; the supervised gold breaks ties lexicographically.

Diversity families are verifiable construction classes, not inferred reasoning:
graph coloring quotients color-label permutations into an unlabeled partition of
vertices; set partition quotients group and item order. Family counts are obtained
by exhaustive finite enumeration for each actual instance, never assumed from a
template or selected using a model outcome.
"""

from datetime import date, timedelta
from fractions import Fraction
import itertools
import json
import math


PROBE_CANDIDATES = ("graph_path", "calendar_arithmetic", "unit_conversion")
PROBE_INSTANCE_LIMITS = {
    "graph_path": 9**12,
    "calendar_arithmetic": 36525 * 732,
    "unit_conversion": 100000 * 30,
}
DIVERSITY_CANDIDATES = ("graph_coloring", "set_partition")
DIVERSITY_INSTANCE_LIMITS = {"graph_coloring": 256, "set_partition": 256 * 64}

_LAYERS = (("S",), ("A", "B"), ("C", "D"), ("E", "F"), ("T",))
_PATH_EDGES = tuple((left, right) for source, target in zip(_LAYERS, _LAYERS[1:])
                    for left in source for right in target)
_PATHS = tuple(tuple(path) for path in itertools.product(*_LAYERS))
_EPOCH = date(2000, 1, 1)
_UNITS = (
    ("length", "cable", (("millimetres", 1), ("centimetres", 10), ("metres", 1000), ("kilometres", 1000000))),
    ("mass", "sand", (("milligrams", 1), ("grams", 1000), ("kilograms", 1000000))),
    ("duration", "a process", (("seconds", 1), ("minutes", 60), ("hours", 3600), ("days", 86400))),
)
_UNIT_PAIRS = tuple((dimension, object_name, source, target)
                    for dimension, object_name, units in _UNITS
                    for source in units for target in units if source != target)
_CYCLE_EDGES = tuple(sorted((min(left, right), max(left, right))
                            for left, right in zip((0, 4, 1, 5, 2, 6, 3, 7), (4, 1, 5, 2, 6, 3, 7, 0))))
_EXTRA_EDGES = tuple((left, right) for left in range(4) for right in range(4, 8)
                     if (left, right) not in _CYCLE_EDGES)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _fraction_text(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _index(candidate, index, limits):
    if candidate not in limits or type(index) is not int or not 0 <= index < limits[candidate]:
        raise ValueError("unknown candidate or out-of-range data-instance index")
    if math.gcd(104729, limits[candidate]) != 1:
        raise ValueError("instance permutation must be bijective")
    return (index * 104729 + 8191) % limits[candidate]


def _example(candidate, prompt, completion, instance, payload):
    return {
        "candidate": candidate,
        "prompt": prompt,
        "completion": completion,
        "semantic_key": _canonical(instance),
        "checker_payload": payload,
    }


def _path_candidates(edges):
    costs = {(left, right): weight for left, right, weight in edges}
    return [(sum(costs[pair] for pair in zip(path, path[1:])), path) for path in _PATHS]


def _graph_path(index):
    edges = []
    for left, right in _PATH_EDGES:
        index, weight = divmod(index, 9)
        edges.append([left, right, weight + 1])
    cost, path = min(_path_candidates(edges))
    listing = "; ".join(f"{left}->{right}: {weight}" for left, right, weight in edges)
    prompt = (
        f"Directed weighted graph edges: {listing}. Construct a cheapest directed path from S to T. "
        "Path cost is the sum of edge weights. Any minimum-cost path is valid. "
        "Return only the visited node names separated by single spaces, including S and T."
    )
    return _example("graph_path", prompt, " ".join(path), ["directed_weighted_path", edges, "S", "T"],
                    {"edges": edges, "minimum_cost": cost})


def _calendar(index):
    offset_index, start_index = divmod(index, 36525)
    offset = offset_index - 366
    if offset >= 0:
        offset += 1
    start = _EPOCH + timedelta(days=start_index)
    target = start + timedelta(days=offset)
    prompt = (
        f"On the Gregorian calendar, start on {start.isoformat()} and move {abs(offset)} calendar days "
        f"{'forward' if offset > 0 else 'backward'}. The starting date is day zero. "
        "Return only the resulting date in YYYY-MM-DD format."
    )
    return _example("calendar_arithmetic", prompt, target.isoformat(),
                    ["calendar_day_offset", start.isoformat(), offset], {"answer": target.isoformat()})


def _unit(index):
    pair_index, quantity = divmod(index, 100000)
    quantity += 1
    dimension, object_name, source, target = _UNIT_PAIRS[pair_index]
    value = Fraction(quantity * source[1], target[1])
    answer = _fraction_text(value)
    prompt = (
        f"A technician records the {dimension} of {object_name} as {quantity} {source[0]}. "
        f"Convert this quantity to {target[0]}. Use exact standard metric and time-unit ratios. "
        "Return only the reduced integer or fraction, without a unit, decimal approximation, or explanation."
    )
    return _example("unit_conversion", prompt, answer,
                    ["unit_conversion", dimension, quantity, source[0], target[0]], {"answer": answer})


def make_probe(candidate, index):
    """Index denotes an instance. Split membership/order are frozen by callers."""
    index = _index(candidate, index, PROBE_INSTANCE_LIMITS)
    if candidate == "graph_path":
        return _graph_path(index)
    if candidate == "calendar_arithmetic":
        return _calendar(index)
    return _unit(index)


def verify_probe(example, text):
    if not isinstance(text, str) or not text or len(text) > 512:
        return False
    try:
        candidate, payload = example["candidate"], example["checker_payload"]
        if candidate == "graph_path":
            path = tuple(text.strip().split(" "))
            if path not in _PATHS or len(path) != len(set(path)):
                return False
            costs = {(left, right): weight for left, right, weight in payload["edges"]}
            return sum(costs[pair] for pair in zip(path, path[1:])) == payload["minimum_cost"]
        if candidate in ("calendar_arithmetic", "unit_conversion"):
            return text.strip() == payload["answer"]
        return False
    except (TypeError, ValueError, KeyError):
        return False


def _normalize_colors(colors):
    mapping = {}
    normalized = []
    for color in colors:
        if color not in mapping:
            mapping[color] = len(mapping)
        normalized.append(mapping[color])
    return normalized


def _color_families(edges):
    """Enumerate restricted-growth color vectors, once per color permutation."""
    preceding = {node: set() for node in range(8)}
    for left, right in edges:
        preceding[max(left, right)].add(min(left, right))
    families = []

    def extend(colors):
        node = len(colors)
        if node == 8:
            families.append(_canonical(colors))
            return
        for color in range(min(max(colors, default=-1) + 1, 2) + 1):
            if all(colors[other] != color for other in preceding[node]):
                extend(colors + [color])

    extend([])
    return tuple(sorted(families))


def _partition_families(values, target):
    """Enumerate all equal-sum 3x4 partitions, quotienting item/group order."""
    valid_groups = tuple(group for group in itertools.combinations(sorted(values), 4) if sum(group) == target)
    families = []

    def extend(remaining, groups):
        if not remaining:
            families.append(_canonical(sorted(groups)))
            return
        smallest = min(remaining)
        for group in valid_groups:
            group_set = frozenset(group)
            if smallest in group_set and group_set <= remaining:
                extend(remaining - group_set, groups + [list(group)])

    extend(frozenset(values), [])
    return tuple(sorted(families))


def enumerate_diversity_families(example):
    """Return exhaustive, unique canonical families for the actual item."""
    candidate, payload = example["candidate"], example["checker_payload"]
    if candidate == "graph_coloring":
        return _color_families(payload["edges"])
    if candidate == "set_partition":
        return _partition_families(payload["values"], payload["target"])
    raise ValueError("unknown diversity candidate")


def make_diversity(candidate, index):
    index = _index(candidate, index, DIVERSITY_INSTANCE_LIMITS)
    if candidate == "graph_coloring":
        edges = sorted(_CYCLE_EDGES + tuple(edge for bit, edge in enumerate(_EXTRA_EDGES) if index & (1 << bit)))
        edges = [list(edge) for edge in edges]
        prompt = (
            f"Undirected graph with vertices 0 through 7 and edges {_canonical(edges)}. "
            "Construct any proper vertex coloring using colors 0, 1, and 2 (at most three colors). "
            "Endpoints of every edge must differ. Return only a JSON array of eight integer colors, "
            "in vertex order 0 through 7."
        )
        example = _example(candidate, prompt, "", ["undirected_graph_coloring", edges, 3], {"edges": edges})
    else:
        step, first = divmod(index, 256)
        first, step = first + 1, step + 1
        values = [first + step * position for position in range(12)]
        target = sum(values) // 3
        prompt = (
            f"Partition the set {_canonical(values)} into exactly three groups of exactly four integers each, "
            f"with every group's sum equal to {target}. Use every given integer exactly once. "
            "Return only a JSON array containing the three arrays of four integers. Group and item order do not matter."
        )
        example = _example(candidate, prompt, "", ["equal_sum_set_partition", values, 3, 4, target],
                           {"values": values, "target": target})
    families = enumerate_diversity_families(example)
    if not families:
        raise ValueError("frozen diversity instance has no solution")
    example["completion"] = families[0]
    example["family_count"] = len(families)
    return example


def verify_diversity(example, text):
    """Return canonical output and item-local family, or None for any failure.

    Color-label permutations may be different outputs but are one family.
    Partition group/item permutations are one canonical output and one family.
    Family concentration should be computed within an item before aggregation.
    """
    if not isinstance(text, str) or not text or len(text) > 2048:
        return None
    try:
        candidate, payload = example["candidate"], example["checker_payload"]
        value = json.loads(text)
        if candidate == "graph_coloring":
            if type(value) is not list or len(value) != 8 or any(type(color) is not int or color not in (0, 1, 2) for color in value):
                return None
            if any(value[left] == value[right] for left, right in payload["edges"]):
                return None
            return {"canonical_solution": _canonical(value), "strategy_family": _canonical(_normalize_colors(value))}
        if candidate == "set_partition":
            if type(value) is not list or len(value) != 3 or any(type(group) is not list or len(group) != 4 for group in value):
                return None
            flattened = [item for group in value for item in group]
            if any(type(item) is not int for item in flattened) or sorted(flattened) != payload["values"]:
                return None
            if any(sum(group) != payload["target"] for group in value):
                return None
            canonical = _canonical(sorted(sorted(group) for group in value))
            return {"canonical_solution": canonical, "strategy_family": canonical}
        return None
    except (TypeError, ValueError, KeyError, RecursionError):
        return None
