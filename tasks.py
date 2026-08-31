"""Frozen, synthetic v2 acquisition candidates and exact binary checkers.

``seed`` is a data-instance index, never an independent training seed. Each
candidate enumerates a finite mathematical instance space through a fixed
bijective permutation. Keys describe instances, not their index or wording.
No dataset, model, network, or outcome-dependent configuration is consulted.

The Fraction/AST arithmetic helpers below are selectively adapted from
kintsugi-v1 tasks.py at d2aa7cd94a0a618169496f0235fa021ea46c0372. No v1 task
families, datasets, observations, or orchestration are imported.
"""

import ast
from fractions import Fraction
import itertools
import json
import math
import re
import sqlite3
import xml.etree.ElementTree as ET


CANDIDATES = (
    "arithmetic_derivations", "equation_derivations", "nl_sql",
    "spreadsheet_formulas", "json_extraction", "xml_extraction",
    "base_conversion", "modular_numerals", "string_programs",
    "finite_state_rewrite",
)
INSTANCE_LIMITS = {
    "arithmetic_derivations": math.comb(24, 4) * 32,
    "equation_derivations": 32 * 128 * 4096,
    "nl_sql": 4 * 128 * 256 * 4 * 4,
    "spreadsheet_formulas": 3 * 4 * 28 * 32 * 32,
    "json_extraction": 16 * 16 * 8 * 128 * 2,
    "xml_extraction": 16 * 16 * 8 * 128 * 2,
    "base_conversion": 65536 * 8 * 7,
    "modular_numerals": 65536 * 8 * 32 * 4 * 4,
    "string_programs": 8**8 * 7,
    "finite_state_rewrite": 2**16 * 4 * 2,
}

_BASES = (2, 3, 5, 7, 8, 10, 12, 16)
_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_REGIONS = ("north", "south", "east", "west")
_FIRST_NAMES = (
    "Ada", "Bea", "Cora", "Dara", "Eli", "Faye", "Gus", "Hana",
    "Ivo", "Jia", "Kira", "Leon", "Mira", "Noel", "Omar", "Pia",
)
_LAST_NAMES = (
    "Arden", "Bell", "Cole", "Dunn", "Evans", "Ford", "Gray", "Hale",
    "Ives", "Jones", "King", "Lane", "Moss", "Nash", "Owen", "Park",
)
_CITIES = ("York", "Bath", "Leeds", "Derby", "Truro", "Exeter", "Perth", "Dover")
_CELLS = tuple(f"{column}{row}" for column in "ABCD" for row in range(1, 9))
_ROW_RANGES = tuple(itertools.combinations(range(1, 9), 2))
_SHEET_WITNESSES = tuple(
    {cell: (index * factor) % 32 + 1 + offset for index, cell in enumerate(_CELLS)}
    for factor, offset in ((1, 0), (7, 32), (13, 64))
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def semantic_key(example):
    """Return the generated, canonical mathematical/structured instance key."""
    return example["semantic_key"]


def _example(candidate, prompt, completion, instance, payload):
    return {
        "candidate": candidate,
        "prompt": prompt,
        "completion": completion,
        "semantic_key": _canonical(instance),
        "checker_payload": payload,
    }


def _unpack(index, *radices):
    values = []
    for radix in radices:
        index, value = divmod(index, radix)
        values.append(value)
    if index:
        raise ValueError("instance index overflow")
    return values


def _combination(index, size=4, low=2, high=25):
    """Unrank increasing operands without an RNG or rejection sampling."""
    result = []
    for remaining in range(size - 1, -1, -1):
        for value in range(low, high + 1):
            count = math.comb(high - value, remaining)
            if index < count:
                result.append(value)
                low = value + 1
                break
            index -= count
    return result


# Selective v1 reuse: exact Fraction arithmetic, allowlisted AST parsing and
# evaluation. The instance families and derivation validation are new in v2.
def _apply(left, right, op):
    if op == "+":
        value = left + right
    elif op == "-":
        value = left - right
    elif op == "*":
        value = left * right
    elif op == "/" and right:
        value = left / right
    else:
        raise ZeroDivisionError
    if abs(value) > 10_000 or value.denominator > 1_000:
        raise ValueError("arithmetic guard")
    return value


def _fraction_text(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _parse_answer(answer):
    if not isinstance(answer, str) or not 1 <= len(answer) <= 1024:
        raise ValueError("answer length")
    if any(character not in "0123456789+-*/() \t\r\n\v\f" for character in answer):
        raise ValueError("answer alphabet")
    depth = maximum = 0
    for character in answer:
        depth += character == "("
        depth -= character == ")"
        if depth < 0:
            raise ValueError("parentheses")
        maximum = max(maximum, depth)
    if depth or maximum > 64:
        raise ValueError("parentheses")
    tokens = re.findall(r"\d+|[()+\-*/]", answer)
    if "".join(tokens) != "".join(answer.split()):
        raise ValueError("tokenization")
    normalized = " ".join(str(int(token)) if token.isdigit() else token for token in tokens)
    return ast.parse(normalized, mode="eval").body


def _evaluate(node):
    if isinstance(node, ast.Constant) and type(node.value) is int and node.value >= 0:
        return Fraction(node.value), [node.value], 1, 1
    if not isinstance(node, ast.BinOp) or type(node.op) not in (ast.Add, ast.Sub, ast.Mult, ast.Div):
        raise ValueError("AST node")
    left, left_leaves, left_nodes, left_depth = _evaluate(node.left)
    right, right_leaves, right_nodes, right_depth = _evaluate(node.right)
    op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(node.op)]
    value = _apply(left, right, op)
    nodes, depth = left_nodes + right_nodes + 1, max(left_depth, right_depth) + 1
    if nodes > 31 or depth > 5:
        raise ValueError("AST guard")
    return value, left_leaves + right_leaves, nodes, depth


def _read_fraction(text):
    if not re.fullmatch(r"-?\d{1,8}(?:/[1-9]\d{0,7})?", text.strip()):
        raise ValueError("fraction")
    return Fraction(text.strip())


def _arithmetic(index):
    left_op, right_op, root_op, operands_index = _unpack(index, 4, 4, 2, math.comb(24, 4))
    a, b, c, d = _combination(operands_index)
    left = f"({a}{'+-*/'[left_op]}{b})"
    right = f"({c}{'+-*/'[right_op]}{d})"
    expression = f"({left}{'+-'[root_op]}{right})"
    expressions = [left, right, expression]
    work = [f"{part} = {_fraction_text(_evaluate(_parse_answer(part))[0])}" for part in expressions]
    completion = "\n".join(work) + f"\n<answer>{expression}</answer>"
    prompt = (
        f"Derive the exact rational value of {expression}. Return exactly four lines: "
        "the left child expression = its reduced value; the right child expression = its reduced value; "
        "the original whole expression = its reduced value; then <answer>the original whole expression</answer>. "
        "Repeat the unsimplified expressions on the left of each equality. Use integers or reduced fractions; no prose."
    )
    return _example("arithmetic_derivations", prompt, completion,
                    ["arithmetic_derivation", expression], {"expressions": expressions})


def _verify_arithmetic(payload, text):
    lines = text.strip().splitlines()
    if len(lines) != 4:
        return False
    for line, expression in zip(lines[:3], payload["expressions"]):
        if line.count("=") != 1:
            return False
        lhs, rhs = line.split("=")
        node = _parse_answer(lhs)
        expected = _parse_answer(expression)
        if ast.dump(node) != ast.dump(expected) or _evaluate(node)[0] != _read_fraction(rhs):
            return False
        if rhs.strip() != _fraction_text(_read_fraction(rhs)):
            return False
    match = re.fullmatch(r"<answer>([^<>]+)</answer>", lines[3].strip())
    return bool(match and ast.dump(_parse_answer(match[1])) == ast.dump(_parse_answer(payload["expressions"][-1])))


def _equation(index):
    a, b, c = _unpack(index, 32, 128, 4096)
    a, b, c = a + 2, b + 1, c - 2048
    result = Fraction(c - b, a)
    completion = f"{a}*x = {c - b}\nx = {_fraction_text(result)}"
    prompt = (
        f"Solve {a}*x + {b} = {c} by an exact equation derivation. Return exactly two lines: "
        f"first subtract {b} from both sides and simplify to {a}*x = integer; "
        f"then divide both sides by {a} and simplify to x = reduced integer/fraction. No other text."
    )
    return _example("equation_derivations", prompt, completion,
                    ["equation_derivation", a, b, c], {"a": a, "b": b, "c": c})


def _verify_equation(payload, text):
    if len(text.strip().splitlines()) != 2:
        return False
    match = re.fullmatch(r"\s*(\d+)\s*\*\s*x\s*=\s*(-?\d+)\s*\nx\s*=\s*(-?\d+(?:/\d+)?)\s*", text)
    if not match:
        return False
    a, b, c = (payload[name] for name in ("a", "b", "c"))
    result = _read_fraction(match[3])
    return int(match[1]) == a and int(match[2]) == c - b and result == Fraction(c - b, a) and match[3] == _fraction_text(result)


def _sql_witnesses(region, units, price):
    witnesses = []
    for number in range(3):
        rows = [
            [1, region, "apple", units + 1, price - 1],
            [2, region, "pear", units + 2, price - 2],
            [3, region, "boundary_units", units, price - 1],
            [4, region, "boundary_price", units + 1, price],
            [5, region, "plum", units - 1, price - 1],
            [6, region, "plum", units + 1, price + 1],
            [7, _REGIONS[(_REGIONS.index(region) + 1) % 4], "outside", units + 4, price - 1],
        ]
        # Different qualifying products make fixed literal-answer SQL fail even
        # for COUNT or MAX. Boundary rows test every registered predicate.
        for extra in range(number):
            rows.append([8 + extra, region, ("cocoa", "melon")[extra], units + 3 + extra, price - 3])
        witnesses.append(rows)
    return witnesses


def _sql_expected(rows, region, units, price, comparisons, aggregation):
    groups = {}
    for _, row_region, product, row_units, row_price in rows:
        unit_ok = row_units >= units if comparisons < 2 else row_units > units
        price_ok = row_price <= price if comparisons % 2 == 0 else row_price < price
        if row_region == region and unit_ok and price_ok:
            groups.setdefault(product, []).append((row_units, row_price))
    result = []
    for product, values in sorted(groups.items()):
        scores = (sum(u * p for u, p in values), sum(u for u, _ in values), len(values), max(p for _, p in values))
        result.append([product, scores[aggregation]])
    return result


def _sql(index):
    region, units, price, comparisons, aggregation = _unpack(index, 4, 128, 256, 4, 4)
    region, units, price = _REGIONS[region], units + 2, price + 10
    unit_op = ">=" if comparisons < 2 else ">"
    price_op = "<=" if comparisons % 2 == 0 else "<"
    expression = ("SUM(units * price_cents)", "SUM(units)", "COUNT(*)", "MAX(price_cents)")[aggregation]
    description = ("total revenue in cents (units times price_cents)", "total units", "row count", "maximum price_cents")[aggregation]
    completion = (
        f"SELECT product, {expression} FROM sales WHERE region = '{region}' "
        f"AND units {unit_op} {units} AND price_cents {price_op} {price} GROUP BY product ORDER BY product;"
    )
    witnesses = _sql_witnesses(region, units, price)
    expected = [_sql_expected(rows, region, units, price, comparisons, aggregation) for rows in witnesses]
    prompt = (
        "SQLite schema: sales(id INTEGER, region TEXT, product TEXT, units INTEGER, price_cents INTEGER). "
        "Every column is non-NULL. Return one read-only SELECT query and nothing else. "
        f"For rows in region '{region}' with units {unit_op} {units} and price_cents {price_op} {price}, "
        f"return each product and its {description}, grouped by product and sorted by product ascending. "
        "The query must work for any contents of this schema."
    )
    payload = {"witnesses": witnesses, "expected": expected}
    return _example("nl_sql", prompt, completion,
                    ["sql_query", region, units, price, unit_op, price_op, expression], payload)


def _run_sql(query, rows):
    if not isinstance(query, str) or len(query) > 4096 or not re.match(r"\s*SELECT\b", query, re.IGNORECASE):
        raise ValueError("one SELECT query required")
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE sales(id INTEGER, region TEXT, product TEXT, units INTEGER, price_cents INTEGER)")
        connection.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?)", rows)
        connection.execute("PRAGMA query_only = ON")
        if hasattr(connection, "setlimit"):
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 8192)
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 4096)
            connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 16)
            connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 24)
            connection.setlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 1)

        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and arg1 == "sales" and arg2 in ("id", "region", "product", "units", "price_cents", ""):
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in ("sum", "count", "min", "max", "abs", "coalesce"):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        remaining = 40

        def progress():
            nonlocal remaining
            remaining -= 1
            return int(remaining <= 0)

        connection.set_authorizer(authorize)
        connection.set_progress_handler(progress, 1000)
        cursor = connection.execute(query)
        result = cursor.fetchmany(33)
        if len(result) > 32 or any(len(row) != 2 for row in result):
            raise ValueError("SQL result bound")
        return [list(row) for row in result]
    finally:
        connection.close()


def _verify_sql(payload, text):
    if len(payload["witnesses"]) != 3 or len(payload["expected"]) != 3:
        return False
    return all(_run_sql(text, rows) == expected for rows, expected in zip(payload["witnesses"], payload["expected"]))


def _range_cells(start, end):
    if start not in _CELLS or end not in _CELLS or start[0] > end[0] or start[1] > end[1]:
        raise ValueError("spreadsheet range")
    return [f"{column}{row}" for column in "ABCD" if start[0] <= column <= end[0]
            for row in range(int(start[1]), int(end[1]) + 1)]


def _formula_value(formula, sheet):
    """Interpret a small spreadsheet grammar; never Python eval or exec."""
    if not isinstance(formula, str) or not formula.startswith("=") or len(formula) > 512:
        raise ValueError("formula envelope")
    source = formula[1:].strip()
    if not re.fullmatch(r"[A-Z0-9+*/(),:\s.\-]+", source) or re.search(r"\bRANGE\b", source):
        raise ValueError("formula alphabet")
    source = re.sub(r"\b([A-D][1-8])\s*:\s*([A-D][1-8])\b", r"RANGE(\1,\2)", source)
    tree = ast.parse(source, mode="eval").body
    if len(list(ast.walk(tree))) > 96:
        raise ValueError("formula AST bound")

    def visit(node, depth=0):
        if depth > 12:
            raise ValueError("formula depth")
        if isinstance(node, ast.Name) and node.id in _CELLS:
            return Fraction(sheet[node.id])
        if isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 10000:
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            value = visit(node.operand, depth + 1)
            if not isinstance(value, Fraction):
                raise ValueError("range operand")
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if not isinstance(left, Fraction) or not isinstance(right, Fraction):
                raise ValueError("range operand")
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            else:
                value = left / right
            if abs(value) > 10**12 or value.denominator > 10**12:
                raise ValueError("formula numeric bound")
            return value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            if node.func.id == "RANGE":
                if len(node.args) != 2 or not all(isinstance(arg, ast.Name) for arg in node.args):
                    raise ValueError("range arguments")
                return [Fraction(sheet[cell]) for cell in _range_cells(node.args[0].id, node.args[1].id)]
            if node.func.id not in ("SUM", "MIN", "MAX") or not 1 <= len(node.args) <= 32:
                raise ValueError("spreadsheet function")
            values = []
            for arg in node.args:
                value = visit(arg, depth + 1)
                values.extend(value if isinstance(value, list) else [value])
            return {"SUM": sum, "MIN": min, "MAX": max}[node.func.id](values)
        raise ValueError("formula AST node")

    value = visit(tree)
    if not isinstance(value, Fraction):
        raise ValueError("scalar formula required")
    return value


def _spreadsheet(index):
    function, column, rows, scale, offset = _unpack(index, 3, 4, 28, 32, 32)
    function = ("SUM", "MIN", "MAX")[function]
    column = "ABCD"[column]
    start, end = _ROW_RANGES[rows]
    selected_range = f"{column}{start}:{column}{end}"
    scale, offset = _CELLS[scale], _CELLS[offset]
    completion = f"={function}({selected_range})*{scale}+{offset}"
    prompt = (
        "Write a spreadsheet formula for a sheet with numeric cells A1:D8. "
        f"Take the { {'SUM': 'sum', 'MIN': 'minimum', 'MAX': 'maximum'}[function]} of {selected_range}, "
        f"multiply it by cell {scale}, then add cell {offset}. Return only the formula beginning with '='. "
        "Allowed syntax: integer constants, cell references, rectangular ranges in SUM/MIN/MAX, "
        "+, -, *, /, and parentheses. It must work for arbitrary numeric cell values."
    )
    expected = [_fraction_text(_formula_value(completion, sheet)) for sheet in _SHEET_WITNESSES]
    return _example("spreadsheet_formulas", prompt, completion,
                    ["spreadsheet_formula", function, selected_range, scale, offset], {"expected": expected})


def _verify_spreadsheet(payload, text):
    if len(payload["expected"]) != 3:
        return False
    return all(_formula_value(text.strip(), sheet) == Fraction(expected)
               for sheet, expected in zip(_SHEET_WITNESSES, payload["expected"]))


def _record(index):
    first, last, city, units, active = _unpack(index, 16, 16, 8, 128, 2)
    return {"name": f"{_FIRST_NAMES[first]} {_LAST_NAMES[last]}", "city": _CITIES[city], "units": units + 1, "active": bool(active)}


def _extraction(candidate, index):
    record = _record(index)
    source = (
        f"Current customer: {record['name']}; current city: {record['city']}; "
        f"confirmed units: {record['units']}; account status: {'active' if record['active'] else 'inactive'}. "
        f"Ignore the previous order (units: {record['units'] + 3}) and archived city: Lincoln."
    )
    if candidate == "json_extraction":
        completion = _canonical(record)
        instruction = (
            "Extract the current record as one strict JSON object with exactly keys name, city, units, active. "
            "name and city are strings, units is an integer, active is a JSON boolean. No extra keys, repeated keys, or prose. "
        )
    else:
        completion = "<record>" + "".join(
            f"<{key}>{str(value).lower() if type(value) is bool else value}</{key}>" for key, value in record.items()
        ) + "</record>"
        instruction = (
            "Extract the current record as strict XML: one record root with exactly these children in order: "
            "name, city, units, active. units is decimal; active is true or false. "
            "No attributes, extra elements, declarations, entities, comments, or prose. "
        )
    # Both formats share an underlying extraction-instance key: changing only
    # JSON to XML must not make the same source record a new semantic instance.
    return _example(candidate, instruction + source, completion,
                    ["record_extraction", record], {"record": record})


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _verify_json(payload, text):
    record = json.loads(text, object_pairs_hook=_unique_object)
    expected = payload["record"]
    return type(record) is dict and record.keys() == expected.keys() and all(
        type(record[key]) is type(value) and record[key] == value for key, value in expected.items()
    )


def _verify_xml(payload, text):
    if any(marker in text for marker in ("<!", "<?", "&")):
        return False
    root = ET.fromstring(text)
    expected = payload["record"]
    if root.tag != "record" or root.attrib or (root.text or "").strip() or [child.tag for child in root] != list(expected):
        return False
    for child, value in zip(root, expected.values()):
        target = str(value).lower() if type(value) is bool else str(value)
        if child.attrib or list(child) or (child.tail or "").strip() or child.text != target:
            return False
    return True


def _numeral(number, base, width=0):
    digits = []
    while number:
        number, digit = divmod(number, base)
        digits.append(_DIGITS[digit])
    return ("".join(reversed(digits)) or "0").rjust(width, "0")


def _base(index):
    number, source, target = _unpack(index, 65536, 8, 7)
    number += 1
    target += target >= source
    source, target = _BASES[source], _BASES[target]
    completion = _numeral(number, target)
    prompt = (
        f"Convert {_numeral(number, source)} from base {source} to base {target}. "
        "Return only the target numeral, uppercase A=10, B=11, etc.; no prefix, spaces, or leading zeros."
    )
    return _example("base_conversion", prompt, completion,
                    ["base_conversion", number, source, target], {"answer": completion})


def _modular(index):
    number, multiplier, offset, modulus, target = _unpack(index, 65536, 8, 32, 4, 4)
    multiplier, modulus, target = multiplier + 2, (97, 101, 103, 107)[modulus], (2, 7, 10, 16)[target]
    answer = _numeral((multiplier * number + offset) % modulus, target)
    prompt = (
        f"Let n be the hexadecimal numeral {_numeral(number, 16)}. Compute ({multiplier}*n + {offset}) mod {modulus}, "
        f"using the nonnegative remainder, then express the result in base {target}. "
        "Constants other than n are decimal. Return only the target numeral; uppercase digits and no leading zeros."
    )
    return _example("modular_numerals", prompt, answer,
                    ["modular_numeral", number, multiplier, offset, modulus, target], {"answer": answer})


def _string(index):
    input_index, rotation = _unpack(index, 8**8, 7)
    source = _numeral(input_index, 8, 8).translate(str.maketrans("01234567", "abcdefgh"))
    rotation += 1
    value = source[::-1]
    value = value[rotation:] + value[:rotation]
    answer = value.translate(str.maketrans("abcdefgh", "bcdefgha"))
    prompt = (
        f"Execute this string program on '{source}': (1) reverse the string; "
        f"(2) rotate left by {rotation} positions; "
        "(3) simultaneously substitute a->b, b->c, c->d, d->e, e->f, f->g, g->h, h->a. "
        "Return only the final string, without quotes or spaces."
    )
    return _example("string_programs", prompt, answer,
                    ["string_program", source, "reverse", rotation, "bcdefgha"], {"answer": answer})


def _finite_state(index):
    input_index, start, variant = _unpack(index, 2**16, 4, 2)
    source = _numeral(input_index, 2, 16)
    transitions = []
    for state in range(4):
        transitions.append([
            [(state + 1) % 4, "abcd"[(state + variant) % 4]],
            [(state + 2 + variant) % 4, "abcd"[(3 - state + variant) % 4]],
        ])
    state, output = start, []
    for bit in source:
        state, emitted = transitions[state][int(bit)]
        output.append(emitted)
    table = "\n".join(f"state {state}, input {bit}: next {transitions[state][bit][0]}, emit {transitions[state][bit][1]}"
                      for state in range(4) for bit in range(2))
    answer = "".join(output)
    prompt = (
        f"Run this deterministic finite-state transducer left to right on {source}, starting in state {start}. "
        f"For each bit, emit one character and enter the listed next state.\n{table}\n"
        "Return only the emitted string, without quotes, spaces, or the final state."
    )
    return _example("finite_state_rewrite", prompt, answer,
                    ["finite_state_transduction", source, start, transitions], {"answer": answer})


def make_example(candidate, seed):
    """Build one deterministic instance. Out-of-space indices fail, never wrap."""
    if candidate not in CANDIDATES or type(seed) is not int or not 0 <= seed < INSTANCE_LIMITS[candidate]:
        raise ValueError("unknown candidate or out-of-range data-instance index")
    # 104729 is coprime to every declared space size: this is a bijection,
    # interleaving mathematical parameters without random collision repair.
    index = (seed * 104729 + 8191) % INSTANCE_LIMITS[candidate]
    if candidate == "arithmetic_derivations":
        return _arithmetic(index)
    if candidate == "equation_derivations":
        return _equation(index)
    if candidate == "nl_sql":
        return _sql(index)
    if candidate == "spreadsheet_formulas":
        return _spreadsheet(index)
    if candidate in ("json_extraction", "xml_extraction"):
        return _extraction(candidate, index)
    if candidate == "base_conversion":
        return _base(index)
    if candidate == "modular_numerals":
        return _modular(index)
    if candidate == "string_programs":
        return _string(index)
    return _finite_state(index)


def verify(example, text):
    """Exact deterministic binary success, including all required derivation steps."""
    if not isinstance(text, str) or not text or len(text) > 8192:
        return False
    try:
        candidate, payload = example["candidate"], example["checker_payload"]
        if candidate == "arithmetic_derivations":
            return bool(_verify_arithmetic(payload, text))
        if candidate == "equation_derivations":
            return bool(_verify_equation(payload, text))
        if candidate == "nl_sql":
            return bool(_verify_sql(payload, text))
        if candidate == "spreadsheet_formulas":
            return bool(_verify_spreadsheet(payload, text))
        if candidate == "json_extraction":
            return bool(_verify_json(payload, text))
        if candidate == "xml_extraction":
            return bool(_verify_xml(payload, text))
        if candidate in ("base_conversion", "modular_numerals", "string_programs", "finite_state_rewrite"):
            return text.strip() == payload["answer"]
        return False
    except (ValueError, TypeError, KeyError, SyntaxError, ZeroDivisionError, RecursionError, OverflowError, sqlite3.Error, sqlite3.Warning, ET.ParseError):
        return False
