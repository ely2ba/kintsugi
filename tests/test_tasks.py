from fractions import Fraction
import json
import math
import re
import sqlite3
import unittest

import tasks


class SyntheticTaskTests(unittest.TestCase):
    def test_fixed_candidates_spaces_and_index_validation(self):
        self.assertEqual(len(tasks.CANDIDATES), 10)
        for candidate in tasks.CANDIDATES:
            limit = tasks.INSTANCE_LIMITS[candidate]
            self.assertGreater(limit, 4 * 120 * 64 + 10000)
            self.assertEqual(math.gcd(104729, limit), 1)
            for seed in (-1, limit, True, 1.5, "1"):
                with self.subTest(candidate=candidate, seed=seed):
                    with self.assertRaises(ValueError):
                        tasks.make_example(candidate, seed)
        with self.assertRaises(ValueError):
            tasks.make_example("unknown", 0)

    def test_gold_is_correct_reproducible_and_serializable(self):
        for candidate in tasks.CANDIDATES:
            limit = tasks.INSTANCE_LIMITS[candidate]
            for seed in (*range(24), limit // 2, limit - 1):
                with self.subTest(candidate=candidate, seed=seed):
                    example = tasks.make_example(candidate, seed)
                    self.assertEqual(example, tasks.make_example(candidate, seed))
                    self.assertEqual(example, json.loads(json.dumps(example)))
                    self.assertEqual(set(example), {"candidate", "prompt", "completion", "semantic_key", "checker_payload"})
                    self.assertIs(tasks.verify(example, example["completion"]), True)
                    self.assertIs(tasks.verify(example, "unrelated or invalid output"), False)
                    self.assertIs(tasks.verify(example, None), False)
                    self.assertIs(tasks.verify(example, "x" * 8193), False)

    def test_semantic_keys_do_not_collide_within_candidate(self):
        # Check separated windows too, not only an adjacent prefix. The mixed-
        # radix enumeration and coprime permutation give the full-space bound.
        for candidate in tasks.CANDIDATES:
            limit = tasks.INSTANCE_LIMITS[candidate]
            seeds = list(range(256)) + list(range(30720, 30976)) + list(range(limit - 256, limit))
            keys = [tasks.make_example(candidate, seed)["semantic_key"] for seed in seeds]
            with self.subTest(candidate=candidate):
                self.assertEqual(len(keys), len(set(keys)))
                self.assertTrue(all("seed" not in key for key in keys))

    def test_underlying_record_cannot_cross_splits_via_format_change(self):
        for seed in range(16):
            json_example = tasks.make_example("json_extraction", seed)
            xml_example = tasks.make_example("xml_extraction", seed)
            self.assertEqual(tasks.semantic_key(json_example), tasks.semantic_key(xml_example))
            # A split collision check must reject a repeat, including backups.
            keys = [tasks.semantic_key(json_example), tasks.semantic_key(xml_example)]
            self.assertNotEqual(len(keys), len(set(keys)))

    def test_arithmetic_requires_each_correct_derivation_and_tree(self):
        example = tasks.make_example("arithmetic_derivations", 73)
        lines = example["completion"].splitlines()
        self.assertFalse(tasks.verify(example, lines[-1]))
        wrong = lines.copy()
        wrong[0] = wrong[0].split("=")[0] + "=9999"
        self.assertFalse(tasks.verify(example, "\n".join(wrong)))
        wrong = lines.copy()
        wrong[0], wrong[1] = wrong[1], wrong[0]
        self.assertFalse(tasks.verify(example, "\n".join(wrong)))
        wrong = lines.copy()
        wrong[-1] = "<answer>__import__('os').system('anything')</answer>"
        self.assertFalse(tasks.verify(example, "\n".join(wrong)))
        self.assertFalse(tasks.verify(example, example["completion"] + "\nextra"))
        self.assertFalse(tasks.verify(example, example["completion"].replace("<answer>", "<answer><answer>")))

    def test_reused_arithmetic_rejects_unsafe_or_unbounded_ast(self):
        for expression in ("2**3", "2//3", "-2", "True", "x", "[2]", "1/0", "(" * 65 + "2" + ")" * 65):
            with self.subTest(expression=expression):
                with self.assertRaises((ValueError, SyntaxError, ZeroDivisionError)):
                    tasks._evaluate(tasks._parse_answer(expression))
        self.assertEqual(tasks._evaluate(tasks._parse_answer("(2/3)+(4/5)"))[0], Fraction(22, 15))

    def test_equation_requires_both_correct_steps_and_reduced_fraction(self):
        example = tasks.make_example("equation_derivations", 19)
        first, last = example["completion"].splitlines()
        self.assertFalse(tasks.verify(example, last))
        self.assertFalse(tasks.verify(example, first + "\n\n" + last))
        self.assertFalse(tasks.verify(example, "2*x = 99\n" + last))
        result = Fraction(last.split("=")[1].strip())
        equivalent_unreduced = f"x = {result.numerator * 2}/{result.denominator * 2}"
        self.assertFalse(tasks.verify(example, first + "\n" + equivalent_unreduced))

    def test_sql_gold_and_equivalent_query_work_on_all_witnesses(self):
        for seed in range(20):
            example = tasks.make_example("nl_sql", seed)
            query = example["completion"]
            self.assertTrue(tasks.verify(example, query.lower()))
            self.assertGreater(len({json.dumps(value) for value in example["checker_payload"]["expected"]}), 1)
            for rows, expected in zip(example["checker_payload"]["witnesses"], example["checker_payload"]["expected"]):
                self.assertEqual(tasks._run_sql(query, rows), expected)

    def test_sql_literal_answer_and_missing_predicates_fail(self):
        example = tasks.make_example("nl_sql", 0)
        self.assertFalse(tasks.verify(example, "SELECT 'apple', 123"))
        first_result = example["checker_payload"]["expected"][0]
        constants = " UNION ALL ".join(f"SELECT '{name}', {value}" for name, value in first_result)
        self.assertFalse(tasks.verify(example, constants))
        query = example["completion"]
        where_start, group_start = query.index(" WHERE "), query.index(" GROUP BY ")
        self.assertFalse(tasks.verify(example, query[:where_start] + query[group_start:]))

    def test_sql_witnesses_expose_each_predicate_boundary_for_all_aggregates(self):
        for seed in range(48):
            example = tasks.make_example("nl_sql", seed)
            query = example["completion"]
            wrong_queries = (
                re.sub(r"units (>=|>)", lambda match: "units " + (">" if match[1] == ">=" else ">="), query),
                re.sub(r"price_cents (<=|<)", lambda match: "price_cents " + ("<" if match[1] == "<=" else "<="), query),
                re.sub(r"region = '[a-z]+' AND ", "", query),
            )
            for wrong in wrong_queries:
                with self.subTest(seed=seed, wrong=wrong):
                    self.assertFalse(tasks.verify(example, wrong))

    def test_sql_authorizer_denies_mutations_metadata_and_unsafe_functions(self):
        example = tasks.make_example("nl_sql", 2)
        rows = example["checker_payload"]["witnesses"][0]
        attacks = (
            "DROP TABLE sales", "DELETE FROM sales RETURNING product, units", "PRAGMA table_info(sales)",
            "SELECT name, sql FROM sqlite_master", "SELECT load_extension('anything'), 1",
            "SELECT random(), 1", "SELECT sqlite_version(), 1", "SELECT readfile('/etc/passwd'), 1",
            "SELECT 1, 2; DROP TABLE sales", "ATTACH DATABASE ':memory:' AS other",
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT n,n FROM r",
        )
        for attack in attacks:
            with self.subTest(attack=attack):
                self.assertFalse(tasks.verify(example, attack))
                with self.assertRaises((ValueError, sqlite3.Error, sqlite3.Warning)):
                    tasks._run_sql(attack, rows)

    def test_sql_bounds_expensive_selects(self):
        rows = tasks.make_example("nl_sql", 0)["checker_payload"]["witnesses"][2]
        query = "SELECT COUNT(*), COUNT(*) FROM " + ", ".join(f"sales s{i}" for i in range(12))
        with self.assertRaises(sqlite3.OperationalError):
            tasks._run_sql(query, rows)

    def test_spreadsheet_exact_arithmetic_and_generalization(self):
        sheet = {cell: index + 1 for index, cell in enumerate(tasks._CELLS)}
        self.assertEqual(tasks._formula_value("=SUM(A1:A3)/2", sheet), Fraction(3))
        self.assertEqual(tasks._formula_value("=MIN(A1:A3)+MAX(A1:A3)", sheet), Fraction(4))
        self.assertEqual(tasks._formula_value("=1/3+1/6", sheet), Fraction(1, 2))
        example = tasks.make_example("spreadsheet_formulas", 0)
        self.assertEqual(len(set(example["checker_payload"]["expected"])), 3)
        self.assertFalse(tasks.verify(example, "=" + example["checker_payload"]["expected"][0]))

    def test_spreadsheet_rejects_python_and_unsupported_grammar(self):
        example = tasks.make_example("spreadsheet_formulas", 1)
        attacks = (
            "=__import__('os')", "=A1.__class__", "=SUM(A1:A2, bad=3)", "=2**99999999",
            "=1//2", "=A1[0]", "=SUM([1,2])", "=A9", "=E1", "=1/0", "=1.5",
            "=SUM(RANGE(A1,A2))", "=RANGE(A1,A2)", "=SUM(A3:A1)", "=SUM()",
            "=IF(A1,2,3)", "=SUM(A1:A2);1", "=" + "(" * 250 + "1" + ")" * 250,
        )
        for attack in attacks:
            with self.subTest(attack=attack):
                self.assertFalse(tasks.verify(example, attack))

    def test_json_exact_types_duplicates_extra_fields_and_trailing_text(self):
        example = tasks.make_example("json_extraction", 0)
        record = example["checker_payload"]["record"]
        self.assertTrue(tasks.verify(example, json.dumps(record, indent=2)))
        for altered in ({**record, "extra": 1}, {**record, "units": str(record["units"])},
                        {**record, "active": int(record["active"])}, {**record, "units": True}):
            self.assertFalse(tasks.verify(example, json.dumps(altered)))
        duplicate = example["completion"][:-1] + ',"units":' + str(record["units"]) + "}"
        self.assertFalse(tasks.verify(example, duplicate))
        self.assertFalse(tasks.verify(example, example["completion"] + " trailing text"))
        self.assertFalse(tasks.verify(example, "[" + example["completion"] + "]"))

    def test_xml_rejects_entities_attributes_nested_or_extra_fields(self):
        example = tasks.make_example("xml_extraction", 0)
        gold = example["completion"]
        attacks = (
            gold.replace("<record>", '<record version="1">'),
            gold.replace("</record>", "<extra>1</extra></record>"),
            gold.replace("<name>", "<name><nested/>", 1),
            gold.replace("<city>", '<city x="1">'),
            gold.replace("</record>", "<!-- comment --></record>"),
            gold.replace("<name>", "<name>&amp;"),
            '<!DOCTYPE record [<!ENTITY x "entity">]>' + gold,
            '<?xml version="1.0"?>' + gold,
            gold + "trailing text",
            gold.replace("<active>", "<active>true</active><active>", 1),
        )
        for attack in attacks:
            with self.subTest(attack=attack):
                self.assertFalse(tasks.verify(example, attack))

    def test_number_conversion_has_distinct_bases_and_exact_output(self):
        for seed in range(30):
            example = tasks.make_example("base_conversion", seed)
            _, number, source, target = json.loads(example["semantic_key"])
            self.assertNotEqual(source, target)
            self.assertEqual(int(example["completion"], target), number)
            self.assertFalse(tasks.verify(example, "0" + example["completion"]))
            self.assertFalse(tasks.verify(example, example["completion"] + " explanation"))

    def test_modular_numbers_recompute_exactly(self):
        for seed in range(30):
            example = tasks.make_example("modular_numerals", seed)
            _, n, a, b, modulus, base = json.loads(example["semantic_key"])
            self.assertEqual(int(example["completion"], base), (a * n + b) % modulus)

    def test_transductions_are_grounded_in_actual_input_and_program(self):
        for candidate in ("string_programs", "finite_state_rewrite"):
            example = tasks.make_example(candidate, 10)
            key = json.loads(example["semantic_key"])
            self.assertIn(key[1], example["prompt"])
            wrong = ("a" if example["completion"][0] != "a" else "b") + example["completion"][1:]
            self.assertFalse(tasks.verify(example, wrong))
            self.assertFalse(tasks.verify(example, json.dumps(example["completion"])))


if __name__ == "__main__":
    unittest.main()
