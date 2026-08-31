import copy
import json
import math
import unittest

from backend import ACCOUNTING_KEYS
from costs import MEASUREMENT_UNIT_KEYS, TASK_UNIT_KEYS, project_m2, token_cost
from protocol import ARMS, ORDERS, TASK_SLOTS


class TokenCostTests(unittest.TestCase):
    def setUp(self):
        self.prices = {"train": 4.0, "prefill": 2.0, "cached": 0.5, "sample": 8.0}

    def test_direct_accounting_fields_use_the_registered_price_categories(self):
        rates = {"gradient_target_tokens": 0, "train_tokens": 4, "forward_tokens": 4,
                 "prefill_tokens": 2, "cached_tokens": 0.5, "sample_tokens": 8,
                 "scoring_prefill_tokens": 2, "scoring_discarded_sample_tokens_estimate": 8}
        self.assertEqual(set(rates), set(ACCOUNTING_KEYS))
        for key, expected in rates.items():
            with self.subTest(key=key):
                self.assertEqual(token_cost({key: 1_000_000}, self.prices), expected)

    def test_accounting_is_additive_without_gradient_double_billing(self):
        first = {key: (index + 1) * 100_000 for index, key in enumerate(ACCOUNTING_KEYS)}
        second = {key: (index + 2) * 50_000 for index, key in enumerate(ACCOUNTING_KEYS)}
        combined = {key: first[key] + second[key] for key in ACCOUNTING_KEYS}
        self.assertAlmostEqual(token_cost(combined, self.prices), token_cost(first, self.prices) + token_cost(second, self.prices))
        exposed = {**combined, "gradient_target_tokens": 10**15}
        self.assertEqual(token_cost(exposed, self.prices), token_cost(combined, self.prices))

    def test_zero_costs_and_unrounded_small_amounts_are_valid(self):
        self.assertEqual(token_cost({}, self.prices), 0.0)
        self.assertEqual(token_cost({"sample_tokens": 1000}, dict.fromkeys(self.prices, 0)), 0.0)
        self.assertEqual(token_cost({"train_tokens": 1}, self.prices), 0.000004)

    def test_negative_nonfinite_fractional_and_unknown_accounting_rejected(self):
        for bad in (-1, float("nan"), float("inf"), 1.5, True, "10"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    token_cost({"train_tokens": bad}, self.prices)
        with self.assertRaises(ValueError):
            token_cost({"unknown_tokens": 0}, self.prices)
        for bad in (-1, float("nan"), float("inf"), True, "1", 10**1000):
            with self.subTest(price=bad):
                with self.assertRaises(ValueError):
                    token_cost({}, {**self.prices, "train": bad})
        with self.assertRaises(ValueError):
            token_cost({}, {"train": 1})
        with self.assertRaises(ValueError):
            token_cost({}, {**self.prices, "other": 1})

    def test_numeric_overflow_is_not_reported_as_a_valid_cost(self):
        with self.assertRaises(ValueError):
            token_cost({"train_tokens": 10**1000}, self.prices)
        with self.assertRaises(ValueError):
            token_cost({"train_tokens": 10_000_000}, {**self.prices, "train": 1e308})


class M2ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.task_units = {slot: dict(zip(TASK_UNIT_KEYS, (1.0, 2.0, 3.0, 0.5))) for slot in TASK_SLOTS}
        self.measurements = dict(zip(MEASUREMENT_UNIT_KEYS, (1.0, 2.0, 3.0, 4.0, 5.0)))

    def test_exact_twelve_lineage_counts_and_retention_deduplication(self):
        projection = project_m2(self.task_units, self.measurements)
        counts = projection["counts"]
        self.assertEqual(counts["lineages"], 12)
        self.assertEqual(counts["lineage_cycles"], 84)
        self.assertEqual(counts["repair_opportunities"], 56)
        self.assertEqual(counts["learn_only_physical_checkpoints"], 28)
        self.assertEqual(counts["repair_arm_physical_checkpoints"], 112)
        self.assertEqual(counts["physical_checkpoints"], 140)
        self.assertEqual(counts["task_heldout_evaluations"], 560)
        self.assertEqual(counts["current_A_heldout_in_learning"], 84)
        self.assertEqual(counts["additional_task_heldout"], 476)
        additional = {"T1": 73, "T2": 83, "T3": 68, "T4": 63, "T5": 63, "T6": 73, "T7": 53}
        for task, expected in additional.items():
            self.assertEqual(counts["per_task"][task], {
                "learn_to_competence": 4, "learn_to_damage": 8, "repair": 8,
                "native_heldout_total": expected + 12, "native_heldout_included": 12,
                "native_heldout_additional": expected,
            })

    def test_retention_count_matches_independent_state_listing_and_position_formula(self):
        # Build the whole physical-state table, independently of the projection's
        # running acquired-task counters, then select states following acquisition.
        states = [(name, arm, cycle, state) for name in ORDERS for arm in ARMS
                  for cycle in range(1, 8) for state in (("A",) if arm == "learn-only" else ("A", "B"))]
        self.assertEqual(len(states), 140)
        projected = project_m2(self.task_units, self.measurements)["counts"]["per_task"]
        for task in TASK_SLOTS:
            total = sum(cycle >= ORDERS[order].index(task) + 1 for order, _, cycle, _ in states)
            already_in_learning = sum(state == "A" and cycle == ORDERS[order].index(task) + 1
                                      for order, _, cycle, state in states)
            formula = 5 * sum(8 - (sequence.index(task) + 1) for sequence in ORDERS.values())
            self.assertEqual(total, formula)
            self.assertEqual(already_in_learning, 12)
            self.assertEqual(projected[task]["native_heldout_additional"], total - already_in_learning)

    def test_line_items_subtotals_and_total_add_up_without_rounding(self):
        projection = project_m2(self.task_units, self.measurements)
        self.assertEqual(len(projection["line_items"]), 7 * 4 + 5)
        self.assertEqual(projection["subtotals_usd"], {
            "acquisition": 7 * (4 * 1 + 8 * 2), "repair": 7 * 8 * 3,
            "retention": 476 * 0.5, "checkpoint_measurements": 140 * 15,
        })
        self.assertEqual(projection["total_usd"], 2646)
        self.assertEqual(math.fsum(projection["subtotals_usd"].values()), projection["total_usd"])
        for line in projection["line_items"]:
            self.assertEqual(line["subtotal_usd"], line["count"] * line["unit_usd"])
            self.assertIsInstance(line["count"], int)
            self.assertTrue(line["unit"])
        self.assertEqual(json.loads(json.dumps(projection)), projection)

    def test_zero_units_are_valid_and_fractional_unit_cost_is_not_ceiled(self):
        zero_tasks = {slot: dict.fromkeys(TASK_UNIT_KEYS, 0) for slot in TASK_SLOTS}
        zero_measurements = dict.fromkeys(MEASUREMENT_UNIT_KEYS, 0)
        self.assertEqual(project_m2(zero_tasks, zero_measurements)["total_usd"], 0.0)
        zero_tasks["T1"]["learn_to_competence"] = 0.0000013
        self.assertEqual(project_m2(zero_tasks, zero_measurements)["total_usd"], 4 * 0.0000013)

    def test_optional_prices_are_provenance_not_an_extra_multiplier(self):
        prices = {"train": 100.0, "prefill": 200.0, "cached": 300.0, "sample": 400.0}
        original = copy.deepcopy((self.task_units, self.measurements, prices))
        unpriced = project_m2(self.task_units, self.measurements)
        priced = project_m2(self.task_units, self.measurements, prices)
        self.assertEqual(unpriced["total_usd"], priced["total_usd"])
        self.assertEqual(priced["prices_usd_per_million_tokens"], prices)
        self.assertEqual((self.task_units, self.measurements, prices), original)

    def test_incomplete_or_extra_design_units_and_nonfinite_costs_are_rejected(self):
        missing = copy.deepcopy(self.task_units)
        missing.pop("T7")
        with self.assertRaises(ValueError):
            project_m2(missing, self.measurements)
        with self.assertRaises(ValueError):
            project_m2({**self.task_units, "T8": self.task_units["T1"]}, self.measurements)
        with self.assertRaises(ValueError):
            project_m2(self.task_units, {key: value for key, value in self.measurements.items() if key != "kl"})
        for invalid in (-1, float("nan"), float("inf"), True, "1"):
            with self.subTest(invalid=invalid):
                changed = copy.deepcopy(self.task_units)
                changed["T1"]["repair"] = invalid
                with self.assertRaises(ValueError):
                    project_m2(changed, self.measurements)
                with self.assertRaises(ValueError):
                    project_m2(self.task_units, {**self.measurements, "diversity": invalid})
        with self.assertRaises(ValueError):
            project_m2(self.task_units, {**self.measurements, "kl": 1e308})


if __name__ == "__main__":
    unittest.main()
