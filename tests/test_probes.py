from datetime import date, timedelta
from fractions import Fraction
import itertools
import json
import unittest

import probes
import tasks


class FixedProbeTests(unittest.TestCase):
    def test_candidate_order_and_instance_limits(self):
        self.assertEqual(probes.PROBE_CANDIDATES, ("graph_path", "calendar_arithmetic", "unit_conversion"))
        self.assertTrue(set(probes.PROBE_CANDIDATES).isdisjoint(tasks.CANDIDATES))
        for candidate in probes.PROBE_CANDIDATES:
            self.assertGreater(probes.PROBE_INSTANCE_LIMITS[candidate], 2 * 32 * 64 + 512)
            for index in (-1, True, probes.PROBE_INSTANCE_LIMITS[candidate]):
                with self.assertRaises(ValueError):
                    probes.make_probe(candidate, index)

    def test_probe_gold_reproducible_serializable_and_valid(self):
        for candidate in probes.PROBE_CANDIDATES:
            for index in (*range(32), probes.PROBE_INSTANCE_LIMITS[candidate] - 1):
                with self.subTest(candidate=candidate, index=index):
                    example = probes.make_probe(candidate, index)
                    self.assertEqual(example, probes.make_probe(candidate, index))
                    self.assertEqual(example, json.loads(json.dumps(example)))
                    self.assertIs(probes.verify_probe(example, example["completion"]), True)
                    self.assertIs(probes.verify_probe(example, "invalid answer"), False)

    def test_disjoint_instance_indices_produce_disjoint_semantic_keys(self):
        all_keys = set()
        for candidate in probes.PROBE_CANDIDATES:
            train = {probes.make_probe(candidate, index)["semantic_key"] for index in range(1024)}
            validation = {probes.make_probe(candidate, index)["semantic_key"] for index in range(1024, 2048)}
            self.assertEqual(len(train), 1024)
            self.assertEqual(len(validation), 1024)
            self.assertTrue(train.isdisjoint(validation))
            self.assertTrue(all_keys.isdisjoint(train | validation))
            all_keys.update(train | validation)
        acquisition_keys = {tasks.make_example(candidate, 0)["semantic_key"] for candidate in tasks.CANDIDATES}
        self.assertTrue(all_keys.isdisjoint(acquisition_keys))

    def test_graph_path_cost_minimum_all_ties_and_duplicate_rejection(self):
        # All edge weights 1: eight valid tied shortest paths; gold is fixed.
        example = probes._graph_path(0)
        self.assertEqual(example["completion"], "S A C E T")
        for path in probes._PATHS:
            self.assertTrue(probes.verify_probe(example, " ".join(path)))
        for path in ("S A A C E T", "S A E T", "S C A E T", "S A C E", "S  A C E T", "['S','A','C','E','T']"):
            self.assertFalse(probes.verify_probe(example, path))
        example = probes.make_probe("graph_path", 51)
        for cost, path in probes._path_candidates(example["checker_payload"]["edges"]):
            self.assertEqual(probes.verify_probe(example, " ".join(path)), cost == example["checker_payload"]["minimum_cost"])

    def test_calendar_handles_leap_day_and_century_rule(self):
        # Direct unpermuted instances let the boundaries be tested explicitly.
        for start, offset, expected in (("2000-02-28", 1, "2000-02-29"),
                                        ("2000-03-01", -1, "2000-02-29"),
                                        ("2099-12-31", 60, "2100-03-01")):
            start_index = (date.fromisoformat(start) - probes._EPOCH).days
            offset_index = offset + 366 if offset < 0 else offset + 365
            example = probes._calendar(offset_index * 36525 + start_index)
            self.assertEqual(example["completion"], expected)
            self.assertTrue(probes.verify_probe(example, expected))
            self.assertFalse(probes.verify_probe(example, expected.replace("-", "/")))

    def test_unit_conversion_exact_fraction_all_registered_pairs(self):
        for pair_index, (_, _, source, target) in enumerate(probes._UNIT_PAIRS):
            example = probes._unit(pair_index * 100000 + 16)
            self.assertEqual(Fraction(example["completion"]), Fraction(17 * source[1], target[1]))
            self.assertFalse(probes.verify_probe(example, example["completion"] + " " + target[0]))


class DiversityTests(unittest.TestCase):
    def test_diversity_candidates_and_instance_bounds(self):
        self.assertEqual(probes.DIVERSITY_CANDIDATES, ("graph_coloring", "set_partition"))
        for candidate in probes.DIVERSITY_CANDIDATES:
            for index in (-1, True, probes.DIVERSITY_INSTANCE_LIMITS[candidate]):
                with self.assertRaises(ValueError):
                    probes.make_diversity(candidate, index)

    def test_every_graph_instance_has_at_least_four_enumerated_families(self):
        keys = set()
        for index in range(probes.DIVERSITY_INSTANCE_LIMITS["graph_coloring"]):
            example = probes.make_diversity("graph_coloring", index)
            families = probes.enumerate_diversity_families(example)
            self.assertEqual(example["family_count"], len(set(families)))
            self.assertGreaterEqual(len(families), 4)
            self.assertNotIn(example["semantic_key"], keys)
            keys.add(example["semantic_key"])
            for family in families:
                verified = probes.verify_diversity(example, family)
                self.assertEqual(verified["strategy_family"], family)

    def test_graph_family_enumeration_matches_full_brute_force(self):
        for index in (0, 1, 17, 255):
            example = probes.make_diversity("graph_coloring", index)
            brute_force = set()
            for colors in itertools.product(range(3), repeat=8):
                verified = probes.verify_diversity(example, json.dumps(colors))
                if verified:
                    brute_force.add(verified["strategy_family"])
            self.assertEqual(set(probes.enumerate_diversity_families(example)), brute_force)

    def test_color_label_swaps_are_one_family_but_distinct_outputs(self):
        example = probes.make_diversity("graph_coloring", 9)
        gold = json.loads(example["completion"])
        swapped = [(value + 1) % 3 for value in gold]
        original = probes.verify_diversity(example, json.dumps(gold))
        renamed = probes.verify_diversity(example, json.dumps(swapped))
        self.assertEqual(original["strategy_family"], renamed["strategy_family"])
        self.assertNotEqual(original["canonical_solution"], renamed["canonical_solution"])

    def test_coloring_rejects_invalid_shapes_types_and_edges(self):
        example = probes.make_diversity("graph_coloring", 2)
        for text in ("[]", "[0,0,0,0,0,0,0,0]", "[0,0,0,0,1,1,1,3]",
                     "[0,0,0,0,1,1,1,true]", '{"0":0}', example["completion"] + " extra"):
            self.assertIsNone(probes.verify_diversity(example, text))

    def test_partition_families_are_exhaustive_canonical_and_valid(self):
        for index in (0, 1, 17, 127, probes.DIVERSITY_INSTANCE_LIMITS["set_partition"] - 1):
            example = probes.make_diversity("set_partition", index)
            families = probes.enumerate_diversity_families(example)
            self.assertGreaterEqual(len(families), 4)
            self.assertEqual(len(families), len(set(families)))
            self.assertEqual(example["family_count"], len(families))
            for family in families:
                verified = probes.verify_diversity(example, family)
                self.assertEqual(verified["strategy_family"], family)
                groups = json.loads(family)
                permuted = [list(reversed(group)) for group in reversed(groups)]
                self.assertEqual(probes.verify_diversity(example, json.dumps(permuted)), verified)

    def test_partition_rejects_duplicates_missing_items_and_wrong_sums(self):
        example = probes.make_diversity("set_partition", 0)
        groups = json.loads(example["completion"])
        groups[0][0] = groups[1][0]
        self.assertIsNone(probes.verify_diversity(example, json.dumps(groups)))
        groups = json.loads(example["completion"])
        groups[0][0], groups[1][0] = groups[1][0], groups[0][0]
        self.assertIsNone(probes.verify_diversity(example, json.dumps(groups)))
        self.assertIsNone(probes.verify_diversity(example, json.dumps([example["checker_payload"]["values"]])))
        self.assertIsNone(probes.verify_diversity(example, "[]"))

    def test_diversity_gold_and_keys_are_serializable_reproducible_disjoint(self):
        all_keys = set()
        for candidate in probes.DIVERSITY_CANDIDATES:
            for index in range(32):
                example = probes.make_diversity(candidate, index)
                self.assertEqual(example, probes.make_diversity(candidate, index))
                self.assertEqual(example, json.loads(json.dumps(example)))
                self.assertIsNotNone(probes.verify_diversity(example, example["completion"]))
                self.assertLess(len(example["completion"]), 128)
                self.assertNotIn(example["semantic_key"], all_keys)
                all_keys.add(example["semantic_key"])
        for candidate in probes.PROBE_CANDIDATES:
            self.assertNotIn(probes.make_probe(candidate, 0)["semantic_key"], all_keys)
        for candidate in tasks.CANDIDATES:
            self.assertNotIn(tasks.make_example(candidate, 0)["semantic_key"], all_keys)


if __name__ == "__main__":
    unittest.main()
