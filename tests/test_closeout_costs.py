"""Pure closeout-cost integration tests; journal records and manifests are local fakes."""

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend import ACCOUNTING_KEYS
import closeout
from costs import project_m2, token_cost
from protocol import ORDERS, TASK_SLOTS


class CloseoutCostTests(unittest.TestCase):
    def setUp(self):
        # Deliberately simple fictitious test rates, never published prices.
        self.prices = {"train": 1_000_000, "prefill": 2_000_000, "cached": 3_000_000, "sample": 4_000_000}
        self.snapshot = {"train_and_forward": self.prices["train"], "prefill": self.prices["prefill"],
                         "cached_prefill": self.prices["cached"], "sample": self.prices["sample"],
                         "unit": "USD per million tokens", "source": "offline fixture"}
        self.journal = SimpleNamespace(completed={}, rows=[])
        self.clock = 0
        self.thresholds = {"gate_competence": 0.7, "minimum_movement": 0.15,
                           "heldout_competence": 0.5, "damage_low": 26, "damage_high": 36, "recovery_target": 41}

    def add_operation(self, name, usage=None, *, elapsed=0.5, result=None):
        start = datetime(2026, 8, 31, tzinfo=timezone.utc) + timedelta(seconds=self.clock)
        self.clock += 10  # Deliberate idle intervals distinguish wall from active time.
        finish = start + timedelta(seconds=elapsed)
        result = dict(result or {})
        if usage is not None:
            result["accounting"] = dict(usage)
        begun = {"type": "inflight", "operation": name, "timestamp": start.isoformat()}
        complete = {"type": "complete", "operation": name, "timestamp": finish.isoformat(),
                    "elapsed_seconds": elapsed, "result": result}
        self.assertNotIn(name, self.journal.completed)
        self.journal.rows.extend((begun, complete))
        self.journal.completed[name] = complete
        return result

    def add_event(self, slot="T1", realization=1, *, scale=1, points=None, start_gate=0.0):
        branch = f"screen/{slot}/selected/{realization}"
        points = points or [{"gate": 0.72, "if_score": 42}, {"gate": 0.8, "if_score": 35}]

        def usage(**values):
            return {key: value * scale for key, value in values.items()}

        prefix = branch + "/learn/"
        start = {"gate": start_gate, "if_score": 43, "heldout": 0.0}
        start_result = self.add_operation(prefix + "start", usage(forward_tokens=10), elapsed=1, result=start)
        history, gradient = [], 0
        gradients = (4, 6, 10)
        for step, values in enumerate(points, 1):
            gradient += gradients[step - 1] * scale
            self.add_operation(prefix + f"update/{step:03d}",
                               usage(train_tokens=20 * step, gradient_target_tokens=gradients[step - 1]), elapsed=2 * step)
            evaluation_usage = usage(forward_tokens=2 * step + 3, cached_tokens=3) if step == 1 else usage(
                forward_tokens=2 * step + 3, prefill_tokens=2, sample_tokens=1)
            measured = self.add_operation(prefix + f"evaluate/{step:03d}", evaluation_usage,
                                         elapsed=2 * step + 1, result=values)
            history.append({**measured, "step": step, "target_tokens": gradient})
        history[-1]["heldout"] = 0.9
        self.add_operation(prefix + f"heldout/{len(points):03d}", usage(forward_tokens=11), elapsed=8,
                           result={"heldout": 0.9})
        learning = {"start": start_result, "points": history,
                    "decision": {"checkpoint": history[-1]}, "target_tokens": gradient}
        # Nested repeated accounting is metadata, not another billed operation.
        self.add_operation(prefix + "complete", result=learning, elapsed=0.2)
        repair_usage = usage(train_tokens=10, forward_tokens=2, gradient_target_tokens=3,
                             prefill_tokens=5, cached_tokens=2, sample_tokens=3,
                             scoring_prefill_tokens=4, scoring_discarded_sample_tokens_estimate=1)
        for step in range(1, 6):
            self.add_operation(branch + f"/repair/update/{step:03d}", repair_usage)
        self.add_operation(branch + "/repair/criterion/005", usage(sample_tokens=2, prefill_tokens=3, cached_tokens=1))
        repair = {"decision": {"repair_steps": 5}, "target_tokens": 15 * scale}
        self.add_operation(branch + "/repair/complete", result=repair)
        event = {"branch": branch, "slot": slot, "arm": "fixed", "cycle": 1,
                 "learning": learning, "repair": repair,
                 "A": {"state_path": branch + "/A-state", "sampler_path": branch + "/A-sampler"},
                 "B": {"state_path": branch + "/B-state", "sampler_path": branch + "/B-sampler"}}
        self.add_operation(branch + "/event", result=event)
        return event

    def stage_for(self, events):
        selections = {}
        for slot in dict.fromkeys(event["slot"] for event in events):
            selections[slot] = {"candidate": TASK_SLOTS[slot][0], "thresholds": dict(self.thresholds),
                                "selected": {"learning_rate": 3e-5, "batch_size": 16,
                                             "realizations": [event for event in events if event["slot"] == slot]}}
        return {"screening": {"selected": selections}, "persistence": {"events": []},
                "if_thresholds": {key: self.thresholds[key] for key in ("damage_low", "damage_high", "recovery_target")}}

    def projection_fixture(self):
        events = [self.add_event(slot, repeat, scale=repeat) for slot in TASK_SLOTS for repeat in (1, 2)]
        stage = self.stage_for(events)
        selected_probes = {}
        for kind, counts in (("structured", (8, 16)), ("language", (15, 45))):
            branches = [f"probe/{kind}/selected/state/{index}" for index in range(2)]
            for branch, count in zip(branches, counts):
                self.add_operation(branch + "/update/001", {"train_tokens": count})
                self.add_operation(branch + "/complete", result={"curve": [{"accounting": {"train_tokens": count}}]})
            selected_probes[kind] = {
                "candidate": "graph_path" if kind == "structured" else "wikipedia_vi",
                "learning_rate": 1e-5, "standard_budget": 32 if kind == "structured" else 245,
                "reference_target": 1.0, "delta_l": 0.25, "minimum_headroom": 1.0,
                "noise_bounds": {}, "clock_noise": {}, "states": [], "passes": True,
                "standard_branches": branches,
            }
        stage["probes"] = {"selected": selected_probes}
        realized = [self.add_operation(f"diversity/selected/realization/{index}", {"sample_tokens": count})
                    for index, count in enumerate((4, 8, 12))]
        stage["diversity"] = {"selected": {"candidate": "graph_coloring", "realizations": realized,
                                             "qualification": {"passes": True}, "noise": {}}}
        physical = {}
        for key, sample, scoring in (("A", 2, 5), ("B", 4, 7)):
            physical[key] = {
                "if_heldout": self.add_operation(f"closeout/if-heldout/{key}", {"sample_tokens": sample}),
                "kl": self.add_operation(f"closeout/kl/{key}",
                                         {"scoring_prefill_tokens": scoring, "scoring_discarded_sample_tokens_estimate": 1}),
            }
        # Spent M1 work counts toward M1 cost, never the selected M2 unit recipe.
        self.add_operation("screen/T1/unselected/learn/update/001", {"train_tokens": 9999})
        self.add_operation("probe/structured/reference/update/001", {"train_tokens": 8888})
        return stage, {"physical_checkpoints": physical}

    def test_selected_control_prefix_is_distinct_from_full_damage_and_heldout_charged_once(self):
        first = self.add_event(realization=1)
        second = self.add_event(realization=2, scale=2)
        self.add_operation("screen/T1/unselected/learn/update/001", {"train_tokens": 9999})
        stage = self.stage_for([first, second])
        result = closeout.measured_task_units(self.journal, stage, self.prices)["T1"]
        # One realization: competence=10+20+14+11; damage adds40+15.
        # Repair=5*(12 train +18 prefill +6 cached +16 sample)+17 criterion.
        self.assertEqual(result, {"learn_to_competence": 82.5, "learn_to_damage": 165,
                                  "repair": 415.5, "native_heldout": 16.5})
        self.assertEqual(closeout.cost_of_prefix(self.journal, first["branch"] + "/learn/", self.prices), 110)

    def test_control_prefix_requires_minimum_movement_not_gate_only(self):
        event = self.add_event(start_gate=0.65,
                               points=[{"gate": 0.72, "if_score": 42}, {"gate": 0.85, "if_score": 35}])
        units = closeout.measured_task_units(self.journal, self.stage_for([event]), self.prices)["T1"]
        self.assertEqual(units["learn_to_competence"], 110)
        self.assertEqual(units["learn_to_damage"], 110)

    def test_nested_summaries_and_prefix_neighbors_do_not_duplicate_billing(self):
        self.add_operation("exact/update/001", {"train_tokens": 5})
        self.add_operation("exact/complete", result={"copied": {"accounting": {"train_tokens": 5}}})
        self.add_operation("exactly/update/001", {"train_tokens": 100})
        self.assertEqual(closeout.cost_of_prefix(self.journal, "exact/", self.prices), 5)
        self.assertEqual(closeout.cost_of_prefix(self.journal, "", self.prices), 105)

    def test_measured_projection_uses_selected_standard_costs_and_preserves_full_design_shape(self):
        stage, measured = self.projection_fixture()
        with patch("closeout._read", return_value=self.snapshot) as read:
            projection = closeout.measured_projection("/unused", self.journal, stage, measured)
        read.assert_called_once_with(Path("/unused/manifests/prices.json"))
        expected_tasks = {slot: {"learn_to_competence": 82.5, "learn_to_damage": 165,
                                 "repair": 415.5, "native_heldout": 16.5} for slot in TASK_SLOTS}
        expected_measurements = {"structured_probe": 12, "language_probe": 30, "diversity": 32, "if_heldout": 12, "kl": 16}
        self.assertEqual(projection["task_units_usd"], expected_tasks)
        self.assertEqual(projection["measurement_units_usd"], expected_measurements)
        expected = project_m2(expected_tasks, expected_measurements, self.prices)
        self.assertEqual(projection["total_usd"], expected["total_usd"])
        self.assertEqual(projection["line_items"], expected["line_items"])
        self.assertEqual(projection["counts"]["physical_checkpoints"], 140)
        self.assertEqual(projection["counts"]["additional_task_heldout"], 476)
        self.assertEqual(projection["price_source"], self.snapshot)
        # Includes unselected screening and extended-reference expenditure once.
        self.assertEqual(projection["m1_estimated_usd"], 7 * (110 + 220 + 277 + 554) + 84 + 96 + 56 + 9999 + 8888)

    def test_unknown_recovery_duration_is_not_reported_as_exact_active_time(self):
        event = self.add_event()
        name = event["branch"] + "/learn/evaluate/001"
        self.journal.completed[name]["timing_complete"] = False
        row = closeout.lifecycle_exposures(self.journal, self.stage_for([event]), self.prices)[0]
        self.assertIsNone(row["competence"]["active_wall_seconds"])
        self.assertFalse(row["competence"]["timing_complete"])
        self.assertGreater(row["competence"]["wall_seconds"], 0)

    def test_projection_discloses_unknown_legacy_sampling_usage(self):
        stage, measured = self.projection_fixture()
        self.journal.rows.append({"type": "recovery_authorized", "operation": "reference/task/evaluate/015",
                                  "prior_accounting": "unavailable"})
        with patch("closeout._read", return_value=self.snapshot):
            projection = closeout.measured_projection("/unused", self.journal, stage, measured)
        self.assertEqual(projection["unreconciled_sampling_operations"], ["reference/task/evaluate/015"])

    def test_launch_packet_preserves_projection_schema_without_authorizing_M2(self):
        stage, measured = self.projection_fixture()
        with patch("closeout._read", return_value=self.snapshot):
            projection = closeout.measured_projection("/unused", self.journal, stage, measured)
        stage["persistence"]["events"] = [{} for _ in range(5)]

        def manifest(path):
            if path.parent.name == "tasks":
                return {"metric": "verifier_success", "splits": {"main": {"sha256": "frozen-main-" + path.stem}}}
            if path.parent.name == "orders":
                return {"order": path.stem, "slots": list(ORDERS[path.stem])}
            self.fail(f"unexpected manifest read: {path}")

        with patch("closeout._read", side_effect=manifest):
            packet = closeout.launch_packet("/unused", {"test_commit": "tested", "freeze_sha256": "frozen"},
                                            stage, measured, projection, [{"event": "measured-exposure"}])
        self.assertEqual(packet["projection"], projection)
        self.assertEqual(packet["projection"]["currency"], "USD")
        self.assertEqual(packet["projection"]["counts"]["lineages"], 12)
        self.assertFalse(packet["main_run_authorized"])
        self.assertIsNone(packet["publication_freeze_commit"])
        self.assertEqual(packet["persistence"]["observed_events"], 5)
        self.assertEqual(len(packet["orders"]), 4)
        self.assertEqual(len(packet["tasks"]), 7)

    def test_exposures_distinguish_competence_band_entry_valid_damage_and_heldout(self):
        event = self.add_event(points=[{"gate": 0.4, "if_score": 35}, {"gate": 0.8, "if_score": 25},
                                       {"gate": 0.9, "if_score": 34}])
        row = closeout.lifecycle_exposures(self.journal, self.stage_for([event]), self.prices)[0]
        expected = {
            "damage_band_entry": (1, 4, 20, 35, 44, 6, 23),
            "competence": (2, 10, 60, 82, 99, 15, 45),
            "valid_damage": (3, 20, 120, 151, 176, 28, 67),
        }
        for name, (updates, gradient, train, priced, usd, active, wall) in expected.items():
            with self.subTest(dose=name):
                dose = row[name]
                self.assertEqual(dose["updates"], updates)
                self.assertEqual(dose["gradient_target_tokens"], gradient)
                self.assertEqual(dose["train_tokens"], train)
                self.assertEqual(dose["train_priced_tokens"], priced)
                self.assertEqual(dose["estimated_usd"], usd)
                self.assertEqual(dose["active_wall_seconds"], active)
                self.assertEqual(dose["wall_seconds"], wall)
                self.assertEqual(dose["started_at"], "2026-08-31T00:00:00+00:00")
                self.assertEqual(dose["finished_at"], self.journal.completed[event["branch"] + f"/learn/evaluate/{updates:03d}"]["timestamp"])
        self.assertEqual(closeout.cost_of_prefix(self.journal, event["branch"] + "/learn/", self.prices), 187)
        self.assertEqual(row["repair_updates"], 5)
        self.assertEqual(row["repair_gradient_target_tokens"], 15)
        self.assertEqual(set(row["repair_accounting"]), set(ACCOUNTING_KEYS))
        self.assertEqual(row["repair_accounting"]["forward_tokens"], 10)
        self.assertEqual(row["repair_accounting"]["cached_tokens"], 11)
        self.assertEqual(row["repair_accounting"]["scoring_discarded_sample_tokens_estimate"], 5)
        self.assertEqual(token_cost(row["repair_accounting"], self.prices), row["repair_estimated_usd"])
        self.assertEqual(row["repair_estimated_usd"], 277)
        self.assertGreater(row["wall_seconds"], row["active_wall_seconds"])
        self.assertEqual(row["finished_at"], self.journal.completed[event["branch"] + "/event"]["timestamp"])

    def test_nonattainment_remains_none_and_baseline_competence_is_observed_zero_updates(self):
        event = self.add_event(points=[{"gate": 0.2, "if_score": 42}, {"gate": 0.3, "if_score": 42}])
        stage = self.stage_for([event])
        row = closeout.lifecycle_exposures(self.journal, stage, self.prices)[0]
        self.assertIsNone(row["competence"])
        self.assertIsNone(row["damage_band_entry"])
        self.assertIsNone(row["valid_damage"])
        event["learning"]["start"]["gate"] = 0.8
        row = closeout.lifecycle_exposures(self.journal, stage, self.prices)[0]
        self.assertEqual(row["competence"]["updates"], 0)
        self.assertEqual(row["competence"]["gradient_target_tokens"], 0)
        self.assertEqual(row["competence"]["estimated_usd"], 10)
        self.assertEqual(row["competence"]["wall_seconds"], 1)
        self.assertIsNone(row["valid_damage"])

    def test_missing_start_timestamp_does_not_fabricate_wall_time_from_active_time(self):
        event = self.add_event()
        stage = self.stage_for([event])
        self.journal.rows = [row for row in self.journal.rows if row["type"] != "inflight"]
        measured = closeout.lifecycle_exposures(self.journal, stage, self.prices)[0]
        self.assertIsNone(measured["started_at"])
        self.assertIsNone(measured["wall_seconds"])
        self.assertIsNotNone(measured["finished_at"])
        self.assertGreater(measured["active_wall_seconds"], 0)
        self.assertIsNone(measured["competence"]["started_at"])
        self.assertIsNone(measured["competence"]["wall_seconds"])

    def test_closeout_cost_queries_do_not_mutate_journal_or_stage(self):
        stage, measured = self.projection_fixture()
        before = copy.deepcopy((stage, measured, self.journal.completed, self.journal.rows))
        with patch("closeout._read", return_value=self.snapshot):
            closeout.measured_projection("/unused", self.journal, stage, measured)
        closeout.lifecycle_exposures(self.journal, stage, self.prices)
        self.assertEqual((stage, measured, self.journal.completed, self.journal.rows), before)


if __name__ == "__main__":
    unittest.main()
