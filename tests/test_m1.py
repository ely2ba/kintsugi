import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibrate import Journal
import m1
from protocol import LEARNING_RATES


class MemoryJournal:
    def __init__(self):
        self.calls = []

    def call(self, operation, inputs, function):
        self.calls.append(operation)
        return function()


def checkpoint(name):
    return {"state_path": name + "-state", "sampler_path": name + "-sampler"}


def successful_event(branch, **extra):
    return {"branch": branch, "primary_eligible": True, "repair_success": True,
            "target_tokens": 100, "A": checkpoint(branch + "-A"), "B": checkpoint(branch + "-B"), **extra}


def measurement_manifest():
    return {"noise_repeats": 3,
            "paired_noise_bound": {"method": "v1_operational_2_5_sd", "multiplier": 2.5},
            "kl": {"prompt_count": 32, "samples": 1, "max_tokens": 512, "temperature": 1.0, "seed": 20260831}}


class ScreeningTests(unittest.TestCase):
    def test_deterministic_task_noise_does_not_make_evaluation_calls(self):
        with patch("m1._task_metric") as metric:
            noise = m1.task_noise(None, checkpoint("origin"), {"candidate": "task", "data_sha256": "data"},
                                  MemoryJournal(), 3)
        metric.assert_not_called()
        self.assertEqual((noise["sample_sd"], noise["bound"], noise["paid_repetitions"]), (0.0, 0.0, 0))

    def test_reference_frozen_then_all_lrs_and_both_realizations_start_origin(self):
        origin, calls, journal = checkpoint("origin"), [], MemoryJournal()
        task = {"candidate": "candidate", "slot": "T1", "batch_size": 16, "data_sha256": "data"}
        reference = {"references": {"status": "defined", "gate0": 0.0, "gate_ref": 1.0,
                                     "heldout0": 0.0, "heldout_ref": 1.0}}

        def sweep(*args):
            calls.append("reference-freeze")
            return reference

        def event(api, common_origin, start, previous_b, given_task, split, rate, arm,
                  cycle, thresholds, prompts, given_journal, branch):
            self.assertEqual(common_origin, origin)
            self.assertEqual(start, origin)
            self.assertIsNone(previous_b)
            self.assertEqual((arm, cycle), ("fixed", 1))
            self.assertEqual(thresholds["gate_competence"], 0.7)
            calls.append((rate, split))
            dose = {1e-5: 500, 3e-5: 100, 1e-4: 100}[rate]
            return successful_event(branch, target_tokens=dose)

        with patch("m1.calibrate.reference_sweep", side_effect=sweep), \
             patch("m1.task_noise", return_value={"valid": True, "sample_sd": 0.01}), \
             patch("m1.run_event", side_effect=event):
            result = m1.screen_candidate(None, origin, task, {"recovery_target": 41}, [], journal, 3)
        self.assertEqual(calls, ["reference-freeze"] + [(rate, split) for rate in LEARNING_RATES for split in ("screen1", "screen2")])
        self.assertEqual(result["selected"]["learning_rate"], 3e-5)
        self.assertEqual(result["selected"]["median_target_tokens"], 100)
        self.assertEqual(journal.calls[0], "screen/candidate/thresholds")

    def test_invalid_reference_has_no_screening_calls(self):
        with patch("m1.calibrate.reference_sweep", return_value={"references": {"status": "no_complete_valid_trajectory"}}), \
             patch("m1.run_event") as event, patch("m1.task_noise") as noise:
            result = m1.screen_candidate(None, {}, {"candidate": "bad"}, {}, [], MemoryJournal(), 3)
        self.assertIsNone(result["selected"])
        event.assert_not_called()
        noise.assert_not_called()

    def test_backup_only_after_primary_failure_and_never_after_primary_success(self):
        calls = []
        slots = {"T1": ("primary1", "backup1"), "T2": ("primary2", "backup2")}

        def candidate(api, origin, task, *args):
            calls.append(task["candidate"])
            return {"candidate": task["candidate"], "selected": None if task["candidate"] == "primary1" else {"learning_rate": 1e-5}}

        with patch("m1.TASK_SLOTS", slots), patch("m1.load_task", side_effect=lambda root, name: {"candidate": name}), \
             patch("m1.screen_candidate", side_effect=candidate):
            result = m1.screen_tasks(None, {}, Path("."), {}, [], MemoryJournal(), 3)
        self.assertEqual(calls, ["primary1", "backup1", "primary2"])
        self.assertEqual(result["status"], "task_screening_complete")

    def test_both_fail_hard_stops_before_next_slot(self):
        calls = []

        def candidate(api, origin, task, *args):
            calls.append(task["candidate"])
            return {"candidate": task["candidate"], "selected": None}

        with patch("m1.TASK_SLOTS", {"T1": ("primary", "backup"), "T2": ("later", "later-backup")}), \
             patch("m1.load_task", side_effect=lambda root, name: {"candidate": name}), \
             patch("m1.screen_candidate", side_effect=candidate):
            result = m1.screen_tasks(None, {}, Path("."), {}, [], MemoryJournal(), 3)
        self.assertEqual(calls, ["primary", "backup"])
        self.assertEqual(result["status"], "m1_failed")
        self.assertFalse(result["m1_complete"])

    def test_zero_gain_and_zero_noise_needs_contract_attention_not_backup(self):
        reference = {"references": {"status": "defined", "gate0": 0.0, "gate_ref": 0.0,
                                     "heldout0": 0.0, "heldout_ref": 0.0}}
        with patch("m1.calibrate.reference_sweep", return_value=reference), \
             patch("m1.task_noise", return_value={"valid": True, "sample_sd": 0.0}), \
             patch("m1.run_event") as event:
            result = m1.screen_candidate(None, {}, {"candidate": "flat"}, {}, [], MemoryJournal(), 3)
        self.assertEqual(result["status"], "contract_attention_needed")
        event.assert_not_called()
        with patch("m1.TASK_SLOTS", {"T1": ("primary", "backup")}), \
             patch("m1.load_task", return_value={}), patch("m1.screen_candidate", return_value=result) as candidate:
            stopped = m1.screen_tasks(None, {}, Path("."), {}, [], MemoryJournal(), 3)
        self.assertEqual(candidate.call_count, 1)
        self.assertEqual(stopped["status"], "contract_attention_needed")


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.origin = checkpoint("origin")
        self.selections = {slot: {"candidate": slot, "selected": {"learning_rate": 3e-5}, "thresholds": {}}
                           for slot in ("T3", "T2", "T6")}

    def test_common_then_fixed_and_rolling_correct_physical_starts(self):
        calls = []

        def event(api, origin, start, previous_b, task, split, lr, arm, cycle, *args):
            branch = args[-1]
            calls.append({"task": task["candidate"], "cycle": cycle, "arm": arm,
                          "start": start, "previous_b": previous_b, "branch": branch})
            self.assertEqual((split, lr), ("persistence", 3e-5))
            return successful_event(branch)

        with patch("m1.load_task", side_effect=lambda root, name: {"candidate": name}), patch("m1.run_event", side_effect=event):
            result = m1.persistence(None, self.origin, Path("."), self.selections, [], MemoryJournal())
        self.assertEqual([(row["task"], row["cycle"], row["arm"]) for row in calls],
                         [("T3", 1, "fixed"), ("T2", 2, "fixed"), ("T6", 3, "fixed"),
                          ("T2", 2, "rolling"), ("T6", 3, "rolling")])
        common_b = checkpoint(calls[0]["branch"] + "-B")
        self.assertEqual(calls[0]["start"], self.origin)
        self.assertIsNone(calls[0]["previous_b"])
        self.assertEqual(calls[1]["start"], common_b)
        self.assertEqual(calls[3]["start"], common_b)
        self.assertEqual(calls[2]["start"], checkpoint(calls[1]["branch"] + "-B"))
        self.assertEqual(calls[4]["start"], checkpoint(calls[3]["branch"] + "-B"))
        self.assertTrue(result["probe_gate_pending"])

    def test_persistence_first_invalid_event_hard_stops(self):
        events = [successful_event("common"), successful_event("fixed2", primary_eligible=False)]
        with patch("m1.load_task", return_value={}), patch("m1.run_event", side_effect=events) as event:
            result = m1.persistence(None, self.origin, Path("."), self.selections, [], MemoryJournal())
        self.assertEqual(event.call_count, 2)
        self.assertEqual(result["status"], "m1_failed")
        self.assertFalse(result["m1_complete"])

    def test_state_panel_has_all_selected_and_persistence_states_dedup_aliases(self):
        events = [successful_event("screen1"), successful_event("screen2")]
        events[1]["B"] = events[1]["A"]
        selections = {"T1": {"selected": {"realizations": events}}}
        persisted = {"events": [successful_event("common"), successful_event("fixed2")]}
        panel = m1.state_panel(self.origin, selections, persisted)
        self.assertEqual(len(panel), 8)  # origin + 4 screening + 4 persistence - one alias
        aliased = next(row for row in panel if row["state"] == "screen2/A")
        self.assertEqual(aliased["aliases"], ["screen2/B"])


class LaunchGateTests(unittest.TestCase):
    def test_missing_freeze_blocks_before_backend_or_credentials(self):
        with tempfile.TemporaryDirectory() as directory, patch("m1.Backend.connect") as connect:
            with self.assertRaisesRegex(RuntimeError, "freeze.json"):
                m1.run(directory, project_id="explicit-project", keychain_service="explicit-service")
            connect.assert_not_called()

    def test_pending_noise_method_and_missing_counts_have_no_defaults(self):
        self.assertEqual(m1.validate_measurement_manifest(measurement_manifest())["noise_repeats"], 3)
        for change in ({"paired_noise_bound": {"method": "pending"}}, {"paired_noise_bound": {}},
                       {"noise_repeats": None}, {"noise_repeats": 2}, {"noise_repeats": 4}, {"kl": {}},
                       {"paired_noise_bound": {"method": "v1_operational_2_5_sd", "multiplier": 3.0}},
                       {"paired_noise_bound": {"method": "observed_max_absolute_difference"}},
                       {"deterministic_bounds": {"validation_nll": 0.01}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                m1.validate_measurement_manifest({**measurement_manifest(), **change})
        descriptive = measurement_manifest()
        descriptive["kl"]["prompt_selection"] = "first 32 hash-ordered frozen repair prompts"
        descriptive["paired_noise_bound"]["note"] = "explicitly registered construction"
        self.assertEqual(m1.validate_measurement_manifest(descriptive), descriptive)

    def test_partial_runner_does_not_authorize_paid_launch(self):
        checked = {"freeze_sha256": "frozen", "measurement": measurement_manifest(),
                   "paid_launch_allowed": False}
        with tempfile.TemporaryDirectory() as directory, patch("m1.preflight", return_value=checked), \
             patch("m1.Backend.connect") as connect:
            with self.assertRaisesRegex(RuntimeError, "full M1 runner"):
                m1.run(directory, project_id="project", keychain_service="service")
            connect.assert_not_called()
        self.assertTrue(m1.M1_RUNNER_COMPLETE)  # Complete code still needs its tested public input freeze.

    def test_existing_scientific_stop_returns_without_connecting(self):
        checked = {"freeze_sha256": "frozen", "measurement": measurement_manifest()}
        project_id = "explicit-project"
        identity = {"freeze_sha256": "frozen", "project_sha256": hashlib.sha256(project_id.encode()).hexdigest(),
                    "model": m1.MODEL, "lora_seed": m1.LORA_SEED}
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "runs/m1/journal.jsonl")
            journal.call("m1/identity", identity, lambda: identity)
            result = {"status": "m1_failed", "failure": "task", "m1_complete": False}
            journal.call("m1/stopped", {}, lambda: result)
            with patch("m1.preflight", return_value=checked), patch("m1.Backend.connect") as connect:
                self.assertEqual(m1.run(directory, project_id=project_id, keychain_service="explicit"), result)
                connect.assert_not_called()

    def test_unresolved_operation_blocks_even_after_successful_preflight(self):
        checked = {"freeze_sha256": "frozen", "measurement": measurement_manifest()}
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "runs/m1/journal.jsonl")
            with self.assertRaises(TimeoutError):
                journal.call("paid", {}, lambda: (_ for _ in ()).throw(TimeoutError()))
            with patch("m1.preflight", return_value=checked), patch("m1.Backend.connect") as connect:
                with self.assertRaises(m1.calibrate.AmbiguousOperation):
                    m1.run(directory, project_id="explicit-project", keychain_service="explicit")
                connect.assert_not_called()

    def test_run_requires_explicit_keychain_service_and_has_no_m2_command(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            m1.main(["run", "--project-id", "project"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            m1.main(["m2"])

    def test_preflight_cli_reports_blocker_without_claiming_completion(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch("m1.preflight", side_effect=RuntimeError("missing freeze")), \
             patch("m1.Backend.connect") as connect:
            self.assertEqual(m1.main(["preflight"]), 2)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["m1_complete"])
        self.assertFalse(report["main_run_authorized"])
        connect.assert_not_called()


class ProbeStageTests(unittest.TestCase):
    def setUp(self):
        self.origin = checkpoint("origin")
        self.panel = [{"state": "cycle0", "checkpoint": self.origin},
                      {"state": "screen/task/A", "checkpoint": checkpoint("A")}]
        self.calls = []

    def load(self, root, candidate):
        probe_class = "structured" if candidate in m1.STRUCTURED_PROBES else "language"
        return {"candidate": candidate, "class": probe_class, "batch_size": 32 if probe_class == "structured" else 8,
                "train": [], "val": [], "data_sha256": candidate}

    def sweep(self, api, start, probe, rate, journal, branch, evaluate, *, extended):
        self.calls.append({"candidate": probe["candidate"], "rate": rate, "start": start,
                           "branch": branch, "extended": extended})
        budget, cadence = m1.PROBE_BUDGETS[probe["class"]]
        budget *= 2 if extended else 1
        steps = m1.rules.registered_probe_steps(budget, cadence)
        reduction = 8 if extended else 7
        return {"status": "complete", "steps": steps, "losses": [10 - reduction * step / budget for step in steps]}

    def test_all_18_pairs_full_panel_and_three_independent_standard_origin_executions(self):
        with patch("m1.load_probe", side_effect=self.load), patch("m1.calibrate.probe_sweep", side_effect=self.sweep):
            result = m1.calibrate_probes(None, self.origin, Path("."), self.panel, MemoryJournal(), measurement_manifest())
        self.assertEqual(result["status"], "probe_calibration_complete")
        self.assertEqual(len(result["candidates"]), 18)
        self.assertEqual(len(self.calls), 18 * 5)  # extended + 2 states + 2 extra origin executions
        for candidate in m1.STRUCTURED_PROBES + m1.LANGUAGE_PROBES:
            for rate in LEARNING_RATES:
                calls = [call for call in self.calls if (call["candidate"], call["rate"]) == (candidate, rate)]
                self.assertEqual(sum(call["extended"] for call in calls), 1)
                standard_origin = [call for call in calls if not call["extended"] and call["start"] == self.origin]
                self.assertEqual(len(standard_origin), 3)
                self.assertEqual(len({call["branch"] for call in standard_origin}), 3)
        language = result["selected"]["language"]
        self.assertIn(245, language["state_curves"][0]["steps"])
        self.assertNotIn(245, m1.rules.registered_probe_steps(490, 25))
        self.assertEqual(len(language["noise_replicates"]), 3)
        self.assertEqual(language["noise_bounds"], {"t50": 0.0, "tdelta": 0.0})
        self.assertEqual(result["retention_noise_bound"], 0.0)

    def test_undefined_repeat_clock_invalidates_candidate_before_selection(self):
        def sweep(*args, **kwargs):
            result = self.sweep(*args, **kwargs)
            probe, rate, branch = args[2], args[3], args[5]
            if probe["candidate"] == "graph_path" and rate == 1e-5 and branch.endswith("/noise/1"):
                result["losses"] = [10.0] * len(result["steps"])
            return result

        with patch("m1.load_probe", side_effect=self.load), patch("m1.calibrate.probe_sweep", side_effect=sweep):
            result = m1.calibrate_probes(None, self.origin, Path("."), self.panel, MemoryJournal(), measurement_manifest())
        failed = next(row for row in result["candidates"] if row["candidate"] == "graph_path" and row["learning_rate"] == 1e-5)
        self.assertFalse(failed["passes"])
        self.assertEqual(failed["noise_bounds"], {"t50": None, "tdelta": None})
        self.assertEqual(failed["failure"], "unstable_cycle0_probe_clocks")
        self.assertEqual(result["selected"]["structured"]["learning_rate"], 3e-5)

    def test_failed_state_coverage_has_no_extra_noise_runs_and_no_smaller_panel(self):
        def sweep(*args, **kwargs):
            result = self.sweep(*args, **kwargs)
            if args[1] != self.origin:
                result["losses"] = [10.0] * len(result["steps"])
            return result

        with patch("m1.load_probe", side_effect=self.load), patch("m1.calibrate.probe_sweep", side_effect=sweep):
            result = m1.calibrate_probes(None, self.origin, Path("."), self.panel, MemoryJournal(), measurement_manifest())
        self.assertEqual(result["status"], "m1_failed")
        self.assertEqual(len(self.calls), 18 * 3)
        self.assertTrue(all(len(row["states"]) == 2 for row in result["candidates"]))
        self.assertFalse(any("/noise/" in call["branch"] for call in self.calls))


def diversity_result(valid_per_item=2, *, length=10, truncated=0.0):
    return {"pass1": valid_per_item / 8, "pass8": 1.0, "coverage_gap": 1 - valid_per_item / 8,
            "unique_valid_outputs": float(valid_per_item), "strategy_families": float(valid_per_item),
            "strategy_family_concentration": 1 / valid_per_item, "sampled_token_surprisal": 1.0,
            "mean_completion_tokens": float(length), "truncation_rate": truncated,
            "results": [{"valid_outputs": valid_per_item, "lengths": [length] * 8} for _ in range(100)]}


class DiversityStageTests(unittest.TestCase):
    def setUp(self):
        self.examples = [{"prompt_tokens": [1, 2], "family_count": 4}] * 100
        self.manifest = {"path": "data/diversity/panel.jsonl", "sha256": "frozen",
                         "sampling": {"samples": 8, "temperature": 1.0, "max_tokens": 512, "seed": 100},
                         "safe_length_rule": {"max_truncation_rate": 0.0, "p95_tokens_at_most": 384}}

    def test_first_realization_qualifies_three_full_panels_supply_noise(self):
        calls = []

        class API:
            def sampler(self, path):
                return "sampler"

            def sample(self, sampler, prompts, **kwargs):
                calls.append((len(prompts), kwargs))
                return {"groups": [], "accounting": {"sample_tokens": 800}}

        outcomes = [diversity_result(valid) for valid in (2, 1, 3)]
        with patch("m1._read", return_value=self.manifest), patch("m1.data.load_rows", return_value=self.examples), \
             patch("m1.measure.diversity_summary", side_effect=outcomes):
            result = m1.calibrate_diversity(API(), checkpoint("origin"), Path("."), MemoryJournal(), measurement_manifest())
        self.assertEqual(result["selected"]["candidate"], "graph_coloring")
        self.assertEqual([call[1]["seed"] for call in calls], [100, 101, 102])
        self.assertTrue(all(call[0] == 100 and call[1]["samples"] == 8 for call in calls))
        selected = result["selected"]
        self.assertEqual(selected["qualification"]["qualification_realization"], 0)
        self.assertEqual(len(selected["realizations"]), 3)
        expected = 2.5 * 2**0.5 * 0.125
        self.assertAlmostEqual(selected["noise"]["coverage_gap"]["bound"], expected)

    def test_failed_first_panel_does_not_get_extra_noise_repeats(self):
        calls = []

        class API:
            def sampler(self, path):
                return "sampler"

            def sample(self, sampler, prompts, **kwargs):
                calls.append(kwargs["seed"])
                return {"groups": [], "accounting": {}}

        with patch("m1._read", return_value=self.manifest), patch("m1.data.load_rows", return_value=self.examples), \
             patch("m1.measure.diversity_summary", side_effect=[diversity_result(8), *[diversity_result(2)] * 3]):
            result = m1.calibrate_diversity(API(), checkpoint("origin"), Path("."), MemoryJournal(), measurement_manifest())
        self.assertEqual(result["selected"]["candidate"], "set_partition")
        self.assertEqual(calls, [100, 100, 101, 102])
        self.assertEqual(len(result["attempts"]), 2)

    def test_first_panel_length_guard_and_family_coverage(self):
        self.assertFalse(m1.diversity_qualification(self.examples, diversity_result(length=385), self.manifest)["passes"])
        self.assertFalse(m1.diversity_qualification(self.examples, diversity_result(truncated=0.01), self.manifest)["passes"])
        examples = [{"family_count": 4}] * 79 + [{"family_count": 3}] * 21
        self.assertFalse(m1.diversity_qualification(examples, diversity_result(), self.manifest)["passes"])


if __name__ == "__main__":
    unittest.main()
