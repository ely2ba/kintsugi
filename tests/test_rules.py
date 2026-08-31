"""Deterministic SPEC v2 rule tests; no model, network, or paid execution."""

import copy
import math
import unittest

import rules


def thresholds():
    return {**rules.if_thresholds(43),
            **rules.acquisition_thresholds(0, 10, 0, 10, 0.1)}


def learning_points(values, cap=False):
    values = list(values)
    if cap:
        values.extend([values[-1]] * (120 - len(values)))
    return [{"step": index, "gate": gate, "if_score": score, "heldout": heldout}
            for index, (gate, score, heldout) in enumerate(values, 1)]


def reference_trajectory(rate, gate=8, heldout=7):
    return {"learning_rate": rate, "points": [
        {"step": step, "gate": 0 if step == 0 else gate, "heldout": 0 if step == 0 else heldout}
        for step in range(0, 121, 5)]}


def probe_state(name="state", headroom=4, t50=16, final_progress=None):
    steps = list(range(0, 33, 4))
    progress = [step / (2 * t50) for step in steps]
    if final_progress is not None:
        progress = [min(value, final_progress) for value in progress]
    return {"state": name, "steps": steps,
            "losses": [6 + headroom - headroom * value for value in progress]}


def candidate_summary(name, rate=1e-5, headroom=4, t50=16, passes=True):
    return {"candidate": name, "learning_rate": rate, "minimum_headroom": headroom,
            "median_t50": t50, "standard_budget": 32, "passes": passes}


def observed_clocks():
    return {"t50": 16, "tdelta": 8, "t50_status": "observed", "tdelta_status": "observed"}


def scheduled_cycles():
    orders = {
        "O1": ("T1", "T2", "T3", "T4", "T5", "T6", "T7"),
        "O2": ("T2", "T4", "T1", "T6", "T3", "T7", "T5"),
        "O3": ("T5", "T3", "T6", "T1", "T7", "T4", "T2"),
        "O4": ("T7", "T6", "T2", "T5", "T4", "T3", "T1"),
    }
    return [{"arm": arm, "order": order, "cycle": cycle, "task": task,
             "lineage": f"{arm}-{order}", "primary_eligible": True, "repair_success": True,
             "probes": {probe: {checkpoint: observed_clocks() for checkpoint in ("A", "B")}
                        for probe in ("structured", "language")}}
            for arm in ("fixed", "rolling") for order, tasks in orders.items()
            for cycle, task in enumerate(tasks, 1)]


def forty_two_cycles():
    rows = scheduled_cycles()
    for index in range(7):
        task_rows = [row for row in rows if row["task"] == f"T{index + 1}"]
        for excluded in (index % 8, (index + 1) % 8):
            task_rows[excluded]["primary_eligible"] = False
    return rows


def positive_trainability(**changes):
    arguments = {
        "t50_effect": 12, "tdelta_effect": 4, "relative_t50_effect": 0.12,
        "t50_noise_bound": 2, "tdelta_noise_bound": 1,
        "order_effects": {"O1": 1, "O2": 1, "O3": 1, "O4": -1},
        "task_adjusted_effect": 1, "coverage_met": True,
    }
    arguments.update(changes)
    return rules.trainability_claim(**arguments)


class AcquisitionRulesTest(unittest.TestCase):
    def test_if_integer_rounding_and_inclusive_endpoints(self):
        self.assertEqual(rules.if_thresholds(43), {"damage_low": 26, "damage_high": 36, "recovery_target": 41})
        for score in range(61):
            actual = rules.if_thresholds(score)
            self.assertEqual(actual["damage_low"], math.ceil(0.60 * score))
            self.assertEqual(actual["damage_high"], math.floor(0.85 * score))
            self.assertEqual(actual["recovery_target"], math.ceil(0.95 * score))
        for score in (-1, 61, 43.5, float("nan")):
            with self.assertRaises(ValueError):
                rules.if_thresholds(score)

    def test_independent_reference_maxima_and_registered_120_budget(self):
        trajectories = [reference_trajectory(rate) for rate in (1e-5, 3e-5, 1e-4)]
        trajectories[0]["points"][1]["gate"] = 20
        trajectories[1]["points"][-1]["heldout"] = 30
        result = rules.acquisition_references(trajectories)
        self.assertEqual(result["status"], "defined")
        self.assertEqual((result["gate_ref"], result["heldout_ref"]), (20, 30))
        self.assertEqual(result["registered_steps"], list(range(0, 121, 5)))
        self.assertEqual(len(result["valid_points"]), 75)
        trajectories[0]["points"].append({"step": 125, "gate": 100, "heldout": 100})
        with self.assertRaises(ValueError):
            rules.acquisition_references(trajectories)

    def test_failure_prefix_contributes_but_no_later_points(self):
        trajectories = [reference_trajectory(rate) for rate in (1e-5, 3e-5, 1e-4)]
        trajectories[0]["points"][1]["gate"] = 20
        trajectories[0]["points"][2]["gate"] = float("nan")
        trajectories[0]["points"][3].update(gate=1000, heldout=1000)
        trajectories[1]["failure_step"] = 7
        trajectories[1]["points"][1]["heldout"] = 25
        trajectories[1]["points"][2].update(gate=2000, heldout=2000)
        result = rules.acquisition_references(trajectories)
        self.assertEqual((result["gate_ref"], result["heldout_ref"]), (20, 25))
        self.assertEqual(result["complete_learning_rates"], [1e-4])
        self.assertEqual([row["last_valid_step"] for row in result["trajectory_status"]], [5, 5, 120])

    def test_at_least_one_complete_valid_trajectory_is_required(self):
        trajectories = [reference_trajectory(rate) for rate in (1e-5, 3e-5, 1e-4)]
        for trajectory in trajectories:
            trajectory["points"][-1]["valid"] = False
        result = rules.acquisition_references(trajectories)
        self.assertEqual(result["status"], "no_complete_valid_trajectory")
        self.assertIsNone(result["gate_ref"])
        self.assertIsNone(result["heldout_ref"])
        for trajectory in trajectories:
            trajectory["points"] = trajectory["points"][:2]
        self.assertEqual(rules.acquisition_references(trajectories)["status"], "incomplete_sweep")

    def test_unfinished_lr_is_not_a_frozen_reference_or_numeric_failure(self):
        trajectories = [reference_trajectory(rate) for rate in (1e-5, 3e-5, 1e-4)]
        trajectories[0]["points"] = trajectories[0]["points"][:2]
        result = rules.acquisition_references(trajectories)
        self.assertEqual(result["status"], "incomplete_sweep")
        self.assertFalse(result["sweep_complete"])
        self.assertIsNone(result["gate_ref"])
        self.assertEqual(result["complete_learning_rates"], [3e-5, 1e-4])

    def test_repeat_zero_noise_does_not_fail_or_average_baseline(self):
        trajectories = [reference_trajectory(rate) for rate in (1e-4, 3e-5, 1e-5)]
        trajectories[0]["points"][0].update(gate=0.2, heldout=0.3)
        trajectories[1]["points"][0].update(gate=0.1, heldout=0.2)
        frozen = {"gate": 0.04, "heldout": 0.05}
        result = rules.acquisition_references(trajectories, frozen)
        self.assertEqual(result["status"], "defined")
        self.assertEqual((result["gate0"], result["heldout0"]), (0.04, 0.05))
        self.assertEqual(len(result["cycle0_observations"]), 3)
        fallback = rules.acquisition_references(trajectories)
        self.assertEqual((fallback["gate0"], fallback["heldout0"]), (0, 0))

    def test_reference_rejects_unregistered_or_missing_interior_points(self):
        for replacement in (4, 6, 10):
            trajectories = [reference_trajectory(rate) for rate in (1e-5, 3e-5, 1e-4)]
            trajectories[0]["points"][1]["step"] = replacement
            with self.assertRaises(ValueError):
                rules.acquisition_references(trajectories)
        with self.assertRaises(ValueError):
            rules.acquisition_references([reference_trajectory(1e-5)])

    def test_oriented_negative_nll_and_noise_vs_reference_movement(self):
        result = rules.acquisition_thresholds(-5, -3, -6, -2, 0.01)
        self.assertAlmostEqual(result["gate_competence"], -3.6)
        self.assertEqual(result["heldout_competence"], -4)
        self.assertAlmostEqual(result["minimum_movement"], 0.3)
        noisy = rules.acquisition_thresholds(-5, -3, -6, -2, 0.2)
        self.assertEqual(noisy["minimum_movement"], 1)
        zero = rules.acquisition_thresholds(0, 0, 0, 0, 0)
        self.assertEqual(zero["minimum_movement"], 0)
        self.assertEqual(zero["gate_competence"], 0)

    def test_start_headroom_movement_and_band_equalities(self):
        for score in (26, 36):
            start = {"if_score": 41, "gate": 5.5}
            result = rules.learning_decision("fixed", start, learning_points([(7, score, 5)]), thresholds())
            self.assertTrue(result["primary_eligible"])
            self.assertEqual(result["classification"], "valid_acquisition")
        status = rules.acquisition_status(5.5, 7, thresholds())
        self.assertEqual((status["headroom"], status["movement"]), (1.5, 1.5))
        self.assertTrue(status["sufficient_headroom"] and status["sufficient_movement"])

    def test_repair_and_control_have_different_first_stop_conditions(self):
        start = {"if_score": 43, "gate": 0}
        points = learning_points([(7, 43, 5), (8, 36, 5), (9, 30, 8)])
        repair = rules.learning_decision("fixed", start, points, thresholds())
        control = rules.learning_decision("learn-only", start, points, thresholds())
        self.assertEqual(repair["stop_step"], 2)
        self.assertEqual(control["stop_step"], 1)
        self.assertFalse(control["primary_eligible"])
        self.assertEqual(repair["checkpoint"]["gate"], 8)
        transferred = rules.learning_decision("learn-only", {"if_score": 43, "gate": 6},
                                              learning_points([(7, 43, 5), (7.5, 43, 5)]), thresholds())
        self.assertEqual(transferred["stop_step"], 2)
        self.assertEqual(transferred["classification"], "already_competent")

    def test_heldout_cannot_change_selected_stopping_checkpoint(self):
        result = rules.learning_decision("fixed", {"if_score": 43, "gate": 0},
                                         learning_points([(7, 36, 4), (9, 30, 8)]), thresholds())
        self.assertEqual(result["stop_step"], 1)
        self.assertEqual(result["classification"], "heldout_competence_fail")
        self.assertFalse(result["primary_eligible"])
        pending = rules.learning_decision("fixed", {"if_score": 43, "gate": 0},
                                          learning_points([(7, 36, None)]), thresholds())
        self.assertTrue(pending["stop"] and pending["awaiting_heldout"])
        self.assertEqual(pending["classification"], "heldout_pending")
        self.assertNotIn("heldout_competence_fail", pending["flags"])

    def test_every_unique_learning_failure_category(self):
        scenarios = [
            ("undamageable", {"if_score": 43, "gate": 0}, [(7, 37, 5)], True),
            ("band_overshoot", {"if_score": 43, "gate": 0}, [(7, 25, 5)], True),
            ("damage_before_competence", {"if_score": 43, "gate": 0}, [(0, 30, 0), (7, 25, 5)], True),
            ("competence_unmet", {"if_score": 43, "gate": 0}, [(6, 43, 0)], True),
            ("heldout_competence_fail", {"if_score": 43, "gate": 0}, [(7, 36, 4)], False),
            ("already_competent", {"if_score": 43, "gate": 6}, [(7, 36, 5)], False),
            ("unrestored_start", {"if_score": 40, "gate": 0}, [(7, 36, 5)], False),
        ]
        for expected, start, values, cap in scenarios:
            with self.subTest(category=expected):
                result = rules.learning_decision("rolling", start, learning_points(values, cap), thresholds())
                self.assertEqual(result["classification"], expected)
                self.assertEqual(result["flags"], [expected])
                self.assertFalse(result["primary_eligible"])

    def test_overlapping_failure_flags_have_no_invented_precedence(self):
        result = rules.learning_decision("fixed", {"if_score": 40, "gate": 6},
                                         learning_points([(7, 36, 4)]), thresholds())
        self.assertEqual(result["classification"], "mixed_gate_failure")
        self.assertEqual(set(result["flags"]), {"already_competent", "unrestored_start", "heldout_competence_fail"})
        overshoot = rules.learning_decision("fixed", {"if_score": 43, "gate": 0},
                                            learning_points([(0, 25, 0), (7, 25, 5)], True), thresholds())
        self.assertEqual(overshoot["classification"], "mixed_gate_failure")
        self.assertEqual(set(overshoot["flags"]), {"band_overshoot", "damage_before_competence"})
        unmet = rules.learning_decision("fixed", {"if_score": 43, "gate": 0},
                                        learning_points([(6, 30, 0)], True), thresholds())
        self.assertEqual(set(unmet["flags"]), {"competence_unmet", "damage_before_competence"})

    def test_no_unique_failure_category_is_mixed_and_censored_not_cap_dose(self):
        result = rules.learning_decision("fixed", {"if_score": 43, "gate": 0},
                                         learning_points([(7, 43, 5), (6, 30, 5), (7, 25, 5)], True), thresholds())
        self.assertEqual(result["classification"], "mixed_gate_failure")
        self.assertEqual(result["flags"], [])
        self.assertEqual(result["stop_step"], 120)
        self.assertIsNone(result["valid_damage_step"])
        self.assertEqual(result["valid_damage_status"], "right_censored")
        unmet = rules.learning_decision("fixed", {"if_score": 43, "gate": 0},
                                        learning_points([(6, 43, 0)], True), thresholds())
        self.assertIsNone(unmet["first_competence_step"])
        self.assertEqual(unmet["competence_status"], "right_censored")

    def test_incomplete_learning_is_not_a_cap_failure(self):
        result = rules.learning_decision("fixed", {"if_score": 43, "gate": 0},
                                         learning_points([(3, 43, 0)]), thresholds())
        self.assertFalse(result["stop"])
        self.assertEqual(result["classification"], "learning")
        self.assertIsNone(result["stop_step"])
        self.assertEqual(result["competence_status"], "incomplete")
        self.assertEqual(result["flags"], [])

    def test_if_measurements_must_be_real_integer_criterion_counts(self):
        with self.assertRaises(ValueError):
            rules.learning_decision("fixed", {"if_score": 43, "gate": 0},
                                    learning_points([(7, 35.5, 5)]), thresholds())
        with self.assertRaises(ValueError):
            rules.repair_decision("fixed", 30, [{"step": 5, "if_score": 41.5}], thresholds())


class RepairAndCohortTest(unittest.TestCase):
    def test_first_scheduled_check_and_achieved_overshoot(self):
        waiting = rules.repair_decision("fixed", 30, [], thresholds())
        self.assertEqual(waiting["next_check_step"], 5)
        self.assertFalse(waiting["stop"])
        checks = [{"step": 5, "if_score": 45}, {"step": 10, "if_score": 50}]
        result = rules.repair_decision("fixed", 30, checks, thresholds())
        self.assertEqual((result["repair_steps"], result["criterion_score"]), (5, 45))
        self.assertEqual(result["time_to_success"], 5)
        with self.assertRaises(ValueError):
            rules.repair_decision("fixed", 30, [{"step": 4, "if_score": 45}], thresholds())

    def test_cap_failure_is_not_an_observed_success_time(self):
        checks = [{"step": step, "if_score": 40} for step in range(5, 151, 5)]
        failure = rules.repair_decision("rolling", 30, checks, thresholds())
        self.assertEqual(failure["status"], "repair_failure")
        self.assertFalse(failure["repair_success"])
        self.assertTrue(failure["repair_effect_observed"])
        self.assertEqual(failure["repair_steps"], 150)
        self.assertIsNone(failure["time_to_success"])
        self.assertEqual(failure["time_to_success_status"], "right_censored")
        self.assertEqual(failure["censor_step"], 150)
        checks[-1]["if_score"] = 41
        success = rules.repair_decision("rolling", 30, checks, thresholds())
        self.assertTrue(success["repair_success"])
        self.assertEqual(success["time_to_success"], 150)
        self.assertIsNone(success["censor_step"])

    def test_noop_and_control_identity_are_not_zero_repair_effects(self):
        for arm, score, status in (("fixed", 41, "no_repair_required"), ("learn-only", 20, "learn_only_control")):
            with self.subTest(arm=arm):
                result = rules.repair_decision(arm, score, [], thresholds())
                self.assertEqual(result["status"], status)
                self.assertEqual(result["repair_steps"], 0)
                self.assertEqual(result["identity_difference"], 0)
                self.assertFalse(result["repair_effect_observed"])
                self.assertIsNone(result["repair_success"])
                effect = rules.repair_effect(8, 8, arm, 0)
                self.assertEqual(effect["identity_difference"], 0)
                self.assertIsNone(effect["repair_effect"])
                self.assertFalse(effect["repair_effect_observed"])
        actual = rules.repair_effect(8, 10, "fixed", 150)
        self.assertEqual(actual["repair_effect"], 2)
        missing = rules.repair_effect(None, None, "fixed", 5)
        self.assertIsNone(missing["repair_effect"])
        with self.assertRaises(ValueError):
            rules.repair_effect(8, 10, "fixed", 0)
        with self.assertRaises(ValueError):
            rules.repair_decision("fixed", 41, [{"step": 5, "if_score": 43}], thresholds())

    def test_failed_repair_stays_primary_and_strict_cohort_is_separate(self):
        learning = rules.learning_decision("fixed", {"if_score": 43, "gate": 0},
                                           learning_points([(7, 30, 5)]), thresholds())
        self.assertTrue(learning["primary_eligible"])
        rows = [{"lineage": lineage, "arm": "fixed", "cycle": cycle,
                 "primary_eligible": not (lineage == "broken" and cycle == 4),
                 "repair_success": not (lineage == "whole" and cycle == 3)}
                for lineage in ("whole", "broken") for cycle in range(1, 8)]
        rows.append({"lineage": "control", "arm": "learn-only", "cycle": 1,
                     "primary_eligible": True, "repair_success": True})
        cohorts = rules.cohort_summary(rows)
        self.assertEqual(len(cohorts["primary"]), 13)
        self.assertEqual(len(cohorts["criterion_matched"]), 12)
        self.assertEqual(cohorts["strict_lineages"], ["whole"])
        self.assertEqual(len(cohorts["strict"]), 7)
        self.assertIn(rows[2], cohorts["primary"])
        self.assertNotIn(rows[2], cohorts["criterion_matched"])
        self.assertEqual(len(cohorts["lifecycle"]), 15)


class RetentionTest(unittest.TestCase):
    def test_strict_denominator_noise_and_movement_guards(self):
        for acquired, noise, minimum in ((0.25, 0.05, 0.1), (0.3, 0.01, 0.3), (0, 0, 0), (-1, 0, 0)):
            with self.subTest(acquired=acquired):
                result = rules.normalized_retention(0, acquired, 1, noise, minimum)
                self.assertIsNone(result["value"])
                self.assertEqual(result["status"], "undefined_denominator")
        self.assertIsNotNone(rules.normalized_retention(0, 0.300001, 0.2, 0.01, 0.3)["value"])

    def test_retention_not_clipped_and_accepts_oriented_negative_nll(self):
        self.assertEqual(rules.normalized_retention(-5, -3, -6, 0.01, 0.1)["value"], -0.5)
        self.assertEqual(rules.normalized_retention(-5, -3, -2, 0.01, 0.1)["value"], 1.5)
        self.assertIsNone(rules.normalized_retention(-5, -3, None, 0.01, 0.1)["value"])

    def test_prior_task_mean_excludes_current_and_undefined_with_coverage(self):
        rows = [
            {"task": "T1", "baseline": 0, "acquired": 1, "current": 0.5, "noise_sd": 0, "minimum_movement": 0.1},
            {"task": "T2", "baseline": 0, "acquired": 0.1, "current": 0, "noise_sd": 0, "minimum_movement": 0.1},
            {"task": "T3", "baseline": 0, "acquired": 1, "current": 1.5, "noise_sd": 0, "minimum_movement": 0.1},
        ]
        summary = rules.retention_summary(rows, "T3")
        self.assertEqual(summary["current"], 1.5)
        self.assertEqual(summary["prior_mean"], 0.5)
        self.assertEqual(summary["all_mean"], 1)
        self.assertEqual(summary["prior_coverage"], {"defined": 1, "total": 2, "fraction": 0.5})
        first_cycle = rules.retention_summary(rows[:1], "T1")
        self.assertIsNone(first_cycle["prior_mean"])
        self.assertEqual(first_cycle["prior_coverage"], {"defined": 0, "total": 0, "fraction": None})


class ProbeRulesTest(unittest.TestCase):
    def test_first_crossing_interpolation_equalities_and_nonmonotonic_curve(self):
        steps, losses = [0, 4, 8, 12, 16], [10, 9, 7, 8, 6]
        self.assertEqual(rules.interpolated_crossing_time(steps, losses, 8), 6)
        self.assertEqual(rules.interpolated_crossing_time(steps, losses, 9), 4)
        clocks = rules.trainability_clocks(steps, losses, 6, 1, 0.1)
        self.assertEqual(clocks["t50"], 6)
        self.assertEqual(clocks["tdelta"], 4)
        self.assertEqual(clocks["best_loss_reduction"], 4)
        self.assertEqual(clocks["final_loss"], 6)
        self.assertEqual(clocks["t50_status"], "observed")

    def test_headroom_equalities_and_independent_companion_guard(self):
        normalized = rules.trainability_clocks([0, 4], [1, 0.4], 0, 0.25, 0.2)
        self.assertEqual(normalized["t50_status"], "observed")
        self.assertIsNone(normalized["tdelta"])
        self.assertEqual(normalized["tdelta_status"], "undefined_headroom")
        companion = rules.trainability_clocks([0, 4], [10, 8], 6, 1, 0.6)
        self.assertEqual(companion["tdelta"], 2)
        self.assertEqual(companion["tdelta_status"], "observed")

    def test_undefined_and_censored_clocks_never_become_zero_or_budget(self):
        for losses, reference in (([6, 5], 6), ([5, 4], 6)):
            result = rules.trainability_clocks([0, 4], losses, reference, 1, 0)
            self.assertIsNone(result["t50"])
            self.assertIsNone(result["tdelta"])
            self.assertEqual(result["t50_status"], "undefined_headroom")
            self.assertIsNone(result["progress"])
        censored = rules.trainability_clocks([0, 4, 8], [10, 9.8, 9.6], 6, 1, 0)
        self.assertIsNone(censored["t50"])
        self.assertIsNone(censored["tdelta"])
        self.assertEqual(censored["t50_status"], "right_censored")
        self.assertEqual(censored["censor_step"], 8)
        initial_only = rules.trainability_clocks([0], [10], 6, 1, 0)
        self.assertIsNone(initial_only["t50"])
        self.assertIsNone(rules.interpolated_crossing_time([0, 4], [8, 7], 8))
        observed_at_budget = rules.trainability_clocks([0, 4, 8], [10, 9, 8], 6, 1, 0)
        self.assertEqual(observed_at_budget["t50"], 8)
        self.assertEqual(observed_at_budget["t50_status"], "observed")

    def test_curve_validation_and_unclipped_progress(self):
        for steps, losses in (([], []), ([0, 0], [2, 1]), ([4, 0], [2, 1]), ([0, 4], [2, float("nan")])):
            with self.assertRaises(ValueError):
                rules.trainability_clocks(steps, losses, 0, 1, 0)
        result = rules.trainability_clocks([0, 4, 8], [10, 11, 5], 6, 1, 0)
        self.assertEqual(result["progress"], [0, -0.25, 1.25])
        self.assertAlmostEqual(rules.validation_loss_auc([0, 4, 8], [10, 9, 8]), 9)

    def test_probe_twice_budget_reference_and_nondivisible_final_evaluation(self):
        self.assertEqual(rules.registered_probe_steps(245, 25)[-2:], [225, 245])
        steps = rules.registered_probe_steps(490, 25)
        self.assertEqual(steps[-2:], [475, 490])
        losses = [10 - step / 1000 for step in steps]
        result = rules.probe_reference_target(steps, losses, 245, 25)
        self.assertEqual(result["reference_step"], 490)
        with self.assertRaises(ValueError):
            rules.probe_reference_target(steps[:-1], losses[:-1], 245, 25)

    def test_m1_dynamic_range_endpoints_and_final_progress_equality(self):
        for t50 in (6.4, 25.6):
            result = rules.m1_probe_candidate("graph", 1e-5, [probe_state(t50=t50)], 6, 0.1, 32, 4)
            self.assertTrue(result["passes"], result)
            self.assertAlmostEqual(result["median_t50"], t50)
        final_equal = rules.m1_probe_candidate("graph", 1e-5, [probe_state(final_progress=0.60)], 6, 0.1, 32, 4)
        self.assertTrue(final_equal["passes"])
        self.assertAlmostEqual(final_equal["states"][0]["final_progress"], 0.60)
        for t50 in (6.3, 25.7):
            result = rules.m1_probe_candidate("graph", 1e-5, [probe_state(t50=t50)], 6, 0.1, 32, 4)
            self.assertFalse(result["passes"])
            self.assertFalse(result["states"][0]["checks"]["dynamic_coverage"])

    def test_m1_minimum_headroom_delta_and_every_state_requirement(self):
        states = [probe_state("A", 4), probe_state("B", 8)]
        result = rules.m1_probe_candidate("graph", 3e-5, states, 6, 0.6, 32, 4)
        self.assertEqual(result["delta_l"], 1)
        self.assertEqual(result["minimum_headroom"], 4)
        self.assertTrue(result["passes"])
        for failing_state in (probe_state("B", 0.4), probe_state("B", 0), probe_state("B", 4, final_progress=0.599)):
            failed = rules.m1_probe_candidate("graph", 3e-5, [states[0], failing_state], 6, 0.1, 32, 4)
            self.assertFalse(failed["passes"])
        no_crossing = rules.m1_probe_candidate("graph", 3e-5, [probe_state(final_progress=0.4)], 6, 0.1, 32, 4)
        self.assertIsNone(no_crossing["median_t50"])
        self.assertFalse(no_crossing["passes"])

    def test_probe_selection_priority_is_lr_then_headroom_then_center_then_order(self):
        records = [candidate_summary("A", 1e-5, 4), candidate_summary("A", 3e-5, 100), candidate_summary("B", 1e-5, 5)]
        self.assertEqual(rules.select_probe(records, ["A", "B"])["candidate"], "B")
        records[0]["passes"] = False
        self.assertEqual(rules.select_probe(records, ["A", "B"])["learning_rate"], 3e-5)
        records = [candidate_summary("A", headroom=5, t50=12), candidate_summary("B", headroom=5, t50=15)]
        self.assertEqual(rules.select_probe(records, ["A", "B"])["candidate"], "B")
        records = [candidate_summary("B", t50=17), candidate_summary("A", t50=15)]
        self.assertEqual(rules.select_probe(records, ["A", "B"])["candidate"], "A")
        for record in records:
            record["passes"] = False
        self.assertIsNone(rules.select_probe(records, ["A", "B"]))

    def test_recipe_requires_both_repairs_then_median_target_dose_and_lower_lr(self):
        recipes = [{"learning_rate": rate, "batch_size": 32, "realizations": [
            {"realization": name, "primary_eligible": True, "repair_success": True, "target_tokens": dose}
            for name, dose in zip(("r1", "r2"), doses)]}
            for rate, doses in ((1e-5, (100, 300)), (3e-5, (190, 190)))]
        self.assertEqual(rules.select_task_recipe(recipes)["learning_rate"], 3e-5)
        recipes[1]["realizations"][1]["repair_success"] = False
        self.assertEqual(rules.select_task_recipe(recipes)["learning_rate"], 1e-5)
        recipes[1]["realizations"][1]["repair_success"] = True
        for row in recipes[1]["realizations"]:
            row["target_tokens"] = 200
        self.assertEqual(rules.select_task_recipe(recipes)["learning_rate"], 1e-5)
        recipes[1]["batch_size"] = 64
        with self.assertRaises(ValueError):
            rules.select_task_recipe(recipes)


class CoverageAndClaimsTest(unittest.TestCase):
    def test_42_of_56_and_six_of_eight_per_task(self):
        rows = forty_two_cycles()
        result = rules.coverage_requirements(rows, ("structured", "language"))
        self.assertEqual((result["eligible"], result["scheduled"]), (42, 56))
        self.assertTrue(result["passes"])
        self.assertTrue(all(value["eligible"] == 6 and value["scheduled"] == 8 for value in result["per_task"].values()))
        next(row for row in rows if row["primary_eligible"])["primary_eligible"] = False
        self.assertFalse(rules.coverage_requirements(rows)["passes"])

    def test_task_minimum_and_all_orders_are_independent_requirements(self):
        rows = scheduled_cycles()
        for row in [row for row in rows if row["task"] == "T1"][:3]:
            row["primary_eligible"] = False
        result = rules.coverage_requirements(rows)
        self.assertEqual(result["eligible"], 53)
        self.assertFalse(result["per_task"]["T1"]["passes"])
        self.assertFalse(result["passes"])
        rows = scheduled_cycles()
        for row in rows:
            if row["arm"] == "fixed" and row["order"] == "O4":
                row["primary_eligible"] = False
        result = rules.coverage_requirements(rows)
        self.assertEqual(result["eligible"], 49)
        self.assertTrue(all(value["passes"] for value in result["per_task"].values()))
        self.assertFalse(result["passes"])

    def test_80_percent_requires_both_clocks_at_both_endpoints(self):
        rows = forty_two_cycles()
        for row in [row for row in rows if not row["primary_eligible"]][:3]:
            row["primary_eligible"] = True
        eligible = [row for row in rows if row["primary_eligible"]]
        self.assertEqual(len(eligible), 45)
        for row in eligible[36:]:
            row["probes"]["structured"]["B"]["tdelta"] = None
            row["probes"]["structured"]["B"]["tdelta_status"] = "right_censored"
        result = rules.coverage_requirements(rows, ("structured",))
        self.assertTrue(result["passes"])
        self.assertEqual(result["probe_coverage"]["structured"]["fraction"], 0.8)
        eligible[35]["probes"]["structured"]["A"]["t50_status"] = "right_censored"
        result = rules.coverage_requirements(rows, ("structured",))
        self.assertFalse(result["passes"])
        self.assertEqual(result["eligible"], 45)
        self.assertTrue(result["manipulation_passes"])

    def test_unclaimed_probe_does_not_mask_other_axis_and_missing_is_not_zero(self):
        rows = scheduled_cycles()
        for row in rows:
            row["probes"]["language"] = {}
        self.assertTrue(rules.coverage_requirements(rows, ("structured",))["passes"])
        self.assertFalse(rules.coverage_requirements(rows, ("structured", "language"))["passes"])
        zero = {"A": observed_clocks(), "B": observed_clocks()}
        zero["A"]["t50"] = 0
        self.assertFalse(rules.paired_clocks_available(zero))
        empty = rules.coverage_requirements([], ("structured",))
        self.assertIsNone(empty["probe_coverage"]["structured"]["fraction"])
        self.assertFalse(empty["passes"])
        with self.assertRaises(ValueError):
            rules.coverage_requirements(rows + [copy.deepcopy(rows[0])])

    def test_trainability_claim_all_conditions_and_strict_thresholds(self):
        self.assertEqual(positive_trainability()["claim"], "trainability_debt")
        failures = [
            {"relative_t50_effect": 0.10}, {"t50_effect": 2}, {"tdelta_effect": 1},
            {"tdelta_effect": -4}, {"relative_t50_effect": -0.12},
            {"order_effects": {"O1": 1, "O2": 1, "O3": -1, "O4": -1}},
            {"task_adjusted_effect": -1}, {"coverage_met": False}, {"t50_effect": None},
        ]
        for change in failures:
            with self.subTest(change=change):
                result = positive_trainability(**change)
                self.assertFalse(result["passes"])
                self.assertIsNone(result["claim"])
        restoration = positive_trainability(t50_effect=-12, tdelta_effect=-4, relative_t50_effect=-0.12,
                                            order_effects={"O1": -1, "O2": -1, "O3": -1, "O4": None},
                                            task_adjusted_effect=-1)
        self.assertEqual(restoration["claim"], "plasticity_restoration")
        self.assertEqual(restoration["orders"], {"valid": 3, "consistent": 3})

    def test_global_trainability_retains_domain_disagreement(self):
        debt = positive_trainability()
        neutral = positive_trainability(t50_effect=0.1, tdelta_effect=0.1, relative_t50_effect=0.01)
        self.assertTrue(rules.global_trainability_claim(debt, debt)["passes"])
        self.assertTrue(rules.global_trainability_claim(debt, neutral)["passes"])
        opposite = positive_trainability(t50_effect=-12, tdelta_effect=-4, relative_t50_effect=-0.12,
                                         order_effects={"O1": -1, "O2": -1, "O3": -1, "O4": 1},
                                         task_adjusted_effect=-1)
        result = rules.global_trainability_claim(debt, opposite)
        self.assertEqual(result["status"], "domain_dependent")
        self.assertIsNone(result["claim"])
        unavailable = positive_trainability(t50_effect=None)
        self.assertFalse(rules.global_trainability_claim(debt, unavailable)["passes"])

    def test_retention_and_diversity_claim_noise_margins_and_strategy_evidence(self):
        orders = {"O1": -1, "O2": -1, "O3": -1, "O4": 1}
        self.assertEqual(rules.retention_claim(-0.06, 0.04, orders)["claim"], "retention_loss")
        self.assertFalse(rules.retention_claim(-0.05, 0.01, orders)["passes"])
        self.assertFalse(rules.retention_claim(-0.06, 0.06, orders)["passes"])
        self.assertFalse(rules.retention_claim(None, 0.01, orders)["passes"])
        self.assertEqual(rules.diversity_claim(-0.04, 0.02, orders, -1)["claim"], "diversity_loss")
        self.assertFalse(rules.diversity_claim(-0.03, 0.01, orders, -1)["passes"])
        self.assertFalse(rules.diversity_claim(-0.04, 0.04, orders, -1)["passes"])
        self.assertFalse(rules.diversity_claim(-0.04, 0.01, orders, None)["passes"])
        self.assertFalse(rules.diversity_claim(-0.04, 0.01, orders, 1)["passes"])


if __name__ == "__main__":
    unittest.main()
