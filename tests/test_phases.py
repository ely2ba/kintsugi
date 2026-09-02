"""Offline phase integration checks: finite local states, no SDK or API calls."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from calibrate import Journal, canonical, learning_phase, probe_sweep, repair_phase
from protocol import ORDERS, schedule_seed


class MemoryJournal(Journal):
    """Exercise the real journal and evaluation scope in a temporary directory."""
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        super().__init__(Path(self.temporary.name) / "journal.jsonl")

    def __del__(self):
        self.temporary.cleanup()


class PhaseBackend:
    def __init__(self):
        self.states = {"origin-state": {"step": 71}, "A-state": {"step": 29}, "previous-B-state": {"step": 47}}
        self.branches, self.updates, self.repairs, self.saves, self.teachers = [], [], [], [], []

    def branch(self, state, *, resume=False):
        self.branches.append((state, resume))
        client = dict(self.states[state])
        if not resume:
            client["step"] = 0  # A fresh optimizer must not inherit phase history.
        return client

    def train_step(self, client, rows, *, learning_rate, step, warmup_steps):
        if client["step"] != step - 1:
            raise AssertionError("training replayed a step or restored stale optimizer state")
        client["step"] = step
        self.updates.append({"step": step, "rows": [row["id"] for row in rows],
                             "lr": learning_rate, "warmup": warmup_steps})
        return {"valid": True, "accounting": {"gradient_target_tokens": len(rows)}}

    def sampler(self, sampler_path):
        self.teachers.append(sampler_path)
        return sampler_path

    def repair_step(self, client, teacher, prompts, *, step, seed):
        if client["step"] != step - 1:
            raise AssertionError("repair replayed a step or restored stale optimizer state")
        client["step"] = step
        self.repairs.append({"step": step, "teacher": teacher,
                             "prompts": [list(prompt) for prompt in prompts], "seed": seed})
        return {"accounting": {"gradient_target_tokens": 256}}

    def save(self, client, name, *, step, resume=False):
        if client["step"] != step:
            raise AssertionError("saved stale client")
        checkpoint = {"state_path": f"{name}-state-{step}", "sampler_path": f"{name}-sampler-{step}", "step": step}
        self.states[checkpoint["state_path"]] = dict(client)
        self.saves.append((name, step, resume))
        return checkpoint

    def save_state(self, client, name, *, step):
        if client["step"] != step:
            raise AssertionError("saved stale client")
        checkpoint = {"state_path": f"{name}-state-{step}", "step": step}
        self.states[checkpoint["state_path"]] = dict(client)
        self.saves.append((name, step, "state_only"))
        return checkpoint


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.origin = {"state_path": "origin-state", "sampler_path": "origin-sampler"}
        self.a = {"state_path": "A-state", "sampler_path": "A-sampler"}
        self.previous_b = {"state_path": "previous-B-state", "sampler_path": "previous-B-sampler"}
        self.thresholds = {"damage_low": 26, "damage_high": 36, "recovery_target": 41,
                           "gate_competence": 0.7, "heldout_competence": 0.5, "minimum_movement": 0.15}
        self.task = {"candidate": "arithmetic_derivations", "batch_size": 16,
                     "learning_rate": 3e-5, "data_sha256": "frozen-learning-data"}
        self.rows = [{"id": index} for index in range(120 * 16)]
        self.pool = [[index, 2000 + index] for index in range(2000)]

    def learning_evaluator(self, points, *, start=None, heldout=0.9):
        calls = []
        start = start or {"if_score": 43, "gate": 0.0, "heldout": 0.0}

        def evaluate(kind, client, sampler, step):
            self.assertEqual(client["step"], step)
            calls.append((kind, step, sampler))
            if kind == "start":
                return dict(start)
            if kind == "heldout":
                return {"heldout": heldout(step) if callable(heldout) else heldout}
            return dict(points(step))

        return evaluate, calls

    def learn(self, backend, evaluate, *, arm="fixed", journal=None, branch="learn/test"):
        return learning_phase(backend, self.origin, self.task, self.rows, arm, self.thresholds,
                              journal or MemoryJournal(), branch, evaluate)

    def repair(self, backend, evaluate, *, arm="fixed", cycle=2, task="T3", branch="repair/test", journal=None, a_score=35):
        return repair_phase(backend, self.a, a_score, self.origin, self.previous_b, arm, cycle, task,
                            self.pool, self.thresholds, journal or MemoryJournal(), branch, evaluate)

    def test_learning_selects_first_real_A_and_counts_only_actual_exposure(self):
        backend = PhaseBackend()
        points = {1: {"gate": 0.3, "if_score": 40}, 2: {"gate": 0.8, "if_score": 38},
                  3: {"gate": 0.9, "if_score": 36}}
        evaluate, calls = self.learning_evaluator(lambda step: points[step])
        result = self.learn(backend, evaluate)
        self.assertEqual(result["A"]["step"], 3)
        self.assertEqual(result["decision"]["classification"], "valid_acquisition")
        self.assertTrue(result["primary_eligible"])
        self.assertEqual(result["target_tokens"], 3 * 16)
        self.assertEqual([point["step"] for point in result["points"]], [1, 2, 3])
        self.assertEqual([(kind, step) for kind, step, _ in calls],
                         [("start", 0), ("update", 1), ("update", 2), ("update", 3), ("heldout", 3)])
        self.assertEqual([row for update in backend.updates for row in update["rows"]], list(range(48)))
        self.assertEqual(backend.branches, [("origin-state", False)])
        self.assertTrue(all(update["warmup"] == 10 for update in backend.updates))

    def test_learn_only_ignores_both_high_and_low_IF_outside_damage_band(self):
        for protected in (43, 12):
            with self.subTest(if_score=protected):
                backend = PhaseBackend()
                evaluate, _ = self.learning_evaluator(lambda step: {"gate": 0.4 if step == 1 else 0.8, "if_score": protected})
                result = self.learn(backend, evaluate, arm="learn-only")
                self.assertEqual(result["A"]["step"], 2)
                self.assertEqual(len(backend.updates), 2)
                self.assertFalse(result["primary_eligible"])

    def test_heldout_failure_saves_selected_A_and_never_searches_later_steps(self):
        backend = PhaseBackend()
        evaluate, calls = self.learning_evaluator(lambda step: {"gate": 0.8, "if_score": 35},
                                                   heldout=lambda step: 0.1 if step == 1 else 0.9)
        result = self.learn(backend, evaluate)
        self.assertEqual(result["A"]["step"], 1)
        self.assertEqual(result["decision"]["classification"], "heldout_competence_fail")
        self.assertFalse(result["primary_eligible"])
        self.assertEqual(len(backend.updates), 1)
        self.assertEqual([(kind, step) for kind, step, _ in calls if kind == "heldout"], [("heldout", 1)])

    def test_learning_cap_and_terminal_classifications(self):
        cases = {
            "competence_unmet": lambda step: {"gate": 0.1, "if_score": 43},
            "undamageable": lambda step: {"gate": 0.8, "if_score": 43},
            "band_overshoot": lambda step: {"gate": 0.8, "if_score": 43 if step == 1 else 25},
            "damage_before_competence": lambda step: {"gate": 0.1 if step < 3 else 0.8, "if_score": 34 if step == 1 else 25},
        }
        for classification, points in cases.items():
            with self.subTest(classification=classification):
                backend = PhaseBackend()
                evaluate, calls = self.learning_evaluator(points)
                result = self.learn(backend, evaluate)
                self.assertEqual(len(backend.updates), 120)
                self.assertEqual(result["A"]["step"], 120)
                self.assertEqual(result["target_tokens"], 120 * 16)
                self.assertEqual(result["decision"]["classification"], classification)
                self.assertEqual(result["decision"]["valid_damage_status"], "right_censored")
                self.assertFalse(result["primary_eligible"])
                self.assertEqual([step for kind, step, _ in calls if kind == "update"], list(range(1, 121)))
                self.assertEqual([step for kind, step, _ in calls if kind == "heldout"], [120])

    def test_start_validity_labels_do_not_prevent_real_learning(self):
        cases = ((43, 0.65, "already_competent"), (40, 0.0, "unrestored_start"),
                 (40, 0.65, "mixed_gate_failure"))
        for protected, gate, classification in cases:
            with self.subTest(classification=classification):
                backend = PhaseBackend()
                evaluate, _ = self.learning_evaluator(lambda step: {"gate": 0.8, "if_score": 35},
                                                       start={"if_score": protected, "gate": gate, "heldout": 0.0})
                result = self.learn(backend, evaluate)
                self.assertEqual(len(backend.updates), 1)
                self.assertEqual(result["decision"]["classification"], classification)
                self.assertFalse(result["primary_eligible"])

    def test_noop_and_control_alias_A_with_zero_cost_and_no_model_calls(self):
        for arm, score, status in (("fixed", 41, "no_repair_required"),
                                   ("rolling", 45, "no_repair_required"),
                                   ("learn-only", 12, "learn_only_control")):
            with self.subTest(arm=arm, score=score):
                backend = PhaseBackend()
                result = repair_phase(backend, self.a, score, self.origin, None, arm, 1, "T3", [],
                                      self.thresholds, MemoryJournal(), f"alias/{arm}",
                                      lambda *_: self.fail("no-op must not evaluate or contact a model"))
                self.assertEqual(result["B"], {**self.a, "alias_of": "A"})
                self.assertEqual(result["target_tokens"], 0)
                self.assertEqual(result["checks"], [])
                self.assertEqual(result["decision"]["status"], status)
                self.assertEqual(result["decision"]["identity_difference"], 0.0)
                self.assertFalse(result["decision"]["repair_effect_observed"])
                self.assertFalse(backend.branches or backend.updates or backend.repairs or backend.saves or backend.teachers)

    def test_fixed_and_rolling_teachers_use_frozen_checkpoint_paths(self):
        for arm, cycle, expected in (("fixed", 1, "origin-sampler"), ("fixed", 7, "origin-sampler"),
                                     ("rolling", 1, "origin-sampler"), ("rolling", 2, "previous-B-sampler")):
            with self.subTest(arm=arm, cycle=cycle):
                backend = PhaseBackend()
                result = self.repair(backend, lambda sampler, step: {"if_score": 41}, arm=arm, cycle=cycle)
                self.assertEqual(result["B"]["step"], 5)
                self.assertEqual(backend.teachers, [expected])
                self.assertEqual({update["teacher"] for update in backend.repairs}, {expected})
                self.assertEqual(backend.branches, [("A-state", False)])

    def test_task_prompt_and_rollout_schedules_are_paired_across_orders_and_arms(self):
        schedules = []
        for order, sequence in ORDERS.items():
            for arm in ("fixed", "rolling"):
                backend = PhaseBackend()
                self.repair(backend, lambda sampler, step: {"if_score": 41}, arm=arm,
                            cycle=sequence.index("T3") + 1, task="T3", branch=f"{order}/{arm}/repair")
                schedules.append([(update["prompts"], update["seed"]) for update in backend.repairs])
                for update in backend.repairs:
                    self.assertEqual(len(update["prompts"]), 64)
                    self.assertEqual(len({tuple(prompt) for prompt in update["prompts"]}), 64)
                    self.assertEqual(update["seed"], schedule_seed("T3", "repair-rollouts", update["step"]))
        self.assertTrue(all(schedule == schedules[0] for schedule in schedules))
        backend = PhaseBackend()
        self.repair(backend, lambda sampler, step: {"if_score": 41}, task="T6")
        self.assertNotEqual([(update["prompts"], update["seed"]) for update in backend.repairs], schedules[0])

    def test_repair_first_scheduled_success_including_150_and_cap_failure(self):
        for success_step in (10, 150, None):
            with self.subTest(success_step=success_step):
                backend, calls = PhaseBackend(), []

                def evaluate(sampler, step):
                    calls.append(step)
                    return {"if_score": 41 if success_step is not None and step >= success_step else 40}

                result = self.repair(backend, evaluate)
                terminal = success_step or 150
                self.assertEqual(len(backend.repairs), terminal)
                self.assertEqual(calls, list(range(5, terminal + 1, 5)))
                self.assertEqual(result["B"]["step"], terminal)
                self.assertEqual(result["target_tokens"], terminal * 256)
                self.assertEqual(result["decision"]["status"], "repaired" if success_step else "repair_failure")
                self.assertEqual(result["decision"]["time_to_success"], success_step)
                self.assertEqual(result["decision"]["time_to_success_status"], "observed" if success_step else "right_censored")

    def test_learning_resume_at_durable_update_or_gate_never_replays_paid_work(self):
        for boundary in ("update/003", "evaluate/003"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as directory:
                backend = PhaseBackend()
                path = Path(directory) / "learning.jsonl"
                evaluate, calls = self.learning_evaluator(lambda step: {"gate": 0.8, "if_score": 38 if step < 3 else 35})

                class StopAtBoundary(Journal):
                    def call(self, operation, inputs, function, **kwargs):
                        result = super().call(operation, inputs, function, **kwargs)
                        if operation == "learn/test/" + boundary:
                            raise InterruptedError("stopped after a durable operation")
                        return result

                with self.assertRaises(InterruptedError):
                    self.learn(backend, evaluate, journal=StopAtBoundary(path))
                self.assertFalse(Journal(path).pending)
                result = self.learn(backend, evaluate, journal=Journal(path))
                self.assertEqual([update["step"] for update in backend.updates], [1, 2, 3])
                self.assertEqual([(kind, step) for kind, step, _ in calls],
                                 [("start", 0), ("update", 1), ("update", 2), ("update", 3), ("heldout", 3)])
                self.assertEqual(result["A"]["step"], 3)
                self.assertIn(("learn-test-state-3", True), backend.branches)
                before = (len(backend.branches), len(backend.updates), len(backend.saves), len(calls))
                self.assertEqual(self.learn(backend, evaluate, journal=Journal(path)), result)
                self.assertEqual(before, (len(backend.branches), len(backend.updates), len(backend.saves), len(calls)))

    def test_learning_preoptimizer_recovery_abandons_client_and_restores_prior_optimizer(self):
        class APIConnectionError(Exception):
            pass

        class FailOnceBackend(PhaseBackend):
            failed = False

            def save(self, client, name, *, step, resume=False):
                state = f"tinker://run/weights/{name}-state-{step}"
                sampler = f"tinker://run/sampler_weights/{name}-sampler-{step}"
                checkpoint = {"state_path": state, "sampler_path": sampler, "step": step}
                self.states[state] = dict(client)
                self.saves.append((name, step, resume))
                return checkpoint

            def train_step(self, client, rows, *, learning_rate, step, warmup_steps):
                if step == 3 and not self.failed:
                    self.failed = True
                    raise APIConnectionError("connection lost before optimizer call")
                return super().train_step(client, rows, learning_rate=learning_rate,
                                          step=step, warmup_steps=warmup_steps)

        with tempfile.TemporaryDirectory() as directory:
            backend = FailOnceBackend()
            path = Path(directory) / "learning.jsonl"
            evaluate, _ = self.learning_evaluator(
                lambda step: {"gate": 0.8, "if_score": 38 if step < 3 else 35})
            with self.assertRaises(APIConnectionError):
                self.learn(backend, evaluate, journal=Journal(path))
            operation = "learn/test/update/003"
            journal = Journal(path)
            attempt = journal.attempts[operation]
            self.assertTrue(attempt["blocked"])
            digest = journal.pending[operation]["inputs_sha256"]
            evidence = {"old_client": "abandoned", "source_step": 2}
            journal._record({
                "type": "premutation_recovery_authorized", "operation": operation,
                "inputs_sha256": digest, "recoverable": True, "failed_attempt": 1,
                "failed_error_type": "APIConnectionError",
                "failure_boundary": "forward_backward_before_optimizer",
                "forward_backward_applied": "unknown", "optimizer_applied": False,
                "remote_gradient_state_abandoned": True, "failed_client_process_exited": True,
                "checkpoint_created": False,
                "training_updates_replayed": False,
                "source_checkpoint": "tinker://run/weights/learn-test-state-2",
                "remote_training_runs_created": [], "evidence": evidence,
                "evidence_sha256": hashlib.sha256(canonical(evidence).encode()).hexdigest(),
                "reason": "Abandon the failed client and restore the durable step-2 optimizer state.",
            })
            result = self.learn(backend, evaluate, journal=Journal(path))
            self.assertEqual(result["A"]["step"], 3)
            self.assertEqual([update["step"] for update in backend.updates], [1, 2, 3])
            self.assertIn(("tinker://run/weights/learn-test-state-2", True), backend.branches)
            self.assertEqual(Journal(path).completed[operation]["attempt"], 2)

    def test_repair_resume_after_durable_criterion_restores_without_replay(self):
        for success_step in (5, 10):
            with self.subTest(success_step=success_step), tempfile.TemporaryDirectory() as directory:
                backend, calls = PhaseBackend(), []
                path = Path(directory) / "repair.jsonl"

                def evaluate(sampler, step):
                    calls.append(step)
                    return {"if_score": 41 if step >= success_step else 40}

                class StopAfterCriterion(Journal):
                    def call(self, operation, inputs, function, **kwargs):
                        result = super().call(operation, inputs, function, **kwargs)
                        if operation == "repair/test/criterion/005":
                            raise InterruptedError("stopped after a durable repair criterion")
                        return result

                with self.assertRaises(InterruptedError):
                    self.repair(backend, evaluate, journal=StopAfterCriterion(path))
                self.assertFalse(Journal(path).pending)
                result = self.repair(backend, evaluate, journal=Journal(path))
                self.assertEqual([update["step"] for update in backend.repairs], list(range(1, success_step + 1)))
                self.assertEqual(calls, list(range(5, success_step + 1, 5)))
                self.assertEqual(result["B"]["step"], success_step)
                self.assertIn(("repair-test-state-5", True), backend.branches)
                before = (len(backend.branches), len(backend.repairs), len(backend.saves), len(backend.teachers), len(calls))
                self.assertEqual(self.repair(backend, evaluate, journal=Journal(path)), result)
                self.assertEqual(before, (len(backend.branches), len(backend.repairs), len(backend.saves), len(backend.teachers), len(calls)))

    def test_standard_and_extended_probe_budgets_cadences_and_fresh_optimizers(self):
        for kind, standard, cadence in (("structured", 32, 4), ("language", 245, 25)):
            for extended in (False, True):
                with self.subTest(kind=kind, extended=extended):
                    budget = standard * (2 if extended else 1)
                    backend, calls, journal = PhaseBackend(), [], MemoryJournal()
                    probe = {"candidate": "graph_path" if kind == "structured" else "wikipedia_vi", "class": kind,
                             "batch_size": 16, "data_sha256": "frozen-probe-data",
                             "train": [{"id": index} for index in range(budget * 16)]}

                    def evaluate(client, step):
                        self.assertEqual(client["step"], step)
                        calls.append(step)
                        return {"nll": 10.0 - step / 1000}

                    result = probe_sweep(backend, self.origin, probe, 1e-5, journal, "probe/test", evaluate, extended=extended)
                    expected = sorted(set(range(0, budget + 1, cadence)) | {budget})
                    self.assertEqual(result["steps"], expected)
                    self.assertEqual(calls, expected)
                    self.assertEqual([update["step"] for update in backend.updates], list(range(1, budget + 1)))
                    self.assertEqual([row for update in backend.updates for row in update["rows"]], list(range(budget * 16)))
                    self.assertEqual(backend.branches, [("origin-state", False)])
                    self.assertTrue(all(update["warmup"] == 0 for update in backend.updates))
                    self.assertTrue(all(save[2] == "state_only" for save in backend.saves))
                    before = (len(backend.branches), len(backend.updates), len(backend.saves), len(calls))
                    self.assertEqual(probe_sweep(backend, self.origin, probe, 1e-5, journal, "probe/test", evaluate, extended=extended), result)
                    self.assertEqual(before, (len(backend.branches), len(backend.updates), len(backend.saves), len(calls)))


if __name__ == "__main__":
    unittest.main()
