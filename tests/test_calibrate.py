import json
import tempfile
import unittest
from pathlib import Path

from calibrate import AmbiguousOperation, Journal, batch_at, reference_sweep
from protocol import LEARNING_RATES, REFERENCE_EVAL_STEPS


class FakeBackend:
    """Local state-machine double: never instantiates an API client."""
    def __init__(self, invalid_lr=None):
        self.states = {"origin-state": {"step": 0, "lr": None}}
        self.branches = []
        self.updates = []
        self.evaluations = []
        self.invalid_lr = invalid_lr

    def branch(self, state, resume=False):
        self.branches.append((state, resume))
        return dict(self.states[state])

    def train_step(self, client, rows, *, learning_rate, step, warmup_steps):
        if step != client["step"] + 1 or warmup_steps != 10:
            raise AssertionError("wrong step or warmup on resumed branch")
        self.updates.append((learning_rate, step))
        client.update(step=step, lr=learning_rate)
        if learning_rate == self.invalid_lr and step == 11:
            return {"valid": False, "failure": "nonfinite completed loss", "accounting": {"train_tokens": 32}}
        return {"valid": True, "accounting": {"train_tokens": 32, "gradient_target_tokens": 16}}

    def save(self, client, name, *, step, resume=False):
        state = f"{name}-state-{step}"
        self.states[state] = dict(client)
        return {"state_path": state, "sampler_path": f"{name}-sampler-{step}"}

    def evaluate(self, client, sampler, step):
        if client["step"] != step:
            raise AssertionError("evaluation used stale weights")
        self.evaluations.append((client["lr"], step))
        return {"gate": -10 + min(step, 20) / 20,
                "heldout": -12 + min(step, 55) / 55,
                "accounting": {"forward_tokens": 16}}


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "journal.jsonl"
        self.task = {"candidate": "test", "batch_size": 16,
                     "reference": [{"tokens": [1, 2]}] * (120 * 16),
                     "data_sha256": "frozen-test-data"}
        self.origin = {"state_path": "origin-state", "sampler_path": "origin-sampler"}

    def test_registered_sweep_has_no_early_stop_and_independent_origin(self):
        backend = FakeBackend()
        result = reference_sweep(backend, self.origin, self.task, Journal(self.path), backend.evaluate)
        self.assertEqual(len(backend.updates), 360)
        self.assertEqual(backend.branches, [("origin-state", False)] * 3)
        self.assertEqual(len(backend.evaluations), 75)
        for trajectory in result["trajectories"]:
            self.assertEqual([p["step"] for p in trajectory["points"]], list(REFERENCE_EVAL_STEPS))
        # Everything is durable; replay makes no model, optimizer, save or eval calls.
        old_counts = (len(backend.updates), len(backend.evaluations), len(backend.branches))
        self.assertEqual(reference_sweep(backend, self.origin, self.task, Journal(self.path), backend.evaluate), result)
        self.assertEqual(old_counts, (len(backend.updates), len(backend.evaluations), len(backend.branches)))

    def test_numerical_failure_truncates_only_that_lr(self):
        backend = FakeBackend(invalid_lr=3e-5)
        result = reference_sweep(backend, self.origin, self.task, Journal(self.path), backend.evaluate)
        failed = result["trajectories"][1]
        self.assertEqual(failed["failure_step"], 11)
        self.assertEqual([p["step"] for p in failed["points"]], [0, 5, 10])
        self.assertEqual(len(result["trajectories"][2]["points"]), 25)

    def test_safe_checkpoint_resume_restores_optimizer_without_replay(self):
        backend = FakeBackend()

        class StopAfterCheckpoint(Journal):
            def call(self, operation, inputs, function):
                result = super().call(operation, inputs, function)
                if operation.endswith("1e-05/update/031"):
                    raise InterruptedError("local stop after durable checkpoint")
                return result

        with self.assertRaises(InterruptedError):
            reference_sweep(backend, self.origin, self.task, StopAfterCheckpoint(self.path), backend.evaluate)
        reference_sweep(backend, self.origin, self.task, Journal(self.path), backend.evaluate)
        self.assertEqual(len(backend.updates), 360)
        self.assertEqual(len(set(backend.updates)), 360)
        self.assertIn(("reference-test-1e-05-state-31", True), backend.branches)

    def test_short_reference_rejected_before_calls(self):
        backend = FakeBackend()
        task = {**self.task, "reference": self.task["reference"][:-1]}
        with self.assertRaises(ValueError):
            reference_sweep(backend, self.origin, task, Journal(self.path), backend.evaluate)
        self.assertFalse(backend.branches)

    def test_batch_does_not_wrap_or_reuse(self):
        self.assertEqual(batch_at(list(range(8)), 2, 3), [4, 5])
        with self.assertRaises(ValueError):
            batch_at(list(range(8)), 2, 5)


class JournalTests(unittest.TestCase):
    def test_ambiguous_call_never_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            calls = []

            def interrupted():
                calls.append(1)
                raise ConnectionError("unknown response")

            with self.assertRaises(ConnectionError):
                Journal(path).call("one", {"step": 1}, interrupted)
            with self.assertRaises(AmbiguousOperation):
                Journal(path).call("one", {"step": 1}, interrupted)
            self.assertEqual(calls, [1])

    def test_changed_contract_and_nonfinite_results_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "journal.jsonl")
            journal.call("one", {"lr": 1e-5}, lambda: {"value": 1})
            with self.assertRaises(RuntimeError):
                journal.call("one", {"lr": 1e-4}, lambda: {"value": 2})
            with self.assertRaises(ValueError):
                journal.call("two", {}, lambda: {"value": float("nan")})
            self.assertIn("two", journal.pending)


if __name__ == "__main__":
    unittest.main()
