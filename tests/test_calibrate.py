import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calibrate import (AmbiguousOperation, InvalidMeasurement, Journal, batch_at, canonical,
                       probe_sweep, reference_sweep)
import eval_cache
from protocol import LEARNING_RATES, REFERENCE_EVAL_STEPS


class FakeBackend:
    """Local state-machine double: never instantiates an API client."""
    def __init__(self, invalid_lr=None):
        self.states = {"origin-state": {"step": 0, "lr": None}}
        self.branches = []
        self.updates = []
        self.evaluations = []
        self.full_saves, self.state_only_saves = [], []
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
        self.full_saves.append((client["lr"], step))
        return {"state_path": state, "sampler_path": f"{name}-sampler-{step}"}

    def save_state(self, client, name, *, step):
        state = f"{name}-state-{step}"
        self.states[state] = dict(client)
        self.state_only_saves.append((client["lr"], step))
        return {"state_path": state}

    def evaluate(self, client, sampler, step):
        if client["step"] != step:
            raise AssertionError("evaluation used stale weights")
        if ((step == 0 and sampler != "origin-sampler")
                or (step and not sampler.endswith(f"-sampler-{step}"))):
            raise AssertionError("evaluation used stale sampler weights")
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
        evaluated_updates = set(REFERENCE_EVAL_STEPS) - {0}
        self.assertEqual([step for _, step in backend.full_saves],
                         [step for _ in LEARNING_RATES for step in sorted(evaluated_updates)])
        self.assertEqual([step for _, step in backend.state_only_saves],
                         [step for _ in LEARNING_RATES
                          for step in range(1, 121) if step not in evaluated_updates])
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
            def call(self, operation, inputs, function, **kwargs):
                result = super().call(operation, inputs, function, **kwargs)
                if operation.endswith("1e-05/update/031"):
                    raise InterruptedError("local stop after durable checkpoint")
                return result

        with self.assertRaises(InterruptedError):
            reference_sweep(backend, self.origin, self.task, StopAfterCheckpoint(self.path), backend.evaluate)
        reference_sweep(backend, self.origin, self.task, Journal(self.path), backend.evaluate)
        self.assertEqual(len(backend.updates), 360)
        self.assertEqual(len(set(backend.updates)), 360)
        self.assertIn(("reference-test-1e-05-state-31", True), backend.branches)

    def test_verified_post_update_save_recovery_does_not_repeat_update(self):
        class SamplerSaveFailure(FakeBackend):
            def save(self, client, name, *, step, resume=False):
                saved = super().save(client, name, step=step, resume=resume)
                if not hasattr(self, "interrupted_checkpoint"):
                    self.interrupted_checkpoint = saved
                    raise TypeError("sampler save rejected overwrite after state saved")
                return saved

        backend = SamplerSaveFailure()
        with self.assertRaises(TypeError):
            reference_sweep(backend, self.origin, self.task, Journal(self.path), backend.evaluate)
        journal = Journal(self.path)
        operation = "reference/test/1e-05/update/005"
        self.assertEqual(set(journal.pending), {operation})
        self.assertEqual(backend.updates, [(1e-5, step) for step in range(1, 6)])
        checkpoint = backend.interrupted_checkpoint
        self.assertEqual(backend.states[checkpoint["state_path"]]["step"], 5)
        # Mirrors an explicit verified recovery, not automatic handling of an
        # ambiguous call. Lost training loss stays null; scheduled evals remain.
        journal._append({"type": "complete", "operation": operation,
                         "inputs_sha256": journal.pending[operation]["inputs_sha256"],
                         "elapsed_seconds": None,
                         "result": {"valid": True, "nll": None, "q": None,
                                    "optimizer_applied": True,
                                    "accounting": {"train_tokens": 32, "gradient_target_tokens": 16},
                                    "checkpoint": checkpoint}})
        result = reference_sweep(backend, self.origin, self.task, Journal(self.path), backend.evaluate)
        self.assertEqual(len(backend.updates), 360)
        self.assertEqual(len(set(backend.updates)), 360)
        self.assertIn((checkpoint["state_path"], True), backend.branches)
        self.assertEqual([p["step"] for p in result["trajectories"][0]["points"]], list(REFERENCE_EVAL_STEPS))

    def test_short_reference_rejected_before_calls(self):
        backend = FakeBackend()
        task = {**self.task, "reference": self.task["reference"][:-1]}
        with self.assertRaises(ValueError):
            reference_sweep(backend, self.origin, task, Journal(self.path), backend.evaluate)
        self.assertFalse(backend.branches)

    def test_sampling_evaluation_resume_preserves_all_unique_updates_and_cadence(self):
        backend = FakeBackend()
        task = {**self.task, "metric": "verifier_success"}
        attempted = []

        def evaluate(client, sampler, step):
            attempted.append((client["lr"], step))
            if (client["lr"], step) == (1e-5, 15) and attempted.count((1e-5, 15)) == 1:
                raise eval_cache.SamplingTransportError("interrupted after a cached prompt")
            return backend.evaluate(client, sampler, step)

        with self.assertRaises(eval_cache.SamplingTransportError):
            reference_sweep(backend, self.origin, task, Journal(self.path), evaluate)
        pending = Journal(self.path)
        operation = "reference/test/1e-05/evaluate/015"
        self.assertEqual(set(pending.pending), {operation})
        self.assertTrue(pending.pending[operation]["recoverable"])
        self.assertEqual(len(backend.updates), 15)
        before = self.path.read_bytes()
        result = reference_sweep(backend, self.origin, task, Journal(self.path), evaluate)
        self.assertTrue(self.path.read_bytes().startswith(before))
        self.assertEqual(len(backend.updates), 360)
        self.assertEqual(len(set(backend.updates)), 360)
        self.assertEqual(len(backend.evaluations), 75)
        self.assertEqual(attempted.count((1e-5, 15)), 2)
        self.assertEqual([point["step"] for point in result["trajectories"][0]["points"]], list(REFERENCE_EVAL_STEPS))
        journal = Journal(self.path)
        self.assertEqual(journal.completed[operation]["attempt"], 2)
        self.assertTrue(journal.completed[operation]["timing_complete"])
        calls = list(attempted)
        reference_sweep(backend, self.origin, task, journal, evaluate)
        self.assertEqual(attempted, calls)

    def test_native_nll_evaluation_failure_remains_ambiguous(self):
        backend = FakeBackend()
        task = {**self.task, "metric": "negative_nll"}

        def interrupted(*args):
            raise ConnectionError("forward response unknown")

        with self.assertRaises(ConnectionError):
            reference_sweep(backend, self.origin, task, Journal(self.path), interrupted)
        journal = Journal(self.path)
        self.assertFalse(journal.recoverable_pending())
        self.assertNotIn("recoverable", next(iter(journal.pending.values())))
        with self.assertRaises(AmbiguousOperation):
            reference_sweep(backend, self.origin, task, journal, backend.evaluate)
        self.assertEqual(backend.updates, [])

    def test_batch_does_not_wrap_or_reuse(self):
        self.assertEqual(batch_at(list(range(8)), 2, 3), [4, 5])
        with self.assertRaises(ValueError):
            batch_at(list(range(8)), 2, 5)


class ProbeTests(unittest.TestCase):
    class Backend:
        def __init__(self):
            self.states = {"origin-state": {"step": 0}}
            self.branches, self.updates, self.state_saves = [], [], []

        def branch(self, state, resume=False):
            self.branches.append((state, resume))
            return dict(self.states[state])

        def train_step(self, client, rows, *, learning_rate, step, warmup_steps):
            if warmup_steps != 0 or step != client["step"] + 1:
                raise AssertionError("probe update did not resume exactly")
            client["step"] = step
            self.updates.append(step)
            return {"valid": True, "accounting": {"train_tokens": len(rows)}}

        def save_state(self, client, name, *, step):
            state = f"{name}-state-{step}"
            self.states[state] = dict(client)
            self.state_saves.append((name, step))
            return {"state_path": state, "resume_slot": step % 2}

        def save(self, *args, **kwargs):
            raise AssertionError("probe sweep must not export sampler weights")

    def test_probe_updates_save_only_state_and_resume_without_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            backend = self.Backend()
            probe = {"class": "structured", "candidate": "probe", "batch_size": 2,
                     "train": [{"tokens": [1, 2]}] * 64, "data_sha256": "frozen-probe"}
            evaluated = []

            def evaluate(client, step):
                self.assertEqual(client["step"], step)
                evaluated.append(step)
                return {"valid": True, "nll": 10.0 - step / 10}

            class StopAfterState(Journal):
                def call(self, operation, inputs, function, **kwargs):
                    result = super().call(operation, inputs, function, **kwargs)
                    if operation.endswith("/update/017"):
                        raise InterruptedError("stop after optimizer state is durable")
                    return result

            with self.assertRaises(InterruptedError):
                probe_sweep(backend, {"state_path": "origin-state"}, probe, 1e-4,
                            StopAfterState(path), "probe/run", evaluate)
            result = probe_sweep(backend, {"state_path": "origin-state"}, probe, 1e-4,
                                 Journal(path), "probe/run", evaluate)
            self.assertEqual(backend.updates, list(range(1, 33)))
            self.assertEqual(backend.state_saves, [("probe-run", step) for step in range(1, 33)])
            self.assertIn(("probe-run-state-17", True), backend.branches)
            self.assertEqual(result["steps"], list(range(0, 33, 4)))
            self.assertEqual(evaluated, list(range(0, 33, 4)))


class JournalTests(unittest.TestCase):
    operation = "reference/test/1e-05/evaluate/015"

    def legacy_pending(self, path, operation=None):
        journal = Journal(path)
        operation = operation or self.operation
        digest = hashlib.sha256(canonical({"state": "immutable"}).encode()).hexdigest()
        journal._append({"type": "inflight", "operation": operation, "inputs_sha256": digest})
        return digest

    def test_all_new_callables_have_scope_and_completed_replay_does_not_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "journal.jsonl")

            def local_work():
                self.assertIsNotNone(eval_cache.current_scope())
                return {"value": 1}

            self.assertEqual(journal.call("local", {}, local_work), {"value": 1})
            with patch("calibrate.eval_cache.evaluation_scope") as scope:
                self.assertEqual(journal.call("local", {}, lambda: self.fail("replayed")), {"value": 1})
                scope.assert_not_called()
            self.assertIsNone(eval_cache.current_scope())

    def test_failed_sampling_attempt_resumes_same_hash_and_sums_known_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            journal = Journal(path)
            journal.call("prior", {}, lambda: {"saved": True})

            def interrupted():
                raise eval_cache.SamplingTransportError("secret response or credential must not be logged")

            with patch("calibrate.time.monotonic", side_effect=[10.0, 12.0]), self.assertRaises(eval_cache.SamplingTransportError):
                journal.call(self.operation, {"state": "immutable"}, interrupted, recoverable=True)
            before = path.read_bytes()
            self.assertNotIn(b"secret response", before)
            resumed = Journal(path)
            self.assertEqual(resumed.call("prior", {}, lambda: self.fail("prior replayed")), {"saved": True})
            with patch("calibrate.time.monotonic", side_effect=[20.0, 23.0]):
                result = resumed.call(self.operation, {"state": "immutable"}, lambda: {"score": 1}, recoverable=True)
            self.assertEqual(result, {"score": 1})
            self.assertTrue(path.read_bytes().startswith(before))
            completed = Journal(path).completed[self.operation]
            self.assertEqual(completed["elapsed_seconds"], 5.0)
            self.assertEqual(completed["attempt_elapsed_seconds"], 3.0)
            self.assertTrue(completed["timing_complete"])
            self.assertEqual(completed["attempt"], 2)

    def test_legacy_evaluation_needs_explicit_matching_authorization_and_has_unknown_prior_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            digest = self.legacy_pending(path)
            journal = Journal(path)
            before = path.read_bytes()
            with self.assertRaises(AmbiguousOperation):
                journal.call(self.operation, {"state": "immutable"}, lambda: 1, recoverable=True)
            self.assertEqual(path.read_bytes(), before)
            journal._append({"type": "recovery_authorized", "operation": self.operation,
                             "inputs_sha256": digest, "recoverable": True,
                             "reason": "User approved cached immutable-checkpoint evaluation recovery."})
            journal = Journal(path)
            with patch("calibrate.time.monotonic", side_effect=[20.0, 23.0]):
                journal.call(self.operation, {"state": "immutable"}, lambda: {"score": 1}, recoverable=True)
            completed = Journal(path).completed[self.operation]
            self.assertEqual(completed["elapsed_seconds"], 3.0)
            self.assertFalse(completed["timing_complete"])

    def test_explicit_authorization_recovers_only_matching_blocked_sampling_setup_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            # Record the historical failure shape directly: before this fix the
            # setup exception was not translated at the sampler boundary.
            journal = Journal(path)
            digest = hashlib.sha256(canonical({"state": "immutable"}).encode()).hexdigest()
            journal._record({"type": "inflight", "operation": self.operation,
                             "inputs_sha256": digest, "recoverable": True})
            journal._record({"type": "failed", "operation": self.operation,
                             "inputs_sha256": digest, "attempt": 1,
                             "attempt_status": "failed", "elapsed_seconds": 2.0,
                             "retryable": False, "error_type": "APIConnectionError"})
            blocked = Journal(path)
            self.assertFalse(blocked.recoverable_pending())
            blocked._record({"type": "recovery_authorized", "operation": self.operation,
                             "inputs_sha256": digest, "recoverable": True,
                             "failed_attempt": 1, "failed_error_type": "APIConnectionError",
                             "reason": "User approved resuming the cached evaluation after setup failed."})
            resumed = Journal(path)
            self.assertTrue(resumed.recoverable_pending())
            result = resumed.call(self.operation, {"state": "immutable"},
                                  lambda: {"score": 1}, recoverable=True)
            self.assertEqual(result, {"score": 1})
            self.assertEqual(Journal(path).completed[self.operation]["attempt"], 2)

    def test_blocked_authorization_rejects_mismatched_or_permanent_failures(self):
        for failure_type, authorized_type in (("EvaluationRecoveryError", "EvaluationRecoveryError"),
                                              ("APIConnectionError", "ConnectionError")):
            with self.subTest(failure_type=failure_type, authorized_type=authorized_type), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "journal.jsonl"
                journal = Journal(path)
                digest = hashlib.sha256(canonical({"state": "immutable"}).encode()).hexdigest()
                journal._record({"type": "inflight", "operation": self.operation,
                                 "inputs_sha256": digest, "recoverable": True})
                journal._record({"type": "failed", "operation": self.operation,
                                 "inputs_sha256": digest, "attempt": 1,
                                 "attempt_status": "failed", "elapsed_seconds": 2.0,
                                 "retryable": False, "error_type": failure_type})
                journal._append({"type": "recovery_authorized", "operation": self.operation,
                                 "inputs_sha256": digest, "recoverable": True,
                                 "failed_attempt": 1, "failed_error_type": authorized_type,
                                 "reason": "must not authorize"})
                with self.assertRaises(AmbiguousOperation):
                    Journal(path)

    def test_verified_repair_client_setup_failure_resumes_exact_update_once(self):
        operation, inputs = "screen/task/0.0001/1/repair/update/005", {"step": 5, "from": "state-4"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            journal = Journal(path)
            digest = hashlib.sha256(canonical(inputs).encode()).hexdigest()
            journal._record({"type": "inflight", "operation": operation, "inputs_sha256": digest})
            journal._record({"type": "failed", "operation": operation, "inputs_sha256": digest,
                             "attempt": 1, "attempt_status": "failed", "elapsed_seconds": 2.0,
                             "retryable": False, "error_type": "APIConnectionError"})
            self.assertFalse(Journal(path).resumable_pending())
            Journal(path)._record({
                "type": "setup_recovery_authorized", "operation": operation,
                "inputs_sha256": digest, "recoverable": True, "failed_attempt": 1,
                "failed_error_type": "APIConnectionError", "training_updates_replayed": False,
                "source_checkpoint": "tinker://run/weights/state-4",
                "remote_training_runs_created": [], "evidence": {},
                "evidence_sha256": hashlib.sha256(canonical({}).encode()).hexdigest(),
                "reason": "Remote metadata proves client creation never reached Tinker.",
            })
            resumed = Journal(path)
            self.assertTrue(resumed.setup_recoverable_pending())
            self.assertTrue(resumed.resumable_pending())
            calls = []
            result = resumed.call(operation, inputs, lambda: calls.append(5) or {"step": 5})
            self.assertEqual(result, {"step": 5})
            self.assertEqual(calls, [5])
            completed = Journal(path).completed[operation]
            self.assertEqual(completed["attempt"], 2)
            self.assertFalse(Journal(path).pending)

    def test_setup_recovery_rejects_unverified_or_nonrepair_work(self):
        cases = (
            ("screen/task/learn/update/005", "APIConnectionError", []),
            ("screen/task/repair/update/005", "TimeoutError", []),
            ("screen/task/repair/update/005", "APIConnectionError", ["new-run"]),
        )
        for operation, failure_type, created in cases:
            with self.subTest(operation=operation, failure_type=failure_type, created=created), \
                    tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "journal.jsonl"
                journal = Journal(path)
                digest = hashlib.sha256(canonical({}).encode()).hexdigest()
                journal._record({"type": "inflight", "operation": operation, "inputs_sha256": digest})
                journal._record({"type": "failed", "operation": operation, "inputs_sha256": digest,
                                 "attempt": 1, "attempt_status": "failed", "elapsed_seconds": 1.0,
                                 "retryable": False, "error_type": failure_type})
                journal._append({
                    "type": "setup_recovery_authorized", "operation": operation,
                    "inputs_sha256": digest, "recoverable": True, "failed_attempt": 1,
                    "failed_error_type": failure_type, "training_updates_replayed": False,
                    "source_checkpoint": "tinker://run/weights/state-4",
                    "remote_training_runs_created": created, "evidence": {},
                    "evidence_sha256": hashlib.sha256(canonical({}).encode()).hexdigest(),
                    "reason": "must reject",
                })
                with self.assertRaises(AmbiguousOperation):
                    Journal(path)

    def test_verified_teacher_scoring_failure_resumes_before_mutation(self):
        operation, inputs = "screen/task/0.0001/1/repair/update/006", {"step": 6, "from": "state-5"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            journal = Journal(path)
            digest = hashlib.sha256(canonical(inputs).encode()).hexdigest()
            journal._record({"type": "inflight", "operation": operation, "inputs_sha256": digest})
            journal._record({"type": "failed", "operation": operation, "inputs_sha256": digest,
                             "attempt": 1, "attempt_status": "failed", "elapsed_seconds": 2.0,
                             "retryable": False, "error_type": "APIConnectionError"})
            evidence = {"traceback_boundary": "Backend.score"}
            journal._record({
                "type": "premutation_recovery_authorized", "operation": operation,
                "inputs_sha256": digest, "recoverable": True, "failed_attempt": 1,
                "failed_error_type": "APIConnectionError", "failure_boundary": "teacher_scoring",
                "forward_backward_applied": False, "optimizer_applied": False,
                "training_updates_replayed": False, "source_checkpoint": "tinker://run/weights/state-5",
                "remote_training_runs_created": [], "evidence": evidence,
                "evidence_sha256": hashlib.sha256(canonical(evidence).encode()).hexdigest(),
                "reason": "Traceback and remote request time prove failure preceded mutable work.",
            })
            resumed = Journal(path)
            self.assertTrue(resumed.premutation_recoverable_pending())
            self.assertFalse(resumed.setup_recoverable_pending())
            self.assertEqual(resumed.call(operation, inputs, lambda: {"step": 6})["step"], 6)
            self.assertEqual(Journal(path).completed[operation]["attempt"], 2)

    def test_verified_sft_preoptimizer_failure_resumes_from_checkpoint(self):
        previous = "screen/task/3e-05/2/learn/update/029"
        operation = "screen/task/3e-05/2/learn/update/030"
        source = "tinker://run/weights/state-29"
        inputs = {"step": 30, "from": source}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            journal = Journal(path)
            previous_inputs = {"step": 29, "from": "tinker://run/weights/state-28"}
            journal.call(previous, previous_inputs,
                         lambda: {"checkpoint": {"state_path": source}, "step": 29})
            digest = hashlib.sha256(canonical(inputs).encode()).hexdigest()
            journal._record({"type": "inflight", "operation": operation, "inputs_sha256": digest})
            journal._record({"type": "failed", "operation": operation, "inputs_sha256": digest,
                             "attempt": 1, "attempt_status": "failed", "elapsed_seconds": 1.0,
                             "retryable": False, "error_type": "APIConnectionError"})
            evidence = {"source_checkpoint": "state-29", "old_client": "abandoned"}
            journal._record({
                "type": "premutation_recovery_authorized", "operation": operation,
                "inputs_sha256": digest, "recoverable": True, "failed_attempt": 1,
                "failed_error_type": "APIConnectionError",
                "failure_boundary": "forward_backward_before_optimizer",
                "forward_backward_applied": "unknown", "optimizer_applied": False,
                "remote_gradient_state_abandoned": True, "failed_client_process_exited": True,
                "checkpoint_created": False,
                "training_updates_replayed": False,
                "source_checkpoint": source,
                "remote_training_runs_created": [], "evidence": evidence,
                "evidence_sha256": hashlib.sha256(canonical(evidence).encode()).hexdigest(),
                "reason": "The failed client is abandoned before restoring the preceding optimizer checkpoint.",
            })
            resumed = Journal(path)
            self.assertTrue(resumed.premutation_recoverable_pending())
            self.assertEqual(resumed.call(operation, inputs, lambda: {"step": 30})["step"], 30)
            self.assertEqual(Journal(path).completed[operation]["attempt"], 2)

    def test_sft_premutation_recovery_requires_abandoning_remote_gradient_state(self):
        previous = "reference/task/1e-05/update/001"
        operation = "reference/task/1e-05/update/002"
        source = "tinker://run/weights/state-1"
        inputs = {"step": 2, "from": source}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            journal = Journal(path)
            journal.call(previous, {"step": 1, "from": "tinker://run/weights/origin"},
                         lambda: {"checkpoint": {"state_path": source}, "step": 1})
            digest = hashlib.sha256(canonical(inputs).encode()).hexdigest()
            journal._record({"type": "inflight", "operation": operation, "inputs_sha256": digest})
            journal._record({"type": "failed", "operation": operation, "inputs_sha256": digest,
                             "attempt": 1, "attempt_status": "failed", "elapsed_seconds": 1.0,
                             "retryable": False, "error_type": "APIConnectionError"})
            evidence = {"remote_status": "ambiguous"}
            journal._append({
                "type": "premutation_recovery_authorized", "operation": operation,
                "inputs_sha256": digest, "recoverable": True, "failed_attempt": 1,
                "failed_error_type": "APIConnectionError",
                "failure_boundary": "forward_backward_before_optimizer",
                "forward_backward_applied": "unknown", "optimizer_applied": False,
                "remote_gradient_state_abandoned": False, "failed_client_process_exited": True,
                "checkpoint_created": False,
                "training_updates_replayed": False,
                "source_checkpoint": source,
                "remote_training_runs_created": [], "evidence": evidence,
                "evidence_sha256": hashlib.sha256(canonical(evidence).encode()).hexdigest(),
                "reason": "must reject reuse of ambiguous remote gradient state",
            })
            with self.assertRaises(AmbiguousOperation):
                Journal(path)

    def test_marked_hard_interruption_can_resume_but_prior_duration_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"

            def interrupted():
                raise KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                Journal(path).call(self.operation, {}, interrupted, recoverable=True)
            journal = Journal(path)
            self.assertTrue(journal.recoverable_pending())
            journal.call(self.operation, {}, lambda: 1, recoverable=True)
            self.assertFalse(Journal(path).completed[self.operation]["timing_complete"])

    def test_pending_contract_mismatch_and_nonrecoverable_caller_never_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"

            def interrupted():
                raise eval_cache.SamplingTransportError()

            with self.assertRaises(eval_cache.SamplingTransportError):
                Journal(path).call(self.operation, {"state": "immutable"}, interrupted, recoverable=True)
            before = path.read_bytes()
            journal = Journal(path)
            with self.assertRaisesRegex(RuntimeError, "resume contract changed"):
                journal.call(self.operation, {"state": "different"}, lambda: self.fail("executed"), recoverable=True)
            with self.assertRaises(AmbiguousOperation):
                journal.call(self.operation, {"state": "immutable"}, lambda: self.fail("executed"))
            with self.assertRaises(AmbiguousOperation):
                journal.call("unrelated", {}, lambda: self.fail("executed"))
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_scientific_results_never_become_retryable(self):
        def invalid_measurement():
            raise InvalidMeasurement("numerically invalid")

        for function in (invalid_measurement, lambda: {"score": float("nan")}):
            with self.subTest(function=function), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "journal.jsonl"
                with self.assertRaises((InvalidMeasurement, ValueError)):
                    Journal(path).call(self.operation, {}, function, recoverable=True)
                journal = Journal(path)
                self.assertFalse(journal.recoverable_pending())
                self.assertFalse(journal.rows[-1]["retryable"])
                with self.assertRaises(AmbiguousOperation):
                    journal.call(self.operation, {}, lambda: 1, recoverable=True)

    def test_only_dedicated_sampling_transport_failures_can_resume(self):
        for error in (RuntimeError("nonfinite sampled logprobs"),
                      eval_cache.EvaluationRecoveryError("cache identity mismatch"),
                      ValueError("malformed sample"), ConnectionError("not from durable sampling"),
                      TimeoutError("not from durable sampling")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "journal.jsonl"

                def failure():
                    raise error

                with self.assertRaises(type(error)):
                    Journal(path).call(self.operation, {}, failure, recoverable=True)
                journal = Journal(path)
                self.assertFalse(journal.recoverable_pending())
                self.assertFalse(journal.rows[-1]["retryable"])
                with self.assertRaises(AmbiguousOperation):
                    journal.call(self.operation, {}, lambda: self.fail("retried permanent failure"), recoverable=True)

    def test_parser_rejects_unseen_events_hash_mismatch_and_invalid_authorization(self):
        mutations = [
            {"type": "complete", "inputs_sha256": "different", "result": 1},
            {"type": "failed", "inputs_sha256": "different"},
            {"type": "resume", "inputs_sha256": "different"},
            {"type": "recovery_authorized", "inputs_sha256": "different", "reason": "approved", "recoverable": True},
            {"type": "recovery_authorized", "reason": "", "recoverable": True},
            {"type": "complete", "operation": "unseen", "result": 1},
            {"type": "inflight", "operation": "unseen"},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "journal.jsonl"
                digest = self.legacy_pending(path)
                journal = Journal(path)
                journal._append({"operation": self.operation, "inputs_sha256": digest, **mutation})
                with self.assertRaises(AmbiguousOperation):
                    Journal(path)

    def test_training_origin_and_probe_cannot_be_marked_or_authorized(self):
        for operation in ("reference/test/1e-05/update/015", "m1/origin", "learn/A", "repair/B", "probe/test/evaluate/015"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "journal.jsonl"
                journal = Journal(path)
                with self.assertRaises(AmbiguousOperation):
                    journal.call(operation, {}, lambda: self.fail("executed"), recoverable=True)
                digest = self.legacy_pending(path, operation)
                journal = Journal(path)
                journal._append({"type": "recovery_authorized", "operation": operation,
                                 "inputs_sha256": digest, "recoverable": True, "reason": "not allowed"})
                with self.assertRaises(AmbiguousOperation):
                    Journal(path)

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
