"""Offline tests: no credentials, live clients or network calls.

The installed-SDK contract test imports TrainingClient only to autospec its
method signatures; no actual SDK client is constructed or called.
"""

import asyncio
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import struct
import tarfile
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import create_autospec, patch

import backend
import eval_cache


class Future:
    def __init__(self, result=None, error=None):
        self.value, self.error = result, error

    def result(self):
        if self.error:
            raise self.error
        return self.value


class ModelInput:
    @classmethod
    def from_ints(cls, tokens):
        return NS(tokens=list(tokens), length=len(tokens))


class Tokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return 99 if token == "<|im_end|>" else -1

    def encode(self, text, add_special_tokens=False):
        self.text, self.special = text, add_special_tokens
        return [1, 2, 3]

    def decode(self, tokens, skip_special_tokens=False):
        return ",".join(map(str, tokens))


class Sampler:
    def __init__(self):
        self.calls, self.scored = [], []
        self.sequences = None
        self.score_values = None

    def sample(self, prompt, samples, params):
        self.calls.append((prompt, samples, params))
        sequences = self.sequences if self.sequences is not None else [
            NS(tokens=[4, 99], logprobs=[-0.5, -0.5], stop_reason="stop") for _ in range(samples)]
        return Future(NS(sequences=sequences, prompt_cache_hit_tokens=1))

    def compute_logprobs(self, prompt):
        self.scored.append(prompt)
        return Future(self.score_values if self.score_values is not None else
                      [None] + [-0.25] * (prompt.length - 1))


class Client:
    def __init__(self):
        self.calls, self.optimizers, self.saves = [], [], []
        self.sampler = Sampler()
        self.refreshes = 0
        self.output_value = -2.0
        self.output_error = None
        self.optim_value = 1.0

    def _output(self, datums, loss_fn):
        self.calls.append((datums, loss_fn))
        return Future(NS(loss_fn_outputs=[{"logprobs": NS(data=[self.output_value] * d.model_input.length)}
                                          for d in datums]), error=self.output_error)

    def forward(self, datums, loss_fn):
        return self._output(datums, loss_fn)

    def forward_backward(self, datums, loss_fn):
        return self._output(datums, loss_fn)

    def optim_step(self, params):
        self.optimizers.append(params)
        return Future(NS(metrics={"grad_norm": self.optim_value}))

    def save_weights_and_get_sampling_client(self, **kwargs):
        self.refreshes += 1
        return self.sampler

    def save_state(self, name, ttl_seconds=None, overwrite=False):
        self.saves.append(("state", name, {"ttl_seconds": ttl_seconds, "overwrite": overwrite}))
        return Future(NS(path=f"tinker://run/weights/{name}"))

    def save_weights_for_sampler(self, name, ttl_seconds=None):
        self.saves.append(("sampler", name, {"ttl_seconds": ttl_seconds}))
        return Future(NS(path=f"tinker://run/sampler_weights/{name}"))


def make_backend(service=None):
    return backend.Backend(service or NS(), NS(ModelInput=ModelInput, Datum=NS),
                           NS(AdamParams=NS, SamplingParams=NS), Tokenizer(),
                           seed=17, checkpoint_ttl=86400)


def adapter_bytes(*, bad_name=False, bad_shape=False):
    entries, data = {}, bytearray()
    for index, name in enumerate(backend._expected_modules()):
        for side in ("A", "B"):
            key = f"{name}.lora_{side}.weight"
            if bad_name and index == 0 and side == "A":
                key = "base_model.model.lm_head.lora_A.weight"
            shape = [32, 2] if side == "A" else [2, 32]
            if bad_shape and index == 0 and side == "A":
                shape = [16, 4]
            start = len(data)
            data.extend(b"\0" * (math.prod(shape) * 4))
            entries[key] = {"dtype": "F32", "shape": shape, "data_offsets": [start, len(data)]}
    header = json.dumps(entries, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + data


def write_adapter(directory, *, alpha=32, **kwargs):
    directory = Path(directory)
    (directory / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": alpha}))
    path = directory / "adapter_model.safetensors"
    path.write_bytes(adapter_bytes(**kwargs))
    return path


class BackendTests(unittest.TestCase):
    def test_completion_alignment_and_normalization(self):
        datum = make_backend().datum({"prompt_tokens": [1, 2, 3], "completion_tokens": [4, 99]})
        self.assertEqual(datum.model_input.tokens, [1, 2, 3, 4])
        self.assertEqual(datum.loss_fn_inputs["target_tokens"], [2, 3, 4, 99])
        self.assertEqual(datum.loss_fn_inputs["weights"], [0.0, 0.0, 0.5, 0.5])
        datum = make_backend().datum({"tokens": [1, 2, 3]})
        self.assertEqual(datum.loss_fn_inputs["weights"], [0.5, 0.5])

    def test_invalid_rows_fail_before_paid_call(self):
        for row in ({"tokens": [1]}, {"tokens": [1, -2]},
                    {"prompt_tokens": [], "completion_tokens": [2]},
                    {"prompt_tokens": [1], "completion_tokens": []},
                    {"tokens": [1, 2], "prompt_tokens": [1]}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                make_backend().datum(row)

    def test_renderer_nonthinking_without_truncation(self):
        api = make_backend()
        api.render_prompt("hello" * 10000)
        self.assertIn("hello" * 10000, api.tokenizer.text)
        self.assertTrue(api.tokenizer.text.endswith("assistant\n<think>\n\n</think>\n\n"))
        self.assertFalse(api.tokenizer.special)

    def test_oriented_nll_is_target_token_weighted(self):
        output = NS(loss_fn_outputs=[{"logprobs": NS(data=[-100, -1])},
                                     {"logprobs": NS(data=[-3, -3, -3])}])
        self.assertEqual(backend._nll(output, [[0, 1], [1, 1, 1]]), (2.5, 4))
        result = make_backend().evaluate_nll(Client(), [
            {"prompt_tokens": [1, 2, 3], "completion_tokens": [4, 5]}])
        self.assertEqual((result["nll"], result["q"]), (2.0, -2.0))
        self.assertEqual(result["accounting"]["forward_tokens"], 4)
        self.assertEqual(result["accounting"]["gradient_target_tokens"], 0)

    def test_training_counts_and_warmup(self):
        client, api = Client(), make_backend()
        row = {"prompt_tokens": [1, 2, 3], "completion_tokens": [4, 5]}
        result = api.train_step(client, [row], learning_rate=1e-4, step=1)
        self.assertAlmostEqual(result["learning_rate"], 1e-5)
        self.assertEqual(result["accounting"]["gradient_target_tokens"], 2)
        self.assertEqual(result["accounting"]["train_tokens"], 4)
        result = api.train_step(client, [row], learning_rate=1e-4, step=1, warmup_steps=0)
        self.assertEqual(result["learning_rate"], 1e-4)
        self.assertEqual(client.calls[0][1], "cross_entropy")

    def test_returned_nonfinite_is_invalid_but_transport_is_ambiguous(self):
        client, api = Client(), make_backend()
        client.output_value = math.nan
        result = api.train_step(client, [{"tokens": [1, 2]}], learning_rate=1e-4, step=1)
        self.assertFalse(result["valid"])
        self.assertFalse(result["optimizer_applied"])
        self.assertEqual(client.optimizers, [])
        self.assertEqual(result["accounting"]["train_tokens"], 1)
        client.output_error = TimeoutError("ambiguous")
        with self.assertRaises(TimeoutError):
            api.train_step(client, [{"tokens": [1, 2]}], learning_rate=1e-4, step=1)
        self.assertEqual(len(client.calls), 2)  # Exactly one invocation per request.

    def test_returned_bad_optimizer_is_not_replayed(self):
        client = Client()
        client.optim_value = math.inf
        result = make_backend().train_step(client, [{"tokens": [1, 2]}], learning_rate=1e-4, step=1)
        self.assertFalse(result["valid"])
        self.assertTrue(result["optimizer_applied"])
        self.assertEqual(len(client.optimizers), 1)

    def test_fresh_branch_resume_and_origin_scope(self):
        calls = []
        service = NS(create_lora_training_client=lambda **kw: calls.append(("origin", kw)),
                     create_training_client_from_state=lambda path: calls.append(("fresh", path)),
                     create_training_client_from_state_with_optimizer=lambda path: calls.append(("resume", path)))
        api = make_backend(service)
        api.origin()
        api.branch("tinker://00000000-0000-0000-0000-000000000000:train:0/weights/anchor")
        api.branch("tinker://run/weights/step-1", resume=True)
        self.assertEqual([call[0] for call in calls], ["origin", "fresh", "resume"])
        self.assertEqual(calls[0][1], {"base_model": backend.MODEL, "rank": 32, "seed": 17,
                                      "train_attn": True, "train_mlp": True, "train_unembed": False})
        with self.assertRaises(ValueError):
            api.branch("tinker://run/sampler_weights/wrong")

    def test_save_both_kinds_unique_step_name_and_ttl(self):
        client = Client()
        result = make_backend().save(client, "m1-T1-reference-lr1", step=5)
        self.assertEqual(result["state_path"], "tinker://run/weights/m1-T1-reference-lr1-step-000005")
        self.assertIn("/sampler_weights/", result["sampler_path"])
        self.assertEqual([row[0] for row in client.saves], ["state", "sampler"])
        self.assertEqual(client.saves[0][2], {"ttl_seconds": 86400, "overwrite": False})
        self.assertEqual(client.saves[1][2], {"ttl_seconds": 86400})

    def test_resume_save_rotates_two_short_lived_slots_then_durable_final(self):
        client, api = Client(), make_backend()
        api.checkpoint_ttl = 90 * 86400
        checkpoints = [api.save(client, "branch", step=step, resume=True) for step in (1, 2, 3)]
        self.assertEqual([row["resume_slot"] for row in checkpoints], [1, 0, 1])
        self.assertEqual(checkpoints[0]["state_path"], checkpoints[2]["state_path"])
        state_saves = [row for row in client.saves if row[0] == "state"]
        sampler_saves = [row for row in client.saves if row[0] == "sampler"]
        self.assertEqual([row[2]["overwrite"] for row in state_saves], [False, False, True])
        self.assertTrue(all("overwrite" not in row[2] for row in sampler_saves))
        self.assertEqual([row[1] for row in sampler_saves],
                         ["branch-step-000001", "branch-step-000002", "branch-step-000003"])
        self.assertEqual(len({row["sampler_path"] for row in checkpoints}), 3)
        self.assertTrue(all(row[2]["ttl_seconds"] == 2 * 86400 for row in client.saves))
        final = api.save(client, "branch-A", step=3)
        self.assertIsNone(final["resume_slot"])
        self.assertEqual(final["ttl_seconds"], 90 * 86400)
        self.assertNotIn("overwrite", client.saves[-1][2])

    def test_checkpoint_calls_bind_to_installed_pinned_sdk_signatures(self):
        try:
            version = importlib.metadata.version("tinker")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("pinned Tinker SDK is required; run this contract test with .venv/bin/python")
        self.assertEqual(version, backend.SDK_VERSION)
        from tinker import TrainingClient

        # Autospec enforces the real SDK signatures, not a permissive **kwargs
        # double. Returning only local futures keeps this completely offline.
        client = create_autospec(TrainingClient, instance=True)
        client.save_state.side_effect = lambda name, ttl_seconds=None, overwrite=False: Future(
            NS(path=f"tinker://run/weights/{name}"))
        client.save_weights_for_sampler.side_effect = lambda name, ttl_seconds=None: Future(
            NS(path=f"tinker://run/sampler_weights/{name}"))
        with self.assertRaises(TypeError):
            client.save_weights_for_sampler("not-submitted", overwrite=False)
        api = make_backend()
        api.checkpoint_ttl = 90 * 86400
        resumes = [api.save(client, "branch", step=step, resume=True) for step in (1, 2, 3, 4)]
        durable_a = api.save(client, "branch-A", step=4)
        durable_b = api.save(client, "branch-B", step=7)

        states = client.save_state.call_args_list
        samplers = client.save_weights_for_sampler.call_args_list
        self.assertEqual([call.args[0] for call in states[:4]],
                         ["branch-resume-1", "branch-resume-0", "branch-resume-1", "branch-resume-0"])
        self.assertEqual([call.kwargs["overwrite"] for call in states[:4]], [False, False, True, True])
        self.assertEqual([call.args[0] for call in samplers],
                         ["branch-step-000001", "branch-step-000002", "branch-step-000003", "branch-step-000004",
                          "branch-A-step-000004", "branch-B-step-000007"])
        self.assertTrue(all(call.kwargs == {"ttl_seconds": 2 * 86400} for call in samplers[:4]))
        self.assertTrue(all(call.kwargs == {"ttl_seconds": 90 * 86400} for call in samplers[4:]))
        self.assertTrue(all("overwrite" not in call.kwargs for call in samplers))
        self.assertEqual(len({row["sampler_path"] for row in resumes}), 4)
        self.assertEqual(resumes[0]["state_path"], resumes[2]["state_path"])
        for saved, label in ((durable_a, "branch-A-step-000004"), (durable_b, "branch-B-step-000007")):
            self.assertEqual(saved["name"], label)
            self.assertEqual(saved["sampler_name"], label)
            self.assertEqual(saved["state_path"], f"tinker://run/weights/{label}")
            self.assertEqual(saved["sampler_path"], f"tinker://run/sampler_weights/{label}")
            self.assertIsNone(saved["resume_slot"])

    def test_sampling_parameters_alignment_and_accounting(self):
        sampler = Sampler()
        result = make_backend().sample(sampler, [[1, 2, 3], [1, 2]], samples=2,
                                      max_tokens=9, temperature=1, seed=31)
        params = sampler.calls[0][2]
        self.assertEqual((params.top_p, params.top_k, params.stop, params.seed), (1.0, -1, [99], 31))
        self.assertEqual(sampler.calls[1][2].seed, 32)
        self.assertEqual(result["groups"][0][0]["text"], "4")
        self.assertEqual(result["accounting"], backend.accounting(prefill_tokens=3, cached_tokens=7, sample_tokens=8))

    def test_immutable_sampler_is_tagged_and_requires_journal_scope(self):
        sampler, calls = Sampler(), []
        api = make_backend(NS(create_sampling_client=lambda **kwargs: calls.append(kwargs) or sampler))
        path = "tinker://run/sampler_weights/A-step-000015"
        self.assertIs(api.sampler(path), sampler)
        self.assertEqual(sampler._kintsugi_immutable_model_path, path)
        self.assertEqual(calls[0]["model_path"], path)
        with self.assertRaisesRegex(eval_cache.EvaluationRecoveryError, "evaluation_scope"):
            api.sample(sampler, [[1, 2]], samples=1, max_tokens=3, temperature=0, seed=1)
        self.assertEqual(sampler.calls, [])

    def test_evaluation_scope_never_retries_mutable_sampling_or_training(self):
        class FailingSampler(Sampler):
            def sample(self, prompt, samples, params):
                self.calls.append((prompt, samples, params))
                return Future(error=TimeoutError("ambiguous mutable sample"))

        sampler, client, api = FailingSampler(), Client(), make_backend()
        client.output_error = TimeoutError("ambiguous optimizer input")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TimeoutError):
                with eval_cache.evaluation_scope(directory, "not-recoverable-sampling", "a" * 64):
                    api.sample(sampler, [[1, 2]], samples=1, max_tokens=3, temperature=1, seed=1)
            with self.assertRaises(TimeoutError):
                with eval_cache.evaluation_scope(directory, "not-recoverable-training", "a" * 64):
                    api.train_step(client, [{"tokens": [1, 2]}], learning_rate=1e-4, step=1)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(len(sampler.calls), 1)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.optimizers, [])

    def test_sampling_rejects_bad_logprobs_cap_and_eos(self):
        sampler = Sampler()
        for tokens, values in (([4], None), ([4, 99], [-1]), ([4], [math.nan]),
                               ([99, 4], [-1, -1]), ([4, 5, 6], [-1, -1, -1])):
            sampler.sequences = [NS(tokens=tokens, logprobs=values, stop_reason="length")]
            with self.subTest(tokens=tokens, values=values), self.assertRaises((ValueError, RuntimeError)):
                make_backend().sample(sampler, [[1]], samples=1, max_tokens=2, temperature=1, seed=1)

    def test_teacher_scoring_alignment_and_estimated_usage(self):
        teacher = Sampler()
        result = make_backend().score(teacher, [[1, 2]], [[{"tokens": [3, 99]}]])
        self.assertEqual(result["logprobs"], [[[-0.25, -0.25]]])
        self.assertEqual(result["accounting"]["scoring_prefill_tokens"], 4)
        self.assertEqual(result["accounting"]["scoring_discarded_sample_tokens_estimate"], 1)
        for values in ([None, -1, -1], [None, -1, None, -1], [None, -1, math.inf, -1]):
            teacher.score_values = values
            with self.assertRaises(RuntimeError):
                make_backend().score(teacher, [[1, 2]], [[{"tokens": [3, 99]}]])

    def test_repair_onpolicy_advantages_refresh_and_counts(self):
        client, teacher, api = Client(), Sampler(), make_backend()
        result = api.repair_step(client, teacher, [[1, 2, 3]] * 64, step=1, seed=23)
        datums, loss_fn = client.calls[0]
        self.assertEqual(loss_fn, "importance_sampling")
        self.assertEqual(len(datums), 256)
        self.assertEqual(datums[0].loss_fn_inputs["advantages"], [0.0, 0.0, 0.25, 0.25])
        self.assertEqual(datums[0].loss_fn_inputs["logprobs"], [0.0, 0.0, -0.5, -0.5])
        self.assertEqual(result["accounting"]["gradient_target_tokens"], 512)
        self.assertEqual(result["accounting"]["train_tokens"], 1024)
        self.assertEqual(result["accounting"]["scoring_prefill_tokens"], 1280)
        self.assertAlmostEqual(result["learning_rate"], 1e-5)
        api.repair_step(client, teacher, [[1, 2, 3]] * 64, step=2, seed=24)
        self.assertEqual(client.refreshes, 2)

    def test_historical_projects_rejected_before_credentials_or_sdk(self):
        denied = hashlib.sha256(b"historical-test-project").hexdigest()
        with patch("backend.subprocess.run") as command, patch("backend.HISTORICAL_PROJECT_HASHES", {denied}):
            for project_id in ["", "historical-test-project", " Historical-Test-Project "]:
                with self.assertRaises(ValueError):
                    backend.Backend.connect(project_id, "keychain", seed=1, checkpoint_ttl=10)
            command.assert_not_called()

    def test_sampling_and_scoring_submit_batches_before_consuming(self):
        sampler, api = Sampler(), make_backend()
        events = []

        class ObservedFuture(Future):
            def result(self):
                events.append("result")
                return super().result()

        original_sample, original_score = sampler.sample, sampler.compute_logprobs

        def sample(*args):
            events.append("submit")
            return ObservedFuture(original_sample(*args).value)

        def score(*args):
            events.append("submit")
            return ObservedFuture(original_score(*args).value)

        sampler.sample, sampler.compute_logprobs = sample, score
        api.sample(sampler, [[1, 2], [3, 4]], samples=1, max_tokens=3, temperature=1, seed=1)
        self.assertEqual(events, ["submit", "submit", "result", "result"])
        events.clear()
        api.score(sampler, [[1, 2]], [[{"tokens": [3]}, {"tokens": [4]}]])
        self.assertEqual(events, ["submit", "submit", "result", "result"])

    def test_one_attempt_guard_never_retries(self):
        calls = []

        async def paid():
            calls.append(1)
            raise TimeoutError("ambiguous")

        with self.assertRaises(TimeoutError):
            asyncio.run(backend._one_attempt(paid))
        self.assertEqual(calls, [1])


class AdapterTests(unittest.TestCase):
    def test_inspection_rejects_alpha_missing_names_and_bad_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_adapter(directory)
            info = backend.inspect_adapter(path)
            self.assertEqual((info["rank"], info["alpha"], len(info["layout"])), (32, 32, 248))
            for options in ({"alpha": 16}, {"bad_name": True}, {"bad_shape": True}):
                write_adapter(directory, **options)
                with self.subTest(options=options), self.assertRaises(RuntimeError):
                    backend.inspect_adapter(path)

    def test_inspection_rejects_layout_change_and_truncated_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_adapter(directory)
            with self.assertRaises(RuntimeError):
                backend.inspect_adapter(path, expected_layout=[])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(RuntimeError):
                backend.inspect_adapter(path)

    def test_safe_archive_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_adapter(directory)
            archive = directory / "sampler.tar"
            with tarfile.open(archive, "w") as bundle:
                for name in ("adapter_config.json", "adapter_model.safetensors"):
                    bundle.add(directory / name, arcname=name)
            destination = directory / "extracted"
            self.assertEqual(backend.safe_extract_sampler(archive, destination), destination)
            self.assertTrue((destination / "adapter_model.safetensors").is_file())
            with self.assertRaises(FileExistsError):
                backend.safe_extract_sampler(archive, destination)

    def test_archive_rejects_traversal_links_duplicate_and_extra_files(self):
        for name, kind in (("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                           ("unexpected", tarfile.REGTYPE), ("adapter_model.safetensors", tarfile.SYMTYPE),
                           ("adapter_config.json", tarfile.REGTYPE)):
            with self.subTest(name=name, kind=kind), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                archive = directory / "sampler.tar"
                with tarfile.open(archive, "w") as bundle:
                    config = b'{"r":32,"lora_alpha":32}'
                    for member_name, data in (("adapter_config.json", config),
                                               ("adapter_model.safetensors", adapter_bytes())):
                        entry = tarfile.TarInfo(member_name)
                        entry.size = len(data)
                        bundle.addfile(entry, io.BytesIO(data))
                    extra = tarfile.TarInfo(name)
                    extra.type, extra.linkname = kind, "/outside"
                    bundle.addfile(extra, io.BytesIO())
                with self.assertRaises(RuntimeError):
                    backend.safe_extract_sampler(archive, directory / "out")
                self.assertFalse((directory / "out").exists())

    def test_scaled_geometry_and_cosine(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional numpy not installed")
        a = np.eye(2)
        b = np.diag([3.0, 4.0])
        result = backend.lora_geometry(a, b, alpha=4, previous=(a, -b, 4))
        self.assertAlmostEqual(result["frobenius"], 10.0)
        self.assertAlmostEqual(result["stable_rank"], 25 / 16)
        self.assertAlmostEqual(result["effective_rank"], math.exp(-3/7 * math.log(3/7) - 4/7 * math.log(4/7)))
        self.assertAlmostEqual(result["cosine_to_previous"], -1.0)
        zero = backend.lora_geometry(a, np.zeros((2, 2)), alpha=4, previous=(a, b, 4))
        self.assertEqual(zero, {"frobenius": 0.0, "effective_rank": 0.0,
                                "stable_rank": 0.0, "cosine_to_previous": None})


if __name__ == "__main__":
    unittest.main()
