"""Immutable sampling recovery tests; no SDK clients or network calls."""

import asyncio
from contextlib import asynccontextmanager, contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import create_autospec, patch

import backend
import eval_cache
from test_backend import Future, ModelInput, Sampler, make_backend

INPUTS = hashlib.sha256(b"frozen evaluation inputs").hexdigest()
PATH = "tinker://run/sampler_weights/A-step-000015"
_sleep = asyncio.sleep


class StatusError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class Holder:
    def run_coroutine_threadsafe(self, coroutine):
        return NS(result=lambda: asyncio.run(coroutine))


def immutable_sampler(session="session-1"):
    sampler = Sampler()
    sampler.holder = Holder()
    sampler._sampling_session_id = session
    sampler._request_id_counter = 0
    sampler._kintsugi_immutable_model_path = PATH
    return sampler


class Transport:
    def __init__(self):
        self.sent, self.retrieved = [], []
        self.send_errors, self.receive_errors = {}, {}
        self.active = self.peak = 0

    async def send(self, sampler, unit, prompt, params):
        self.active += 1
        self.peak = max(self.peak, self.active)
        identity = unit["sampling_session_id"], unit["seq_id"]
        self.sent.append((identity, list(prompt.tokens), params.seed))
        await _sleep(0)
        self.active -= 1
        errors = self.send_errors.get(prompt.tokens[0], [])
        if errors:
            raise errors.pop(0)
        return ":".join(map(str, identity))

    async def receive(self, sampler, unit):
        self.retrieved.append(unit["request_id"])
        await _sleep(0)
        prompt = unit["inputs"]["prompt_tokens"]
        errors = self.receive_errors.get(prompt[0], [])
        if errors:
            raise errors.pop(0)
        return {"tokens": [prompt[0] + 10, 99], "count": unit["inputs"]["num_samples"]}

    def decode(self, response):
        return NS(sequences=[NS(tokens=response["tokens"], logprobs=[-0.5, -0.5], stop_reason="stop")
                             for _ in range(response["count"])], prompt_cache_hit_tokens=1)


async def no_delay(delay):
    if not 0 <= delay <= 30:
        raise AssertionError("unbounded backoff")
    await _sleep(0)


@contextmanager
def transport_boundary(transport):
    with patch("eval_cache._send", transport.send), patch("eval_cache._receive", transport.receive), \
            patch("eval_cache._decode", transport.decode), patch("eval_cache.asyncio.sleep", no_delay):
        yield


def sample(sampler, count=10, *, seed=31):
    return make_backend().sample(sampler, [[index, 2] for index in range(count)], samples=2,
                                 max_tokens=512, temperature=0, seed=seed)


def records(directory):
    paths = list(Path(directory).glob("*.jsonl"))
    return [json.loads(line) for path in paths for line in path.read_text().splitlines()]


class EvaluationCacheTests(unittest.TestCase):
    def test_midbatch_502_preserves_identity_alignment_and_accounting(self):
        transport, sampler = Transport(), immutable_sampler()
        transport.send_errors[3] = [StatusError(502)]
        transport.receive_errors[4] = [StatusError(502)]
        with tempfile.TemporaryDirectory() as directory, transport_boundary(transport):
            with eval_cache.evaluation_scope(directory, "reference/evaluate/015", INPUTS):
                result = sample(sampler)
            self.assertEqual(transport.peak, 8)
            identities = [identity for identity, prompt, seed in transport.sent if prompt[0] == 3]
            self.assertEqual(identities, [("session-1", 3)] * 2)
            self.assertEqual(transport.retrieved.count("session-1:4"), 2)
            self.assertEqual([group[0]["tokens"][0] for group in result["groups"]], list(range(10, 20)))
            self.assertEqual(result["accounting"], backend.accounting(prefill_tokens=10, cached_tokens=30, sample_tokens=40))
            self.assertEqual(sampler.calls, [])  # The SDK public retry wrapper is never called.
            rows = records(directory)
            self.assertTrue(any(row["event"] == "failure" and row["billing_uncertain"] for row in rows))
            before = len(transport.sent), len(transport.retrieved), len(rows)
            with eval_cache.evaluation_scope(directory, "reference/evaluate/015", INPUTS):
                replay = sample(immutable_sampler("new-unused-session"))
            self.assertEqual(replay, result)
            self.assertEqual(before, (len(transport.sent), len(transport.retrieved), len(records(directory))))

    def test_resume_known_future_and_remaining_only_keeps_successful_window_peers(self):
        transport = Transport()
        complete = eval_cache._complete

        async def interrupted(scope, sampler, unit, prompt, params, normalize):
            if prompt.tokens[0] == 3:
                scope.append({"event": "attempt", "unit": unit["unit"], "phase": "submit",
                              "number": 1, "billing_uncertain": False})
                request_id = await transport.send(sampler, unit, prompt, params)
                scope.append({"event": "ack", "unit": unit["unit"], "request_id": request_id})
                raise RuntimeError("simulated process interruption after acknowledgement")
            return await complete(scope, sampler, unit, prompt, params, normalize)

        with tempfile.TemporaryDirectory() as directory, transport_boundary(transport):
            with patch("eval_cache._complete", interrupted), self.assertRaisesRegex(RuntimeError, "interruption"):
                with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                    sample(immutable_sampler())
            self.assertEqual(len(transport.sent), 8)
            self.assertEqual(sum(row["event"] == "done" for row in records(directory)), 7)
            with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                result = sample(immutable_sampler("resumed-session"))
            self.assertEqual(len(result["groups"]), 10)
            self.assertEqual(len(transport.sent), 10)
            self.assertEqual(transport.retrieved.count("session-1:3"), 1)
            self.assertEqual([seed for _, _, seed in transport.sent[-2:]], [39, 40])
            self.assertEqual([identity[0] for identity, _, _ in transport.sent[-2:]], ["resumed-session"] * 2)

    def test_resume_unacknowledged_submission_reuses_persisted_session_sequence(self):
        transport = Transport()
        transport.send_errors[0] = [asyncio.CancelledError()]
        with tempfile.TemporaryDirectory() as directory, transport_boundary(transport):
            with self.assertRaises(asyncio.CancelledError):
                with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                    sample(immutable_sampler(), 1)
            with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                sample(immutable_sampler("different-session"), 1)
            self.assertEqual([row[0] for row in transport.sent], [("session-1", 0)] * 2)
            attempts = [row for row in records(directory) if row["event"] == "attempt" and row["phase"] == "submit"]
            self.assertEqual([row["billing_uncertain"] for row in attempts], [False, True])

    def test_response_persisted_before_crash_is_decoded_without_another_remote_call(self):
        transport = Transport()
        append = eval_cache._EvaluationScope.append

        def interrupt_after_response(scope, record):
            append(scope, record)
            if record["event"] == "received":
                raise RuntimeError("simulated interruption after response fsync")

        with tempfile.TemporaryDirectory() as directory, transport_boundary(transport):
            with patch("eval_cache._EvaluationScope.append", interrupt_after_response), self.assertRaisesRegex(RuntimeError, "fsync"):
                with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                    sample(immutable_sampler(), 1)
            self.assertTrue(any(row["event"] == "received" for row in records(directory)))
            self.assertFalse(any(row["event"] == "done" for row in records(directory)))
            with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                result = sample(immutable_sampler("resumed-session"), 1)
            self.assertEqual((len(transport.sent), len(transport.retrieved)), (1, 1))
            self.assertEqual(result["accounting"], backend.accounting(prefill_tokens=1, cached_tokens=3, sample_tokens=4))

    def test_calls_are_distinct_realizations_but_replay_binds_full_inputs(self):
        transport, sampler = Transport(), immutable_sampler()
        with tempfile.TemporaryDirectory() as directory, transport_boundary(transport):
            with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                first, second = sample(sampler, 1), sample(sampler, 1)
            self.assertEqual(len(transport.sent), 2)
            self.assertNotEqual(transport.sent[0][0], transport.sent[1][0])
            with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                self.assertEqual(sample(sampler, 1), first)
                self.assertEqual(sample(sampler, 1), second)
            for changed in ("seed", "path", "count", "operation_digest"):
                sampler2 = immutable_sampler()
                if changed == "path":
                    sampler2._kintsugi_immutable_model_path = PATH + "-changed"
                with self.subTest(changed=changed), self.assertRaises(eval_cache.EvaluationRecoveryError):
                    with eval_cache.evaluation_scope(directory, "evaluate", "a" * 64 if changed == "operation_digest" else INPUTS):
                        sample(sampler2, 2 if changed == "count" else 1, seed=32 if changed == "seed" else 31)
            self.assertEqual(len(transport.sent), 2)

    def test_retry_bounds_then_manual_resume_and_410_or_permanent_error_never_resubmits(self):
        for phase, error in (("send", 400), ("send", 401), ("receive", 404), ("receive", 410)):
            transport = Transport()
            getattr(transport, phase + "_errors")[0] = [StatusError(error)]
            with self.subTest(phase=phase, error=error), tempfile.TemporaryDirectory() as directory, transport_boundary(transport):
                for _ in range(2):
                    with self.assertRaises(eval_cache.EvaluationRecoveryError) as raised:
                        with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                            sample(immutable_sampler(), 1)
                    self.assertNotIsInstance(raised.exception, eval_cache.SamplingTransportError)
                self.assertEqual(len(transport.sent), 1)
                self.assertEqual(len(transport.retrieved), int(phase == "receive"))
        transport = Transport()
        transport.send_errors[0] = [StatusError(502)] * 3
        with tempfile.TemporaryDirectory() as directory, transport_boundary(transport), patch("eval_cache.MAX_SUBMIT_ATTEMPTS", 3):
            with self.assertRaises(eval_cache.SamplingTransportError):
                with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                    sample(immutable_sampler(), 1)
            self.assertEqual(len(transport.sent), 3)
            with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                recovered = sample(immutable_sampler("resume-session"), 1)
            self.assertEqual(len(recovered["groups"]), 1)
            self.assertEqual(len(transport.sent), 4)
            self.assertEqual(len({row[0] for row in transport.sent}), 1)
            attempts = [row["number"] for row in records(directory)
                        if row["event"] == "attempt" and row["phase"] == "submit"]
            self.assertEqual(attempts, [1, 2, 3, 4])

    def test_connection_408_429_and_5xx_retry_same_request(self):
        for error in (ConnectionError("lost"), TimeoutError("lost"), StatusError(408), StatusError(429), StatusError(503)):
            transport = Transport()
            transport.send_errors[0] = [error]
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory, transport_boundary(transport):
                with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                    sample(immutable_sampler(), 1)
                self.assertEqual([row[0] for row in transport.sent], [("session-1", 0)] * 2)

    def test_corruption_unknown_records_and_missing_scope_halt_before_submission(self):
        transport = Transport()
        with transport_boundary(transport), self.assertRaisesRegex(eval_cache.EvaluationRecoveryError, "evaluation_scope"):
            sample(immutable_sampler(), 1)
        for damage in ("truncated", "checksum", "unknown"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory, transport_boundary(transport):
                with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                    sample(immutable_sampler(), 1)
                before = len(transport.sent)
                path = next(Path(directory).glob("*.jsonl"))
                if damage == "truncated":
                    path.write_text(path.read_text()[:-1])
                elif damage == "checksum":
                    path.write_text(path.read_text().replace('"sample_tokens":4', '"sample_tokens":9'))
                else:
                    rows = records(directory)
                    extra = {"event": "unknown", "unit": "0:0", "index": len(rows), "previous": rows[-1]["sha256"]}
                    with path.open("a") as stream:
                        stream.write(json.dumps({**extra, "sha256": eval_cache._digest(extra)}) + "\n")
                with self.assertRaises(eval_cache.EvaluationRecoveryError):
                    with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                        sample(immutable_sampler(), 1)
                self.assertEqual(len(transport.sent), before)

    def test_invalid_returned_output_is_terminal_not_transport_retry(self):
        transport = Transport()
        with tempfile.TemporaryDirectory() as directory, transport_boundary(transport):
            with patch("eval_cache._decode", return_value=NS(sequences=[])), self.assertRaisesRegex(RuntimeError, "sequence count"):
                with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                    sample(immutable_sampler(), 1)
            with self.assertRaises(eval_cache.EvaluationRecoveryError) as raised:
                with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                    sample(immutable_sampler(), 1)
            self.assertNotIsInstance(raised.exception, eval_cache.SamplingTransportError)
            self.assertEqual((len(transport.sent), len(transport.retrieved)), (1, 1))

    def test_real_pinned_rest_signatures_and_protobuf_decode(self):
        try:
            version = importlib.metadata.version("tinker")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("run the SDK boundary test with .venv/bin/python")
        self.assertEqual(version, eval_cache.SDK_VERSION)
        from tinker import types
        from tinker.lib.internal_client_holder import InternalClientHolder
        from tinker.resources.sampling import AsyncSamplingResource
        from tinker.resources.futures import AsyncFuturesResource
        from tinker.proto import tinker_public_pb2 as pb

        # Autospec every invoked transport/holder callable against the installed
        # SDK. No ServiceClient, SamplingClient or HTTP client is constructed.
        sampling = create_autospec(AsyncSamplingResource, instance=True)
        futures = create_autospec(AsyncFuturesResource, instance=True)
        sampling.asample.return_value = types.UntypedAPIFuture(request_id="known-future")
        import struct
        payload = pb.SampleResponse(prompt_cache_hit_tokens=1)
        sequence = payload.sequences.add(stop_reason=pb.STOP_REASON_STOP)
        sequence.tokens = struct.pack("<ii", 4, 99)
        sequence.logprobs = struct.pack("<ff", -0.5, -0.5)
        futures.retrieve.return_value = NS(headers={"content-type": "application/x-protobuf"},
                                           http_response=NS(content=payload.SerializeToString()))
        client = NS(sampling=sampling, futures=NS(with_raw_response=futures))
        holder = create_autospec(InternalClientHolder, instance=True)

        @contextmanager
        def aclient(pool):
            yield client

        @asynccontextmanager
        async def rate_limit(estimated):
            yield

        holder.aclient.side_effect = aclient
        holder.estimate_bytes_count_in_model_input.return_value = 16
        holder.sample_dispatch_rate_limit.side_effect = rate_limit
        holder.run_coroutine_threadsafe.side_effect = lambda coroutine: NS(result=lambda: asyncio.run(coroutine))
        sampler = immutable_sampler()
        sampler.holder = holder
        api = make_backend()
        api.sdk, api.types = NS(ModelInput=types.ModelInput), types
        with tempfile.TemporaryDirectory() as directory:
            with eval_cache.evaluation_scope(directory, "evaluate", INPUTS):
                result = api.sample(sampler, [[1, 2]], samples=1, max_tokens=512, temperature=0, seed=31)
        sent = sampling.asample.call_args.kwargs
        request = sent["request"]
        self.assertEqual((request.sampling_session_id, request.seq_id, request.sampling_params.seed), ("session-1", 0, 31))
        self.assertEqual((request.prompt_logprobs, request.topk_prompt_logprobs), (False, 0))
        self.assertEqual((sent["max_retries"], sent["timeout"]), (0, 30))
        retrieved = futures.retrieve.call_args.kwargs
        self.assertEqual(retrieved["request"].request_id, "known-future")
        self.assertFalse(retrieved["request"].allow_metadata_only)
        self.assertEqual((retrieved["max_retries"], retrieved["timeout"]), (0, 30))
        self.assertEqual(result["groups"][0][0]["tokens"], [4, 99])
        self.assertEqual(result["accounting"], backend.accounting(prefill_tokens=1, cached_tokens=1, sample_tokens=2))


if __name__ == "__main__":
    unittest.main()
