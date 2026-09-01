"""Durable, identity-preserving recovery for immutable Tinker sampling only.

The journal supplies one evaluation_scope per operation. Each sampling call and
prompt retains its original session/sequence ID, then its acknowledged future
ID. Completed responses and their accounting are replayed, never resampled.
Transport retries do not establish exactly-once billing: their uncertainty is
recorded separately and does not change the returned response token accounting.
Importing this module is offline; the narrow SDK boundary is loaded lazily.
"""

import asyncio
import base64
from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

SDK_VERSION = "0.25.0"
WINDOW = 8
MAX_SUBMIT_ATTEMPTS = 8
MAX_RETRIEVE_ATTEMPTS = 60
MAX_TRANSIENT_FAILURES = 8
MAX_SECONDS = 300
REQUEST_TIMEOUT = 30
_scope = ContextVar("immutable_sampling_evaluation", default=None)


class EvaluationRecoveryError(RuntimeError):
    """Recovery cannot safely continue with the recorded request identity."""


class SamplingTransportError(EvaluationRecoveryError):
    """Bounded transport/pending-future recovery stopped, not a new request."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate cache field")
        result[key] = value
    return result


def current_scope():
    return _scope.get()


@contextmanager
def evaluation_scope(directory, operation, inputs_sha256):
    """Bind journal identity; replay starts sampling-call ordinals at zero.

    Files are append-only, checksummed and fsynced. An unfinished/truncated or
    unknown record fails closed. A nonblocking file lock rejects two processes
    attempting the same operation. No file is created for nonsampling work.
    """
    scope = _EvaluationScope(directory, operation, inputs_sha256)
    token = _scope.set(scope)
    try:
        if scope.path.exists():
            scope.open()
        yield scope
        if scope.call_index != len(scope.calls):
            raise EvaluationRecoveryError("evaluation omitted previously recorded sampling calls")
    finally:
        _scope.reset(token)
        scope.close()


class _EvaluationScope:
    def __init__(self, directory, operation, inputs_sha256):
        if (not isinstance(operation, str) or not operation
                or not isinstance(inputs_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", inputs_sha256)):
            raise ValueError("evaluation requires an operation and SHA256 input digest")
        self.operation, self.inputs_sha256 = operation, inputs_sha256
        self.path = Path(directory) / (hashlib.sha256(operation.encode()).hexdigest() + ".jsonl")
        self.file = None
        self.calls, self.units = {}, {}
        self.call_index, self.record_index = 0, 0
        self.previous = "0" * 64

    def open(self):
        if self.file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        self.file = os.fdopen(descriptor, "a+", encoding="utf-8")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.close()
            raise EvaluationRecoveryError("evaluation is already active in another process") from error
        try:
            self.file.seek(0)
            for line in self.file:
                if not line.endswith("\n"):
                    raise ValueError("truncated cache record")
                record = json.loads(line, object_pairs_hook=_object)
                digest = record.pop("sha256")
                if (record.pop("index") != self.record_index
                        or record.pop("previous") != self.previous):
                    raise ValueError("cache record chain is inconsistent")
                envelope = {**record, "index": self.record_index, "previous": self.previous}
                if digest != _digest(envelope):
                    raise ValueError("cache record checksum mismatch")
                self.apply(record)
                self.record_index += 1
                self.previous = digest
            if not self.record_index:
                if existed:
                    raise ValueError("existing evaluation cache is empty")
                self.append({"event": "scope", "version": 1, "operation": self.operation,
                             "inputs_sha256": self.inputs_sha256})
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except Exception as error:
            self.close()
            raise EvaluationRecoveryError("invalid or incompatible evaluation cache") from error

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def append(self, record):
        envelope = {**record, "index": self.record_index, "previous": self.previous}
        digest = _digest(envelope)
        self.file.write(_json({**envelope, "sha256": digest}) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        self.apply(record)
        self.record_index += 1
        self.previous = digest

    def apply(self, record):
        event = record["event"]
        if event == "scope":
            if self.record_index or record != {"event": "scope", "version": 1,
                                               "operation": self.operation,
                                               "inputs_sha256": self.inputs_sha256}:
                raise ValueError("scope identity changed")
            return
        if not self.record_index:
            raise ValueError("missing scope header")
        if event == "call":
            if (set(record) != {"event", "call", "inputs_sha256", "count"}
                    or type(record["call"]) is not int or record["call"] != len(self.calls)
                    or type(record["count"]) is not int or record["count"] <= 0
                    or not re.fullmatch(r"[0-9a-f]{64}", record["inputs_sha256"])):
                raise ValueError("invalid sampling call record")
            self.calls[record["call"]] = record
            return
        if event == "request":
            if set(record) != {"event", "unit", "call", "prompt_index", "inputs",
                               "sampling_session_id", "seq_id"}:
                raise ValueError("invalid sampling identity record")
            call, index = record["call"], record["prompt_index"]
            if (type(call) is not int or call not in self.calls or type(index) is not int
                    or not 0 <= index < self.calls[call]["count"]
                    or record["unit"] != f"{call}:{index}" or record["unit"] in self.units
                    or not isinstance(record["inputs"], dict)
                    or not isinstance(record["sampling_session_id"], str) or not record["sampling_session_id"]
                    or type(record["seq_id"]) is not int or record["seq_id"] < 0):
                raise ValueError("sampling identity is invalid or duplicated")
            identity = record["sampling_session_id"], record["seq_id"]
            if any((unit["sampling_session_id"], unit["seq_id"]) == identity for unit in self.units.values()):
                raise ValueError("sampling request identity reused")
            self.units[record["unit"]] = {**record, "attempts": {"submit": 0, "retrieve": 0},
                                           "failures": {"submit": 0, "retrieve": 0}}
            return
        unit = self.units[record["unit"]]
        if "result" in unit or unit.get("terminal"):
            raise ValueError("record follows completed or terminal sampling unit")
        if event == "attempt":
            phase = record["phase"]
            if (set(record) != {"event", "unit", "phase", "number", "billing_uncertain"}
                    or phase not in ("submit", "retrieve")
                    or (phase == "retrieve") != ("request_id" in unit)
                    or "response" in unit or record["number"] != unit["attempts"][phase] + 1
                    or record["billing_uncertain"] != (phase == "submit" and record["number"] > 1)):
                raise ValueError("invalid sampling attempt")
            unit["attempts"][phase] = record["number"]
        elif event == "ack":
            if (set(record) != {"event", "unit", "request_id"} or "request_id" in unit
                    or not unit["attempts"]["submit"]
                    or not isinstance(record["request_id"], str) or not record["request_id"]):
                raise ValueError("invalid sampling acknowledgement")
            unit["request_id"] = record["request_id"]
        elif event == "received":
            if (set(record) != {"event", "unit", "response"} or "request_id" not in unit
                    or "response" in unit or not unit["attempts"]["retrieve"]):
                raise ValueError("invalid sampling response record")
            unit["response"] = record["response"]
        elif event == "done":
            if (set(record) != {"event", "unit", "result"} or "response" not in unit
                    or set(record["result"]) != {"group", "accounting"}):
                raise ValueError("invalid completed sampling result")
            unit["result"] = record["result"]
        elif event == "failure":
            if (set(record) != {"event", "unit", "phase", "retryable", "error_type",
                               "status_code", "billing_uncertain"}
                    or record["phase"] not in ("submit", "retrieve", "decode")
                    or type(record["retryable"]) is not bool
                    or record["billing_uncertain"] != (record["phase"] == "submit")):
                raise ValueError("invalid sampling failure record")
            if record["phase"] in unit["failures"]:
                unit["failures"][record["phase"]] += 1
            unit["terminal"] = not record["retryable"]
        else:
            raise ValueError("unknown sampling cache event")

    def bind(self, inputs):
        self.open()
        call = self.call_index
        record = {"event": "call", "call": call, "inputs_sha256": _digest(inputs),
                  "count": len(inputs)}
        if call in self.calls:
            if self.calls[call] != record:
                raise EvaluationRecoveryError("frozen sampling inputs or call ordering changed")
        else:
            self.append(record)
        self.call_index += 1
        return call

    def plan(self, sampler, call, index, inputs):
        key = f"{call}:{index}"
        if key in self.units:
            unit = self.units[key]
            if unit["inputs"] != inputs:
                raise EvaluationRecoveryError("frozen per-prompt sampling inputs changed")
            return unit
        session, sequence = sampler._sampling_session_id, sampler._request_id_counter
        if not isinstance(session, str) or not session or type(sequence) is not int or sequence < 0:
            raise EvaluationRecoveryError("sampler lacks a valid pinned SDK request identity")
        sequence = max([sequence] + [unit["seq_id"] + 1 for unit in self.units.values()
                                     if unit["sampling_session_id"] == session])
        self.append({"event": "request", "unit": key, "call": call, "prompt_index": index,
                     "inputs": inputs, "sampling_session_id": session, "seq_id": sequence})
        sampler._request_id_counter = sequence + 1
        return self.units[key]


def _sdk():
    if importlib.metadata.version("tinker") != SDK_VERSION:
        raise EvaluationRecoveryError("sampling recovery requires the verified Tinker SDK")
    from tinker import types
    from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
    return types, ClientConnectionPoolType


async def _send(sampler, unit, prompt, params):
    """One HTTP submission, no public sample() or holder retry wrapper."""
    types, pools = _sdk()
    request = types.SampleRequest(
        sampling_session_id=unit["sampling_session_id"], seq_id=unit["seq_id"],
        num_samples=unit["inputs"]["num_samples"], prompt=prompt, sampling_params=params,
        prompt_logprobs=False, topk_prompt_logprobs=0)
    estimated = sampler.holder.estimate_bytes_count_in_model_input(prompt)
    async with sampler.holder.sample_dispatch_rate_limit(estimated):
        with sampler.holder.aclient(pools.SAMPLE) as client:
            future = await client.sampling.asample(
                request=request, max_retries=0, timeout=REQUEST_TIMEOUT,
                extra_headers={"X-Tinker-Sampling-Backpressure": "1"})
    if not isinstance(future.request_id, str) or not future.request_id:
        raise EvaluationRecoveryError("sampling acknowledgement omitted its request ID")
    return future.request_id


async def _receive(sampler, unit):
    """Retrieve the SAME future ID once; 410 is terminal, never resubmitted."""
    types, pools = _sdk()
    with sampler.holder.aclient(pools.RETRIEVE_PROMISE) as client:
        response = await client.futures.with_raw_response.retrieve(
            request=types.FutureRetrieveRequest(request_id=unit["request_id"], allow_metadata_only=False),
            timeout=REQUEST_TIMEOUT, max_retries=0,
            extra_headers={"Accept": "application/x-protobuf", "X-Tinker-Request-Type": "Sample",
                           "X-Tinker-Request-Iteration": str(unit["attempts"]["retrieve"] - 1)})
    if "application/x-protobuf" in response.headers.get("content-type", ""):
        return {"format": "tinker-protobuf", "base64": base64.b64encode(response.http_response.content).decode()}
    payload = await response.json()
    if isinstance(payload, dict) and payload.get("type") == "try_again":
        return None
    # Like SDK 0.25.0, accept SampleResponse only as protobuf. Application
    # errors, missing futures and unexpected formats cannot trigger sampling.
    raise EvaluationRecoveryError("sampling future returned an error or unexpected response format")


def _decode(response):
    if set(response) != {"format", "base64"} or response["format"] != "tinker-protobuf":
        raise EvaluationRecoveryError("unknown cached sampling response format")
    types, _ = _sdk()
    from tinker.proto.response_conv import deserialize_proto_response
    return deserialize_proto_response(base64.b64decode(response["base64"], validate=True), types.SampleResponse)


def is_transient_sampling_transport(error):
    """Classify only connection, timeout, 408/429 and 5xx transport failures.

    This public predicate is shared by immutable sampler construction and the
    identity-preserving request path. It deliberately excludes application,
    validation, cache and permanent HTTP failures.
    """
    status = getattr(error, "status_code", None)
    if status is not None:
        return status in (408, 429) or 500 <= status <= 599
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    try:
        from tinker import APIConnectionError
    except ImportError:
        return False
    return isinstance(error, APIConnectionError)


def _failure(scope, unit, phase, error, retryable):
    scope.append({"event": "failure", "unit": unit["unit"], "phase": phase,
                  "retryable": retryable, "error_type": type(error).__name__,
                  "status_code": getattr(error, "status_code", None),
                  "billing_uncertain": phase == "submit"})


async def _complete(scope, sampler, unit, prompt, params, normalize):
    if "result" in unit:
        return unit["result"]
    if unit.get("terminal"):
        raise EvaluationRecoveryError("sampling unit has a permanent recorded failure")
    deadline = time.monotonic() + MAX_SECONDS
    # Each explicit journal resume gets a bounded recovery window. The durable
    # counters remain monotonic; only this invocation's retry allowance resets.
    initial_attempts, initial_failures = dict(unit["attempts"]), dict(unit["failures"])
    while "response" not in unit:
        phase = "retrieve" if "request_id" in unit else "submit"
        limit = MAX_RETRIEVE_ATTEMPTS if phase == "retrieve" else MAX_SUBMIT_ATTEMPTS
        remaining = deadline - time.monotonic()
        attempts = unit["attempts"][phase] - initial_attempts[phase]
        failures = unit["failures"][phase] - initial_failures[phase]
        if (attempts >= limit or failures >= MAX_TRANSIENT_FAILURES
                or remaining <= 0):
            raise SamplingTransportError("bounded sampling recovery exhausted; original identity retained")
        scope.append({"event": "attempt", "unit": unit["unit"], "phase": phase,
                      "number": unit["attempts"][phase] + 1,
                      "billing_uncertain": phase == "submit" and unit["attempts"][phase] > 0})
        try:
            call = _receive(sampler, unit) if phase == "retrieve" else _send(sampler, unit, prompt, params)
            value = await asyncio.wait_for(call, timeout=min(REQUEST_TIMEOUT, remaining))
        except Exception as error:
            retryable = is_transient_sampling_transport(error)
            _failure(scope, unit, phase, error, retryable)
            if not retryable:
                raise EvaluationRecoveryError("permanent sampling failure; original identity retained") from error
            failures = unit["failures"][phase] - initial_failures[phase]
            attempts = unit["attempts"][phase] - initial_attempts[phase]
            if failures < MAX_TRANSIENT_FAILURES and attempts < limit:
                delay = min(30, 2 ** (failures - 1), max(0, deadline - time.monotonic()))
                await asyncio.sleep(delay)
            continue
        if phase == "submit":
            scope.append({"event": "ack", "unit": unit["unit"], "request_id": value})
        elif value is None:
            await asyncio.sleep(min(1, max(0, deadline - time.monotonic())))
        else:
            # Preserve the complete wire response before decoding/verification.
            scope.append({"event": "received", "unit": unit["unit"], "response": value})
    try:
        result = normalize(_decode(unit["response"]), unit["inputs"]["prompt_tokens"])
    except Exception as error:
        _failure(scope, unit, "decode", error, False)
        raise
    scope.append({"event": "done", "unit": unit["unit"], "result": result})
    return result


def sample_batch(sampler, sdk, prompts, parameters, samples, normalize):
    """Return aligned per-prompt results, with at most eight requests in flight."""
    scope = current_scope()
    if scope is None:
        raise EvaluationRecoveryError("immutable sampling requires a journal evaluation_scope")
    if getattr(sampler, "_sampling_client_sidecar_handle", None) is not None:
        raise EvaluationRecoveryError("sampling sidecar is outside the verified recovery boundary")
    inputs = [{"model_path": sampler._kintsugi_immutable_model_path, "prompt_tokens": prompt,
               "sampling_params": (params.model_dump(mode="json") if hasattr(params, "model_dump") else vars(params)),
               "num_samples": samples, "prompt_logprobs": False, "topk_prompt_logprobs": 0}
              for prompt, params in zip(prompts, parameters)]
    call = scope.bind(inputs)
    results = []
    for offset in range(0, len(prompts), WINDOW):
        units = [(scope.plan(sampler, call, index, inputs[index]),
                  sdk.ModelInput.from_ints(prompts[index]), parameters[index])
                 for index in range(offset, min(offset + WINDOW, len(prompts)))]

        async def window():
            return await asyncio.gather(*[_complete(scope, sampler, unit, prompt, params, normalize)
                                          for unit, prompt, params in units], return_exceptions=True)

        # Consume the entire window even if one fails, so its successful peers
        # remain durable. No later window is submitted after a failure.
        completed = sampler.holder.run_coroutine_threadsafe(window()).result()
        for result in completed:
            if isinstance(result, BaseException):
                raise result
        results.extend(completed)
    return results
