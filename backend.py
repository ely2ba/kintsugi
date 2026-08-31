"""Small Tinker boundary for SPEC.md; importing this module is entirely offline.

Adapted selectively from kintsugi-v1 repair.py/probe.py/lineage.py/analyze.py at
d2aa7cd94a0a618169496f0235fa021ea46c0372. No v1 data, thresholds or run paths.

The runner owns durable inflight/done records around EVERY remote operation.
Exceptions are deliberately not retried: an unfinished operation is ambiguous,
not permission to repeat paid work. Client-returning methods return SDK clients;
measurement/update/save methods return JSON-safe dictionaries with ``accounting``.
``train_tokens`` and ``forward_tokens`` are full input lengths, independently of
``gradient_target_tokens`` (completion-mask positions, not prompt positions).
Scorer usage is separately marked as estimated because compute_logprobs returns
no cache/billing details and internally discards a one-token sample.
"""

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import struct
import subprocess
import tarfile
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

from protocol import TOKENIZER_REVISION

MODEL = "Qwen/Qwen3.5-4B"
RANK = ALPHA = 32
SDK_VERSION = "0.25.0"
# Rejection guards only. These projects must never receive v2 work.
HISTORICAL_PROJECT_HASHES = frozenset({
    "2dfe3c1d0aa4a06b76bd01b4fd95def6b831705e7bdce6af402c1e11eee91e90",
    "b39ae3e52ac2aaf5410f321b86db2c14a6bf27e146b98da87f7f6e3f21be68c1",
})
ACCOUNTING_KEYS = (
    "gradient_target_tokens", "train_tokens", "forward_tokens", "prefill_tokens",
    "cached_tokens", "sample_tokens", "scoring_prefill_tokens",
    "scoring_discarded_sample_tokens_estimate",
)


def accounting(**values):
    if values.keys() - set(ACCOUNTING_KEYS):
        raise ValueError("unknown accounting field")
    return {key: values.get(key, 0) for key in ACCOUNTING_KEYS}


def _tokens(values):
    values = list(values)
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("token sequences must contain nonnegative integers")
    return values


def token_row(row):
    """Return complete tokens and an unnormalized next-token target mask."""
    if "tokens" in row:
        if "prompt_tokens" in row or "completion_tokens" in row:
            raise ValueError("ambiguous token row")
        tokens = _tokens(row["tokens"])
        if len(tokens) < 2:
            raise ValueError("language rows need at least two tokens")
        return tokens, [1] * (len(tokens) - 1)
    prompt, completion = _tokens(row["prompt_tokens"]), _tokens(row["completion_tokens"])
    return prompt + completion, [0] * (len(prompt) - 1) + [1] * len(completion)


def _values(value):
    return list(value.data if hasattr(value, "data") else value)


def _finite(values, label):
    if any(value is None or not math.isfinite(float(value)) for value in values):
        raise RuntimeError(f"nonfinite or unavailable {label}")
    return [float(value) for value in values]


def _nll(output, masks):
    if len(output.loss_fn_outputs) != len(masks):
        raise RuntimeError("cross-entropy output count mismatch")
    numerator = denominator = 0
    for result, mask in zip(output.loss_fn_outputs, masks):
        if "logprobs" not in result:
            raise RuntimeError("cross-entropy logprobs missing")
        values = _finite(_values(result["logprobs"]), "cross-entropy logprob")
        if len(values) != len(mask):
            raise RuntimeError("cross-entropy logprob alignment failed")
        numerator -= sum(value * weight for value, weight in zip(values, mask))
        denominator += sum(mask)
    if not denominator:
        raise RuntimeError("no target tokens")
    return numerator / denominator, denominator


def _optimizer_finite(output):
    _finite([value for value in (output.metrics or {}).values()
             if isinstance(value, (int, float))], "optimizer metric")


def _checkpoint_path(path, kind):
    parsed = urlparse(str(path))
    if (parsed.scheme != "tinker" or not parsed.netloc or parsed.query or parsed.fragment
            or not re.fullmatch(rf"/{kind}/[A-Za-z0-9_.-]+", parsed.path)):
        raise ValueError(f"expected a Tinker /{kind}/ checkpoint path")
    return str(path)


async def _one_attempt(function, *args, **kwargs):
    """Disable the pinned SDK's extra holder retry loop, not just HTTP retries."""
    return await function(*args, **kwargs)


class Backend:
    def __init__(self, service, sdk, types, tokenizer, *, seed, checkpoint_ttl,
                 retry_config=None):
        if type(seed) is not int or seed < 0 or type(checkpoint_ttl) is not int or checkpoint_ttl <= 0:
            raise ValueError("explicit nonnegative seed and positive checkpoint TTL required")
        self.service, self.sdk, self.types, self.tokenizer = service, sdk, types, tokenizer
        self.seed, self.checkpoint_ttl = seed, checkpoint_ttl
        self.retry_config = retry_config
        self.end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (type(self.end_token) is not int or self.end_token < 0
                or self.end_token == getattr(tokenizer, "unk_token_id", None)):
            raise ValueError("tokenizer lacks the Qwen end-of-turn token")

    @classmethod
    def connect(cls, project_id, keychain_service, *, seed, checkpoint_ttl):
        """Explicit live boundary; cached tokenizer, Keychain only, no retries.

        This is NOT invoked on import or by tests. The pinned SDK has no alpha
        argument: save/download/inspect cycle 0 before permitting any training.
        """
        if (not isinstance(project_id, str) or not project_id.strip()
                or hashlib.sha256(project_id.strip().lower().encode()).hexdigest() in HISTORICAL_PROJECT_HASHES):
            raise ValueError("an explicit fresh v2 project ID is required")
        if not isinstance(keychain_service, str) or not keychain_service.strip():
            raise ValueError("an explicit Keychain service is required")
        if os.environ.get("TINKER_PROJECT_ID") not in (None, "", project_id):
            raise RuntimeError("environment project differs from the explicit v2 project")
        if os.environ.get("TINKER_SUBPROCESS_SAMPLING", "").lower() in ("1", "true", "yes"):
            raise RuntimeError("subprocess sampling would bypass the no-retry holder guard")
        import importlib.metadata
        if importlib.metadata.version("tinker") != SDK_VERSION:
            raise RuntimeError("Tinker SDK version differs from the verified backend")
        import tinker
        from tinker import types
        from tinker.lib.retry_handler import RetryConfig
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION, local_files_only=True)
        credential = subprocess.run(
            ["security", "find-generic-password", "-s", keychain_service, "-w"],
            capture_output=True, text=True, check=False,
        )
        if credential.returncode or not credential.stdout.strip():
            raise RuntimeError("the supplied Keychain credential is unavailable")
        # The server can require credential-command auth even when api_key was
        # passed explicitly. Bind that path to the same supplied Keychain item.
        os.environ["TINKER_CREDENTIAL_CMD"] = shlex.join(
            ["security", "find-generic-password", "-s", keychain_service, "-w"])
        service = tinker.ServiceClient(project_id=project_id,
                                       api_key=credential.stdout.strip(), max_retries=0)
        # The holder loop retries even when HTTP max_retries=0. Keep this change
        # instance-local and version-checked; never modify an installed SDK file.
        service.holder.execute_with_retries = _one_attempt
        return cls(service, tinker, types, tokenizer, seed=seed,
                   checkpoint_ttl=checkpoint_ttl,
                   retry_config=RetryConfig(enable_retry_logic=False))

    def origin(self):
        return self.service.create_lora_training_client(
            base_model=MODEL, rank=RANK, seed=self.seed,
            train_attn=True, train_mlp=True, train_unembed=False,
        )

    def branch(self, state_path, *, resume=False):
        """Fresh Adam for a new branch; resume=True ONLY for the same branch."""
        path = _checkpoint_path(state_path, "weights")
        create = (self.service.create_training_client_from_state_with_optimizer if resume
                  else self.service.create_training_client_from_state)
        return create(path)

    def sampler(self, sampler_path):
        return self.service.create_sampling_client(
            model_path=_checkpoint_path(sampler_path, "sampler_weights"),
            base_model=MODEL, retry_config=self.retry_config,
        )

    def render_prompt(self, prompt):
        """Frozen Qwen non-thinking rendering, with no silent prompt truncation."""
        text = (f"<|im_start|>user\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return _tokens(self.tokenizer.encode(text, add_special_tokens=False))

    def datum(self, row):
        tokens, mask = token_row(row)
        denominator = sum(mask)
        return self.sdk.Datum(model_input=self.sdk.ModelInput.from_ints(tokens[:-1]),
                              loss_fn_inputs={"target_tokens": tokens[1:],
                                              "weights": [v / denominator for v in mask]})

    def evaluate_nll(self, client, rows):
        """Return {nll, q=-nll, target_tokens, accounting}; token-weighted NLL."""
        rows = list(rows)
        if not rows:
            raise ValueError("empty evaluation batch")
        datums, masks = [self.datum(row) for row in rows], [token_row(row)[1] for row in rows]
        output = client.forward(datums, loss_fn="cross_entropy").result()
        usage = accounting(forward_tokens=sum(d.model_input.length for d in datums))
        try:
            nll, count = _nll(output, masks)
        except RuntimeError as error:
            return {"valid": False, "failure": str(error), "nll": None, "q": None,
                    "target_tokens": sum(map(sum, masks)), "accounting": usage}
        return {"valid": True, "nll": nll, "q": -nll, "target_tokens": count, "accounting": usage}

    def train_step(self, client, rows, *, learning_rate, step, warmup_steps=10):
        """One SFT update; probes explicitly pass warmup_steps=0."""
        lr = _learning_rate(learning_rate, step, warmup_steps)
        rows = list(rows)
        if not rows:
            raise ValueError("empty training batch")
        datums, masks = [self.datum(row) for row in rows], [token_row(row)[1] for row in rows]
        output = client.forward_backward(datums, loss_fn="cross_entropy").result()
        usage = accounting(gradient_target_tokens=sum(map(sum, masks)),
                           train_tokens=sum(d.model_input.length for d in datums))
        try:
            nll, _ = _nll(output, masks)
        except RuntimeError as error:
            return {"valid": False, "failure": str(error), "step": step, "learning_rate": lr,
                    "optimizer_applied": False, "accounting": usage}
        optimizer = client.optim_step(self.types.AdamParams(
            learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0,
        )).result()
        try:
            _optimizer_finite(optimizer)
        except RuntimeError as error:
            return {"valid": False, "failure": str(error), "step": step, "learning_rate": lr,
                    "optimizer_applied": True, "accounting": usage}
        return {"valid": True, "nll": nll, "q": -nll, "learning_rate": lr, "step": step,
                "optimizer_applied": True, "accounting": usage}

    def sample(self, sampler, prompt_tokens, *, samples, max_tokens, temperature, seed):
        """Return {groups, accounting}; groups have text/tokens/logprobs/stop flags."""
        prompts = [_tokens(prompt) for prompt in prompt_tokens]
        if (not prompts or type(samples) is not int or samples <= 0 or type(max_tokens) is not int
                or max_tokens <= 0 or not math.isfinite(temperature) or temperature < 0
                or type(seed) is not int or seed < 0):
            raise ValueError("invalid sampling parameters")
        futures = []
        for index, prompt in enumerate(prompts):
            params = self.types.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                               top_p=1.0, top_k=-1, stop=[self.end_token],
                                               seed=seed + index)
            futures.append(sampler.sample(self.sdk.ModelInput.from_ints(prompt), samples, params))
        groups, usage = [], accounting()
        for prompt, future in zip(prompts, futures):
            response = future.result()
            if len(response.sequences) != samples:
                raise RuntimeError("sample returned the wrong sequence count")
            hit = response.prompt_cache_hit_tokens
            if type(hit) is not int or not 0 <= hit <= len(prompt):
                raise RuntimeError("invalid prompt cache-hit accounting")
            usage["prefill_tokens"] += len(prompt) - hit
            usage["cached_tokens"] += hit + (samples - 1) * len(prompt)
            group = []
            for sequence in response.sequences:
                tokens = _tokens(sequence.tokens)
                if sequence.logprobs is None or len(tokens) != len(sequence.logprobs):
                    raise RuntimeError("sample token/logprob alignment failed")
                logprobs = _finite(sequence.logprobs, "sample logprob")
                if len(tokens) > max_tokens:
                    raise RuntimeError("sample exceeded completion cap")
                if self.end_token in tokens[:-1]:
                    raise RuntimeError("sample continued beyond end-of-turn token")
                reason = str(sequence.stop_reason)
                text = self.tokenizer.decode(tokens[:-1] if tokens[-1] == self.end_token else tokens,
                                             skip_special_tokens=False)
                group.append({"text": text, "tokens": tokens, "logprobs": logprobs,
                              "stop_reason": reason, "truncated": reason.lower().endswith("length")})
                usage["sample_tokens"] += len(tokens)
            groups.append(group)
        return {"groups": groups, "accounting": usage}

    def score(self, sampler, prompt_tokens, groups):
        """Teacher or drift scorer: completion-aligned logprobs and usage estimates."""
        prompts = [_tokens(prompt) for prompt in prompt_tokens]
        if not prompts or len(prompts) != len(groups) or any(not group for group in groups):
            raise ValueError("scorer prompt/group alignment failed")
        trajectories = [(index, prompt, _tokens(sample["tokens"]))
                        for index, (prompt, group) in enumerate(zip(prompts, groups)) for sample in group]
        futures = [sampler.compute_logprobs(self.sdk.ModelInput.from_ints(prompt + tokens))
                   for _, prompt, tokens in trajectories]
        scored, usage = [[] for _ in prompts], accounting()
        for (index, prompt, tokens), future in zip(trajectories, futures):
            values = future.result()
            if len(values) != len(prompt) + len(tokens):
                raise RuntimeError("scorer full-trajectory alignment failed")
            scored[index].append(_finite(values[len(prompt):], "scorer completion logprob"))
            usage["scoring_prefill_tokens"] += len(prompt) + len(tokens)
            usage["scoring_discarded_sample_tokens_estimate"] += 1
        return {"logprobs": scored, "accounting": usage}

    def repair_step(self, client, teacher, prompt_tokens, *, step, seed):
        """One on-policy reverse-KL update, refreshing the student sampler first."""
        _learning_rate(1e-4, step, 10)
        prompts = [_tokens(prompt) for prompt in prompt_tokens]
        if len(prompts) != 64:
            raise ValueError("repair requires exactly 64 prompt groups")
        student = client.save_weights_and_get_sampling_client(retry_config=self.retry_config)
        sampled = self.sample(student, prompts, samples=4, max_tokens=4096, temperature=1.0, seed=seed)
        groups = sampled["groups"]
        scored = self.score(teacher, prompts, groups)
        datums, divergences = [], []
        for prompt, group, teacher_group in zip(prompts, groups, scored["logprobs"]):
            for sample, anchor in zip(group, teacher_group):
                full, prefix = prompt + sample["tokens"], len(prompt) - 1
                reverse_kl = _finite([student_lp - teacher_lp for student_lp, teacher_lp
                                      in zip(sample["logprobs"], anchor)], "reverse KL")
                divergences.extend(reverse_kl)
                datums.append(self.sdk.Datum(
                    model_input=self.sdk.ModelInput.from_ints(full[:-1]),
                    loss_fn_inputs={"target_tokens": full[1:],
                                    "logprobs": [0.0] * prefix + sample["logprobs"],
                                    "advantages": [0.0] * prefix + [-v for v in reverse_kl]},
                ))
        output = client.forward_backward(datums, loss_fn="importance_sampling").result()
        if len(output.loss_fn_outputs) != len(datums):
            raise RuntimeError("repair output count mismatch")
        for result in output.loss_fn_outputs:
            for value in result.values():
                _finite(_values(value), "repair output")
        lr = _learning_rate(1e-4, step, 10)
        _optimizer_finite(client.optim_step(self.types.AdamParams(
            learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
        )).result())
        usage = {key: sampled["accounting"][key] + scored["accounting"][key]
                 for key in ACCOUNTING_KEYS}
        usage.update(gradient_target_tokens=len(divergences),
                     train_tokens=sum(d.model_input.length for d in datums))
        return {"step": step, "learning_rate": lr,
                "reverse_kl": sum(divergences) / len(divergences), "accounting": usage}

    def save(self, client, name, *, step, resume=False):
        """Save both kinds, rotating only the optimizer-state resume slots.

        Resume slots are only for the same branch, with contiguous steps recorded
        by the runner. Sampler saves do not support overwrite, so their names
        always identify the step; resume samplers retain the short TTL. Final
        A/B checkpoints require a separate durable save.
        """
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or type(step) is not int or step < 0:
            raise ValueError("safe checkpoint name and nonnegative step required")
        if resume and step < 1:
            raise ValueError("resume checkpoints require a completed update")
        sampler_label = f"{name}-step-{step:06d}"
        label = f"{name}-resume-{step % 2}" if resume else sampler_label
        ttl = min(self.checkpoint_ttl, 2 * 86400) if resume else self.checkpoint_ttl
        options = {"ttl_seconds": ttl}
        if resume:
            options["overwrite"] = step > 2
        state = client.save_state(label, **options).result()
        state_path = _checkpoint_path(state.path, "weights")
        sampled = client.save_weights_for_sampler(sampler_label, ttl_seconds=ttl).result()
        sampler_path = _checkpoint_path(sampled.path, "sampler_weights")
        if urlparse(state_path).netloc != urlparse(sampler_path).netloc:
            raise RuntimeError("saved state and sampler belong to different training runs")
        if (urlparse(state_path).path.split("/")[-1] != label
                or urlparse(sampler_path).path.split("/")[-1] != sampler_label):
            raise RuntimeError("saved checkpoint names differ from the requested step")
        return {"name": label, "sampler_name": sampler_label, "step": step,
                "state_path": state_path, "sampler_path": sampler_path,
                "ttl_seconds": ttl, "resume_slot": step % 2 if resume else None,
                "accounting": accounting()}

    def download_sampler(self, sampler_path, destination, *, max_bytes=2 * 1024**3):
        """Download one signed archive; refuse overwrites and unsafe members."""
        path = _checkpoint_path(sampler_path, "sampler_weights")
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        response = self.service.create_rest_client().get_checkpoint_archive_url_from_tinker_path(path).result()
        if urlparse(response.url).scheme != "https":
            raise RuntimeError("checkpoint download requires HTTPS")
        with tempfile.TemporaryDirectory(prefix="kintsugi-download-") as temporary:
            archive = Path(temporary) / "sampler.tar"
            with urlopen(response.url, timeout=60) as remote, archive.open("xb") as handle:
                size = 0
                while chunk := remote.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise RuntimeError("sampler archive exceeds size limit")
                    handle.write(chunk)
            extracted = safe_extract_sampler(archive, destination, max_bytes=max_bytes)
        return {"sampler_path": path, "directory": str(destination),
                "adapter": inspect_adapter(extracted / "adapter_model.safetensors"),
                "accounting": accounting()}


def _learning_rate(learning_rate, step, warmup_steps):
    if (not math.isfinite(learning_rate) or learning_rate <= 0 or type(step) is not int or step < 1
            or type(warmup_steps) is not int or warmup_steps < 0):
        raise ValueError("invalid optimizer schedule")
    return learning_rate * min(1.0, step / warmup_steps) if warmup_steps else learning_rate


def safe_extract_sampler(archive, destination, *, max_bytes=2 * 1024**3):
    """Extract only the known root-level sampler files, never links or traversal."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    required = {"adapter_config.json", "adapter_model.safetensors"}
    allowed = required | {"checkpoint_complete"}
    with tarfile.open(archive, "r:*") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if (len(set(names)) != len(names) or not required <= set(names) or set(names) - allowed
                or any(not member.isfile() or member.size < 0 for member in members)
                or sum(member.size for member in members) > max_bytes
                or any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts for name in names)):
            raise RuntimeError("unsafe or unexpected sampler archive contents")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".kintsugi-extract-", dir=destination.parent) as stage:
            for member in members:
                with bundle.extractfile(member) as source, (Path(stage) / member.name).open("xb") as target:
                    shutil.copyfileobj(source, target)
            # Publish only after all bytes and local metadata have validated.
            inspect_adapter(Path(stage) / "adapter_model.safetensors")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            os.rename(stage, destination)
    return destination


def _expected_modules():
    # Qwen3.5-4B's tested export layout: 32 layers, full attention every fourth.
    names = []
    for layer in range(32):
        modules = [f"mlp.{name}_proj" for name in ("down", "gate", "up")]
        modules += ([f"self_attn.{name}_proj" for name in ("q", "k", "v", "o")]
                    if layer % 4 == 3 else [f"linear_attn.in_proj_{name}" for name in ("q", "k", "v", "z")]
                    + ["linear_attn.out_proj"])
        names.extend(f"base_model.model.model.layers.{layer}.{name}" for name in modules)
    return sorted(names)


def inspect_adapter(tensor_path, *, expected_layout=None):
    """Verify alpha/rank, every A/B pair, model scope, shapes and tensor bounds."""
    tensor_path = Path(tensor_path)
    metadata = json.loads((tensor_path.parent / "adapter_config.json").read_text())
    if (metadata.get("r") != RANK or metadata.get("lora_alpha") != ALPHA
            or any(metadata.get(key) for key in ("rank_pattern", "alpha_pattern", "use_rslora",
                                                 "use_dora", "use_qalora", "fan_in_fan_out",
                                                 "modules_to_save", "lora_bias"))):
        raise RuntimeError("adapter metadata violates rank-32 alpha-32 ordinary LoRA")
    with tensor_path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise RuntimeError("truncated safetensors header")
        size = struct.unpack("<Q", prefix)[0]
        if size <= 0 or size > 16 * 1024**2:
            raise RuntimeError("invalid safetensors header size")
        header = json.loads(handle.read(size))
    entries = {key: value for key, value in header.items() if key != "__metadata__"}
    expected = {f"{name}.lora_{side}.weight" for name in _expected_modules() for side in ("A", "B")}
    if set(entries) != expected:
        raise RuntimeError("adapter tensor names differ from the Qwen attention/MLP-only layout")
    intervals, layout = [], []
    for name in _expected_modules():
        shapes = []
        for side in ("A", "B"):
            entry = entries[f"{name}.lora_{side}.weight"]
            shape, offsets = entry["shape"], entry["data_offsets"]
            width = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}.get(entry["dtype"])
            if (len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape)
                    or width is None or len(offsets) != 2
                    or any(type(v) is not int or v < 0 for v in offsets)
                    or offsets[1] - offsets[0] != math.prod(shape) * width):
                raise RuntimeError("invalid adapter shape, dtype or tensor offsets")
            intervals.append(tuple(offsets))
            shapes.append(shape)
        if shapes[0][0] != RANK or shapes[1][1] != RANK:
            raise RuntimeError("adapter A/B shape does not match rank")
        layout.append({"layer": name, "a_shape": shapes[0], "b_shape": shapes[1]})
    intervals.sort()
    if (intervals[0][0] != 0 or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:]))
            or intervals[-1][1] + 8 + size != tensor_path.stat().st_size):
        raise RuntimeError("overlapping, gapped or truncated adapter tensor storage")
    if expected_layout is not None and layout != expected_layout:
        raise RuntimeError("adapter layout changed from the frozen origin")
    return {"rank": RANK, "alpha": ALPHA, "scale": ALPHA / RANK, "layout": layout}


def lora_geometry(a, b, *, alpha, previous=None):
    """Registered scaled B@A geometry in compact rank space; no dense delta W."""
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if (a.ndim != 2 or b.ndim != 2 or not a.shape[0] or b.shape[1] != a.shape[0]
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("invalid LoRA factors or alpha")
    scale = alpha / a.shape[0]
    _, rb = np.linalg.qr(b, mode="reduced")
    _, ra = np.linalg.qr(a.T, mode="reduced")
    # Some BLAS builds leave spurious floating-point status flags. Check the
    # resulting compact matrix explicitly instead of trusting those flags.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        compact = rb @ ra.T * scale
    if not np.isfinite(compact).all():
        raise RuntimeError("nonfinite compact LoRA delta")
    singular = np.linalg.svd(compact, compute_uv=False)
    positive = singular[singular > 0]
    probabilities = positive / positive.sum() if len(positive) else positive
    norm = float(np.linalg.norm(singular))
    result = {"frobenius": norm,
              "effective_rank": float(np.exp(-(probabilities * np.log(probabilities)).sum())) if len(positive) else 0.0,
              "stable_rank": float((singular ** 2).sum() / singular[0] ** 2) if len(positive) else 0.0,
              "cosine_to_previous": None}
    if not all(math.isfinite(result[key]) for key in ("frobenius", "effective_rank", "stable_rank")):
        raise RuntimeError("nonfinite LoRA geometry")
    if previous is not None:
        old_a, old_b, old_alpha = previous
        old_a, old_b = np.asarray(old_a, dtype=np.float64), np.asarray(old_b, dtype=np.float64)
        if old_a.shape != a.shape or old_b.shape != b.shape:
            raise ValueError("adapter factor layout changed")
        old_norm = lora_geometry(old_a, old_b, alpha=old_alpha)["frobenius"]
        if old_norm and norm:
            inner = float(np.sum((old_b.T @ b) * (old_a @ a.T)) * (old_alpha / old_a.shape[0]) * scale)
            result["cosine_to_previous"] = inner / (old_norm * norm)
    return result


def adapter_geometry(tensor_path, *, previous_path=None):
    """Per-module drift panel. Imports numpy/safetensors only when requested."""
    import numpy as np
    from safetensors import safe_open

    def factors(path, expected_layout=None):
        path = Path(path)
        info = inspect_adapter(path, expected_layout=expected_layout)
        # Library validation, including the safetensors storage format itself.
        with safe_open(str(path), framework="np") as tensors:
            keys = list(tensors.keys())
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
            arrays = {}
            for key in keys:
                entry = header[key]
                start, end = entry["data_offsets"]
                handle.seek(8 + size + start)
                raw = handle.read(end - start)
                if entry["dtype"] == "BF16":
                    array = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
                else:
                    array = np.frombuffer(raw, dtype={"F16": "<f2", "F32": "<f4", "F64": "<f8"}[entry["dtype"]])
                arrays[key] = array.reshape(entry["shape"])
        return info, arrays

    info, arrays = factors(tensor_path)
    old = factors(previous_path, info["layout"])[1] if previous_path else None
    rows = []
    for module in info["layout"]:
        name = module["layer"]
        a_key, b_key = f"{name}.lora_A.weight", f"{name}.lora_B.weight"
        previous = (old[a_key], old[b_key], ALPHA) if old is not None else None
        rows.append({"layer": name, "alpha": ALPHA, "rank": RANK, "scale": ALPHA / RANK,
                     **lora_geometry(arrays[a_key], arrays[b_key], alpha=ALPHA, previous=previous)})
    return rows
