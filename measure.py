"""Native, frozen measurements; callers journal each complete operation.

No sampling or SDK import occurs at module import. These functions select no
thresholds, seeds, repetitions, learning rates, checkpoints, or stopping points.
"""

from collections import Counter
import math
import statistics

from backend import ACCOUNTING_KEYS, accounting, token_row
import if_suite
from protocol import IF_MAX_TOKENS
import tasks


def _add_usage(total, usage):
    for key in ACCOUNTING_KEYS:
        value = usage.get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError("invalid token accounting")
        total[key] += value


def evaluate_probe_loss(backend, client, rows):
    """Completion-masked, target-token-weighted NLL in fixed batches of 16."""
    rows = list(rows)
    if not rows:
        raise ValueError("evaluation rows must be nonempty")
    # Validate all local alignment before submitting any paid forward operation.
    for row in rows:
        token_row(row)
    usage, results = accounting(), []
    numerator = count = 0
    for start in range(0, len(rows), 16):
        result = backend.evaluate_nll(client, rows[start:start + 16])
        _add_usage(usage, result["accounting"])
        results.append({"first_row": start, "examples": len(rows[start:start + 16]),
                        "nll": result.get("nll"), "target_tokens": result["target_tokens"],
                        "valid": result.get("valid", True)})
        if not result.get("valid", True):
            return {"valid": False, "q": None, "nll": None,
                    "failure": result.get("failure", "invalid forward result"),
                    "accounting": usage, "results": results}
        if (not math.isfinite(result["nll"]) or type(result["target_tokens"]) is not int
                or result["target_tokens"] <= 0):
            raise RuntimeError("backend returned invalid NLL or denominator")
        numerator += result["nll"] * result["target_tokens"]
        count += result["target_tokens"]
    nll = numerator / count
    return {"valid": True, "q": -nll, "nll": nll, "target_tokens": count,
            "accounting": usage, "results": results}


def evaluate_task(backend, client, sampler_path, rows, manifest, seed):
    """Same native metric for gate and held-out sets; all Q values larger=better."""
    rows = list(rows)
    if not rows:
        raise ValueError("task evaluation rows must be nonempty")
    metric = manifest["metric"]
    if metric == "negative_nll":
        return evaluate_probe_loss(backend, client, rows)
    if metric != "verifier_success":
        raise ValueError("unknown registered native task metric")
    if manifest["evaluation"] != {"max_tokens": 512, "samples": 1, "temperature": 0.0}:
        raise ValueError("exact-task evaluation differs from the frozen sampling recipe")
    sampled = backend.sample(backend.sampler(sampler_path), [row["prompt_tokens"] for row in rows],
                             samples=1, max_tokens=512, temperature=0.0, seed=seed)
    if len(sampled["groups"]) != len(rows) or any(len(group) != 1 for group in sampled["groups"]):
        raise RuntimeError("exact-task result alignment failed")
    results = [{"semantic_key": row["semantic_key"], "passed": bool(tasks.verify(row, group[0]["text"])),
                "output": group[0]} for row, group in zip(rows, sampled["groups"])]
    passed = sum(result["passed"] for result in results)
    return {"valid": True, "q": passed / len(rows), "passed": passed, "total": len(rows),
            "accounting": sampled["accounting"], "results": results}


def evaluate_if(backend, sampler_path, split, seed):
    """Unchanged hash-bound suite: raw if_score count and separate score fraction."""
    if split not in ("criterion", "heldout"):
        raise ValueError("IF split must be criterion or heldout")
    selected = if_suite.items(split=split)
    prompts = [backend.render_prompt(item["prompt"]) for item in selected]
    sampled = backend.sample(backend.sampler(sampler_path), prompts, samples=1,
                             max_tokens=IF_MAX_TOKENS, temperature=0.0, seed=seed)
    if len(sampled["groups"]) != len(selected) or any(len(group) != 1 for group in sampled["groups"]):
        raise RuntimeError("IF result alignment failed")
    outputs = {item["id"]: group[0]["text"] for item, group in zip(selected, sampled["groups"])}
    result = if_suite.evaluate(outputs, split=split)
    return {"valid": True, **result, "if_score": result["passed"],
            "outputs": {item["id"]: group[0] for item, group in zip(selected, sampled["groups"])},
            "accounting": sampled["accounting"]}


def forward_kl(backend, sampler_path, prompt_tokens, groups):
    """E_cycle0[log p_cycle0 - log p_current], on frozen cycle-0 trajectories.

    The empirical average is over completion tokens only. It can be negative
    from sampling error; it is never clipped, resampled, or sign-reversed.
    """
    prompts = list(prompt_tokens)
    if not prompts or len(prompts) != len(groups) or any(not group for group in groups):
        raise ValueError("frozen trajectory prompt/group alignment failed")
    for prompt, group in zip(prompts, groups):
        for sample in group:
            token_row({"prompt_tokens": prompt, "completion_tokens": sample["tokens"]})
            if (len(sample["tokens"]) != len(sample["logprobs"])
                    or any(value is None or not math.isfinite(value) for value in sample["logprobs"])):
                raise ValueError("invalid frozen cycle-0 completion logprobs")
    scored = backend.score(backend.sampler(sampler_path), prompts, groups)
    if len(scored["logprobs"]) != len(groups):
        raise RuntimeError("current scorer group alignment failed")
    values = []
    for group, current_group in zip(groups, scored["logprobs"]):
        if len(group) != len(current_group):
            raise RuntimeError("current scorer trajectory alignment failed")
        for sample, current in zip(group, current_group):
            if len(sample["logprobs"]) != len(current):
                raise RuntimeError("current scorer completion alignment failed")
            values.extend(anchor - value for anchor, value in zip(sample["logprobs"], current))
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("nonfinite forward-KL estimate")
    return {"valid": True, "kl_to_cycle0": statistics.mean(values),
            "completion_tokens": len(values), "accounting": scored["accounting"]}


def diversity_summary(examples, groups):
    """Verified eight-sample coverage and item-local strategy concentration.

    pass@1 is mean binary success across all eight draws; pass@8 is mean any
    success per item. Concentration is sum(p_family**2) within each item among
    valid draws, then mean over items with a valid draw. No-valid concentration
    is unavailable, not zero. Counts and denominators accompany aggregates.
    """
    from probes import verify_diversity
    if not examples or len(examples) != len(groups) or any(len(group) != 8 for group in groups):
        raise ValueError("diversity requires exactly eight samples per nonempty item")
    results, all_logprobs, all_lengths, truncations = [], [], [], []
    for example, group in zip(examples, groups):
        canonical, families, bits, lengths = set(), Counter(), [], []
        for sample in group:
            tokens, logprobs = sample["tokens"], sample["logprobs"]
            if (not tokens or len(tokens) != len(logprobs)
                    or any(value is None or not math.isfinite(value) for value in logprobs)):
                raise ValueError("invalid diversity token/logprob alignment")
            verified = verify_diversity(example, sample["text"])
            bits.append(verified is not None)
            if verified is not None:
                canonical.add(verified["canonical_solution"])
                families[verified["strategy_family"]] += 1
            all_logprobs.extend(logprobs)
            lengths.append(len(tokens))
            truncations.append(bool(sample["truncated"]))
        valid_count = sum(bits)
        concentration = sum((count / valid_count) ** 2 for count in families.values()) if valid_count else None
        results.append({"semantic_key": example["semantic_key"], "valid": bits,
                        "valid_outputs": valid_count, "unique_valid_outputs": len(canonical),
                        "strategy_families": len(families), "strategy_family_counts": dict(families),
                        "strategy_family_concentration": concentration,
                        "lengths": lengths, "truncated": [bool(sample["truncated"]) for sample in group]})
        all_lengths.extend(lengths)
    pass1 = sum(row["valid_outputs"] for row in results) / (8 * len(results))
    pass8 = sum(row["valid_outputs"] > 0 for row in results) / len(results)
    concentrations = [row["strategy_family_concentration"] for row in results
                      if row["strategy_family_concentration"] is not None]
    return {"pass1": pass1, "pass8": pass8, "coverage_gap": pass8 - pass1,
            "unique_valid_outputs": statistics.mean(row["unique_valid_outputs"] for row in results),
            "strategy_families": statistics.mean(row["strategy_families"] for row in results),
            "strategy_family_concentration": statistics.mean(concentrations) if concentrations else None,
            "concentration_items": len(concentrations), "total_items": len(results),
            "sampled_token_surprisal": -statistics.mean(all_logprobs),
            "mean_completion_tokens": statistics.mean(all_lengths),
            "truncation_rate": statistics.mean(truncations), "results": results}


def noise_summary(repeated, paired=None):
    """Descriptive repeats of one physical-state statistic, not seed inference.

    Callers provide the fixed repetition schedule and matched replicate pairs.
    No confidence level, multiplier, noise-bound selection or count is invented.
    """
    repeated = list(repeated)
    paired = [] if paired is None else list(paired)
    if not repeated or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                           for value in repeated):
        raise ValueError("finite repeated measurements are required")
    if any(len(pair) != 2 or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                                for value in pair) for pair in paired):
        raise ValueError("paired measurements require finite two-element pairs")
    return {"count": len(repeated), "mean": statistics.mean(repeated),
            "sample_sd": statistics.stdev(repeated) if len(repeated) > 1 else None,
            "paired_absolute_differences": [abs(left - right) for left, right in paired]}
