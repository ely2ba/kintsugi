# Running the registered study

The scientific authority is [SPEC.md](SPEC.md). This file describes execution,
not additional scientific options. No paid v2 call has been made.

## Current boundary

Local implementation and deterministic tests are in progress. M1 is **not yet
ready to launch**. In particular, the rule for constructing the §17 noise bounds
awaits the author's definition. `manifests/measurement.json` records that missing
value explicitly; there is no default multiplier or confidence bound.

M2 has no execution command. It requires the separate launch authorization after
M1 passes and the full twelve-lineage projection is delivered.

## Local checks and data

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

Public manifests bind the task generators, sources, splits, token rows, and four
orders. Downloaded source text, training rows, checkpoints and operational logs
stay in ignored local directories. Corpus preparation makes no model calls.

Task batch sizes are selected once from unscored, tokenizer-only workload
measurements, using the fixed rule in `data.py`. Natural-language batches contain
16 documents/chunks of at most 513 tokens; structured and language probes retain
the tested batch sizes of 32 and 8 respectively. Task gate and held-out sets each
contain 128 examples. Synthetic acquisition evaluation uses one deterministic
512-token-capped completion. The original IF suite remains unchanged at 60/30
prompts and a 96-token cap.

Reference, screening-1, screening-2, main, and persistence training rows are held
apart. The extra persistence split is calibration-only. Natural documents are
assigned before chunking; a document never crosses splits. The legal source's
URL identifies the collection, not a transcript, so transcript identity uses
normalized full-text hashes. Synthetic keys identify underlying instances, not
wording or random seeds. Primary and backup extraction formats use the same
underlying instance-to-split assignment.

All main task and repair schedules are keyed by task slot, never by arm or order.
The two screening realizations are disjoint data, not extra training seeds.

## Reference versus screening

Each acquisition reference uses all three registered LRs from cycle 0, for 120
updates with ten-update warmup. Gate and held-out metrics are evaluated at
0,5,…,120. Protected IF cannot stop this sweep. Gate and held-out maxima are
selected independently; invalid trajectories contribute no later points.

The first valid update-zero measurement in registered LR order supplies the
canonical cycle-0 baseline. Additional update-zero observations are retained;
they are not averaged or required to be bit-identical. References cannot freeze
until all three trajectories have either completed or become numerically invalid.

Screening is separate: task gate and IF are evaluated after every update, and
the first real joint crossing selects A. A held-out failure does not trigger a
search for a later, more favorable checkpoint. Probe references alone use twice
their standard budgets. The non-divisible language endpoints, 245 and 490, are
explicitly evaluated.

## Recovery and accounting

Each remote operation has an append-only start/completion record. Completed
operations are reused on resume. A request with an unknown outcome halts; it is
not automatically retried. New branches have fresh Adam state. Only recovery of
the same interrupted branch restores optimizer state.

Two short-lived state/sampler slots protect completed updates. Physical A/B
checkpoints receive separate durable saves. No-op B checkpoints alias A and
reuse its measurements; they do not become zero-valued repair observations.

Token accounting separates gradient-bearing targets, full train/forward inputs,
sampling prefill/cache/completions, and teacher scoring. `compute_logprobs` usage
is an estimate until billing events reconcile it. Monetary projections use the
live Tinker sheet, not v1's historical totals. No credentials or signed download
URLs belong in public artifacts.

A fresh, explicit v2 Tinker project is required. The backend rejects the two
historical projects. Before training the origin, download and validate its
adapter metadata: rank 32, alpha 32, expected attention/MLP tensors, no unembedding.
The archive reader also checks paths, tensor shapes and complete A/B pairs.

On 2026-08-31, an administrative download of the existing v1 origin sampler
successfully exercised the new download/extraction path: 259,590,376 tensor-file
bytes, rank 32, alpha 32, complete expected layout. This made no training,
sampling, forward or scoring call. V2 origin metadata must still be checked on
its own export; the v1 check does not substitute for that assertion.

## Source attribution

Unchanged components are selectively adapted from the terminal v1 implementation
identified in [PROVENANCE.md](PROVENANCE.md). The natural corpora are pinned
versions of [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia),
[Pile of Law](https://huggingface.co/datasets/pile-of-law/pile-of-law) Supreme Court
oral-argument transcripts, and the abstract-only unlabeled portion of
[PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA). Exact revisions,
files, licenses and split commitments are in `manifests/`.
