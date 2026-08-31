# Running the registered study

The scientific authority is [SPEC.md](SPEC.md). This file describes execution,
not additional scientific options. M1 was launched on 2026-08-31; M2 is not authorized.

## Current boundary

The M0/M1 runner and its deterministic tests are implemented. The author has
frozen the measurement-noise rule in SPEC §17.0 and
`manifests/measurement.json`. Execution requires the public tested-code/input
freeze. A separate `kintsugi-v2` Tinker project has been created; its operational
ID stays local. The initial public input-freeze commit is
`8dc01c8a85ea2d1e6706a76ad4270d30b23438bd`, with 188 passing local tests.
The current tested implementation is recorded in `manifests/freeze.json`.

On 2026-08-31, execution stopped during the first acquisition-reference update:
the sampler-save SDK method rejected the `overwrite` argument after the update
and optimizer-state save had completed. The original remote update-1 state was
verified and restored with its Adam moments, then both checkpoint types were
exported without repeating training. The unfinished journal entry was completed
append-only; the original cycle-0 checkpoint and measurements were preserved.
Update 1's training-loss response and active-call duration were not retained and
are explicitly unavailable. Its token counts were reconstructed from the exact
frozen batch; no scheduled gate or held-out evaluation was lost.

The fix uses step-unique sampler names, since the pinned SDK supports overwriting
optimizer states but not sampler exports. Regression tests enforce the installed
SDK signatures and exercise multiple saves and recovery without repeated updates.
The implementation revision preserves the original journal identity and records
the new tested code separately. Preflight requires the original SPEC and all
input-manifest hashes to remain identical. This is an engineering correction,
not a scientific-contract change or a calibration-gate result.

The recovery implementation (`499c61d`) passed 197 tests and was publicly frozen
in `1716e0e` before continuation. M1 resumed at update 2, which completed with both
checkpoint saves. M2 remains unauthorized.

Initialization first launched under `938fb1f`. Before any sampling, scoring,
forward pass, or training update, an overlapping diversity-repeat seed range
was corrected in `138b24d` and re-frozen in `8dc01c8`. The original untrained
cycle-0 state and sampler exports were preserved; no scientific trajectory was
restarted. The initial administrative journal is retained alongside the active
journal. This is an implementation correction, not a scientific-contract change.

M2 has no execution command. It requires the separate launch authorization after
M1 passes and the full twelve-lineage projection is delivered.

The runner stops immediately on a registered M1 failure. If every gate passes,
it downloads and measures selected A/B checkpoints, completes retention and the
registered drift panel, and writes `runs/m1/launch_packet.json`. That packet is
not an M2 authorization: the completed calibration package and selections must
be published, and their exact freeze commit supplied in the launch message.

## Local checks and data

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python m1.py preflight
```

After the public freeze passes preflight, the explicit M1 command is
`.venv/bin/python m1.py run --project-id <fresh-v2-project-id> --keychain-service <keychain-item>`.
It has no implicit project, credential, or main-run fallback. A completed
measurement handoff or scientific stop is returned locally on resume, without
connecting to the service again.

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

Repair keeps v1's 1,024-token user-prompt preprocessing before chat rendering;
acquisition prompts are not truncated by that rule. This affects 49 of the
2,000 frozen repair prompts. The repair completion cap remains 4,096 tokens.
Diversity candidates qualify on their first frozen sampling realization;
three independent complete realizations under the selected recipe supply its
registered noise bound. Their SD is not divided by the square root of three.
Their per-prompt sampling-seed ranges are disjoint: realization `r` starts at
the panel seed plus `r × panel_size`. This avoids sharing random streams between
different prompt positions across repeats; the main-panel seed is unchanged.

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

Two short-lived optimizer-state slots protect completed updates; sampler exports
use unique step names with the same short expiry. Physical A/B checkpoints receive
separate durable saves. No-op B checkpoints alias A and reuse its measurements;
they do not become zero-valued repair observations.

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
