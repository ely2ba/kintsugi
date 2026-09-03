# M1 anomaly-resolution diagnostic

**Date:** 2026-09-03
**Status:** complete; diagnostic only; no selection or design authority

## Answer and disposition

The exact v1 cycle-1/f01 intervention still produces substantial damage on the
60-prompt protected IF criterion suite in the current environment. This diagnostic
did not evaluate the separate 30-prompt held-out IF suite.

| Measurement | Protected IF criterion |
|---|---:|
| Fresh replay origin, criterion suite | 44/60 |
| After the exact 20-update v1 intervention | 37/60 |
| Change | **−7 points** |

At item level, 9 criterion prompts changed from pass to fail and 2 changed from
fail to pass, for the net 7-point drop.

The five original v1 cycle-1/f01 A scores were 35, 36, 36, 34, and 37/60. Relative
to the single registered 43/60 cycle-0 score, those are differences of 6–9 points;
they are not seed-matched paired drops because the lineages used different
evaluation seeds. The replay's 37/60 endpoint is inside the historical A range,
and its genuine paired 7-point drop is inside that S0-referenced difference range.
It is not an exact replication of fixed-0's original 36/60 endpoint. It is also
one point above v1's old registered 26–36 damage band, while lying exactly at v2's
current 27–37 band ceiling. The replay baseline and the v2 M1 cycle-0 baseline
both happened to total 44/60, but they are different frozen sampling realizations:
seeds `1738054771` and `774298527`, respectively. Their item vectors disagree on
`c-forbidden-11` and `c-forbidden-14`, one pass exchanged in each direction.

The task itself was acquired strongly: its historical-cadence validation NLL fell
from 0.608718 to 0.075775, an 87.55% reduction. There was no intermediate IF grid in
v1; the replay therefore added none.

This resolves the requested fork in favor of the first disposition: stop the
diagnostic. The replay supports—and is consistent with—the changed acquisition
intervention explaining the v1→v2 criterion-suite discrepancy; it cannot establish
that as the unique cause. The offline diff does not identify one narrower causal
component because task semantics, role structure, demonstration, completion
contract, token allocation, corpus, and exposure horizon all changed together,
and Tinker does not expose an immutable digest for the server base weights. No
T2–T7 screening, repair, threshold change, LR extension, cap extension, or
replacement task ran.

### Execution provenance

The 44/60 replay baseline first completed in `runs/m1-diagnostic/journal.jsonl`
and was reused through the immutable request cache by the final recovery run; it
was not resampled. Two local wrapper attempts failed before any training update
completed: first a `TypeError` while adapting already-encoded validation datums,
then a local cache-consumption error while importing the completed baseline. Both
failures remain in the append-only diagnostic journals. The clean final journal
contains 30 completed operations and zero failed operations.

## Offline intervention diff

These interventions are **not the same task**, byte-wise or semantically.

| Property | v1 cycle-1 `f01_n3_left` | v2 T1 `arithmetic_derivations` |
|---|---|---|
| Task semantics | Construct an expression from an unordered set of three integers and a target. Choose/order operators while obeying the forced tree `((number ? number) ? number)`. | Evaluate an already-specified balanced four-operand expression `((a op b) +/- (c op d))`; derive both children and the root. |
| Roles | System + user + assistant | User + assistant; no system message |
| System text | Full task contract plus a worked example | None |
| User text | Only `Numbers: …` and `Target: …` | Full task contract and supplied expression |
| Assistant prefix | `<think>\n\n</think>\n\n` | Identical |
| Demonstration | Repeated worked construction example in every system prompt | None |
| Completion | Two operation equalities, then the constructed expression in `<answer>`; 3 lines total | Left child, right child, and root equalities, then the supplied expression in `<answer>`; 4 lines total |
| Tokenizer | Repository did not pin a revision. The sole local snapshot is `851bf6e…`, and retokenizing with it reproduces every archived per-step token count. | Explicitly pinned to `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Loss | Completion-only, per-example-normalized cross entropy | Identical |
| Batch | 64 | 64 |
| LR / warmup | Nominal `1e-4`; steps 1–10 linearly `1e-5,…,1e-4`; steps 11–20 at `1e-4` | Grid `{1e-5,3e-5,1e-4}`. The `1e-4` branch's first 20 values match v1, but it continues to a 120-update cap. |
| Learning stop / evaluation cadence | Fixed 20-update dose; validation NLL at updates 0 and 20; protected IF at the endpoint | Task gate and protected IF after every update; stop at the first joint competence-plus-damage crossing, otherwise right-censor at update 120; held-out task score at the selected endpoint |
| Adam | Fresh state; β₁ 0.9, β₂ 0.95, ε `1e-12`, weight decay 0, SDK-default gradient clip 0 | Identical |
| Training corpus/order | 1,280 unique rows, used once as 20 contiguous batches; 320 easy, 640 medium, 320 hard; ordered-row SHA-256 `d12631a2…23aa2` | Two disjoint 7,680-row screening realizations, each used as 120 contiguous batches and reused across LRs; screen 1 seed `299145980`, SHA-256 `4b1efd5b17d85130e8c91b6d4a39cefd4dcca547a84b116e6fad9cb34cd6725e`; screen 2 seed `1090867175`, SHA-256 `e359e4325ba8083c8181b879e38bc4f432f3cd23e17aeac05e5ba2c17034de10`; separate 7,680-row reference corpus |
| LoRA | `Qwen/Qwen3.5-4B`; rank 32; seed 1337; attention+MLP trained; unembedding off; exported alpha 32 | Identical |

Each v1 example had more prompt tokens on average but fewer supervised
completion-target tokens on average.

| Corpus | n | Prompt tokens min / mean / median / p95 / max | Completion+EOT tokens min / mean / median / p95 / max | Model-input tokens | Target tokens |
|---|---:|---|---|---:|---:|
| v1 f01 | 1,280 | 162 / 164.5703 / 165 / 166 / 167 | 32 / 38.2398 / 39 / 42 / 45 | 258,317 | 48,947 |
| v2 screen 1 | 7,680 | 92 / 94.6647 / 95 / 96 / 96 | 46 / 59.4357 / 59 / 67 / 72 | 1,175,811 | 456,466 |
| v2 screen 2 | 7,680 | 92 / 94.6674 / 95 / 96 / 96 | 46 / 59.4294 / 59 / 67 / 72 | 1,175,784 | 456,418 |

Even over equal first-20-update exposure, v1 used 258,317 model-input and 48,947
target tokens, while v2 screen 1 used 195,921 model-input and 76,065 target tokens.

## Exact representative datums

For both datums below, let `P` be the listed prompt-token sequence and `C` the
listed completion-token sequence, including final EOT token 248046. The exact
training datum is:

```text
model_input  = (P || C)[:-1]
target_tokens = (P || C)[1:]
```

### v1: first ordered f01 training row

Raw task: numbers `[9,25,13]`, target `47`, expression `((9+13)+25)`.

Exact rendered text:

```text
<|im_start|>system
Example only—do not copy its numbers: 2 + 3 = 5
5 * 4 = 20
<answer>((2+3)*4)</answer>
Now combine the 3 numbers with +, -, *, / and parentheses to reach the target, using each exactly once. Return exactly 3 lines and no other text: 2 short binary-operation equality lines, then one <answer>...</answer> line. Inside <answer>, repeat the fully parenthesized expression using every original number; never put only the numeric target. Use exactly this grouping, with ? replaced by operators: ((number ? number) ? number).<|im_end|>
<|im_start|>user
Numbers: 9 25 13
Target: 47<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Gold completion + EOT:

```text
9 + 13 = 22
(9+13) + 25 = 47
<answer>((9+13)+25)</answer><|im_end|>
```

Prompt token IDs (165):

```text
[248045,8678,198,12929,1132,2218,2887,524,2880,1141,4947,25,220,17,478,220,18,283,220,20,198,20,348,220,19,283,220,17,15,198,27,8944,45871,17,10,18,4653,19,12173,8944,29,198,6820,15491,279,220,18,4947,440,478,11,82956,11439,593,321,71458,310,5372,279,2100,11,1608,1754,6681,2957,13,3301,6681,220,18,4965,321,874,975,1414,25,220,17,2716,7620,63998,21106,4965,11,1179,799,361,8944,29,25573,8944,29,1500,13,26482,361,8944,7813,12771,279,6999,36688,80019,7258,1608,1396,3889,1324,26,2496,2113,1132,279,23311,2100,13,5272,6681,411,47532,11,440,907,12215,539,19024,25,1718,3946,907,1324,8,907,1324,553,248046,198,248045,846,198,26359,25,220,24,220,17,20,220,16,18,198,6196,25,220,19,22,248046,198,248045,74455,198,248068,271,248069,271]
```

Completion token IDs (39):

```text
[24,478,220,16,18,283,220,17,17,198,7,24,10,16,18,8,478,220,17,20,283,220,19,22,198,27,8944,45871,24,10,16,18,7030,17,20,12173,8944,29,248046]
```

Exact 203-position target mask: `0.0 × 164`, then `(1/39) × 39`.

### v2: first arithmetic screen-1 row

Raw task: derive `((8/18)-(24/25))` and its two child values.

Exact rendered text:

```text
<|im_start|>user
Derive the exact rational value of ((8/18)-(24/25)). Return exactly four lines: the left child expression = its reduced value; the right child expression = its reduced value; the original whole expression = its reduced value; then <answer>the original whole expression</answer>. Repeat the unsimplified expressions on the left of each equality. Use integers or reduced fractions; no prose.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Gold completion + EOT:

```text
(8/18) = 4/9
(24/25) = 24/25
((8/18)-(24/25)) = -116/225
<answer>((8/18)-(24/25))</answer><|im_end|>
```

Prompt token IDs (95):

```text
[248045,846,198,21483,520,279,4581,23665,869,314,1718,23,14,16,23,49629,17,19,14,17,20,4430,3301,6681,2943,4965,25,279,2047,1623,7258,283,1141,10723,869,26,279,1245,1623,7258,283,1141,10723,869,26,279,3889,4220,7258,283,1141,10723,869,26,1179,361,8944,29,1719,3889,4220,7258,510,8944,13867,43311,279,6758,71328,22666,383,279,2047,314,1754,21106,13,5272,24959,466,10723,62700,26,874,58655,13,248046,198,248045,74455,198,248068,271,248069,271]
```

Completion token IDs (67):

```text
[7,23,14,16,23,8,283,220,19,14,24,198,7,17,19,14,17,20,8,283,220,17,19,14,17,20,198,1148,23,14,16,23,49629,17,19,14,17,20,578,283,471,16,16,21,14,17,17,20,198,27,8944,45871,23,14,16,23,49629,17,19,14,17,20,578,510,8944,29,248046]
```

Exact 161-position target mask: `0.0 × 94`, then `(1/67) × 67`.

## Replay identity and input verification

- v1 `v1-final` peeled commit: `4ab1d5ecff12bddc85e45ca0348358d3295f47b4`
- archived M2 task-body SHA-256: `972dc54471787613275ccf95cf17fe25d7a3de7ce8dea0cb85c179a5e75b9d33`
- exact ordered f01 train-subset SHA-256: `d12631a20876c4709f27244f0da7b5db07c3336667fb9cdee1026e8190423aa2`
- actual run's `tasks.py` SHA-256: `bea16fc36af7d8fbea162277865fcfda5a17b8865e571217b76f16344766b229`
- all IF rows SHA-256: `a51b4a12dab1103d135ac9fad931b3f0dfab3071dcf119067e93d45d6f2728e1`
- criterion IF rows SHA-256: `5defb4057d3fbc38ae8317f1704e69a6e735f84bd234e8d17b33aca10919c157`
- held-out IF rows SHA-256: `46ffb14c2fec5670ec4342dc3e883cd63b4e69b515b77fbe4bea717967b376c2`
- replay IF realization: archived pair-0 cycle-1 seed `1738054771`, one sample per prompt, temperature 0, 96-token cap, top-p 1, top-k −1, EOT stop
- SDK: Tinker `0.25.0`; tokenizer: Transformers `5.5.4`, revision `851bf6e…`
- Tinker reported both replay runs as uncorrupted LoRA over `Qwen/Qwen3.5-4B`, owner `tml:organization_user:0d41e482-eb21-4e18-894e-f9fd05603039`, rank 32, server maximum context 65,536
- origin run: `2fca9c97-09de-56a9-bc6b-dad09e2a0698:train:0`; learning run: `b4ae52ed-f067-5a28-8395-2f863190b5af:train:0`
- downloaded origin and A adapters both validate alpha 32, scale 1, and all 248 expected attention/MLP module pairs with no unembedding tensors
- replay-origin adapter file SHA-256: `b116a1086cacf54d1e8f145dacd6e3e206f9f83dd767381d808dc84665d2fcfe`, byte-identical to both downloaded v1 and v2 cycle-0 adapters
- replay-A adapter file SHA-256: `6c20842bc33f72165b239c4677a601c8b1450b510e74f424a7220bcf315cd5cb`
- local `result.json` SHA-256: `712c94dde1eb8822858572a0ea434d005bb8a1bedb8c09dd6fc4214b052ac168`
- clean final replay journal SHA-256: `be3f3ec69514a6f6e0f17d80a208e95a99183bab9156ac5d41a57179eec1fe40`

Tinker exposes the base-model alias and owner but not an immutable base-weight
revision or digest. The local adapter identity therefore does not independently
prove equality of historical and current server base weights. It does show an
identical LoRA origin/layout, and the behavioral replay directly shows that the
historical intervention remains capable of causing the historical-scale
criterion-suite IF drop.
