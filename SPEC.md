# SPEC — Kintsugi

## Does repeated behavioral repair preserve a model’s ability to keep learning?

Corrected replication, version 2.0  
Prospective scientific contract

## 0. Scope

Kintsugi v1 is the completed pilot and remains available as `kintsugi-v1`. It motivates this design but contributes no observations to the v2 analysis.

Kintsugi v2 is a new experiment in the canonical `kintsugi` repository. It retains the same model family, protected instruction-following behavior, and on-policy repair method, while correcting the pilot’s three central failures:

1. fixed learning doses did not continue producing comparable damage;
2. related acquisition tasks generated enough transfer to reduce task and probe headroom;
3. one shared task order confounded task identity with lifecycle position.

This document contains only the scientific contract. Execution, billing, checkpoint storage, repository provenance, and crash recovery belong in separate operational documents.

---

# 1. Research question

When a language model is repeatedly taught new capabilities and then repaired through checkpoint-anchored on-policy distillation, does restoring its visible behavior also preserve it as a substrate for future learning?

Visible restoration means that the protected instruction-following score returns.

Functional restoration means that repair does not progressively:

- impair future supervised learning;
- erase capabilities acquired in earlier cycles;
- reduce output diversity;
- increase the cost of future repair;
- or introduce a harmful fixed-versus-rolling anchor tradeoff.

The primary distinction is:

> A checkpoint can look repaired without necessarily being equally capable of learning, retaining knowledge, or being repaired again.

## 1.1 Primary hypotheses

### H1 — Repair debt

Repair restores the protected behavior but progressively worsens at least one hidden functional property: future trainability, accumulated-skill retention, diversity, or maintenance cost.

### H2 — Functional restoration

Repair restores the protected behavior without a practically meaningful deterioration in those functional properties.

### H3 — Plasticity restoration

Repair makes subsequent learning materially faster or more effective than it was immediately before repair.

### H4 — Anchor tradeoff

Fixed-anchor and rolling-anchor repair produce different lifecycle outcomes. For example, a fixed anchor may better preserve the original behavior but interfere with accumulated knowledge, while a rolling anchor may preserve accumulated knowledge but propagate drift from earlier maintenance cycles.

## 1.2 Secondary lifecycle question

Does accumulated post-training history make the model harder or easier to damage?

The experiment therefore records how much new-task training is required to:

1. reach task competence;
2. push protected behavior into the damage band.

This damageability result is secondary and does not substitute for the repair-effect analysis.

---

# 2. Experimental design

## 2.1 Model

- Model: `Qwen/Qwen3.5-4B`
- Renderer: non-thinking Qwen renderer
- Adaptation: rank-32 LoRA
- LoRA alpha: 32
- Train attention modules: yes
- Train MLP modules: yes
- Train unembedding: no
- Common origin: one frozen cycle-0 checkpoint shared by every lineage

Model family, adaptation scope, rank, and renderer remain fixed throughout the study.

## 2.2 Protected behavior

Protected behavior is the model’s native instruction-following ability, measured using the hash-bound v1 suite:

- 60 criterion prompts;
- 30 separate held-out prompts;
- deterministic programmatic checkers;
- no LLM judge.

The criterion suite determines learning damage and repair stopping. The held-out suite never influences either stopping rule.

Let \(S_0\) be the cycle-0 criterion score.

Define:

\[
D_{\mathrm{low}}=\lceil 0.60S_0\rceil
\]

\[
D_{\mathrm{high}}=\lfloor 0.85S_0\rfloor
\]

\[
R=\lceil 0.95S_0\rceil
\]

where:

- \([D_{\mathrm{low}},D_{\mathrm{high}}]\) is the inclusive valid damage band;
- \(R\) is the visible-recovery target.

If \(S_0=43\), the valid damage band is 26–36 and the repair target is 41.

## 2.3 Arms

Each task order has three lineages.

### Learn-only

The model learns every task but receives no behavioral repair. This is the unmaintained lifecycle control.

### Fixed-anchor repair

After every learning phase, repair distills toward the original cycle-0 checkpoint.

### Rolling-anchor repair

Cycle 1 also uses cycle 0 as teacher. From cycle 2 onward, repair distills toward the preceding post-repair checkpoint.

## 2.4 Design size

- Seven heterogeneous task types
- Seven lifecycle cycles
- Four counterbalanced task orders
- Three arms per order
- Twelve lineages total
- Eight repair lineages
- Four learn-only lineages
- Eighty-four lineage-cycles
- Fifty-six scheduled repair opportunities

The four orders are controlled history conditions, not four random training seeds from a wider population. Claims are therefore conditional on the fixed model, tasks, data, and random schedules.

---

# 3. Learning tasks

The final curriculum contains exactly one selected task from each of seven slots.

| Slot | Primary candidate | Predeclared backup |
|---|---|---|
| T1 — symbolic derivation | Exact arithmetic-expression derivations | Symbolic equation derivations |
| T2 — low-resource language modelling | Icelandic Wikipedia | Basque Wikipedia |
| T3 — executable structured generation | Natural-language-to-SQL over synthetic SQLite databases | Spreadsheet-formula synthesis |
| T4 — constrained extraction | Strict JSON extraction from synthetic records | Strict XML extraction |
| T5 — number representation | Base-conversion and number-system problems | Modular numeral transformations |
| T6 — symbolic transduction | String-transformation programs | Finite-state rewrite tasks |
| T7 — natural-domain language modelling | Public legal-domain text | Biomedical abstracts |

Every task has disjoint:

- reference-calibration data;
- two screening realizations;
- main-run training data;
- gate data;
- held-out evaluation data.

Synthetic tasks use semantic disjointness. Natural-language tasks use document-level disjointness.

A backup is screened only when its primary candidate fails. If neither candidate in any slot passes M1, v2 stops. The task count and cycle count are not reduced.

---

# 4. Task orders

The four orders are:

| Order | Sequence |
|---|---|
| O1 | T1 → T2 → T3 → T4 → T5 → T6 → T7 |
| O2 | T2 → T4 → T1 → T6 → T3 → T7 → T5 |
| O3 | T5 → T3 → T6 → T1 → T7 → T4 → T2 |
| O4 | T7 → T6 → T2 → T5 → T4 → T3 → T1 |

Position matrix:

| Task | O1 | O2 | O3 | O4 |
|---|---:|---:|---:|---:|
| T1 | 1 | 3 | 4 | 7 |
| T2 | 2 | 1 | 7 | 3 |
| T3 | 3 | 5 | 2 | 6 |
| T4 | 4 | 2 | 6 | 5 |
| T5 | 5 | 7 | 1 | 4 |
| T6 | 6 | 4 | 3 | 2 |
| T7 | 7 | 6 | 5 | 1 |

Every task therefore appears:

- once in an early position, 1–2;
- at least once in a middle position, 3–5;
- once in a late position, 6–7;
- in four distinct positions.

Across the four orders, all 24 directed task adjacencies are unique. No immediate predecessor→task pair is repeated.

This does not constitute complete carryover balance: only 24 of the 42 possible directed task adjacencies occur. The experiment controls lifecycle depth and avoids repeated local adjacency, but it does not estimate every possible sequence-context interaction.

## 4.1 Common randomness

Whenever the same task appears in different orders, it uses the same:

- training corpus;
- example order;
- gate items;
- held-out items;
- task-specific optimizer recipe;
- evaluation seeds.

Each task also has one frozen repair-prompt and rollout-seed schedule shared across orders. Fixed and rolling lineages within an order receive identical repair randomness.

This isolates accumulated history as far as practical but means the results are conditional on the frozen task and repair schedules. The design does not estimate variance across training or repair seeds.

---

# 5. Task competence and acquisition headroom

All task scores are oriented so that larger is better:

\[
Q=
\begin{cases}
\text{accuracy or verifier success}, & \text{exact tasks}\\
-\text{validation NLL}, & \text{loss-based tasks}
\end{cases}
\]

For task \(j\), M1 produces:

- \(Q^{g}_{j,0}\): cycle-0 gate score;
- \(Q^{g}_{j,\mathrm{ref}}\): best registered gate score in the extended reference sweep;
- \(Q^{h}_{j,0}\): cycle-0 held-out score;
- \(Q^{h}_{j,\mathrm{ref}}\): corresponding held-out reference score.

The gate competence threshold is:

\[
C^{g}_j
=
Q^{g}_{j,0}
+
0.70
\left(
Q^{g}_{j,\mathrm{ref}}-Q^{g}_{j,0}
\right)
\]

The held-out competence floor is:

\[
C^{h}_j
=
Q^{h}_{j,0}
+
0.50
\left(
Q^{h}_{j,\mathrm{ref}}-Q^{h}_{j,0}
\right)
\]

Define a minimum acquisition movement:

\[
M_j
=
\max
\left[
5\sigma_{Q,j},
\,
0.15
\left(
Q^{g}_{j,\mathrm{ref}}-Q^{g}_{j,0}
\right)
\right]
\]

where \(\sigma_{Q,j}\) is the M1 measurement-noise estimate for that task’s gate metric.

A valid acquisition event requires both:

\[
C^{g}_j-Q^{g}_{j,\mathrm{start}}\ge M_j
\]

and

\[
Q^{g}_{j,A}-Q^{g}_{j,\mathrm{start}}\ge M_j
\]

This prevents a task that has already been largely acquired through transfer from being treated as a genuinely new learning event.

---

# 6. Learning intervention

## 6.1 Recipe calibration

For each task candidate, M1 evaluates a predeclared learning-rate grid:

\[
\{10^{-5},\,3\times10^{-5},\,10^{-4}\}
\]

Batch size is fixed by task type before outcome inspection from:

\[
\{16,32,64\}
\]

using sequence length and target-token volume.

A recipe qualifies only if the same learning rate and batch size produce a valid acquisition-and-damage event on two disjoint calibration realizations and both events can subsequently be repaired.

Among qualifying recipes, select the one with the lowest median gradient-bearing target-token dose required to reach valid checkpoint A. Exact ties use the lower learning rate.

The main-run learning cap is 120 updates per task.

## 6.2 Start-of-cycle measurements

Before each learning phase, record:

- protected criterion score \(S_{\mathrm{start}}\);
- task gate score \(Q^{g}_{j,\mathrm{start}}\);
- task held-out score;
- both fixed-probe starting losses.

A primary-valid cycle must begin with:

\[
S_{\mathrm{start}}\ge R
\]

and sufficient task-acquisition headroom as defined in Section 5.

If the model begins below \(R\), the cycle continues but is labelled `unrestored_start` and cannot enter the primary matched-damage analysis.

If task headroom is insufficient, the cycle continues but is labelled `already_competent`.

## 6.3 Damage-to-criterion training

Training proceeds one optimizer update at a time with a fresh optimizer.

After every update:

- evaluate the task gate;
- evaluate the 60-prompt protected criterion suite;
- record gradient-bearing target tokens and total training tokens.

Stop at the first real checkpoint satisfying both:

\[
Q^{g}_j\ge C^{g}_j
\]

and

\[
D_{\mathrm{low}}
\le
S_{\mathrm{IF}}
\le
D_{\mathrm{high}}
\]

This checkpoint is \(A_k\).

After selection, evaluate the task held-out set. The acquisition generalizes only if:

\[
Q^{h}_{j,A}\ge C^{h}_j
\]

## 6.4 Learning-phase classifications

If no valid A checkpoint exists by the cap, save the terminal checkpoint and assign one mutually exclusive primary classification.

### `undamageable`

Task competence was reached, but protected behavior never fell to or below \(D_{\mathrm{high}}\) after competence.

### `band_overshoot`

Competence was reached, but the protected score crossed from above \(D_{\mathrm{high}}\) to below \(D_{\mathrm{low}}\) without an observed in-band checkpoint.

### `damage_before_competence`

Protected behavior reached or passed below the damage band before competence, and no later checkpoint satisfied both conditions.

### `competence_unmet`

The task gate threshold was not reached.

### `heldout_competence_fail`

The selected checkpoint passed the gate threshold but failed the held-out competence floor.

### `already_competent`

The cycle began without the minimum registered acquisition headroom.

### `unrestored_start`

The cycle began below the repair target \(R\).

### `mixed_gate_failure`

No preceding category applies uniquely.

Every classification remains part of the lifecycle record. Nothing is rerun or retuned because a cycle receives an unfavorable label.

Cycles with actual damage below \(R\), including overshoots, proceed to repair so that the lineage remains a real lifecycle. Only valid in-band cycles enter the primary matched-damage cohort.

## 6.5 Exposure measurements

For competence and damage-band entry, report:

- optimizer updates;
- gradient-bearing target tokens;
- total train-priced tokens;
- estimated cost;
- wall-clock time.

Gradient-bearing target tokens are the primary cross-task exposure unit. Update count remains a task-internal operational quantity.

---

# 7. Learn-only control

The learn-only lineage uses the same task data, recipe, and per-update evaluations as the repair lineages.

It stops at the first checkpoint satisfying:

\[
Q^{g}_j\ge C^{g}_j
\]

and the minimum acquisition-movement requirement. It ignores the protected damage band for stopping.

The control records:

- protected score after every update;
- first competence crossing;
- first entry into the damage band, if any;
- first crossing below the band, if any;
- task and protected scores at stopping;
- competence and damage doses.

It receives no repair. Its one physical checkpoint may be referenced as both A and B for bookkeeping, but it supplies no repair-effect observation.

---

# 8. Repair intervention

## 8.1 Repair recipe

Every repair uses:

- on-policy reverse-KL distillation;
- the frozen 2,000-prompt public repair pool;
- 64 prompt groups per update;
- four student rollouts per prompt;
- temperature 1;
- top-p 1;
- top-k −1;
- 4,096-token completion cap;
- fresh Adam optimizer;
- learning rate \(10^{-4}\);
- ten-update linear warmup;
- criterion checks every five repair updates;
- 150-update hard cap.

## 8.2 Anchors

- Fixed arm: teacher is always cycle 0.
- Rolling arm: teacher is the preceding post-repair checkpoint.
- Cycle 1 uses cycle 0 in both arms.

## 8.3 No-repair-required cycles

If checkpoint A already satisfies:

\[
S_A\ge R
\]

repair performs zero updates.

The cycle is labelled `no_repair_required`. Its numerical A–B identity difference is zero, but it supplies no observation of the effect of repair because no intervention occurred.

Such cycles:

- remain zero-cost observations in literal lifecycle-maintenance analysis;
- do not enter repair-effect statistics;
- do not count as evidence that repair is harmless.

## 8.4 Repair stopping

Otherwise, repair stops at:

1. the first scheduled check with score at least \(R\); or
2. the 150-update cap.

The resulting checkpoint is B.

A cap-bound repair is labelled `repair_failure`, but B is still measured. Repair success is an outcome, not a condition for inclusion in the repair-policy analysis.

Record the achieved criterion score so that overshoot beyond \(R\) can be assessed.

---

# 9. Fixed future-learning probes

V2 uses two fixed probes whose task types never appear in the acquisition curriculum.

## 9.1 Structured-probe candidates

1. graph-path or route construction with exact verification;
2. calendar and date arithmetic;
3. unit-conversion word problems.

## 9.2 Language-probe candidates

1. Vietnamese Wikipedia;
2. Indonesian Wikipedia;
3. Finnish Wikipedia.

Exact corpora, splits, and evaluation sets are frozen before M1.

## 9.3 Candidate learning rates and budgets

Probe learning-rate grid:

\[
\{10^{-5},\,3\times10^{-5},\,10^{-4}\}
\]

Standard probe budgets:

- structured probe: 32 updates, evaluated every four updates;
- language probe: 245 updates, evaluated every 25 updates.

For every candidate-LR pair, the cycle-0 reference sweep uses twice the standard budget.

The reference target is:

\[
L^*_{\mathrm{ref}}
=
\min_t L_{\mathrm{cycle0}}(t)
\]

over registered evaluation points in that extended sweep.

This is a fixed-budget reference target, not an asymptotic loss floor.

## 9.4 M1 state panel

Each candidate probe is tested on:

- cycle 0;
- both A and B checkpoints from every selected task’s two screening realizations;
- every A and B checkpoint in the persistence experiment;
- both fixed and rolling terminal persistence states.

For a checkpoint:

\[
H=L(0)-L^*_{\mathrm{ref}}
\]

The candidate must satisfy, at every M1 state:

\[
H\ge 5\sigma_{\mathrm{eval}}
\]

and under the standard probe budget:

\[
P(t_{\mathrm{final}})
=
\frac{L(0)-L(t_{\mathrm{final}})}
{L(0)-L^*_{\mathrm{ref}}}
\ge 0.60
\]

Its \(t_{50}\) crossing must be observed and bracketed between registered evaluation points.

To avoid floor and ceiling compression, \(t_{50}\) must occur between 20% and 80% of the standard probe budget at every M1 state.

## 9.5 Companion absolute-reduction clock

For each candidate-LR pair, define:

\[
\delta L
=
0.25
\times
\min_{\text{M1 states}} H
\]

Every M1 state must satisfy:

\[
H\ge \delta L+5\sigma_{\mathrm{eval}}
\]

and must reach the \(\delta L\) target under the standard budget.

## 9.6 Probe selection

For each probe class:

1. discard every candidate-LR pair failing any headroom or dynamic-coverage rule;
2. within each candidate, retain the lowest learning rate that passes;
3. select the candidate maximizing minimum headroom across the M1 state panel;
4. ties go to the candidate whose median \(t_{50}\) is closest to half the standard budget;
5. remaining ties follow the listed candidate order.

If no candidate in either probe class passes, v2 stops at M1.

## 9.7 Probe outputs

At every physical A and B checkpoint, each probe starts from a fresh optimizer with the same frozen data order.

Report:

- initial loss \(L_0\);
- best loss reduction \(\Delta L\);
- final loss;
- validation-loss AUC;
- normalized progress \(P(t)\);
- \(t_{50}\);
- \(t_\Delta\).

Definitions:

\[
P(t)
=
\frac{L(0)-L(t)}
{L(0)-L^*_{\mathrm{ref}}}
\]

\(t_{50}\) is the interpolated update at which \(P(t)\ge0.5\).

\(t_\Delta\) is the interpolated update at which:

\[
L(t)\le L(0)-\delta L
\]

Interpolation is allowed only between two registered points that bracket the first crossing.

Undefined and right-censored clocks remain unavailable. They are never replaced by zero, the budget limit, or an extrapolated value.

---

# 10. Accumulated-skill retention

Every acquisition task retains its native held-out metric.

For aggregation, use the oriented score \(Q\), with larger always better.

For task \(j\):

- \(Q_{j,0}\): cycle-0 held-out score;
- \(Q_{j,A_j}\): held-out score immediately after acquisition;
- \(Q_{j,c}\): held-out score at a later checkpoint.

Define normalized retention:

\[
R_{j,c}
=
\frac{Q_{j,c}-Q_{j,0}}
{Q_{j,A_j}-Q_{j,0}}
\]

Interpretation:

- \(R=1\): post-acquisition improvement fully retained;
- \(R=0\): returned to cycle-0 performance;
- \(R<0\): worse than cycle 0;
- \(R>1\): improved beyond the original acquired level.

Scores are not clipped.

Normalized retention is defined only when the acquisition denominator exceeds both:

- five times task-metric noise;
- the registered minimum acquisition movement \(M_j\).

At every A and B checkpoint, report:

- native metrics for every acquired task;
- current-task retention;
- mean prior-task normalized retention;
- mean all-task normalized retention.

The primary repair-retention quantity is the A-to-B change in mean prior-task normalized retention.

---

# 11. Diversity

Diversity is measured on a separate multi-solution, exactly verifiable construction domain not used in acquisition or either probe.

M1 selects one panel from a predeclared candidate set such as:

- graph coloring;
- constrained route construction;
- set-partition construction.

A panel qualifies only if:

- at least 80% of items admit at least four verifier-distinct valid solution families;
- cycle-0 pass@1 is above 0.05 and below 0.80;
- pass@8 exceeds pass@1 by at least 0.05;
- completion length remains safely below the sampling cap;
- the panel is semantically disjoint from every learn and probe dataset.

Report:

- pass@1;
- pass@8;
- pass@8 minus pass@1;
- number of unique valid outputs;
- number of verified strategy families;
- strategy-family concentration;
- sampled-token surprisal;
- output length and truncation.

The pass@8–pass@1 quantity is treated as a sampling-coverage gap, not a complete measure of behavioral diversity.

---

# 12. Drift and lifecycle cost

At every physical A and B checkpoint, report:

- forward KL to cycle 0 on a frozen trajectory set;
- scaled LoRA \(\Delta W\) Frobenius norm;
- effective rank;
- stable rank;
- cosine to the preceding physical checkpoint.

These are descriptive mechanism measurements. They cannot replace behavioral outcomes or establish causality by themselves.

Lifecycle cost includes:

- target tokens to task competence;
- target tokens to valid damage;
- repair updates;
- repair tokens;
- estimated monetary cost;
- wall-clock time.

---

# 13. M1 launch gates

M1 calibrates whether the experiment will remain identifiable, not merely whether one initial cycle works.

## 13.1 Task screening

Each of the seven selected task candidates must:

1. have meaningful cycle-0 acquisition headroom;
2. reach its gate competence threshold;
3. exceed its minimum acquisition-movement requirement;
4. land inside the complete damage band;
5. satisfy held-out competence;
6. do so on two disjoint calibration realizations using the same recipe;
7. repair successfully to \(R\) on both realizations.

If a primary candidate fails, its registered backup is tested. If both fail in any slot, v2 stops.

## 13.2 Persistence experiment

After task screening, run a three-cycle mini-lifecycle using:

\[
T3 \rightarrow T2 \rightarrow T6
\]

Cycle 1 begins from cycle 0 and is common to both anchor policies:

1. acquire T3;
2. enter the valid damage band;
3. repair using the cycle-0 teacher;
4. save common \(B_1\).

Then branch into fixed and rolling lineages.

Both branches perform:

- cycle 2 on T2;
- cycle 3 on T6.

The persistence gate requires all five repair opportunities to satisfy:

- restored start \(S_{\mathrm{start}}\ge R\);
- sufficient task-acquisition headroom;
- valid task competence and movement;
- held-out competence;
- A inside the inclusive damage band;
- successful repair to \(R\);
- valid structured- and language-probe \(t_{50}\) and \(t_\Delta\).

The five required events are:

- one common cycle-1 event;
- two fixed-anchor events;
- two rolling-anchor events.

Any failure stops the project at M1. No task, threshold, probe, or recipe is improvised after the failure.

## 13.3 Additional M1 requirements

Before M2, M1 must also establish:

- cycle-0 noise estimates for stochastic measurements;
- paired probe-noise bounds;
- a valid diversity panel;
- measured task-specific token exposure;
- measured repair cost;
- the complete twelve-lineage cost projection.

M2 launches only when the complete twelve-lineage design is funded. No order, arm, task, cycle, or primary measurement is dropped for budget reasons.

---

# 14. Primary eligibility

A repair-arm cycle is prospectively eligible for the primary repair-policy analysis when, before repair begins:

1. it started with protected behavior restored:

\[
S_{\mathrm{start}}\ge R
\]

2. it had sufficient task-acquisition headroom;
3. the task gate threshold was reached;
4. minimum acquisition movement was reached;
5. held-out competence was reached;
6. checkpoint A lies inside the inclusive damage band;
7. A lies below the repair target and therefore requires an actual repair.

Repair success is not an eligibility condition. A repair that reaches the cap without recovering remains part of the primary repair-policy result.

Axis-specific measurement availability is reported separately. A cycle can remain eligible for visible restoration or retention even when one probe clock is unavailable.

Flagged, overshoot, no-op, and control cycles remain in lifecycle and validity tables but do not enter the matched-damage primary cohort.

---

# 15. Primary analyses

## 15.1 Visible restoration

Across all prospectively eligible cycles, report:

- probability of reaching \(R\);
- updates and tokens to criterion;
- achieved stopping score;
- held-out instruction-following change;
- repair failures.

This analysis includes failed repairs.

## 15.2 Repair-policy effect on future trainability

For each fixed probe, compare B with A:

\[
\Delta t_{50}=t_{50,B}-t_{50,A}
\]

\[
\Delta t_{\Delta}=t_{\Delta,B}-t_{\Delta,A}
\]

Positive values mean slower learning after repair.

The primary repair-policy analysis uses B at either:

- the first successful criterion crossing; or
- the 150-step repair cap.

This estimates the result of applying the registered repair policy, not merely the result among successful repairs.

## 15.3 Criterion-matched functional restoration

A separate analysis includes only cycles where visible repair reached \(R\).

It asks:

> Conditional on the benchmark saying repair worked, what happened to future trainability, prior skills, and diversity?

This analysis is explicitly conditional on successful visible restoration and is not substituted for the all-eligible repair-policy result.

## 15.4 Strict whole-lineage analysis

A secondary strict-robustness analysis includes only lineages with seven primary-eligible cycles.

Failure of one cycle does not erase the per-cycle primary analysis. The strict analysis instead asks whether conclusions survive under uninterrupted seven-cycle validity.

---

# 16. Coverage requirements

A single global repair-effect headline requires:

- at least 42 of 56 scheduled repair-arm cycles to be primary eligible;
- at least six of eight scheduled repair-arm observations to be eligible for every task type;
- all four orders represented in each repair arm;
- at least 80% of eligible cycles to have both valid clocks for each probe used in a trainability claim.

If these requirements fail, report task-specific and validity results, but do not issue one pooled statement about repeated repair.

A manipulation-coverage failure is itself a design result. It is not repaired by silently restricting the analysis to whichever tasks happened to work.

---

# 17. Statistical summaries and claim rules

## 17.1 Task- and position-adjusted summaries

Per-cycle analyses include:

- task fixed effects;
- lifecycle position;
- repair arm;
- arm-by-position interaction;
- order as the repeated design block.

Fixed-versus-rolling comparisons are paired within order.

The four order conditions do not support population-level seed inference. The paper reports:

- within-order effects;
- consistency across orders;
- task-adjusted descriptive aggregates;
- measurement-level uncertainty where available.

Cycles, prompts, and sampled completions are never treated as independent training seeds.

## 17.2 Trainability claims

A domain-specific trainability-debt or restoration claim requires:

1. \(t_{50}\) and \(t_\Delta\) effects agree in direction;
2. both exceed their M1 paired-noise bounds;
3. relative \(t_{50}\) change exceeds 10%;
4. at least three of four order-level summaries have the same direction;
5. the task-adjusted aggregate has that direction;
6. coverage requirements are met.

A global trainability claim requires the structured and language probes to agree, or at minimum requires that the second probe show no opposite practically meaningful effect.

If the probes disagree, the result is reported as domain-dependent trainability rather than collapsed into one plasticity number.

## 17.3 Retention claims

The primary retention effect is the A-to-B change in mean prior-task normalized retention.

A repair-induced retention effect requires:

- absolute change greater than 0.05;
- effect above the M1 noise bound;
- same direction in at least three of four valid orders.

Native task metrics remain visible alongside normalized retention.

## 17.4 Diversity claims

A diversity effect requires:

\[
\left|
\Delta(\mathrm{pass@8}-\mathrm{pass@1})
\right|
>0.03
\]

plus:

- M1 noise-bound exceedance;
- same direction in at least three of four orders;
- consistent evidence from strategy-family coverage or concentration.

Sampled-token surprisal alone cannot establish diversity collapse.

---

# 18. Maintenance and damageability

Report three distinct repair-cost quantities.

## 18.1 Literal lifecycle burden

Every cycle is included. No-repair-required events contribute zero repair cost.

This measures the operational cost of following the complete maintenance policy.

## 18.2 Repair-required burden

Includes every cycle where positive repair work occurred.

This measures cost conditional on the model requiring intervention.

## 18.3 Matched-damage repair cost

Includes only primary-eligible in-band cycles.

This is the controlled comparison of like-for-like damage.

Where identifiable, model mean repair steps using zero-compatible Poisson pseudo-maximum likelihood with:

- task fixed effects;
- lifecycle position;
- repair arm.

No pseudocounts are used. Raw repair-step sequences are always reported. Failed repairs are right-censored for time-to-success and remain visible as failures.

For acquisition, model task-adjusted trends in:

- gradient-bearing target tokens to competence;
- gradient-bearing target tokens to valid damage.

A cap-bound acquisition is right-censored, not treated as if the cap were its true required dose.

---

# 19. Interpretation

| Observed pattern | Interpretation |
|---|---|
| Protected behavior recovers, while trainability, retention, or diversity deteriorates | Repair debt: visible restoration is not functional restoration |
| Repair produces materially faster future learning | Repair acts as plasticity restoration in this regime |
| Fixed and rolling anchors differ consistently | Anchor choice creates a lifecycle stability–memory tradeoff |
| Repair cost rises for comparable in-band damage | Maintenance debt |
| Damage requires progressively more target tokens | Accumulated history makes the model harder to disrupt |
| Damage requires progressively fewer target tokens | Accumulated history makes the model increasingly fragile |
| Too few cycles produce valid damage or new-task acquisition | Manipulation failure; no global repair-composition claim |
| Probe clocks lose coverage despite M1 validation | Instrument failure; no affected trainability claim |
| Effects remain inside practical margins | No practically large effect discovered; not proof of equivalence |

This twelve-lineage experiment is a discovery study. It cannot issue a formal universal or equivalence-based “Composes” verdict.

A flat result must be stated as:

> No registered practically large repair-debt or restoration effect was discovered under this model, task set, adapter configuration, and lifecycle design.

---

# 20. Primary endpoints

## 20.1 Co-primary trainability endpoints

For both the structured and language probes:

- relative A-to-B change in \(t_{50}\);
- A-to-B change in \(t_\Delta\) as the required companion.

## 20.2 Co-primary visible-repair endpoints

- criterion-recovery success;
- updates and tokens to criterion;
- held-out instruction-following change;
- stopping-score dispersion.

## 20.3 Secondary endpoints

- prior-task normalized-retention change;
- current-task retention;
- diversity and strategy coverage;
- damage dose;
- competence dose;
- literal and matched-damage repair cost;
- probe \(L_0\), \(\Delta L\), final loss, and AUC;
- KL to cycle 0;
- LoRA norm, rank, and cosine;
- fixed-versus-rolling contrasts;
- strict seven-valid-cycle lineage results.

## 20.4 Practical margins

| Axis | Practical margin |
|---|---:|
| Trainability | ±10% relative \(t_{50}\) |
| Prior-task normalized retention | ±0.05 |
| Sampling-coverage gap | ±0.03 |
| Seven-cycle comparable-damage repair-cost growth | +30% |
| Visible restoration | Must reach \(R\); held-out behavior reported separately |

These margins determine scientific importance. They do not create an equivalence claim at four order conditions.

---

# 21. Out of scope

The following are not part of v2:

- forward-KL repair;
- EMA or blended teachers;
- different LoRA ranks;
- full-weight training;
- additional model families;
- RL acquisition phases;
- thinking-mode training;
- dynamic task selection;
- result-triggered extra lineages;
- gauge-rescaling interventions;
- LLM-judged evaluation;
- new mechanistic probes beyond the registered drift panel.

Any such experiment requires a separate contract after v2.

---

# 22. Registration boundary

This specification is published before any paid v2 call.

M1 may fill in only the quantities explicitly defined as calibration outputs:

- selected primary or backup tasks;
- task recipes;
- competence thresholds;
- noise estimates;
- selected fixed probes and their learning rates;
- reference targets;
- \(\delta L\) values;
- selected diversity panel;
- measured cost projection.

Those values are frozen before M2.

After M2 begins, the interventions, validity rules, primary endpoints, coverage requirements, practical margins, and claim rules do not change in response to outcomes. Any unavoidable deviation is reported as a deviation rather than rewritten as part of the original design.
