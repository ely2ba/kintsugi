# M1 damage-capacity diagnostic

**Date:** 2026-09-04

**Status:** complete; diagnostic only; no selection or design authority

## Result

Both domains learned in the descriptive sense that NLL improved on both the gate
and disjoint held-out sets. Neither learning intervention moved the protected IF
score into the registered 27–37 damage band at any measured update.

| Slot and domain | Did it learn? | Gate NLL, start → terminal | Held-out NLL, start → terminal | Protected IF, start → terminal (observed range) | Entered 27–37? | First band entry / token exposure |
|---|---|---:|---:|---:|---|---|
| T2 — Icelandic Wikipedia | Yes, descriptively | 2.207482 → 1.985479 (−0.222003; −10.06%) | 2.109094 → 1.867182 (−0.241912; −11.47%) | 44 → 44 (41–45) | No | None; no crossing through update 120 / 832,203 gradient-target tokens |
| T7 — legal-domain corpus | Yes, descriptively | 2.532325 → 2.355522 (−0.176804; −6.98%) | 2.590807 → 2.407688 (−0.183120; −7.07%) | 43 → 41 (40–43) | No | None; no crossing through update 120 / 980,870 gradient-target tokens |

“Did it learn?” is deliberately limited here. Formal registered task competence
cannot be classified because M1 stopped before the candidate-specific reference
sweeps that define competence and acquisition-headroom thresholds. The table only
asserts the directly observed, directionally consistent reduction in gate and
held-out NLL.

## Fixed execution

Each primary candidate used screening realization 1, batch size 16, nominal LR
`1e-4`, the existing 10-update linear warmup, and one complete 120-update pass from
the common frozen cycle-0 checkpoint. Gate NLL and protected IF were measured at
update 0 and after every update. Held-out NLL was measured at update 0 and update
120. No repair, alternative LR, backup candidate, threshold change, cap change, or
selection step ran.

The full 121-point gate-NLL and protected-IF trajectories, including cumulative
token exposure at every update, are in
[`artifacts/m1_damage_capacity_trajectories.csv`](artifacts/m1_damage_capacity_trajectories.csv).

## Execution integrity

Both child journals contain 120 contiguous, valid optimizer updates and complete
gate/IF trajectories. The T7 update-120 checkpoint, gate result, and IF result were
durable before its terminal deterministic held-out forward pass encountered a
connection reset. That read-only measurement was replayed once from the identical
immutable checkpoint under an explicit recovery record; no training operation was
repeated. All parent and child journals end complete with no pending operation.

- T2 screen-1 SHA-256: `f719191df5529af2f16a6b6761287494a332e619d4603b32097b33f19ffb9180`
- T7 screen-1 SHA-256: `b2ddcb5999e8726c5bc4d597d146eacea668cfd677e21a4cca60f912476695a7`
- trajectory CSV SHA-256: `251f6b7509e32b3084c2926673bed2ebab892706dbeac4fc0f6af76583344eb8`
- complete local result SHA-256: `9d8b82c653e2f02074fbab941b14405ced38afa08bf9d665696002266b063c05`

The diagnostic stops here. It does not revise or redesign Kintsugi v2.
