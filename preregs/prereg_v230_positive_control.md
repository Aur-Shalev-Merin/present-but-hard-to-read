# v230 pre-registration: does the protocol say yes to an unwritten, used intermediate?

Workstream paper3 (Ouro note). Claimed 2026-10-28. Design and pilot log: `docs/v230_design.md` (pilots 1-5).
Frozen 2026-11-01, before any protocol statistic was computed on a v230 model.

## 1. Why

Two outside reviews (2026-10-28) gave the same blocker: every positive control of the protocol is a value the
model is about to write (v221 inb, v224 inA) or has written. Its power for its stated purpose, a value computed but
never written, is unshown, and the Ouro "not used" and Kudo readings rest on a detector that has never said yes there.

v230 builds that case from scratch. Task: `[pad] x op z ; [k fillers] w =` -> y = (T[(x op z) mod 23] + w) mod 23,
T a fixed permutation, so s = x op z is never written and the answer does not fix it. Ground truth comes from DAS, a
causal method independent of the protocol. This study runs the protocol, with its rules fixed, at cells where DAS says
s is computed and used and at cells where it says not.

## 2. Disclosures (what was seen before the freeze)

1. **Ground truth and its thresholds come from the pilots** (section 3). The thresholds (0.70, 0.15) were chosen after
   seeing the DAS numbers. Only DAS-side quantities were seen.
2. **One quantity close to the protocol was seen:** s's class-mean energy at ';', layer 4, seed 0, unregressed (0.41 of
   the state's variance; pilot 2's guard diagnostic). No threshold here is set from it.
3. **Guarded DAS was seen:** with the linear copy span of x, z, op removed, DAS on s at ';' falls to 0.16-0.42 in seeds
   1, 2, 4 (0.69-0.75 in seed 0). This bears on the clamp's copy-orthogonalised condition (cBo, which removes z's code).
4. **The clamp departs from v215 in three ways**, each fixed after a pre-freeze known-answer check on w at its own token
   (section 6), never on s: class codes are taken jointly (least squares on one-hot s, x, z, w, op) rather than as
   separate class means; each subspace keeps only the directions above a shuffled-label null; the random subspace has
   the dimension of the subspace it controls.
5. **The share swap is v224's statistic ported** to modulus 23, the copies x, z, w and the answer, and this model's
   layers. The port reproduces v224's numbers exactly on v224's own design (section 6).
6. **No share statistic and no clamp on s** has been computed on any v230 model state.

## 3. Ground truth (fixed now, from pilots 1-5)

DAS (`v230_pilot.das`): learned m-dim subspace (m 32; seed 0's non-';' cells at m 16 from pilots 1-2), 2,000 steps,
held-out interchange accuracy (IIA). S: the source's s, target T[s_src] + w_base. X / Z: one operand takes the source's
value, target T[x_src op z_base] + w_base; if ';' held the operands for later reads, X or Z would score.

Rule at a (k, position), over layers 1-6:
- **POSITIVE** (s computed and used there): unguarded S IIA >= 0.70 at some layer, and X and Z IIA <= 0.15 at every
  layer.
- **NEGATIVE** (s not used from there): unguarded S IIA <= 0.15 at every layer.
- **INDETERMINATE** otherwise; excluded from scoring.

DAS known answers: W, X and Z at their own tokens, layer 0, IIA 0.99-1.00 in all five seeds (random <= 0.01).

| seed | ';' (k 6 pos 3 / k 0 pos 9): best S, layer | X, Z max | other positions: S max | label |
|---|---|---|---|---|
| 0 | 0.91 / 0.92 (L4) | 0.08 | 0.03 (filler, w); '=' 0.13-0.18 (L5) | ';' POSITIVE; fillers, w NEGATIVE; '=' INDETERMINATE |
| 1 | 0.78 / 0.94 (L4) | 0.05 | 0.07 | ';' POSITIVE; rest NEGATIVE |
| 2 | 0.98 / 0.98 (L5) | 0.05 | 0.04 | ';' POSITIVE; rest NEGATIVE |
| 3 | 0.91 / 0.92 (L3) | 0.05 | 0.04 | ';' POSITIVE; rest NEGATIVE |
| 4 | 0.87 / 0.87 (L3) | 0.05 | 0.06 | ';' POSITIVE; rest NEGATIVE |

Positions (k 6): ';' (3), a middle filler (6), the last filler (9), w (10), '=' (11); (k 0): ';' (9), w, '='. At k 6,
';' is 8 tokens before '='; at k 0, 2 tokens.

## 4. Models

- **pos**, seeds 0-4: `v230_pilot.train('std', 'pos', seed)` on the pilot schedules (STEPS 30k; seed 2 60k). Training is
  bitwise deterministic on Kaggle's T4 (seed 0's checkpoint identical across pilots 1-5, seeds 1 and 2 across two
  runs each). The run retrains each model and **stops if its sha256 differs** from the checkpoint DAS measured:
  0 423563683be2c8f9, 1 0a46ffeb90ae782b, 2 39aac1088343cf2d, 3 fe7a13c4e6aa5fd9, 4 017e54b930608bae.
- **neg** (negative control), seeds 0-4: y = (T1[x] + T2[z] op w) mod 23, same tokens; op matters but x op z is never
  needed. 30k schedule. Gate: held-out accuracy >= 0.95, else that model is UNSCORABLE (reported, not replaced).

## 5. The protocol arm (`v230_protocol.py`, frozen)

**Items** per k in {6, 0}: 100 families x 60 (x, z) pairs (no repeats within a family, seed 2300 + k) x both
operations; twins share x, z and w; w drawn per pair. 12,000 items. The same items for every model.

**States:** residual after blocks 1-5 at each position (**BAND**: every block output that a later position can read; layer
0 is the token embedding, constant at ';', fillers and '=', and layer 6 at an earlier position is read by nothing).

**Share swap** (v224 rules): per operation, position and BAND layer, family-demeaned states; copies (one-hot x, z, w and
the answer y, family-demeaned) regressed out on training folds (5 by family, v178's seed); class means of each label on
the training residuals; share = 1 - sum held-out error / sum held-out residual, averaged over BAND. Labels S = (x + z)
mod 23, M = (x - z) mod 23, R0-R3 (S or M of a shuffled pair, seed 2231). N = (share_S plus + share_M minus) / 2,
D+ = share_S - share_M on plus, D- = share_M - share_S on minus, I = (D+ + D-) / 2. Cluster bootstrap over families,
10,000 resamples, seed 2241, 97.5% CIs.
- UNSCORABLE: the model is right on < 80% of a version's items. NO-SIGNAL: N's CI lower <= 0.
- **FOLLOWS:** D+ > 0, D- > 0, I's CI lower > 0 and I >= BAR x N, **BAR = max(0.30, TAU230)** (section 6).
- DOES-NOT-FOLLOW: I's CI upper < 0.10 x N. UNCLEAR: otherwise.

**Clamp** (v215 / v221 rules, with disclosure 4): at the position, per BAND layer, codes fitted on the other half of the
families (split seed 2151) by least squares of family-demeaned states on one-hot s, x, z, w, op (family-demeaned). U =
s's code, D = z's code, each keeping the directions whose singular value exceeds the largest from 5 shuffled-label fits
(seed 2154); Uo = U orthogonalised against D; R random with Uo's dimension (seed 2152). Pairs: 1,000 bases among the
items the model gets right (seed 2151); the source has the same op, x and w and a different z, so s and the answer
change and w does not. Conditions at the token, at every BAND layer (h <- h - A A^T (h - h_src)): cB (U), cBo (Uo),
cD (D), cR (R); none; full (the whole state at every block output 1-6). Outcome: the argmax at '='; hit = the
source's answer. Statistic hit(cBo) - hit(cR), cluster bootstrap over the base's family, 10,000 resamples, seed 2153,
97.5% CI.
- UNSCORABLE: hit(full) < 0.30. **USED:** difference >= 0.10 and CI lower > 0. NOT-USED: CI upper < 0.02. UNCLEAR.

Both arms run at every position, both k, every model.

## 6. Pre-freeze record

- **Port check** (`python v230_protocol.py --selftest`): on v224's b design (computed world, c 0.5) the ported statistic
  equals v224_share's exactly (N 0.149, I +0.086 [+0.085, +0.086], I / N 0.58, FOLLOWS).
- **Synthetic worlds on this design** (BAR 0.30 for the test): a code of s at 0.25 and 1 x the copies' RMS (with pair
  mixing) -> FOLLOWS (I / N 0.85, 0.99); copies only -> NO-SIGNAL; paircode, paircode_op, fourier at lam 2 ->
  DOES-NOT-FOLLOW (I / N 0.00-0.06; fourier N 0.47: Fourier products hold x + z and x - z alike).
- **Null bar** (`--nullbar`, `aurshalevmerin/astrocyte-v230-nullbar`): pair, fourier, mlp, paircode, paircode_op (none
  holds x op z; copies x, z, w and the answer; d 192), lam in {0.05, ..., 16} x the copies' RMS, seeds 1-6, 270 runs.
  TAU230 = max(0.20, 1.5 x the largest I / N among runs with signal, rounded up to 0.05) = **0.35**, so **BAR = 0.35**.
  Signal in 204 of 270 runs; FOLLOWS (at 0.35) in none. Largest I / N: fourier at lam 0.05 (0.225, 0.202, 0.169;
  N 0.0007-0.0009, where the ratio is noise, as in v224), mlp lam 1 0.144; pair, paircode, paircode_op <= 0.043.
  (results_v230/nullbar, local; v230_nullbar.json.)
- **Clamp known answer** (w at its own token, k 6, pairs differ in w only; checkpoints of seeds 0, 3, 4; CPU):
  - v215's separate class means: cBo - cR +0.49 (seed 0), +0.18 (seed 3), but the z-copy subspace alone moved the
    answer in 40% and 97% of pairs (w drawn per pair: separate class means of z carry w's code). Changed to joint codes.
  - Joint codes, full 22 dimensions: cBo +0.77 / +0.90, z's subspace still 11% / 50% (a faint copy's full code spans
    high-variance noise directions). Changed to null-ranked dimensions, random matched to Uo.
  - Final: seed 0 hit cBo 0.863, full 0.874, cD 0.001, cR 0.000 -> USED (+0.863 [+0.838, +0.887]); seed 3 cBo 0.993,
    full 0.992, cD 0.000, cR 0.007 -> USED (+0.986); seed 4 UNSCORABLE (full 0.018: '=' reads w within block 1 there,
    as the layer-1 W DAS (0.05) showed). Ranks: w 7-21, z's copy 0-3 at w's token.
- **Mechanics** (TINY=1, CPU: pos, neg, null, verdict end to end): pass (one fix: a position with no correct base item
  is UNSCORABLE for the clamp instead of an error). The checkpoint-hash stop is skipped only under TINY. TINY models
  do not learn, so their cells are UNSCORABLE: the USED and FOLLOWS branches are exercised by the known-answer check
  and the synthetic worlds, not by TINY.
- **Independent code review** (a separate reviewer, before the freeze): no correctness bug. Applied: the run stops if TAU230
  is unset (otherwise no cell could be FOLLOWS and the fresh nulls would pass vacuously); the verdict stops unless all
  5 pos, 5 neg and 270 fresh null outputs are present; Uo keeps directions with singular value > 1e-3 (absolute; U is
  orthonormal) instead of relative to the largest. Known answers unchanged after these edits.

## 7. Verdicts and readings

Scored cells: ';' (POSITIVE in all five pos models at both k); NEGATIVE cells (clamp only: there is no readability
ground truth, since s may be present but unused); every cell of the neg models.

**Per arm (share swap: yes = FOLLOWS; clamp: yes = USED), per k:**
- **DETECTS:** yes at ';' in >= 4 of 5 pos models. **SEED-DEPENDENT:** 2-3. **MISSES:** <= 1. UNSCORABLE counts as not yes.
- **NOT-SPECIFIC:** any yes in a neg model (any position, either k), or (clamp) any yes at a NEGATIVE cell of a pos
  model, or (share) any FOLLOWS among fresh null worlds (seeds 7-12, the null bar's grid).

**Readings for the paper** (k 6 decides "unwritten, used, 8 tokens before the answer"; k 0 is the short distance):
- **Share swap DETECTS at k 6, specific:** the paper states that the share swap says yes to an unwritten, used value
  8 tokens before the answer in this setting, and no in models that never need it. It is the missing positive control
  for the swap's FOLLOWS (Ouro P1) and for its negatives (Kudo's format, the answer-only states).
- **Share swap MISSES or SEED-DEPENDENT at k 6:** the swap's negatives carry little weight beyond "not shown"; the paper
  says so in the main text. A positive (Ouro P1) keeps its force only if the swap is specific.
- **Clamp DETECTS at k 6, specific:** the clamp has shown power for an unwritten, used value at a distance; Ouro's
  NOT-USED is reported as a result of an instrument with shown power (in this setting).
- **Clamp MISSES or SEED-DEPENDENT at k 6:** Ouro's "not used" is weakened to "no use found by a clamp that misses a
  used value in (n) of 5 models here"; the abstract drops "not used" as a finding.
- **NOT-SPECIFIC (either arm):** that arm's yes answers lose force (Ouro P1 FOLLOWS for the swap; v221 inb USED for the
  clamp); the paper reports which cells said yes.
- **The clamp's power transfers to v215 / v221 / v222 only as far as their codes match this one** (disclosure 4):
  joint codes equal separate class means when the other copies are balanced within the item set. Before the paper
  cites a clamp result here as support for those verdicts, it checks that balance in their item sets (or re-scores
  them with joint codes) and states which.
- **DETECTS at k 0 but not k 6:** the paper states the distance limit.
- In every case the paper says the setting is a 6-layer model trained from scratch on a synthetic task: power here
  does not transfer automatically to Ouro or Qwen.

**Predictions:** share swap DETECTS at k 6 70%, at k 0 75%; clamp DETECTS at k 6 60%, at k 0 60%; share swap specific
85%; clamp specific 85%.

## 8. Frozen files (sha256, first 16 hex, LF)

- `v230_protocol.py` 0c6be76ba6e14a38 (TAU230 0.35)
- `build_kaggle_v230_protocol.py` 8b6019c25285b1f3
- reused unchanged: `v230_pilot.py` a5ba00309f35d1ab (pilot 5), `v220_train.py` 6fbfc35236c2376d, `v178_analyze.py`
  90bbc8a848dc5544

Run order (aurshalevmerin): `astrocyte-v230-protocol` (GPU, MODE=main: gpu0 pos, gpu1 neg) and
`astrocyte-v230-nullfresh` (CPU, MODE=null) in parallel; then `python v230_protocol.py verdict <outputs>` (aggregation
of their JSON files only, run locally).

## 9. RESULT (2026-11-01; frozen code, freeze f50ed67 / origin v230-freeze cfc8972)

Kernels `aurshalevmerin/astrocyte-v230-protocol` v1 (GPU, exit codes 0, 0) and `astrocyte-v230-nullfresh` v1 (CPU, exit
0). All five pos checkpoints matched their sha256 (held-out 0.995-1.000); neg models held-out 0.983-0.997 (all pass
0.95). `python v230_protocol.py verdict results_v230`: 5 pos, 5 neg, 270 fresh null runs.

**Clamp: DETECTS at k 6 (5/5) and k 0 (5/5), specific.** ';' cBo - cR: k 6 +0.90, +0.67, +0.60, +0.93, +0.97; k 0
+0.85, +0.90, +0.62, +0.91, +0.98 (seeds 0-4; every CI lower bound >= +0.54; full patch 0.82-0.99, cR <= 0.003). No
USED at any NEGATIVE cell (largest: w's token, seed 3, k 0, +0.091) and none in a neg model. '=' (seed 0 INDETERMINATE)
reaches +0.076 / +0.087, below the 0.10 bar.

**Share swap: DETECTS at k 6 (5/5) and k 0 (5/5), NOT-SPECIFIC.** ';' FOLLOWS in every pos model at both k (I/N
0.78-0.91). No FOLLOWS in any neg model (largest I/N 0.04). One fresh null run of 270 is FOLLOWS: fourier, lam 0.05,
seed 8, I/N 0.363 against the bar 0.35, with N = 0.0009. By the frozen rule the share swap is NOT-SPECIFIC; per section
7, its yes answers (Ouro P1 FOLLOWS) lose force, and the paper reports which cells said yes.

Descriptive, not part of any label (post hoc):
- The null that crossed is the regime flagged before the freeze (section 6: the largest null I/N, 0.225, was fourier at
  lam 0.05 with N ~ 0.0007, where the ratio is noise). Its N is about 50 times below the smallest N of any pos-model
  cell (0.043). A floor on N would have excluded it; none was pre-registered, so the label stands.
- In the pos models the share swap is FOLLOWS at every scored position (fillers, w's token and '=', 40/40 cells),
  including positions where DAS finds no s and the clamp finds no use. The swap reads s where it is present, not where
  it is used; the clamp separates the two here.

**Transfer check (section 7's added reading; post hoc, 2026-11-01).** Item-set balance, as the RMS over target classes
of the family-demeaned conditional mean of another variable's one-hot (0 = balanced; a variable equal to the target
gives about 0.95): v230 0.04-0.08 (sampling: w drawn per pair); Ouro Study 3 (v194, a+b units vs a, b, c, answer)
0.037-0.040 (sampling: 200 of 1,000 unit combinations per family); v215 / v222 Kudo families 0.13-0.34 (structural: B
is a function of d1, o1, d2). None is exactly balanced, so joint codes and separate class means are not identical on
any of them. Their states are not kept locally, so they were not re-scored. Instead v215's original clamp (separate
class means of s and of the z copy, Uo at sv > 1e-3 x max, random subspace of 22 dims; same pairs, same checkpoints;
`v230_posthoc_classmean.py`, CPU, outputs `results_v230/posthoc_classmean_clamp.json`) was run on the five pos models:

- ';' USED in 5/5 at both k: cBo - cR k 6 +0.89, +0.56, +0.49, +0.81, +0.84; k 0 +0.89, +0.73, +0.49, +0.83, +0.86
  (frozen joint-code clamp: +0.60 to +0.98). cB - cR (Ouro's form, no copy projected out) +0.75 to +0.93.
- No USED at a NEGATIVE cell. Seed 0's '=' (INDETERMINATE, not scored) is USED under class means (+0.21 / +0.24) and
  UNCLEAR under joint codes (+0.08 / +0.09).
- The z-copy subspace from class means moves s at ';' in 47-96% of pairs (joint codes: 6-37%): class means of a copy
  carry part of s's code here, so the copy-projected statistic loses some power (seed 2 +0.49 vs +0.60).

So the clamp as used for Ouro and Kudo also detects s here; the change to joint codes was needed for the known answer
at w's token, not for detection at ';'.

**Readability check (post hoc, 2026-11-01, after two outside reviews asked what the swap's FOLLOWS at unused positions
detects).** Independent of the swap: on family-demeaned states with the copies (x, z, w, op, answer; one-hot)
regressed out on training folds, a ridge classifier decodes s on held-out families (`v230_posthoc_readability.py`,
k 6, layers 1-5, CPU; `results_v230/posthoc_readability.json`). Chance 0.043. Best layer (layer 5 almost everywhere):

- pos models, fillers (positions 6, 9): s 0.94-1.00 in seeds 0, 1, 3; 0.90 in seed 4; 0.63-0.65 in seed 2. Random
  functions of (x, z) read there at 0.20-0.31. w is not yet seen at the fillers, so this is s itself, not the answer.
- pos models, w's token and '=': s 0.85-1.00 in four seeds, 0.40-0.44 in seed 2 (random functions 0.12-0.16). Here
  the answer and w are present, and s = T^-1[y - w] is a function of both; regressing their one-hot copies does not
  remove a joint code, so these two positions are weaker evidence.
- neg models: s reads no better than random functions anywhere (s 0.07-0.28, random 0.08-0.30).

So s is carried forward to the fillers and read there by a linear probe in every pos model, though DAS swaps it in at
most 0.07 there and the clamp finds no use: present but not used. The share swap's FOLLOWS at those positions is
consistent with this independent probe.
