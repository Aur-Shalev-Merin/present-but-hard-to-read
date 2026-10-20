# v209 pre-registration: is a + b used before the answer has formed? (earlier and answer-free patching in Ouro)

*Drafted 2026-10-19. Status: **FROZEN 2026-10-19** (this commit, pushed to origin, branch `v209-freeze`, before any
main run). No v209 treatment condition (ab, amb or ans patch, or the answer ablation) has been run on a main-set pair.
Before the freeze, only control quantities were computed: the end-of-loop-1 subspaces and family means
(`astrocyte-v209-plan`, section 5), and a GPU smoke test on pilot pairs with random subspaces and control conditions
only. Frozen files (sha256, first 16 hex chars, LF):
- `v209_patch.py` 15addeddfd27e93b; `v209_plan.py` 72a999d9a9bd33f8; `build_kaggle_v209.py` 78376ac521999825;
- `results_v209/plan/v209_patch_plan.npz` d4e9dc3eb360cf28;
- reused, identical to v194's frozen files: `v194_items.json` fceceeb4dcbd4076; `loop_grad_probe.py` 27b80fa0b999bfa7;
  `results_v194/cpu/v194_patch_pairs.json` a6829055c3730361; `v186_analyze.py` b408d943603262a6;
  `v178_analyze.py` 90bbc8a848dc5544.*

## 1. Why

v194 Part B found NOT-USED: at P1, replacing the a + b component of the state with another problem's, at the end of
loop 3 or loop 2, leaves the digit Ouro writes unchanged, while the same swap of the answer component changes it (0.90
at end3). An outside review of note v9.2 (W1) objected that this design cannot detect use:
- By the end of loop 3 the answer is already formed and strongly readable. Swapping a weak a + b component while the
  answer stays intact pits the two against each other, so a null is expected even if Ouro formed the answer from a + b
  in earlier loops.
- The equal-answer pairs (B2) have the same problem: the intact answer is the one being contradicted.
- The loop-2 test, where this matters most, was weak (answer patch 0.33).

The note now states the narrowed claim ("once the answer has formed, the written digit does not depend on it"). v209
tests whether the stronger claim holds, with three patches that do not leave a formed answer to override a + b.

## 2. Design

**Kept from v194, unchanged:** items (`v194_items.json`), the 2,000 B1 pairs (same c, same parity of a + b, different
(a + b) mod 10) and 2,000 B2 pairs (same answer, different (a + b) mod 10 of the same parity, c compensates), the
cross-fitted subspace recipe (family-demeaned P1 states of the other template half, class means mod 10, top 9 right
singular vectors), the forward pass (teacher-forced prompt + answer, fp32 Ouro via `loop_grad_probe`), the random
subspace, and the hit definitions:
- B1: hit = the prediction at P1 is x′'s answer units digit.
- B2: hit = the prediction at P1 is (a″ + b″ − c_x) mod 10, the digit x would write from x″'s a + b and its own c.

**New tests** (`v209_patch.py`), each run on both pair types:
- **e1:** at the end of loop 1, set x's component at P1 in the condition subspace (fitted on end1 states) to x′'s; run
  loops 2–4. Studies 1–2 found a + b and the answer equally readable after loop 1 (+0.16 each).
- **cl (clamp):** at the end of loops 1, 2 and 3, each time set x's component at P1 in that point's subspace to x′'s
  unpatched component; run loop 4. Later loops cannot rebuild x's own a + b at P1.
- **ab3 (answer ablated):** at the end of loop 3, first set x's component in the answer subspace to x's family mean,
  h ← h − U_a U_aᵀ (h − μ_fam); then set the component in the condition subspace, orthogonalised against U_a, to x′'s;
  run loop 4. The orthogonalisation keeps the patch from carrying x′'s answer back in. Condition `abl` ablates only.
- **Conditions:** e1 and cl: ab, amb, rand, and ans (positive control, B1 only); ab3: abl, ab, amb, rand. Plus `none`.

**Plan** (`v209_plan.py`, `astrocyte-v209-plan`, Kaggle CPU): the end1 subspaces U_{ab,amb,ans}_{h}_e1 by v194's recipe,
each plus family's mean P1 end3 state (μ_fam, over all 200 plus items of the family, x included), and v194's frozen
plan arrays copied unchanged. `results_v209/plan/v209_patch_plan.npz`.

**Runs:** 4 parts (pairs k with k mod 4 = part, over v194's B1-then-B2 list), on Kaggle T4: parts 0–1 on account
aurshalevmerin, parts 2–3 on aurmerin (two GPUs per kernel). The plan and v194's pairs are embedded in the kernels.
The analysis (`v209_patch.py analyse`) runs on the downloaded per-pair rows (seconds, CPU).

## 3. Statistic and labels

Per test (e1, cl, ab3) and pair type (B1, B2), six tests:
- Statistic: hit(ab) − max(hit(amb), hit(rand)), cluster bootstrap over x's family, 10,000 resamples, seed 2091,
  two-sided 99.17% CI (5% family-wise over the six tests).
- **UNSCORABLE** (e1, cl): B1's hit(ans) for that test < 0.30, i.e. the method cannot move the prediction; applies to
  both pair types. ab3 has no answer patch; v194's end3 answer patch (0.90) shows that patching at end3 moves the
  prediction.
- **USED:** difference ≥ 0.10 with CI lower bound > 0. **NOT-USED:** CI upper bound < 0.02. **UNCLEAR:** otherwise.
- Selftest (`v209_patch.py selftest`, synthetic rows): a used world → USED in all six, an ignored world → NOT-USED,
  a world where nothing moves → UNSCORABLE for e1 and cl.

**Descriptive** (no label): own-digit and target rates under every condition, in particular under `abl` (does loop 4
rebuild x's own digit once the answer component is gone?), and the subspace overlaps in section 5.

## 4. Predictions and what each outcome means for the note

| test | USED | NOT-USED | UNCLEAR | UNSCORABLE |
|---|---|---|---|---|
| e1, B1 / B2 | 10% / 10% | 25% / 25% | 15% / 15% | 50% / 50% |
| cl, B1 / B2 | 25% / 20% | 45% / 50% | 20% / 20% | 10% / 10% |
| ab3, B1 / B2 | 15% / 15% | 55% / 55% | 30% / 30% | — |

Reading rules (fixed now):
- **B2 decides.** In B1 the answer moves with a + b, and the a + b subspace partly overlaps the answer's (section 5),
  so a B1 USED can come from answer information carried by the patch. A use claim needs USED in B2 for the same
  test; B1 USED with B2 not USED is reported as consistent with answer leakage.
- **USED in B2 for cl or ab3 (or e1):** the note says that at P1, Ouro's written digit follows a + b when the answer
  cannot override it (with the test named), and drops "does not depend on it" for that case.
- **NOT-USED in B2 for every scorable test:** the note says that the written digit does not follow a + b even when it
  is clamped from loop 1 on, or patched after the answer component is removed; the claim stays limited to P1 and to
  class-mean subspaces.
- **UNCLEAR, or e1/cl UNSCORABLE:** the note says so, and keeps the narrowed wording.

## 5. Pre-freeze checks (control quantities only)

- **Plan** (`results_v209/plan/`): end1 subspaces built; mean cos² of the principal angles, half 0 / half 1:
  - end1 vs end3 subspace, same label: ab 0.18 / 0.28, amb 0.13 / 0.21, ans 0.31 / 0.36;
  - at end1: ab vs amb 0.45 / 0.48, ab vs ans 0.27 / 0.27.
  - The large ab/amb overlap at end1 (v194 at end3: 0.26) makes the amb control conservative for e1 and cl: part of an
    a + b effect would also show up under amb and be subtracted. It is reported, not corrected.
- **Selftest:** passed (section 3).
- **GPU smoke test** (`astrocyte-v209-smoke`, T4, `results_v209/smoke/`): 3 B1 and 3 B2 pilot pairs, random
  subspaces for every condition, conditions none, rand, ans (a random subspace here), abl only. The code ran end to end
  for all three tests; unpatched, Ouro predicted its own digit on every pair, and every random-subspace patch and the
  ablation left it unchanged. About 6 s per pair after loading, so about 10 s per pair with all conditions: about
  2.8 h per GPU for 1,000 pairs.

## 6. Run plan

1. Plan (done), selftest (done), smoke test.
2. Freeze: hashes of `v209_patch.py`, `v209_plan.py`, `build_kaggle_v209.py`, the plan file, and the reused v194 files;
   push to origin before any main run.
3. `astrocyte-v209-main` on both accounts (parts 0–1 and 2–3).
4. Analysis on the downloaded rows; RESULT section here; note update per section 4.

## 7. Amendment 1 (2026-10-19, after the freeze, before any main run)

The first push of the main kernels was rejected by Kaggle (HTTP 400): embedding the 1.66 MB plan made the script
2.5 MB (v194's was 0.33 MB). `build_kaggle_v209.py` (78376ac521999825 → **295551fc8ec511bc**) now reads the plan from the
`astrocyte-v209-plan` kernel's output (kernel_sources), as v194 read its plan from `astrocyte-v194-cpu`, and aborts
unless the file's sha256 is the frozen d4e9dc3eb360cf28. It also names the two main kernels by their parts
(`astrocyte-v209-main01`, `-main23`), both on account aurshalevmerin (two GPU sessions, T4 ×2 each). No analysis
file, rule or input changed; no main-set pair had been run.

## 8. RESULT (2026-10-19; `results_v209/v209_patch_analysis.txt`)

Runs: `astrocyte-v209-main01` (parts 0–1) and `-main23` (parts 2–3), Kaggle T4 ×2 each, at the frozen and amended
hashes; the plan's hash checked in both (d4e9dc3eb360cf28); all four parts exited 0 (2.4–3.0 h per GPU). 2,000 B1
and 2,000 B2 pairs, 36 families each. The analysis ran on the four main parts only (`results_v209/main_all/`; the
smoke test's 6 pilot rows are excluded). Unpatched, Ouro predicts its own digit on every pair.

| test | positive control: hit(ans) | B1: hit ab / amb / rand → statistic [99.17% CI] | label | B2: hit ab / amb / rand → statistic [99.17% CI] | label |
|---|---|---|---|---|---|
| e1 | 0.009 | 0.003 / 0.002 / 0.001 → +0.001 [+0.000, +0.003] | UNSCORABLE | 0.004 / 0.002 / 0.001 → +0.002 [−0.002, +0.006] | UNSCORABLE |
| cl | 0.914 | 0.042 / 0.058 / 0.003 → −0.016 [−0.034, +0.001] | NOT-USED | 0.009 / 0.004 / 0.001 → +0.005 [−0.001, +0.012] | NOT-USED |
| ab3 | — | 0.102 / 0.097 / 0.098 → +0.005 [−0.011, +0.015] | NOT-USED | 0.089 / 0.090 / 0.086 → −0.001 [−0.016, +0.013] | NOT-USED |

Predicted: e1 UNSCORABLE 50%; cl NOT-USED 45% / 50%; ab3 NOT-USED 55%.

**Descriptive.**
- **e1:** loops 2–4 undo a patch made at the end of loop 1, even of the answer subspace (hit 0.009; own digit kept in
  97–100% of pairs under every condition).
- **cl:** clamping the answer subspace at the ends of loops 1–3 moves the prediction to x′'s digit in 91.4% of pairs;
  clamping a + b does not (own digit kept in 90.5% of B1 and 97.4% of B2 pairs). In B1, amb slightly out-hits ab
  (0.058 against 0.042), as in v194 at end3.
- **ab3:** with the answer component set to its family mean, loop 4 writes x's own digit in only 31–32% of pairs, and
  every condition hits x′'s target at about chance (0.084–0.102 for one digit in ten). Neither x's own a + b
  component (abl) nor x′'s (ab) restores or steers the digit.

**Reading, per section 4 (NOT-USED in B2 for every scorable test).** At P1 the written digit does not follow a + b
even when its component is clamped from the end of loop 1 to the end of loop 3, or swapped in after the answer
component has been removed; with the answer component removed, the a + b component that remains does not rebuild the
answer. The single patch at the end of loop 1 is unscorable, since later loops undo any patch made there. The claim
stays limited to P1 and to subspaces spanned by class means.

**Note added after the result (2026-10-19, after review; not a change of label).** The ab3 test has no positive
control, and the ablation alone moves the digit off x's own in about 68% of pairs, with every condition then at
chance. Its NOT-USED is therefore weak evidence: it may reflect a disrupted state rather than an unused value. The
note rests the not-used claim on the clamp test (answer clamp 0.914) and describes ab3 as weaker.
