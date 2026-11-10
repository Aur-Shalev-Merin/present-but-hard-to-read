# v234 pre-registration: the Kudo re-test without a chain, with clamp pairs that share the answer

Workstream paper3 (Ouro note, arXiv v2). Claimed 2026-11-06. Follows v215, v216, v233 and the post hoc quick checks of
2026-11-06. FROZEN 2026-11-08, before any run with a learned subspace.

## 1. Why

In the re-test of Kudo et al. without a chain (v216), every position is UNCLEAR on the share swap, and the clamp of b's
directions moves the answer in 1% (end of b's equation, NOT-USED) and 3% (before the answer digit, UNCLEAR) of pairs.
Those clamps had to project b's subspace off the answer's as well as the copy's, because within a v216 family k is fixed
and the answer fixes b; the projection left 35-54% of b, where a clamp of this form missed a used value in small models
(v233, primary arm). So v216 cannot say whether b is held and used before the answer.

v234 removes the reason for the answer projection. k varies within a family, and each clamp pair shares the answer
while differing in b (and so in k): x = (k, b) and x' = (k', b') with k o2 b = k' o2 b'. In such a pair the answer's
code carries no difference, so a clamp cannot act through it; if the model uses b's own code at the clamped position,
the clamp predicts a third answer, P = k o2 b' (x's k with x''s b), which neither prompt has. The pairs also share d1,
so only the d2 and k copies need projecting out.

**Disclosures.**
1. v216's results are known (above), as are v233's (power of this clamp form when the projection leaves 0.55-0.67 of
   the value; none at 0.26-0.44) and the 2026-11-06 correction of the share of b kept in the chain study.
2. Designed after the fourth outside reviews (one asked for pairs with equal answers).
3. Seen before the freeze (`astrocyte-v234-smoke`): answer accuracy on 40 random items and the patch mechanics with
   random subspaces in place of every learned one on 20 pairs per position (no learned subspace was used), including
   the whole-state patch (the positive control). Numbers in section 6. The scorability gate was moved from the copy
   patch (cD) to the whole-state patch (v216's convention) after the smoke, because the smoke showed the whole-state
   patch reaching P at eq; no learned-subspace outcome had been seen.
4. Items are new (v215's generator, new seed and crossing); same model (Qwen2.5-7B base, fp16), format, positions,
   layers and band as v216.

## 2. Design

- **Items** (`v234_items.py`): Kudo's Implicit format ("a=k+b, b=d1+d2 ; a?", target " a = A", three demonstrations
  in the same format). A family fixes the letters, the demonstrations and o2, and crosses k in {0, 1, 2} (o2 = +) or
  {9, 8, 7} (o2 = -) with every (d1, d2), d2 <= d1, d1 + d2 <= 7, and o1 in {+, -} (versions plus and minus): 120 items
  per family, 60 families per o2, 14,400 items (seed 2340). b ranges 0-7 and every answer is one digit.
- **Extraction** (`v234_extract.py`, v216's): one teacher-forced pass per item; states at eq (b's last digit, d2),
  last ('?') and pre (the token before the answer digit), hidden_states 0, 4, ..., 28; correct = the answer digit is the
  argmax over the ten digits at pre.
- **Plan** (`v234_plan.py`): per position, band layer (12, 16, 20, 24) and half of the families, family-centred class
  means fitted on the other half (v216's basis()): U = b (8 classes, 7 dims), D = d2's copy, K = k's copy, An = the
  answer; Uo = U projected off D and K (v216's procedure). R = random, Uo's dimension. Share of b left =
  ||U - QQ^T U||^2 / dim(U) (v216 / v233 definition). Pairs: up to 1,000 per position from items whose answer and whose
  partner's answer are correct; partner = same family, version and d1, different d2, equal answer (seed 2341).
- **Clamp** (`v234_patch.py`, v216's): at the band layers, h <- h - U U^T (h - h') at the position, for U in cB (b),
  **cBo (primary)**, cA (the answer: a built-in null, since the answers are equal), cD (d2's copy), cK (k's copy), cR
  (random); full = x''s whole state at layers 11-27. Outcome: the argmax digit at pre. hit = P; kept = A.

## 3. Rules (per position; v216's, outcome changed to P)

- Statistic: hit(cBo) - hit(cR) over the pairs, cluster bootstrap over x's family, 10,000 resamples, seed 2343,
  97.5% CI.
- **UNSCORABLE** if hit(full) < 0.30 (the whole state at the position does not carry the change a used b predicts,
  so the position has no positive control); else **USED** if >= 0.10 with CI lower > 0; **NOT-USED** if CI upper < 0.02;
  **UNCLEAR** otherwise.
- **Power annotation:** a USED or NOT-USED label counts as shown-power only if the position's mean share of b left
  (over band layers and halves) is at least 0.55, the lower end of the v233 arm that detected a used value down to share
  0.005. Below 0.55 the label is reported as "(power not shown)".
- **Design check (all positions):** hit(cA) - hit(cR), same bootstrap. If it is >= 0.10 with CI lower > 0 at a
  position, the premise that equal answers cancel the answer's code fails there, and that position's label is replaced
  by DESIGN-FAILED.
- Primary position: **eq**. pre and last are secondary (same rules).

## 4. Readings for the paper (fixed now)

- **eq USED (shown power):** without a chain, b's own code at the end of its equation, beyond the copies, is used to
  form the answer, in problems the model answers correctly. The paper's no-chain verdict on Kudo et al. changes from
  "cannot tell" to "a used sub-answer before the answer at this position" (the share swap there stays UNCLEAR), and the
  claim that sub-answers are not computed until written is contradicted for this setting.
- **eq NOT-USED (shown power):** b's own directions beyond the copies are not used at layers 12-24 at the end of its
  equation even when the answer cannot carry them; the answer is formed from the copies there, or b is used before
  layer 12. The paper says this supports Kudo et al.'s claim for this setting at this position.
- **UNCLEAR, UNSCORABLE, DESIGN-FAILED, or a label without shown power:** the no-chain verdict stays "cannot tell";
  the paper reports the number.
- pre and last: reported with the same wording as secondary results.

Predictions (eq): USED 30%, NOT-USED 35%, UNCLEAR 30%, UNSCORABLE or DESIGN-FAILED 5%.

## 5. Run plan

Freeze -> push as `v234-freeze` -> `V234_MODE=x` (GPU, extraction) -> `V234_MODE=plan` (CPU) -> `V234_MODE=patch`
(GPU, the clamp and its analysis) -> download to `results_v234/`.

Run note (2026-11-08, after the freeze, before any patch outcome): x and plan ran on account aurshalevmerin. Its two GPU
slots were then held by another workstream's kernels, so the patch step ran on account aurmerin (still Kaggle T4 x2).
The plan kernel's two outputs (`v234_plan.npz` sha256 278d7754b2a637e4, `v234_pairs.json` 982885e625623e24) were
downloaded unchanged and uploaded as the private dataset `aurmerin/astrocyte-v234-plan-out`. The patch kernel's
`run.py` is byte-identical to the frozen build (b2787fb94ba5b557); only its metadata changed (owner, and a dataset
source in place of the kernel source). No rule changed.

## 6. Pre-freeze checks

- `astrocyte-v234-smoke` (kernel v1, T4 x2; items sha ac4121c22f88fe31): answer accuracy 0.825 on 40 random items
  (0.157 s/item); positions on the first item eq = '3' (d2), last = '?', pre = ' ' before the answer digit.
- Patch mechanics on 20 pairs per position from family 0 (no correctness filter), every subspace random: hit(P) 0.000
  for every subspace condition at every position and kept A 1.000 (random 7-dim clamps do nothing). Whole-state patch:
  hit(P) 1.000 at eq (kept A 0.000), 0.250 at pre (kept 0.750), 0.000 at last (kept 1.000). So at eq the whole state
  carries what a used b predicts and the answer combines it with x's own k; pre and last are likely UNSCORABLE.
- `v234_patch.py --selftest` (label rule) passes; `v234_plan.py` was run on fabricated states (mechanics only).

## 7. Hashes (sha256 of the LF-normalised file, first 16 hex)

- New: `v234_items.py` fa7374fd6378d679 (ITEMS_SHA ac4121c22f88fe31), `v234_extract.py` 7c570802c92af672,
  `v234_plan.py` 025497441b9d6f9b, `v234_patch.py` a6126d0008bf4a0a, `build_kaggle_v234.py` 9a23d3a352f18491.
- Dependencies, unchanged from v233-freeze: `v215_pilot.py` d049a0b2d81fb80b, `v215_items.py` 9249f0523d7219eb,
  `v216_items.py` a2ebd48b899fed40, `build_kaggle_v215.py` 49ce949efc2c42a8.

## 8. RESULT (2026-11-08; x and plan on aurshalevmerin, patch on `aurmerin/astrocyte-v234-patch` v1; `results_v234/`)

**eq (primary): UNCLEAR.** cBo - cR +0.038 [+0.023, +0.056] (1,000 pairs; above zero, below the 0.10 bar). The whole
state passes the gate (hit(full) 0.865). Share of b left 0.847, so a USED or NOT-USED label would have counted as
shown power; UNCLEAR carries no power annotation. pre and last: UNSCORABLE (hit(full) 0.000 and 0.002).

| position | none | cB | cBo | cA | cD | cK | cR | full |
|---|---|---|---|---|---|---|---|---|
| eq, hit P | 0.000 | 0.203 | 0.039 | 0.099 | 0.884 | 0.002 | 0.001 | 0.865 |
| pre, hit P | 0.000 | 0.060 | 0.041 | 0.021 | 0.236 | 0.066 | 0.000 | 0.000 |
| last, hit P | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 |

**Design check, omitted by the frozen analysis code.** `v234_patch.py analyse` did not compute the section 3 design
check. It was computed after the result, from the saved per-pair rows, exactly as section 3 specifies (cluster bootstrap
over family, seed 2343, 10,000 resamples, 97.5%): eq +0.098 [+0.077, +0.121] (ok, just below the 0.10 bar), pre +0.021
[+0.012, +0.031] (ok), last +0.000 (ok). No label changes. The eq value sits next to the bar; b's subspace lies about
40% in the answer's (plan log), so clamping the answer's subspace also swaps part of b.

**Descriptive (not part of the rule):** at eq the d2 copy's subspace carries the whole effect (cD - cR +0.883
[+0.854, +0.909] vs full 0.865), b's unprojected subspace +0.202, and b's directions beyond the copies +0.038.

**Reading (section 4, fixed):** the no-chain verdict on Kudo et al. stays "cannot tell"; the paper reports the number.
Descriptively, at the end of b's equation what the answer uses is d2's copy; b's own code beyond the copies adds a small
but nonzero amount (interval above zero, below the bar).
