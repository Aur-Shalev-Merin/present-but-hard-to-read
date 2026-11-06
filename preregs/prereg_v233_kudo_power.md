# v233 pre-registration: clamp power for the form used in the re-test of Kudo et al.

Workstream paper3 (Ouro note, arXiv v2). Claimed 2026-11-05. Follows v215, v216, v230, v231. Frozen 2026-11-05, before any degraded cell
was run.

## 1. Why

The paper's limitations say the clamp's power at a faint signal was shown "not for the form used for Kudo et al.":
v231 tested the class-mean form used for Ouro (no copy projection) and the v230 joint-code form, but not the form of
v215 / v216, which takes separate class means of b and projects them off the class means of the copied digit (d2) and,
without a chain, the answer. In Kudo's re-test that clamp moved the written b or answer in 0-3% of pairs (NOT-USED at
the end of b's equation without a chain, otherwise UNCLEAR), keeping 35-64% of b's directions after the projection. Its
power at Kudo's signal level is unknown, so those results cannot be read. v233 measures it with v231's design.

**Disclosures.**
1. v231's results are known (Ouro form USED 5/5 down to share 0.005 in Ouro's regime; v230 joint-code form fails on
   faint or small samples). The Kudo form differs from the Ouro form only by the projection, so its power is expected
   to lie between them.
2. Kudo's needed shares at the clamped positions are known from v228 (held-out N 0.013-0.077) and set the reading's
   thresholds below.
3. The known-answer check (section 5) runs the Kudo form at native strength on seed 0 (the only clamp outcome seen
   before the freeze).
4. Designed after a reviewer pointed to the untested form; the doses, shapes, samples, pairs and rules are v231's.
5. The known-answer check (seed 0, native strength) showed the z-only projection leaves 0.74 of s, more than Kudo's
   clamps kept; the z-and-x projection (0.51) was added and made primary before any degraded cell was run.

## 2. Design (`v233_kudo_power.py`)

- **Models:** the five v230 pos models, loaded from v231's saved checkpoints (aurshalevmerin/astrocyte-v231-power) and
  checked against v230's sha256 (abort on mismatch). Position ';' with k 6, where DAS puts s; band layers 1-5.
- **Degradation (v231's `noises()`, unchanged):** noise added to the clamp's FITTING states only, so that s's
  population share is native, 0.1, 0.03, 0.01, 0.005 or 0.002; shapes state (the model's own residual covariance) and
  iso; per-layer calibration, same seeds as v231.
- **Samples:** native (all of the fitting half) and **kudo** (782 items, 34 per class: d / n_per_class 192 / 34 = 5.6;
  Kudo's re-test 3584 / 600 = 6.0; the same subset as v231's 'ouro' sample).
- **Form kudo:** separate family-centred class means of s (22 dims) fitted on the other half of the families,
  projected off a copy basis (left singular vectors of U - D D^T U with singular value > 1e-3, v216's procedure); cBo
  = the projected basis, cR a random subspace of the same dimension. ';' precedes any answer, so there is no answer
  projection. Two copy bases:
  - **zx (primary):** the class means of z and of x together. On seed 0 this leaves 0.51 of s's basis, inside the
    0.35-0.64 Kudo's clamps kept, so the test is about as hard as Kudo's on that count.
  - **z (secondary):** the class means of z alone (the operand that differs within a pair, as d2 did in v215):
    Kudo's procedure, which here leaves 0.74 of s's basis (an easier test than Kudo's).
  The primary was set to zx after the known-answer check showed the z projection leaves more of s than Kudo's did
  (disclosure 5).
- **Statistic per cell:** hit(cBo) - hit(cR) over v230's 1,000 pairs (differing only in z), cluster bootstrap over
  families, 10,000 resamples, seed 2153, 97.5% CI; USED >= 0.10 with CI lower > 0; NOT-USED CI upper < 0.02;
  UNSCORABLE if the whole-state patch hits < 0.30; UNCLEAR otherwise (v230 / v231).
- **Descriptive:** hit(cB) (unprojected, = v231's Ouro form), hit(cD) (z's class means alone), the share of s's basis
  left after the projection (Kudo's clamps: 0.35-0.64), and each cell's realised share.

## 3. Rules

- Cell label: **DETECTS** if USED in >= 4 of 5 models, **SEED-DEPENDENT** 2-3, **MISSES** <= 1. A model whose dose and
  shape missed its calibration target at any layer (more than 10% or 0.0005 off) counts as not USED in that cell.
- **Detection limit** per (shape, sample, projection): the smallest dose such that native and every dose down to it DETECT.
- **Kudo reading** (primary: zx projection, state-shaped noise, Kudo-matched sample; the same rule on the z projection
  is reported as secondary):
  - **POWER-AT-KUDO-LEVEL:** limit 0.01 or below (under Kudo's smallest needed share at a clamped position, 0.013).
  - **AT-THE-LIMIT:** limit 0.03 (inside Kudo's range: power at the positions with the larger shares only).
  - **UNINFORMATIVE:** limit 0.1, native only, or none.

## 4. Readings for the paper (fixed now)

- **POWER-AT-KUDO-LEVEL:** the limitation "not for the form used for Kudo et al." is replaced by the measured power,
  and Section 7 says that b's own directions beyond the copies (and the answer) are barely used before the chain, by a
  clamp that detects a used value at that signal level in small models. The overall verdict on Kudo's claim stays
  "cannot tell": the share swap is UNCLEAR there and the answer fixes b within a family.
- **AT-THE-LIMIT:** as above, but only for the clamped positions whose needed share is at least 0.03; the others
  remain without shown power.
- **UNINFORMATIVE:** the paper says the Kudo clamp has no shown power at its signal level, so its NOT-USED and UNCLEAR
  results say little.
- If the primary (zx) and secondary (z) readings differ, the paper reports both and says which matches Kudo's
  share of b left and which matches Kudo's procedure.

Predictions: POWER-AT-KUDO-LEVEL 50%, AT-THE-LIMIT 25%, UNINFORMATIVE 25%.

## 5. Pre-freeze checks

- Known answers (`astrocyte-v233-known`, Kaggle CPU, seed-0 checkpoint): (1) the Kudo form's unprojected cB and the
  whole-state patch equal v231's Ouro-form cB and full exactly (same bases and pairs); (2) cBo is orthogonal to cD.
  Results (kernel v3, `v233_kudo_power.py` 279eed05a8e37853): KNOWN 1 PASS (cB 0.940 vs 0.940; full 0.942 vs 0.942);
  KNOWN 2 PASS for z (max |cos| 5.4e-07, 22 dims kept, 0.74 of s left) and for zx (1.6e-06, 22 dims, 0.51 left).
  Seen at native strength on seed 0 only: z USED +0.891 [+0.867, +0.912] (cBo 0.905, cD 0.486, cR 0.014); zx USED
  +0.710 [+0.681, +0.740] (cBo 0.724, cD 0.863).
- Verdict code tested on fabricated result files (all paths, both readings).
- Dose calibration is v231's code, already checked there (KNOWN 3, within 10% or 0.0005).

## 6. Run plan

Freeze -> push as `v233-freeze` -> `V233_MODE=run python build_kaggle_v233.py`, push astrocyte-v233-power
(aurshalevmerin, CPU) -> the kernel runs the verdict; download to `results_v233/`.

## 7. Hashes (sha256 of the LF-normalised file, first 16 hex)

- New: `v233_kudo_power.py` 279eed05a8e37853 (the version the known-answer kernel v3 ran); `build_kaggle_v233.py`
  c75d445bdd8217e7.
- Unchanged, as on origin/v232-freeze: `v231_power.py` 376b13d56fcfe043 (= v231's frozen hash); `v230_protocol.py`
  0c6be76ba6e14a38; `v230_pilot.py` a5ba00309f35d1ab; `v220_train.py` 6fbfc35236c2376d; `v178_analyze.py`
  90bbc8a848dc5544.

## 8. RESULT (2026-11-05; aurshalevmerin/astrocyte-v233-power v1, Kaggle CPU; `results_v233/power/`)

Verdict as written by the kernel (`v233_verdict.txt`):

- **Primary (zx projection, state-shaped noise, Kudo-matched sample): no detection limit -> UNINFORMATIVE.** The cell
  is SEED-DEPENDENT already at native strength (USED 3/5: +0.264, +0.010, -0.039, +0.101, +0.121), then 3/5 at 0.1, 2/5
  at 0.03 and 0.01, 0/5 at 0.005 and 0.002.
- **Secondary (z projection, Kudo's procedure): limit 0.005 -> POWER-AT-KUDO-LEVEL** (USED 5/5 at every dose down to
  0.005, +0.23 to +0.78; at 0.002 USED in all five but three are voided by calibration misses, so MISSES by the rule).
- Other limits: iso / native: zx 0.005, z 0.005; iso / kudo: zx none (native SEED-DEPENDENT), z 0.005; state / native:
  zx 0.03, z 0.005.
- Calibration misses (cells counted as not USED): 0.002 only (iso: seeds 0, 1, 2; state: seeds 1, 2, 3, 4).

**Descriptive.**
- Share of s's basis left after the projection, Kudo-matched sample, state noise: zx 0.26-0.44, z 0.55-0.67 (native
  sample: zx 0.32-0.51, z 0.59-0.74). The seed-0 known-answer figure (0.51) was the highest of the five; in the
  Kudo-matched sample zx leaves less than Kudo's clamps kept in most models. Kudo's clamps kept 0.43-0.64 with the
  short chain (v215, end of b's equation) and 0.35-0.54 without a chain (v216). So the z arm matches the upper part of
  the chain study's range, and the zx arm the lower part of the no-chain range and below.
- The projected clamp's power tracks the share left: in the Kudo-matched sample at native strength the zx clamp is
  NOT-USED by the clamp's rule in seed 2 (-0.039 [-0.060, -0.019]; share left 0.26) and UNCLEAR in seed 1 (+0.010
  [-0.008, +0.030]; 0.29), although the unprojected class-mean clamp moves the answer in 0.97 and 0.80 of pairs there
  and DAS puts s at this position. A clamp that projects off the class means of the inputs can therefore call a used
  value NOT-USED when the projection removes about three quarters of its directions, because here the value's code
  lies largely in the inputs' class-mean directions.
- cD (the copy basis alone) moves the answer in 0.64-0.98 of pairs at native strength (it carries part of s's code).

**Reading (section 4, readings differ).** The primary reading governs: the Kudo-form clamp has no shown power at its
signal level when the projection leaves as little of the value as the primary arm does, so the Kudo re-test's
NOT-USED and UNCLEAR clamp results say little. The paper reports both: the secondary arm (Kudo's procedure, which here
leaves 0.55-0.67, the upper part of the chain study's 0.43-0.64) has power down to 0.005; the primary arm (0.26-0.44,
overlapping the no-chain study's 0.35-0.54 at its lower end) has none even at full strength in two of five models.
Prediction recorded at the freeze: POWER 50 / LIMIT 25 / UNINFORMATIVE 25; outcome UNINFORMATIVE (primary).
