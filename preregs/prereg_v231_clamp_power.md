# v231 pre-registration: clamp power for a faint unwritten value

Frozen 2026-11-02, before any clamp outcome below native strength was computed.

## 1. Why

v230 (`docs/prereg_v230_positive_control.md`, section 9) showed the clamp detects an unwritten value used eight tokens
later in 5/5 small models, including in the form used for Ouro (post hoc). In those models s holds 32-47% of the
remaining variance at ';'; Ouro's a+b about 0.7% (held-out), and Ouro's clamp subspace was fitted from far fewer items
per dimension. Two outside reviews asked for the clamp's detection rate as the value gets fainter, and where
Ouro falls. Ouro's a+b clamp moved the written digit in 1.7% of pairs (patch after loop 3) and about 4% (clamp through
loops 1-3). The question here: would a used value that faint have moved it more?

## 2. Disclosures (what was seen before the freeze)

1. All v230 results, including the post hoc class-mean clamp (+0.49 to +0.89 at ';', cB - cR +0.75 to +0.93).
2. Known answers on seed 0's checkpoint at native strength (section 5), including the Ouro form with its a - b
   control: cX (the other-sign subspace) moves the answer in 78% of pairs, so Ouro's own comparison, cB - max(cX, cR),
   is +0.16 even at full strength. Because of this the primary Ouro-form statistic was changed, before any weakened
   run, to cB - cR (does s's own subspace move the answer?), with Ouro's comparison kept as descriptive.
3. Dose calibration: realised shares on seed 0 (states only, no clamp outcome).
4. No clamp outcome below native strength was computed.

## 3. Design (`v231_power.py`)

The five v230 pos models, retrained on a Kaggle T4 and stopped unless each matches v230's sha256; position ';', k 6;
v230's items, pairs (z differs), 1,000 bases, band layers 1-5 and bootstrap. The forward pass is untouched: the model
still uses s. Only the states the clamp's subspaces are fitted on are degraded.

- **Dose:** s's population share of the residual (the share swap's N with copies x, z, w and the answer regressed;
  population = mean of held-out and in-sample estimates, which removes the ~1/items-per-class bias of either), lowered
  by added variance drawn once per item: native (0.32-0.47), 0.1, 0.03, 0.01, 0.005, 0.002, calibrated per layer
  (known answer: within 10%). Ouro's a+b: held-out 0.006-0.008 with about 576 training items per class, so its
  population share is about 0.008-0.010.
- **Shape:** state (primary): Gaussian with the covariance of the model's own residual at that layer (states minus the
  joint codes of s, x, z, w, op); iso: isotropic Gaussian.
- **Sample:** ouro (primary): 782 fitting items per half (34 per class; d / n_per_class 5.6 vs Ouro's 2048 / 360 =
  5.7); native: the whole half (~260 per class).
- **Form:** ouro (primary for the Ouro reading): separate family-centred class means of s, no copy projection,
  statistic hit(cB) - hit(cR) (cR random, same dimension); Ouro's comparison hit(cB) - max(hit(cX), hit(cR)) recorded,
  descriptive. v230: the frozen v230 clamp (joint null-ranked codes, copy projected, cBo - cR).
- Grid: native x 2 samples x 2 forms, plus 5 doses x 2 shapes x 2 samples x 2 forms: 44 cells x 5 models.

## 4. Rules

- **Per model and cell:** v215 / v230 rule on the form's statistic: USED if >= 0.10 with the 97.5% interval's lower end
  > 0; NOT-USED if the upper end < 0.02; UNSCORABLE if the whole-state patch hits < 0.30; else UNCLEAR.
- **Per cell:** DETECTS if USED in >= 4/5 models; SEED-DEPENDENT 2-3; MISSES <= 1 (UNSCORABLE counts as not USED).
- **Detection limit** per (shape, sample, form): the faintest dose at which the cell and every stronger dose
  (native included) DETECTS; none if native does not.
- **Calibration on every seed:** a (seed, dose, shape) whose population share misses its target at any band layer by
  more than 10% or 0.0005 counts as not USED in all its cells (biased against showing power), and is reported.
- **Ouro reading** (form ouro, shape state, sample ouro):
  - **POWER-AT-OURO-LEVEL:** detection limit 0.005 or fainter. The paper may say Ouro's NOT-USED comes from a clamp
    whose form detects a used value at Ouro's signal level, in this setting.
  - **AT-THE-LIMIT:** limit 0.01. The paper states the limit and that Ouro (about 0.008-0.010) sits at it; NOT-USED is
    reported as weak.
  - **UNINFORMATIVE:** limit 0.03 or stronger, or none. The paper drops "not used" for Ouro and says the clamp in
    that form cannot detect a value as faint as Ouro's.
- **Predictions:** POWER-AT-OURO-LEVEL 45%, AT-THE-LIMIT 25%, UNINFORMATIVE 30%.
- The iso shape, the native sample and the v230 form are reported alongside, as the detection curve; they do not
  change the reading. If iso and state disagree in Ouro's regime, the paper says so.

## 5. Pre-freeze record

- Known answers on seed 0's local checkpoint (CPU):
  1. Form v230 with no added variance reproduces `v230_protocol.clamp`: hits identical, statistic 0.903 and interval
     [0.877, 0.926] identical up to float32 rounding (3e-9): PASS.
  2. Form ouro's cB and cR hits equal the post hoc class-mean clamp (0.940, 0.014): PASS. At full strength the form's
     statistic cB - cR is +0.926 [0.904, 0.946] (USED); Ouro's comparison, cB - max(cX, cR), is +0.161 (cX 0.779).
  3. Dose calibration, population share per band layer: 0.1 -> 0.0998-0.1001; 0.03 -> 0.0296-0.0301; 0.01 ->
     0.0098-0.0103; 0.005 -> 0.0048-0.0052; 0.002 -> 0.0020-0.0024. PASS under the tolerance "10% of the target or
     0.0005, whichever is larger", which was set after the first calibration runs missed at 0.002 (a 10% band there is
     below the share estimator's noise); the first two runs, with fewer refinement steps and then the held-out share
     as the target, missed at 0.005 and 0.002 and were fixed in the calibration code, not by moving targets.
- Code review (separate reviewer, code only): no bug that changes a label or the reading. Applied: `verdict` checks every
  seed's calibration (below); an early UNSCORABLE return now carries `diff`. Noted: the state-shaped noise covariance
  is estimated on all 12,000 items (s, x, z, w and op removed, so it carries no s information across halves).
- Synthetic outputs through `verdict` give the expected labels (limit 0.01 -> AT-THE-LIMIT).

## 6. Frozen files (sha256, first 16 hex, LF)

- `v231_power.py` 376b13d56fcfe043
- `build_kaggle_v231.py` 5628b1f4410f18ca
- reused unchanged (as frozen for v230): `v230_protocol.py` 0c6be76ba6e14a38, `v230_pilot.py` a5ba00309f35d1ab,
  `v220_train.py` 6fbfc35236c2376d, `v178_analyze.py` 90bbc8a848dc5544

Run order (aurshalevmerin): `astrocyte-v231-power` (GPU, both T4s: gpu0 seeds 2, 0; gpu1 seeds 1, 3, 4), then
`python v231_power.py verdict <outputs>` locally. A checkpoint that differs from v230's sha256 stops its GPU's
remaining seeds; `verdict` then stops for want of five seeds.

## 7. RESULT (2026-11-02; frozen code, freeze 1fe4d95 / origin v231-freeze 780d1a3)

Kernel `aurshalevmerin/astrocyte-v231-power` v1 (GPU). All five checkpoints matched v230's sha256. Calibration hit every
target within tolerance except 0.002 state-shaped at layer 2 in seeds 0 and 1, whose cells count as not USED (rule).
`python v231_power.py verdict results_v231`:

**Ouro reading: POWER-AT-OURO-LEVEL.** In Ouro's regime (Ouro form, model-shaped noise, 34 items per class) the
detection limit is 0.005: the clamp is USED in 5/5 models at every dose down to 0.005 (cB - cR +0.73 to +0.90 at 0.005),
and SEED-DEPENDENT at 0.002 (3/5; the two misses are the voided calibration cells, whose statistics were +0.74 and
+0.90). Ouro's a+b sits at about 0.008-0.010.

Detection limits (faintest dose with it and every stronger dose DETECTS):

| shape | sample | Ouro form | v230 form |
|---|---|---|---|
| model-shaped | Ouro-matched | 0.005 | none (native SEED-DEPENDENT 3/5) |
| model-shaped | native | 0.005 | 0.1 |
| isotropic | Ouro-matched | 0.005 | none |
| isotropic | native | 0.002 | 0.1 |

Descriptive (not part of any label):
- Hit rates in Ouro's regime at 0.005: s's own subspace moves the answer to the partner's in 0.77-0.98 of pairs
  (model-shaped) and 0.44-0.72 (isotropic); the random subspace in 0.01-0.12. Ouro's a+b subspace moved its digit in
  0.017 (patch) and about 0.04 (clamp).
- Ouro's own comparison, cB - max(cX, cR) with cX the other-sign subspace, is weak here whatever the dose: +0.00 to
  +0.15 under model-shaped noise (cX moves the answer about as often as cB), +0.32 to +0.52 under isotropic noise at
  0.01-0.005. So in this setting the informative number is s's own subspace against the random one, not the comparison
  with a - b.
- Model-shaped noise is crowded into s's span (variance per dimension inside it 2.5-5.5 times the average, seeds 0
  and 3; Ouro's context: 6.5-11.7). Crowding inside the span leaves the class-mean span in place, which is why
  model-shaped noise costs little; isotropic noise, mostly outside the span, costs more but stays above the bar at
  0.005.
- The v230 form (joint codes, shuffled-label rank, copy projected) keeps 0-1 directions per layer once the signal is
  faint or the sample is Ouro-sized, and then does nothing (statistic 0.00): it MISSES everywhere in Ouro-sized samples
  and below 0.1 in native ones. It is not a usable clamp for faint values.
- Not covered: a value whose code is faint inside the model (arm B), and the v215 form used for Kudo (class means with
  the copy projected out).
