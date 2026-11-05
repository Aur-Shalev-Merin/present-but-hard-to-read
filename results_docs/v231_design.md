# v231 design draft (NOT frozen): clamp power for a faint unwritten value

Workstream paper3. Claimed 2026-11-01. Follows v230 (`docs/prereg_v230_positive_control.md`, section 9).

## 1. Why

v230 showed the clamp detects an unwritten, used value eight tokens before the answer in 5/5 small models (+0.60 to
+0.98), with the class-mean form used for Ouro also detecting it (+0.49 to +0.89, post hoc). Two outside reviews (4/10 each,
2026-11-01) agree this does not cover Ouro: there s explains 32-47% of the remaining variance at ';', while Ouro's a+b
explains under 1%, and Ouro's clamp subspace is fitted from far fewer items per dimension. Both asked for one thing:
the clamp's detection rate as the value gets fainter, and where Ouro falls on that curve.

What the Ouro clamp faced (v194 / v209, from the paper and their logs):
- d = 2048; a+b subspace = 9 family-centred class means (10 classes), fitted on the other template half: about 3,600
  plus items, 360 per class, so d / n_per_class ~ 5.7 (v230: 192 / 260 ~ 0.74).
- a+b's share of the remaining variance (share swap N, copies and answer regressed): about 0.006-0.008.
- Context variance is crowded into the digit span: inside the span of the family-centred digit means, the context's
  variance per dimension is 6.5-11.7 times its average in Ouro (paper, "a small change, not a crowded one").
- Statistic: hit(a+b) minus the larger of hit(a-b) and hit(random); no copy projection (v194), separate class means.

## 2. Design (arm A: the fitting side, no retraining)

The clamp fails on a faint value mainly through its subspace estimate: class means of a faint value in crowded states
span mostly noise. Arm A reproduces Ouro's estimation regime on the five frozen v230 pos checkpoints at ';' (k 6),
with the forward pass untouched (the model still uses s; only the clamp's fitting data are degraded):

- **Dose (signal share).** Fitting states get added variance, drawn per item, so that s's class-mean share of the
  family-demeaned fitting states falls to a target: native (~0.4), 0.1, 0.03, 0.01, 0.005, 0.002. Ouro's ~0.007 sits
  between the last three. The added variance is fixed random Gaussian with a set crowding factor c: its variance per
  dimension inside s's native span is c times its average. c = 1 (isotropic) and c = 10 (Ouro-like).
- **Sample.** Fitting items per class: native (~260) and Ouro-matched (d / n_per_class = 5.7, ~34 per class).
- **Clamp forms.** (i) as in Ouro: separate class means, no copy projection, statistic hit(cB) - max(hit(cX), hit(cR))
  with cX a control subspace of the same kind (class means of a random function of (x, z), as a-b was for Ouro);
  (ii) as frozen in v230: joint null-ranked codes, copy projected, cBo - cR. Same pairs as v230 (z differs).
- Grid: 6 doses x 2 crowding x 2 samples x 2 forms x 5 models = 240 clamps (Kaggle CPU, est. under 1 h).

Known answers before the freeze: the native dose with no added variance and the native sample reproduces v230's
frozen clamp numbers exactly (form ii) and the post hoc class-mean numbers (form i); the dose calibration hits each
target share within 10%. Only states and shares are inspected before the freeze, no clamp outcome below native dose.

## 3. Proposed rules (to be fixed before any run)

- Per cell (dose x crowding x sample x form): DETECTS if USED (v215 rule: >= 0.10, interval lower end > 0) in >= 4/5
  models; SEED-DEPENDENT 2-3; MISSES <= 1.
- **Detection limit:** per (crowding, sample, form), the faintest dose at which DETECTS holds, with every stronger dose
  also DETECTS.
- **Ouro reading:** in Ouro's regime (form i, c = 10, Ouro-matched sample):
  - DETECTS at 0.005 (below Ouro's ~0.007): the paper may say Ouro's NOT-USED comes from a clamp with shown power at
    Ouro's signal level, in this setting.
  - MISSES or SEED-DEPENDENT at 0.01 or above: Ouro's NOT-USED is uninformative; the paper drops "not used" for Ouro
    and says the clamp cannot detect a value that faint.
  - In between: the paper states the detection limit and that Ouro is at it.
- The same curve is reported for form ii (what a rerun of Ouro's clamp with the current method could detect).

## 4. Arm B (optional, exploratory): a value that is faint inside the model

Arm A degrades the measurement, not the model. A model whose own s code is faint would be the direct test, but no
training change controls that cleanly (distractor prefixes are likely ignored at ';'). Proposed only as an optional,
exploratory arm.

## 5. Not settled

- Whether the share target should be the share swap's N (copies regressed) or the raw class-mean energy (what the
  Ouro clamp's subspace actually saw); Ouro's raw energy is not available locally.
- Whether 5 models are enough per cell, or the grid should be cut to Ouro's regime plus the native control.
