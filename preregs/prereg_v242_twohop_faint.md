# v242 pre-registration: the two-hop clamp at a faint signal

Workstream paper3 (Ouro note; TMLR version). Claimed 2026-11-11. Follows v231 (power at a faint signal in small models)
and v235 (the two-hop positive control). FROZEN 2026-11-12, before any clamp with a
learned subspace on degraded data.

## 1. Why

v235 showed that in a pretrained model (Qwen2.5-7B, two-hop recall) the clamp finds an unwritten value that is used: the
country at the city's last token (e1), +0.789 over a random subspace. But the country is a strong signal there (its share
of the state's variance at e1, with the copy projected out, is 0.15-0.24), and the paper's NOT-USED and UNCLEAR readings
in Ouro concern a value about fifty times fainter (a+b: about 0.008-0.010). Power at that level has been shown only in
small transformers trained from scratch (v231), by degrading the clamp's fitting data. This is the paper's main
remaining limitation: "at Ouro's signal level, only in small models trained from scratch". v242 runs v231's design on
v235's pretrained setting.

**Disclosures.**
1. All results of v215-v238 are known, including v235's clamp (USED, +0.789; whole state 0.849) and v231 (in small
   models the class-mean clamp is USED in 5 of 5 down to share 0.005).
2. Dose calibration was run before the freeze on pilot 2's saved states (`v235_states.npz`; states only, no clamp
   outcome): native shares 0.145-0.237 at the band; every target hit exactly as a population share; in-sample estimates
   on the degraded states within about 0.002 of it (section 6).
3. `astrocyte-v242-smoke` (if run before the freeze) replaces every learned subspace with a random one: it shows the
   mechanics, the calibration on the GPU states and the whole-state patch (known from v235), no learned-subspace outcome.

## 2. Design (`v242_faint.py`; one GPU kernel)

- **Model, items, pairs, band, position, outcome:** v235's, unchanged (helpers imported from the frozen `v235_clamp.py`):
  Qwen2.5-7B fp16; the 1,038 known cities (25 countries); one pair per city with a city of another country (seed 2353);
  hidden_states 8, 12, 16 at e1; outcome = argmax over the 25 capitals' first tokens at the last position; hit = the
  source city's capital.
- **The forward pass is untouched.** Only the states the clamp's subspaces are fitted on are degraded: the model still
  uses the country, and the question is whether a clamp fitted on faint data still finds it.
- **Dose:** the country's population share of the e1 state's variance with the copy's subspace C projected out, per
  band layer and fitting half: sb / (sb + msw), the ANOVA between-class component over between plus within. Native
  0.15-0.24. Targets 0.1, 0.03, 0.01, 0.005, 0.002. Noise is drawn once per item and scaled so the degraded fitting
  states' population share, sb / (sb + msw + s^2 Wn) (Wn the drawn noise's variance), equals the target.
- **Shape:** state (primary): Gaussian with the covariance of the half's within-country residual at that layer;
  iso: isotropic Gaussian.
- **Sample:** each half has about 20.5 cities per country, so d / items-per-class is about 175 (Ouro's fit: about 5.7);
  this fit is already far sparser than Ouro's, so no separate sample arm.
- **Form:** v235's primary clamp: U = the 25 country class means (centred; at most 24 dims), C = the same from the
  city name's token embeddings (not degraded), Uo = U projected off C; R random with Uo's dimension (seed 2423 + cell).
  A pair uses the bases fitted on the half that does not contain x.
- **Cells:** native (no noise) + 5 doses x 2 shapes = 11. Rows per pair: none, full (x''s whole state at the band;
  ground truth), and per cell cUo and cR.

## 3. Rules

- **Per cell:** statistic hit(cUo) - hit(cR); cluster bootstrap over x's country (25 clusters), 10,000 resamples, seed
  2425, 97.5% interval; v235's label: **USED** if >= 0.10 with lower > 0; **NOT-USED** if upper < 0.02; else UNCLEAR.
- **Ground truth:** hit(full) - hit(cR, native) must be USED; otherwise **GROUND-TRUTH-FAILED** (no reading).
- **Calibration:** a cell whose population share misses its target at any band layer or half by more than 10% (or
  0.0005) counts as not USED, and is reported.
- **Detection limit** per shape: the faintest dose at which the cell and every stronger dose, native included, are
  USED; none if native is not.
- **Reading (shape state):**
  - **POWER-AT-OURO-LEVEL:** limit 0.005 or fainter.
  - **AT-THE-LIMIT:** limit 0.01.
  - **UNINFORMATIVE:** limit 0.03 or stronger, or none.
- The iso shape is reported alongside, as the detection curve; it does not change the reading. If iso and state
  disagree in Ouro's regime, the paper says so.

## 4. Readings for the paper (fixed now)

- **POWER-AT-OURO-LEVEL:** in a pretrained model too, a class-mean clamp fitted on data as faint as Ouro's a+b (and
  far sparser) still detects a used unwritten value. The Limitations sentence "at Ouro's signal level, only in small
  models trained from scratch" is replaced by "in small models and in two-hop recall in a pretrained model, by degrading
  the clamp's fitting data". Ouro's own result stays "use unresolved": its clamps predate the copy projection and its
  pre-registered comparison (with a-b) proved weak, which v242 does not change.
- **AT-THE-LIMIT:** the paper states that in the pretrained control the clamp's detection limit is 0.01, at Ouro's
  level (0.008-0.010), and that a NOT-USED at that level is weak evidence.
- **UNINFORMATIVE:** the paper states that in a pretrained model the clamp, fitted on data this faint and this sparse,
  misses a used value, so its readings at Ouro's level are uninformative there; the Limitations sentence says so.
- **GROUND-TRUTH-FAILED:** reported, no reading.
- The study degrades the measurement, not the model: a value the model itself codes faintly is not tested (as v231).

Predictions: POWER-AT-OURO-LEVEL 40%, AT-THE-LIMIT 25%, UNINFORMATIVE 30%, GROUND-TRUTH-FAILED 5%.

## 5. Run plan

Freeze -> push as `v242-freeze` (stacked on `v238-freeze`) -> `V242_MODE=faint python build_kaggle_v242.py` -> push
`aurmerin/astrocyte-v242-faint` (GPU T4) -> download to `results_v242/faint/`. The kernel runs the analysis at the end.

**Run note (2026-11-12, after the freeze, before any treatment output).** The push to `aurmerin` failed: first the
account's two GPU slots were taken by another study's kernels, then its weekly GPU quota ran out. The run went to
`aurshalevmerin/astrocyte-v242-faint` instead. The built kernel script (run.py) is the same file for both, SHA-256 bbc66c4d8013b114;
only the account in `kernel-metadata.json` differs. The platform is still Kaggle T4.

## 6. Pre-freeze checks

- `v242_faint.py --selftest` passes (calibration hits each target as a population share; verdict and detection-limit
  logic; rows).
- Dose calibration on pilot 2's saved states (`python v242_faint.py calib results_v235/pilot2/v235_states.npz`; states
  only): native shares 0.1502 / 0.1453 (layer 8, halves 0 / 1), 0.2039 / 0.1983 (12), 0.2370 / 0.2319 (16); every
  target hit as a population share; in-sample estimates on the degraded states: state 0.1 0.096-0.102, 0.03
  0.029-0.032, 0.01 0.009-0.012, 0.005 0.003-0.006, 0.002 0.002-0.004; iso within 0.0003 of every target; every basis
  keeps 24 dims.
- Local mechanics (CPU, a tiny random Qwen2 with Qwen's tokenizer, V242_SMOKE=1, 4 pairs): runs end to end through
  the analysis; in that model the country has no signal, so every dose is flagged as a calibration miss and the ground
  truth fails, as the rules require.
- `astrocyte-v242-smoke` (Kaggle T4, 40 pairs, every learned subspace random; `results_v242/smoke/`): the GPU states
  reproduce pilot 2's exactly (native shares identical to four decimals) and every dose is hit with no calibration miss;
  the whole-state patch moves the answer to the source's capital in 0.925 of pairs and every random-subspace row in
  0.000, so the hooks and the analysis work end to end (the 40 pairs come from few countries, so the intervals are
  degenerate; the smoke's labels mean nothing). 1.3 s per pair with 24 rows.

## 7. Hashes (sha256 of the LF-normalised file, first 16 hex)

- New: `v242_faint.py` e81b40ab5c25ddfe, `build_kaggle_v242.py` 057928ed45b230f3.
- Dependencies, unchanged from v235-freeze: `v235_clamp.py` 089a2bdeba402573, `v235_pilot.py` 442760543b204add,
  `v235_cities.json` 5299759bc458e12e, `v235_pilot2_items.json` 1fae350dc4df2422, `build_kaggle_v215.py` 49ce949efc2c42a8.

## 8. RESULT (2026-11-12; `aurshalevmerin/astrocyte-v242-faint`, Kaggle T4; `results_v242/faint/`)

**AT-THE-LIMIT.** The detection limit is 0.01 for the state shape (primary) and for iso as well.

- The run was clean: 1,038 pairs, 1.53 s per pair, 1,814 s in total. The native shares match pilot 2 to four decimals.
  Every cell hit its target population share exactly, so there were **no calibration misses**. In-sample estimates
  on the degraded states fell within 0.0019 of the target for state and 0.0004 for iso.
- **Ground truth:** full 0.850; full - cR(native) +0.850 [+0.809, +0.885], USED.
- hit(cUo) - hit(cR) for each cell, with the 97.5% cluster-bootstrap interval. hit(cR) is 0.000 in every cell.

| share | state | iso |
|---|---|---|
| native (0.15-0.24) | +0.789 [+0.732, +0.839] USED | (same cell) |
| 0.1 | +0.774 [+0.712, +0.828] USED | +0.771 [+0.709, +0.823] USED |
| 0.03 | +0.654 [+0.588, +0.710] USED | +0.611 [+0.542, +0.672] USED |
| 0.01 | +0.254 [+0.198, +0.312] USED | +0.128 [+0.084, +0.173] USED |
| 0.005 | +0.058 [+0.034, +0.086] UNCLEAR | +0.013 [+0.005, +0.023] UNCLEAR |
| 0.002 | +0.008 [+0.003, +0.013] NOT-USED | +0.002 [+0.000, +0.005] NOT-USED |

- **Reading (section 4, fixed before the run):** in the pretrained control, the clamp's detection limit is 0.01. That
  is at Ouro's level (a+b: about 0.008-0.010), so a NOT-USED at that level is weak evidence.
- **Points the readings did not anticipate (descriptive, no new labels):**
  - The effect does not drop off at the limit; it shrinks steadily. At 0.01 the clamp moves the answer in 25% of
    pairs (state) and 13% (iso), against 79% native.
  - At 0.002 the rules label a value the model does use NOT-USED: the upper bound, 0.013, is under the 0.02 bar.
    Both shapes show this, though state's lower bound is above zero (+0.003). So in this control, a NOT-USED label
    for a value as faint as 0.002 would be wrong.
  - Every label agrees between the two shapes. At 0.01 and 0.005, state's effect is about twice iso's.
- **The paper:** the Limitations sentence and the two-hop paragraph say that in a pretrained model the clamp's
  detection limit, with its fitting data degraded, is 0.01, at Ouro's a+b level. A NOT-USED there is weak evidence,
  and below it the clamp can call a used value NOT-USED. Ouro's own reading stays "use unresolved".
