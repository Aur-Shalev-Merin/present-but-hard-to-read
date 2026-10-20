# v212 pre-registration: a calibrated null for the protocol's mixture tests (outside review W3)

*Drafted 2026-10-19. Status: **FROZEN 2026-10-19** (this commit, pushed to origin, branch `v212-freeze`, before
stage 1). Synthetic data only; no model is run. Before the freeze only the selftests (section 5) were run, on worlds
with seed 99, which no stage uses. Frozen files (sha256, first 16 hex chars, LF):
- `v212_mixture_null.py` f312680b6e87a5b8; `build_kaggle_v212.py` 7ab7db78bc07ddfe;
- reused: `v194_analyze.py` 6aa87b4ebe90bd4d (v194's amended, frozen file); `v186_analyze.py` b408d943603262a6;
  `v178_analyze.py` 90bbc8a848dc5544; `v194_items.json` fceceeb4dcbd4076.*

## 1. Why

The note's protocol compares the target value with the alternatives a generic mixture of the inputs would also make
readable. Its line test asks whether the needed form (a + b on plus problems, a − b on minus problems, each the answer
plus c) tops the other forms on its line answer + k c. In Study 3 it does, after loop 4, by +0.036 (plus) and +0.045
(minus). The note assumes that a generic mixture would make the four forms on a line about equally readable. An
outside review of note v9.2 (W3) pointed out that this is asserted, not shown:
- same-frequency products of periodic number features (which the note cites) favour answer ± c over answer ± 2c, a
  structured, non-computational reason for the needed form to read well;
- margins of +0.036 to +0.075 are hard to interpret without the spread generic mixing produces.

v212 measures that spread on synthetic states.

## 2. Design (`v212_mixture_null.py`)

**Items:** Study 3's real items (`v194_items.json`, main and bare sets), so the families, templates, folds, label
distributions and twin pairs are the real ones.

**Synthetic P1 state** (as v194's selftest, d = 64), with no a + b or a − b term:
- family offset (20 · N(0, 1) per family), noise 0.05, copies W_a[a] + W_b[b] + W_c[c], and a strong answer code
  3 · W_ans[answer units digit];
- plus one generic nonlinear mixture M of the four units digits (a, b, c, answer), scaled to λ times the RMS norm of
  the answer code:
  - **pair:** the sum over the six pairs (x, y) of tanh(2 · U_x[x] ⊙ V_y[y]) with random codes;
  - **mlp:** a random one-layer network, R2 · relu(R1 [one-hot a; b; c; answer] + r0), 256 hidden units;
  - **fourier:** a random projection of the same-frequency products of cos and sin(2π f x / 10) across the six pairs,
    f ∈ {1, 2, 5} (the reviewer's case).
- **Two stages** (selftest v2 showed that at λ ≥ 0.8 the needed form reads at the ceiling, +0.775, far above Ouro's
  +0.237, so the strength must be calibrated, not gridded):
  - **Stage 1, calibration:** seed 100 only, λ ∈ {0.005, 0.01, 0.02, 0.04, 0.08}, per family; only the needed form's
    excess is computed (the calibration target; no line margin or I).
  - **Matched strength λ\*** (frozen rule, `match`): log-linear interpolation of the needed excess against log λ to
    Study 3's observed 0.237, between the two bracketing grid points, rounded to 3 significant figures. If the curve
    never brackets 0.237, the family is reported as UNCALIBRATED and stage 1 is repeated on the next grid down or up
    (×1/16 or ×16) before any stage-2 run.
  - **Stage 2:** seeds {0, 1, 2, 3, 4, 5} at λ\*, full statistics: 12 line margins per family (6 seeds × 2 versions).
- The copy-ceiling fits do not depend on the world and are computed once per kernel.

**Statistics per world**, with the note's own code (`v194_analyze` probe jobs: the v178 probe on family-demeaned
states, mod-5 labels, copy ceiling = one-hot a, b, c, folds by template):
- needed: the needed form's excess over its ceiling (mean of plus and minus);
- line: needed − max over the other three forms on its line, per version (the note's line test);
- I = (D+ + D−)/2, the operation-swap statistic.

## 3. Reading rule (fixed now)

Per family, at λ\* (12 null line margins):
- **SEPARABLE:** both observed margins (+0.036, +0.045) exceed the largest null margin.
- **NOT-SEPARABLE:** the null's mean margin is at least the observed mean (+0.0405).
- **PARTLY:** otherwise.
- Overall label: the worst family (NOT-SEPARABLE < PARTLY < SEPARABLE).

**Descriptive:** I under each family at λ\*, against the observed +0.157. Since the needed form equals the
answer plus c in both versions, a generic answer × c mixture can produce I > 0 by itself; the note already says that
the labels cannot separate the two readings, and this measures how large such an I gets.

## 4. Predictions and what each outcome means for the note

- Predicted: SEPARABLE 45%, PARTLY 35%, NOT-SEPARABLE 20%. Fourier mixing favours answer + c and answer − c equally,
  and answer − c is on the line, so it should not produce a positive line margin on its own; random MLPs should give
  margins scattered around a negative mean (the needed form is one of four).
- **SEPARABLE:** the note states that the line margins exceed what three families of generic mixing produce at
  matched strength, and drops the limitation sentence on uncalibrated mixtures.
- **PARTLY:** the note gives the null's range next to the observed margins and keeps the line test as weak evidence.
- **NOT-SEPARABLE:** the note says that generic mixing reproduces the line margins, so the line test does not
  separate a held a ± b from a mixture of the answer with c; the computation claim is reduced to the operation-swap
  result.
- In every case the note reports the null's I next to the observed one.

## 5. Selftest (`v212_mixture_null.py selftest`, Kaggle CPU)

Worlds use seed 99, which is neither a calibration nor a stage-2 seed.
- **Version 1** (seed 0 for the Fourier world, which was then a grid seed) was superseded before it finished; its
  output was not read.
- **Version 2** (`results_v212/selftest/`, the single-stage design): explicit needed-form code → needed +0.775, line
  +0.709 / +0.735, I +0.758 (OK); pure Fourier mixing at λ = 0.8 → needed +0.775, line +0.009 / −0.003, I +0.059 (OK).
  This showed the saturation that led to the two-stage design, and about 21 min per world (before caching the
  ceiling).
- **Version 3** (this design, `results_v212/selftest_v3/`): explicit needed-form code → needed +0.773, line
  +0.709 / +0.735 (OK); pure Fourier mixing at λ = 0.8 → line +0.009 / −0.003 (OK); matching rule on the toy curve →
  λ\* = 0.0283 (want 0.0283, OK). With BLAS pinned to one thread per worker, both worlds took about 1 min in all.

## 6. Run plan

1. Selftest v3; freeze (hashes); push before stage 1.
2. Stage 1: `astrocyte-v212-cal-pair`, `-cal-mlp`, `-cal-fourier` (Kaggle CPU); `match` → λ\* per family, recorded here.
3. Stage 2: `astrocyte-v212-run-<family>` at λ\*.
4. `analyse` on the downloaded rows (seconds); RESULT here; note update per section 4.

## 7. Stage 1 (2026-10-19; `results_v212/cal_*/`, `results_v212/v212_match.txt`)

Needed-form excess at seed 100 (no line margin or I computed):

| λ | 0.005 | 0.01 | 0.02 | 0.04 | 0.08 |
|---|---|---|---|---|---|
| pair | 0.012 | 0.033 | 0.079 | 0.127 | 0.157 |
| mlp | 0.002 | 0.004 | 0.015 | 0.044 | 0.100 |
| fourier | 0.039 | 0.119 | 0.259 | 0.442 | 0.600 |

`match`: fourier λ\* = 0.018; pair and mlp never reach 0.237 on this grid, so by section 2 they are UNCALIBRATED and
stage 1 is repeated for them on the grid ×16, {0.08, 0.16, 0.32, 0.64, 1.28}, before any stage-2 run for them.

**Amendment 1** (before that repeat and before any stage-2 run): `build_kaggle_v212.py` (7ab7db78bc07ddfe →
d7a54621ae2d3308) passes an optional λ list (`V212_LAMS`) to `calib`, which the frozen `v212_mixture_null.py`
already accepts. Nothing else changed. Stage 2 for fourier runs at λ\* = 0.018 now.

**Stage 1, repeat for pair and mlp (grid ×16, then ×16 again for mlp; `results_v212/cal16_*/`, `cal256_mlp/`).**
- pair: 0.157 / 0.171 / 0.187 / 0.216 / 0.243 at λ = 0.08 … 1.28 → λ\* = 1.09 (`results_v212/v212_match2.txt`).
- mlp: 0.100 / 0.141 / 0.169 / 0.183 / 0.188 at λ = 0.08 … 1.28, and 0.188 / 0.195 / 0.195 / 0.195 / 0.196 at
  λ = 1.28 … 20.48. It levels off below 0.237, so mlp is **UNCALIBRATED**: a random one-layer network of the four
  digits never makes the needed form as readable as it is in Ouro. No stage 2 is run for it.
- The λ = 0.08 and 1.28 rows, computed twice, are identical (same seed, same world).

## 8. RESULT (2026-10-19; `results_v212/v212_analysis.txt`, stage-2 rows in `results_v212/run_*/`)

| family | λ\* | needed (stage 2) | null line margins (6 seeds × plus, minus) | mean | max | label | null I mean / max |
|---|---|---|---|---|---|---|---|
| pair | 1.09 | 0.273 | +0.023 … +0.119 | +0.078 | +0.119 | NOT-SEPARABLE | +0.095 / +0.135 |
| fourier | 0.018 | 0.208 | −0.063 … +0.123 | +0.051 | +0.123 | NOT-SEPARABLE | +0.085 / +0.109 |
| mlp | — | — | — | — | — | UNCALIBRATED | — |

Observed in Ouro (Study 3, P1 end4): line margins +0.036 / +0.045 (mean +0.0405); I +0.157.

**Overall: NOT-SEPARABLE** (predicted 20%).

**Descriptive.**
- In the Fourier worlds the forms on each line read: needed (answer + c) 0.16–0.27, answer − c 0.13–0.22, answer ± 2c
  0.01–0.03. The reviewer's mechanism holds for ± 2c. The needed form also beats answer − c, because it is a function
  of two inputs (a ± b), which pairwise mixing makes readable directly, while answer − c needs all three.
- Generic mixing with no a ± b term produces a sign-following I of +0.07 to +0.14 at matched strength. Ouro's +0.157
  is above every null world, but mixing accounts for much of it.

**Reading, per section 4 (NOT-SEPARABLE).** Generic mixing of the inputs with the answer reproduces the line margins,
so the line test does not separate a held a ± b from such a mixture. The note's computation claim is reduced to the
operation-swap result, reported next to the null's I (+0.09 on average, at most +0.14, against +0.157), and the
protocol's line step is described as weak against pairwise mixing.
