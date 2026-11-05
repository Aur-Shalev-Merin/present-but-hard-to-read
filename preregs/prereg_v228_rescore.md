# v228 pre-registration: re-scoring the note's targets with the validated share swap

Workstream paper3 (Ouro note). Claimed 2026-10-28. Follows v224 and v227 (the share swap is VALID).

## 1. Why

The accuracy-based operation swap failed its positive control (v221), so the note demoted it: the Kudo et al.
agreement rests mostly on it, and Ouro's FOLLOWS carries the caveat that generic mixing also earns it. A share-based
swap (v224) has since passed two positive controls it was never tuned on (Qwen2.5-7B a step, I / N 0.65; Mistral-7B
b step, I / N 0.48, v227) and no null world (0 / 420), with TAU 0.30 set from a 420-run null grid. This study applies
it, frozen, to the note's targets.

**Disclosures.** (1) Every target's result under the old swap is known. (2) The short-chain eq / last positions were
seen under the share statistic in v223 and v224's development run, so they are not re-scored here (their dev numbers
are reported as such: eq UNCLEAR, D+ -0.051, D- +0.087; last UNCLEAR). (3) Control 1 of v224 was replaced (v227)
after v224's results were known. (4) Ouro-1.4B is not one of the validation models.

## 2. Statistic (`v228_rescore.py`, frozen)

`v224_share.analyse` with two generalisations fixed here, before any target state is touched:

- **Copies:** one-hot of every digit varied in the prompt (Kudo: d1, d2; Ouro Study 3: the units digits of a, b, c).
  In v224 / v227 these were the step's two operands, the only varied digits there.
- **Answer:** where the answer is about to be written or fixes b within a family, its digit's one-hot is regressed
  out with the copies (primary). Selftest: an answer-only world (no b) is FOLLOWS with copies only (I / N 0.79) and
  NO-SIGNAL with the answer regressed; an answer + b world stays FOLLOWS with the answer regressed (I / N 0.78).
- Bootstrap cluster: whatever links a plus family to its minus twin (Kudo: fam; Ouro: pair).
- Labels S = (u + v) mod 5, M = (u - v) mod 5 from the step's operands (Kudo d1, d2; Ouro a, b units), R0..R3, TAU 0.30,
  FOLLOWS / DOES-NOT-FOLLOW / UNCLEAR / NO-SIGNAL exactly as v224 section 2; 10,000 resamples, seed 2241, 97.5% CIs.
- Accuracy gates: each original study's (all passed there); not re-applied.
- Selftest: with copies (u, v) and cluster fam, the generalised code reproduces v224_share.analyse exactly.

## 3. Targets (states from the original extractions, aurshalevmerin)

- **T1, Kudo et al.'s chain format** (v215 fam states, Qwen2.5-7B, layers 16 / 20 / 24 / 28): **eq** (end of b's
  equation; primary), last. Copies d1, d2 (the validated form). Descriptive: the answer regressed too.
- **T2, answer only** (v216 fam states): **eq, pre (the token before the answer digit), last** (each primary).
  Copies d1, d2 and the answer A regressed. Descriptive: copies only.
- **T3, Ouro-1.4B Study 3** (v194 states, main set, 36 plus / 36 minus families): **P1** (the answer token before the
  units digit; primary), P0 (last prompt token). Copies a, b, c and the answer's units digit regressed. States: the last
  two loops, mid3 / end3 / mid4 / end4 (the upper half of the effective depth, as v224's layers are of Qwen's).
  Descriptive: copies only.

## 4. Predictions and readings

Predictions: T1 eq not FOLLOWS 75%; T2 pre not FOLLOWS 65%; T3 P1 FOLLOWS 50%.

- **T1 / T2, DOES-NOT-FOLLOW or NO-SIGNAL:** no sign-specific b at that position by a swap shown to detect one; the
  note's agreement with Kudo et al. at that position rests on a working test (with the clamps and patches).
- **T1 / T2, FOLLOWS:** a sign-specific code of b at that position (T1: if the answer-regressed descriptive is also
  FOLLOWS, of b rather than the answer); the note reports a disagreement with Kudo et al. there.
- **T1 / T2, UNCLEAR:** as now (not resolved).
- **T3 P1, FOLLOWS:** Ouro's sign-following stands with a validated test, beyond the copies and the answer, and beyond
  the null-calibrated bar (so the "generic mixing also earns it" caveat is replaced); the note keeps the operand form
  caveat (a + b and "answer + c" are the same function on units digits) and that Ouro is not a validation model.
- **T3 P1, UNCLEAR / DOES-NOT-FOLLOW / NO-SIGNAL:** the note drops "its sign follows the operation" for Ouro
  ("not shown" or "does not", by label).

## 5. Pre-freeze checks

- Selftest (`python v228_rescore.py --selftest`): reproduces v224 (I +0.08595 both, FOLLOWS both); answer-only world
  FOLLOWS -> NO-SIGNAL with the answer regressed; answer + computed world FOLLOWS (I / N 0.78).
- Inputs: v215 items sha 8c08adec21d20ae3 (asserted against the saved meta); v216 items 497db0bcb9748f30 (asserted);
  v194 merged meta items (main set, 14,400).
- No share statistic has been computed on any T1-T3 state.

## 6. Frozen files (sha256, first 16 hex, LF; 2026-10-28)

- `v228_rescore.py` 1d1e588493f56ba0
- `build_kaggle_v228.py` 2152c4d4e0ef6b47
- reused unchanged: `v224_share.py` 2b0bc8868d13b00e (TAU 0.30), `v216_items.py` a2ebd48b899fed40, `v215_items.py`
  9249f0523d7219eb, `v178_analyze.py` 90bbc8a848dc5544, and v224's other frozen files

Run: `astrocyte-v228-rescore` (CPU, aurshalevmerin; sources astrocyte-v215-x, astrocyte-v216-x, astrocyte-v194-ouro).

## 7. RESULT (2026-10-28; frozen code, `aurshalevmerin/astrocyte-v228-rescore`)

| target | position | N | D+ | D- | I [97.5% CI] | I / N | label |
|---|---|---|---|---|---|---|---|
| T1 Kudo, chain | **eq** | 0.077 | -0.059 | +0.099 | +0.020 [0.019, 0.022] | 0.26 | **UNCLEAR** |
| | last | 0.018 | -0.006 | +0.025 | +0.010 [0.009, 0.011] | 0.56 | UNCLEAR |
| T2 Kudo, answer only (answer regressed) | **eq** | 0.055 | -0.044 | +0.057 | +0.006 [0.004, 0.008] | 0.11 | **UNCLEAR** |
| | **pre** | 0.039 | -0.010 | +0.041 | +0.015 [0.013, 0.017] | 0.39 | **UNCLEAR** |
| | **last** | 0.023 | -0.006 | +0.033 | +0.013 [0.011, 0.016] | 0.59 | **UNCLEAR** |
| T3 Ouro Study 3 (copies and answer regressed) | **P1** | 0.007 | +0.002 | +0.007 | +0.004 [0.003, 0.005] | 0.63 | **FOLLOWS** |
| | P0 | 0.001 | -0.000 | +0.001 | +0.000 [-0.000, 0.001] | 0.40 | UNCLEAR |

Descriptive variants agree: T1 with the answer regressed UNCLEAR at eq and last; T2 with copies only UNCLEAR at all
three; T3 P1 with copies only FOLLOWS (I / N 0.64), P0 UNCLEAR.

Predictions: T1 eq not FOLLOWS (75%) right; T2 pre not FOLLOWS (65%) right; T3 P1 FOLLOWS (50%) right.

**Reading (section 4).**
- **Kudo et al. (T1, T2): UNCLEAR at every position, with D+ < 0 everywhere**: on plus problems the difference explains
  more than the sum, so the state carries a fixed preference for the difference on both kinds of problem, not a code
  that follows the operation. The validated swap does not resolve these positions; the note's agreement with Kudo et
  al. rests on the clamps and the whole-state patches, not on the swap.
- **Ouro (T3): P1 FOLLOWS**, beyond the copies and the answer and above the null-calibrated bar: the sign-following
  stands with a validated test, and the "generic mixing also earns it" caveat is replaced. Caveats the note keeps:
  a + b and "answer + c" are the same function on units digits; Ouro is not a validation model; and the effect is
  small in share terms (N 0.007, against 0.11-0.15 in the positive controls), although its CI is well clear of zero.
