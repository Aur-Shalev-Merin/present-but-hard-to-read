# v181 minimal-pair readout (2026-10-11): the copied digit IS at the last token, swamped by context

Code: `v181_extract.py`, `v181_analyze.py` (rules and interpretation map fixed before the data), `build_kaggle_v181.py`
(95f7490, 82395e2; account aurmerin). Output: `results_v181/v181_analysis.txt`.

**Setup.**
- 137 families from 77 GSM-Symbolic templates. Each family is one question with one units digit set to 0–9.
- Variants are token-for-token identical except that digit; minimal pairs verified 1370/1370.
- Frozen Ouro-1.4B, v176–v180 prompt format. Forward check against Ouro's own logits: |diff| 0.

**Q1: does the state move with the digit?** Within-family spread divided by between-family spread:
- last token: 0.046–0.069 across loops;
- '?': 0.035–0.067;
- +1: about 0.10.
- So the digit moves the later states by about **5–7% of the variation between questions**. That is small, but not zero.

**Q2: can it be read? (chance 0.10; READS ≥ 0.30)**

| slot | standard probe | contrastive probe (family-demeaned) |
|---|---|---|
| +1 | 0.49 (mid1) → 0.14 | 1.00 → 0.67 |
| +3 | 0.18 → 0.12 | 0.94 → 0.32 |
| sentence end | 0.24 → 0.13 | 0.92 → 0.37 |
| '?' | 0.13–0.18 | 0.89–0.91 → 0.67 |
| **last token** | **0.15–0.23** | **0.97 / 0.97 / 0.95 / 0.92 / 0.93** (mid1, end1–4) |

**Reading.**
- This is case 2 of the map fixed before the data: **present but swamped**. The copied digit leaves a trace at the last
  prompt token at every loop. The trace uses a code shared across question types, since the probe is CV-grouped by
  template.
- The trace is about 6% of the context variation, which a standard probe trained across questions cannot pull out.
  At the last token it holds through all 4 loops. At +3 and the sentence end it fades with loops.
- **This corrects v179's reading.** "Values stay on their token" was wrong. Values reach the last token, but as a small
  signal under the context.
- **Scope (review note, agreed).** The contrastive readout needs all 10 variants of the same question. It is a
  presence test, not a decoder for one ordinary prompt. In the note it is written up as "the digit's effect is
  present at the last token, with a shared code", not as a new way to read hidden values.
- **Contrast with v180.** In bare arithmetic the operands dominate the last-token state, and a standard probe reads
  them at 0.96–1.00. In word problems the same information is present, but under about 16× more context variation.

**Value or digit? (exploratory, CPU-only Kaggle kernel `aurmerin/astrocyte-v181-value`, `v181_value_check.py`;
output `results_v181/value/v181_value_check.txt`)**

| slot, state | 10-class acc | mean distance of errors (uniform ≈ 3.3) | R² linear value 0–9 | R² circular | acc with top 1 / 2 / 4 PCs |
|---|---|---|---|---|---|
| last, end1 | 0.968 | 3.75 | 0.57 | 0.48 | 0.21 / 0.25 / 0.45 |
| last, end4 | 0.926 | 3.77 | 0.57 | 0.50 | 0.21 / 0.26 / 0.30 |
| '?', end4 | 0.672 | 3.11 | 0.32 | 0.18 | 0.18 / 0.19 / 0.21 |
| +3, end4 | 0.323 | 3.31 | 0.00 | −0.01 | 0.10 / 0.10 / 0.12 |

**Reading.** The code is not a single number-size direction.
- A magnitude code would give mostly adjacent-digit errors (distance ≈ 1) and would work from the top component.
- Instead, the errors are no closer than random, and 1–4 components give 0.2–0.45, against 0.93–0.97 with 256.
- A linear value component exists (R² 0.57), but the readout relies on a higher-dimensional, digit-identity-like code.
- This points to the copied digit itself, for example its token identity carried by attention, rather than the size
  of the answer in progress, whose change with the digit is template-specific.
- It is still not the clean control. That control is a number that cannot affect the answer.

## "Swamped" measurement (2026-10-13; post hoc; reading rule fixed here before running)

A reviewer asked for a direct test of the note's claim that the copied digit is present but swamped at the last prompt
token. The 5–7% variance ratio alone does not explain why a single-prompt probe fails: a small trace is easy to read
if the context hardly varies along its directions. Script `v181_swamp_check.py`. Data: the v181 states at the last
prompt token, all five loop states, folds grouped by template (the v178 folds). Everything is fitted on training
templates and evaluated on held-out templates.

**Measures, per state:**
1. **Overlap.**
   - Digit subspace: the 9-dimensional span of the family-centred digit means, fitted on training templates.
   - Context covariance: covariance of the held-out family means.
   - Overlap O = context variance per dimension inside the digit subspace ÷ context variance per dimension over all
     2,048 dimensions. O = 1 means context is isotropic, O > 1 means it concentrates on the digit's directions.
   - Also reported: signal-to-noise inside the subspace, SNR_U, against that over all dimensions, SNR_all.
2. **Structured linear decoder.**
   - Shrinkage LDA: digit means from the family-centred training states, plus the Ledoit–Wolf-shrunk covariance of
     training items around their digit means (family offset + residual).
   - It is applied to raw single held-out prompts. This is the best linear single-prompt decoder the minimal-pair
     estimates support, so its held-out accuracy is the empirical ceiling.
   - The same rule on samples from the fitted Gaussian is also reported. That model-based ceiling is optimistic.
3. **Learning curve.**
   - The frozen standard probe (v178 pipeline) is trained on 25 / 50 / 75 / 100% of each training fold's templates
     (3 random draws below 100%) and tested on the full held-out fold.
   - The minimal-pair probe gets the same curve, for comparison.

**Reading** (loop-end states end1–end4):
- **SWAMPED** if the structured decoder is at most 0.40 at every loop end, and the standard probe gains at most 0.05
  from 50% to 100% of templates (mean over loop ends).
- **SAMPLE-LIMITED** if that gain is at least 0.10, or the structured decoder reaches at least 0.60 at any loop end.
- Otherwise **MIXED**.
- O and SNR are descriptive. They say why, not whether.

**Result (2026-10-13; `results_v181/swamp/v181_swamp_check.txt`): SWAMPED.**

| state | overlap O | SNR in digit subspace | SNR all dims | structured LDA | Gaussian ceiling | standard probe 25/50/75/100% templates | minimal-pair probe 25/50/75/100% |
|---|---|---|---|---|---|---|---|
| mid1 | 11.7 | 0.005 | 0.0004 | 0.174 | 0.752 | 0.134 / 0.141 / 0.163 / 0.153 | 0.918 / 0.948 / 0.963 / 0.969 |
| end1 | 7.2 | 0.014 | 0.0006 | 0.170 | 0.904 | 0.155 / 0.175 / 0.187 / 0.193 | 0.918 / 0.953 / 0.964 / 0.968 |
| end2 | 6.5 | 0.023 | 0.0010 | 0.214 | 0.920 | 0.187 / 0.207 / 0.215 / 0.227 | 0.908 / 0.934 / 0.940 / 0.947 |
| end3 | 9.2 | 0.011 | 0.0008 | 0.198 | 0.879 | 0.158 / 0.174 / 0.191 / 0.183 | 0.870 / 0.906 / 0.920 / 0.923 |
| end4 | 9.4 | 0.010 | 0.0006 | 0.177 | 0.854 | 0.152 / 0.171 / 0.178 / 0.185 | 0.879 / 0.908 / 0.918 / 0.926 |

- The structured decoder is at most 0.214 at the loop ends, and the standard probe gains +0.015 from 50% to 100% of
  templates (mean over loop ends). That meets the SWAMPED rule.
- The context varies 6.5–11.7 times more per dimension inside the digit subspace than on average.
- The Gaussian-model ceiling (0.75–0.92) far exceeds the empirical decoder (0.17–0.21). Held-out templates move the
  state along directions the training templates' covariance does not cover.
- The standard probe still rises slightly from 25% to 100% (+0.02 to +0.04). So "unreadable at this scale" is
  established; "unreadable in principle" is not.

## Why is the digit swamped? Mechanism check (2026-10-15; post hoc; reading rule fixed here before running)

An outside review (2026-10-15) asked what causes the crowding, and proposed **same-code interference**. The digit
code is shared across numbers in different roles (v184: a probe trained on needed numbers reads irrelevant ones at
0.78–0.97). So every other number in the prompt may write its own digit code into the same subspace at the last
token, and the target digit is one term in that sum. A rival account is **generic crowding**: the context varies most
along a few directions, and any token's identity subspace overlaps them, whatever the token. Script
`v185_swamp_mechanism.py`; nothing is refitted except what is described here. Data: saved states only, no new forward
passes.

**M1. Overlap by token type** (v185 states: the same 77 contexts with a colour word or a digit varied in an added
sentence; last prompt token; mid1 and end1–end4; v178 folds by template).
- O as in `v181_swamp_check.py` (`overlap_lda`, unchanged), for each family type: col_start, col_end, dig_start,
  dig_end.
- r = mean O(colour) / mean O(digit), over end1–end4 and both insertion points.
- Validity: mean O(digit) ≥ 3. Otherwise **UNINFORMATIVE** (the added digit is not crowded, so there is nothing to
  compare).
- **NUMBER-SPECIFIC** if r ≤ 0.5; **GENERIC** if r ≥ 0.75; **MIXED** otherwise.
- Same-code interference predicts NUMBER-SPECIFIC: prompts hold many numbers and almost no colour words.

**M2. Regression on the other numbers' digits** (v185 digit families, both insertion points pooled; and v181).
- Per fold: fit the 9-dim digit subspace U on the training families (family-centred class means, as in
  `overlap_lda`). Project the held-out families' mean states onto U, centred with the training families' mean.
- Features (primary): for each family, the count of each units digit 0–9 over the other integers in its question
  (v181: every integer except the varied number; v185: every integer of the base question). Secondary: the count of each
  digit value over all digit characters in the question, except the varied one.
- Ridge, α = 1, fitted on the training families. Out-of-sample R² pooled over the 9 dimensions, with the training mean
  as the baseline, averaged over folds by size.
- Controls: the same targets projected onto 20 random orthonormal 9-dim subspaces (seed 185); for v185 also the colour
  subspace, fitted the same way from the colour families.
- ΔR² = R²(U) − the larger of the random subspaces' 95th percentile and (v185) R²(colour subspace). Mean over
  end1–end4.
- **EXPLAINS** if R²(U) ≥ 0.10 and ΔR² ≥ 0.05 in both data sets; **DOES-NOT-EXPLAIN** if R²(U) < 0.05 or ΔR² < 0.02 in
  both; **PARTIAL** otherwise.

**Reading.** NUMBER-SPECIFIC with EXPLAINS supports same-code interference: the note says the other numbers' traces
crowd the digit's subspace, and a decoder that subtracts them becomes worth trying. GENERIC with DOES-NOT-EXPLAIN
supports generic crowding: the note says any token's subspace sits where the context varies most. Other combinations
are reported as they come, without a mechanism claim.

**M3 (descriptive; not about swamping). Study 3's copies at P0, every state** (v194 states, `v194_analyze.job`
unchanged). The paper's P0 sentence (Section 5.4) uses the best-state copy of b; this gives its value after loop 4,
where the P0 test was read. If b after loop 4 is at least 0.85 in both versions, the note may say Study 3's P0 test has
strong copies; otherwise it states the value.

**Result (2026-10-15; `results_v185/mech/v185_mech_v185.txt`, `results_v181/mech/v185_mech_v181.txt`,
`results_v185/mech/v185_mech_v194.txt`; kernels `astrocyte-v185-mech`, `astrocyte-v181-mech`, files at the hashes of
commit 23e8ad4).** No mechanism claim: M1 MIXED, M2 DOES-NOT-EXPLAIN.

**M1: MIXED** (r = 0.551; validity met, mean O(digit) 9.75 ≥ 3).

| type | mid1 | end1 | end2 | end3 | end4 |
|---|---|---|---|---|---|
| colour, start | 12.96 | 6.87 | 4.18 | 4.69 | 4.05 |
| colour, end | 13.26 | 6.92 | 4.73 | 6.19 | 5.35 |
| digit, start | 8.88 | 10.58 | 8.79 | 8.69 | 8.12 |
| digit, end | 10.89 | 12.39 | 8.07 | 12.17 | 9.17 |

- The colour word's subspace is crowded too: O 4–7 at the loop ends (13 at mid1), far above 1. So much of the crowding
  is generic.
- The digit's is about twice as crowded (8–12, the same range as v181's 6.5–11.7) and stays so across loops, while the
  colour's falls from end1 to end4.
- Descriptive: the structured single-prompt decoder reads the colour better than the digit at late loops (0.28–0.38
  against 0.15–0.16 at end4), in line with the standard probe (colour up to 0.40).

**M2: DOES-NOT-EXPLAIN** in both data sets (R²(U) < 0.05).

| data, features (end1–end4) | R²(digit subspace) | ΔR² |
|---|---|---|
| v185, units digits (primary) | −0.091 | −0.008 |
| v185, all digit characters | −0.108 | +0.009 |
| v181, units digits (primary) | −0.135 | −0.050 |
| v181, all digit characters | −0.206 | −0.090 |

- Every out-of-sample R² is negative, for the digit subspace, the random subspaces and the colour subspace alike. That
  is what a 10-feature ridge on about 120 training families gives when the true R² is near zero.
- The synthetic selftest shows the regression finds the effect when other numbers' codes add into the subspace (R²
  0.87). So an additive, position-independent code from the other numbers does not explain where a context sits in the
  digit subspace. Codes that depend on position or role are not tested.

**Reading (per the rule: other combination, no mechanism claim).** The crowding is partly generic: a colour word's
subspace is crowded too, at about half the digit's level. The extra crowding of the digit's subspace is not predicted by
the other numbers' digits. The note can say that the crowding is not specific to numbers and that the other numbers'
digits do not account for it; it has no mechanism.

**M3 (descriptive): Study 3's copies at P0, by state** (mid1 end1 mid2 end2 mid3 end3 mid4 end4).

| | a | b | c |
|---|---|---|---|
| plus | 1.000 … 0.995 | 0.979 0.944 0.932 0.923 0.811 0.829 0.691 0.770 | 0.998 … 0.959 |
| minus | 1.000 … 0.998 | 0.985 0.950 0.958 0.939 0.903 0.898 0.847 0.808 | 0.995 … 0.765 |

- The copy of b at P0 after loop 4 is 0.770 (plus) and 0.808 (minus): below the 0.85 bar, so the note states the value
  rather than calling Study 3's P0 test strong. It is above Study 1's 0.66.
