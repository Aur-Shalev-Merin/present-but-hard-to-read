# v205 pre-registration: is "present but swamped" robust? (anisotropy, nonlinear probes, many templates, a meaningless-change null; Study 3's P0 with carry and borrow)

*Drafted 2026-10-16. Status: **FROZEN 2026-10-16** (this commit, pushed to origin before any v205 run). No v205
measurement has been made on real states. The pilots ran the real model on a few prompts for timing and checks only.
The red-team's Ouro v181 overlap nulls (section 1) are known. Frozen files (sha256, first 16 hex chars, LF):
- `v205_extract.py` 11cc20b80558cd77 (pilot v2 ran this file); `v205_saved.py` 386619e2928cd51b and `v205_analyze.py`
  ee70344632eb5650 (the selftest kernel ran these exact files, section 6); `build_kaggle_v205.py` ae6e1f2fd66a8846;
- `v202_extract.py` 1c68dc20dde2d0d8; `v178_analyze.py` 90bbc8a848dc5544; `v181_swamp_check.py` 0a239557c1360a98;
  `v194_analyze.py` 6aa87b4ebe90bd4d; `v186_analyze.py` b408d943603262a6; `v178_labels.json` 11f0d3852b766479.*

## 1. Why

Two outside reviews of the note (v7.4–7.5, 2026-10-16) raised four points that could weaken its headline:
1. **Anisotropy.** The overlap O (7–12 in Ouro, 15–24 in Qwen) divides by the average variance over all dimensions,
   most of which barely vary. A generic direction of the active subspace might score as high. "Qwen crowds more" might
   only mean Qwen's states are more anisotropic.
2. **Template count.** The standard probes saw about 60 training templates. A curve that is flat from 30 to 60 says
   little about the hundreds or thousands of questions that probing studies usually use.
3. **Nonlinear probes.** "No single-prompt *linear* decoder reads it" leaves open whether a small MLP would.
4. **Study 3's P0 signal** (I = +0.017 [0.007, 0.028]) could come from the tens digit Ouro is about to write, which
   depends on the carry and borrow from the units. That is sign-specific. The P0 ceiling did not include them.

A further review suggested the cleanest test of point 1. Measure the same overlap for changes at the digit's position that
carry no meaning: random token swaps and random embedding noise. If they crowd as much as the digit, swamping is simply
what happens to any change, and the note gains a one-line mechanism.

**Red-team before the freeze (2026-10-16).** An independent code-level red-team found no fault in the minimal-pair
probe, the standard probe, the structured decoder, the pairs, the copy ceiling or the patching. It found two serious
issues; this session re-ran the first and got identical numbers:
1. **O is not specific to the digit.** In Ouro v181 at the last token, O for the true digit subspace (end1–end4 6.5–9.4)
   is at or below O for a subspace fitted to labels shuffled within each family (9.0–11.6). The top-9 within-family
   residual PCs give 15–23, and random directions give 0.9–1.1. Any change within a question lands on the directions
   where questions vary most, so the note's "7–12 times larger along the digit's own directions" is generic, not
   aimed at the digit. The in-subspace signal-to-noise (0.5–2%) and the SWAMPED verdict are unaffected.
   **These Ouro v181 numbers are known before the freeze.** A1's shuffled-label and residual-PC nulls on the other
   data (Qwen v202, both designs; Ouro v185) and in B1 are not.
2. **The data are thinner than the text says.** There are 77 distinct questions (a template's two families are one
   question with a different number varied), so a training fold holds about 62 contexts. B2 (6,249 distinct GSM8K
   questions) answers this for Qwen. An Ouro version needs a GPU. Also, a decoder whose covariance includes the test
   question's other variants reads single prompts at about 0.9 (implicit demeaning), so the claim is about decoders
   trained without the test question.

Changes made for this, before the freeze: A1 and B1 report each O against its own nulls (labels shuffled within family,
5 draws; top-9 residual PCs), and B1's mechanism reading uses them (section 5).

## 2. Part A: saved states only (`v205_saved.py`)

**A1. Anisotropy profile (descriptive).**
- Per fold (v178 folds by template), U is fitted on the training families as in `overlap_lda`, and O(U) is overlap_lda's
  O.
- Baseline: O of 50 random 9-dim subspaces drawn inside the top-k principal components of the training prompts' raw
  states (what the standard probe's PCA is fitted on), k = 16, 64, 256. R_k = O(U) / baseline. Also the share of U inside
  the top-k PCs.
- Data: Ouro v181 and Qwen v202 (digit families), and Ouro v185 and Qwen v202-v185 (colour and added digit).
- **Own nulls (added after the red-team):** O for a subspace fitted to labels shuffled within each family (5 draws), and
  O for the top-9 PCs of the within-family residuals. Reported as O/shuffled; for the colour vs digit comparison, the
  ratio of O/shuffled replaces the raw O ratio as the reading.
- **No label.** The selftest shows the baseline depends on k. A code lying generically in a 64-dim active subspace gives
  R64 ≈ 1.0 but R256 ≈ 3.7, while a context concentrated on the code gives R64 ≈ 2.1. So whether the crowding is
  special to the token is judged by B1 below, not by A1.

**A2. MLP probe.**
- StandardScaler → PCA 256 → MLP (one hidden layer of 256, early stopping, α by inner 3-fold grouped CV over
  {1e-4, 1e-2, 1}), with the same outer folds.
- Run on raw (standard) and family-demeaned (contrastive, positive control) states at the last prompt token: Ouro v181
  (mid1, end1–end4) and Qwen v202 (l4–l28).
- Per model, over Q (Ouro end1–end4; Qwen l7–l28):
  - **UNINFORMATIVE:** contrastive MLP max < 0.70.
  - **NONLINEAR-READABLE:** standard MLP max ≥ 0.40.
  - **STILL-SWAMPED:** standard MLP max ≤ 0.30.
  - **PARTIAL:** otherwise.

**A3. Study 3 at P0 with carry and borrow in the ceiling.**
- Ceiling = one-hot(a_u, b_u, c_u) + the first step's carry (plus) or borrow (minus) + the second step's borrow + their
  product. Otherwise v194's statistic is unchanged (I, F, pair bootstrap, seed 194, 97.5% CI).
- Reported at P0 and P1 after loop 4, with the original and the carry/borrow ceiling. The original-ceiling rows reproduce
  v194's +0.017 and +0.157.
- **EXPLAINED:** the P0 carry/borrow CI includes 0. **REMAINS:** its lower bound > 0.

## 3. Part B: new Qwen2.5-1.5B states (`v205_extract.py`, `v205_analyze.py`)

The v202 forward pass, prompt, prefix cache and checks are unchanged; v202's helpers are imported.

**B1. Random-perturbation null.**
- The 137 v181 families with their original digit. At the varied digit's token there are 10 fixed "classes" per kind:
  - **swap:** the digit's token id is replaced by one of 10 fixed random word tokens (' ' + 3–8 lowercase letters, seed
    205), keeping position and length;
  - **noise:** one of 10 fixed Gaussian vectors is added to that token's input embedding, with norm equal to the median
    embedding norm (seed 205).
- 2,740 prompts.
- The same overlap O and probes as the v202 digit families on the same contexts. r = mean O(kind) / mean O(digit) over
  Q = l7, l14, l21, l28.
- Validity per kind: the minimal-pair probe's mean over Q ≥ 0.50 (the change reaches the last token). Otherwise
  **UNINFORMATIVE**.
- **GENERIC:** r ≥ 0.75 for both kinds (any change at that token lands in equally crowded directions).
- **TOKEN-SPECIFIC:** r ≤ 0.50 for both kinds.
- **MIXED:** otherwise.

**B2. Template-count curve** (raised before the freeze after a further review: 1,900 questions might stop too early, and a
flat curve can be flat by construction).
- Every usable GSM8K training question: index ≥ 4, so the four shots are excluded, and containing an integer of ≥ 2
  digits. There are 6,249.
- In each question, the last such integer's units digit is set at random (seed 205). One prompt per question.
- A fixed random 1,000 (permutation seed 2051) are the test set. The other 5,249 are the training pool.
- The run is split into 3 shards of about 2,080 prompts.
- **Nested training subsets.** One permutation (seed 205) of the pool gives N = 60, 125, 250, 500, 1,000, 2,000, N_max/2,
  4,000 and N_max = 5,249. Every point is scored on the same 1,000 test questions.
- **Two probes:**
  - **pca:** the v178 probe (PCA 256, C in {const, 0.01, 0.1, 1});
  - **full:** standardized logistic regression on all 1,536 dimensions, C in {const, 1e-4, 1e-3, 1e-2, 0.1, 1, 10}.

  The chosen C is reported at every N, so a C stuck at the top of its grid is visible. The full-state probe is the one
  for which "more questions than directions" matters, because the pca probe has only 256 inputs.
- **Yardstick, measured:** the effective rank of the training pool's states at each Q layer (participation ratio, and
  the number of components to 95% variance). N_max is stated as a multiple of each.
- **Gate:** the pca probe on the digit's own token (+0, l7) at N_max ≥ 0.90, else **UNSCORABLE**.
- **Label on level and slope.**
  - level = the max over Q and both probes of the accuracy at N_max.
  - gain = the max over Q and probes of acc(N_max) − acc(N_max/2), the last doubling.
  - **STAYS-SWAMPED:** level ≤ 0.30 and gain ≤ 0.02.
  - **CLIMBS:** level ≥ 0.50 or gain ≥ 0.05 (high, or still rising).
  - **PARTIAL:** otherwise.
- Descriptive: an MLP at N_max.
- An Ouro version needs a GPU (no prefix cache in our Ouro code) and waits for quota. It is not part of this freeze. Until
  it runs, the note states the scaling result for Qwen only.

**B3. Positive control for B2.**
- Minimal pairs on the first 150 of B2's test questions, in permutation order, each with its units digit set to 0–9:
  1,500 prompts.
- The v178 probe, contrastive and standard, with folds by question.
- If the contrastive mean over Q is < 0.50, the digit is not shown to be present in B2's prompts. B2's label is then
  reported as **UNINFORMATIVE**, since a flat curve would not show swamping.

## 4. Predictions (before any v205 measurement)

- A2: STILL-SWAMPED in both models 65%, PARTIAL 20%, NONLINEAR-READABLE 15%.
- A3: EXPLAINED 60%.
- B1: GENERIC 50%, MIXED 30%, TOKEN-SPECIFIC 15%, UNINFORMATIVE 5%.
- B2: STAYS-SWAMPED 40%, PARTIAL 30%, CLIMBS 25%, UNINFORMATIVE 5%. B3 present: 90%.

## 5. What each outcome means for the note

- **Overlap wording, whatever the labels:** the abstract's and Section 3.3's "7–12 times larger along the digit's own
  directions" is withdrawn as a digit-specific claim (red-team, verified). The note reports the in-subspace
  signal-to-noise and states that the subspace of any within-question change, shuffled labels included, is equally or
  more crowded. B1 and A1's nulls supply the Qwen numbers for that sentence.
- **B1 GENERIC:** swamping is the default for any change at a token. The note says the variation between questions lies
  along the same directions the network gives to any input change. The abstract's "7–12 times" is replaced by the
  in-subspace signal-to-noise (0.5–2%) plus that sentence. **TOKEN-SPECIFIC:** a token's identity is crowded more than a
  meaningless change, which is a finding to report and leaves the mechanism open. **MIXED:** reported as measured.
- **B2 STAYS-SWAMPED:** the warning applies to probing studies with thousands of questions; the headline gets stronger.
  **CLIMBS:** the note states how many questions a standard probe needs (the curve), and the headline becomes "probes
  trained on tens of templates miss it". **PARTIAL:** both statements, with the curve.
- **A2 STILL-SWAMPED:** "no single-prompt decoder we built" now includes an MLP. **NONLINEAR-READABLE:** the claim
  narrows to linear readouts.
- **A3 EXPLAINED:** the note says the P0 signal is accounted for by the carry and borrow of the output. **REMAINS:** it
  says a small signal remains beyond them.

## 6. Pilot (timing and checks only) and selftests

`astrocyte-v205-pilot` (Kaggle CPU). No probe is fitted on real states in the pilot.

**Pilot v1** (earlier B2 code: 2,400 questions, no shards, no B3; `results_v205/pilot_v1/`):
- B1 with the real model on 3 families: the swap tokens are ' tent', ' stere', ' terra', ' learns', ' bas', ' easiest',
  ' priv', ' counted', ' upright', ' matte'; the noise norm is 1.032 (median embedding norm).
- Cached vs full forward: 6.9e-5 (swap) and 1.5e-4 (noise) on the states, about 2e-5 on the logits.
- Speed: about 2.1 s per prompt (B1) and 2.3 s per prompt (B2).
- Selftests:
  - A1 anisotropy world R64 0.97 against concentrated world 2.13;
  - B2 readable world → CLIMBS, swamped world → STAYS-SWAMPED.

**Pilot v2** (final extraction code; `results_v205/pilot_v2/`):
- B2 shard 2 on 40 questions and B3 on 4 questions with the real model;
- 6,249 usable questions;
- cached vs full forward 1.3e-4 (B2) and 1.1e-4 (B3) on the states, about 2e-5 on the logits;
- about 2.1 s per prompt;
- selftests as in v1.

**Selftest kernel on the final analysis files** (`results_v205/selftest_v2/`; `v205_saved.py` 386619e2,
`v205_analyze.py` ee703446):
- A1: anisotropy world R64 0.99, excess world 2.13;
- B2: readable → CLIMBS, swamped → STAYS-SWAMPED.

Local test (done): every extraction part runs on a tiny random Qwen2, with cached vs full |diff| ≈ 1e-6. 6,249 usable
GSM8K questions.

## 7. Run plan

1. Pilot v2 (section 6).
2. Freeze: hashes of every file. Push to origin before any run below.
3. In parallel on Kaggle CPU:
   - `astrocyte-v205-amain` and `astrocyte-v205-aaur` (Part A);
   - `astrocyte-v205-b1` (about 1.6 h);
   - `astrocyte-v205-b2s0`, `-b2s1`, `-b2s2` (about 1.4 h each);
   - `astrocyte-v205-b3` (about 1 h).
4. `astrocyte-v205-bcpu`: the Part B analysis.

## 8. Amendment (2026-10-16, after the freeze; numerical only, no change to any measurement, rule or label)

`astrocyte-v205-aaur` (Ouro v181) stopped at end4 with "SVD did not converge" (LAPACK, Kaggle CPU). The states are
finite and the same SVD converges locally. `v205_saved.py` now computes right singular vectors through `right_sv`, which
falls back to the eigendecomposition of AᵀA when the SVD fails; the vectors agree to 1e-15 on a test matrix, and the
selftest is unchanged. `v205_saved.py` 386619e2928cd51b → **339d264ae7385d0a**. The partial output of the failed run
(mid1–end3) was seen. It matches the red-team's known shuffled-label numbers; no A2 or A3 output existed. Re-run:
`astrocyte-v205-aaur` version 2. `astrocyte-v205-amain` had not started and runs the fixed file.

## 9. RESULT (2026-10-16; `results_v205/{aaur,amain,bcpu}/`)

Runs: `astrocyte-v205-b1`, `-b2s0/1/2`, `-b3`, `-amain`, `-bcpu` (main account) and `astrocyte-v205-aaur` v2 (second
account), all Kaggle CPU, at the frozen hashes (`v205_saved.py` at the amended 339d264a).

**A2 MLP: STILL-SWAMPED in both models** (predicted 65%).
- Ouro: standard MLP 0.105–0.167 (mid1–end4), contrastive 0.90–0.95.
- Qwen: standard MLP 0.099–0.107 (l4–l28), contrastive 0.70–0.98.

**A3 Study 3 with carry and borrow: EXPLAINED at P0** (predicted 60%).

| | original ceiling | carry/borrow ceiling |
|---|---|---|
| P0 end4 | I +0.017 [0.007, 0.028] | I −0.013 [−0.028, +0.002] |
| P1 end4 | I +0.157 [0.141, 0.172] | I +0.126 [0.107, 0.145] |

- At P0 the small signal is accounted for by the carry/borrow of the output.
- At P1, I stays well above the 0.05 bar with the stricter ceiling (both D's positive: +0.079, +0.174).

**A1 own nulls (descriptive).**
- Digit O / shuffled-label O: Ouro 0.83 (end1–end4); Qwen 1.01 (l7–l28).
- Colour vs added digit, O/shuffled:
  - Ouro: colour 0.53–0.83, digit 0.85–1.13;
  - Qwen: colour 1.03–1.16 and digit 0.96–1.15 at l7–l21; at l28 colour 0.46–0.51 and digit 0.77–0.90.
- Measured against their own nulls, neither token's subspace is more crowded than generic within-question directions. The
  colour is less crowded than its null in Ouro and at Qwen's last layer, where it becomes readable.

**B1 random-perturbation null: GENERIC** (predicted 50%).
- r = O(kind)/O(digit): swap 0.889, noise 1.067. Minimal-pair readout: swap 0.985, noise 0.870 (valid).
- O/shuffled: digit 0.68–1.19, swap 0.50–1.06, noise 0.75–1.24.
- Any change at the digit's token, even meaningless embedding noise, lands in equally crowded directions.
- Descriptive, and important for the reading: the readouts split by signal size, not by crowding.
  - Word swaps change the state far more (in-subspace SNR 0.011–0.205, against the digit's 0.0009–0.0037), and a
    standard probe reads them at 0.24–0.74.
  - Embedding noise is as small as the digit's change (SNR 0.0001–0.0010) and as unreadable (0.105–0.145).

**B3 positive control: present.** Minimal pairs on 150 B2 test questions: 0.81–0.99 (mean over Q 0.912). Standard probe:
0.14–0.21.

**B2 template-count curve: PARTIAL** (predicted 30%).
- Gate: the digit's own token reads at 1.000.
- Effective rank of the training states: participation ratio 9.4 / 14.9 / 24.9 / 43.0 and components to 95% variance
  135 / 213 / 305 / 591 (l7 / l14 / l21 / l28). So N_max = 5,249 is 8.9–38.9 times the 95% rank.
- Best single-prompt accuracy at N_max: 0.190 (l28, full-state probe; chance 0.10). The pca probe reaches 0.108–0.162.
  The MLP reaches 0.113–0.143.
- Gain over the last doubling (2,624 → 5,249): +0.044 at most. That maximum comes from l7's full-state probe (0.100 →
  0.144); at l28 it is +0.009.
- Chosen C is not stuck at the top of the grid at large N (l28 full: 0.01–1 from N = 2,000 on).
- Full curve: l28 full 0.100, 0.114, 0.117, 0.131, 0.127, 0.153, 0.181, 0.188, 0.190 for N = 60 … 5,249.
- So the digit becomes slightly readable with thousands of training questions, rising from chance to 0.14–0.19. It
  stays far below the minimal-pair readout (0.91 on the same prompts), and far below 0.50.

**Reading, per section 5.**
- **Overlap wording.** "7–12 times larger along the digit's own directions" is withdrawn. The variation between questions
  lies along the directions the network gives to any change at that token (B1 GENERIC; A1). What makes the digit
  unreadable is how small its change is: 0.1–0.4% of the variation in its subspace in Qwen, 0.5–2% in Ouro.
- **B2 PARTIAL.** Both statements are true. With tens of templates a standard probe reads nothing. With thousands of
  distinct questions (9–39 times the effective rank) it reads 0.14–0.19, still far below what the minimal pairs show is
  there. This is shown for Qwen; the Ouro curve awaits a GPU.
- **A2.** "No single-prompt decoder we built" now includes an MLP.
- **A3.** The P0 signal is explained by the output's carry and borrow. P1's FOLLOWS survives the stricter ceiling.
