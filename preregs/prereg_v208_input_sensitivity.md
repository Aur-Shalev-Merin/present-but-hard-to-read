# v208 pre-registration: how a change at the digit's token reaches the summary position, and does removing the network's amplified directions help a probe? (Qwen2.5-1.5B and Ouro-1.4B, Kaggle CPU)

*Drafted 2026-10-18. Status: **FROZEN 2026-10-18** (this commit, pushed to origin before any v208 extraction run). No
v208 statistic of section 3 has been computed on a real model. The pilot ran both real models on 2 prompts for the cache
check, the step check and timing only, and saved no responses. Frozen files (sha256, first 16 hex chars, LF):
- `v208_extract.py` 8d533a2514a0020c and `v208_analyze.py` d5a1546bec7358dd: the pilot kernel ran these exact files.
- `v208_method.py` 1f30cc38627104c2. Its selftest was revised after the pilot (section 5); the measurement code is
  unchanged.
- `build_kaggle_v208.py` 25f36f659d80e914.
- Unchanged dependencies: `v205_extract.py` 11cc20b80558cd77; `v202_extract.py` 1c68dc20dde2d0d8; `loop_grad_probe.py`
  27b80fa0b999bfa7; `v205_analyze.py` ee70344632eb5650; `v178_analyze.py` 90bbc8a848dc5544; `v181_swamp_check.py`
  0a239557c1360a98; `v178_labels.json` 11f0d3852b766479.*

## 1. Why

The Ouro note (v9.1) ends on an open question. v205 and v207 showed three things:
- at the input, a word swap and a random embedding change are equally large (1.42 against 1.46);
- at the last prompt token, only the word swap is readable;
- digits start about half as far apart as words, and their trace at the last token is smaller still than that alone
  predicts.

So why does a change along a real token's direction reach the summary position while random changes of the same size do
not, and are digit changes carried less well than word changes?

A cross-paper idea (the ρ^k paper's non-normal loop maps) also suggested a possible method: if the directions the network
amplifies most are the ones that hide the digit, projecting them out of the states might let a single-prompt probe read
it. The red-team found that the digit's subspace sits in generic high-variance directions, so removing those directions
may remove the signal as well. Part B tests the method directly; the author asked for it to be kept.

Both parts use forward passes only (finite differences), so they run on Kaggle CPU and use no GPU quota.

## 2. Design

**Models and prompts.** Qwen2.5-1.5B and Ouro-1.4B, fp32, frozen, the 4-shot answer-only prompt of v176–v206.
- Prompts: 24 v181 questions with their original digit, one per template (the 'first' anchor of the first 24 templates
  in `v202_extract.families` order).
- j = the anchor's units-digit token.

**Part A: linear response at the summary position** (`v208_extract.py`, `v208_analyze.py`).
- Directions at the input embedding of token j, as unit vectors:
  - DIG: the 9 other digits minus the original;
  - PAIR: 30 random word-pair differences (words as v205's swap words, seed 208);
  - ISO: 64 isotropic Gaussian directions (seed 208);
  - SWAPD: the 10 v205 B1 swap words minus the digit;
  - NOISED: the 10 v205 B1 noise directions.
- Linear response r(u) = (h(e_j + s u) − h(e_j − s u)) / 2s, with s = 0.02 × the median embedding norm (confirmed by
  the pilot's step check, section 5).
- Full-size changes, for the linearity description: the 9 digit swaps, the 10 v205 swap words, and the 10 v205 noise
  vectors.
- Slots: j, the end of the question, and the last prompt token. States: Qwen after layers 4, 7, 14, 21, 28; Ouro mid1 and
  end1–end4.
- Decision slot: the last prompt token. Decision states: Qwen l7–l28; Ouro end1–end4.
- Statistics, per prompt, slot and state, with g²(X) = the mean over class X of |r(u)|²:
  - R_tok = g²(PAIR) / g²(ISO): selectivity for directions in which token embeddings differ. 1 means none. It equals the
    alignment between the network's input sensitivity at j and the embedding covariance.
  - D = g²(DIG) / g²(PAIR): digit-difference directions against word-difference directions, per unit input.
- Intervals: medians over the 24 prompts, with 95% percentile bootstrap intervals over prompts (2,000 resamples, seed
  208).

**Part B: the method test** (`v208_method.py`).
- Amplified output directions A_m: the top-m right singular vectors of the stacked ISO responses at the last slot, over
  all 24 prompts, per model and decision state. This is a randomized range finder for the input-to-last-token Jacobians
  at the varied digit. m = 16 is primary; m = 4 and 64 are descriptive.
- Baselines with the same m:
  - PC: the top-16 principal components of the B2 training pool's own states;
  - RND: a random 16-dim subspace.
- Probe: v205's full-state probe (`v205_analyze.fit_c`, unchanged), trained on the whole B2 training pool (5,249
  questions) and scored on B2's fixed 1,000 test questions. Qwen uses v205 B2; Ouro uses v206 B2. The states are taken
  with A's span removed.
- Statistic: D_X = acc(projected by X) − acc(raw), with a paired bootstrap over test questions (2,000, seed 208).
- The amplified directions come from v208's 24 prompts, not from the B2 questions, so nothing is fitted on test data.

## 3. Labels (per model)

**Part A, Q1 selectivity** (last slot, 4 decision states):
- TUNED: R_tok ≥ 3 with lower bound > 1.5 at ≥ 3 of 4 states.
- FLAT: R_tok ≤ 1.5 with upper bound < 2 at ≥ 3 of 4.
- MIXED otherwise.

**Part A, Q2 digit directions:**
- LOW-GAIN: D ≤ 0.5 with upper bound < 0.7 at ≥ 3 of 4.
- DISTANCE-ONLY: D ≥ 0.8 with lower bound > 0.6 at ≥ 3 of 4.
- MIXED otherwise.

**Linearity** (bearing on the finite changes): the median cos(Δh, |δ| r(u)) at the last slot must be ≥ 0.7 for both the
swap and the noise changes at ≥ 3 of 4 states. Then LINEAR: the Q1 result explains the note's word-vs-noise finding.
Otherwise LOCAL-ONLY: Q1 is reported as a statement about small changes only.

**Part B:**
- HELPS: D_J ≥ 0.05 with lower bound > 0 at ≥ 3 of 4 states. It is then SPECIFIC if D_J − D_PC ≥ 0.03 with lower bound
  > 0 at ≥ 3 of 4, otherwise GENERIC.
- NO-HELP: D_J ≤ 0.02 with upper bound < 0.05 at ≥ 3 of 4.
- MIXED otherwise.

**Descriptive, with no labels:**
- Part A: the ratios at slots j and qmark (where along the way a change is lost), g²(SWAPD)/g²(NOISED), and full-size
  squared gains.
- Part B: the share of B3's digit-mean energy inside A_16 against 16/d; the B3 minimal-pair readout after removing A_16;
  D_J at m = 4 and 64.

## 4. Predictions (written before any real-model response) and what each outcome means

**Q1:**
- Qwen: TUNED 70%, MIXED 20%, FLAT 10%.
- Ouro: TUNED 65%, MIXED 25%, FLAT 10%.
- What it means:
  - TUNED + LINEAR: the word-vs-noise puzzle has a mechanism. The network's input sensitivity at a token is concentrated
    on the directions in which token embeddings vary, and random directions mostly fall outside them. The note can say
    so in one sentence, if the author wants.
  - FLAT or LOCAL-ONLY: the puzzle stays open. The difference then comes from nonlinear effects of large changes.

**Q2:**
- Qwen: LOW-GAIN 45%, DISTANCE-ONLY 25%, MIXED 30%.
- Ouro: LOW-GAIN 40%, DISTANCE-ONLY 25%, MIXED 35%.
- What it means:
  - LOW-GAIN: the digit's small trace has two parts, its input distance (about 2×) and a lower gain along digit
    directions. The descriptive slot-j/qmark ratios then say whether the gain is lost at the token itself or on the way
    to the last token.
  - DISTANCE-ONLY: per unit input, digit directions are carried like word directions. v207's extra SNR gap then comes
    from the context variance in the digit's subspace, not from transport.

**Part B:**
- Qwen: NO-HELP 55%, MIXED 20%, HELPS (GENERIC) 15%, HELPS (SPECIFIC) 10%.
- Ouro: NO-HELP 50%, MIXED 25%, HELPS (GENERIC) 15%, HELPS (SPECIFIC) 10%.
- What it means:
  - HELPS (SPECIFIC): a method candidate that needs neither minimal pairs nor thousands of questions. It would be a
    follow-up paper's starting point, to be replicated before any claim.
  - HELPS (GENERIC): removing high-variance directions helps, but the network's amplified directions add nothing over
    the states' own top PCs. That is known practice and no method.
  - NO-HELP: the method idea is closed. The descriptive digit-energy share says whether this is because the digit lives
    in the amplified directions.

**Where the results go.** This is a new thread, not part of either paper. Any sentence in the note is the author's
call.

## 5. Pilots (timing and checks only)

**Pilot v1** (`astrocyte-v208-pilot`, Kaggle CPU, 2026-10-18; `results_v208/pilot_v1/`). Both real models, 2 prompts
each. No statistic of section 3 was computed and no responses were saved.

| | Qwen2.5-1.5B | Ouro-1.4B |
|---|---|---|
| cache check (max abs state diff, threshold 1e-3) | 1.28e-4 | 1.43e-5 |
| step check, max / median rel. change s vs s/2, at 0.005 / 0.01 / 0.02 / 0.05 × median norm | 5.3e-3 / 2.5e-3 / **1.6e-3** / 9.6e-3 (medians ≤ 3.9e-3) | 1.1e-2 / 5.4e-3 / **2.7e-3** / 7.8e-3 (medians ≤ 7.8e-3) |
| time per forward (segment of 41–50 tokens) | 1.6–1.8 s | 5.6–6.3 s |
| estimated time per prompt (277 forwards) | 7–8 min | 26–29 min |

- **Step.** 0.02 × the median norm has the smallest error in both models, far inside the 0.05 gate, so it is frozen.
- **Shards.** Qwen: 2 shards of 12 prompts, about 1.6 h each. Ouro: 4 shards of 6 prompts, about 2.8 h each. The Kaggle
  limit is 12 h.
- **Part A selftest:** 4/4 worlds.
- **Part B selftest** (scikit-learn, in the pilot): 1 of 2 worlds failed. The failing 'helps' world put the context in
  directions orthogonal to the digit, with the same distribution in training and test. A linear probe learns to ignore
  such directions, so removing them cannot help, and the world was wrong rather than the code.
  - The selftest was rewritten to include the shift that defeats single-prompt probes: test contexts vary along
    directions the training contexts do not.
  - 'specific' world: the responses span those directions, and the label is HELPS (SPECIFIC).
  - 'nohelp' world: the responses span the digit code, and the label is NO-HELP.
  - The GENERIC and MIXED branches of the rule are checked on fixed rows.
  - The revised selftest passes locally (scikit-learn 1.9.1), and it runs again as the first step of each Part B kernel.
  - A side observation from building the selftest: when test contexts are very large along a direction, even a 0.2%
    error in an estimated subspace leaves enough of it to hurt the probe. So D_J against D_PC partly measures how
    precisely each subspace is estimated, not only which directions it contains. This is noted for the reading of
    SPECIFIC; the rule is unchanged.

Local tests (tiny random models, 2026-10-18): the Qwen and Ouro pipelines run end to end on CPU; cache check 1.3e-6; step
check ≤ 3e-3; the Part A selftest passes 4/4.

## 5a. AMENDMENT 1 (2026-10-18, after the freeze, before any Part B statistic)

**Status when written.** The extraction kernels were running and no v208 response had been read. Part B reads the v205
and v206 B2 states, which earlier studies analysed, but none of Part B's projected-probe accuracies had been computed.

**What changed.** A review note pointed at the risk the pilot section already described: the SPECIFIC label could
reflect one subspace being estimated more precisely than the other, rather than different directions. In the synthetic
check, top PCs that captured 99.8% of the nuisance still left the probe at 0.52, against 0.87 for exact directions. So:

- **New control, PC16m:** the top-16 principal components of a random subsample of the training pool (seed 208) with as
  many states as there are nudge responses (24 prompts × 64 = 1,536). Both subspaces are then estimated from the same
  number of vectors.
- **SPECIFIC now requires** D_J − D_PC ≥ 0.03 and D_J − D_PCm ≥ 0.03, each with lower bound > 0, at ≥ 3 of 4 states.
  Otherwise a HELPS result is GENERIC. The HELPS, NO-HELP and MIXED rules are unchanged.
- **Descriptive additions:**
  - D_J for J estimated from each half of the prompts (J16a, J16b);
  - split-half stability of both subspaces: the mean squared cosine between the 16-dim spans estimated from disjoint
    halves (J: two halves of the prompts; PC: two disjoint training subsamples of half the matched size).
  - If J's subspace is much less stable than the PCs', a SPECIFIC result is read with that caveat.

**Files.** `v208_method.py` 1f30cc38627104c2 → 919fe830b0bfa91a: measurement as before plus the three added conditions,
the amended label and the extended selftest, which passes locally. `build_kaggle_v208.py` is unchanged
(25f36f659d80e914). The extraction code is unchanged, so the running kernels are unaffected.

## 6. Run plan

1. Pilot kernel `astrocyte-v208-pilot` (Kaggle CPU, account aurshalevmerin). Record the results in section 5. Fix the
   shard counts from the timing.
2. Freeze: hashes in the header, commit, scan for KGAT_, push to origin.
3. Extraction kernels: `astrocyte-v208-qwen0..`, `astrocyte-v208-ouro0..`.
4. Analysis kernels: `astrocyte-v208-acpu` (Part A), `-bqwen` and `-bouro` (Part B). Part B reads the v205/v206 B2/B3
   outputs, which are on aurshalevmerin, so every v208 kernel runs on that account.
5. Results in section 7 and `docs/v208_result.md`.

## 7. RESULT (2026-10-18; `docs/v208_result.md`, `results_v208/`)

- **Q1: FLAT** in both models (R_tok 1.16–1.28 Qwen, 1.36–1.50 Ouro; predicted TUNED 65–70%).
- **Q2: DISTANCE-ONLY** in both. D is 2.7–8.5 (Qwen) and 4.4–5.0 (Ouro): digit directions are carried more strongly than
  word-pair directions to first order.
- **Linearity: LOCAL-ONLY.** The word swap's linear prediction has cos ≈ 0 with the actual change and is 4.5–15× too small.
  Per the rule, Q1 and Q2 describe small changes only.
- **Part B: NO-HELP** in both. D_J16 is −0.034 to +0.006; plain PCs reach at most +0.018. The amplified 16-dim subspace
  is unstable across halves of the prompts (0.14–0.59, against PCs 0.78–0.92). It holds 25–74% of the digit's
  minimal-pair energy, yet removing it leaves the minimal-pair readout unchanged.
- Reading: the method idea is closed. The note's word-vs-noise difference is a nonlinear, token-identity effect, not
  linear tuning.
