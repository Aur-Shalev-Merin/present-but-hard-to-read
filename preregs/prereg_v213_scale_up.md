# v213 pre-registration: does "present but hard to read" hold at larger scale and with many more questions?

*Drafted 2026-10-21. Status: **FROZEN 2026-10-21** (this commit, pushed to origin, branch `v213-freeze`, before any
extraction run). No v213 probe has been fitted on real states. Before the freeze, only checks that compute no probe
statistic were run: a GPU pilot (timing and correctness), a smoke test of every extraction part on a few prompts, the
analysis selftest on synthetic states, and data checks on the item pool (section 5). Frozen files (sha256, first 16
hex chars, LF):
- `v213_extract.py` 8c65d944e204cb1f; `v213_items.py` 9e0d413d39366d38 (ITEMS_SHA db8f37a5f24b8c64);
  `v213_analyze.py` fafebeb1ab446d67 (the Kaggle selftest ran this exact file); `build_kaggle_v213.py` 3c8cba00aa845dfa;
  `v213_pilot.py` b5bdf9e864fa6c23.
- reused, identical to their earlier frozen versions: `v202_extract.py` 1c68dc20dde2d0d8; `v205_extract.py`
  11cc20b80558cd77; `v202_analyze.py` 46d9d8aca3bf0459; `v205_analyze.py` ee70344632eb5650; `v205_saved.py`
  339d264ae7385d0a; `v178_analyze.py` 90bbc8a848dc5544; `v181_swamp_check.py` 0a239557c1360a98;
  `v178_labels.json` 11f0d3852b766479.
- The smoke test ran `v213_extract.py` as frozen and an earlier `v213_items.py` (before the reworded-twin filter and
  the hash constant); the frozen items were rebuilt twice locally and once on Kaggle with the same hash.*

## 1. Why

The arXiv note (2610.28498) shows that a digit copied from the question is present at the last prompt token
(minimal pairs about 0.9) yet hard to read from single prompts, in Ouro-1.4B and Qwen2.5-1.5B. Two objections recur in
the outside reviews and are the obvious ones for a conference version:
- **Scale:** two models of about 1.5B parameters.
- **"Still rising":** with 5,249 training questions the best single-prompt probe reaches 0.36 in Ouro and 0.19 in
  Qwen, and Ouro's curve was still climbing (+0.060 over the last doubling). More data might close the gap.

v213 repeats the note's Section 3 measurements in a larger standard model (Qwen2.5-7B) and a larger looped model
(Ouro-2.6B), and extends the many-question curve from 5,249 to 45,249 training questions.

## 2. Design

**Models** (`v213_extract.py`):
- **qwen7b:** Qwen/Qwen2.5-7B (28 layers, d = 3,584, one token per digit), fp16 over two T4 GPUs. Stored states after
  layers 0, 4, 7, 14, 21, 28 (as v202); Q = l7, l14, l21, l28.
- **ouro26:** ByteDance/Ouro-2.6B (48 layers × 4 loops, d = 2,048; model code identical to Ouro-1.4B's), fp32 on one
  T4. Stored states h0, mid1 (after layer 24 of loop 1), end1–end4; Q = end1–end4.
- Same 4-shot answer-only prompt as all earlier studies. Prefix cache as v202 (Qwen) and v206 (Ouro), checked against a
  full forward on the first prompt of each part: Ouro absolute max difference < 1e-3; Qwen in fp16, relative difference
  < 1e-2 (the extraction aborts otherwise).

**Parts:**
- **v181:** the 137 minimal-pair families of the note (v202_extract.families), digit 0–9 (1,370 prompts per model).
- **b1:** v205's random-perturbation null on the same families: 10 fixed random word tokens swapped in, and 10 fixed
  Gaussian vectors of the median input-embedding norm added to the digit's embedding (2,740 prompts).
- **b2:** the many-question pool (`v213_items.py`), one prompt per question with the units digit of the last integer of
  two or more digits set at random:
  - GSM8K: v205's items unchanged, 5,249 training and 1,000 test questions.
  - Orca-Math (microsoft/orca-math-word-problems-200k): usable questions without a GSM8K near-duplicate (4-word
    shingles, numbers masked, ≥ 30% shared); 1,000 test questions; 40,000 training questions, each checked to be
    neither a near-duplicate of an Orca test question nor a **reworded twin** of any test question, GSM8K or Orca
    (content-word Jaccard ≥ 0.30 against every question of each test set that shares a word occurring in at most 30
    questions of that set). Test questions with a reworded twin in the GSM8K training pool are flagged.
  - Item-list hash (sha256 prefix of the canonical JSON): **db8f37a5f24b8c64**; every kernel rebuilds the items and
    aborts on a different hash.
- **b3:** minimal pairs on the first 100 GSM8K and the first 100 Orca test questions (2,000 prompts).

**Analysis** (`v213_analyze.py`): the frozen v202 and v205 functions, with v206's renaming for Ouro (end1 → 'l7'), and an
extended many-question curve (B2x):
- Training order: the GSM8K training pool in item order, shuffled with seed 205 (the same kind of random subset as
  v205/v206, not the identical subsets), then the Orca training pool in its fixed order.
- Grid: 60, 125, 250, 500, 1,000, 2,000, 2,624, 4,000, 5,249, 10,000, 20,000, 22,624 (N_max/2), 45,249 (N_max).
- Probes as v205 (PCA-256 and full-state logistic regression, C chosen by inner 3-fold CV grouped by question; MLP at
  N_max). Each fit is scored on the GSM8K test set (primary), the Orca test set, and both test sets without their
  twin-flagged questions (secondary).

## 3. Labels (per model)

- **A (v202's rule, unchanged), last prompt token over Q:** ABSENT if the minimal-pair probe is < 0.30 at every Q state;
  READABLE if the standard probe ≥ 0.50 or the structured decoder ≥ 0.60; REPLICATES if minimal pairs ≥ 0.80 and the
  standard probe ≤ 0.30 and the decoder ≤ 0.40 and the gain from half to all templates ≤ 0.05; else MIXED.
  UNSCORABLE if fewer than 99% of minimal pairs are valid, the standard probe reads the digit on its own token below
  0.90, or the cached-vs-full check fails.
- **B1 (v205's rule, unchanged):** GENERIC / TOKEN-SPECIFIC / MIXED / UNINFORMATIVE, from the overlap of the word swap
  and the noise relative to the digit's (r ≥ 0.75 generic, ≤ 0.50 token-specific) and their minimal-pair readouts.
- **B2x (new), GSM8K test set, at N_max over Q and both probes:** level = the best accuracy; gain = the best
  acc(N_max) − acc(N_max/2).
  - **CLOSES:** level ≥ 0.80.
  - **STILL-CLIMBING:** level < 0.80 and gain ≥ 0.05.
  - **PLATEAUS-BELOW:** level ≤ 0.50 and gain ≤ 0.02 on the GSM8K test set, **and** level ≤ 0.50 and gain ≤ 0.02 on the
    Orca test set (the added data's own distribution).
  - **SHIFT-LIMITED:** flat and low on the GSM8K test set (level ≤ 0.50, gain ≤ 0.02) but not on the Orca test set: the
    added questions help their own distribution and not GSM8K, so the curve cannot show that more data fails to help.
  - **PARTIAL:** otherwise.
  - UNSCORABLE if the PCA probe on the digit's own token reads it below 0.90 at N_max.
  - The GSM8K-only part (N ≤ 5,249) is also labelled with v205's rule, for comparison with v205/v206.
- **B3 (v205's rule):** the minimal-pair mean over Q ≥ 0.50 on each test set = present. B2x is UNINFORMATIVE if the
  GSM8K test questions are not shown present.

## 4. Predictions and what each outcome means for the paper

| | REPLICATES | MIXED | READABLE | other |
|---|---|---|---|---|
| A, qwen7b | 55% | 25% | 15% | 5% |
| A, ouro26 | 60% | 25% | 10% | 5% |

| | CLOSES | STILL-CLIMBING | PARTIAL | PLATEAUS-BELOW | SHIFT-LIMITED |
|---|---|---|---|---|---|
| B2x, qwen7b | 10% | 35% | 30% | 15% | 10% |
| B2x, ouro26 | 15% | 45% | 25% | 5% | 10% |

B1: GENERIC 55% in each model.

Reading rules (fixed now):
- **A REPLICATES in both:** the paper says the presence-vs-readability gap holds in a 7B standard model and a 2.6B
  looped model. **READABLE in either:** the paper says single prompts read the digit at that scale, and scopes the
  claim to the smaller models (this would be a central change).
- **B2x CLOSES:** the main claim changes from "hard to read" to "readable from single prompts, but only with tens of
  thousands of training questions"; the paper reports the cost and drops any suggestion that the digit is out of reach.
- **B2x PLATEAUS-BELOW:** the paper says that adding 40,000 more questions does not close the gap, on either test set.
- **B2x SHIFT-LIMITED:** the paper says the added questions did not transfer to GSM8K, reports the Orca test curve, and
  makes no claim that more data fails to help.
- **STILL-CLIMBING or PARTIAL:** the paper gives the curve and states the claim for the sizes tested.
- In every case the paper reports the Orca test set and the twin-free subsets next to the primary numbers, and B1 next
  to the 1.5B results.

## 5. Pre-freeze checks (no probe statistic computed)

- **GPU pilot** (`results_v213/pilot/`): Ouro-2.6B loads in fp32 (12.5 GB), cached vs full 2.9e-5, our loop forward
  matches the model's own forward exactly (max logit difference 0), 0.84 s/prompt on one T4. Qwen2.5-7B in fp16 over
  two T4s (7.8 + 9.7 GB): cached vs full 0.25% relative, no non-finite values, fp16 vs bf16 states differ by 1.4–1.7%
  (bf16 rounding), 0.11 s/prompt.
- **Item pool:**
  - Orca-Math: 169,812 of 200,035 questions usable; 128,251 without a GSM8K near-duplicate.
  - The reworded-twin filter dropped 33,758 training candidates.
  - After it, no Orca training question has content-word Jaccard ≥ 0.30 with any GSM8K or Orca test question
    (checked by a separate exhaustive scan).
  - Remaining twins are inside GSM8K's own pool (kept as in v205/v206): 96 GSM8K and 68 Orca test questions are
    flagged.
  - The hash is stable across two local rebuilds.
- **Smoke test** (`results_v213/smoke/`): every part on both models, on 3 families or 12 prompts. All cached, every
  token check passed, every cached-vs-full check under its limit; arrays finite with the expected shapes.
- **Selftest** (`v213_analyze.py --selftest`): synthetic worlds through B2x must give readable → CLOSES, data-hungry →
  STILL-CLIMBING, swamped → PLATEAUS-BELOW.
  - The first Kaggle run failed on a code bug: the GSM8K-only grid point was missing from the scaled-down grid.
    Fixed; the real grid always contained it.
  - The second run labelled the "data-hungry" world PLATEAUS-BELOW. At 48 dimensions and signal 0.25 the synthetic
    probe stopped improving early, so the world was not data-hungry.
  - The worlds were retuned to 256 dimensions (hungry: signal 0.25, chosen from a scan of plain logistic fits on
    synthetic data), with 1,000 test items to keep chance fluctuation under the 0.02 plateau bar.
  - After a review, PLATEAUS-BELOW was made to require a flat Orca test curve too, and SHIFT-LIMITED was added,
    with a fourth selftest world: the digit unreadable in the GSM8K-like items and readable but data-hungry in the
    Orca-like items (its own code) → SHIFT-LIMITED.
  - Drawing that world's code shifted the random draws, and the data-hungry world fell to PARTIAL: at 256 dimensions
    the gain per doubling sits at the 0.05 bar across seeds (scan: +0.04 to +0.07). The worlds were moved to 1,024
    dimensions, signal 0.20 for the data-hungry parts, where a plain-logistic scan gives +0.06 to +0.09 per doubling
    over three seeds at a level near 0.53.
  - Local run and Kaggle run (`astrocyte-v213-selftest`, `results_v213/selftest/`) of the final file: CLOSES /
    STILL-CLIMBING / PLATEAUS-BELOW / SHIFT-LIMITED, all OK.
- **Item hash on Kaggle** (`astrocyte-v213-check`, `results_v213/check/`): db8f37a5f24b8c64, identical to the two local
  rebuilds.

## 6. Run plan

1. Selftest and the Kaggle item-hash check; freeze (file hashes, ITEMS_SHA); push before any extraction run.
2. `astrocyte-v213-x-qwen7b` (T4 ×2; ~2 h) and `astrocyte-v213-x-ouro26` (T4 ×2: GPU 0 v181, b1, b3 and half the pool;
   GPU 1 the other half; ~7 h).
3. CPU analysis kernels per model: A, B1, B3 together; B2x on its own.
4. RESULT section here; paper update per section 4.

## 7. Amendment 1 (2026-10-21, after the freeze, during extraction, before any probe was fitted on real states)

A reviewer noted that PLATEAUS-BELOW takes the largest gain over the 8 Q × probe combinations on each test set, so
with 1,000 test questions chance fluctuation can turn a truly flat curve into PARTIAL. The effect only makes the
plateau label harder to earn, never easier, so **the labels are unchanged**. Added, descriptive only: the **mean** gain
over the 8 combinations from N_max/2 to N_max on each test set, with a 95% bootstrap interval over test questions
(paired, 10,000 resamples, seed 2132). The paper reports it next to the label.
`v213_analyze.py` fafebeb1ab446d67 → **eaf96e1a782d9fa9** (curve_job also returns per-question correctness; one
reporting block added). The four-world selftest passes on the amended file (local run).

## 8. Execution note 1 (2026-10-22, B2x run split; analysis unchanged)

The single-kernel B2x run for Qwen (`astrocyte-v213-a-qwen7b-B2`, started 2026-10-21 22:53) was still running after
~11 h, close to Kaggle's 12 h CPU limit: the full-state probe at N_max is slow, and `pool.map`'s default chunking puts
the largest jobs on one worker. As a fallback, `v213_split.py` runs the frozen `v213_analyze.b2x` **unchanged**
(`v213_analyze.py` still eaf96e1a782d9fa9) with its process pool swapped out: four part kernels
(`astrocyte-v213-s-<model>-<K>`, K = layer index 0-3) each fit one layer's curve jobs (both probes, every size), that
layer's MLP and, in part 0, the gate, largest first and one job per task, saving each result as it finishes; a combine
kernel (`astrocyte-v213-c-<model>`) feeds the saved results back into b2x, which prints the tables, labels and
amendment-1 intervals. Every fit is deterministic (fixed folds, PCA and MLP seeds), so the output equals a single
run's. Check (local, synthetic world with a different signal strength per layer): the split output equals the single
run's line for line. Used only for a model whose single-kernel run does not finish; no probe result was seen before
this note.
Used for Ouro: the single-kernel run `astrocyte-v213-a-ouro26-B2` (started 09:40) was cancelled at ~10:15, with no
output, and the four `astrocyte-v213-s-ouro26-<K>` part kernels were started instead.
Used for Qwen too: the single-kernel run `astrocyte-v213-a-qwen7b-B2` ended without output at Kaggle's 12 h limit
(status CANCEL_ACKNOWLEDGED). Ouro part 1's first attempt failed before any code ran (empty log; five CPU sessions
were running) and was rerun unchanged. The Ouro parts took 13-30 min each, so the single run's length came from the
pool's chunking, not the fits.
Qwen parts (2026-10-22/23): each part finished every job within ~30 min except the full-state probe at N_max
(45,249), which ran past Kaggle's 12 h limit in all four parts (the same probe at 22,624 took minutes; in Ouro the
N_max job took 10-30 min). The saved results survived the time-out. Rest kernels (`astrocyte-v213-r-qwen7b-<K>`,
`v213_split.py rest K`) load each part's saved results and run only its missing job, with the frozen code, in one
process with 4 BLAS threads (multi-threaded BLAS may change fits in the last digits; selection rule and probe
unchanged), logging each logistic fit's time and iteration count. Combine for Qwen takes the rest kernels' files
(`V213_MODE=c-qwen7b-r`). Local check: dropping a part's N_max job and recovering it with the rest path gives output
identical to a single run.
Rest kernels (2026-10-23): the time-limited part kernels' saved files were not mounted as kernel sources, so all four
rest kernels re-ran their parts in full (4 BLAS threads). Parts 1-3 finished (~1.4-1.9 h each; the N_max full-state
fit then took ~35 min per layer). Part 0 was killed for memory (exit -9, ~68 min) after its N_max job but before the
end, with no output. Its re-run (`astrocyte-v213-r-qwen7b-0` v2) loads the time-limited part 0's 27 saved results
from a private dataset (`astrocyte-v213-qwen7b-part0-saved`) and runs the one missing job; the wrapper now keeps a
single shard memory-mapped instead of copying it. Combine: `astrocyte-v213-c-qwen7b-r`.

## 9. Execution note 2 (2026-10-23, twin-free rows)

`v213_extract.py` does not copy the items' `twin` flag into its metadata, so the analysis kernels' `gsm_clean` /
`orca_clean` rows equal the full test sets. The pre-registered twin-free accuracies were recomputed from the saved
per-question correctness (`_hit`, amendment 1) with the flags rebuilt locally by `v213_items.build()` (96 GSM8K and 68
Orca test questions flagged, as in the extraction metadata's counts); each full-set accuracy recomputed from `_hit`
matches the kernel's. No label uses these rows.

## 10. RESULT (2026-10-22/23; `results_v213/`)

| | Qwen2.5-7B | Ouro-2.6B |
|---|---|---|
| A (v202 rule) | **REPLICATES**: mp mean 0.846, std max 0.272, decoder 0.261, gain +0.007 | **MIXED**: mp mean 0.786 (< 0.80; 0.92 at end1, 0.72-0.78 after), std max 0.167, decoder 0.166, gain +0.008 |
| B1 (v205 rule) | **GENERIC**: r swap 0.90, noise 0.84 | **MIXED**: r swap 0.34 (word swap in quieter directions, standard probe 0.82-0.86), noise 1.28 |
| B3 | present: GSM 0.861, Orca 0.815 | present: GSM 0.787, Orca 0.741 |
| B2x gate | 1.000 | 1.000 |
| B2x GSM8K test, best at 5,249 / 45,249 | 0.430 / **0.519** (l28 full) | 0.217 / **0.308** (end1 full) |
| last-doubling gain, max / mean [95% CI] | +0.026 / +0.008 [-0.001, +0.018] | +0.016 / +0.004 [-0.006, +0.014] |
| Orca test at 45,249, gain max / mean | 0.407, +0.072 / +0.014 [+0.005, +0.023] | 0.272, +0.031 / +0.018 [+0.008, +0.029] |
| twin-free GSM8K / Orca (note 2) | 0.510 (+0.025) / 0.408 (+0.071) | 0.306 (+0.018) / 0.270 (+0.038) |
| MLP at N_max (GSM8K) | 0.12-0.29 | 0.17-0.19 |
| **B2x label** | **PARTIAL** (level > 0.50; gain between 0.02 and 0.05) | **SHIFT-LIMITED** |
| GSM8K-only part, v205 rule | CLIMBS (0.430, +0.067) | PARTIAL (0.217, +0.035) |

Predictions: A REPLICATES 55% (Qwen yes) / 60% (Ouro: MIXED, 25%); B1 GENERIC 55% each (Qwen yes, Ouro MIXED);
B2x Qwen PARTIAL 30%, Ouro SHIFT-LIMITED 10%.

Reading per section 4:
- Neither curve CLOSES: at 45,249 questions the best single-prompt probe is 0.52 (Qwen-7B) and 0.31 (Ouro-2.6B) on
  GSM8K test questions whose minimal pairs read 0.86 and 0.79. Averaged over layers and probes, the last doubling
  adds nothing measurable on GSM8K in either model.
- Qwen-7B's last layer is the exception: its full-state probe reaches 0.43 with the 5,249 GSM8K questions alone
  (Qwen-1.5B: 0.19) and 0.52 with Orca added, still rising slowly; the other layers stay at 0.19-0.29. The paper gives
  the curve and states the claim for the sizes tested (PARTIAL), with the last layer named.
- Ouro: the added questions stopped helping on GSM8K and still help a little on their own distribution (SHIFT-LIMITED):
  no claim that more data fails to help.
- A and B1 for Ouro-2.6B are reported as measured (MIXED), without a general claim.
