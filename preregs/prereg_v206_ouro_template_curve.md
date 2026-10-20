# v206 pre-registration: does Ouro's swamped digit stay unreadable with thousands of distinct questions? (v205 B2/B3 on Ouro, Kaggle CPU)

*Drafted 2026-10-16. Status: **FROZEN 2026-10-16** (this commit, pushed to origin before any v206 run). No v206
probe has been fitted; the pilots ran the real model on 8 prompts per part, for timing and the cache check only. Frozen
files (sha256, first 16 hex chars, LF):
- `v206_extract.py` e813cc2b33bd5933. Pilot v3 ran this file except for one default, the B3 question count (60 → 100);
  the code path is the same.
- `v206_analyze.py` af70ba71273adcd0 (the selftest kernel ran this exact file); `build_kaggle_v206.py` 0496b0416c916014.
- `v205_extract.py` 11cc20b80558cd77; `v202_extract.py` 1c68dc20dde2d0d8; `loop_grad_probe.py` 27b80fa0b999bfa7;
  `v205_analyze.py` ee70344632eb5650; `v205_saved.py` 339d264ae7385d0a; `v178_analyze.py` 90bbc8a848dc5544;
  `v181_swamp_check.py` 0a239557c1360a98.*

## 1. Why

v205's B2 showed that in Qwen2.5-1.5B a standard probe trained on up to 5,249 distinct GSM8K questions reads the copied
digit at most at 0.19, while minimal pairs read it at 0.91 (PARTIAL by its rule). The note leads with Ouro-1.4B. Ouro's
swamped measurement rests on 77 distinct questions, so its scaling claim is still Qwen-only. Both Kaggle accounts are out
of GPU quota this week, but Kaggle CPU kernels do not use it, and with a prefix cache Ouro runs at about 8 s per prompt
on CPU (section 5).

## 2. Design: v205's B2 and B3, unchanged except the model

- **Items:** `v205_extract.gsm_items`, unchanged. Every usable GSM8K training question (6,249), one prompt each with the
  last ≥ 2-digit integer's units digit set at random (seed 205). The same fixed 1,000 test questions (permutation seed
  2051) and the same 5,249-question training pool.
- **Model and forward:** Ouro-1.4B in fp32 (`loop_grad_probe`, as in v181). 24 layers × 4 loops, with the loop-end state
  normed and fed back. States: h0, mid1, end1–end4 at slots +0, '?', last. Q = end1, end2, end3, end4.
- **Prefix cache:**
  - The 4-shot head is run once and its key-value states are kept in a minimal per-(loop, layer) cache. Ouro's own
    `UniversalTransformerCache` does not construct under transformers 4.54.1, and Ouro's attention only calls `update`.
  - Only the longest token prefix shared by full prompts is cached, minus 2 tokens of margin, because Ouro's tokenizer
    merges across the head/question boundary: 222 of 225 tokens.
  - Every prompt is checked against that prefix; a prompt that differs runs in full.
  - Cached vs full forward is checked on the first prompt of each kernel; the run aborts if the max |diff| is ≥ 1e-3.
- **B2:** 4 shards (items k with k % 4 = shard). Nested training subsets, both probes (pca: the v178 probe; full: all
  2,048 dimensions with C from 1e-4 to 10, chosen C reported), the effective rank, and the MLP at N_max: all
  `v205_analyze.b2`, unchanged.
- **B3:** minimal pairs on the first 100 test questions (in permutation order) × 10 digits, i.e. 1,000 prompts, with
  `v205_analyze.b3` unchanged. v205 used 150 questions; 100 keeps the CPU run under 2.5 h.
- **Analysis** (`v206_analyze.py`): v205's functions with Q set to Ouro's loop ends. The +0 apparatus gate is read at end1
  (v205 read it at Qwen's l7).

## 3. Labels (v205's, unchanged)

- **Gate:** the pca probe on the digit's own token (+0, end1) at N_max ≥ 0.90, else **UNSCORABLE**.
- level = max over Q and both probes of the accuracy at N_max; gain = max over Q and probes of acc(N_max) − acc(N_max/2).
- **STAYS-SWAMPED:** level ≤ 0.30 and gain ≤ 0.02. **CLIMBS:** level ≥ 0.50 or gain ≥ 0.05. **PARTIAL:** otherwise.
- **B3:** if the minimal-pair mean over Q is < 0.50, B2 is **UNINFORMATIVE**.

## 4. Predictions and what each outcome means for the note

- Predicted: PARTIAL 45%, STAYS-SWAMPED 35%, CLIMBS 15%, UNINFORMATIVE 5%. Ouro's 77-question numbers (standard probe
  0.15–0.23) are a little above Qwen's (0.12–0.14), so it may climb a little higher.
- **STAYS-SWAMPED or PARTIAL with level ≤ 0.30:** the note states the many-question result for both models.
- **CLIMBS:** the note says Ouro's digit becomes readable with thousands of questions, with the curve and the effective
  rank, and scopes "swamped" to tens of templates for Ouro.
- **UNINFORMATIVE:** the note says so; the Qwen result stands alone.

## 5. Pilots (timing and checks only; `results_v206/pilot_v*/`)

- **v1:** stopped. Ouro's `UniversalTransformerCache` cannot be constructed under transformers 4.54.1 (property
  `key_cache` has no setter).
- **v2:** ran, but cached no prompt. Ouro's tokenizer merges across the head/question boundary, so the full head never
  matched. Uncached, it takes about 27–33 s per prompt. Its "check" compared two uncached runs and showed nothing.
- **v3:** 222 of 225 head tokens cached; 8 of 8 prompts cached in each part. Cached vs full: max |logit diff| 1.4e-5 /
  1.5e-5, max |state diff| 2.3e-5 / 1.7e-5. About 7–9 s per prompt after the first, so a 1,562-prompt shard takes about
  3.5 h and B3 about 2.2 h.
- Selftest of `v206_analyze.py` (Kaggle CPU, the frozen file; `results_v206/selftest/`): readable world → CLIMBS,
  swamped world → STAYS-SWAMPED.

## 6. Run plan

1. Selftest (section 5).
2. Freeze: hashes; push to origin before any run below.
3. On Kaggle CPU, main account, in parallel: `astrocyte-v206-b2s0`–`b2s3` and `astrocyte-v206-b3` (5 sessions, the
   account's limit).
4. `astrocyte-v206-bcpu`: the analysis.

## 7. RESULT (2026-10-17; `results_v206/bcpu/v206_analysis.txt`)

Runs: `astrocyte-v206-b2s0`–`b2s3` (1,562–1,563 prompts each, 3.6–4.0 h), `-b3` (1,000 prompts, 2.5 h) and `-bcpu`,
all on Kaggle CPU at the frozen hashes. Every prompt used the cache; the cached-vs-full checks were 1.4e-5 to 2.3e-5.

**B2: CLIMBS** (predicted 15%). Gate: the digit's own token reads at 1.000.

| N | 60 | 250 | 1,000 | 2,000 | 2,624 | 4,000 | 5,249 |
|---|---|---|---|---|---|---|---|
| best pca probe (over end1–end4) | 0.10 | 0.13 | 0.18 | 0.24 | 0.25 | 0.25 | 0.26 |
| best full-state probe | 0.11 | 0.11 | 0.21 | 0.31 | 0.33 | 0.36 | 0.36 |

- level 0.363 (end2, full-state); gain over the last doubling +0.060 (end1 full-state, 0.260 → 0.320), which is above
  the 0.05 bar. So the label is CLIMBS, on the slope rather than the level: below 0.50, but still rising.
- Full-state probe at N_max by state: end1 0.320, end2 0.363, end3 0.306, end4 0.250. MLP at N_max 0.16–0.24.
- Chosen C is mid-grid (0.01–1) at large N.
- Effective rank (participation ratio / components to 95% variance): end1 20 / 552, end2 40 / 652, end3 53 / 713,
  end4 62 / 753. N_max is 7.0–9.5 times the 95% rank.

**B3: present.** Minimal pairs on 100 test questions read the digit at 0.94–0.98 (mean 0.959). A standard probe trained on
these 1,000 prompts (folds by question) reads 0.25–0.35.

**Reading, per section 4 (CLIMBS).**
- In Ouro, "swamped" holds for tens of templates (0.10–0.14 at N ≤ 250). With thousands of distinct questions a
  full-state probe reads the digit at up to 0.36, and it is still rising. That is well above chance and still far
  below the minimal-pair readout (0.96).
- The note must scope Ouro's swamping to small training sets and give the curve.
- The contrast with Qwen (0.19 at N_max, slope at most +0.044) is descriptive. The two models differ in effective rank
  and in how fast the probe climbs. Both are far below their minimal-pair readouts.
