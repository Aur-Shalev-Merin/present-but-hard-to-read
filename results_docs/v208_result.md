# v208 result: the last token is not linearly tuned to token directions, and removing amplified directions does not help a probe

**Prereg:** `docs/prereg_v208_input_sensitivity.md`, frozen b0f3ce2 and amendment 1 f581b9c, both pushed before the runs
they govern. **Runs:** Kaggle CPU, account aurshalevmerin, 2026-10-18:
- extraction: `astrocyte-v208-qwen0/1`, `-ouro0..3`;
- analysis: `-acpu`, `-bqwen`, `-bouro`, at the frozen and amended hashes.

**Outputs:** `results_v208/`.

## Validity
- Every extraction shard passed both checks:
  - cache check, max abs state diff 8.6e-6 to 1.3e-4 (threshold 1e-3);
  - step check, max rel. change s vs s/2 at most 3.4e-3 (gate 0.05).
- The Part A selftest (4/4) and the Part B selftest (2 worlds + 5 rule cases) passed inside the analysis kernels.

## Part A (24 prompts per model, last prompt token, decision states Qwen l7–l28, Ouro end1–end4)

| | Qwen2.5-1.5B | Ouro-1.4B |
|---|---|---|
| R_tok = g²(word-pair dirs) / g²(random dirs) | 1.16–1.28 | 1.36–1.50 |
| **Q1** | **FLAT** (predicted 10%) | **FLAT** (predicted 10%) |
| D = g²(digit dirs) / g²(word-pair dirs) | 2.7–8.5 | 4.4–5.0 |
| **Q2** | **DISTANCE-ONLY** (predicted 25%) | **DISTANCE-ONLY** (predicted 25%) |
| linearity: cos(full change, linear prediction), swap / noise / digit | ≈0.0 / 0.38–0.50 / 0.34–0.53 | −0.12–0.07 / 0.54–0.68 / 0.36–0.41 |
| full change size over linear prediction, swap | 4.5–10.4× | 4.9–15.1× |
| **Linearity** | **LOCAL-ONLY** | **LOCAL-ONLY** |
| full-size squared gain, word swap / noise (descriptive) | 10–25 | 38–373 |
| full-size squared gain, digit swap / word swap (descriptive) | 0.12–0.94 | 0.23–0.36 |

**What it says.**
- **The natural linear explanation fails.** To first order, a change at the digit's token reaches the last token about as
  strongly along random directions as along the directions in which word embeddings differ (R_tok 1.2–1.5, against the
  ≥ 3 that TUNED needed). So "the network is tuned to the token manifold" does not explain the note's word-vs-noise
  finding.
- **The word swap's effect is almost entirely nonlinear.** Its linear prediction is orthogonal to the actual change
  (cos ≈ 0) and 4.5–15 times too small. A real word in the digit's place changes the last token's state 10–370 times more
  (squared, per unit input) than noise of the same size. The equal-sized noise acts roughly linearly and weakly.
  - Read descriptively: what matters is which token is present, not how far the input moved.
  - Noise added to the digit leaves it recognisably a digit; a different real token is processed as that token.
- **Digit directions are not carried less, per unit input, in the linear response.** D is 3–8: small digit-direction
  changes reach the last token more strongly than word-pair-direction changes, the opposite of v207's guess.
  - Q2's label is DISTANCE-ONLY by the rule (D ≥ 0.8), but the effect goes beyond "like words".
  - For finite swaps the order reverses: a digit-to-digit swap moves the last token 0.1–0.9 times as much, per unit
    input, as a digit-to-word swap.
- **The digit's small trace is again a property of finite token changes,** not of linear gain. Changing which digit is
  present matters much less than changing the category of the token.
- By the prereg, the LOCAL-ONLY label makes Q1 and Q2 statements about small changes only.

## Part B: removing the amplified output directions (full probe, 5,249 training questions, 1,000 test questions)

| | Qwen (l7 / l14 / l21 / l28) | Ouro (end1 / end2 / end3 / end4) |
|---|---|---|
| raw accuracy | 0.147 / 0.177 / 0.176 / 0.190 | 0.321 / 0.363 / 0.306 / 0.250 |
| D_J16 (remove amplified dirs) | −0.004 / −0.005 / +0.006 / +0.000 | −0.034 / +0.002 / +0.002 / −0.001 |
| D_PC16 (remove top PCs) | +0.015 / −0.003 / +0.011 / +0.006 | +0.008 / +0.018 / +0.005 / +0.018 |
| D_PC16m (matched count) | +0.009 / −0.008 / +0.008 / +0.001 | +0.009 / +0.012 / +0.003 / +0.015 |
| **label** | **NO-HELP** (predicted 55%) | **NO-HELP** (predicted 50%) |

**Descriptive.**
- **Removing the amplified directions never helps.** Ouro's end1 is slightly hurt (−0.034 [−0.057, −0.012]). Plain top
  PCs help by at most +0.018, which is not useful either. All intervals are within about ±0.06 of zero.
- **There is no stable shared amplified subspace.** The split-half stability of the 16 amplified directions is 0.29–0.59
  (Qwen) and 0.14–0.22 (Ouro), against 0.78–0.92 for the PCs. The directions the network amplifies vary from prompt to
  prompt. A per-prompt version would need a Jacobian per prompt, which is too costly to be a practical probe.
- **The digit lives largely in those directions, yet redundantly.** The amplified directions hold 25–74% of the digit's
  minimal-pair class-mean energy (random 1%). Removing them leaves the minimal-pair readout unchanged (0.94 → 0.95 in
  Ouro, 0.81–0.99 unchanged in Qwen). The digit's code is spread over many directions.

## Predictions scorecard
- Q1: wrong in both models. TUNED was 65–70%; FLAT, the 10% outcome, happened.
- Q2: the 25% outcome, DISTANCE-ONLY, in both; the effect goes further than the label.
- Part B: NO-HELP, the most likely outcome (50–55%).

## What this changes
- **Method idea: closed.** Projecting out the network's amplified directions does not help a single-prompt probe in
  either model. The directions are not even stable across prompts.
- **The note's open question:** narrowed, not answered.
  - The word-vs-noise difference is not a small-change property. It is a nonlinear effect of putting a different real
    token there.
  - The digit's small trace is not low linear gain. If the author wants, one descriptive sentence fits in Sec 3.4:
    "To first order, the last token is about as sensitive to random input directions as to token directions; the word
    swap's effect is nonlinear."
  - The question of which mechanism carries token identity to the last token (attention to the token, its keys and
    values) remains, and needs the attention-transport extraction.
- **Cross-paper link:** the amplified-direction hypothesis (swamping as the forward face of a non-normal loop map) gets
  no support from this test. Paper 1 is unaffected.
