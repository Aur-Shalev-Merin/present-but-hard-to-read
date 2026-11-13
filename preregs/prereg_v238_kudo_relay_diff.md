# v238 pre-registration: the relay test without a chain, with pairs whose answers differ

Workstream paper3 (Ouro note, arXiv v2). Claimed 2026-11-09. Follows v216, v234 and v236.
FROZEN 2026-11-09, before any run on the model.

## 1. Why

v236 replaced the whole state at the five positions between b's equation and the answer position (' ;', the
question's letter, '?', the target's letter, ' =') after every layer, in pairs that share the answer, and the answer
moved to the one a carried b predicts in 0.3% and 0.2% of pairs (NOT-HELD). Because those pairs share the answer (and
differ in k), the test sees b, or the differing operand, carried apart from k; it cannot see a finished answer, or b
together with k, carried at those positions (outside review of the paper, 2026-11-09). v216 covers that case at the
last prompt token alone (whole state from layer 11, different answers: 0.00).

v238 repeats v236 with pairs that share k and differ in one operand, so b and the answer differ. Then a carried b', a
carried operand and a carried finished answer all predict the same output, x''s answer A'. If the relay patch does not
move the answer to A', nothing at those positions that depends on b is carried in a form the answer uses.

**Disclosures.**
1. All results of v215-v236 are known, including v236's pilot and relay test (section 1) and the outside reviews of
   2026-11-09 that prompted this study.
2. No pilot was run for v238. The design (positions, rows, layers, gate, thresholds) is v236's, unchanged; only the
   pairs and the scored output differ.
3. No run on the model precedes the freeze, since every row is a whole-state patch and would show real outcomes. The
   mechanics were checked locally on a tiny random Qwen2 (section 6).

## 2. Design (`v238_relay.py`; one GPU kernel)

- **Items:** v234's frozen items (`v234_items.build()`, ITEMS_SHA ac4121c22f88fe31); Qwen2.5-7B base, fp16, Kaggle T4 x2
  (account aurmerin).
- **Pairs:** same family, version and k, both answered correctly (argmax over the ten digits at pre); up to 1,000 per
  type, candidates shuffled with seed 2380:
  - **E1:** same d2, different d1 (operand token d1; the end of b's equation, d2, is the same token in both).
  - **E2:** same d1, different d2 (operand token = the end of b's equation).

  b and the answer differ in every pair. hit = x''s answer A'; kept = x's answer A.
- **Rows** (v236's `ROWS` and `row_spec`, imported unchanged): none; **relay** (the five positions, after every layer
  1-27; primary); **op** (the operand token, every layer; gate); oth (the other operand token, every layer;
  descriptive); op_L and pre_L after single layers 16, 18, 20, 21, 22, 23, 24 (descriptive).

## 3. Rules (per pair type; v236's)

- Statistic: hit(relay) - hit(none), cluster bootstrap over x's family, 10,000 resamples, seed 2383, 97.5% interval.
- **UNSCORABLE** if hit(op) < 0.30; else **HELD** if the statistic is >= 0.10 with the lower bound > 0; **NOT-HELD** if
  the upper bound is < 0.02; **UNCLEAR** otherwise.
- **Verdict:** UNSCORABLE if both types are; HELD if any scorable type is HELD; NOT-HELD if every scorable type is
  NOT-HELD; UNCLEAR otherwise.
- Reported alongside, not part of the verdict: kept rates, oth, the single-layer rows.

## 4. Readings for the paper (fixed now)

- **NOT-HELD:** without a chain, in correctly answered problems, nothing at the five positions between b's equation and
  the answer position that depends on b (b itself, the differing operand, or a finished answer) is carried there in a
  form the answer uses, after any layer. With v236, the paper's no-chain reading of Kudo et al. drops the equal-answer
  caveat: the sub-answer and the answer are not carried through those positions; the end of b's equation and the
  answer position remain as stated (a small effect at the former; undecided at the latter).
- **HELD:** something that depends on b is carried at those positions and used. Since v236 found b not carried apart
  from k, the paper says it is most likely b combined with k, or the finished answer, held before the answer
  position, which goes against Kudo et al.'s claim in this setting; the paper reports it as such.
- **UNCLEAR:** the equal-answer caveat stays, and the paper reports the number.
- **UNSCORABLE:** reported, no reading.

Predictions: NOT-HELD 75%, UNCLEAR 12%, HELD 8%, UNSCORABLE 5%.

## 5. Run plan

Freeze -> push as `v238-freeze` (stacked on `v236-freeze`) -> `python build_kaggle_v238.py` -> push
`aurmerin/astrocyte-v238-relay` (GPU T4) -> download to `results_v238/relay/`. The kernel runs the analysis at the end.

## 6. Pre-freeze checks

- `v238_relay.py --selftest` passes: items sha; 3,000 candidate pairs per type checked to share family, version, k and
  the other operand and to differ in b and the answer (candidates E1 72,000, E2 28,800); row specs; labels.
- Local mechanics (CPU, tiny random Qwen2 with 28 layers and Qwen's tokenizer, `V238_MECH=1`, 3 pairs per type, no
  correctness filter): runs end to end through the analysis; patching every position after every layer reproduces x''s
  logits exactly (max difference 0.0, against 4.2e-3 to 8.4e-3 between x and x').

## 7. Hashes (sha256 of the LF-normalised file, first 16 hex)

- New: `v238_relay.py` 4df49e85f60885cb, `build_kaggle_v238.py` dcfc8ff6ddecc96a.
- Dependencies, unchanged from v236-freeze: `v236_relay.py` 1fed70f2c09a18bd, `v236_pilot.py` 1491005d4bb6f3db,
  `v234_items.py` fa7374fd6378d679 (ITEMS_SHA ac4121c22f88fe31), `v215_pilot.py` d049a0b2d81fb80b,
  `v215_items.py` 9249f0523d7219eb, `v216_items.py` a2ebd48b899fed40, `build_kaggle_v215.py` 49ce949efc2c42a8.

## 8. RESULT (2026-11-10, `aurmerin/astrocyte-v238-relay` v1, frozen build; `results_v238/relay/`)

**Verdict: UNCLEAR** (both types scorable and UNCLEAR: above zero, below the 0.10 bar).

| type | pairs (families) | none | relay | op (gate) | oth | relay - none [97.5% CI] | label |
|---|---|---|---|---|---|---|---|
| E1 (same k and d2, different d1) | 1,000 (120) | 0.000 | 0.081 | 0.998 | 0.069 | +0.081 [+0.061, +0.103] | UNCLEAR |
| E2 (same k and d1, different d2) | 1,000 (120) | 0.000 | 0.083 | 0.999 | 0.000 | +0.083 [+0.062, +0.107] | UNCLEAR |

(hit = the answer becomes x''s answer A'.) With the five positions swapped after every layer the answer stays A in
0.865 (E1) and 0.887 (E2) of pairs, so about 13% and 11% change, 8% of pairs to A'.

Descriptive (single layers; hit A'):

| row | L16 | L18 | L20 | L21 | L22 | L23 | L24 |
|---|---|---|---|---|---|---|---|
| E1 op | 0.815 | 0.811 | 0.755 | 0.747 | 0.672 | 0.030 | 0.000 |
| E1 pre | 0.007 | 0.009 | 0.075 | 0.086 | 0.130 | 0.763 | 0.900 |
| E2 op | 0.976 | 0.972 | 0.635 | 0.612 | 0.402 | 0.006 | 0.000 |
| E2 pre | 0.010 | 0.013 | 0.266 | 0.292 | 0.452 | 0.947 | 0.960 |

- Against v236 (equal answers, same positions and code: 0.003 and 0.002), the relay effect appears only when the pairs'
  answers differ. v236 rules out b (or the differing operand) carried apart from k at these positions, so what carries
  the 8% is most likely b combined with k, or the finished answer, held weakly before the answer position.
- The answer position holds the answer from layer 23 (pre after layer 23: 0.76 and 0.95), the layer after which the
  operand tokens stop mattering.

Reading (section 4, fixed): UNCLEAR. The equal-answer caveat stays, and the paper reports the number.

## 9. Post hoc position map (2026-11-10; descriptive, no decision rule; `astrocyte-v238-posmap`, `results_v238/posmap/`)

Same 2,000 pairs, model and hook (`v238_posmap.py`, pairs exported to `v238_pairs.json`). hit = x''s answer A', with a
97.5% cluster-bootstrap interval over families (seed 2393).

| row (every layer unless noted) | E1 | E2 |
|---|---|---|
| relay (five positions) | 0.081 [0.061, 0.103] | 0.083 [0.062, 0.107] |
| ' ;' alone | 0.008 [0.003, 0.015] | 0.014 [0.006, 0.024] |
| question letter, '?', target letter, each alone | 0.000 | 0.000-0.003 |
| ' =' alone (the answer line's '=', the token before the answer position) | 0.058 [0.041, 0.077] | 0.054 [0.037, 0.074] |
| all but ' =' | 0.011 [0.005, 0.019] | 0.020 [0.010, 0.032] |
| all but any other one | 0.073-0.080 | 0.066-0.084 |
| five positions after layer 8 / 12 / 16 only | 0.001 / 0.003 / 0.006 | 0.000 / 0.003 / 0.008 |
| five positions after layer 20 / 22 / 23 only | 0.030 / 0.024 / 0.044 | 0.041 / 0.033 / 0.042 |

Most of v238's relay effect sits at ' =', the second-to-last token of the answer line " a = A", and appears only from
layer 20; the question's own tokens (' ;', the letter, '?') carry at most 0.014 between them. So the answer-related
carry that v238 found is formed at the answer line, one token before the answer position, late in the network.

## 10. Post hoc timing of ' =' alone (2026-11-11; descriptive, no decision rule; `astrocyte-v238-teqmap`, `results_v238/teqmap/`)

Same pairs, model and hook (`v238_teqmap.py`). The target line's ' =' alone, after single layers (hit = x''s answer A'):

| after layer | 8 | 12 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | every layer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.007 | 0.024 | 0.027 | 0.025 | 0.040 | 0.046 | 0.060 |
| E2 | 0.000 | 0.000 | 0.001 | 0.002 | 0.001 | 0.010 | 0.028 | 0.027 | 0.031 | 0.041 | 0.038 | 0.054 |

The five positions after layer 17 / 18 / 19 alone: 0.006 / 0.008 / 0.016 (E1), 0.008 / 0.008 / 0.026 (E2). ' =' carries
nothing the answer uses through layer 18; it starts at layer 19 and is largest after layers 23-24, when the answer
position finishes reading the operands. The every-layer row reproduces section 9 (0.060 / 0.054 against 0.058 / 0.054).
