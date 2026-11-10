# v236 pre-registration: without a chain, is b carried anywhere between its equation and the answer?

Workstream paper3 (Ouro note, arXiv v2). Claimed 2026-11-08. Follows v216, v234 and the v236 pilot.
FROZEN 2026-11-09, before any relay run.

## 1. Why

v234 tested, without a chain, whether b's own code at the end of b's equation (eq) is used to form the answer, with pairs
that share the answer. The result was UNCLEAR (+0.038), and the whole effect at eq went through d2's copy, because in
those pairs the eq token is d2 itself and differs between the two prompts. The v236 pilot asked the question without a
subspace. It used whole-state patches only, at every position from d1 to the answer prefix, one layer at a time and with
all layers together, and two pair types:

- D1, new: same d2, different d1, so the eq token is the same in both prompts.
- D2: v234's pairs.

In both types, the answer moved to the prediction of a used b (P) only when the operand's own token was patched (0.80 to
0.93 through layer 20, 0.00 at layer 24). Patching eq in D1 moved it at most 0.06 of the time. Every single position
from ';' to the answer prefix moved it at most 0.02 of the time. At the answer position (pre) it moved 0.04 (D1) and
0.19 (D2) at layer 20.

A single-position patch can miss a value that is held redundantly at several positions. v236 closes that gap: it swaps
every position between b's equation and the answer at once (the relay), at every layer. If b, or the operands it is
made from, were carried forward at any of those positions in a form the answer uses, the relay patch would move the
answer to P.

**Disclosures.**
1. The pilot's results (`results_v236/pilot/`, `astrocyte-v236-pilot` v1, 300 pairs per type) are known; the numbers
   are above. The rows, layers and gate were chosen from them.
2. All results of v215-v235 are known.
3. No pilot pair is reused: the study takes the pilot's candidate order (seed 2360), skips the candidates the pilot
   tried (D1 445, D2 476), and also skips any later candidate that is one of those pairs in the other order.
4. There is no Kaggle smoke. Every row is a whole-state patch, so a smoke would show real outcomes. The mechanics were
   checked locally instead, on a tiny random Qwen2 (section 6).
5. Designed after v234's result, to press further on the no-chain question.

## 2. Design (`v236_relay.py`; one GPU kernel)

- **Items:** v234's frozen items (`v234_items.build()`, ITEMS_SHA ac4121c22f88fe31): Kudo et al.'s Implicit format
  without a chain ("a=k+b, b=d1+d2 ; a?", target " a = A", three demonstrations). The model is Qwen2.5-7B base in fp16,
  on Kaggle T4 x2 (account aurmerin).
- **Pairs:** pairs share family and version and have equal answers A (k varies). Both prompts must be answered
  correctly, judged by the argmax over the ten digits at pre. Up to 1,000 pairs per type:
  - **D1:** same d2, different d1. The operand token is d1, and the other operand token is eq.
  - **D2:** same d1, different d2. The operand token is eq, and the other operand token is d1.

  P = k o2 b' (x's k with x''s b) is the answer a used b' predicts. P is never A. hit = P; kept = A.
- **Positions:**
  - relay: every token strictly between eq and pre. These are ' ;', the question letter, '?', the target letter and
    ' ='. There are 5 tokens in every item (checked on all 14,400).
  - pre itself is excluded, because it is the answer position.
- **Rows:** one batched pass per pair. At the listed positions and layers, x''s hidden_states l (the output of decoder
  layer l - 1) replace x's. "All" means layers 1-27.
  - none: no patch.
  - **relay:** the relay positions, all layers (primary).
  - **op:** the operand token, all layers (the positive control and the gate).
  - oth: the other operand token, all layers (descriptive).
  - op_L, pre_L: the operand token, or pre, at a single layer L in 16, 18, 20, 21, 22, 23, 24 (descriptive; this
    locates where the answer position reads the operands).

## 3. Rules (per pair type)

- **Statistic:** hit(relay) - hit(none), with a cluster bootstrap over x's family: 10,000 resamples, seed 2363, 97.5%
  interval.
- **Labels:**
  - **UNSCORABLE** if hit(op) < 0.30. The operand patch does not move the answer, so the type has no positive control.
  - **HELD** if the statistic is >= 0.10 and the lower bound is > 0.
  - **NOT-HELD** if the upper bound is < 0.02.
  - **UNCLEAR** otherwise.

  These are the v215, v234 and v235 thresholds.
- **Verdict:**
  - UNSCORABLE if both types are UNSCORABLE.
  - HELD if any scorable type is HELD.
  - NOT-HELD if every scorable type is NOT-HELD.
  - UNCLEAR otherwise.
- Reported alongside the verdict but not part of it: kept rates, oth, and the single-layer op and pre rows.

## 4. Readings for the paper (fixed now)

- **NOT-HELD:** without a chain, in problems the model answers correctly, no position between b's equation and the
  answer carries b or its operands in a form the answer uses. This holds even with all five positions swapped at every
  layer. The answer position reads each operand from that operand's own token. The paper's no-chain verdict on Kudo et
  al. changes from "cannot tell" to "consistent with their claim in this setting: b is not formed before the answer
  position". The paper also says what the result does not settle: whether b is formed at the answer position itself,
  as a step before the answer digit. There the protocol cannot tell b from the answer.
- **HELD:** some position between b's equation and the answer carries operand-dependent information that the answer
  uses. The paper says the answer does not read the operands only from their own tokens, and that telling b from a
  relayed operand copy there needs a clamp study. The no-chain verdict stays "cannot tell", and the paper reports
  where the relay sits.
- **UNCLEAR:** the verdict stays "cannot tell", and the paper reports the number.
- **UNSCORABLE:** reported, with no reading.

If one type is UNSCORABLE, the reading names the type it rests on.

Predictions: NOT-HELD 70%, UNCLEAR 15%, HELD 10%, UNSCORABLE 5%.

## 5. Run plan

The steps are:

1. Freeze.
2. Push as `v236-freeze`, stacked on `v235-freeze`.
3. Build with `V236_MODE=relay python build_kaggle_v236.py`.
4. Push `aurmerin/astrocyte-v236-relay` (GPU T4).
5. Download the output to `results_v236/relay/`.

The kernel runs the analysis at the end.

## 6. Pre-freeze checks

- Pilot (`astrocyte-v236-pilot` v1): see section 1 and `results_v236/pilot/v236_pilot_log.txt`.
- `v236_relay.py --selftest` passes. It checks the labels, the verdict, the bootstrap, the exclusion of pilot pairs
  (both orders) and the row specs: relay = the 5 tokens, all layers 1-27, no layer 28.
- Relay span: on all 14,400 items, range(eq + 1, pre) is exactly (semi, qa, last, ta, teq), 5 tokens.
- Local mechanics (CPU, a tiny random Qwen2 with 28 layers and Qwen's tokenizer, `V236_MECH=1`, 3 pairs per type, no
  correctness filter): the code runs end to end through the analysis. Patching every position at every layer gives
  max|patched - x'| = 0.0 at pre, against max|x - x'| of 1.5e-3 to 9.8e-3, so the hook writes x''s states.
  Candidates after the exclusion: D1 20,238, D2 13,470.

## 7. Hashes (sha256 of the LF-normalised file, first 16 hex)

- New: `v236_relay.py` 1fed70f2c09a18bd, `build_kaggle_v236.py` 2bfdb46ea870a43a, `v236_pilot.py` 1491005d4bb6f3db
  (imported for `candidates` and `positions`; the code the pilot ran).
- Dependencies, unchanged from v234-freeze: `v234_items.py` fa7374fd6378d679 (ITEMS_SHA ac4121c22f88fe31),
  `v215_pilot.py` d049a0b2d81fb80b, `v215_items.py` 9249f0523d7219eb, `v216_items.py` a2ebd48b899fed40,
  `build_kaggle_v215.py` 49ce949efc2c42a8.

## 8. RESULT (2026-11-09, `aurmerin/astrocyte-v236-relay` v1, frozen build; `results_v236/relay/`)

**Verdict: NOT-HELD** (both types scorable and NOT-HELD).

| type | pairs (families) | none | relay | op (gate) | oth | relay - none [97.5% CI] | label |
|---|---|---|---|---|---|---|---|
| D1 (same d2, different d1) | 1,000 (120) | 0.000 | 0.003 | 0.898 | 0.058 | +0.003 [+0.000, +0.007] | NOT-HELD |
| D2 (same d1, different d2) | 1,000 (119) | 0.000 | 0.002 | 0.876 | 0.013 | +0.002 [+0.000, +0.006] | NOT-HELD |

(hit = the answer is P.) With all five positions between b's equation and the answer swapped at every layer, the answer
kept A in 0.994 (D1) and 0.993 (D2) of pairs.

Descriptive (single layers; hit P):

| row | L16 | L18 | L20 | L21 | L22 | L23 | L24 |
|---|---|---|---|---|---|---|---|
| D1 op | 0.803 | 0.791 | 0.786 | 0.784 | 0.695 | 0.018 | 0.001 |
| D1 pre | 0.004 | 0.003 | 0.027 | 0.031 | 0.055 | 0.052 | 0.000 |
| D2 op | 0.868 | 0.844 | 0.613 | 0.601 | 0.396 | 0.005 | 0.000 |
| D2 pre | 0.019 | 0.027 | 0.187 | 0.198 | 0.263 | 0.086 | 0.000 |

- The operand tokens stop mattering between hidden_states 22 and 23: the answer position reads them in decoder layer 22.
- At pre, a partial P signal rises to 0.26 (D2) and 0.06 (D1) at layers 20-22 and is gone by 24. This is at the answer
  position, where the protocol cannot tell b from the answer (section 4).

Reading (section 4, fixed): without a chain, in problems the model answers correctly, no position between b's equation
and the answer carries b or its operands in a form the answer uses. The answer position reads each operand from its own
token (in decoder layer 22). The paper's no-chain verdict on Kudo et al. changes from "cannot tell" to "consistent with
their claim in this setting: b is not formed before the answer position". Not settled: whether b is formed at the
answer position itself before the answer digit.
