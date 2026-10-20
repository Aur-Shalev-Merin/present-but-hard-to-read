# v186 pre-registration: does Ouro compute an unwritten intermediate value?

*Drafted 2026-10-11. **Status: FROZEN 2026-10-12** (this commit). No main-set or bare-set state exists. The only model
outputs seen before the freeze are the pilot's accuracy and copied-digit probes on separate families (section 7).
Frozen files (sha256, first 16 hex chars, LF line endings):
- `v186_extract.py` d47ab8b70c1cd7ec;
- `v186_analyze.py` b408d943603262a6 (selftest OK on all required worlds, section 4);
- `v178_analyze.py` 90bbc8a848dc5544 (probe);
- `loop_grad_probe.py` edc98c31085b0ea7;
- `v186_items.py` 4b5367ed784ba4a5;
- `v186_items.json` 73221b1d7bdb1045.*

## 1. Question

The Ouro note (`paper_ouro_note/draft.md`, v181–v185) found that any single varied token leaves a small trace at the
last prompt token, in a code shared across templates. A minimal-pair probe recovers it at 92–100%. So a probe that
reads a value there may just be reading a copied token. **Does Ouro also combine two copied numbers into a value that
is never written, and does that appear only after loop 1?**

## 2. Design (inference only; frozen Ouro-1.4B, fp32)

**Problems** (`v186_items.py` → `v186_items.json`, sha256 73221b1d7bdb1045; no model involved).
- 12 hand-written two-step templates. The answer is a + b − c, and the intermediate m = a + b is never written.
  - Example: "Maya has 30 stickers. Her friend gives her 30 more stickers. She then uses 13 stickers on a card. How many
    stickers does Maya have left?"
- Answers are 21–87, always two digits.
- A **family** fixes the template, the name and the tens digits of a, b and c. Only the units digits vary.
- **main:** 12 templates × 3 families × 200 of the 1,000 unit-digit combinations (seeded) = 7,200 items. The
  association between m's units digit and the answer's units digit is small (Cramér's V = 0.032), so reading m is
  not reading the answer.
- **pilot:** 12 families with other names and tens digits × 60 combinations = 720 items. They are used only before
  the freeze.
- **bare:** 8 families of "What is A plus B?" × all 100 unit-digit pairs = 800 items. They are the positive control
  for reading a computed digit.

**Prompt and states** (`v186_extract.py`).
- The same 4-shot answer-only prompt as v176–v185. One teacher-forced forward pass per item over the prompt plus
  " <answer>".
- Positions: **P0**, the last prompt token, before any answer token; and **P1**, the answer token just before the
  units digit, whose next-token prediction is the units digit.
- States: the middle (layer 12) and end of each of the 4 loops.
- Answer-only correctness by loop, with the v176b scorer.

**Labels** (units digits):

| label | definition | role |
|---|---|---|
| T1 m | (a_u + b_u) mod 10 | the test |
| T2 ans | the answer's units digit | control: reading the answer |
| T3 amb, apc, rnd | (a_u − b_u) mod 10, (a_u + c_u) mod 10, a fixed random function of (a_u, b_u) | unneeded combinations |
| T4 a, b, c | a_u, b_u, c_u | copying (positive control) |

**Copy ceiling.** The same probe on one-hot codes of (a_u, b_u, c_u). This is the best a linear readout of separate
copies can do. Like XOR, (a + b) mod 10 is not linearly readable from separate codes, so the ceiling for T1 should be
near chance.

**Probe** (`v186_analyze.py`).
- Family-mean subtracted states.
- v178 `probe_predict`: standardise, then PCA 256, then multinomial logistic, with C in {const, 0.01, 0.1, 1} chosen
  by inner CV.
- 5-fold CV grouped by template.

## 3. Primary analysis and labels (state = end of loop 4; two positions)

**Positions.** P0 is the last prompt token. P1 is the answer token just before the units digit. Ouro writes the
leading digit first (v180), so m's units digit may only be needed at P1. Both count, with a Bonferroni correction
over the two positions (97.5% CIs).

**Excess over each label's own copy ceiling.** excess(L) = acc(L, state) − acc(L, copy ceiling), per item. This is
needed because the random function has main effects in a and b alone: its copy ceiling was 0.51 in the selftest,
far above chance.

**Statistics per position:**
- D_ceil = excess(T1).
- D_XOR = excess(T1) − max(excess(amb), excess(apc)), against the XOR-like unneeded combinations.
- D_rnd = excess(T1) − excess(rnd).
- D_T3mean = excess(T1) − mean over T3 of excess(T3).
- Paired cluster bootstrap over families: 10,000 resamples, seed 186, 97.5% CIs.

**Rules per position:**
- COMPUTED: D_ceil ≥ 0.15 and D_XOR ≥ 0.15, both CI lower bounds > 0, and the D_rnd CI lower bound > 0.
- NULL: the D_T3mean CI lies entirely within [−0.05, +0.05] (an equivalence test).
- otherwise MIXED.

| outcome | rule |
|---|---|
| **COMPUTED** | either position COMPUTED (report which) |
| **NOT-DETECTED** | both positions NULL |
| **UNCLEAR** | otherwise |

**Gates.** If any gate fails, the label is UNSCORABLE.
- G1: main-set answer-only accuracy after loop 4 ≥ 0.30.
- G2: T4 a, b and c each ≥ 0.75 at P0, in its best state. This shows the pipeline reads copies.
- G3: on the bare set, the sum's units digit at P1, end4 ≥ 0.90. This shows the pipeline reads a computed digit when
  it is there. At P1 that digit is Ouro's next-token prediction, so this is an apparatus check, not evidence.

**Secondary** (reported, not labels):
- T1 at P0 for every state: does it appear only after loop 1?
- Raw accuracies at P1.
- T1 on items Ouro gets right vs wrong after loop 4.
- Bare set: the sum's units digit at P0 vs its copy ceiling.

## 4. Selftest (synthetic states on the real item structure; family offsets dominate as in v181–v185)

Final selftest: Kaggle CPU kernel, version 5. Output in `results_v186_selftest/v186_selftest_v5.txt`.

| world | content | required | result | excess(m) at P0 |
|---|---|---|---|---|
| compute | copies + a code for m on every item | COMPUTED | COMPUTED (P0, P1) | +0.836 |
| compute_on0.4 | m computed on 40% of items | COMPUTED | COMPUTED (P0, P1) | +0.321 |
| compute_on0.15 | m on 15% of items | (recorded) | UNCLEAR | +0.106 |
| copy | additive codes for a, b, c only | NOT-DETECTED | NOT-DETECTED | 0.000 |
| answer | copies + a code for the answer digit | NOT-DETECTED | NOT-DETECTED | +0.017 |
| mixing | copies + a generic nonlinear a×b interaction | NOT-DETECTED or UNCLEAR | UNCLEAR | +0.836 (amb also +0.841) |
| broken | no copies | UNSCORABLE | UNSCORABLE | n/a |

**Sensitivity.** In these idealised worlds the rule detects m when its excess is about 0.15–0.3 or more. That
corresponds to computing m on roughly 20–40% of items. An effect on 15% of items comes out UNCLEAR, not
NOT-DETECTED, which is the honest outcome.

## 5. Predictions (recorded before any main-set data)

**Gates.**
- G1 passes: accuracy after loop 4 is about 0.5–0.6, as in the pilot.
- G2 passes: with 36 families the copies read better than in the 12-family pilot; b is the risk. P(pass) ≈ 0.8.
- G3 passes: P ≈ 0.95.

**Primary.**

| outcome | probability |
|---|---|
| COMPUTED | 35% |
| UNCLEAR | 40% |
| NOT-DETECTED | 15% |
| UNSCORABLE | 10% |

- If COMPUTED, it is more likely at P1 than at P0, because Ouro writes the tens digit first.

**Secondary.**
- T1 at P0 rises after loop 1. Accuracy is 0.02 after loop 1 and 0.33 after loop 2 in the pilot.
- T1 is higher on items Ouro gets right.

**Basis.**
- Ouro solves these far better with more loops, so something is computed.
- The answer's units digit requires (a + b − c) mod 10, which needs m's units digit at some point.
- But Ouro may compute the answer without forming a + b as a separate value, for example as a − c + b. It may also
  hold it in a code no linear probe shared across templates can read.

## 6. What each outcome means for the note

- **COMPUTED, rising after loop 1:** the note's title question gets an answer. Ouro combines an unwritten intermediate
  across loops.
- **NOT-DETECTED with gates passing:** a clean null from a pipeline that detects both copies (G2) and a computed digit
  (G3).
- **UNSCORABLE (G1):** Ouro can't solve these answer-only. The note stays a methods note and reports the gate.

## 7. Pre-freeze pilot (control labels only; `results_v186_pilot/v186_pilot.txt`)

Pilot set: 12 families, 720 items. They are disjoint from the main set: other names and tens digits.

**Answer-only accuracy by loop:** 0.024 / 0.331 / 0.535 / 0.564, so G1 would pass.
- Per template after loop 4, the range is 0.00 (template 3, the train problem) to 0.85.
- Unlike GSM8K, almost nothing is solved after one loop.

**Copied digits, family-demeaned probe at P0** (mid1 → end4):
- a: 0.98 → 0.97;
- b: 0.81 → 0.58, falling across loops;
- c: 0.96 → 0.79.

So the planned G2 (each ≥ 0.90 at end4) could not pass. G2 became "each ≥ 0.75 in its best state".

**Not interpreted.** b's copy fades across loops. This was seen on a control label with 12 families. No test label
was read.

## 8. Changes before the freeze (no main-set state existed at any point)

1. **G2 relaxed** from "each ≥ 0.90 at end4" to "each ≥ 0.75 in its best state", after the pilot (section 7).
2. **Excess over each label's own ceiling.** The first selftest failed three worlds (compute, copy and answer all came
   out UNCLEAR). The random-function control rnd has main effects: its copy ceiling was 0.51, and 0.97 in the compute
   world. Raw-accuracy comparisons were therefore biased.
   - The rules now use each label's excess over its own ceiling.
   - The null uses the mean of the T3 excesses, since the max is biased upward under the null.
3. **P1 made co-primary** with P0, with Bonferroni over 2 positions, because Ouro writes the leading digit first.
4. **NULL became an equivalence test.** The second selftest (Kaggle CPU) still gave UNCLEAR in the copy and answer
   worlds. With 7,200 items the CI around a harmless D_T3mean of about 0.005 excluded 0, so "the CI includes 0" could
   never pass in a true null world. NULL now requires the whole 97.5% CI to lie within ±0.05.
5. **The random-function control only has to be beaten** (D_rnd CI lower bound > 0); the 0.15 gap is required only
   against a − b and a + c. Flagged in review before the freeze.
   - When the state holds a, b and a + b, any function of (a, b) becomes more readable. In the compute world rnd
     reached 0.97, against 0.51 from copying alone.
   - So a 0.15 gap against rnd would turn a real but weaker effect into UNCLEAR.
   - The selftest now also requires COMPUTED in a partial compute world, where m is computed on 40% of items. A world
     with 15% is recorded to show the sensitivity limit.
   - Weaker code amplitudes (0.5, 0.3) were tried first. In the low-noise synthetic states they are still read
     perfectly (excess 0.84), so they do not test weakness. Computing on only a fraction of items does.
6. **Multiprocessing** needs a `__main__` guard on Windows. The selftest and analysis now run as CPU-only Kaggle
   kernels.

## RESULT (2026-10-12): **UNCLEAR**

Run: Kaggle T4 ×2 extraction (`aurshalevmerin/astrocyte-v186-ouro`, frozen files unchanged; tokenisation 8,000/8,000,
logit check 0). Frozen analysis on a CPU-only kernel (`astrocyte-v186-cpu`). Output: `results_v186/cpu/v186_analysis.txt`.

**Gates: all pass.**
- G1: accuracy by loop 0.023 / 0.436 / 0.649 / 0.660.
- G2: copies at their best state are a 1.000, b 0.944, c 0.996. At end4 they are 0.954 / 0.658 / 0.858; b again fades
  across loops.
- G3: the bare sum digit at P1 reads 1.000.

**P0 (last prompt token), end4:**
- m reads 0.148 against a copy ceiling of 0.164, so excess(m) = −0.016.
- D_XOR = −0.075; D_T3mean = +0.041 [+0.027, +0.055].
- Position verdict: **MIXED**. The CI's upper end, 0.055, just misses the ±0.05 equivalence band, so this is
  effectively null but not certified.

**P1 (just before the answer's units digit), end4:**
- excess(m) = **+0.205 [+0.172, +0.236]**.
- Excess of the controls: amb +0.087, apc +0.070, rnd −0.182.
- D_XOR (vs amb) = **+0.118 [+0.094, +0.141]**, below the 0.15 bar; D_rnd = +0.387.
- Position verdict: **MIXED**.

**PRIMARY → UNCLEAR.**

**Secondary:**
- **T1 at P0 by state** falls: 0.226 (mid1) → 0.148 (end4), below the copy ceiling at the end. At the last prompt
  token, m does not build up across loops.
- **Raw accuracies at P1:** m 0.369, ans 0.878, amb 0.246, apc 0.231, rnd 0.325.
- **T1 at P0 by correctness:** 0.159 on items Ouro gets right after loop 4, 0.127 on items it gets wrong.
- **Bare set:** the sum's units digit at P0 reads **0.708 against a copy ceiling of 0.100**. Before writing anything,
  in bare addition, the last digit of a + b is readable far beyond what copies allow. No unneeded-combination control
  was run on the bare set, so the mixing alternative is not excluded there.

**Predictions vs result:**

| prediction | result |
|---|---|
| G1 passes | ✓ (0.66) |
| G2 passes | ✓ |
| G3 passes | ✓ |
| COMPUTED 35% / UNCLEAR 40% / NOT-DETECTED 15% / UNSCORABLE 10% | UNCLEAR |
| if positive, more likely at P1 | ✓ P1 is where m's excess is (+0.21); P0 has none |
| T1 at P0 rises after loop 1 | ✗ it falls |
| T1 higher on correct items | ✓ slightly (0.159 vs 0.127) |

**Reading** (not part of the label).
- At the last prompt token, there is no sign that Ouro has formed the unwritten a + b.
- Just before it writes the answer's last digit, (a + b) mod 10 is readable well beyond copying.
- It is also preferred over unneeded combinations of the same numbers, by +0.12 with the CI excluding 0. That falls
  short of the pre-set 0.15 margin. At that position some generic combination is also present: a − b and a + c read
  +0.07 to +0.09 above their ceilings.
- The bare-addition result shows the pipeline does find a computed digit ahead of writing when the problem is bare
  arithmetic.
- The honest summary is partial evidence of an intermediate at the moment it is needed, not a demonstrated latent
  computation.

## Post-hoc note (2026-10-12; exploratory; does not change the label)

The v186 states were re-scored with mod-5 labels for v188's bar calibration (`v188_calibrate.py`,
`results_v188_calib/v188_calibrate_v186.txt`). Mod 10, every combination shares its parity with a + b or with the
answer digit; mod 5 does not.

- **Bare-addition secondary** ("the sum's units digit at P0 0.708 vs ceiling 0.100"): not specific to the sum.
  Unneeded combinations read as well or better: (a − b) mod 5 has excess +0.545 against +0.434 for (a + b) mod 5, so
  D_XOR = −0.111.
- **P1:** the margin over the answer-and-c combinations, which were not among v186's controls, is +0.048
  [0.032, 0.063] mod 5.

Both hints shrink under the cleaner controls. v188 tests them on new templates.
