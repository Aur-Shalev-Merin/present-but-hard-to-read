# v194 pre-registration: does the readable sign follow the problem? (operation swap + causal patching)

*Drafted 2026-10-13 (as `prereg_v194_ouro_operation_swap_DRAFT.md`). Status: **FROZEN 2026-10-13** (this commit, pushed to origin before any main-set run). No main-set state exists. Frozen files (sha256, first 16 hex chars, LF):
- `v194_items.py` 43f21df67801425c; `v194_items.json` fceceeb4dcbd4076;
- `v194_extract.py` a0f6bc4d9a8447a4;
- `v194_analyze.py` cbdccb2c90c00599;
- `v194_patch.py` 0033d5b21884dc93;
- `v178_analyze.py` 90bbc8a848dc5544; `v186_analyze.py` b408d943603262a6; `loop_grad_probe.py` 27b80fa0b999bfa7 (the same Ouro code as v188).

The selftest in section 4 ran on `v194_analyze.py` before one pre-freeze change (G2's bar for b, section 5); it is re-run on the frozen file in parallel with the main extraction, before any result is read.*

## 1. Why

Studies 1 and 2 (v186, v188) found the unwritten a + b readable at P1, just before Ouro writes the answer's units
digit. It reads 0.09–0.13 better than its sign-flipped twin a − b (post hoc).
- That comparison only shows a preference for the needed sign if nothing else favours plus. Every problem in both
  studies needed a + b, so a fixed preference for sums at that position would give the same result.
- Bare addition showed the opposite preference (a − b read best), and a − c beat a + c in word problems. So a
  sign-specific bias exists, and its direction depends on context.
- The clean test changes the computation and keeps everything else fixed (Part A).
- Readable is not used: only an intervention shows whether Ouro relies on a + b (Part B).

## 2. Part A (primary): operation swap

**Items** (`v194_items.py` → `v194_items.json`, sha256 fceceeb4dcbd4076; no model involved).
- Twelve new templates, each in two versions that differ only in the first step's phrase:
  - **plus**, e.g. "gave her B more stickers": first step a + b, answer a + b − c;
  - **minus**, e.g. "took B of her stickers": first step a − b, answer a − b − c.
- A plus family and its minus twin share name, tens digits and unit-digit combinations, so every plus item has a
  minus twin with identical numbers.
- Tens digits: a_t in 6–8, b_t and c_t in 1–2, with a_t − b_t − c_t ≥ 3 and a_t + b_t − c_t ≤ 8. So every answer is
  a positive two-digit number: plus 52–98, minus 12–69.
- **main:** 12 templates × 2 versions × 3 families × 200 of the 1,000 unit-digit combinations = 14,400 items.
- **bare:** 16 families of "What is A plus B?" × all 100 unit-digit pairs = 1,600 items, for gate G3.
- **pilot:** 12 × 2 × 1 family × 30 combinations = 720 items, with other names and tens digits. Used before the
  freeze, on control labels only (section 5).
- The two versions' answers have different tens digits (P1 sits on the tens token). This is acceptable because
  probes are trained and scored within each version.

**States** (`v194_extract.py` = `v188_extract.py` with v194 file names, and the item numbers, version and pair id
kept in the meta).
- One teacher-forced forward pass over the 4-shot answer-only prompt plus " <answer>".
- P0 is the last prompt token, P1 the answer token before the units digit; mid and end of each loop.
- Answer-only correctness after each loop is recorded.

**Probe** (`v194_analyze.py`).
- The v178 probe on family-demeaned states, folds grouped by template, separately for each version.
- Labels mod 5 on units digits: AB = (a + b) mod 5, AMB = (a − b) mod 5.
- The copy ceiling is the same probe on one-hot (a_u, b_u, c_u). excess = acc(state) − acc(ceiling), per item.

**Primary statistic** (P1, end4):
- D+ = excess(AB) − excess(AMB) on plus items;
- D− = excess(AMB) − excess(AB) on minus items;
- **I = (mean D+ + mean D−) / 2.**
- A fixed preference for either sign adds the same amount to one D and subtracts it from the other, so it cancels in
  I. It is reported separately as F = (mean D− − mean D+) / 2.
- Paired cluster bootstrap over pair ids (a plus family with its minus twin): 10,000 resamples, seed 194, 97.5% CI
  (Bonferroni over Parts A and B).

**Labels.**
- **FOLLOWS:** I ≥ 0.05, CI lower bound > 0, mean D+ > 0 and mean D− > 0.
- **DOES-NOT-FOLLOW:** I's CI upper bound < 0.02.
- **UNCLEAR:** otherwise.

**Bar rationale and power.**
- Studies 1–2 gave D+ = 0.087 and 0.130. If the sign follows the problem with a similar effect, I ≈ 0.09–0.13
  whatever the fixed preference; 0.05 is about half the smaller value.
- In Study 2 (7,200 items) the family-bootstrap half-width for a contrast of this kind was ≈ 0.025. With 7,200 items
  per version, I's half-width should be ≈ 0.02, so the bar can be resolved.

**Gates.** If any fails, the label is UNSCORABLE.
- G1: answer-only accuracy after loop 4 ≥ 0.30 in each version.
- G2: copies at P0, best state, in each version: a and c ≥ 0.75, b ≥ 0.60 (lowered from 0.75 after the pilot; section 5).
- G3: bare sum digit at P1, end4 ≥ 0.90.

**Secondary** (reported, not labels):
- the same statistic at P1 end3 and at P0 end4;
- per version, the needed form against the other forms on its line answer + k·c, at end4 and end3;
- by-state excess of AB and AMB per version.

## 3. Part B (secondary): causal patching

Run after Part A unless Part A is UNSCORABLE. The procedure is frozen here. Its inputs, the subspaces and the pairs,
are computed from the Part A states by the frozen `v194_analyze.py` (`patch_plan`).

**Plan** (`v194_patch_plan.npz`, `v194_patch_pairs.json`).
- The templates are split into halves {0–5} and {6–11}.
- For each half, 9-dimensional subspaces are fitted on the OTHER half's plus items at P1 end3: the family-centred
  class means of (a + b) mod 10, (a − b) mod 10 and the answer digit, plus a random orthonormal subspace (seed 1941).
- **Pairs (x, x′):** plus items in the same family, with
  - the same c_u;
  - the same parity of a_u + b_u, so a parity-only effect cannot produce a hit;
  - a different (a_u + b_u) mod 10;
  - x answered correctly after loop 4.
  
  1,000 random pairs are drawn per half.

**Intervention** (`v194_patch.py`).
- Run x and x′ to the end of loop 3, with the teacher-forced prompt and answer as in the extraction.
- At P1, replace x's component in subspace U with x′'s: h ← h + U Uᵀ (h_x′ − h_x).
- Run loop 4 and read the predicted next token at P1, the units digit.
- Positions before P1 are unaffected (causal attention), so loop 4 sees x's own context.
- Conditions: ab, amb, rand, ans (positive control), and none.
- **Hit:** the prediction equals x′'s answer units digit, which is (a′ + b′ − c) mod 10, the digit x would have if its
  a + b were x′'s.

**Statistic.** hit(ab) − max(hit(amb), hit(rand)), with a cluster bootstrap over x's family: 10,000 resamples, seed
1942, 97.5% CI.

**Labels.**
- **UNSCORABLE:** hit(ans) < 0.30, meaning the method cannot move the prediction.
- **USED:** difference ≥ 0.10 with CI lower bound > 0.
- **NOT-USED:** CI upper bound < 0.02.
- **UNCLEAR:** otherwise.

**Choices made at freeze** (OPEN in the draft):
- Patch at end3 only, where a + b read best in Studies 1–2.
- Use the mod-10 subspaces, because the prediction is a units digit; parity is controlled by the pair rule.
- Do not freeze other positions separately, since causal attention already keeps everything before P1 as x's own.

## 4. Selftest (synthetic states on the real item structure)

Results: `results_v194_selftest/v194_selftest.txt` (Kaggle CPU, real probe). Every required world gave its required
label:

| world | content at P1 | required | result |
|---|---|---|---|
| follows | code of the needed sign (a + b on plus, a − b on minus) | FOLLOWS | FOLLOWS, I +0.757 |
| partial_0.4 | the same on 40% of items | (recorded) | FOLLOWS, I +0.322 |
| fixed_plus | a + b code on both versions | DOES-NOT-FOLLOW | DOES-NOT-FOLLOW, I +0.008 [+0.002, +0.013] |
| fixed_difference | a − b code on both versions | DOES-NOT-FOLLOW | DOES-NOT-FOLLOW, I +0.003 |
| generic_mixing | tanh(U[a] ∘ V[b]) | DOES-NOT-FOLLOW or UNCLEAR | DOES-NOT-FOLLOW, I +0.002 |
| broken | no copies | UNSCORABLE | UNSCORABLE |

Part B code was checked on 6 pilot pairs with random subspaces and control conditions only (none, rand, ans), so no
a + b patch was run before the freeze: `results_v194_patchsmoke/` (Kaggle T4; 3 pairs): the code ran end to end; unpatched, Ouro predicted its own digit, and random-subspace patches left it unchanged.

## 5. Pilot (control labels only; 720 items, other names and tens digits)

Results: `results_v194_pilot/v194_pilot.txt` (Kaggle T4 ×2). Tokenisation ok 720/720; logit check 0.

| | plus | minus |
|---|---|---|
| answer-only accuracy, loops 1–4 | 0.044 / 0.453 / 0.644 / 0.658 | 0.036 / 0.672 / 0.728 / 0.728 |
| copy a at P0, best state | 0.997 | 0.997 |
| copy b at P0, best state | 0.772 | 0.719 |
| copy c at P0, best state | 0.933 | 0.922 |

- The copy of b fades across loops in both versions, as in Studies 1–2. The copy of c fades in the minus version (0.92 → 0.54).
- **Change before freezing: G2's bar for b goes from 0.75 to 0.60.** a and c stay at 0.75.
  - Reason: the pilot reads b at 0.72–0.77 with a third of the main set's families. In v186 the copy of b rose from 0.81 in its pilot to 0.94 in the main set, but that is not guaranteed.
  - A copy read at 0.60 is still six times chance.
- No target label (a + b, a − b) was computed on the pilot.

## 6. Predictions (made before any main-set state)

- Gates pass: P ≈ 0.85. The main risk is lower accuracy on minus items.
- **Part A:** FOLLOWS 45%, UNCLEAR 30%, DOES-NOT-FOLLOW 15%, UNSCORABLE 10%.
  - If FOLLOWS: F (the fixed preference) near zero or slightly positive, since bare addition favoured differences.
  - End3 shows a larger I than end4, as with a + b in Studies 1–2.
- **Part B:** USED 30%, UNCLEAR 35%, NOT-USED 20%, UNSCORABLE 15%.

## 7. What each outcome means for the note

- **A FOLLOWS:** the a + b signal is the step the problem needs, not a sum preference; the computation claim firms
  up.
- **A DOES-NOT-FOLLOW:** the Studies 1–2 signal was a position-specific preference; the computation claim is
  withdrawn.
- **B USED:** Ouro relies on the intermediate; the computation section becomes a positive claim.
- **B NOT-USED:** readable but not used, like the copied tokens.

## 8. Interpretation note (added 2026-10-14, after the freeze and before any main-set result; no threshold or label changes)

A side note pointed out a limit that sections 2 and 7 understate.
- **Part A cannot rule out the "answer + c" rival.** In both versions the needed first step equals the final answer
  plus c: (a + b − c) + c = a + b and (a − b − c) + c = a − b. A state that holds only the answer it is about to
  write, with c added back, would also make the readable sign follow the problem.
- **What FOLLOWS does show:** it rules out a fixed, position-specific preference for sums, which is what the sign-flip
  comparison in Studies 1–2 could not exclude.
- **What it does not show:** that a + b is formed as a step before the answer. Section 7's "the step the problem
  needs" is to be read as "the combination the problem needs", not as evidence of order.
- **Part B bears on the rival only partly.** Pairs share c, so within a pair a + b and the answer move together. If
  the state holds an answer × c mixture, the a + b subspace can carry answer information, and an a + b patch can then
  move the prediction without a separate intermediate. Comparing it with the answer-subspace patch helps but does not
  separate the two fully.
- **Added descriptive diagnostic (not a label):** the principal angles between the a + b and answer subspaces, and
  between the a + b and a − b subspaces, in each half's patch plan. They are computed from `v194_patch_plan.npz`
  after the CPU analysis and reported with Part B. Large overlap with the answer subspace weakens any USED reading.

## 9. Amendment to Part B (2026-10-14; after the main extraction finished, before any analysis of it; Part A unchanged)

The main-set states exist, but no analysis has been run on them and no Part A or Part B result has been seen. Two side
notes showed that Part B as frozen could not answer the questions it is meant to answer.

**A. Answer-fixed pairs (B2).** As section 8 notes, B1's pairs share c, so a + b and the answer move together. B2
adds pairs (x, x″) in the same family with:
- the same answer;
- a different (a_u + b_u) mod 10 of the same parity (c shifts to compensate);
- x correct after loop 4.

1,000 pairs are drawn per half, from a separate random stream (seed 1943), so B1's pairs and subspaces are byte-for-byte
unchanged (checked against the frozen code).
- Conditions: none, ab, amb, rand.
- **Hit:** the prediction equals (a″ + b″ − c_x) mod 10, the digit x would write if it used x″'s a + b with its own c.
- Under "the state holds the answer, with c added back", x″ carries the same answer, so an ab patch should not
  produce hits. So B2, not B1, is the test of the answer + c rival.

**B. A second patch point: the end of loop 2.**
- In Studies 1–2 the answer digit was already readable at +0.53 after loop 2 and +0.63 after loop 3. So patching at
  the end of loop 3 tests only whether loop 4 recomputes the answer from a + b; a NOT-USED there would not show that
  a + b is never used.
- Each pair is therefore also patched at the end of loop 2, after which loops 3 and 4 run. The subspaces for this
  point are fitted on end2 states (U_*_h_e2), cross-fitted as before. The random subspace is shared.

**Labels.**
- Part B now has four tests: B1 and B2, each at end3 and end2. Each gets the same rule as before (UNSCORABLE if B1's
  positive control at that point < 0.30; USED if the difference is ≥ 0.10 with CI lower bound > 0; NOT-USED if the CI
  upper bound is < 0.02; else UNCLEAR).
- Each uses a 99.375% CI (Bonferroni: Part A 2.5%, and 0.625% for each Part B test). This is stricter than the frozen
  97.5%.
- The overlap diagnostic of section 8 is reported for both points.

**Code** (diffs since the freeze touch only the Part B plan and the patching script): `v194_analyze.py` cbdccb2c90c00599 → **6aa87b4ebe90bd4d** (the `patch_plan` function only: B2 pairs and end2 subspaces); `v194_patch.py` 0033d5b21884dc93 → **f94d3d27dc97ed92**. All other frozen files are unchanged

**Checks.**
- A plan built from fake states gives B1 pairs and subspaces identical to the frozen code's.
- The patching readout ran on synthetic rows.
- The GPU smoke test (pilot pairs, random subspaces, control conditions only: none, rand, ans) covered both pair
  types and both patch points: `results_v194_patchsmoke/v5/` ran 3 B1 and 3 B2 pairs at both points; unpatched, Ouro predicted its own digit, and the B2 targets differed from it as intended.

**Updated Part B predictions:**
- B1 at end3: USED 30%.
- B1 at end2: USED 35%.
- B2 at end3: USED 15%.
- B2 at end2: USED 20%.
- B2 is the harder test, so it is the less likely to pass.

## 10. RESULT, Part A (2026-10-14; `results_v194/cpu/v194_analysis.txt`)

Run: extraction `aurshalevmerin/astrocyte-v194-ouro` (T4 ×2), analysis `astrocyte-v194-cpu` (CPU; `v194_analyze.py`
6aa87b4e, whose Part A code is identical to the frozen cbdccb2c).

**Gates: all pass.**
- G1: accuracy after loop 4, plus 0.674 and minus 0.765. By loop: plus 0.029 / 0.452 / 0.671 / 0.674; minus
  0.036 / 0.703 / 0.767 / 0.765.
- G2: copies at their best state, plus a 1.000 / b 0.979 / c 0.998; minus a 1.000 / b 0.985 / c 0.995. With all 36
  families per version, b reads well above both the lowered bar (0.60) and the original one (0.75).
- G3: bare sum digit at P1, end4: 1.000.

**Primary (P1, end4): FOLLOWS** (predicted 45%).

| | plus (needs a + b) | minus (needs a − b) |
|---|---|---|
| excess of a + b | +0.207 | +0.078 |
| excess of a − b | +0.082 | +0.267 |
| needed − flipped | D+ +0.124 | D− +0.189 |

- I = **+0.157 [+0.141, +0.172]** (bar 0.05).
- The fixed preference for a − b is F = +0.032 [+0.012, +0.053]: small, in the direction bare addition suggested.

**Secondary.**
- P1 end3: I +0.132 [+0.118, +0.146], FOLLOWS (D+ +0.176, D− +0.088; F −0.044).
- P0 end4: I +0.017 [+0.007, +0.028], UNCLEAR. Before any answer is written the needed sign is barely preferred, as
  in Studies 1–2.
- **Line test** (needed form minus the best other form on answer + k·c):
  - plus, end4: +0.036 [+0.009, +0.062];
  - plus, end3: +0.190;
  - minus, end4: +0.045 [+0.020, +0.070];
  - minus, end3: +0.052.
- **By state:** the needed form peaks at end3 in plus (+0.508) and at end2 in minus (+0.422), and alternates mid/end
  as before.

**Reading, per sections 7–8.**
- The combination the problem needs is readable at P1 more than its sign-flipped twin, and the advantage follows the
  operation. So the Studies 1–2 signal was not a fixed preference for sums.
- This does not separate a step formed before the answer from "the answer with c added back", because in both
  versions the needed form equals answer + c. Part B's answer-fixed pairs (B2) test that.

**Patch plan written:**
- B1: 2,267 and 2,585 candidate pairs per half, 1,000 each taken.
- B2: 2,104 and 2,396 candidates per half, 1,000 each taken.

Part B was launched (`astrocyte-v194-patch`, `v194_patch.py` f94d3d27).

## 11. RESULT, Part B (2026-10-14; `results_v194/patch/v194_patch_analysis.txt`)

Run: `aurshalevmerin/astrocyte-v194-patch` (T4 ×2, `v194_patch.py` f94d3d27): 2,000 B1 and 2,000 B2 pairs, 36
families each. The analysis was recomputed locally from the per-pair rows (`gpu0/`, `gpu1/`), and its output is
identical.

**Positive control (B1, answer subspace): scorable at both points.**
- End3: hit 0.902; the prediction leaves x's own digit in 96.5% of pairs.
- End2: hit 0.331, just above the 0.30 bar. After a loop-2 answer patch, loops 3 and 4 restore x's own digit in 63.5%
  of pairs.
- Unpatched, Ouro predicts its own digit on every pair (1.000; pairs require x correct, with teacher forcing).

**All four tests: NOT-USED** (predicted USED 30% / 35% / 15% / 20%).

| test | hit(ab) | hit(amb) | hit(rand) | hit(ab) − max(amb, rand) [99.375% CI] | label |
|---|---|---|---|---|---|
| B1, end3 | 0.017 | 0.032 | 0.001 | −0.015 [−0.027, −0.004] | NOT-USED |
| B1, end2 | 0.007 | 0.006 | 0.002 | +0.001 [−0.005, +0.007] | NOT-USED |
| B2, end3 | 0.003 | 0.002 | 0.000 | +0.001 [−0.003, +0.007] | NOT-USED |
| B2, end2 | 0.003 | 0.003 | 0.001 | +0.001 [−0.002, +0.003] | NOT-USED |

- Under the a + b patch, Ouro keeps its own digit in 95.9–98.8% of pairs.
- In B1 at end3, the a − b patch scores slightly above the a + b patch (0.032 against 0.017). Both are small, and this
  gets no reading.

**Overlap diagnostic** (mean cos² of the principal angles, half 0 / half 1; top cosine in brackets).

| point | a + b vs answer | a + b vs a − b | a + b vs random |
|---|---|---|---|
| end3 | 0.130 / 0.144 [0.62, 0.64] | 0.264 / 0.268 [1.00] | 0.004 / 0.005 |
| end2 | 0.190 / 0.162 [0.76, 0.64] | 0.361 / 0.345 [1.00] | 0.003 / 0.007 |

The a + b and a − b subspaces share one direction exactly. That is presumably parity, which a + b and a − b always
share and which the pair rule holds fixed. The a + b subspace overlaps the answer subspace only partly.

**Reading, per sections 7–9: B NOT-USED.** The a + b value is readable but not used at P1, just before the last digit, like the copied tokens. This does not show that a + b is never used: the end2 tests are weak (see Limits), and other positions were not patched.
- Replacing the 9-dimensional a + b component at P1 with another problem's, at the end of loop 3 or of loop 2, does not
  change the digit Ouro writes. The same swap of the answer component does change it (0.90 at end3).
- This holds both when the answer moves with a + b (B1) and when it is held fixed (B2).
- Together with Part A: the readable combination follows the operation, but Ouro does not read it at P1 to write the
  units digit.

**Limits.**
- Only P1 is patched. An a + b formed and used at earlier positions, and read from there through attention, is not
  tested, because the positions before P1 keep x's own states.
- The subspaces are linear spans of class means; a code outside them, nonlinear or redundant, is not tested.
- The end2 tests are weak. The control moves the prediction in only a third of pairs, because loops 3 and 4 largely
  re-derive P1 from the context, so a NOT-USED there rests on a+b hits of 0.003–0.007 against a control at 0.33.

## 12. Post-hoc descriptive check (2026-10-15)

The note first cited b's best-state copy at P0 (0.979 / 0.985) next to the P0 test, which was read after loop 4. b's
copy at P0 after loop 4 is 0.770 (plus) and 0.808 (minus) (`results_v185/mech/v185_mech_v194.txt`; rule and full table
in `docs/v181_result.md`, "Swamping mechanism check", M3). So Study 3's P0 test has better copies than Study 1's (0.66)
but not strong ones.
