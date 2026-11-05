# v216 pre-registration: without a chain of thought, is the sub-answer formed, and used?

Workstream paper3 (Ouro note). Claimed 2026-10-24. Follows v215 (docs/prereg_v215_kudo_retest.md).

## 1. Why

v215 found that with a chain of thought the sub-answer b is not derived before the chain: the chain computes it. In
their Implicit setting (Appendix A, Table 6), Kudo et al. have Qwen2.5-7B answer directly ("a = 6") at 77.8%, their
probe reads b at 69.1% and the answer a at 40.8% before the output, and they conclude that "when the model fails to
solve a problem, the corresponding (sub-)answers are also unlikely to be represented in its internal states". (Their
text says the accuracies are N/A; the table, the source of the numbers above, gives them; only the threshold positions
are N/A. Their figure caption says Level 2; the table's step counts and the example are Level 3.) Here the model must
form b, or a shortcut, without writing it: our pilot (v216 pilot, 2026-10-24) has Qwen2.5-7B at 0.675 on Level 3
answer-only (majority 0.109), 0.85-0.88 on three of the four operation pairs and 0.33 on minus-minus. This study asks
whether b is present before the answer, beyond copies, the answer and generic mixing, and whether the answer uses it.

**Framing, fixed now.** Every outcome is reported. A positive result would be the paper's computed value that is
present but hard to read, and a correction of an inference (their probe at 69%, read as "unlikely represented"), not of
their chain-of-thought conclusion, which v215 supports.

## 2. Task, items, positions

- v215's 240 families and 6 (k, o2) settings, in their Implicit format: target ` a = A`, demonstrations in the same
  format (`v216_items.py`, built from `v215_items` with the implicit renderer). 12,000 prompts.
- Positions: **eq** (the d2 token), **last** (the final '?'), and **pre**, the token just before the answer digit
  (` a = ` fed in; the analogue of P1 in Studies 1-3). States at pre depend only on the prompt and ` a = `, which is
  the same for every item of a family.
- Qwen2.5-7B fp16 over two T4s, hidden_states 0, 4, ..., 28; `correct` = the answer digit is the argmax at pre.

## 3. Part A: presence (as v215, plus the answer)

- v215's probe, labels, copy ceiling, band {12, 16, 20, 24}, ex_N, I, bootstrap (seed 216), per position.
- **The answer confound.** Before the answer the state should hold A = k o2 B. Within a family k and o2 are fixed, so
  A is a fixed function of B there; the family-demeaned probe, trained across families with different k, separates
  them only partly, and a state that mixes the answer with copies can make the needed form of b readable (v212). So:
  - Secondary labels: excess of A mod 5 (the answer) and of the forms on the answer line, (A + j k) mod 5 for
    j = -2..2 (the needed form is one of them: B = A - k on o2 = +, B = k - A on o2 = -), reported per position.
  - **Answer-mixing null** (v212's design rebuilt on these items): synthetic states with a family offset, copies of
    d1 and d2, a strong answer code (A's value), and one generic nonlinear mixture of (d1, d2, o1, A) (pair products,
    same-frequency Fourier products, a random network), NO b term; strength matched to the observed ex_N at each
    COMBINED position (v212's two-stage calibration), 6 seeds. **BEYOND-ANSWER-MIXING** if the observed I exceeds every
    calibrated world's I, else NOT-SEPARABLE.
- Labels per position: UNSCORABLE (G1 answer accuracy < 0.50 in either version, or G2 copies d1, d2 < 0.50 at the
  position), COMBINED / NOT-COMBINED / UNCLEAR on ex_N, FOLLOWS / DOES-NOT-FOLLOW / UNCLEAR on I (v215 rules).
- **Reading.** "b is formed before the answer" requires COMBINED, FOLLOWS and BEYOND-ANSWER-MIXING at that position.
- **Split by the model's answer (pre-registered, secondary; review note).** Their sentence is about failures ("when
  the model fails to solve a problem, the corresponding (sub-)answers are also unlikely to be represented"). From the
  same family-demeaned probes, ex_N and I are averaged within correctly and wrongly answered items separately (cluster
  bootstrap, seeds 2164-2165), per position. Wrong answers concentrate in minus-minus items (pilot 0.33 correct there),
  so this split is partly an operation split; it is reported with that caveat, and no label is attached to it.

## 4. Part B: is it used by the answer?

- Pairs (x, x'): same family, version and d1, different d2 (so a different b and a different answer), x correct;
  1,000 per position, at positions labelled COMBINED.
- Subspaces at each band layer, fitted on the other half of the families: B (class means of b), the d2 copy
  (class means of d2), the answer (class means of A), and a random 9-dim control.
- Conditions (hooks on the decoder-layer outputs at the position, band layers 11, 15, 19, 23): full (whole state,
  layers 11-27; positive control), cA (answer subspace), cD (copy), cB (B subspace), **cBo** (B orthogonalised
  against both the copy and the answer subspaces; primary), cR (random).
- Outcome: the answer digit at pre (teacher-forced ` a = `); hit = x''s answer.
- Labels per position: UNSCORABLE hit(full) < 0.30; **USED** hit(cBo) - hit(cR) >= 0.10, CI lower > 0; NOT-USED CI
  upper < 0.02; UNCLEAR otherwise (cluster bootstrap over families, 10,000, seed 2163, 97.5% CI). cB, cA, cD
  descriptive. If projecting out the copy and the answer leaves less than half of b's directions at the band layers,
  the cBo test is reported as weak.
- A use claim needs USED where Part A gives COMBINED + FOLLOWS + BEYOND-ANSWER-MIXING.

## 5. Predictions

- COMBINED at every position 85%. FOLLOWS at pre 45%, at last 30%, at eq 10%. BEYOND-ANSWER-MIXING where FOLLOWS 40%.
- Part B at pre: full >= 0.30 85%; USED 20%, NOT-USED 40%, UNCLEAR 40%.

## 6. What each outcome means for the paper (fixed now)

What would count as what (review note):
- **A correction of their inference** ("below the 90% threshold, so unlikely represented"): b formed (the full
  reading) on all items or on correctly answered items, while a standard probe reads it below threshold.
- **A contradiction of their sentence about failures:** the same on wrongly answered items (the split above), which is
  weaker evidence (fewer items, mostly minus-minus).
- **Support for their sentence:** b formed on correct items but not on wrong ones.

- **b formed (COMBINED + FOLLOWS + BEYOND-ANSWER-MIXING) at pre or last:** without a chain, Qwen2.5-7B forms the
  sub-answer before the answer, in a form their probe reads at 69% and reads as "unlikely represented". With v215:
  with a chain the chain computes b; without one, the model computes it internally. If also USED: the answer uses it.
- **FOLLOWS but NOT-SEPARABLE:** a sign-specific signal that answer mixing explains (as Study 3): no claim that b is
  held apart from the answer.
- **COMBINED only, or nothing:** the model reaches the answer without a separately readable b (a shortcut, or b only
  inside the answer's code); their reading stands in substance, and the paper says so.

## 7. Pre-freeze checks

- Items (`python v216_items.py`): 12,000, the same problems as v215's families (checked).
- Part A selftest (`python v216_analyze.py --selftest`, 60 synthetic families): b computed -> COMBINED + FOLLOWS at
  every position (ex_N +0.42, I +0.39-0.40); copies only -> NOT-COMBINED, not FOLLOWS (ex_N -0.14 to -0.15); the
  answer only, no b -> NOT-COMBINED (ex_N -0.11) but I +0.02-0.04: an answer code alone gives a small sign-following
  signal (A depends on the operation), which is why FOLLOWS alone is not the evidence.
- Null selftest (`python v216_null.py --selftest`, lam 1): with the answer code, pair mixing ex_N +0.46, I +0.001;
  Fourier +0.47, I +0.003; the random network ex_N +0.25, I **+0.095**. Answer mixing can produce a large
  sign-following signal with no b: the null is a demanding bar.
- Split: `by_correct` reuses Part A's probe jobs and bootstrap (selftested above); its own synthetic check (b only in
  the "correct" items) was started locally and stopped unfinished after 45 min (slow on Windows), so the split's
  averaging step has no separate synthetic check. It carries no label.
- Smoke on Kaggle (`astrocyte-v216-smoke`, 2026-10-24): extraction of 40 items (answer correct 39/40; positions eq =
  '0', last = '?', pre = ' ', answer token next); patch mechanics with random subspaces in place of every learned one:
  full-state control changes the answer to x''s at eq 20/20 and at pre 19/20, at last 0/20 (so Part B at last will be
  UNSCORABLE by rule); random-subspace clamps change nothing. No minimal-pair statistic or learned-subspace
  intervention on real states before the freeze.

## 8. Frozen files (sha256, first 16 hex, LF; 2026-10-24)

- `v216_items.py` a2ebd48b899fed40
- `v216_extract.py` 536be548e7c532b8
- `v216_analyze.py` 92ff548f3dfc5b91
- `v216_null.py` 517b60e85cc401ba
- `v216_patch.py` 3b96a44a6795a8ad
- `v215_analyze.py` e9d53523b1da41e2
- `v215_items.py` 9249f0523d7219eb
- `v215_pilot.py` d049a0b2d81fb80b
- `v178_analyze.py` 90bbc8a848dc5544
- `build_kaggle_v216.py` b32f934910f1553a
- ITEMS_SHA 497db0bcb9748f30

Run order: `V216_MODE=x` (GPU) -> `a` (CPU) -> `null` (CPU, COMBINED positions) -> `patch` (GPU).

## 9. Execution note 1 (2026-10-24, after the first Part A run; no statistic changed)

The frozen `v216_analyze.py` calls `v215_analyze.part_a` unchanged, and part_a gates on v215's G1 (correct >= 0.90 in
each version; there the model wrote the chain, 0.998). Section 3 above registers G1 for v216 as answer accuracy < 0.50
in either version. The first run (astrocyte-v216-a version 1) therefore labelled every position UNSCORABLE (correct:
plus 0.876, minus 0.638), wrote no Part B plan, and gave the null nothing to calibrate. Its statistics are the
measurement and are kept (`results_v216/a_v1/v216_stdout.txt`).

Correction: `relabel()` in `v216_analyze.py` recomputes the Part A labels from part_a's own statistics with v215's
rules and the registered G1 (0.50); nothing else changes (probe, folds, bootstrap seeds, thresholds, plan, pairs).
Applied by hand to version 1's numbers: eq COMBINED / DOES-NOT-FOLLOW; pre COMBINED / UNCLEAR; last COMBINED /
UNCLEAR. No position FOLLOWS, so the registered "b formed" reading is already excluded. The analysis kernel is rerun
only to write `v216_partA.json` with these labels and the Part B plan; the null and Part B then run as registered,
and are reported as descriptive, since neither can produce a "b formed" or a use claim without FOLLOWS. The rerun's
statistics must reproduce version 1's (same code and seeds); any difference is reported.

New hash: `v216_analyze.py` 1a7cb2f0c1506129 (was 92ff548f3dfc5b91). (Version 2 of the kernel was a duplicate push,
cancelled within minutes.)

## 10. Post hoc, descriptive: their standard probe in this format (added 2026-10-25, after Parts A and B)

Not pre-registered; carries no label and changes none. Written down before it runs: Kudo et al.'s standard probe as in
v215 Part C (`v216_theirprobe.py`): v215's replication set (same problems and held-out b equations) in the Implicit
format, states from v216's extractor at eq / pre / last, standardised states, logistic C = 0.01, train 10,000 / test
2,000, b's and a's values per position and layer; reported: the maximum (their Table 6: b 69.1%, a 40.8%) and, at that
position and layer, test accuracy on correctly and wrongly answered problems. Kernels astrocyte-v216-tpx (GPU),
astrocyte-v216-tp (CPU). Synthetic check: b written at one position and layer is read there at 0.998, chance elsewhere.

Result (2026-10-25): b 0.738 at pre, layer 28 (theirs 69.1%), 0.63 within the prompt (eq, layer 20); a 0.32 within the
prompt (40.8%), 0.850 at pre (the answer about to be written). On wrongly answered test problems b reads as well or
better (pre 28: 0.765 vs 0.726; eq 20: 0.77 vs 0.56; majority class 0.244 vs 0.230); a reads worse (0.583 vs 0.969).
Wrong answers are 410/616 minus-minus problems (right: 208/1,384), so the split confounds operations.

**10b. Within-operation check (rule recorded before running).** The same probe (same fits, same states), per-item test
hits for b saved; for each operation pair (o1 o2) the difference wrong - right, and the stratified difference (per-pair
differences weighted by the wrong count; pairs with fewer than 20 wrong or right items left out), problem bootstrap
(10,000, seed 2166, 95% CI), at pre layer 28 (the maximum) and eq layer 20 (named above). Reading: **the sub-answer is
not less represented on failures** if the stratified difference at pre 28 has CI lower > -0.05 (at most a small
deficit); **b reads better on failures** if CI lower > 0; **less represented on failures** if CI upper < 0. Majority
class per group reported. Kernel astrocyte-v216-tp, version 2.

Result 10b (2026-10-25): pre 28 stratified wrong - right **-0.0389 [-0.0866, +0.0083] -> unclear** (per pair: ++ +0.192
(50 wrong), +- +0.080 (51), -+ -0.027 (105), -- -0.085 (410)); eq 20 -0.0334 [-0.0670, -0.0001] -> less represented on
failures. The raw advantage on failures was the operation mix (minus-minus: most failures, and b reads best there,
0.85). Their sentence about failures is not contradicted; within operations b reads at most slightly worse on failures.

## 11. RESULT (2026-10-25)

- **Part A** (labels with the registered G1, execution note 1): eq COMBINED (+0.379) / DOES-NOT-FOLLOW (I +0.002);
  pre COMBINED (+0.271) / UNCLEAR (+0.018 [0.014, 0.023]); last COMBINED (+0.137) / UNCLEAR (+0.035 [0.030, 0.041],
  D+ -0.045). Answer-mixing null: NOT-SEPARABLE at all three (null I max +0.018, +0.047, +0.057).
- **Part B** (1,000 pairs per position; weak: 35-54% of b's directions left after projecting out copy and answer):
  eq NOT-USED (cBo - cR +0.005 [-0.001, +0.011]; full 0.858, cD 0.829, cA 0.354, cB 0.280); pre UNCLEAR (+0.025
  [+0.014, +0.039]; full 0.847, cA 0.462, cD 0.244, cB 0.349); last UNSCORABLE (full 0.000).
- **Reading (section 6):** COMBINED only. Without a chain, Qwen2.5-7B reaches the answer from the copied operands; b's
  information sits inside the copy and answer codes, not apart from them. Kudo et al.'s reading stands in substance.
- Split by the model's answer (no label): larger I on wrongly answered items (pre +0.052, last +0.080 vs +0.005,
  +0.017), confounded by operations as 10b shows for their probe.
- Their probe (section 10, post hoc): b 0.738 (theirs 69.1%); failures within operations: unclear / slightly lower.
- Predictions: COMBINED everywhere (85%) held; FOLLOWS nowhere (predicted 45% at pre, 30% at last, 10% at eq); Part B
  at pre: full >= 0.30 held, UNCLEAR (40%).
