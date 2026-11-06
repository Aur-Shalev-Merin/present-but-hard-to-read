# Post hoc quick checks after the fourth outside reviews (2026-11-06)

Three descriptive checks on saved data, asked for by the fourth round of outside reviews of the Ouro note. None is
pre-registered and none gives a label; the rules and readings below are recorded before any of them is run. Code:
`posthoc_quickchecks.py`; output `results_posthoc/quickchecks_<check>.json` and the printed lines (saved as `results_posthoc/quickchecks_log.txt`).

## C1. Does a computed value show the copied digit's gap between a standard probe and minimal pairs?

The paper shows the gap only for copied digits and says it did not test computed values. The one setting with a
computed value that is known to be present and used is the v230 small models (s = x op z at ';', DAS ground truth).

- Items: new minimal-pair families built for this check, k = 6: a family fixes x, w and the operation and varies z
  over all 23 values, so s varies over all 23 classes and the prompts differ in one token. 200 families (seed 2330),
  4,600 items.
- Models: the five v230 pos models (frozen checkpoints, `results_v230/protocol/gpu0`). States at ';' and at the middle
  filler, layer 4 (where DAS put s; fixed here, not chosen on the result).
- Probes: v178's probe (StandardScaler, PCA up to 256, logistic regression with C chosen by inner grouped CV, or the
  majority class), 5-fold CV grouped by family. **Standard** = raw states; **minimal pairs** = states minus their
  family mean. Labels s (chance 1/23) and, for comparison, the copied operand z.
- Reading: if the standard probe reads s at ';' at least 0.8 times as well as minimal pairs in every model, the gap
  does not appear for a computed value in these context-free prompts, which is consistent with the paper's account
  that the gap comes from context variance swamping a small change; the paper then says so and scopes the presence
  claim to prompts with varied context. If the standard probe reads s much worse (below 0.5 times) in most models,
  the gap holds for a computed value as well. Anything in between is reported as mixed.
- Limitation recorded now: these prompts have no varied context, so C1 cannot show that a computed value in a
  language model has the gap; it only tests whether the gap is specific to copies.

## C2. How much of b's directions would a random projection remove?

The Kudo clamps keep 35-64% of b's directions after projecting off the copy (and the answer). Baseline: project the
same b basis off a random orthonormal subspace of the same dimension (20 draws, seed 2331) and report the share
left. Sources: the saved plans of v215 (`results_v221/a/v215_plan.npz`, chain format: U and D at each position,
half and layer) and v216 (`results_v216/a_v3/v216_plan.npz`: U, D and An). Share left = ||U - QQ^T U||^2 / dim(U) with
Q an orthonormal basis of the projected-off span (the v216 definition). Also the v233 small models analytically
(a random k-dimensional subspace of R^192 leaves 1 - k/192 in expectation).

- Reading: if a random projection leaves nearly all of b (above 0.95) while the copy projection leaves 35-64%, the
  loss is due to b's class means lying in the copy's and answer's directions (b is a function of the operands), not
  to the projection's size; the paper says this in one clause.

## C3. Does the minimal-pair readout need all ten variants of the test question?

The minimal-pair probe subtracts the mean of the test question's ten variants, which uses information about the test
question. Here: v181 (Ouro-1.4B, copied digit, 137 families of ten), the last prompt token at end1-end4. The probe is
trained as before on family-demeaned training folds (5-fold CV grouped by template, v178 folds); each test item is
instead contrasted with the mean of k other variants of its question drawn at random (seed 2332; k = 1, 2, 4, 9), and
with nothing (k = 0, the standard probe's input; the probe itself is still the minimal-pair one). k = 9 must reproduce
the v181 contrastive accuracy (0.92-0.97) up to a scale change, as a known-answer check.

- Reading: v181's READS bar is 0.30. If k = 1 reads at or above 0.30 at every loop end, one sibling prompt is enough
  and the presence claim does not depend on the full family; the paper says so. If k = 1 falls below 0.30 but some
  k <= 4 reaches it, the paper states the smallest k that does. Otherwise the paper says the readout needs most of the
  family.

## RESULTS (2026-11-06; `results_posthoc/quickchecks_log.txt`, `quickchecks_c1.json`, `_c2.json`, `_c3.json`)

**C1.** No gap for the computed value in these prompts. At ';' (layer 4) the standard probe reads s at 0.993-1.000 in
all five models, the same as minimal pairs (0.993-1.000); at the middle filler 0.859-0.991 against 0.941-0.992 (the
lowest, 0.86 vs 0.94, in seed 2). The copied operand z reads 0.976-0.996 standard. By the recorded reading the gap does
not appear for a computed value in context-free prompts (standard >= 0.8 x minimal pairs in every model), consistent
with the account that the gap comes from context variance; it says nothing about computed values in language models.

**C2.** A random projection of the same dimension leaves 0.996-0.999 of b (v216: 0.9962-0.9966, 13 dims off; chain
plan: 0.9988-0.9990, 4 dims off); the copy (and answer) projections leave far less, so the loss reflects b's class
means lying in the copy's and answer's directions. v233's small models: a random 22 or 44 dims would leave 0.885 or
0.771 in expectation, against 0.55-0.67 and 0.26-0.44 observed.

**Definition mismatch found by C2.** The "share of b left" for the chain study (v215, 0.43-0.64) was computed as the
mean of 1 - cos^2 over the principal angles between b's 9-dim basis and the copy's 4-dim basis (`v215_analyze.py`
line 156), while v216 and v233 use ||U - QQ^T U||^2 / dim(U). With a 4-dim copy the v215 figure overstates the loss:
on the v216 / v233 definition v215's own values (mean cos^2 0.335-0.566) give 1 - (4/9) mean cos^2 = 0.75-0.85 at the
end of b's equation and the last prompt token (the chain re-run in v221's plan gives 0.75-0.84 at the same position
directly). The paper's "35-64%" therefore combined two definitions; on one definition the chain clamps keep 0.75-0.85
and the no-chain clamps 0.35-0.54. v233's primary arm was chosen against the mixed range (disclosure in its prereg);
its pre-registered reading stands as recorded, and the correction is noted there as post hoc.

**C3.** One sibling is enough. Trained as before, the minimal-pair probe reads the copied digit at the last prompt
token, contrasting each test item with k random siblings, at k = 1: 0.785-0.840; k = 2: 0.866-0.909; k = 4: 0.904-0.939;
k = 9: 0.923-0.967 (= v181's 0.923-0.968, the known-answer check); k = 0 (raw state): 0.109-0.146. By the recorded
reading, k = 1 reads above 0.30 at every loop end, so the presence result does not depend on the full family.
