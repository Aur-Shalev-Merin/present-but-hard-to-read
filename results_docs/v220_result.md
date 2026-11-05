# v220 result: does a op b appear before the answer during training?

**Prereg:** `docs/prereg_v220_training_order.md`, frozen b641d4b. Pushed as branch `v220-freeze` (909c24a,
2026-10-25 22:47:27 UTC) before the study kernel was pushed. No amendments.
**Code:** run unchanged. Hashes match section 8: `v220_train.py` 6fbfc35236c2376d, `v220_measure.py`
6394738908134401, `v220_study.py` f6261159c0c73353, `v220_analyze.py` 7386724d21832e4e.
**Run:** Kaggle T4×2, aurmerin/astrocyte-v220-study v1, 2026-10-25, 77 min, exit codes [0, 0].
**Outputs:** `results_v220/study/` (18 JSONs; weight checkpoints not committed).
`python v220_analyze.py results_v220/study` reproduces `results_v220/study/v220_analysis.txt`.

**Validity:**
- Every run measured all its checkpoints: 301 for dig, 201 for mod.
- Final held-out accuracy is ≥ 0.99 in 17 of 18 runs. std dig seed 3 ended at 0.90 (lr 3e-3; the pilot showed the
  same swings).
- The Kaggle download returned only the first page of files. The rest were fetched page by page; all 18 JSONs are
  present.

## Labels (frozen rules)

| format | arch | out (primary) | c | ranking at out | ranking at c |
|---|---|---|---|---|---|
| dig | std | **WITH-OR-AFTER** (2/3; seed 3 NEVER) | NEVER | P1-FIRST | P1-FIRST |
| dig | loop | **NEVER** (3/3) | NEVER | NONE | NONE |
| dig | std2 | **NEVER** (3/3) | NEVER | NONE | NONE |
| mod | std | **WITH-OR-AFTER** (3/3) | WITH-OR-AFTER | BC-FIRST | P1-FIRST |
| mod | loop | **WITH-OR-AFTER** (3/3) | WITH-OR-AFTER | BC-FIRST | P1-FIRST |
| mod | std2 | **WITH-OR-AFTER** (3/3) | WITH-OR-AFTER (2/3) | SPLIT | P1-FIRST |

**No cell is AHEAD.** Architecture comparison: mod NO-DIFFERENCE; dig NA (leads undefined for loop and std2).

**Times at out** (step of the first sustained run; seeds 1, 2, 3):

| cell | t_A (answer probe) | t_I (sign-specific a op b) | lead |
|---|---|---|---|
| dig std | 1600, 1600, 1400 | 7300, 3000, — | −3.56, −0.88, — |
| mod std | 650, 900, 750 | 750, 1150, 1050 | −0.15, −0.28, −0.40 |
| mod loop | 1000, 1000, 1000 | 1250, 1300, 1200 | −0.25, −0.30, −0.20 |
| mod std2 | 1100, 1100, 900 | 3650, 1450, 4450 | −2.32, −0.32, −3.94 |

On the fixed 0.05 bar, the sign-specific signal at out is crossed after the answer in every run where it appears:
100–400 steps after in mod, 1,400–5,700 steps after in dig std. **This is not evidence that the answer forms
first.** The answer signal peaks near 0.97 and crosses 0.05 almost as soon as it starts rising. I peaks near
0.06–0.26 and crosses 0.05 partway up. A fixed bar therefore delays the smaller signal even when both start together
(review note, 2026-10-25). The frozen label, WITH-OR-AFTER, already treats this as not separating the two accounts.

**Scale-free check (post hoc, descriptive, added after the result).** This uses the step of the first sustained run
at 25% and at 50% of each signal's own peak (same smoothing; I with D+ and D− > 0).
- **mod:** the two rise together.

  | cell | position | I reaches 50% of its peak | the answer reaches 50% of its peak |
  |---|---|---|---|
  | loop | out | 1200–1250 | 1300–1400 |
  | std | out | 750–1200 | 850–1200 |
  | std2 | out | 1050–3650 | 1100–1450 |

  At c, the two are within 0–150 steps of each other in either order. Apart from one std2 seed, the gaps are 1–3
  checkpoints, inside the 5-checkpoint smoothing window. So in mod the order is unresolved, not "answer first".
- **dig std** (seeds 1–2, where I is clearly present): I reaches 50% of its peak at 7,300 and 4,600, the answer at
  3,100 and 2,700. Here I lags on this measure too.
- **dig loop and std2:** I peaks at 0.010–0.022, at the level of untrained networks, so its fractions are noise and
  are not interpreted.

## Descriptive (no labels)

**Final-checkpoint values** (mean over seeds; I at out / its peak; answer dA at out):

| cell | I at out | dA at out | I at b |
|---|---|---|---|
| dig std | 0.071 / 0.088 | 0.71 | 0.133 |
| dig loop | 0.018 / 0.021 | 0.67 | 0.016 |
| dig std2 | 0.006 / 0.011 | 0.35 | 0.011 |
| mod std | 0.091 / 0.145 | 0.97 | 0.073 |
| mod loop | 0.080 / 0.102 | 0.97 | 0.031 |
| mod std2 | 0.064 / 0.071 | 0.88 | 0.021 |

- **dig loop and std2** reach about 99% accuracy with no sign-specific a op b at any position: I ≤ 0.034 everywhere,
  at every checkpoint. They solve the task without a readable first partial.
- **dig std at b** (c not yet seen): I reaches 0.13–0.21 in seeds 1–2, but one-sided. a−b is strongly readable on
  minus problems (D− 0.45–0.55), and on plus problems a−b is also more readable than a+b (D+ −0.06 to −0.13). So the
  model forms a−b at b regardless of the operation. That fails the frozen D+ > 0 condition, so t_I(b) is undefined.
- **mod at c:** p1 rises before ac and bc in all three architectures (P1-FIRST in 8 of 9 runs, TIED in one), with
  the answer present at only +0.04–0.08 there. Under the frozen mapping this is partial evidence against
  answer-mixing at c, labelled descriptive. At out, the earliest partial is bc (b−c / b+c), not p1.
- Single-label readouts of the partials fall below their step-0 values at b and c in dig (dp1 down to −0.29). Training
  removes the untrained network's generic digit mixing, which is why these readouts are measured from step 0.

## Reading (prereg section 6)

- **Training order does not separate the accounts in any cell here.** No run shows a sign-specific a op b while the
  answer is absent. In mod the two rise together within the measurement's resolution; the fixed-bar "after" is a
  threshold effect (see the scale-free check). In dig std, a op b lags the answer even on the scale-free measure. The
  ambiguity stands for these models.
- **Three of six cells (dig loop, dig std2, and one dig std seed) never hold a sign-specific first partial.** They
  reach the answer without it, so in these models a readable a op b is not needed for the computation.
- **Architecture:** no looping effect. mod NO-DIFFERENCE; in dig, loop and std2 behave the same (NEVER), and std
  differs from both. That points to size or compute, not looping.
- **Limits** (as frozen): linear probe; one task family; small models; loop vs std differ in learning rate on dig;
  use not tested.
