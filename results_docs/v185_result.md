# v185 word control (2026-10-11): the last-token trace is NOT specific to numbers; it has comparable strength

Code: `v185_extract.py`, `v185_analyze.py` (rules fixed before the data; amendment A1, 514831c, committed before any
output was downloaded or analysed), `build_kaggle_v185.py`, `build_kaggle_v185_cpu.py` (abf0a28).
Output: `results_v185/cpu/v185_analyze.txt`.

**Setup.**
- Same 77 base questions and insertion points as v184, with the sentence "Unrelated fact: a nearby park has 4D
  <colour> benches."
- Families on identical contexts:
  - **col:** vary the colour over 10 single-token colours (D = 3);
  - **dig:** vary D (colour = red).
- 3,080 forwards. Minimal pairs 3080/3080; forward check |diff| 0; apparatus 1.000.

**Result** (last prompt token)

| family | contrastive, mid1 / end1–end4 | standard | within/between spread, end1–end4 |
|---|---|---|---|
| colour, start | 0.996 / 0.997 / 0.996 / 0.996 / 0.995 | 0.19–0.40 | 0.037–0.064 |
| colour, end | 0.995 / 0.992 / 0.996 / 0.992 / 0.996 | 0.22–0.31 | 0.038–0.048 |
| digit, start | 1.000 / 0.997 / 0.996 / 0.996 / 0.992 | 0.17–0.26 | 0.035–0.044 |
| digit, end | 1.000 / 0.997 / 0.987 / 0.965 / 0.975 | 0.14–0.31 | 0.034–0.037 |

- **Frozen verdict:** contrastive at the last token, mean over end1–end4 and both positions: colour 0.995, digit 0.988.
  **NOT NUMBER-SPECIFIC.**
- **A1, secondary:** spread ratio colour/digit = 1.251, **comparable strength** (the colour moves the state slightly
  more).
- The first answer token changed in at most 1.3% of families, for either type.

**Reading.**
- Any single earlier token we varied leaves a small trace at the last prompt token, in a code shared across
  templates. The token can be a relevant number, an irrelevant number or a colour word.
- The trace is about 4–6% of the between-question spread. A minimal-pair probe recovers it almost perfectly, while
  a standard probe misses it.
- So the note's numbers-specific wording ("the numbers are there", "gathered again where the answer begins") is
  withdrawn. The headline becomes methodological: a failed probe at a summary position does not mean the information
  is absent.
- The bare-arithmetic contrast (v180) stands. There the operands dominate the last-token state.
