# v184 irrelevant-number control (2026-10-11): an unrelated number reaches the answer position just as well

Code: `v184_extract.py`, `v184_analyze.py` (rules and interpretation map fixed before the data), `build_kaggle_v184.py`
(GPU extraction), `build_kaggle_v184_cpu.py` (CPU-only analysis) (e447077). Output: `results_v184/cpu/v184_analyze.txt`.

**Setup.**
- v181's 77 base questions, each with the sentence "Unrelated fact: a nearby park has 4D benches." added either at the
  start or just before the final question sentence.
- Families on identical contexts:
  - **irr:** vary D (the unrelated number's units digit);
  - **rel:** vary a relevant number's units digit, with D = 3.
- 3,080 forwards. Minimal pairs verified 3080/3080; forward check |diff| 0; apparatus (+0 at h0) 1.000 for all types.

**Result** (last prompt token, loops: mid1, end1–end4; chance 0.10)

| family | contrastive probe | standard probe | within/between spread |
|---|---|---|---|
| irr, start | 0.999 / 0.997 / 0.995 / 0.987 / 0.988 | 0.18–0.26 | 0.034–0.045 |
| irr, end | 1.000 / 0.992 / 0.984 / 0.955 / 0.957 | 0.14–0.30 | 0.034–0.060 |
| rel, start | 0.977 / 0.973 / 0.974 / 0.973 / 0.978 | 0.18–0.24 | 0.041–0.055 |
| rel, end | 0.968 / 0.962 / 0.968 / 0.955 / 0.951 | 0.15–0.21 | 0.036–0.062 |

- **Transfer:** a probe trained on relevant families reads the irrelevant ones at 0.78–0.97, on held-out templates.
  The code is shared.
- **Ouro's first answer token** changed across the 10 variants in 0–2.6% of families, for irrelevant and relevant
  digits alike. It starts with the leading digit, which a units digit rarely changes. So the unrelated number does not
  change the answer Ouro starts to give. This check cannot tell anything about the relevant families.

**Reading (case 1 of the map fixed before the data).**
- A copied number's digit reaches the last prompt token **whether or not it matters for the answer**. It arrives at
  the same strength (about 4–6% of the between-question spread) and in the same code as a relevant number's.
- So v181's trace is the copied number itself, not only the answer in progress.
- The alternative raised in review is ruled out as the main source. Answer-related information may still exist alongside
  it, but this test does not show it.
