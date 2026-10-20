# v180 positive control (2026-10-11): in bare arithmetic the same probe reads operands at the last token, ~0.97-1.00

Code: `v180_extract.py`, `v180_analyze.py` (rules and interpretation map fixed before the data), `build_kaggle_v180.py`
(5752473). Output: `results_v180/v180_analysis.txt`.

**Setup.**
- 2,000 bare-arithmetic prompts in the v176–v179 4-shot answer-only format, on frozen Ouro-1.4B:
  - add: "What is a plus b?", a, b in [10, 999];
  - mul: "What is a times b?", a in [2, 9], b in [10, 99].
- The probe is identical to v178/v179 (v178 `probe_predict`), with 5-fold CV over items.
- Forward check against Ouro's own logits: |diff| 0.

**Result.**
- **Ouro solves these with no written steps:** add 0.871 / 0.974 / 0.973 / 0.973 across loops 1–4; mul 0.829 / 1.0 /
  1.0 / 1.0.
- **Copied operand digits at the last prompt token:**
  - add: 0.96–0.99 from the mid layer of loop 1 onward;
  - mul: 1.00;
  - at h0 they sit at majority (~0.12), since h0 is the same token for every item.
  - So **copies READ at the last token**: the probe is not too weak to read digits at a summary position.
- **Computed result digit at the last token, against the input-feature baseline:**
  - mul: 0.80–0.82 vs 0.50, so S ≈ +0.31;
  - add: 0.19–0.23 vs 0.12, so S ≈ +0.07 to +0.11, which is weak.
  - For add, the units digit is the *last* digit Ouro writes. The first answer token is the leading digit, so its units
    digit need not exist yet at the last prompt token. Note: v178's target was also a units digit.

**Reading.**
- This is case 1 of the map fixed before the data. In bare arithmetic, Ouro gathers the operands at the answer
  position: the digit is readable 4 tokens after its number and at every loop. In word problems (v179), the same probe
  cannot read a copied digit even 3 tokens after its number.
- v179's blindness is therefore a property of word problems in this model, not a weakness of the probe. The note's
  claim 3 stands, as a contrast.
- **Caveat for any future intermediate-value probe:** target the leading digit or the value itself, not the units
  digit, because Ouro writes the units digit last.

**Correction (2026-10-11, after v185).** "Ouro gathers the operands at the answer position" over-reads this check.
- Within each operation the prompts differ only in the two numbers, so the check works like a built-in minimal pair.
- v181–v185 show that any varied token is easy to read once nothing else varies.
- What v180 shows is that the probe can read digits at the last token. It does not show that operands are gathered
  there more strongly than in word problems; trace sizes were not compared across the two settings.
