# v177 plan: is Ouro's answer-only GSM8K skill reasoning or memorization? Fixed before any v177 data (2026-10-06)

**Why.**
- v176 found that Ouro-1.4B answers 6.5% → 23% of GSM8K problems with no written steps across loops 1–4, which gave a
  GO for an internal-CoT probing study.
- GSM8K is public and widely leaked into training data, so Ouro may recall answers rather than compute them (a review
  note). Recall can also improve with loops, so the loop gain alone doesn't settle it.
- If it is recall, probing the loops for intermediate quantities would find nothing.

**Data: GSM-Symbolic** (Mirzadeh et al. 2024; `apple/GSM-Symbolic`, config `main`, CC BY-NC-ND 4.0; used for evaluation
only, not redistributed).
- 100 GSM8K test problems turned into templates. Their numbers and names are re-sampled and the answers recomputed.
- Variants: the first 5 instances of each template, so 500 items.
- Originals: the 100 source questions.
- Checked locally with Ouro's tokenizer: 500/500 and 100/100 tokenise cleanly, and only the number tokens are scored.

**Measurement.** `v176_answer_only.py` with V176_DATA = symbolic / symbolic_orig.
- Same 4-shot answer-only prompt (GSM8K train 0–3).
- Same fixed scorer (v176b): the number tokens are the argmax, and the next argmax doesn't continue the number.
- Ouro, loops 1–4, fp32.

**Primary** (`v177_analyze.py`; selftest OK on 3 worlds).
- G = accuracy at loop 4 − accuracy at loop 1 on the 500 variants.
- 95% CI by cluster bootstrap over templates (10,000 resamples, seed 177).
- **REASONING-SUPPORTED:** G ≥ 0.0825 (half of v176's +0.165 on GSM8K) and CI lower bound > 0. The loop gain survives
  new numbers, so the GO stands.
- **MEMORY-SUSPECT:** CI upper bound < 0.0825. The loop gain mostly disappears on new numbers, so the v176 GO is
  **withdrawn**.
- **UNCLEAR:** otherwise. Probing would then only go ahead on GSM-Symbolic items, so that memorized answers can't
  drive a probe.

**Secondary** (reported):
- accuracy per loop on variants and on originals;
- the original-minus-variant drop at loop 4, paired by template.
- A drop alone is not proof of memorization: Mirzadeh et al. report drops from number changes even in strong models.
  The primary is the loop gain.

**Same run (v176b, unchanged rule):**
- Huginn GSM8K answer-only rerun with the fixed scorer, read out to 32 loops. Replication band 2.7–9.0% at 32 loops.
- Ouro GSM8K rescored with the same scorer.

**Compute.** Colab A100 notebook `colab_v177_ouro_symbolic.ipynb`; both Kaggle accounts are out of weekly quota.
Roughly 20–30 min. Outputs go to `MyDrive/v177`.

**My predictions** (guesses, not criteria):
- Ouro variants: ≈ 5% at loop 1 and 14–20% at loop 4, so G ≈ 0.10–0.14 → REASONING-SUPPORTED (≈ 60%).
- Originals at loop 4: 22–28%, a drop of 5–10 points.
- Huginn: 3–7% at 32 loops (replicates).

## RESULT (2026-10-06, Colab A100; output pasted by the user; files in MyDrive/v177): UNCLEAR

| set | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---|---|---|
| GSM-Symbolic variants (500, 100 templates) | 2.8% [1.7, 4.6] | 8.6% | 9.0% | 9.2% [7.0, 12.1] |
| their 100 originals | 10.0% [5.5, 17.4] | 14.0% | 16.0% | 18.0% [11.7, 26.7] |

- **PRIMARY:** the loop 4 − loop 1 gain on the variants is **+6.4 points [3.4, 10.0]**. The bar is 8.25, so the label is
  **UNCLEAR**. The gain is clearly above 0 but below half of v176's +16.5.
- **Secondary:** originals minus variants at loop 4, paired by template: **+8.8 points [2.8, 15.4]**. Accuracy halves on
  new numbers.

**Reading.** Post hoc; the label stays UNCLEAR.
- Memorization or surface recall is real: accuracy roughly halves when the numbers change.
- But it sits mostly at **loop 1**: originals 10.0% vs variants 2.8%.
- The **gain from loops is similar in size on both sets**: +8.0 on these 100 originals vs +6.4 on their variants, i.e.
  ~80% of the matched gain.
- The pre-registered bar was taken from a different problem set (the first 200 GSM8K problems, gain +16.5). On these
  100 templates, the originals gain only +8.0.
- So the loops add computation that works on unseen numbers. The higher accuracy on originals is mostly something a
  single loop already retrieves.

**Consequence, as pre-stated for UNCLEAR:** any internal-CoT probing uses **GSM-Symbolic variants only**, so memorized
answers cannot drive a probe. All 50 instances per template (5,000 items) are available. At ~9% that is ≈ 460 correct
answers, and the template structure is fixed while the numbers vary.

## v176b RESULT (same notebook): harness validated

- **Huginn** GSM8K answer-only with the fixed scorer: 0.5 / 1.5 / 0.5 / 3.0 / 4.5 / **5.5%** at 1 / 2 / 4 / 8 / 16 /
  32 loops.
  - Gold log-prob improves from −10.2 to ≈ −4.7 by 8 loops.
  - Lu et al.'s 4.9% band [2.7, 9.0] → **CONSISTENT**. The harness replicates published work.
- **Ouro** rescored with the same scorer: 6.5 / 20.0 / 22.5 / 23.0%, which matches v176 to within one problem. **GO
  stands** on GSM8K.
