# v178 pre-registration (DRAFT, not frozen): does Ouro compute intermediate values inside its loops?

*Drafted 2026-10-06. **Status: FROZEN 2026-10-06** (this commit). No Ouro state for this study existed before the freeze.
Frozen files (sha256, first 16 hex chars, LF line endings):
- `v178_analyze.py` 90bbc8a848dc5544 (selftest OK on 5 worlds);
- `v178_extract.py` 9361c0fb69f5b797;
- `v178_labels.json` 11f0d3852b766479 (84 templates, 4,200 items);
- `loop_grad_probe.py` edc98c31085b0ea7.
**Freeze e99fd90 pushed to origin on 2026-10-10, before any v178 run** (external timestamp). The run is a Kaggle
T4 ×2 kernel (`build_kaggle_v178.py`, 3f7aca9). It runs the frozen files unchanged on two halves of the locked labels
and merges them in label order. Nothing has been measured on Ouro for this study. Freeze = the commit that adds the
analysis script, selftest and locked label-extraction output, before any state is extracted.*

## 1. Question

v176 and v177 showed that Ouro-1.4B solves some multi-step GSM-Symbolic problems with no written steps (2.8% → 9.2% from
loop 1 to loop 4, on numbers it cannot have memorized). The extra loops do something. **Do they compute the
intermediate values of the solution, and at which loop?**

Example (one template; 50 instances with different numbers):
> 1 dozen danishes at $17, 8 dozen tarts at $74, 3 dozen scones at $113. Total? — 17, **592**, 339 → 948.

The first intermediate that isn't copied from the question (here 592) is the probe target.

## 2. Gate 0 (2026-10-06; full texts where marked): PARTIAL, no scoop

- **No paper decodes per-loop intermediate values on GSM8K or GSM-Symbolic in a looped LM**, and none patches across
  instances. Re-check arXiv right before freezing.
- **Closest prior work:**
  - Yu 2411.15862: MLP probe at the last prompt token, synthetic arithmetic; prompted implicit CoT barely computes
    intermediates beyond one step.
  - Kirin 2607.18553: Ouro-RLTT 2.6B, probes predict *success*, not values.
  - Wang & Reid 2609.01924: Ouro's workspace does not cross loop boundaries; it is re-encoded each loop, and writes must
    span all remaining loops.
  - Lu et al. 2507.02199: Huginn, lens-only null.
  - Stolfo 2305.15054: causal mediation on arithmetic, the template for phase 2.
- **Probe design from the literature:**
  - Ridge on log value gives high R² but poor exact recovery (Zhu 2401.03735). MLPs don't help.
  - Periodic or Fourier probes decode number embeddings much better (2506.08966).
  - Hewitt & Liang selectivity controls; split train and test by template.

## 3. The main confound, and why the target is the last digit

If the model only encodes the **question's numbers**, a linear probe can "decode" a product's *size* for free:
log(a·b) = log a + log b. Decoding the magnitude of 592 would say nothing about computation.

**Rejected design: matched distractor.** The first draft used a distractor: the same operation on a different pair of
question numbers. A label-only coverage check (no model) failed its gate, so it was dropped before freezing:
- only 27 templates (1,538 items) had ≥ 40 usable instances, against a gate of ≥ 50 templates;
- many steps use multi-operand expressions or operands that aren't written as numbers ("dozen", "two-thirds");
- distractor magnitudes were badly mismatched (4 vs 1,890), which would bias an R² comparison.

**Adopted design: the target's units digit.** The last digit of a·b, a+b or a−b depends on an *interaction* of the
operands' digits (a modular product or sum). A linear read-out of the operands' representations cannot produce it, so it
is decodable only if the model has computed something that carries it. The design:
- needs no operand parsing, so all 85 templates with a non-copy intermediate stay in;
- matches the periodic structure of number representations (2506.08966).

**Input baseline (how much is linearly available without computation).** The same classifier is trained on features of
the question itself:
- one-hot units and tens digits plus log magnitude of the first 8 question numbers, in order of appearance;
- same folds.

**Selectivity** S_t = held-out accuracy(state at loop t) − held-out accuracy(input baseline).

## 4. Data and labels (locked by a script before freezing; no model involved)

- **Source:** `apple/GSM-Symbolic` main, test. 100 templates × 50 instances.
- **Labels parsed from the worked solutions.** Each step is written `… = $592`; the last equation equals the final
  answer in 4,950/5,000.
- **Target:** the first equation result that is not a question number, not the final answer, an integer, and in
  0 < v < 10⁵. Templates with such a target in all 50 instances: 85/100 (≈ 4,250 items).
- **Primary label:** the target's units digit (10 classes).
- **Copy control label:** the units digit of the largest question number. It is in the input, so it should decode at
  every loop.
- **Final-answer label** (secondary): its units digit.

## 5. States (one forward pass per item; frozen Ouro-1.4B, fp32; no training)

- Same 4-shot answer-only prompt as v176/v177. States are taken at the **last prompt token** (after "Answer:").
  - Pre-loop embedding: h₀.
  - Each loop's mid layer (layer 12) and end state (post-norm), for loops 1–4.
  - So 9 states per item, stored in fp16.
- **Secondary position:** the question's final token ("?").
- The same pass records answer-only correctness (v176b scorer).

## 6. Probes and primary analysis (`v178_analyze.py`, to be written and selftested before freezing)

- **Probe:** standardise → PCA(256, fit on the training fold) → multinomial logistic regression with C ∈ {0.01, 0.1,
  1}, **or the constant majority-class predictor**. The choice is made by inner 3-fold grouped CV. The same procedure is
  used for every state and for the input baseline.
- **Validation:** 5-fold CV **grouped by template** (seeded template shuffle, seed 178), so probes must generalise to
  unseen problem types. Accuracy is pooled over held-out items.
- **Input-baseline features:** for each of the first 8 question numbers, a presence flag, one-hot units digit, one-hot
  tens digit, and log₁₀ magnitude.
- **CI:** paired cluster bootstrap over templates (10,000 resamples, seed 178) on per-item (state correct − baseline
  correct).
- **Majority-class share** (digit 0): 0.308.
- **Apparatus gates.** If either fails, the study is UNSCORABLE:
  - G1, probes work: copy-control accuracy ≥ 0.5 at loop 4's end.
  - G2, no leakage or overfitting: with target labels shuffled within each template, accuracy at loop 4's end ≤
    majority share + 0.02.

**Two splits, both counting toward the verdict** (added before the freeze, after a review note):
- **cross:** folds grouped by template. One decoder must generalise to unseen problem types. Baseline = question
  features.
- **within:** folds by instance index, so every template appears in every fold and only the numbers are new.
  - Baseline = question features **plus template one-hot**.
  - Without it, a state decoder could win just by learning each problem type's usual last digit (some types nearly
    always end in 0), since the state reveals the type while the question numbers don't directly.
  - Inner-CV groups = items.
- **Why both:** if Ouro encodes the value differently for different problem types, the cross split alone would report
  a false null. That null would close the thread.

**H1, a computed intermediate is present.**
- For some loop end t ∈ {1, 2, 3, 4} in **either** split: S_t ≥ 0.05 with CI lower bound > 0.
- Bonferroni over 8 tests (4 loops × 2 splits): 99.375% CIs.
- **Outcomes:**
  - **LATENT-INTERMEDIATE:** H1 holds. Report the first loop and the split.
  - **NO-SELECTIVE-INTERMEDIATE:** every S_t in **both** splits has CI upper bound < 0.05.
  - **UNCLEAR:** otherwise.
- The gates G1 and G2 are computed on the cross split. Shuffled-within-template labels keep each template's digit
  distribution, which the within split may legitimately learn.

**H2, timing (only scored if H1 holds):** the first passing loop is ≤ 2, and S at that loop's mid layer < S at its end,
in the split that passed.

## 6b. Changes made before the freeze (no Ouro state existed at any point)

1. **Matched distractor → units digit** (section 3). A label-only coverage check gave 27 templates against a gate of 50,
   with mismatched magnitudes.
2. **G2 changed** from "|S| at h₀ ≤ 0.03" to the shuffled-label check. h₀ at the last prompt token is the embedding of
   the same token ":" for every item, so it carries no item information and that gate could not fail. The leakage risk
   it was meant to cover is tested better by shuffled labels.
3. **Probe speed-up.** PCA is fitted once per outer fold, not per candidate C. The selftest at full size took > 10 min
   per world before the change.
4. **Constant predictor added to every probe's choices.** In the selftest's operand-only world, the overfit input
   baseline scored 0.283, below the majority share of 0.308. That inflated S by ≈ 0.025 and turned a true null into
   UNCLEAR. With the constant option, the baseline is never below trivial, and that world scores S ≈ 0 →
   NO-SELECTIVE-INTERMEDIATE.
5. **Labels locked:** 84 templates × 50 = 4,200 items. One of the 85 dropped for non-integer final answers.
6. **Within-template split added to the primary** (a review note: a cross-template-only null may be too strict).
   - The within baseline includes template identity.
   - Bonferroni widened from 4 to 8 tests.
   - Selftest world added: the state knows only the template (its usual digit) → must be NO-SELECTIVE. It passes.
   - **Limitation found by the selftest.** A world where the value is present but coded **completely differently for
     each template** scored NO-SELECTIVE in both splits. With 50 items per template, no pooled probe can learn 84
     separate codes, and per-template probes would have ~40 training items for 10 classes. That world is excluded from
     the selftest as undetectable by any method at this sample size.
   - **Interpretation of a null:** NO-SELECTIVE-INTERMEDIATE means "no code for the value that is shared, even
     partly, across problem types and readable by a linear probe at the last prompt token". It does **not** mean "Ouro
     computes no intermediate".
7. **Gate 0 refresh** (2026-10-06, web search): no new paper probes intermediate values in Ouro's loops. Kirin
   2607.18553 (success prediction) is already cited.

## 7. Secondary (reported, not labels)

- The magnitude probe: ridge on log₁₀(value). It is confounded by log-additivity (section 3), so it is reported only as
  a probe-health check.
- A small MLP probe: guards against a null that only says "linear probe too weak".
- Shuffled-within-template labels: R² ≈ 0 sanity.
- The final answer's R² per loop.
- S_t split by correct vs incorrect items (≈ 9% correct, so wide CIs).
- Within-template split, an easier generalisation.
- The "?" position.

## 8. Phase 2: patching (separate pre-registration, written only after phase 1)

- Pairs within a template whose first intermediate differs.
- Copy the last-token state from instance B into A, both at loop t only and at loops t..4 (Wang & Reid: single-loop
  writes are undone).
- **Measure** the shift in log-probability toward the counterfactual answer, i.e. A's answer recomputed with B's
  intermediate, in templates where the final answer is linear in that intermediate. Accuracy is too sparse at 9%.
- **Controls:** a random instance from another template, and patching at h₀.
- Phase 2 runs only if phase 1 is LATENT-INTERMEDIATE or UNCLEAR.

## 9. Predictions (recorded before any data)

- G1 passes; G2 passes (S at h₀ ≈ 0).
- **Primary:** S_t small. LATENT-INTERMEDIATE ≈ 30%, NO-SELECTIVE-INTERMEDIATE ≈ 40%, UNCLEAR ≈ 30%.
  - Units digits are a strict test: answer-only accuracy is only ~9%, so most items may never get the digit right.
  - Yu found prompted models barely compute intermediates.
  - Wang & Reid's re-encoding may hide values from a last-token probe.
- If positive, it appears at loop 2.

## 10. Compute and steps

1. Label-extraction script and lock (no model; local).
2. `v178_analyze.py` with selftest worlds:
   - real computation (the state carries the digit);
   - operand-only (the state carries the operands' digits but not their product's);
   - no signal;
   - broken apparatus (the digit is already in h₀).
3. Re-check arXiv (Gate 0 refresh) → **FREEZE commit**.
4. State extraction on Colab A100: forward only, ≈ 4,250 items, ~15 min, ~160 MB of states. Kaggle quotas are
   exhausted until the weekly reset.
5. Score with the frozen script.

## RESULT (2026-10-10): **UNSCORABLE** (apparatus gate G1 failed)

Run: Kaggle T4 ×2, kernel `aurshalevmerin/astrocyte-v178-ouro`, frozen files unchanged (hashes logged in the kernel
match the frozen hashes in the header). Extraction check vs Ouro's own forward: max |logit diff| 0, max |gate diff| 0 on both GPUs;
tokenisation ok 4,200/4,200. Output: `results_v178/v178_analysis.txt` (states not committed; 2 × 155 MB).

**Frozen output:**
- Majority share 0.308; input-baseline accuracy (cross) 0.308; within baseline 0.361.
- **G1 copy-control accuracy at end4 = 0.408 (need ≥ 0.5) → FAIL.** G2 shuffled = 0.308 (≤ 0.328) → pass.
- **PRIMARY → UNSCORABLE.**
- Secondary: target accuracy 0.308 at every state, both positions (the inner CV chose the constant predictor
  every time); final-answer digit 0.229 at every loop (= its majority); answer-only accuracy per loop
  0.030 / 0.054 / 0.068 / 0.070.

**Predictions vs result:** G1 was predicted to pass; it did not. No H1/H2 prediction is scorable.

### Exploratory diagnostics (after the result; `v178_diag_exploratory.py`; NOT part of the label)

Plain logistic probe (C = 0.1, PCA 256, no constant option), same folds:

| state | copy (cross) | target (cross) | target (within) |
|---|---|---|---|
| last end1 / end2 / end3 / end4 | 0.366 / 0.363 / 0.346 / 0.328 | 0.163 / 0.220 / 0.198 / 0.217 | 0.343 / 0.335 / 0.337 / 0.324 |
| "?" end1 / end2 / end3 / end4 | 0.294 / 0.315 / 0.283 / 0.296 | 0.190 / 0.194 / 0.185 / 0.208 | 0.335 / 0.337 / 0.327 / 0.327 |

Reference points: copy majority **0.408** (so the frozen G1 = 0.408 is exactly the constant guess); target majority
0.308; input features 0.260 (cross) / 0.335 (within); template one-hot alone (within) 0.324; input + template
(within) 0.362.

**Reading.**
- At the last prompt token and at "?", a linear probe cannot read the units digit even of a number **copied from
  the question**. It does worse than guessing the most common digit, at every loop. The instrument is blind at
  these positions, so the flat target result says nothing about whether Ouro computes intermediates.
- Within-template, states score ≈ template identity alone (0.32–0.34 vs 0.324). They carry the problem type, not
  the numbers.
- **Design lesson.** The selftest checked the analysis on synthetic states, but G1's feasibility on the **real**
  model was never piloted. Section 6 allowed control probes before the freeze, and a copy-control pilot on ~500
  items would have shown this before committing. Digit identity is likely readable only at or near the number
  tokens, not at a summary token.

**Status:** the phase-1 question is unanswered. Phase 2 (patching) does not run: its precondition is LATENT or
UNCLEAR. Any follow-up needs a new prereg whose first step is a control-only pilot on real states (where is a
copied digit readable?).
