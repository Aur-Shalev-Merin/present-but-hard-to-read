# v176 plan: answer-only GSM8K accuracy vs loop count (Huginn-3.5B, Ouro-1.4B). Fixed before any v176 data (2026-10-05)

**Purpose.** Before building any internal-chain-of-thought probes, check whether the looped models solve multi-step
problems *without writing steps*, and whether more loops help. If they don't, there is nothing latent to probe.

**Gate 0** (2026-10-05; full texts of 2507.02199, 2609.01924, 2604.11791, 2510.25741): GO-WITH-REFRAME.
- **Huginn answer-only is already measured.** Lu et al. (2507.02199), 8-shot with CoT suppressed by the system prompt:
  ≈ 3.1% at r = 4, ≈ 4.9% at r = 32, then a plateau. With explicit CoT: 24.9 strict / 38.1 flexible.
  - They also found limited evidence of latent CoT. Logit- and Coda-lens rank trajectories show no temporal separation
    between intermediate and final tokens. They used no trained probes and no patching.
  - The Huginn arm here is therefore a **replication and harness check**.
- **Ouro answer-only: open.** Ouro's reported GSM8K (78.9 at T = 4) is 8-shot CoT.
- **Intermediates at successive loops, by trained probes with controls: partial**, a null on Huginn by lens only.
  Causal patching of loop states on reasoning: open.
- **Horizon caveat.** 2609.01924 (Jacobian lens) finds Huginn's interventions act within ~2 recurrences, much shorter
  than the backward-signal estimate 1/(1 − ρ) ≈ 8. Content stays readable across all 16 loops. Any "ρ bounds the chain"
  claim must reconcile the two.

**Measurement.** `v176_answer_only.py`; frozen models; no training.
- **Data:** the first 200 GSM8K test problems.
- **Prompt:** 4-shot, answer only. The shots are GSM8K train 0–3 with their final answers only, in the format
  `Question: …\nAnswer: <n>\n\n`.
- **Huginn:** one 64-loop run per problem (bf16, as in the September probe), read out after 1/2/4/8/16/32/64 loops. The
  recursion doesn't depend on total R, so the readout after t loops equals an R = t run.
- **Ouro:** fp32, read out after each of loops 1–4.
- **Score:** exact greedy match, teacher-forced. Greedy decoding emits ` <n>\n` iff every gold token is the argmax.
  Also the gold log-probability.
- **Checks:**
  - adapter vs the model's own forward: argmax agreement and max logit diff, logged;
  - tiny local run: agreement 1.0 and diff 0.0 for both models;
  - real tokenizers split prompt | answer cleanly (checked locally on both).
- **Trivial baseline:** the fraction of problems whose answer appears verbatim as a number in the question.

**Go/no-go for an internal-CoT probing study on Ouro** (trained probes for intermediates by loop, plus causal patching
with matched controls).
- **GO** requires both:
  - Ouro's best-loop accuracy ≥ 10%;
  - best loop minus loop 1 ≥ 5 points, with a 95% paired bootstrap CI over problems (10,000 resamples, seed 176) that
    excludes 0.
- Otherwise **NO-GO**: Ouro barely solves problems without writing steps, or loops don't help, so there is little latent
  computation to find.
- **Huginn is reported as a replication:** "consistent" if its 32-loop accuracy lies within the 95% binomial CI of
  4.9% at n = 200 (about 2–9%). It does not count toward GO.

**My predictions** (guesses, not criteria):
1. Huginn: 2–7% at 32 loops, flat from 8 loops on (replicates).
2. Ouro: about 3% after loop 1 and 6–10% after loop 4. So NO-GO is slightly more likely than GO (≈ 60/40).
3. Trivial "answer appears in the question" baseline: 5–10%.

**Compute.** One Kaggle kernel, T4 ×2. Each GPU runs half of the Huginn problems, then half of the Ouro problems.
Estimated 30–60 min.

## RESULT (2026-10-05, Kaggle T4 ×2): Ouro GO; Huginn NOT CONSISTENT with Lu et al. (harness question open)

`results_v176/analysis.txt`, from `v176_analyze.py` (committed a599647 before data).
- **Checks:** both models, both GPUs: adapter = model forward (argmax agreement 1.0, diff 0.0); tokenisation 200/200.
- **Trivial baseline:** the answer appears verbatim in the question for 6.5% of problems.

**Ouro-1.4B, answer only**

| loops | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| accuracy | 6.5% [3.8, 10.8] | 19.5% [14.6, 25.5] | 21.5% [16.4, 27.7] | **23.0%** [17.7, 29.3] |
| mean gold log-prob | −5.40 | −3.91 | −3.64 | −3.56 |

- Best loop (4) minus loop 1: **+16.5 points [10.5, 22.5]** → **GO** by the frozen rule.
- Most of the gain comes in the second loop (+13 points). Loops 3 and 4 add +2 and +1.5.

**Huginn-3.5B, answer only: 0/200 at every loop count** (1 to 64). Gold log-prob improves from −16.6 at loop 1 to
≈ −11.9 at 16+ loops.
- Against Lu et al.'s 4.9% at 32 loops (band 2.7–9.0% at n = 200): **NOT CONSISTENT**.
- Possible cause: prompt format. Lu et al. used Huginn's chat template with a system prompt; v176 used a plain few-shot
  format, and the scorer requires the number immediately after "Answer:".
- **Diagnostic launched** (`v176_huginn_diag.py`, post-result): top-5 at the answer position and exact match, plain vs
  chat-template prompt, 20 problems at 32 loops.
- Until it reports, **the harness is not validated by replication**. The Ouro result rests on the passed equivalence
  checks, the clean tokenisation, and its plausibility. Ouro is reported at 78.9% with written CoT at 4 loops, and
  answer-only accuracy rises sharply with loops.

**Predictions vs result:**

| prediction | result |
|---|---|
| Huginn 2–7% at 32 loops | ✗ 0% |
| Ouro ≈ 3% at loop 1 | ✓ 6.5% |
| Ouro 6–10% at loop 4 | ✗ 23%, far higher |
| NO-GO 60% | ✗ GO |
| trivial baseline 5–10% | ✓ 6.5% |

## Diagnostic and v176b (2026-10-06)

**The diagnostic** (`results_v176/diag/diag_stdout.txt`; 20 problems, 32 loops) showed the Huginn zero is a **scoring
artefact**.
- v176 required the token after the number to be a single `\n`.
- Huginn's tokenizer has a merged `\n\n` token, which follows every answer in the few-shot block, so Huginn predicts
  `\n\n` there. Ouro splits `\n\n` into two `\n` tokens, so it was scored correctly.
- Without the terminator requirement, Huginn gets 1/20 in the plain format. The chat template gets 0/20, so prompt
  format was not the cause.
- At the answer position, Huginn's top-5 are spread over leading digits (each ≈ 0.15–0.20).

**v176b** (same data, prompt and checks; Huginn read out to 32 loops):
- **Fixed scorer:** correct = every number token is the argmax, AND the argmax after the number is not a continuation
  (it does not start with a digit, `.` or `,`).
- **Ouro rescored the same way.** The GO already stands on v176, which scored it correctly. v176b only aligns the
  scoring.
- **Huginn replication rule unchanged:** its 32-loop accuracy must lie within the 95% Wilson band of 4.9% at n = 200.

**v176b run 1 was void** (`results_v176b_run1_unfixed/`).
- The scripted edit that removed `\n` from the scored continuation failed silently. The other edits applied, so run 1
  still scored `' <n>\n'` and gave the same 0/200 for Huginn.
- Symptoms:
  - gold log-probs identical to v176;
  - the "next" token read after `\n` was "Question".
- Fixed by a direct edit, verified locally with both real tokenizers: only the number tokens are scored now. Then run 2.
- **Lesson:** after a scripted edit, verify the changed line itself, not just that the script runs.
