# v220 pre-registration: does a op b appear before the answer during training?

Workstream misc (Ouro-note follow-up in small models). Claimed 2026-10-24. Design and pilots: `docs/v220_design.md`.

## 1. Why

In the Ouro note, a value computed from two operands (a+b in a+b−c) was readable before the answer and followed the
operation, but at any one moment that signal cannot be told apart from "the answer was formed, and c was mixed back
in": both give the same probe labels. The selftest below shows the problem directly. A synthetic state holding only
the answer plus a nonlinear mix of the answer with c reads a op b at +0.66, sign-specific I +0.66.

Training order can break the tie in one direction. Under answer-then-mix, the a op b signal is made from the answer,
so it cannot exist at a position before the answer can be read there or from the model's output. So if, during
training, a sign-specific a op b signal is held at a position while the answer is absent at that position and from
the model's output, that position's signal is not answer-then-mix. If a op b only ever appears together with the
answer or after it, the order does not separate the two accounts, and we say so. This asymmetry is fixed now.

Small models trained from scratch allow dense checkpoints, which a pretrained model does not. The result is about
these models, not about Ouro.

## 2. Models, training, checkpoints (fixed from the pilots)

- **Tasks** (`v220_train.py`, answer-only, loss on the answer tokens only):
  - **dig:** "a1 a2 op b1 b2 − c1 c2 =" followed by two answer digits; a, b, c in 10..99, answer in 10..99, fresh
    problems every step.
  - **mod:** "a op b − c =" mod 23, one token per number, one answer token; 20% of the (op, a, b, c) combinations held
    out of training.
  - op is + or −, so each format has a+b−c and a−b−c problems.
- **Architectures** (pre-LN GPT-2 style, d 192, 6 heads):
  - **std:** 6 distinct layers, 2.68M parameters.
  - **loop:** one 2-layer block applied 3 times, 0.90M parameters. Same FLOPs as std.
  - **std2:** 2 distinct layers applied once, 0.90M parameters. Same parameters as loop, a third of the FLOPs.
- **Training** (`v220_study.py`): as the pilot. AdamW (0.9, 0.98), weight decay 0.1, batch 256, warmup 500, cosine
  over 30,000 steps to 10%, fp16 autocast, clip 1.0. No early stop: dig runs 30,000 steps, mod 10,000.
- **Learning rates** (pilots, `docs/v220_design.md`): dig std 3e-3, loop 3e-4, std2 3e-4; mod 1e-3 for all three.
  Pilot steps to 0.99 held-out accuracy: dig 12k / 25k / 20.5k; mod 4k / 4k / 5k.
- **Grid:** 3 architectures × 2 formats × seeds 1, 2, 3 = 18 runs. The pilots used seed 0.
- **Measurement:** at step 0 and every 100 steps (dig) or 50 steps (mod), in fp32 and eval mode, inside the training
  run. That gives 301 and 201 checkpoints per run. Held-out accuracy is logged every 500 steps. fp16 weights are saved
  at 40 log-spaced steps and the last step, for a later use test with its own pre-registration (not part of this one).
- Kaggle T4 ×2, account aurmerin, kernel astrocyte-v220-study, both GPUs.

## 3. Measurement (`v220_measure.py`)

**Probe set** (fixed, seed 220):
- **dig:** per operation, 8 families. A family fixes the tens digits and holds all 1,000 unit triples (ua, ub, uc),
  all of them valid problems. Labels are taken mod 5, so they depend only on the units digits (10 = 0 mod 5). Within a
  family every label varies, and the answer is not a function of a op b (it depends on uc). 16,000 prompts.
- **mod:** per operation, 4,000 random (a, b, c) triples. Labels mod 23.

**Labels**, per operation:

| label | a+b−c | a−b−c |
|---|---|---|
| **p1**, the first partial | a+b | a−b |
| **twin**, the other sign | a−b | a+b |
| **ac** | a−c | a−c |
| **bc** | b−c | b+c (since a−b−c = a−(b+c)) |
| **ans** | the answer | the answer |

**Positions:**
- **b:** the end of b. c is not yet visible.
- **c:** the end of c.
- **out:** where the answer's last token is predicted. For dig this is the answer's tens digit, fed in (teacher
  forcing); for mod it is '='.

**Layers:** the residual stream after every block application (std 6, loop 6, std2 2).

**Probe:** a closed-form ridge classifier on standardised states, λ = 0.01 × n_train, one solve for all labels.
- dig: states are family-demeaned, and the 4 folds are pairs of families.
- mod: 5 random folds, no demeaning, since there is no prompt template to remove.
- This replaces the logistic probe with nested cross-validation used in v178–v216, which is too slow for about 300
  checkpoints × 3 positions × up to 6 layers per run.

**Copy ceiling:** the same probe on one-hot copies of the digits visible at the position:
- dig: ua and ub; at c also uc; at out also the answer's tens digit.
- mod: a and b; at c and out also c.

excess e = probe accuracy on the state − accuracy on the copies. The ceiling depends only on the inputs, so it is
computed once per format.

**Model output:**
- dig: the predicted units digit at out, mod 5.
- mod: the predicted answer at out.

Its accuracy against ans is recorded.

## 4. Locked analysis (`v220_analyze.py`)

**Statistics.** Per run and checkpoint, at a position:
- **I:** the maximum over layers of mean_op(e_p1 − e_twin). D+ and D− are e_p1 − e_twin on op + and op − at that
  layer. I is the sign-specific statistic: a fixed preference for one form cancels.
- **dA:** the maximum over layers of mean_op e_ans, minus the same at step 0.
- **m:** model-output accuracy (mean over operations) minus the larger of ans's copy ceiling and 1/K.
- **dX**, for X in p1, ac and bc: the maximum over layers of mean_op e_X, minus the same at step 0.

Each series is smoothed by a centred running median of 5 checkpoints. A *run* means at least 3 consecutive
checkpoints.

Step 0 is subtracted from single-label readouts because untrained networks already mix the digits. Pre-freeze
measurement of 18 untrained networks (the study's architectures and seeds) gave:
- dig out: e_ans up to +0.062; p1, ac and bc up to +0.043.
- mod out: p1 +0.10 to +0.165; ac and bc about +0.10.
- I at most +0.005 at out, +0.010 at c and +0.028 at b.

**Run labels**, at out (primary) and c (secondary):

| label | condition |
|---|---|
| **AHEAD** | a run with I ≥ 0.05, D+ > 0, D− > 0, dA ≤ 0.02 and m ≤ 0.02 |
| **WITH-OR-AFTER** | no such run, but a run with I ≥ 0.05, D+ > 0 and D− > 0 |
| **NEVER** | otherwise |

**Cell label** (architecture, format, position): the label of at least 2 of the 3 seeds, otherwise SPLIT.

**Times:**
- t_I: the first run of I ≥ 0.05 with D+ and D− > 0;
- t_A: the first run of dA ≥ 0.05;
- t_m: the first run of m ≥ 0.05;
- lead = (t_A − t_I) / t_A.

**Ranking of the partials** (descriptive label, at c and out):
- t_X is the first step of a run with dX ≥ 0.05.
- **X-FIRST:** the earliest X, if the next X starts at least 3 checkpoints later or never. **TIED** otherwise.
  **NONE** if no X reaches the bar.
- Cell: the majority of seeds, otherwise SPLIT.
- Why it matters: an answer-plus-copies state reads all three partials, because (on op + and op −) a op b = A + c,
  a − c = A ∓ b and b ∓ c = ±(A − a). So p1-first is a second signature against answer-mixing, and it holds even when the answer
  is present.

**Position b** (descriptive): t_I at b. c is not yet seen there, so a op b at b cannot be answer-then-mix.

**Architecture comparison** (descriptive, n = 3), per format, on lead at out:
- **LOOP-DIFFERS:** loop's seed range is disjoint from both std's range and std2's range, on the same side.
- **NO-DIFFERENCE:** otherwise.
- **NA:** any lead is undefined.

A looping claim needs the difference against both: std matches compute but not the learning rate on dig, and std2
matches parameters and the learning rate.

## 5. Predictions (at freeze)

- dig out: AHEAD 35%, WITH-OR-AFTER 45%, NEVER 20%, per architecture (about the same for each).
- mod out: AHEAD 15%, WITH-OR-AFTER 60%, NEVER 25%. The pilot's mod accuracy jumps within about 1,000 steps, so the
  partial and the answer may arrive together.
- Ranking at c, dig: P1-FIRST 45%, TIED 35%, another partial first 20%.
- Architecture comparison: NO-DIFFERENCE 75% per format.

## 6. What each outcome means (fixed now)

- **AHEAD** (in a format, for an architecture): in these models, a sign-specific a op b is held at the output position
  while the answer is not readable there and the model cannot yet produce it. There, it is an intermediate, not the
  answer with c mixed back in. This is reported as a training-order resolution of the Ouro note's ambiguity *in small
  trained transformers*, with no claim about Ouro.
- **WITH-OR-AFTER:** training order does not separate the accounts in that cell, and the ambiguity stands. If P1-FIRST
  holds there, it is reported as partial evidence against answer-mixing at the same checkpoints, labelled descriptive.
- **NEVER:** these models do not hold a sign-specific a op b at that position. They reach the answer without a
  readable first partial there.
- **SPLIT:** reported per seed; no cell-level claim.
- **Architecture:** LOOP-DIFFERS against both std and std2 supports a looping effect on the order. A difference against
  only one of them is reported as size, compute or learning rate, not looping.
- Every label is reported, including NEVER and SPLIT.

**Limits stated with any result:**
- The probe is linear and the "answer absent" test is operational: the answer is not linearly readable at the
  position (beyond copies and step 0) and is not produced by the model.
- One task family, small models.
- Loop vs std differs in learning rate on dig.
- The use of a op b by the answer is not tested here.

## 7. Pre-freeze checks (2026-10-25)

- **Pilots** (`results_v220/pilot`, `results_v220/pilot_std2`; learnability only, seed 0): every architecture learns
  both formats answer-only. The learning rates in section 2 come from them.
- **Measurement selftest** (`python v220_measure.py`; synthetic states, excess at out, mean over operations):

  | world | dig | mod |
  |---|---|---|
  | copies only | all labels within ±0.03; I 0.000 | I +0.002 |
  | p1 code | p1 +0.79, I +0.80 | p1 +0.93, I +0.93 |
  | answer code | ans +0.80, I −0.01 | ans +0.97, I +0.003 |
  | answer + c mix | p1 +0.66, I +0.66, ans +0.80 | p1 +0.06, I +0.09, ans +0.93 |

  Answer-mixing gives a large sign-following signal only with the answer present, which is why AHEAD requires the
  answer to be absent. Copy ceilings at out are about chance (dig 0.20–0.22; mod 0.02–0.07).
- **Untrained networks** (control, section 4 numbers): the I threshold of 0.05 is above every step-0 value (maximum
  +0.028). Single-label readouts are not, so they are measured from step 0.
- **Analysis selftest** (`python v220_analyze.py --selftest`; synthetic curves, 3 seeds each): PASS.

  | world | labelled |
  |---|---|
  | partial before the answer | AHEAD |
  | together | WITH-OR-AFTER |
  | partial after the answer | WITH-OR-AFTER |
  | no partial | NEVER |
  | isolated spikes before the answer | WITH-OR-AFTER (spikes removed) |
  | one-sided sign effect (D− < 0) | NEVER |
  | raised step-0 answer baseline | AHEAD (the step-0 subtraction works) |

- **TINY end-to-end** (CPU, `TINY=1`): training, in-loop measurement, JSON output and weight checkpoints run for loop
  dig and std2 mod, and the analysis reads the outputs.
- No probe statistic was computed on any trained checkpoint before the freeze.

## 8. Frozen files (sha256, first 16 hex, LF; 2026-10-25)

- `v220_train.py` 6fbfc35236c2376d (tasks, Net; also the pilot trainer)
- `v220_measure.py` 6394738908134401
- `v220_study.py` f6261159c0c73353
- `v220_analyze.py` 7386724d21832e4e
- `build_kaggle_v220.py` a53f5040adc8a298

Run order: build `V220_SLUG=study python build_kaggle_v220.py`, push astrocyte-v220-study (GPU, `--accelerator
NvidiaTeslaT4`), download, then run `python v220_analyze.py results_v220/study` (CPU).
