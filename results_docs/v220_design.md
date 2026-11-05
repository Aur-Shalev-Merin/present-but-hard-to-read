# v220 design: can training order separate "a+b formed first" from "answer formed, c mixed back in"?

*Drafted 2026-10-24. Status: **PILOT** (learnability only). No pre-registration yet. The study's pre-registration will
be written and frozen after the pilot, before any probe or ladder statistic is computed on any checkpoint.*

## Question
In the Ouro note, a+b in a+b−c problems was readable but could not be told apart from "the answer was formed and c added
back", because the two give identical probe labels at any one time.

Training order can break that tie in one direction. Under answer-then-mix, an a+b signal cannot exist before the answer
is decodable from the state. So if, at some checkpoint, an a+b signal appears before the answer is decodable, the
answer-then-mix account is ruled out. The signal must be beyond the copy ceiling, sign-specific (it follows the
operation, as in Study 3's operation swap), and ahead of the other pairwise partial results. If a+b only ever appears
with or after the answer, the result stays ambiguous. That asymmetry will be stated in the pre-registration.

Literature check (2026-10-24): the generic results are already
published:
- intermediates decodable before they are used (Wang et al. 2024; Ye J. et al. 2025; Lepori et al.);
- looped vs standard developmental stages (Kohli et al. 2026);
- arithmetic circuits across checkpoints (Exoo et al. 2026);
- circuit maps over training (Tigges et al. 2024).

What is not published:
- the timing test of this ambiguity;
- copy-ceiling, sign-flip and operation-swap controls applied over training;
- looped vs standard order on arithmetic.

## Pilot (v220_train.py; Kaggle T4 ×2, account aurmerin, about 2–3 GPU-hours)
- **Tasks:**
  - dig: two-digit a, b, c, answer in 10..99, fresh problems each step;
  - mod: mod 23 with 20% of combinations held out.

  Both have a+b−c and a−b−c versions, answer-only.
- **Models:** std (6 layers, d 192) against loop (a 2-layer block × 3, weight-tied, the same FLOPs).
- **Grid:** learning rates 3e-4, 1e-3, 3e-3, one seed, up to 30k steps (2 architectures × 2 formats × 3 rates = 12 cells).
- **Reads only learnability:**
  - steps to 0.95 and 0.99 held-out accuracy;
  - plus vs minus accuracy (subtraction may be harder: Lin et al. 2025's "variable as subtrahend");
  - stability.
- **Decides:**
  - whether both architectures learn both formats answer-only;
  - the learning rate per architecture (Sesodia et al. 2026: looped and untied models need different rates);
  - the step budget and the log-spaced checkpoint schedule for the study.
- **If a cell never learns,** the study's design changes before any pre-registration. Options: a larger model, a
  curriculum, or dropping that format.

## Study (after the pilot; to be pre-registered)
- 12 runs: std and loop × dig and mod × 3 seeds, at the pilot-chosen rates.
- About 40 log-spaced checkpoints per run, including step 0.
- At each checkpoint:
  - minimal-pair families: vary the units digits of a and b, with c and the answer held fixed where possible;
  - copy ceilings;
  - the pairwise partial results a±b, a−c, b−c and their sign twins, plus the answer, mod 5;
  - the answer-mixing null;
  - patching at about 6 checkpoints per run.
- Reduced measurement set: about 2 CPU-hours.

## Pilot result (2026-10-25; kernel aurmerin/astrocyte-v220-pilot v1; outputs `results_v220/pilot/`)
Learnability only; no probe statistic was computed. One seed. Wall time 25 min on T4 ×2 (both exit codes 0).

| cell | lr 3e-4 | lr 1e-3 | lr 3e-3 |
|---|---|---|---|
| std dig (2.68M params) | 0.99 at 24.5k | 0.95 at 23.5k, final 0.98 | **0.99 at 12k** |
| loop dig (0.90M params) | **0.99 at 25k** | final 0.97 | final 0.97 |
| std mod | 0.99 at 7.5k | **0.99 at 4k** | 0.99 at 5k |
| loop mod | 0.99 at 7k | **0.99 at 4k** | 0.99 at 6.5k |

Findings:
- **Both architectures learn both formats answer-only.** Mod accuracy is on held-out combinations, so this is
  generalisation.
- **The learning rates differ.** For dig, std is best at 3e-3 and loop at 3e-4: loop does not reach 0.99 in 30k steps
  at 1e-3 or 3e-3. For mod, both are best at 1e-3.
- **Mod is a jump.** Accuracy goes from chance to above 0.9 within about 1,000 steps, starting at 1–2.5k steps
  depending on the rate. At the 500-step evaluation interval that transition is only one or two points. The study
  needs dense checkpoints, every 50–100 steps, through the first ~4k steps.
- **Dig is gradual but noisy.** Accuracy rises over 5–25k steps with large swings between evaluations. Minus problems
  collapse transiently, to as low as 0.12 (std at 3e-4 and 1e-3), so subtraction is less stable, as expected. Any
  "decodable from step t" milestone must be defined as the first checkpoint after which the probe stays above
  threshold, not the first crossing.
- **Confound to state:** loop matches std in FLOPs, not parameters (0.90M vs 2.68M).
- **Seed spread is unknown.** The study's 3 seeds will show it, so the checkpoint schedule must not depend on
  pilot step counts. Instead, save fp16 weights every 100 steps throughout. That is about 1.5 GB per std run, or 13 GB
  in total, under Kaggle's 20 GB per kernel when the runs are split over two kernels.

## Size-matched control pilot (2026-10-25; kernel aurmerin/astrocyte-v220-pilot-std2 v1; `results_v220/pilot_std2/`)
Added after a review: loop matches std in FLOPs, not parameters. **std2** is 2 distinct layers applied once (the
loop block without reuse): 0.90M parameters like loop, a third of the FLOPs. Learnability only, one seed, 7 min.

| cell | lr 3e-4 | lr 1e-3 | lr 3e-3 |
|---|---|---|---|
| std2 dig | **0.99 at 20.5k** | 0.99 at 23k | final 0.965 (minus 0.83) |
| std2 mod | **0.99 at 2.5k** | 0.99 at 5k | final 0.97 |

- **The learning-rate split follows size, not looping.** std2 prefers 3e-4 on dig and fails at 3e-3, like loop. So loop
  vs std2 can run at the same rate (3e-4) with the same parameters, which makes it the clean looping comparison. Loop
  vs std (compute-matched) keeps the rate difference, and is stated as such.
- On dig, std2 learns slightly faster than loop (20.5k vs 25k). The loop's extra compute does not speed learning here.
- Mod: all three architectures reach 0.99 by 4–5k at 1e-3, so the study can use one rate, 1e-3, for all of them.
