# v230 design (DRAFT, not frozen): a positive control for an unwritten, used intermediate

Workstream paper3 (Ouro note). Claimed 2026-10-28. Design draft only; the pre-registration is written after the
pilot (section 6) and frozen before any treatment run.

## 1. Why

Two outside reviews (2026-10-28, 4/10 each) gave the same main-track blocker: every positive control of the
protocol is a value the model is about to write (inb, one token before) or has written (its digit, read four tokens
later). Three tokens before b is written, the share swap is already UNCLEAR (D+ < 0). So the protocol's power for its
stated purpose, a value computed but never written, is unshown, and "readable but not used" (Ouro) and the Kudo
reading rest on a detector that has never said yes there.

Needed: a setting where an unwritten intermediate is known, by an independent causal method, to be computed and used,
read before the answer, so we can ask whether the frozen protocol (share swap, TAU, copy-projected clamp) says yes.

## 2. Literature check (2026-10-28; pages fetched, details marked unverified where so)

- No paper found validates a probe + operation-swap + clamp protocol against a ground-truth used latent intermediate.
- No ready pretrained-LLM setting with an unwritten, used arithmetic intermediate:
  - Kudo et al. (arXiv 2412.01113), implicit setting: probes never reach threshold (our re-test agrees).
  - MIB (Mueller et al., ICML 2025, arXiv 2504.13151) arithmetic track: DAS for the carry in Llama-3.1-8B reaches
    IIA 54 (best layer 65); Gemma-2 is at chance. A weak ground truth.
  - Mamidanna et al. (EMNLP 2025, arXiv 2509.09650): mental math is computed at the last token. That argues
    against an intermediate held at an earlier position in pretrained models.
- From-scratch models do hold step-wise intermediates:
  - Lin et al. (ACL 2025 Findings, arXiv 2503.07604): GPT-2 trained on mod-23 chained equations. Patching shows
    step-wise computation, but only with a fixed premise order. Code is public.
  - Brinkmann et al. (arXiv 2402.11917): intermediates are stored at selected positions.
- Pitfalls:
  - A subspace patch can act through a dormant pathway (Makelov et al., arXiv 2311.17030).
  - A model can take shortcuts. Our own v220: two of three from-scratch architectures solved a +/- b - c on two-digit
    numbers with no readable first partial result.
  - Results are sensitive to the format.

So: ground truth from a model trained from scratch, which we control, with DAS as the independent oracle.

## 3. Task (forces the intermediate; mirrors Ouro's design)

Tokens: `x op z ; [k fillers] w =` -> answer `y = (T[(x op z) mod p] + w) mod p`.
- Values: p = 23; op in {+, -}; T a fixed random permutation of Z_p.
- **Why T:** it removes the linear shortcut. With a permutation, y cannot be a signed sum of the operands, so any
  solution must represent something injective in s = x op z (or T[s], the same partition of problems).
- **Why w:** the answer does not fix s, as in Ouro, where answer = a + b - c. So the copy ceiling can include the
  answer without absorbing s.
- **Distance:** k in {0, 3, 6} filler tokens between the first clause and w. The read positions are ';' (the end of
  s's clause), the token before '=', and '='. This gives distances of 1 to about 9 tokens before the answer.
- **Plus/minus twins:** each family fixes (k, w, op pattern) and varies x and z over a factorial grid. Twin families
  differ only in op, as in Study 3 and the Kudo re-test.

**Negative-control model.** Same format, but `y = (T1[x] + T2[z] op w) mod p`. Here op matters but s = x op z is never
needed. The protocol must not give FOLLOWS or USED there.

**Models.** Reuse v220's harness (`v220_train.py`): small transformers trained from scratch, 3 seeds per task, plus
one looped variant if cheap. Train to >= 99% held-out accuracy.

## 4. Ground truth (independent of the protocol)

**DAS** (learned rotation, Geiger et al. / Wu et al.):
- For variable S at each (position, layer).
- Trained on half the pairs; interchange-intervention accuracy (IIA) measured on the held-out half. The prediction
  for (base, source) is T[s_source] + w_base.
- **Copy guard:** the DAS subspace is constrained orthogonal to the copy subspaces of x and z (and w). Otherwise a
  patch of the copies alone would also give T[s'], because the model could recompute s later from the copies.
  "s computed at (p, l)" means copy-orthogonal DAS reaches IIA >= 0.80.
  - Copy span (fixed after pilots 1-2): least squares of the state on one-hot codes of the inputs visible at the
    position (x, z, op; w only from its own token on), solved with a rank-safe solver; rank = singular values above
    the largest from 5 shuffled-label fits. Pilot 1's GPU solver returned 4 junk directions; pilot 2 kept every
    regression direction, and the noise-fitted ones held 84% of s's class-mean energy, so guarded DAS was 0 by
    construction.
- **Controls:**
  - a random subspace of the same dimension (IIA near 0);
  - DAS for W at its own token (sanity; pilot: 0.90, so DAS with m = 16 and 2,000 steps tops out near 0.90);
  - not DAS for S in the negative-control model: there the swap's target is undefined (s is not a variable of that
    task). The negative model's role is in the protocol arm (it must not give FOLLOWS or USED).
- The thresholds are set in the pre-registration, after the pilot. The protocol's numbers are not looked at
  beforehand.

## 5. Protocol arm (frozen code, nothing re-tuned)

At every (position, layer) that has a DAS verdict:
- share swap with the copies (x, z, w, answer) regressed, v224's TAU 0.30, plus a design-matched null as in v229;
- the class-mean clamp with the copies projected out (v221 rules).

**Outcomes:**
- **DETECTS:** where DAS says s is computed and used, the protocol gives FOLLOWS and USED, and in the negative model
  neither. The paper reports the protocol's power on unwritten values and at which distances it holds.
- **MISSES:** DAS says yes, the protocol says UNCLEAR or NOT-USED. The paper presents the protocol as falsification
  tests, and the Ouro "not used" and Kudo readings lose weight. Reported either way.
- **PARTIAL:** detects near the answer but not k tokens earlier. The paper states the distance limit.
- **NO-GROUND-TRUTH:** copy-orthogonal DAS finds s only at '=' (or nowhere before the answer). Then the control
  covers only "unwritten, transformed before output" at the answer position, which still answers review point 6.

## 6. Pilot (allowed before freezing; protocol numbers not inspected)

- Train one seed per task.
- Check accuracy, and check that the copies are present.
- Run DAS only, to see where S lives and to set IIA thresholds and DAS hyper-parameters.
- If s is never held before '=', try the fixed-order chained format of Lin et al. (`a = x op z , c = T(a) + w`)
  before freezing.

### Pilot log (2026-10-29)

- Pilot 1 (Kaggle aurmerin, kernel v1): std model 0.999 held-out (generalises at step 25k of 30k); negative model
  0.997; looped model memorises (0.37). Unguarded DAS: s at ';' rises over layers (IIA 0.09, 0.29, 0.64, 0.90 at
  layers 1-4), patch has no effect from layer 5 on; nothing at fillers or w; at most 0.13 at '='. Same for k = 6
  and k = 0. Guarded numbers invalid (solver bug).
- Pilot 2 (kernel v2): unguarded fits reproduce pilot 1 to three decimals; guarded IIA 0 everywhere (guard rank
  bug, above). Looped model 0.56 after 60k steps: dropped.
- Disclosure: the guard diagnostic computed s's class-mean energy at ';' (0.41 of the state's variance at layer 4,
  unregressed). This is not a protocol decision number (no operation swap, no copy regression), but it is close to
  the protocol's N; the pre-registration must state that it was seen. No threshold is set from it: TAU 0.30 is
  v224's, the bar comes from null worlds rebuilt on this design (as in v229), the clamp rules are v221's, and the
  read positions come from DAS. No further share-of-variance quantity is computed before the freeze; guard
  diagnostics use DAS-side quantities only.
- Pilot 3 (aurshalevmerin, 2026-10-31; commits 745d989, e90af7b): fixed guard plus a pre-flight guard check.
  - Seed 0: check passes (ranks 23-40). Guarded DAS finds s only at ';': layer 3 0.18 / 0.20, layer 4 0.63 / 0.68
    (k = 6 / k = 0), random 0; unguarded 0.90 / 0.91. Elsewhere at most 0.03. W sanity 0.90.
  - Seed 1: generalises at step 7.5k (0.995). Check fails at w's token, layer 1 (x's codes 0.67 inside the span):
    there x's code is 0.55% of the variance, about 4 times the shuffled-label floor, and the part outside the span is
    1.4 times noise, so the fraction is noise-dominated rather than the guard missing a copy. No DAS ran.
  - At ';' layer 4 in both seeds, the copy energy left outside the guard is 2-3 times noise (4-6% of x's code).
  - 0.80 is out of reach with m = 16: the W sanity itself tops out at 0.90, plausibly because a 23-valued variable
    needs up to 22 dimensions.
- Pilot 4 (aurshalevmerin, 2026-10-31; commit feb16ac): noise-referenced guard check, m = 32, up to 60k steps.
  - Seed 1 on the 60k schedule did not generalise (0.40; on pilot 3's 30k schedule it reached 0.995 at 7.5k).
    Generalisation depends on the LR schedule.
  - Seed 0 (30k schedule, same model as pilot 3), key cells, m = 32: W sanity 0.93 (m 16: 0.90). Guarded IIA at ';'
    layer 4 0.69 / 0.75 (m 16: 0.63 / 0.68); layer 3 0.31 / 0.28; '=' layer 5 0.05. Unguarded 0.86-0.92 at ';'.
  - Seed 2 (60k schedule, generalised at 9.5k): guard check passes (rank 26 at ';' layer 4). Unguarded DAS holds s at
    ';' into layer 5 (0.44, 0.69, 0.95, 0.98 at layers 2-5), but guarded only 0.05-0.16. Nothing elsewhere
    (<= 0.05). W sanity 0.52 (base kept 0.04): plausibly '=' already reads w in block 1, so a patch after block 1 is
    partly too late. The sanity site is model-dependent; layer 0 would be cleaner (forward_to does not support it
    yet).
  - Reading: in seed 2, most of what moves the answer at ';' lies in the copy span. The subspace guard cannot tell
    "s written into directions that overlap the copies' codes" from "the copies passed on and s computed later".
- Pilot 5 (aurshalevmerin kernel v3, 2026-10-31 to 11-01; commit 6e7a635): X / Z DAS at ';' (one operand takes the
  source's value, target T[x_src op z_base] + w_base; pairs where s' equals either problem's s are dropped), known
  answers at layer 0, seeds 1, 3, 4 on the 30k schedule, seeds 0 and 2 retrained (both reproduced: 0.99 at 25k and
  9.5k) for X / Z only.
  - Known answers: W, X, Z at their own tokens, layer 0: IIA 0.99-1.00 in all 5 seeds (random <= 0.01). The layer-1
    W check is model-dependent (0.93, 0.53, 0.51, 1.00, 0.05 in seeds 0-4; seed 4 keeps the base answer 0.79), so
    '=' reads w within block 1 in some seeds: layer 0 replaces it as the DAS ceiling.
  - Seeds 1, 3, 4 generalise on the 30k schedule (0.99 at 7.5k, 10k, 7k).
  - X and Z DAS at ';': at most 0.08 at any layer, k, seed (mostly 0.02-0.05; random about the same). No single
    operand patched at ';' moves the answer, at any layer.
  - S DAS at ';', unguarded, best layer (k = 6 / k = 0): seed 0 0.91 / 0.92 (pilot 4), seed 1 0.78 / 0.94 (L4),
    seed 2 0.98 / 0.98 (L5, pilot 4), seed 3 0.91 / 0.92 (L3), seed 4 0.87 / 0.87 (L3). Elsewhere (fillers, w's
    token, '=') at most 0.07 in every seed; layer 6 is 0 everywhere.
  - Guarded S at ';': seed 0 0.69 / 0.75, seed 1 0.17 / 0.32, seed 2 0.16, seed 4 0.41 / 0.42; seed 3's guard
    check failed (z's code outside the span 5.3 times noise at ';' layer 4), so no guarded fit.
  - Reading: in all 5 seeds the answer reads s from ';' (8 tokens before '=' at k = 6), and reads no operand from
    there separately. A joint patch of the x, z and op copies could also give the S target, but then X alone would
    give the X target, and it does not. So the operand test, not the subspace guard, separates "s computed" from
    "copies passed on"; the guard removes part of s in 4 of 5 seeds and is dropped from the ground-truth criterion
    (reported as a secondary column).

## 7. Optional secondary: DAS at the Kudo no-chain position

- Run DAS for b in Qwen2.5-7B, at v216's positions, with the same copy guard.
- If DAS finds b used there, our Kudo reading was a miss.
- If not, it adds independent support.
- Costs one GPU extraction plus DAS training.

## 8. Cost

The task generator and DAS add a few hundred lines to v220's harness. Training runs on a Kaggle T4, about 1-2 hours
per model at v220's scale (to be checked in the pilot). The protocol arm reuses v224/v228/v229 code on CPU. That
makes about 2-4 days with the pre-registration.
