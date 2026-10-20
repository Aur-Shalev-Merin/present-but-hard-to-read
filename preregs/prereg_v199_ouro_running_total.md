# v199 pre-registration: does Ouro keep a running total in the story? (operation swap at a sentence end + clamped patching over its tail)

*Drafted 2026-10-14. Status: **FROZEN 2026-10-14** (this commit, pushed to origin before any main-set run). No
main-set state exists; only the pilot (control labels and conditions) and the synthetic selftests have run. Frozen files
(sha256, first 16 hex chars, LF):
- `v194_items.json` fceceeb4dcbd4076 (unchanged from v194);
- `v199_extract.py` 21ad7985e69f74a0; `v199_analyze.py` af267849e6da1411; `v199_patch.py` b9eca8a653dac65f;
  `v199_pilot.py` 10a3750e3f629851;
- `build_kaggle_v199.py` 0dcad3aafed08c44; `build_kaggle_v199_cpu.py` 53d592e642ef0510;
- `v178_analyze.py` 90bbc8a848dc5544; `v186_analyze.py` b408d943603262a6; `loop_grad_probe.py` 27b80fa0b999bfa7
  (the same Ouro code as v188 and v194).

The real-probe selftest (section 4, v2) and pilot v2 (section 5) ran on exactly these `v199_analyze.py`,
`v199_extract.py` and `v199_patch.py`.*

## 1. Why

v194 found that the unwritten first step (a + b, or a − b) is readable at P1, just before Ouro writes the answer's
units digit, and that its sign follows the operation the problem names (Part A, FOLLOWS). Swapping its subspace at P1
changed nothing (Part B, NOT-USED at all four tests). The reading was scoped to P1. Positions before P1 kept x's own
states, so an intermediate formed earlier, inside the question, and read from there through attention was not tested.

The natural place for such an intermediate is the end of the sentence that gives the second number:
"Her friend gave her 13 more stickers." A reader knows there that Iris now has a + b stickers, and language models
are known to track the states of entities in stories. If Ouro keeps a running total, it should be readable there.
- At that point c has not appeared and the question has not been asked. A value readable there cannot be "the answer
  with c added back", the rival that limited v194's Part A.
- **Part A (primary):** is the needed first step readable at Sb, the period that ends that sentence, beyond what
  copies of a and b allow? And does its sign follow the operation (gave/took)?
- **Part B (secondary; runs if the running total is readable, section 3):** if the first step's subspace on every
  token of the sentence's tail is held at another problem's value throughout the run, does the answer Ouro writes
  change?

## 2. Part A (primary): operation swap at the sentence end

**Items.** The v194 main set, unchanged (`v194_items.json`, sha256 fceceeb4dcbd4076).
- 12 templates × 2 versions (plus "gave her B more stickers", minus "took B of her stickers") × 3 families × 200
  unit-digit combinations = 14,400 items. Each plus item has a minus twin with identical numbers.
- Every question has three statements and a question, in that order: a, then b with the operation, then c.

**States** (`v199_extract.py`; the forward pass, prompt and correctness scorer are v194's).
- One teacher-forced forward pass over the 4-shot answer-only prompt plus " <answer>".
- **Sb** is the period ending the sentence that gives b. **Sc** is the period ending the sentence that gives c.
- The four tokens before Sb (Sb−1 … Sb−4) are also stored. The sentence's **tail** is every token after b's units
  digit up to and including the period: 2–5 tokens by template ("litres." to "books to the shelf."). Positions at or
  before b's units digit are stored but never used.
- States are taken at the mid and end of each loop. Answer-only correctness after each loop is recorded, so G1 is
  recomputed.
- Positions are found by tokenising prefixes, with asserts. They were checked on all 15,120 main and pilot items:
  - every family has a single set of positions;
  - Sb and Sc decode as ".", and bU (b's units digit) decodes as b's units digit;
  - the period is the first token at which every template has stated the operation (in "20 more cars arrived" it comes
    after b), which is why Sb is the primary readout;
  - prompts are about 270 tokens.

**Probe** (`v199_analyze.py`).
- The v178 probe on family-demeaned states, folds grouped by template, separately for each version.
- Labels are taken mod 5 on units digits: AB = (a + b) mod 5, AMB = (a − b) mod 5.
- The copy ceiling is the same probe on one-hot (a_u, b_u, c_u). c has not appeared at Sb, so its part of the ceiling
  is inert there.
- excess = acc(state) − acc(ceiling), per item.

**Primary statistic** (Sb, at each loop end: end1, end2, end3, end4).
- D+ = excess(AB) − excess(AMB) on plus items.
- D− = excess(AMB) − excess(AB) on minus items.
- **I = (mean D+ + mean D−) / 2.** A fixed preference for either sign cancels; it is reported as
  F = (mean D− − mean D+) / 2.
- Paired cluster bootstrap over pair ids (a plus family with its minus twin): 10,000 resamples, seed 199.
- **99.375% CI.** Bonferroni: Part A gets 2.5%, split over the four loop ends; Part B gets the other 2.5%.
- No loop end is singled out in advance. Where an intermediate would form at Sb is unknown; at P1 the needed form
  peaked at end2 or end3, depending on version.

**Labels.**
- **FOLLOWS:** at one or more loop ends, I ≥ 0.05, CI lower bound > 0, mean D+ > 0 and mean D− > 0.
- **DOES-NOT-FOLLOW:** at every loop end, I's CI upper bound < 0.02.
- **UNCLEAR:** otherwise.

**Gates.** If either fails, the label is UNSCORABLE.
- G1: answer-only accuracy after loop 4 ≥ 0.30 in each version. v194 had 0.674 and 0.765 on the same prompts.
- G2: copies a and b at Sb, best state, ≥ 0.75 in each version. The pilot read 0.96–1.00, so the bar stays at 0.75
  (section 5).

**Bar rationale and power.**
- At P1, v194's I was +0.157 (end4) and +0.132 (end3). 0.05 is about a third of that: a running total held as
  clearly as at P1 would pass easily, and one a third as strong would still pass.
- v194's 97.5% interval at P1 had a half-width of 0.016 with the same items. At 99.375% the half-width should be
  about 0.02, so the bar can be resolved.

**Part B gate** (not a label). Part B runs if Part A is FOLLOWS, or if the same FOLLOWS rule holds at Sb−1 (the token
before the period, inside every template's tail) at some loop end, with Part A not UNSCORABLE. This keeps Part B from
being skipped when the total sits one token early. Part B is then judged only by its own test.

**Secondary** (reported, not labels).
- I at the Sb mid states, and at Sb−1 at the loop ends.
- At Sc: I for the first step at each loop end, and I for the answer (a + b − c against a − b − c, mod 5) at every
  state. This asks whether the answer is formed before the question is asked.
- Copies a, b and c at Sb (every state) and at Sc (loop ends).

## 3. Part B (secondary): clamped patching over the sentence's tail

Run only if the Part B gate in section 2 says RUN. Otherwise no running total is readable at the sentence end, there is
nothing to clamp, and Part B is not run (no GPU).

**Plan** (`v199_analyze.py`, `patch_plan`, written by the CPU analysis whatever Part A's label).
- The templates are split into halves {0–5} and {6–11}.
- For each version, half and tail offset k = 0–4 (the token Sb−k), 9-dimensional subspaces are fitted on the OTHER
  half's items of that version whose tail contains Sb−k:
  - the family-centred class means of the needed first step N, which is (a + b) mod 10 in plus and (a − b) mod 10 in
    minus;
  - the same for its sign-flipped twin F;
  - at each patch point: mid1, end1, mid2, end2, mid3, end3, mid4.
- An offset that the other half lacks uses the subspace of the nearest smaller offset. This only affects offset 4 for
  half 0, which uses offset 3.
- A random orthonormal 9-dimensional subspace is drawn per half (seed 1991).
- **Pairs (x, x′):** in the same family, with
  - the same c_u;
  - N_x′ ≠ N_x (mod 10), with the same parity, so a parity-only effect cannot produce a hit;
  - x correct after loop 4.

  500 pairs are drawn per version and half (2,000 in all; seed 1992).

**Intervention** (`v199_patch.py`).
1. Run x′ unpatched and record its states on the tail and at bU (b's units digit) at the seven points.
2. Run x. At each of the seven points, overwrite part of the state with x′'s state at that point:

   | condition | where | what is overwritten |
   |---|---|---|
   | N | every tail token Sb−k | the needed first step's subspace for that point and offset: h ← h + U Uᵀ (h_x′ − h) |
   | F | every tail token | the sign-flipped twin's subspace |
   | rand | every tail token | a random 9-dimensional subspace |
   | full | every tail token | the whole state |
   | Nsb | Sb only | the needed first step's subspace (secondary) |
   | fullsb | Sb only | the whole state (secondary) |
   | btok | bU | the whole state (method control: replaces the second number itself) |
   | none | — | nothing |

3. Read the predicted token at P1 after loop 4, which is the answer's units digit.

**Why the whole tail.** A running total could sit on the words between the number and the period ("more stickers")
as well as on the period. Clamping only the period could then miss a total that is used (review note, before the
freeze). The period-only conditions are kept as secondary readouts.

**Why clamp at every loop.** In v194, a single patch at the end of loop 2 was mostly undone by loops 3–4: the positive
control moved the prediction in only 33% of pairs. Re-applying the patch at the middle and end of every loop keeps the
subspace at x′'s value at every point where it is overwritten. End4 is not patched, because nothing after it reaches
P1.

**Hits.**
- All conditions but btok: the prediction equals x′'s answer units digit. Since c is shared, that is
  (N_x′ − c) mod 10, the digit x would write if it used x′'s first step.
- btok: the prediction equals (a_x ± b_x′ − c_x) mod 10, scored on pairs whose b_u differ.

**Statistic** (tail conditions). hit(N) − max(hit(F), hit(rand)), with a cluster bootstrap over x's family: 10,000
resamples, seed 1993, 97.5% CI.

**Labels.**
- **UNSCORABLE:** hit(btok) < 0.30. Clamping a question position, even the second number itself, cannot steer the
  answer. The pilot's btok hit was 0.857, so the bar stays at 0.30 (section 5).
- **USED:** difference ≥ 0.10 with CI lower bound > 0.
- **NOT-USED:** CI upper bound < 0.02.
- **UNCLEAR:** otherwise.

**Descriptive.**
- hit(full), which shows how much the answer depends on the tail at all;
- hit(Nsb) and hit(fullsb), the period alone;
- each condition's own-digit rate;
- all of the above per version.

**Limits stated in advance.**
- Only the tail is clamped. A running total held at later positions (the c sentence, the question) is not clamped
  there, and it may be re-derived from the unclamped number tokens a and b.
  - hit(full) bounds this. If clamping the whole tail barely moves the answer, the tail is not a route.
  - The secondary Sc answer readout shows whether a total is held later.
- Loop 1's first half (layers 1–12) is not clamped, so later positions see x's own tail there.
- The subspaces are linear spans of class means. The first tail token sits next to b, so its state also carries a
  strong copy of b. N's subspace is fitted to N's class means, and in a balanced design these average out b's main
  effect. F and rand guard against the remainder.

## 4. Selftest

- **Synthetic worlds on the real item structure** (`v199_analyze.py --selftest`, d = 64):
  - copies only;
  - a running total that follows the operation;
  - a fixed preference for differences;
  - generic mixing;
  - partial (40% of items);
  - broken (no copies);
  - tail only (the total one token before the period, not at it), added with the tail clamp.
- **Expected results:** follows → FOLLOWS, with Part B RUN; copies only, fixed difference and generic mixing →
  DOES-NOT-FOLLOW, with Part B SKIP; broken → UNSCORABLE; tail only → not FOLLOWS, with Part B RUN through the Sb−1
  gate; partial 0.4 is recorded.
- **Stub probe** (nearest class mean, local): every world comes out as expected. The patch plan was checked against
  its rule on 2,000 pairs; it has 282 subspaces.
- **Real probe, first version** (Kaggle CPU `astrocyte-v199-selftest` v1, before the tail changes): every world comes
  out as expected.
  - follows: I +0.729 [+0.711, +0.747].
  - partial 0.4: I +0.307.
  - fixed difference: I −0.000, F +0.730.
  - generic mixing: both forms read +0.77, I −0.000.
  - broken: UNSCORABLE.
- **Real probe on the current file** (v2, with the tail-only world; `results_v199/selftest_v2/`): every world comes
  out as expected, with the expected Part B gate. tail only → DOES-NOT-FOLLOW with Part B RUN.
- **Part B readout on synthetic rows:** a planted 30% effect gives USED (+0.245 [+0.224, +0.267]), and a null gives
  NOT-USED.
- **Patching code:** it runs in the pilot (section 5), on GPU. A local check is impossible because the local
  transformers cannot load Ouro's remote code; Kaggle pins 4.54.1. The tail-clamp version needs its own smoke run
  (pilot v2) before the freeze.

## 5. Pilot (control labels and control conditions only; 720 pilot items: other names and tens digits)

`astrocyte-v199-pilot`, v1 (Sb-only patch code; `results_v199/pilot_v1/`), and v2 (the frozen code, with the tail
clamp and six stored positions; `results_v199/pilot_v2/`).

**v2** ran the frozen `v199_extract.py` and `v199_patch.py`, byte for byte.
- States: 720 × 6 × 8 × 2048. Logit check against Ouro's own forward: 0.0. Tails of 2–5 tokens.
- Accuracy and copies are identical to v1, since Sb and Sc are unchanged.
- Patching smoke with the tail clamp: btok hit 0.857 (plus 0.881, minus 0.837); rand hit 0.000, own digit 1.000.
- Wall time: 1.22 s per item for extraction, and 4.2 s per pair for 3 conditions. Main extraction: about 2.4 hours
  per GPU. Part B with 8 conditions: about 2.6 hours per GPU.

**v1 results** (control labels and conditions only).
- **Answer-only accuracy by loop.**
  - plus: 0.044 / 0.453 / 0.644 / 0.658.
  - minus: 0.036 / 0.672 / 0.728 / 0.728.
  - The same as v194's pilot. G1 is safe.
- **Copies at Sb, by state (mid1 … end4), and best.**
  - plus: a 0.975 → 0.581 (best 0.975); b 1.000 → 0.564 (best 1.000).
  - minus: a 0.961 → 0.547 (best 0.961); b 1.000 → 0.631 (best 1.000).
  - G2's bar of 0.75 is safe.
  - c at Sb reads 0.10–0.14, at chance, as it must, since c comes later. This checks that nothing leaks from later
    positions.
- **Copies at Sc, loop ends.**
  - a: 0.92 → 0.55.
  - b: 0.78 → 0.35.
  - c: 0.97 → 0.41.
- **Patching smoke** (103 pairs where Ouro writes its own digit unpatched, 24 families).
  - btok hit 0.857 (plus 0.881, minus 0.837), on the 91 pairs whose b differs. Clamping the second number's token at
    the seven points moves the answer to the one the new b gives. The method control's bar of 0.30 stays.
  - rand: hit 0.000, own digit 1.000. A random 9-dimensional clamp changes nothing.
- **GPU 0:** extraction, then answer-only accuracy by loop and copies a, b, c at Sb and Sc (gate G2 risk). No first
  step or answer label is computed.
- **GPU 1:** patching smoke test on up to 120 pilot pairs, with random subspaces and only the control conditions none,
  rand and btok. It reports the btok hit (the method control's bar), the unpatched own-digit rate, and rand's
  own-digit rate. N, F, full, Nsb and fullsb are not run before the freeze.

## 6. Predictions (made before any main-set state)

- Gates pass: P ≈ 0.85.
- **Part A:** FOLLOWS 25%, UNCLEAR 35%, DOES-NOT-FOLLOW 30%, UNSCORABLE 10%.
  - Bare addition and v194's P0 (I +0.017) showed little sign-specific structure before the answer starts.
  - A sentence end is a common place for summaries, and copies of a and b are strong there.
- **Part B (if run):** USED 35%, UNCLEAR 30%, NOT-USED 25%, UNSCORABLE 10%. Overall chance of "readable and used in
  the tail": about 10%.
- **Secondary, Sc answer I:** some answer signal is likely, since the answer digit already led at P1 after loop 2. I do
  not predict its size.

## 7. What each outcome means for the note

- **B USED:** Ouro keeps a running total in the story and relies on it. To our knowledge this would be the first
  causal evidence of an unwritten intermediate in a pretrained looped model. The computation section becomes a
  positive claim, located in the story rather than at P1.
- **B NOT-USED:** readable but not used, now at both places. The note's "present but not used" reading
  extends to the story positions.
- **Part B gate SKIP:** no running total is readable at the sentence end beyond copies and generic mixing.
  The note says so in one sentence, and the untested routes narrow to other positions and nonlinear codes.
- **The Sc answer readout** is reported descriptively whatever happens. It has no causal test here.

## 8. Run plan and compute

1. Pilot (GPU) and real-probe selftest (CPU): done (sections 4–5).
2. Freeze: this commit, pushed to origin before any main-set run, after scanning for tokens.
3. Main extraction, `astrocyte-v199-ouro` (T4 ×2): about 2.5 hours, 14,400 items.
4. CPU analysis, `astrocyte-v199-cpu` (kernel_sources): Part A and the patch plan, about 1–2 hours.
5. Only if the Part B gate says RUN: Part B, `astrocyte-v199-patch` (T4 ×2). About 2.5 hours: 2,000 pairs × 8
   conditions.

## 9. RESULT, Part A (2026-10-14; `results_v199/cpu/v199_analysis.txt`)

Run: extraction `aurshalevmerin/astrocyte-v199-ouro` (T4 ×2, 2.6–2.7 h per GPU; logit check 0.0 on both), analysis
`astrocyte-v199-cpu` (CPU, 71 min). Every file ran at its frozen hash (`v199_extract.py` 21ad7985, `v199_analyze.py`
af267849, `v178_analyze.py` 90bbc8a8, `v186_analyze.py` b408d943, `loop_grad_probe.py` 27b80fa0, `v194_items.json`
fceceeb4).

**Gates: both pass.**
- G1: accuracy after loop 4, plus 0.674 and minus 0.765. These are v194's numbers exactly, as they should be (same items,
  same prompt, same forward pass).
- G2: copies at Sb, best state: plus a 0.999 / b 1.000; minus a 0.999 / b 1.000.

**Primary (Sb, loop ends): UNCLEAR** (predicted 35%; FOLLOWS 25%, DOES-NOT-FOLLOW 30%).

| Sb | D+ (plus) | D− (minus) | I [99.375% CI] | F [99.375% CI] |
|---|---|---|---|---|
| end1 | −0.010 | +0.025 | +0.008 [−0.009, +0.024] | +0.017 [−0.005, +0.042] |
| end2 | −0.010 | +0.039 | +0.014 [−0.002, +0.032] | +0.024 [+0.001, +0.048] |
| end3 | −0.001 | +0.036 | +0.018 [−0.001, +0.036] | +0.018 [−0.009, +0.045] |
| end4 | +0.005 | +0.018 | +0.011 [−0.007, +0.032] | +0.006 [−0.015, +0.028] |

- No loop end comes near FOLLOWS: the largest I is +0.018, and every upper bound (0.024–0.036) is below the 0.05 bar.
- It is not DOES-NOT-FOLLOW either: every upper bound is above 0.02. So the rule gives UNCLEAR. The data exclude a
  sign-following signal above 0.036 at every loop end, about a quarter of v194's +0.157 at P1 on the same items;
  they do not exclude one of 0.02–0.03.
- D+ is negative at end1–end3: on plus problems a − b reads at least as well as a + b at Sb. The positive F (a fixed
  preference for differences) is what remains, as in bare addition and at P1 (v194 F +0.032).

**Part B gate: SKIP.** At Sb−1 the largest I is +0.001 (end2), and D+ is negative at every loop end. Per section 8, Part
B was not run. The patch plan was written (2,267–2,894 candidate pairs per version and half) and is kept, unused.

**Secondary (not labels).**
- Sb mid states: I −0.002 / +0.017 [+0.001, +0.033] / +0.007 / +0.003 (mid1–mid4). F is positive at mid1–mid3
  (+0.035 to +0.053).
- Both a + b and a − b read above the copy ceiling at Sb early on (+0.04 to +0.13 at mid1–end2, either version), and
  the excess fades by end4 (−0.015 to +0.016). The state mixes a and b, but not according to the operation.
- Sc, first step: I −0.001 to +0.006 at the loop ends.
- **Sc, answer** (a + b − c against a − b − c): I −0.002 to +0.006 at every state, and the needed answer's own excess is
  at most +0.013. The answer is not readable before the question is asked. (I expected some answer signal here; it is
  not there.)
- Copies at Sb: a and b 0.98–1.00 through end2, fading to 0.68–0.72 by end4; c at chance (0.09–0.11), as it has not
  appeared. Copies at Sc fade faster (b 0.89–0.94 at end1–end2 down to 0.40–0.47 at end4).

**Reading, per section 7 (Part B gate SKIP).**
- At the end of the sentence that gives b, the first step is not readable beyond copies of a and b and generic mixing
  of them. Its sign does not follow the operation (largest I +0.018, against +0.157 at P1 on the same problems).
- The answer is not readable at the end of the sentence that gives c either.
- Together with v194: the combination the problem needs appears at P1, when Ouro writes the answer, and not as a
  running total in the story. Ouro does not use it at P1 either.
- Untested routes that remain: other question positions (the question sentence, "Answer:"), and nonlinear or
  non-class-mean codes. A sign-following signal below about 0.03 at Sb is not excluded.
