# v215 pre-registration: is the sub-answer derived before the chain of thought?

Workstream paper3 (Ouro note, arXiv:2610.28498). Claimed 2026-10-23.

## 1. Why

Kudo et al. (Findings of EACL 2026, pp. 1114-1153) conclude that LLMs "have not derived an answer at the moment when
input was fed into the model. Instead, they compute (sub-)answers while generating the reasoning chain". Their evidence:
a standard linear probe per (token position, layer) reads the sub-answer at 17-51% before the chain (Qwen2.5-7B,
Level 3: B 50.5%, A 17.9%) against 95-100% inside it, judged against a 90% threshold; and patching the input part
barely changes the chain. Our note shows a failed single-prompt probe is weak evidence of absence. This study applies
the note's protocol to their task: minimal pairs, a measured copy ceiling, the operation swap, a calibrated mixing
null, and a clamp with a positive control. It tests the presence of a value computed from the operands before the
chain, and its use by the chain.

**Framing, fixed now.** Their probe already reads B well above chance before the chain; their 90% threshold turns that
into "absent", and part of it can come from copies of the input digits alone (a plain sum is linear in the copies; with
both operations the copies alone read B at about 0.27-0.42 in this design, computed from the items alone). The
realistic best case is "present but not used", which refines their claim rather than reversing it. Every outcome
below is reported, and none is preferred in the analysis.

## 2. Task and items (their Level 3 format, re-implemented; their repository has no licence)

- Problem `a = k o2 b , b = d1 o1 d2 ; a?` at character level, three same-level demonstrations before it, target CoT
  ` a = k o2 b , b = d1 o1 d2 , b = B , a = k o2 b , a = k o2 B , a = A` (their simple_backtracking format); every value
  0-9; lowercase letters. Pilot (v215 pilot, 2026-10-23): Qwen2.5-7B exact CoT 0.995, final answer 1.000 (200 problems).
- **Families (Parts A, B):** a family fixes the letters, the three demonstrations and (k, o2) and contains every
  operand pair (d1, d2) with d1 >= d2 that is valid in BOTH versions, plus (o1 = +) and minus (o1 = -). (k, o2) in
  {(0,+), (1,+), (2,+), (9,-), (8,-), (7,-)} so that every family has 20-30 pairs (40-60 prompts), and k varies across
  families (this separates B from the answer A = k o2 B, which within a family is a fixed function of B). 240 families,
  40 per (k, o2): about 12,000 prompts.
- **Short-chain arm (Parts A, B):** the same 240 families (same letters, numbers and demonstrations) in their Simple
  CoT format (App. A): target ` b = B , a = A` with no restated equation, demonstrations in the same format. Pre-chain
  states differ only through the demonstrations' format. Added before the freeze because in the main format no
  pre-chain state can feed the chain's `b =` step (smoke, below), so use is testable only here.
- **Replication set (Part C):** 12,000 single problems drawn as in their generator (all valid (k, o2, d1, o1, d2),
  random letters and demonstrations), split 10,000 / 2,000 so that no test problem shares b's equation (d1, o1, d2)
  with a training problem.
- Positions: **eq** = the d2 token (end of b's equation), **last** = the final '?' before the chain; also ';' for Part C.
  Layers 0, 4, ..., 28 (hidden_states index). Qwen2.5-7B fp16 over two T4s (v213 backend), 0.2 s/prompt (pilot).

## 3. Part A: presence of a computed value before the chain (minimal pairs)

- Probe: `v178_analyze.probe_predict` on family-demeaned states, outer 5 folds grouped by family, separately per
  version (as v194). Labels mod 5: S = (d1 + d2) mod 5, M = (d1 - d2) mod 5; needed N = S on plus, M on minus.
  Copy ceiling: the same probe on one-hot (d1, d2) (k, o2 and o1 are fixed within a family and version). Per item,
  excess(L) = hit(L, state) - hit(L, copy ceiling).
- Statistics per position, averaged over the layer band {12, 16, 20, 24} (fixed now; the pilot's standard probe for b
  peaks at 12-20):
  - ex_N = mean over versions of excess(N) (the needed form read beyond copies; generic mixing of the two digits
    also produces this, see the selftest, so ex_N alone shows only that the operands were combined)
  - I = (D+ + D-)/2, D+ = excess(S) - excess(M) on plus, D- = excess(M) - excess(S) on minus (sign follows the
    operation; a fixed preference for either form cancels)
  - cluster bootstrap over families (plus and minus share a family), 10,000 resamples, seed 215, 97.5% CI per
    position (two positions).
- Labels per position:
  - **UNSCORABLE:** G1 exact-chain accuracy < 0.90 in either version (every family item, teacher forcing: every target
    token is the argmax before it, i.e. greedy decoding writes exactly the chain; in the short-chain arm the chain's b
    step instead, since smoke gave exact chain 0.90 and b step 1.00 on 40 items), or G2 the copies d1 and d2 read by
    the family-demeaned probe at the best layer < 0.50 at that position.
  - **COMBINED:** ex_N >= 0.05 and CI lower > 0.
  - **NOT-COMBINED:** CI upper of ex_N < 0.02.
  - **UNCLEAR:** otherwise.
  - The operation swap is reported with its own label at each position: **FOLLOWS** (I >= 0.05, CI lower > 0, D+ > 0,
    D- > 0), **DOES-NOT-FOLLOW** (CI upper < 0.02), **UNCLEAR**.
- Mixing null (as v212, rebuilt on these items, run after Part A with the procedure fixed now): synthetic states with
  a family offset, copies of d1 and d2, and one generic nonlinear mixture of (d1, d2, o1) (pair products of random
  codes; same-frequency Fourier products; a random one-layer network), with NO computed value (neither B nor A);
  per mixture family the strength lam* is matched to the observed ex_N at each COMBINED position (two-stage
  calibration, v212 rule), then 6 seeds. **BEYOND-MIXING** if the observed I exceeds every calibrated world's I;
  otherwise **NOT-SEPARABLE**; a mixture that cannot reach the observed ex_N is UNCALIBRATED.
- Secondary (descriptive): excess of A mod 5 (the answer); every layer.
- **Reading.** "A computed sub-answer is present before the chain" requires COMBINED, FOLLOWS and BEYOND-MIXING at
  that position. COMBINED alone (or with NOT-SEPARABLE) supports only "the operands are combined there", which
  generic mixing does too. Because the probes are fitted per version, a mixture that depends on o1 cannot by itself
  prefer the needed form, so the null's I is expected near 0 and FOLLOWS carries most of the weight.

## 4. Part B: is it used by the chain? (short-chain arm only; clamp at its COMBINED positions; a use claim only where
## FOLLOWS and BEYOND-MIXING too)

- Main format: not run. Smoke (40 pairs, before the freeze): replacing the whole pre-chain state (eq or last) by
  another problem's at every layer output 11-27 never changes the digit the chain writes at `b =` (0/40; own B kept
  40/40), while the same full patch at the chain's own `b =` token changes it 40/40 (hooks work). The chain restates
  the equation and computes B from that text, so the pre-chain value cannot be used at that step in this format
  (their recency finding). Reported as such.

- Pairs (x, x'): short-chain arm, same family, version and d1, a different d2 (so a different B), x's b step correct;
  1,000 per position, seed 2151. (Smoke: with pairs differing in both operands, the full patch at eq changed B in 90% of
  pairs but to x''s B in only 30%, since it carries only d2's token.)
- Subspace: 9-dim, family-centred class means of B (0-9) at that position and layer, fitted on the other half of the
  families (as v194); a random orthonormal 9-dim control (seed 2152).
- Interventions at that position (hooks on the decoder-layer outputs): **cB** at each band layer (outputs of layers
  11, 15, 19, 23 = hidden_states 12-24) set x's B-subspace component to x''s, so later band layers cannot rebuild it
  (as v209; it can be rebuilt between band layers); **cR** the same for the random subspace; **full** (positive
  control) replace x's whole state at that position by x''s at every layer output 11-27.
- Smoke check before the real run: 20 pairs per position with the random subspace in place of the B subspace
  (mechanics only; no B-subspace intervention is run before the freeze).
- Outcome: teacher-forced up to the chain's first step ` b =` (short format), argmax next digit; hit = x''s B.
- Labels: **UNSCORABLE** hit(full) < 0.30 (the position does not feed that step; consistent with their recency
  finding, and then no use claim either way); **USED** hit(cB) - hit(cR) >= 0.10, CI lower > 0; **NOT-USED** CI upper
  < 0.02; **UNCLEAR** otherwise. Cluster bootstrap over families, 10,000 resamples, seed 2153, 97.5% CI.

## 5. Part C: their probe on our replication set (reference, descriptive)

Standard probe (logistic on single prompts, standardised states) for b's and a's values at eq, ';' and last, every
layer; Acc_preCoT = max over these positions and layers (theirs is over every pre-chain token). Reported next to their
B 50.5%, A 17.9%, and next to Part A at the same positions.

## 6. Predictions

- Main format: COMBINED at eq 40%, at last 25%; FOLLOWS at eq 30%, at last 20%; BEYOND-MIXING where FOLLOWS 50%.
- Short-chain arm: COMBINED at last 45% (the model must produce B right after it); FOLLOWS at last 35%.
- Part B (short arm): UNSCORABLE 25%, NOT-USED 35%, USED 40% (where run).
- Part C: b 0.35-0.70, a 0.15-0.30.

## 7. What each outcome means for the paper (fixed now)

- **COMBINED (+ FOLLOWS, BEYOND-MIXING) at a position, NOT-USED or UNSCORABLE:** "a value computed from both operands
  is present before the chain (at that position), beyond what copies and generic mixing give; their probe and threshold
  call it absent". Their claim that the chain is faithful to the answer stands; the claim that nothing is derived
  before the chain does not. Reported as a refinement, with their numbers.
- **COMBINED but NOT-SEPARABLE:** present beyond copies, but generic mixing explains the sign-following part: weaker
  claim ("something combined from both operands is there"), as in our Study 3.
- **COMBINED but DOES-NOT-FOLLOW (or UNCLEAR):** the two operands are combined at that position beyond copies, but
  nothing sign-specific is readable: the generic-mixing pattern of the selftest. No claim that the sub-answer is
  derived there; reported as consistent with their conclusion (and as a caution that "beyond copies" alone would have
  overstated it). Part B, if run there, is descriptive only.
- **COMBINED + FOLLOWS + BEYOND-MIXING and USED:** the chain uses a value present before it; reported as evidence against "computed during the
  chain" at that position, with the patching caveats.
- **Short-chain arm:** read with the same rules; it answers a narrower question ("when the chain does not restate the
  step, is the value computed before it and used?"). A result there is stated for that format only.
- **NOT-COMBINED at both positions:** confirms their conclusion with a stronger method (minimal pairs and copy
  controls); the note says the protocol agrees with them here.
- In every case: Part C numbers next to theirs; their causal-patching result is cited as complementary.
- Before posting, the authors are contacted with the result (the author's decision).

## 8. Pre-freeze checks

- Part A selftest (`python v215_analyze.py --selftest`, local; 60 synthetic families on the real design): computed
  (a shared code of B) -> COMBINED + FOLLOWS (ex_N +0.42, I +0.39-0.41); copies only -> NOT-COMBINED +
  DOES-NOT-FOLLOW (ex_N -0.15); op-independent pair mixing of (d1, d2) -> COMBINED (ex_N +0.36) but DOES-NOT-FOLLOW
  (I +0.001-0.006). Hence the reading rule above.
- Null selftest (`python v215_null.py --selftest`).
- Items (`python v215_items.py`): 240 families, 6 (k, o2) x 40, 20-30 operand pairs each, 12,000 family items
  (6,000 per version); replication 10,000 / 2,000 with 22 held-out b equations.
- Pilot (done): format and accuracy; timing; standard-probe positions. No minimal-pair statistic was computed.
- Smoke (Kaggle, `astrocyte-v215-smoke` v1-v3): extraction 40 items per part: main format exact chain 40/40, short
  format 36/40 (b step 40/40), replication 40/40; 0 tokenisation boundary mismatches; 0.14-0.22 s/item. Patch mechanics
  (random subspace in place of the B subspace): main format full patch at eq / last 0/40 to x''s B, own B kept 40/40;
  the same full patch at the chain's own `b =` token 40/40 (hooks work). Short format: full patch at eq changes B in
  90% (to x''s B 30% with two-operand pairs), at last 0/20; with d2-only pairs (v4): eq 20/20 to x''s B, last 0/20
  (so Part B at last will be UNSCORABLE by rule). No B-subspace intervention and no minimal-pair statistic on real
  states before the freeze.

## 9. Frozen files (sha256, first 16 hex, LF; 2026-10-23)

- `v215_items.py` 9249f0523d7219eb
- `v215_pilot.py` d049a0b2d81fb80b
- `v215_extract.py` 8920e85c68d600fb
- `v215_analyze.py` 018065ad26f4d887
- `v215_null.py` cfcc60d81dcb6d25
- `v215_patch.py` 4186f6f47baff625
- `v178_analyze.py` 90bbc8a848dc5544
- `build_kaggle_v215.py` 49ce949efc2c42a8
- ITEMS_SHA (fam + rep + fams) 8c08adec21d20ae3

Kernels print the sha of every file they run. Run order: `V215_MODE=x` (GPU) -> `a` (CPU) -> `null` (CPU, if any
position is COMBINED) -> `patch` (GPU, short-chain arm, if a COMBINED position has a plan).

## 10. Amendment 1 (2026-10-23, during extraction, before any analysis of real states)

A reviewer noted that at the eq position (the d2 token) a "used" result from the B-subspace clamp could be a copy
effect: within a family B = d1 +- d2, and x' differs from x only in d2, so the B subspace overlaps the copied digit's
directions; swapping it could change the copied d2 and let the chain compute x''s B from the copy, with nothing
derived early. Changes (Part B only; Parts A and C unchanged):
- The plan also fits the **d2-copy subspace** at each position and band layer (family-centred class means of d2 on the
  other half of the families; d2 <= 4 in this design, so 4 dims) and **cBo**, the B basis orthogonalised against it.
  It reports the overlap (mean and max cos^2 of the principal angles) and the share of B left.
- New conditions: **cBo** (clamp in the orthogonalised B subspace) and **cD** (clamp in the d2-copy subspace, a
  reference for the copy alone). **The primary statistic becomes hit(cBo) - hit(cR)**, with the same labels and
  bootstrap; hit(cB) - hit(cR) is reported as descriptive.
- Reading: USED needs cBo; if cB moves the chain and cBo does not, the reading is "the swap works through the copied
  digit". If the overlap leaves little of B (share left < 0.5 at the band layers), the cBo test is reported as weak.
Files: `v215_analyze.py` 018065ad26f4d887 -> e9d53523b1da41e2, `v215_patch.py` 4186f6f47baff625 -> cc5aa1418b0f278e (other files unchanged; a first amended version a49957490b8569f6 had a line-merge bug in the analysis step, caught by the smoke run and fixed before any real run).
Local check: the plan code runs on the synthetic computed world (B vs copy overlap mean cos^2 0.18-0.19, share of B
left 0.81-0.82); the Part A selftest still passes. Kaggle smoke of the new conditions: `astrocyte-v215-smoke` v5.

## 11. RESULT (2026-10-24; `results_v215/{a,null,patch}/`; files ran at the frozen + amendment-1 hashes)

Gates: G1 main format exact chain 0.998 / 0.998 (plus / minus); short arm b step 0.909 / 0.913 (exact chain 0.837);
replication 0.999. G2 copies 1.000 at both positions in both arms.

**Part A** (band layers 12-24; 97.5% CIs):

| arm | position | ex_N | label | I | swap label | null I max | null label |
|---|---|---|---|---|---|---|---|
| main | eq | +0.384 [+0.379, +0.388] | COMBINED | -0.003 [-0.005, -0.000] | DOES-NOT-FOLLOW | +0.006 | NOT-SEPARABLE |
| main | last | +0.137 [+0.128, +0.146] | COMBINED | +0.028 [+0.023, +0.034] | UNCLEAR | +0.017 | BEYOND-MIXING |
| short | eq | +0.374 [+0.370, +0.378] | COMBINED | +0.001 [-0.002, +0.004] | DOES-NOT-FOLLOW | +0.007 | NOT-SEPARABLE |
| short | last | +0.121 [+0.114, +0.129] | COMBINED | +0.031 [+0.026, +0.037] | UNCLEAR | +0.017 | BEYOND-MIXING |

At eq both S and M read at 0.87-1.00 on minimal pairs in both versions (the generic-mixing pattern of the selftest).
At last D+ is negative (-0.023 / -0.049: on plus items the difference form reads better) and D- positive (+0.080 /
+0.112): mostly a fixed preference for differences with a small sign-following part, which exceeds every calibrated
mixing world but not the 0.05 bar. The Fourier null could not be weakened to the observed ex_N at last (its smallest
grid strength already gave +0.30), so it runs strong there; pair and MLP worlds matched (I at most +0.007).
Predictions: COMBINED at eq 40% / last 25% (both COMBINED); FOLLOWS 30% / 20% (neither).

**Part B** (short arm; descriptive, since no position is FOLLOWS):
- eq (1,000 pairs): hit full 0.963 (positive control passes), cD (d2-copy subspace) 0.962, cB 0.302, cBo 0.021
  [+0.011, +0.033], cR 0.000 -> cBo - cR +0.021, UNCLEAR by rule. The chain's b step follows the copied d2 at its token;
  the B directions beyond the copy move it in 2% of pairs. B vs copy overlap mean cos^2 0.36-0.57 (share of B left
  0.43-0.64; below 0.5 at layer 24), so the cBo test is weak at that layer.
- last (1,000 pairs): full 0.000 -> UNSCORABLE (nothing at the last token feeds the b step, in either format).
- Main format (smoke, section 4): full patch before the chain 0/40.

**Part C** (their probe, train 10,000 / test 2,000, no shared b equation): b Acc_preCoT 0.636 (eq, layer 20; theirs
50.5%), a 0.318 (theirs 17.9%); at the last token b 0.39, a 0.23.

**Reading (section 7).** No position meets COMBINED + FOLLOWS + BEYOND-MIXING, so there is no claim that the
sub-answer is derived before the chain: the protocol agrees with Kudo et al. Their probe's pre-chain readability of b
(0.64 here) is the operands combined generically, which also lifts the "beyond copies" test (ex_N +0.37-0.38) without
any operation-specific value. At the last token a faint sign-following trace exceeds the mixing null (+0.03) but not
the pre-registered bar, and nothing there feeds the chain's b step. Where the chain does not restate the equation, it
reads b's operand from its own token: swapping the copied digit's directions moves the written b in 96% of pairs, the
B directions beyond the copy in 2%. Amendment 1 mattered: the plain B-subspace clamp (30%) would have suggested use.
