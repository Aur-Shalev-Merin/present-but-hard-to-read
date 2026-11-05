# v221 pre-registration: can the protocol say yes? A positive control inside the chain

Workstream paper3 (Ouro note). Claimed 2026-10-26. Follows v215 and v216.

## 1. Why

Every computed-value test in the note so far is negative or unclear (Ouro: not beyond mixing, not used; Kudo et al.'s
task: no sub-answer before the chain, with or without one). An outside review (2026-10-26) asked whether the protocol
can return a positive result at all, and noted that its mixing null never changed a verdict. This study runs the
protocol, unchanged, where a computed value is certainly present and used: inside Kudo et al.'s Simple chain
(" b = B , a = A", v215's short-chain arm), at the step where Qwen2.5-7B writes b. The chain does not restate b's
equation, so B is computed from the operands in the prompt.

**What it can and cannot show (review note, before the freeze).** At the token before the digit, B is the next token,
so a positive result there is close to built in. The study therefore also reads the two tokens before it, "=" (B two
tokens ahead) and the letter b (three tokens ahead). Even there B is about to be written; none of these positions is a
hidden intermediate. A positive result shows that the protocol's bars (copy ceiling, operation swap, mixing null,
copy-controlled clamp) can be passed by a value that is computed and used, in this model, task and probe; it does not
show that the protocol finds hidden intermediates. The paper will say so.

## 2. Task, items, positions

- v215's 240 families and 12,000 prompts in the Simple chain format (`v221_items.py` = `v215_items.build_short`, with
  three positions added), teacher-forced prompt + " b = B , a = A".
- Positions: **eq** (the d2 token; v215 result: COMBINED, DOES-NOT-FOLLOW, clamp cBo 0.021), **last** (the final '?'),
  **letb** (the letter b in the chain), **eqb** (the "=" after it), **inb** (the space before B's digit, where B is
  predicted). Tokenization checked locally: ' t', ' =', ' ', then the digit.
- Qwen2.5-7B fp16 over two T4s, hidden_states 0, 4, ..., 28; `correct` = the B digit is the argmax at inb.

## 3. Part A (v215's rules, unchanged)

`v215_analyze.part_a` at all five positions: labels S = (d1 + d2) mod 5, M = (d1 - d2) mod 5, copy ceiling, band
{12, 16, 20, 24}, ex_N, I, cluster bootstrap (seed 215). G1: b step correct >= 0.90 in each version (v215's short-arm
gate); G2: copies >= 0.50. COMBINED / NOT-COMBINED / UNCLEAR on ex_N; FOLLOWS / DOES-NOT-FOLLOW / UNCLEAR on I.
Mixing null: `v215_null.null` (copies + one generic mixture of d1, d2, o1; no B; three kinds, calibrated to the observed
ex_N, 6 seeds), for each COMBINED position: BEYOND-MIXING if the observed I exceeds every calibrated world's I.

## 4. Part B (v215's clamp, unchanged)

`v215_analyze.patch_plan` for eq, eqb and inb when COMBINED (B subspace, d2-copy subspace, B orthogonalised against the
copy, random; fitted on the other half of the families; 1,000 pairs per position, x correct, x' same family, version
and d1, different d2). `v215_patch` via `v221_patch.py`: conditions none, cB, cBo, cD, cR, full at the band layers at
token p; outcome = the B digit predicted at inb; USED cBo - cR >= 0.10 with CI lower > 0; NOT-USED CI upper < 0.02;
UNSCORABLE full < 0.30; seed 2153.

## 5. Predictions and readings

- inb: COMBINED 97%, FOLLOWS 90%, BEYOND-MIXING 85%, USED 85%. eqb: COMBINED 90%, FOLLOWS 70%, BEYOND-MIXING 60%,
  USED 60%. letb: FOLLOWS 50%. eq: as v215 (COMBINED, DOES-NOT-FOLLOW, NOT-USED) 85%.
- **Protocol passes the control:** COMBINED + FOLLOWS + BEYOND-MIXING + USED at inb, while eq stays negative. Reported as
  a check of the protocol's power for a value about to be written (section 1), and the position furthest from the
  digit at which it passes is stated.
- **Protocol fails the control** (any of FOLLOWS, BEYOND-MIXING or USED missing at inb): the protocol cannot detect a
  value the model is certainly computing and using at that token, so its negative results elsewhere in the note lose
  most of their force. The paper says so in the main text.
- Partial outcomes are reported label by label.

## 6. Post hoc, descriptive: their probe on the copied operands (no label)

`v221_copyprobe.py` on v215's saved replication states (their chain format; eq / semi / last): Kudo et al.'s standard
probe (v215 Part C settings) for d1 and d2, which are in the prompt and read at 1.000 by v215's minimal pairs at every
position. Reported: accuracy per position and layer, and whether the maximum reaches their 90% threshold. A maximum
below 0.90 would mean their rule calls a present operand absent in their own task; above it, the critique of single
-prompt probes does not transfer to their prompts (which differ only in their numbers). Either is reported.

## 7. Pre-freeze checks

- Items (`python v221_items.py`): 12,000, the same problems as v215's short arm; positions ' t' / ' =' / ' ' / digit
  (Qwen tokenizer, local).
- Selftest (`python v221_analyze.py --selftest`, 60 synthetic families, five positions): see section 9.

## 8. Frozen files (sha256, first 16 hex, LF; 2026-10-26)

- `v221_items.py` 2b6bd412a702ccc4 (ITEMS_SHA 50a50828de872803)
- `v221_extract.py` 9803a05dd8bb2666
- `v221_analyze.py` ef20cc70e480d5cc
- `v221_patch.py` e5348b0f791b5fa0
- `v221_copyprobe.py` 83a40fdaf9e2c543
- `build_kaggle_v221.py` 4cd85dc26d3705ed
- reused unchanged: `v215_analyze.py` e9d53523b1da41e2, `v215_null.py` cfcc60d81dcb6d25, `v215_patch.py`
  cc5aa1418b0f278e, `v215_items.py` 9249f0523d7219eb, `v215_pilot.py` d049a0b2d81fb80b, `v178_analyze.py`
  90bbc8a848dc5544

Run order: `V221_MODE=smoke` (GPU, mechanics; pending a free GPU slot at the freeze, so it runs after it and before
the extraction; a failure there is handled by an amendment) -> `x` (GPU) -> `a` (CPU) -> `null` (CPU) and `patch` (GPU).
`copy` (CPU, section 6) runs on v215's saved states and may run any time after the freeze.

## 9. Selftest record (2026-10-26, `python v221_analyze.py --selftest`, 60 synthetic families)

- b computed (a shared code of B at every position): COMBINED + FOLLOWS at all five positions (ex_N +0.418 to +0.419,
  I +0.394 to +0.404).
- Copies only: NOT-COMBINED + DOES-NOT-FOLLOW at all five (ex_N -0.146 to -0.151, I -0.011 to +0.008).
- Plan: positions eq, eqb, inb; 3,000 pairs, 84 bases.
- No statistic on real v221 states has been computed before this freeze.

## 10. Execution notes

1. (2026-10-26) Both GPU slots on the main Kaggle account were taken by another study, so the GPU kernels and the
   CPU kernels that read their outputs (smoke, x, a, null, patch) run on the second account (aurmerin). They are built
   by the frozen `build_kaggle_v221.py`, unchanged; only `kernel-metadata.json` is rewritten afterwards (the account in
   `id` and `kernel_sources`), since Kaggle kernels read outputs from the same account only. The embedded code is
   identical. `copy` ran on the main account (it reads v215's states there).

## 11. Result: their probe on the copied operands (section 6, post hoc, descriptive)

`astrocyte-v221-copy` (CPU, main account; v215 items sha 8c08adec21d20ae3 as frozen). Test accuracy, layers 0, 4, ..., 28:

| label | position | max | their 90% rule |
|---|---|---|---|
| d1 | eq | 1.000 (layer 4) | present |
| d1 | semi | 0.939 (layer 4) | present |
| d1 | last | 0.593 (layer 4) | **absent** |
| d2 | eq | 1.000 (all layers) | present |
| d2 | semi | 1.000 (layers 4, 8) | present |
| d2 | last | 0.998 (layer 4) | present |

At the final token their rule calls the first operand absent, though it is in the prompt and v215's minimal pairs read
it at 1.000 there. At eq and semi it calls both present, so the critique of single-prompt probes transfers to their
prompts at the end of the question, not everywhere. Accuracy falls with depth at every position (d1 at eq: 1.00 at
layer 4, 0.53 at layer 28). Layer 0 at positions after d1 is below the test majority (0.03): the embedding of a fixed
token carries no label and the held-out split's classes differ from the training split's.

## 12. RESULT (2026-10-26; frozen code, run on the second account per execution note 1)

Gates: b step correct plus 0.909, minus 0.913 (G1 >= 0.90 passes); copies 1.000 at every position (G2 passes).

| position | B is | Part A (ex_N) | operation swap (I) | mixing null (max null I) | clamp (cBo - cR) |
|---|---|---|---|---|---|
| eq | not yet written | COMBINED +0.374 | DOES-NOT-FOLLOW +0.001 [-0.002, +0.004] | NOT-SEPARABLE (+0.007) | UNCLEAR +0.021 [+0.011, +0.033] |
| letb | 3 tokens ahead | COMBINED +0.150 | UNCLEAR +0.032 [+0.027, +0.038] | BEYOND-MIXING (+0.017) | not planned |
| eqb | 2 tokens ahead | COMBINED +0.177 | UNCLEAR +0.024 [+0.019, +0.028] | BEYOND-MIXING (+0.017) | UNSCORABLE (full 0.098) |
| inb | next token | COMBINED +0.253 | UNCLEAR +0.017 [+0.013, +0.021] | BEYOND-MIXING (+0.017, margin 0.0004) | **USED +0.607 [+0.565, +0.646]** |
| last | after the question | COMBINED +0.121 | UNCLEAR +0.031 [+0.026, +0.037] | BEYOND-MIXING (+0.017) | not planned |

inb clamp detail (1,000 pairs): hit none 0.000, cB 0.715, cBo 0.607, cD 0.021, cR 0.000, full 0.954. eq reproduces
v215 exactly (cB 0.302, cBo 0.021, cD 0.962, full 0.963), as it should (same items, same code).

**Reading (section 5): the protocol fails the control.** At inb, three of the four bars pass (COMBINED, BEYOND-MIXING
by a margin of 0.0004, USED by a wide one), but the operation swap is UNCLEAR, not FOLLOWS: I is above zero (CI
excludes 0) but well under the 0.05 bar. Read label by label:

- **Clamp: has power.** Removing B's copy-orthogonal directions at the token before the digit changes the digit in
  61% of pairs (random subspace 0%). This is where B is the next token, so it is close to built in (section 1); at eqb
  the clamp cannot be scored (moving the whole state at "=" changes the digit in only 10% of pairs; B is computed at
  inb from the prompt).
- **Operation swap: no power here.** It does not reach FOLLOWS at any position, including where the model is about
  to write B. Descriptive (not a reason to change the label): at inb the computed label and the other operation's
  label are both read near ceiling at layers 24-28 (plus: S 0.98, M 0.99; minus: M 1.00, S 0.98), so the swap has
  little room. The band {12, 16, 20, 24} also averages in layers where both are lower.
- **Mixing null: separates for the first time** (four positions BEYOND-MIXING; at eq NOT-SEPARABLE), but at inb only
  just.
- Predictions: inb FOLLOWS (90%) wrong; inb BEYOND-MIXING (85%) and USED (85%) right; eq as v215 right on Part A and
  the swap, clamp UNCLEAR (as in v215) rather than the predicted NOT-USED.

Consequence for the paper (section 5, fixed in advance): the swap's negatives (DOES-NOT-FOLLOW, UNCLEAR) carry little
force, since the swap fails where B is certainly computed; this goes in the main text. The clamp's negatives keep
their force only where the clamp can be scored, and its power is shown only one token before the digit.
