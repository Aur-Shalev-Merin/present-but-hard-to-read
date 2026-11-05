# v232 pre-registration: a running total at the end of b's sentence in Ouro (share swap + copy-projected clamp)

Workstream paper3 (Ouro note). Claimed 2026-11-02. Follows v199, v228, v229, v231. Frozen 2026-11-03, before any run on Ouro states.

## 1. Why

Two outside reviews of the paper (2026-11-02) said the Ouro result cannot separate "a+b was formed and
the answer computed from it" from "the answer was formed and c added back": at P1, on units digits, these are the
same function. One reviewer named the remedy: the period ending the sentence that gives b (Sb; "Her friend gave her 13
more stickers."). There a and b are in the prompt and c is not, so a units code of a op b at Sb cannot be the answer
plus c. v199 tested Sb only with the accuracy-based swap, which later failed its positive control, and its clamp step
never ran because that swap was UNCLEAR. v232 re-scores Sb with the validated share swap, a bar set on Sb's own
design and a floor on the needed share, and runs v199's frozen clamp with the copies projected out and the comparison
against random only (v231: comparing with the sign-flipped twin is weak).

**Disclosures.**
1. v199's accuracy-swap results at Sb are known: UNCLEAR at every loop end (I at most +0.018, every upper bound at
   most 0.036), Sb-1 likewise; the needed and flipped forms both read above the copy ceiling (for example plus,
   end1: +0.098 and +0.108); the largest I was at end3 (+0.018) and mid2 (+0.017, secondary). The copies of a and b
   read at 0.68-1.00 at Sb and c at chance.
2. The share statistic has not been computed on any v199 state. v199's Part B (clamping) has never run.
3. Ouro is not a validation model of the share swap or the clamp. Clamp power for an unwritten value at this
   distance in a pretrained model is not shown (v231's power is for 6-layer models, at ';' eight tokens before '=').
4. The N floor is new. It is set from v230's one false positive (a null world FOLLOWS at N 0.0009) and from the
   synthetic power check in section 5 (seen before the freeze; synthetic worlds only, no Ouro state).
5. Part B's statistic differs from v199's frozen one (hit(N) - max(hit(F), hit(rand))) because v231 showed the twin
   comparison is weak; the frozen statistic is reported, descriptive. The copy projection is new to this clamp.
6. Designed after the outside reviews and after the post hoc a+b-vs-random check on v194/v209 (UNCLEAR for the loop-3
   patch and the clamp at P1).

## 2. Part A: the share swap at Sb (primary) (`v232_sb.py cpu`)

States: v199's saved main-set states (aurshalevmerin/astrocyte-v199-ouro, merged; 14,400 items, 6 positions x 8
states, d 2048). `v228_rescore.analyse` unchanged: labels S, M, R0..R3 from the units digits of a and b; one-hot copies
of **a and b** regressed (the only varied digits before Sb; c, the answer and the carry are not in the prompt there);
cluster pair; states **mid3, end3, mid4, end4** (v228's rule for Ouro, kept so the choice does not follow disclosure
1); 10,000 resamples, seed 2241, 97.5% CIs; labels as v224 section 2 with TAU = TAU_SB.

**Null on Sb's design, scored first (before any real state is loaded):** v224's five mixture kinds (pair, fourier,
mlp, paircode, paircode_op) built on Study 3's main-set items: copies U1[a] + U2[b], one generic mixture of (a, b, op)
at lam x the copies' RMS, family offsets, N(0, 1) noise drawn once per (family, a, b, op), so that items differing only
in c share one state, as the causal states at Sb do; d 256; **no world holds a op b**; lam in {0.05, 0.1, 0.25, 0.5,
1, 2, 4, 8, 16}, seeds 1-6 (270 runs, 1,000 resamples each). **TAU_SB = max(0.20, 1.5 x the largest I / N among runs
with signal (N CI lower > 0), rounded up to 0.05)**: v224's and v229's rule.

**N floor:** a FOLLOWS needs N >= **N_FLOOR = 0.002** (about twice the needed share of v230's one false positive,
and a third of Ouro P1's 0.007); a FOLLOWS below it is labelled **TOO-FAINT**.

Labels at Sb (primary): FOLLOWS / TOO-FAINT / DOES-NOT-FOLLOW / UNCLEAR / NO-SIGNAL. Secondary: the same at Sb-1 (the
token before the period). Descriptive: both positions over mid1, end1, mid2, end2 (1,000 resamples); every null run.

**Power read at Ouro's mixing (descriptive, run after Part A, reported whatever the label).** For each null kind, the
lam whose seed-1 run has the needed share N nearest Ouro's observed N at Sb; a units code of a op b is planted at comp
0.05, 0.08, 0.12, 0.2, 0.3, 0.5 x the copies' RMS (seed 101) and scored with Part A's rule (TAU_SB, floor). Reported:
the smallest comp labelled FOLLOWS per kind and its I. This says how strong a running total would have to be for the
swap to see it at the mixing level Ouro's states suggest.

## 3. Part B: v199's clamp over the sentence's tail, copies projected out (`v199_patch.py`, frozen; `v232_sb.py b`)

**Plan (CPU, before any GPU run).** v199's frozen subspaces (aurshalevmerin/astrocyte-v199-cpu: for each version, half
of the templates, tail offset 0-4 and patch point mid1-mid4, the 9-dim span of the family-centred class means of the
needed first step N = (a op b) mod 10 and of its twin F, fitted on the other half) are first **recomputed from the
states and checked against the frozen plan** (max ||UU^T - U0U0^T|| < 1e-3, else no plan is written and Part B does
not run). Each frozen subspace is then projected off the copies: the 9-dim class-mean spans of a's and b's units digits
at the same point, offset and fit set; U <- the left singular vectors of U - QQ^T U with singular value > 1e-3 x the
largest (v216's projection). The random subspaces and v199's pairs (500 per version and half; same family, same c,
needed step differing with the same parity, x correct) are unchanged.

**Run (GPU).** `v199_patch.py extract`, byte for byte, one half of the pairs per T4, on the projected plan: at mid1,
end1, mid2, end2, mid3, end3 and mid4 every tail token's component (N, F, rand), the whole tail (full), the period
only (Nsb, fullsb) or b's own token (btok) is set to x''s, and the digit at P1 after loop 4 is read.

**Statistic:** hit(N) - hit(rand), cluster bootstrap over x's family, 10,000 resamples, seed 2321, 97.5% CI.
**UNSCORABLE** if hit(btok) < 0.30 (v199's method control: clamping b's own token cannot steer the answer); **USED**
>= 0.10 with CI lower > 0; **NOT-USED** CI upper < 0.02; **UNCLEAR** otherwise. Descriptive: v199's frozen statistic,
hit(full), hit(Nsb), hit(fullsb), the share of N's directions left after the projection.

Bonferroni: Part A's primary test and Part B each at 97.5%.

## 4. Readings

**Part A (Sb):**
- **FOLLOWS:** by the end of the sentence that gives b, before c is read, a units code of a op b follows the operation
  beyond copies of a and b and a bar set by generic mixing on this design. That cannot be the answer with c added back.
  The paper adds it to the Ouro section and the abstract, as a running total at Sb (the P1 ambiguity stays at P1).
- **TOO-FAINT:** a code follows the operation but below the needed-share floor; reported as such, no claim.
- **UNCLEAR / NO-SIGNAL / DOES-NOT-FOLLOW:** no running total is shown at Sb, and this is **weak evidence of
  absence**: the swap sees a faint code only where it dominates the needed share (section 5), and at Sb the operands
  are already mixed (disclosure 1). The paper's wording is fixed now: "At the end of the sentence that gives b, before
  c is read, the share swap is <label>; under mixing at the level Ouro's states show there, it sees a running total
  only if its I is at least <the power read's smallest I>, about <ratio> times Ouro's at P1, so this does not show
  that none is held." The P1 ambiguity stays as a limitation. The paper does not say "no running total".

**Part B:**
- **USED:** setting the tail's needed-step component, with the copies projected out, to x''s moves the answer to x''s
  in at least 10% more pairs than a random subspace: the running total (or what that subspace holds) is used.
- **NOT-USED:** not used through the tail's needed-step subspace; the paper states that power for this distance in a
  pretrained model is not shown.
- **UNCLEAR / UNSCORABLE:** reported as such.

Predictions: Part A FOLLOWS 20%, TOO-FAINT 10%, UNCLEAR 40%, NO-SIGNAL 10%, DOES-NOT-FOLLOW 20%. Part B USED 10%,
NOT-USED 35%, UNCLEAR 35%, UNSCORABLE 20%.

## 5. Pre-freeze checks (synthetic only; no Ouro state touched)

- `python v232_sb.py --selftest` (Study 3's items): a world with a op b over pairwise mixing (comp 0.5) FOLLOWS
  (I/N 0.90); a op b alone (comp 1.0) FOLLOWS (1.00); copies only NO-SIGNAL; pairwise mixing alone DOES-NOT-FOLLOW
  (I/N 0.01); the projection leaves an orthonormal subspace orthogonal to both copies (max |cos| 3.5e-16); Part B's
  readout gives USED on a planted 30% effect (+0.298 [+0.273, +0.323]) and NOT-USED on none (-0.004 [-0.011, +0.003]).
- Power on Sb's design (scratch, synthetic): with weak mixing (pair lam 0.1-0.25, or none), a planted a op b code at
  comp 0.08 (I 0.005, N 0.003-0.0045, about Ouro P1's level) is FOLLOWS above the floor; at comp 0.05 (I 0.002,
  N 0.0002-0.0015) it is FOLLOWS by the old rule but below the floor (TOO-FAINT); with strong mixing (lam 1) a code
  needs comp 0.12-0.20 (I 0.007-0.019) for FOLLOWS under pair mixing, and is not FOLLOWS up to comp 0.20 under
  paircode_op or fourier mixing. So the swap detects a faint running total only where it dominates the needed share,
  and a non-FOLLOWS at Sb is weak evidence (a review of the draft raised this; the power read in section 2
  and the fixed wording in section 4 were added in response).
- The end-to-end smoke also runs the power read (fake states; every kind reaches FOLLOWS by comp 0.05-0.30).
- End-to-end smoke (scratch, fake states d 64 with v199's own `patch_plan` run on them): `v232_sb.py cpu` runs,
  reproduces v199's plan (1.9e-07), writes the projected plan (9 dims kept, 0.38-0.72 of N's directions left on the
  fake states) and v199's pairs.

## 6. Run plan

Freeze -> push as `v232-freeze` -> `V232_MODE=cpu python build_kaggle_v232.py`, push astrocyte-v232-cpu
(aurshalevmerin, CPU) -> if the plan check passes, `V232_MODE=patch`, push astrocyte-v232-patch (aurshalevmerin, GPU,
`--accelerator NvidiaTeslaT4`, about 2.6 h per T4 by v199's estimate) -> download to `results_v232/`.

## 7. Hashes (sha256 of the LF-normalised file, first 16 hex, as the kernels print them)

- New: `v232_sb.py` 88c621c04627f52e; `build_kaggle_v232.py` c13d78b75b34695f.
- Unchanged, as on origin/v231-freeze: `v199_patch.py` b9eca8a653dac65f (= v199's frozen hash); `v199_extract.py`
  21ad7985e69f74a0; `v228_rescore.py` 1d1e588493f56ba0; `v224_share.py` 2b0bc8868d13b00e; `v186_analyze.py`
  b408d943603262a6; `v178_analyze.py` 90bbc8a848dc5544; `loop_grad_probe.py` 27b80fa0b999bfa7; `v194_items.json`
  fceceeb4dcbd4076.

## 8. RESULT, Part A (2026-11-03; astrocyte-v232-cpu v1, `results_v232/cpu/`)

- **Null on Sb's design (scored first):** 270 runs, 0 FOLLOWS at 0.30; largest I / N with signal 0.201 (fourier)
  -> **TAU_SB 0.35**. No null run passes TAU_SB with N >= 0.002.
- **Sb (primary): UNCLEAR.** N 0.002, I +0.000 [+0.000, +0.001], I / N 0.24, D+ -0.001, D- +0.002. The shares favour
  the difference on both kinds of problem (plus: a+b 0.001, a-b 0.002; minus: a+b 0.001, a-b 0.003). Sb-1
  (secondary): UNCLEAR (N 0.000). Loops 1-2 (descriptive): DOES-NOT-FOLLOW at both (Sb N 0.005, I/N 0.01).
- **Power read at Ouro's mixing** (null lam with N nearest Ouro's 0.0018): a planted code is FOLLOWS from comp 0.05
  in four kinds (I +0.0018 to +0.0024) and from 0.08 under pairwise mixing (I +0.0045). Ouro's I at Sb is at most
  +0.001 (CI upper), against about 0.004 at P1. So at this position's mixing the swap would have seen a running total
  with I of about 0.002-0.0045 (half to just over Ouro's P1 signal), and it sees none; one fainter than that is not
  excluded.
- **Part B plan:** v199's frozen subspaces reproduced (max 1.71e-07); after projecting off a's and b's copies, a
  median 0.63 of the needed step's directions is left (0.29-0.73), 9 dims kept. GPU run launched
  (astrocyte-v232-patch v1).

## 9. RESULT, Part B (2026-11-03; astrocyte-v232-patch v1, `results_v232/patch/`)

- 2,000 pairs, 72 families; unpatched, Ouro writes its own digit in every pair. Method control: clamping b's own token
  moves the answer to the one b' implies in 0.912 of pairs (scorable).
- **NOT-USED:** hit(N, copies projected) - hit(rand) = -0.001 [-0.002, +0.000]. Hits: N 0.000, F 0.000, rand 0.001;
  v199's frozen statistic the same (-0.001 [-0.002, +0.000]).
- **Descriptive, and it limits the reading:** replacing the **whole state** at every tail token (full) moves the answer
  in 0.1% of pairs (own digit kept in 99.5-99.8%), and at the period alone (fullsb) also 0.1%. Nothing at the tail
  tokens, in any subspace, reaches the answer, while b's own token does (91%). So the answer reads b from b's token
  (or later re-reads), not from the sentence's tail. NOT-USED therefore says that the tail is not on the answer's
  path; it does not say whether a running total is formed elsewhere (for example at b's token itself, where a clamp of
  the needed step cannot be separated from b's copy).
