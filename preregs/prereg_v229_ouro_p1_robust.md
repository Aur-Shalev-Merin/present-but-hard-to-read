# v229 pre-registration: Ouro Study 3 P1 under the share swap, carry / borrow and a design-matched null

Workstream paper3 (Ouro note). Claimed 2026-10-28. Follows v228. Frozen 2026-10-28, before any run.

## 1. Why

v228 re-scored Ouro Study 3 with the validated share swap: P1 FOLLOWS (I +0.004 [0.003, 0.005], I / N 0.63, N 0.007),
the paper's one FOLLOWS. A second outside review raised two gaps:

1. **Carry / borrow.** P1 is the answer's tens digit. Within a family the tens digit is fixed by the first-step carry
   (plus) or borrow (minus) and the second-step borrow: checked on all 72 main-set families, 0 exceptions. Both are
   sign-specific. v228 regressed the units digits of a, b, c and the answer, not the carry / borrow, so the token at P1
   itself was not regressed. Under the old accuracy swap, adding the carry / borrow to the ceiling cut P1's I from 0.157
   to 0.126 and removed P0's entirely (v205).
2. **The bar's design.** TAU 0.30 was set from null worlds on Qwen's two-operand designs. Ouro's design has three
   copies, an answer and a carry, and its N (0.007) is far below the controls' (0.11-0.15); v224 found the null ratio
   noisy when N is small.

**Disclosures.** (1) Post hoc: v228's result is known, and this study was designed because of it. (2) The share
statistic with the carry / borrow regressed has not been computed on any Ouro state. (3) Ouro is not a validation
model of the share swap.

## 2. Part A: P1 with the carry / borrow regressed (primary) (`v229_robust.py`, frozen)

`v228_rescore.analyse` unchanged (TAU 0.30, labels as v224 section 2, cluster pair, states mid3 / end3 / mid4 / end4,
10,000 resamples, seed 2241, 97.5% CIs), with one more regressed variable: **cb = c1 + 2 c2** (one-hot, 4 classes),
c1 = first-step carry (a_u + b_u >= 10, plus) or borrow (a_u < b_u, minus), c2 = second-step borrow ((a_u op b_u) mod 10
< c_u). Within a family its one-hot spans c1, c2 and c1 c2, so it covers the tens digit. Copies: a, b, c, ans, cb.
Descriptive: P0; and P1 with v228's copies (must reproduce v228: I +0.004, I / N 0.63).

## 3. Part B: design-matched null

v224's five null kinds (pair, fourier, mlp, paircode, paircode_op) built on Study 3's own main-set items (14,400):
copies U1[a] + U2[b] + U3[c], an answer code and a carry code (per operation and cb), each at the copies' RMS, and one
generic mixture of (a, b, op) at lam x the copies' RMS; family offsets; N(0, 1) noise per state; d 256. **No world holds
a op b.** lam in {0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16} (v224's grid extended down so some worlds reach Ouro's small N);
seeds 1-6: 270 runs, each scored with Part A's statistic (1,000 resamples).

**TAU3** = max(0.20, 1.5 x the largest I / N among runs with signal (N CI lower > 0), rounded up to 0.05): v224's rule.

No answer x c mixture is included: on units digits (a op b) = answer + c, so such a mixture holds the candidate. That
alternative is the one the paper already says no probe label can separate, and this study does not address it.

## 4. Verdict and readings

- **STANDS:** Part A P1 FOLLOWS and I >= TAU3 x N. The paper keeps "its sign follows the operation", now beyond the
  copies, the answer, the carry / borrow and a bar set on Ouro's own design.
- **NOT-BEYOND-DESIGN-MIXING:** Part A P1 FOLLOWS (TAU 0.30) but I < TAU3 x N. The paper says Ouro's sign-following is
  not shown beyond generic mixing at its signal level; the abstract and intro drop the claim.
- **NOT-SHOWN:** Part A P1 not FOLLOWS. The carry / borrow accounts for v228's FOLLOWS; the paper drops "its sign
  follows the operation" for Ouro (it reports Part A's label) and keeps "readable beyond copies, not used".

Predictions: STANDS 35%, NOT-BEYOND-DESIGN-MIXING 20%, NOT-SHOWN 45%.

## 5. Pre-freeze checks

- Selftest (`python v229_robust.py --selftest`, Study 3's items): a world with copies, the answer and a carry / borrow
  code but no a op b is **FOLLOWS (I / N 1.00) with v228's copies** and NO-SIGNAL with cb regressed; worlds that also
  hold a units-digit code of a op b are FOLLOWS with cb regressed (I / N 0.89 at comp 0.5, 0.97 at comp 1.0).
- Mechanics: one world takes about 15 s on one core; Part B runs on 4 processes.
- Items: v194 meta main set (14,400; asserted); cb's tens-digit coverage checked on v194_items.json.
- No share statistic with cb has been computed on any Ouro state.

## 6. Frozen files (sha256, first 16 hex, LF)

- `v229_robust.py` 0874031fe1b6d669
- `build_kaggle_v229.py` cf1932aed74a45cf
- reused unchanged: `v228_rescore.py` 1d1e588493f56ba0, `v224_share.py` 2b0bc8868d13b00e (TAU 0.30),
  `v178_analyze.py` 90bbc8a848dc5544

Run: `astrocyte-v229-robust` (CPU, aurshalevmerin; source astrocyte-v194-ouro).

## 7. RESULT (2026-10-28; frozen code, `aurshalevmerin/astrocyte-v229-robust`)

**Part A (primary), P1 with copies a, b, c, the answer and the carry / borrow regressed: FOLLOWS.** N 0.006
[0.005, 0.007], D+ +0.001 [+0.00001, +0.002], D- +0.006 [+0.005, +0.007], I +0.003 [+0.003, +0.004], I / N 0.57.
Shares: plus S 0.005 vs M 0.004; minus M 0.007 vs S 0.001; random functions 0.001 on average. P0 UNCLEAR (N 0.0005,
I +0.000 [-0.000, +0.000]). With v228's copies the code reproduces v228 exactly (I +0.004 [0.003, 0.005], I / N 0.63).
The carry / borrow takes about a fifth of I (0.0043 -> 0.0034) and most of it from D+, whose interval now barely clears
zero.

**Part B, design-matched null:** FOLLOWS in 0 of 270 runs. Largest I / N among runs with signal 0.149 (paircode_op,
lam 0.1, N 0.0005); 81 runs with signal had N below 0.02, i.e. in Ouro's range, none above 0.149. **TAU3 = 0.25**,
below the frozen 0.30.

**Verdict: STANDS** (P1 FOLLOWS, I / N 0.57 >= TAU3 0.25). Prediction (STANDS 35%) right; the more likely NOT-SHOWN
(45%) did not happen.

**Reading (section 4):** the paper keeps "its sign follows the operation", now beyond the copies, the answer, the
carry / borrow (the token at P1) and a bar set on Ouro's own design at its signal level. It states that the effect is
small (N 0.006) and that on plus problems the sum leads the difference by only 0.001 (interval just above zero); the
answer-plus-c alternative stays unseparated.
