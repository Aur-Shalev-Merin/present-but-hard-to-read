# v224 pre-registration: a share-based operation swap, validated on fresh positive controls

Workstream paper3 (Ouro note). Claimed 2026-10-27. Follows v221 (the accuracy swap failed its positive
control), v222 (the clamp passed at a distance) and v223 (exploratory diagnosis).

## 1. Why

v221: one token before Qwen2.5-7B writes b, the protocol's operation swap (accuracy of S = (d1 + d2) mod 5 against
M = (d1 - d2) mod 5 above their copy ceilings) is UNCLEAR, so the protocol fails its own positive control. v223
(exploratory, on v221's states): there, random functions of the operand pair also read at 0.99-1.00 and their copy
ceilings are 0.80-1.00, so accuracy saturates and the swap has no room. The share of residual variance (after the
copies) that a label's class means explain still follows the operation there (plus: S 0.22 vs M 0.09-0.11; minus:
M 0.25-0.28 vs S 0.06-0.09, layers 24/28).

**Review note (before any of this was frozen):** a statistic shaped on v221's states would pass v221 almost by
construction, and the note's own targets (Kudo, Ouro) are already known. So the statistic is frozen here and judged
only on two positive controls it has never seen, plus fresh null-world seeds. v221's states and the short-chain eq /
last positions (seen in v223) are development data only.

## 2. The statistic (`v224_share.py`, frozen)

Per version, position and layer in {16, 20, 24, 28} (hidden_states; both models have 28 layers): family-demeaned
states; copies (one-hot of the step's two operands u, v, family-demeaned) regressed out by least squares on training
folds (5 folds by family, v178's seed); class means of each label on the training residuals; per held-out item the
squared error to its class mean and its squared residual; share = 1 - sum error / sum residual, averaged over the
four layers. Labels: S = (u + v) mod 5, M = (u - v) mod 5, and R0..R3, S or M of a randomly re-assigned pair (seed
2231; same class sizes). N = (share_S on plus + share_M on minus) / 2; D+ = share_S - share_M on plus; D- = share_M -
share_S on minus; I = (D+ + D-) / 2. Cluster bootstrap over families (both versions together), 10,000 resamples,
seed 2241, 97.5% CIs.

- **UNSCORABLE**: the step is right in < 80% of items in a version (teacher forced).
- **NO-SIGNAL**: N's CI lower <= 0.
- **FOLLOWS**: D+ > 0, D- > 0, I's CI lower > 0 and I >= TAU x N.
- **DOES-NOT-FOLLOW**: I's CI upper < 0.10 x N.
- **UNCLEAR**: otherwise.
- Descriptive: lift = N - mean share of R0..R3 (a pure pair code already gives S and M a lift of about +0.04, so lift
  is not a test), all shares, I / N.

**TAU** = max(0.20, 1.5 x the largest I / N among null-world runs with signal, rounded up to 0.05), computed before
the freeze (section 6): **TAU = 0.30**.

Null worlds (no step value; real family designs, every second family): pair, fourier, mlp (as v215's null, with the
step's operands), paircode (a random vector per operand pair), paircode_op (per operand pair and operation); lam in
{0.25, ..., 16} x the copies' RMS; seeds 1-6 set TAU, fresh seeds 7-12 are run after the freeze.

## 3. Fresh positive controls

1. **Qwen2.5-1.5B, the b step** (v222's items unchanged: 240 families, 12,000 prompts, Simple chain). Operands
   (d1, d2), version = b's operation. Primary position: **inb** (the token before b's digit). Secondary: eqb.
   Descriptive: letb, eq, last. Gate: b correct >= 0.80 per version.
2. **Qwen2.5-7B, the a step** (new items, `v224_items.build_a`: 200 families; each fixes letters, demonstrations and
   b's equation type, and varies (k, B) over 30 pairs with B in 0..4, B <= k <= 9 - B, each in two versions that
   differ only in a's operation). Operands (k, B); B is written in the chain. Primary: **inA** (the token before a's
   digit). Secondary: eqa. Descriptive: digB, eq, last. Gate: a correct >= 0.80 per version.

Neither model-and-step combination has been looked at with any statistic.

## 4. Predictions and readings

Predictions: control 1 inb FOLLOWS 75%; control 2 inA FOLLOWS 80%; no fresh null run FOLLOWS 90%.

- **VALID** (both primaries FOLLOWS, no fresh null run FOLLOWS): the share swap replaces the accuracy swap in the
  protocol. A separate pre-registration (v225) then applies it, unchanged, to the note's targets (Kudo et al.'s
  format and the answer-only states, Ouro's Study 3), stating that those results were known under the old swap and
  that the short-chain eq / last positions were seen in v223.
- **PARTIAL** (one primary FOLLOWS, no null FOLLOWS): reported; the share swap is described as validated in that
  setting only, and the paper keeps the swap demoted. No re-score.
- **FAILS** (neither primary FOLLOWS, or any fresh null run FOLLOWS): the share swap is not used; the paper keeps the
  current wording and reports this attempt.
- UNSCORABLE at a primary counts as not FOLLOWS for the verdict and is reported as such.

## 5. Pre-freeze checks

- Items (`python v224_items.py`): b set = v222 (f917c3b7da3ddc87); a set 12,000, 200 families, A in 0..9.
- Selftest (`python v224_share.py --selftest`): section 6.
- Null grid (`--nullbar`): section 6.
- Development run on v221's states (`astrocyte-v224-dev`): section 6.
- Smoke (`astrocyte-v224-smoke`: 200 items per control; tokens and accuracy only, no states analysed): section 6.

## 6. Pre-freeze record

- Selftest (synthetic, both designs, TAU 0.20 for the test): computed value at c = 0.5 and 1 -> FOLLOWS (I / N 0.58-0.78);
  copies only -> NO-SIGNAL; paircode and paircode_op at lam 2 -> DOES-NOT-FOLLOW (I / N about 0). An earlier gate on
  lift over random pair functions was dropped here because a pure pair code passes it (lift +0.04); R0..R3 were
  changed to shuffled S / M functions with the same class sizes.
- Null grid (5 kinds x 7 lam x seeds 1-6 x both designs, 420 runs; v224_nullbar.json): I / N at most 0.06 in every
  kind except mlp at lam 0.5, where N is near zero (0.0004-0.0008) and the ratio is noise (0.187 and 0.162). By the
  rule, TAU = max(0.20, 1.5 x 0.187 -> 0.30) = **0.30**.
- Development run on v221's states (already seen; `aurmerin/astrocyte-v224-dev`): inb FOLLOWS (I +0.087 [0.079, 0.095],
  I / N 0.62); eqb FOLLOWS (I / N 0.63); letb UNCLEAR (D+ -0.007); eq UNCLEAR (D+ -0.051, D- +0.087: a fixed preference
  for the difference); last UNCLEAR (N 0.014). Not evidence for the statistic; recorded so the fresh results can be
  compared with it.
- Smoke (`aurmerin/astrocyte-v224-smoke`, 200 items each, no states analysed): Qwen2.5-1.5B b set positions as
  expected, b correct 1.000 (first families); Qwen2.5-7B a set positions eq / last / digB / eqa / inA -> '0', '?', B's
  digit, ' =', ' ', a correct 0.98 plus / 0.88 minus.
- No statistic has been computed on either fresh control's states.

## 7. Frozen files (sha256, first 16 hex, LF; 2026-10-27)

- `v224_items.py` 14087b6369b3ce8e (b set = v222 f917c3b7da3ddc87; ITEMS_SHA_A 670ce4ceed4c2ae1)
- `v224_extract.py` c3f5aced78c30939
- `v224_share.py` 2b0bc8868d13b00e (TAU 0.30)
- `build_kaggle_v224.py` 93e4012f544ac21e
- reused unchanged: `v222_items.py` b5813a4fb56ef4f8, `v221_items.py` 2b6bd412a702ccc4, `v215_items.py` 9249f0523d7219eb,
  `v215_pilot.py` d049a0b2d81fb80b, `v178_analyze.py` 90bbc8a848dc5544

Run order (second Kaggle account, aurmerin): `xb` and `xa` (GPU, in parallel), then `main` (CPU).

## 8. RESULT (2026-10-27; frozen code, `aurmerin/astrocyte-v224-xb`, `-xa`, `-main`)

**Verdict: PARTIAL** (control 1 UNSCORABLE, control 2 FOLLOWS, no fresh null run FOLLOWS).

- **Control 1, Qwen2.5-1.5B, b step: UNSCORABLE.** b correct 0.778 (plus) and 0.780 (minus), below the 0.80 gate. The
  smoke's 200 items were the first families only (1.000 there), so it was not a representative accuracy pilot; that
  was a design error in the smoke, recorded here. Descriptive only, no label: inb I +0.049 [0.042, 0.055], D+ +0.032,
  D- +0.066, I / N 0.52 (it would have met the FOLLOWS bar); eqb I / N 0.39 (D+ +0.001); letb, eq, last small.
- **Control 2, Qwen2.5-7B, a step: inA FOLLOWS** (a correct 0.923 / 0.935): N 0.112, D+ +0.061, D- +0.085, I +0.073
  [+0.065, +0.080], I / N 0.65 (bar 0.30). Secondary eqa (A two tokens ahead) FOLLOWS too (I / N 0.61). Descriptive:
  digB UNCLEAR (I / N 0.10, a fixed preference for the difference), eq UNCLEAR and last DOES-NOT-FOLLOW (N 0.003 and
  0.001: almost no signal before the chain).
- **Fresh null seeds 7-12:** FOLLOWS in 0 of 420 runs (5 kinds x 7 lam x 6 seeds x 2 designs).
- Predictions: control 1 FOLLOWS (75%) unscorable; control 2 FOLLOWS (80%) right; no null FOLLOWS (90%) right.

**Reading (section 4):** the share swap is validated in one fresh setting (Qwen2.5-7B, the a step, one and two tokens
before a is written) and rejects every null world; the paper keeps the swap demoted and there is no re-score under
this pre-registration.
