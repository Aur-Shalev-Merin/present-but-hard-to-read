# v188 pre-registration: confirmation of v186 on new templates

*Drafted 2026-10-12. **Status: FROZEN 2026-10-12** (this commit). No v188 state exists. Frozen files (sha256, first 16
hex chars, LF):
- `v188_extract.py` c6ee04639d8f7b35;
- `v188_analyze.py` 8191c92f65f685bd;
- `v186_analyze.py` b408d943603262a6;
- `v178_analyze.py` 90bbc8a848dc5544;
- `loop_grad_probe.py` 27b80fa0b999bfa7. It differs from v186's edc98c31 only in Huginn and gradient-probe options added
  by another thread (LOOP_R, a wikitext fallback, chunked float64 metrics). The Ouro loading and adapter code is
  unchanged.
- `v188_items.py` 48ac8c5b7eb78426;
- `v188_items.json` d13364c7c216d260.

The selftest runs on the mod-5 file, with labels checked against the final bars in section 4. There is no pilot: the templates follow v186's form, and v186
passed every gate with margin (accuracy 0.66, copies ≥ 0.94, bare positive control 1.00).*

## 1. Why

v186 (`docs/prereg_v186_ouro_unwritten_intermediate.md`) was UNCLEAR, with two patterns worth confirming on fresh
material.
- **Bare addition (secondary in v186).** Before Ouro writes anything, the last digit of a + b reads at 0.708 against
  a copy ceiling of 0.100. No unneeded-combination control was run, so generic mixing of a and b is not excluded.
- **Word problems at P1** (the answer token just before the units digit). The unwritten (a + b) mod 10 has excess
  +0.205 over its copy ceiling. It beats the XOR-like controls by +0.118 [0.094, 0.141], below v186's 0.15 bar.
  - The controls did not include the other combinations of the answer digit and c. Since m = (ans + c) mod 10, a
    state that holds the answer being written plus a copy of c could make m readable without m ever being an
    intermediate.

## 2. Design (inference only; frozen Ouro-1.4B; same pipeline as v186)

**Items** (`v188_items.py` → `v188_items.json`, sha256 d13364c7c216d260; no model involved).
- **12 NEW templates.** Same a + b − c form; new wording and objects; names from index 4 onward, unused in v186.
- **main:** 36 families × 200 of the 1,000 unit-digit combinations = 7,200 items. Cramér's V between m and the
  answer digit = 0.032.
- **bare:** 16 families × all 100 unit-digit pairs = 1,600 items, twice v186's 800.

**Extraction.** `v188_extract.py` is identical to the frozen v186_extract.py except for file names (diff checked).
- Positions: P0, the last prompt token; P1, the answer token before the units digit.
- States: mid and end of each of the 4 loops.

**Labels** (units digits).
- a, b, c (copies).
- m = (a+b) mod 10.
- ans.
- Controls: amb = (a−b) mod 10; apc = (a+c) mod 10; rnd, a random function of (a, b); and four new controls:
  - abmc = (a+b−2c) mod 10, which equals (ans−c) mod 10;
  - abc = (a+b+c) mod 10, which equals (m+c) mod 10;
  - ap2b = (a+2b) mod 10;
  - tapb = (2a+b) mod 10.

**Probe and ceiling.** As in v186: family-demeaned states; v178 `probe_predict`; folds grouped by template (word) or
family (bare). The copy ceiling is the same probe on one-hot (a_u, b_u, c_u).
- excess(L) = acc(L, state) − acc(L, ceiling), per item.
- Paired cluster bootstrap over families: 10,000 resamples, seed 188.
- Bonferroni over 2 hypotheses: 97.5% CIs.

## 2b. Change before the freeze: primary labels are mod 5

**The problem: odd/even leaks.** Mod 10, every combination of a, b and c shares its odd/even half with a + b or with
the answer digit:
- a − b, a + b − 2c and a + b + 2c all share it with a + b;
- a + b + c = ans + 2c shares it with ans.

So when the state holds m, or the answer Ouro is writing, controls get free accuracy.
- The first selftest showed this: in the pure-copy world, a + b + c read +0.14 above its ceiling at P1.
- Under the hypothesis, a − b would also inherit m's parity, which may explain part of v186's +0.087 for amb.

**The fix.** Five is prime, so mod-5 combinations not proportional to (a + b) or (a + b − c) are clean.
- **Primary target:** m5 = (a + b) mod 5. Chance is 0.2.
- **H1 controls:** (a − b), (a + 2b), (2a + b) mod 5.
- **H2 controls:** (a − b), (a + c), (a + b − 2c) = (ans − c), (a + b + c) = (ans + 2c), all mod 5.
- **rnd5:** a random function of (a_u, b_u) into 0–4; it must only be beaten.

The bars are unchanged, set from v186's mod-10 estimates. The mod-10 versions are reported as secondary, for
continuity with v186.

## 2c. Bar calibration on the mod-5 scale (`v188_calibrate.py`; `results_v188_calib/v188_calibrate_v186.txt`)

v186's bars were on the mod-10 scale. As a review noted, they cannot be carried to mod 5. Following the earlier rule
(bars at about half of v186's estimates), the finished v186 states were re-scored with the mod-5 labels and v188's
controls. This is an exploratory re-analysis of a completed study on the old templates.

**H1 (v186 bare, P0).**
- Excess: m5 +0.434. The controls read as well or better: amb5 +0.545, ap2b5 +0.453, tapb5 +0.449.
- D_XOR = **−0.111** [−0.203, −0.034]. v186's bare "sum ahead of writing" signal is not specific to the sum. It looks
  like generic mixing of a and b.
- The a-priori 0.15 / 0.15 bars are kept. Nothing supports lowering them.

**H2 (v186 word problems, P1).**
- Excess: m5 +0.176. Controls: amb5 +0.089, apc5 +0.076, abmc5 (ans − c) +0.108, abc5 (ans + 2c) +0.128.
- D_XOR = **+0.048** [+0.032, +0.063].
- Bars at about half: D_ceil ≥ 0.09 and D_XOR ≥ 0.025.

**Selftest against the final bars.** The selftest ran with the earlier H2 bars (0.10 / 0.05), so each world was
re-labelled from its printed statistics:
- copy: the D_ceil CI upper is +0.025 < 0.09 → NOT-REPLICATED ✓;
- ans_c_mixing: the D_XOR CI upper is −0.011 < 0.025 → NOT-REPLICATED ✓;
- bare_mixing: H2 D_ceil CI upper +0.025 → NOT-REPLICATED ✓;
- compute and partial_0.4: REPLICATED ✓;
- broken: UNSCORABLE ✓.

All required labels are unchanged.

## 3. Hypotheses and labels (`v188_analyze.py`; end of loop 4)

**H1, bare addition at P0.** Before writing anything, Ouro holds the sum mod 5.
- Statistics (all mod 5): D_ceil = excess(m5); D_XOR = excess(m5) − max(excess of the three controls); D_rnd = excess(m5) − excess(rnd5).
- **REPLICATED:** D_ceil ≥ 0.15 and D_XOR ≥ 0.15 (both CI lower > 0), and D_rnd CI lower > 0.
- **NOT-REPLICATED:** D_ceil or D_XOR has its CI upper bound below 0.15.
- Otherwise UNCLEAR.

**H2, word problems at P1.** The unwritten (a+b) mod 5 is present just before the answer's units digit.
- Statistics (all mod 5): D_ceil = excess(m5); D_XOR = excess(m5) − max(excess of the four controls); D_rnd as above.
- **REPLICATED:** D_ceil ≥ 0.09 and D_XOR ≥ 0.025 (both CI lower > 0), and D_rnd CI lower > 0.
- **NOT-REPLICATED:** D_ceil CI upper < 0.09, or D_XOR CI upper < 0.025.
- Otherwise UNCLEAR.
- The bars are set from v186's estimates (excess +0.205, D_XOR +0.118), at about half. abmc and abc are new, so a
  state that merely mixes the answer digit with c cannot pass.
- **The 0.05 margin was chosen after v186 narrowly missed its 0.15 margin.** The templates are new, so the test is
  fair, but a REPLICATED here means "a smaller edge than v186 demanded holds on new problems", not "the original bar
  was cleared".
  - The analysis also reports, as a secondary line, whether H2 clears v186's original 0.15 / 0.15 bars.
  - Any write-up states both bars.

**Gates.** If any gate fails, both labels are UNSCORABLE.
- G1: word-problem accuracy after loop 4 ≥ 0.30.
- G2: copies a, b and c each ≥ 0.75 at P0 in their best state.
- G3: on the bare set, the sum's digit at P1, end4 ≥ 0.90.

**Secondary** (reported, not labels):
- H1's and H2's excess by state: does each appear after loop 1?
- Word-problem P0 excess; it was −0.016 in v186.
- Bare accuracy by loop.

## 4. Selftest (synthetic states on the v188 item structure)

| world | content | H1 required | H2 required |
|---|---|---|---|
| compute | m code at bare P0 and word P1 | REPLICATED | REPLICATED |
| partial_0.4 | m on 40% of items | REPLICATED | REPLICATED |
| partial_0.15 | m on 15% of items | (recorded) | (recorded) |
| copy | copies (+ answer code at P1) | NOT-REPLICATED | NOT-REPLICATED |
| ans_c_mixing | word P1: generic interaction of the answer digit and c | NOT-REPLICATED | NOT-REPLICATED or UNCLEAR |
| bare_mixing | bare P0: generic interaction of a and b | NOT-REPLICATED or UNCLEAR | NOT-REPLICATED |
| broken | no copies | UNSCORABLE | UNSCORABLE |

Final selftest (mod 5; Kaggle CPU; `results_v188_selftest/v188_selftest_mod5.txt`). Every required world gave its
required labels:
- compute and partial_0.4: H1 and H2 REPLICATED;
- copy, ans_c_mixing and bare_mixing: NOT-REPLICATED where required;
- broken: UNSCORABLE.

Sensitivity: in partial_0.15, H1 was UNCLEAR (excess +0.138) and H2 REPLICATED (excess +0.120, D_XOR +0.086).

## 5. Predictions (made after the v186 re-scoring in 2c; before any v188 state)

**Gates:** all pass (P ≈ 0.85).

| outcome | H1 (bare, P0) | H2 (word, P1) |
|---|---|---|
| REPLICATED | 10% | 35% |
| NOT-REPLICATED | 70% | 35% |
| UNCLEAR | 20% | 30% |

- **H1:** v186's bare signal was generic mixing (D_XOR −0.11).
- **H2:** v186's mod-5 margin over the answer-and-c controls was only +0.048, close to its 0.025 bar.
- Secondary: in mod 10, both look stronger than in mod 5, because of the parity confound.

## 6. What each outcome means for the note

- **H1 REPLICATED:** in bare addition Ouro holds the last digit of a sum before writing any of it, and it is a
  specific combination, not generic mixing.
- **H2 REPLICATED:** in word problems the unwritten intermediate is present at the moment it is needed. It is not
  explained by the answer digit mixed with c.
- **H2 NOT-REPLICATED:** v186's P1 signal is either specific to its templates or an answer-and-c mixture.

## 7. RESULT (2026-10-12; `results_v188/cpu/v188_analysis.txt`)

**Run.** Extraction `aurshalevmerin/astrocyte-v188-ouro` (T4×2; 4,400 items per GPU, tokenisation ok 8,800/8,800).
Analysis `aurshalevmerin/astrocyte-v188-cpu` (CPU, kernel_sources). Frozen hashes re-checked after the run: all seven
match the list at the top.

**Gates: all pass.**
- G1: word accuracy by loop 0.049 / 0.378 / 0.583 / **0.599** (≥ 0.30). Bare accuracy 0.99 / 1.0 / 1.0 / 1.0.
- G2: copies at P0, best state: a 1.000, b 0.903, c 0.998 (each ≥ 0.75).
- G3: bare sum digit at P1, end4: 1.000 (≥ 0.90).

**H1 (bare, P0): NOT-REPLICATED** (predicted 70%).
- Excess: m5 +0.372; controls amb5 +0.463, ap2b5 +0.349, tapb5 +0.326; rnd5 +0.258.
- D_ceil +0.372 [+0.263, +0.501]; **D_XOR vs amb5 −0.090 [−0.147, −0.039]**; D_rnd +0.115 [+0.064, +0.166].
- Same pattern as the v186 calibration (D_XOR −0.111). Before writing, Ouro's last-token state mixes a and b in a
  way that makes many combinations readable, a − b best of all. The sum is not singled out.

**H2 (word problems, P1): REPLICATED** at the frozen bars (predicted 35%).
- Excess: m5 +0.219; controls amb5 +0.089, apc5 +0.093, abmc5 (ans − c) +0.130, abc5 (ans + 2c) +0.144; rnd5 −0.096.
- D_ceil **+0.219 [+0.194, +0.244]** (bar 0.09); D_XOR vs abc5 **+0.075 [+0.052, +0.098]** (bar 0.025); D_rnd +0.314
  [+0.286, +0.342].
- **At v186's original 0.15 / 0.15 bars (secondary, not a label): NOT-REPLICATED.** D_XOR +0.075 has its upper
  bound 0.098 below 0.15.
- So: on new templates, just before writing the answer's units digit, (a + b) mod 5 reads 0.22 above what the
  copies of a, b and c give, and 0.075 above the best answer-and-c mixture. The edge over the controls is real but
  small; most of m's readability is shared with other combinations.

**Secondary.**
- H1 excess by state is flat from loop 1 (mid1 +0.327 … end4 +0.372): bare mixing is there after one loop.
- H2 excess by state: mid1 +0.026, end1 +0.158, mid2 +0.128, end2 +0.383, mid3 +0.216, end3 **+0.448**, mid4 +0.192,
  end4 +0.219. It is near zero after half a loop, larger at loop ends than mid-loop, and peaks at end3, not end4.
  This was not predicted. The states were not checked against controls by state, so it is descriptive only.
- Word P0, end4: m5 excess −0.007 (v186: −0.016). Before the answer starts, the intermediate is not readable.
- Mod-10 versions (v186-style, not labels): H1 D_XOR −0.161 → NOT-REPLICATED; H2 D_ceil +0.238, D_XOR vs abmc +0.052 →
  REPLICATED at the 0.09 / 0.025 bars. Mod-10 excesses are larger, as predicted from the parity confound.

**Prediction scorecard.** Gates pass (P 0.85). H1 NOT-REPLICATED (P 0.70). H2 REPLICATED (P 0.35, tied for most
likely with NOT-REPLICATED).

**Post-hoc check: the shown answer digit (2026-10-13; exploratory; does not change the label).**
`v188_shown_digit_check.py`, CPU kernel `astrocyte-v188-shown`, output `results_v188/shown/v188_shown_digit_check.txt`.
- **Concern (review).** At P1 the gold answer's tens digit has been fed in, and P1 is that token. Within a family
  it equals base + carry − borrow (carry: a_u + b_u ≥ 10; borrow: (a + b)_u < c_u), so it carries information about
  a + b. The frozen copy ceiling held only (a_u, b_u, c_u).
- **Check.** H2 recomputed with two extra ceilings: *shown* = copies + one-hot of the shown tens digit (all that is
  visible as copied tokens), and *cb* = copies + carry, borrow and their product (stricter: more than is visible).
  Same probe, folds and bootstrap; v186 states too (mod 5).

| | ceiling | m5 excess | D_XOR (vs abc5) | at 0.09 / 0.025 |
|---|---|---|---|---|
| v188 end4 | frozen | +0.219 | +0.075 [+0.052, +0.098] | passes |
| v188 end4 | + shown digit | +0.223 | +0.073 [+0.054, +0.093] | passes |
| v188 end4 | + carry, borrow | +0.193 | +0.048 [+0.023, +0.074] | passes |
| v186 end4 | frozen | +0.176 | +0.048 [+0.030, +0.066] | passes |
| v186 end4 | + shown digit | +0.182 | +0.052 [+0.032, +0.071] | passes |
| v186 end4 | + carry, borrow | +0.155 | +0.027 [+0.007, +0.046] | passes (barely) |

- The shown digit adds nothing to the ceiling (m5 ceiling 0.225 → 0.222). It is a single carry − borrow value per
  item, and a linear probe on separate codes cannot combine it with the copies.
- Giving the ceiling carry and borrow directly, which Ouro does not see as tokens, removes about a third of the
  margin. The margin stays above zero in both studies.
- **End of loop 3, now with all controls** (post hoc; end3 was not a pre-registered state): v188 m5 excess +0.448,
  D_XOR vs apc5 +0.142 [+0.080, +0.206]; with carry and borrow +0.113 [+0.050, +0.178]. v186: +0.424, D_XOR +0.125
  [+0.072, +0.180]. The margin over controls is larger after loop 3 than after loop 4 in both studies.

**Erratum (2026-10-13).** Section 3 says "The 0.05 margin was chosen after v186 narrowly missed its 0.15 margin."
The frozen bar, set in section 2c and used by `v188_analyze.py`, is **0.025**; "0.05" is a leftover from an earlier
draft of the bar (0.10 / 0.05). The frozen text is left as it was.

**Referee checks (2026-10-13; post hoc; reading rules fixed here before running).** An outside review raised five
points that the saved states can answer. Scripts: `v188_referee_checks.py` (v188 and v186 states, CPU kernel
`astrocyte-v188-referee`) and `v181_pca_check.py` (v181 states, CPU kernel on the second account).
1. **All 31 mod-5 combinations.** Mod 5 there are 31 distinct linear forms x·a + y·b + z·c (up to scale). Score each
   at P1 (end4 and end3) as excess over its own copy ceiling. The answer form (a + b − c) is Ouro's next-token
   prediction and readable by construction, so m5 is ranked among the other 30.
   - **Survives:** at end4, m5 ranks first among the 30 in both studies, and in v188 its margin over the runner-up
     has a template-level 97.5% CI with lower bound > 0.
   - **Weakened:** m5 ranks first in v188 but not in v186, or ranks first in both with the CI including 0.
   - **Fails:** m5 does not rank first in v188.
2. **Template-level bootstrap.** Families within a template are correlated, so all margins are also given with
   resampling over the 12 templates (97.5%).
3. **Correct-only.** The margin over the runner-up and over the four frozen controls, on items Ouro answers
   correctly after loop 4 (probe trained on all items; scored on the subset). Reported, not labelled.
4. **Carry-and-borrow ceiling** for every form, so the ranking is also given against the stricter baseline.
5. **Standard probe without the PCA cut** (v181, last prompt token). The frozen standard probe keeps 256 principal
   components of raw states, which may discard the small within-family directions. Rerun the standard probe (a) on
   all dimensions with a wider C grid (1e-4 to 10), (b) with PCA fitted on family-demeaned training states but
   applied to raw states, and (c) as the minimal-pair probe applied to raw (not demeaned) held-out states.
   - If (a) or (b) reads ≥ 0.50 at any loop end, "a standard probe misses it" is an artifact of the PCA step, and
     the note says so. If both stay ≤ 0.30, the claim holds for linear probes on the full state. Otherwise partial.

**Referee checks: results (2026-10-13).** Outputs `results_v188/referee/v188_referee_checks.txt`,
`results_v181/pcacheck/v181_pca_check.txt`.

*All 31 forms (P1, end4, frozen ceiling): **WEAKENED** by the rule above.*
- v188: m5 ranks **1st** of the 30 non-answer forms (+0.219). The runner-up is **a − c** (+0.158); the margin is
  +0.061, template CI [+0.023, +0.098], so the rule's v188 condition holds.
- v186: m5 ranks **2nd** (+0.176), behind **a − c** (+0.198); margin −0.022 [−0.072, +0.026].
- After loop 3 (end3), m5 ranks 1st in both studies, a − c 2nd (margins +0.099 [+0.014, +0.195] and +0.061
  [−0.035, +0.174]).
- **Not foreseen in the rule:** a − c is itself a valid partial result of a + b − c, not an unneeded combination. In all
  four cells (two studies × end3/end4), the top two non-answer forms are a + b and a − c. The third partial result,
  b − c, ranks 5th–7th (b's copy is the weakest of the three). Read post hoc: partial results of the computation are
  readable above every other combination of the same digits; which one leads differs between studies.
- With the carry-and-borrow ceiling the ranks are unchanged (v188 1st, v186 2nd).

*Template-level bootstrap (97.5%), m5 − best of the four frozen controls:* v188 +0.075 [+0.055, +0.095] (family:
[+0.052, +0.098]); with carry and borrow +0.048 [+0.019, +0.077]. v186 +0.048 [+0.030, +0.063]; with carry and borrow
+0.027 [+0.011, +0.044]. Resampling templates does not widen these intervals.

*Correct-only (items Ouro answers correctly after loop 4):* v188 margin over the frozen controls +0.064 [+0.042,
+0.087] (n = 4,316), wrong-only +0.091; v186 +0.051 [+0.020, +0.075]. Against a − c, v188 correct-only +0.048
[−0.001, +0.100].

*Standard probe without the PCA step (v181, last prompt token): **HOLDS**.* Standard probe, frozen 0.15–0.23; all
2,048 dimensions with C from 1e-4 to 10, 0.16–0.24; PCA fitted on demeaned states, 0.16–0.22. The minimal-pair probe
applied to raw held-out states reads 0.11–0.15, against 0.92–0.97 on demeaned states: the digit's direction is there,
but the context's variation swamps it in raw states.

**Within-line check (2026-10-13; post hoc; reading rule fixed here before running).** Suggested by a second
referee. Script `v188_line_check.py` (CPU kernels `astrocyte-v188-line188`, `astrocyte-v188-line186`); per-item
results saved to `v188_line_items_<study>.npz`.
- **Structure.** Mod 5, every form other than the answer lies on at most one line {answer + k·x, k = 1..4}, for a
  copy x ∈ {a, b, c}. Each partial result is one member of a line: a + b = answer + c, a − c = answer − b,
  b − c = answer − a. A generic interaction of the answer digit with one copied number predicts that the four forms on
  a line read about equally well.
- **Statistic.** Per item, excess(partial result) − excess(best other form on its line, chosen by mean), with a
  family-level 97.5% CI (template-level also reported).
- **Already known.** The second referee computed the point estimates from the printed means before this rule was
  written (end4, frozen ceiling, v188 / v186): c-line +0.075 / +0.048, b-line +0.034 / +0.070, a-line +0.022 / +0.028.
  Only the intervals are new.
- **Reading** (end4, frozen ceiling; 6 cells = 3 lines × 2 studies): **SPECIFIC** if at least 5 cells have CI lower
  bound > 0; **MIXED** if 3–4; **GENERIC** if at most 2.
- **Also reported, not labelled:** the same at end3 and with the carry-and-borrow ceiling; the on-line elevation (mean
  excess of the 9 non-partial forms on the answer lines minus the mean of the 15 off-line forms other than the copies),
  with CI; per-template sign counts of the main margins (24 templates).
- **Limit fixed now.** SPECIFIC rules out a generic answer × copy interaction. It cannot separate "a + b was formed
  first" from "the answer was formed and c added back", because both are the same function of the digits; only an
  ordering or causal test can.

**Within-line check: results (2026-10-13).** Per-item results `results_v188/line/v188_line_items_{188,186}.npz`;
statistics `results_v188/line/v188_line_stats.txt` (recomputed locally from the npz with `--stats`; the kernels'
printed copies match, but one text file downloaded as a server-error page). **Reading: SPECIFIC (6 of 6 cells).**

| end4, frozen ceiling | v188 | v186 |
|---|---|---|
| a + b − best other on its line | +0.075 [+0.052, +0.098] (12/12 templates +) | +0.048 [+0.030, +0.066] (11/12) |
| a − c − best other on its line | +0.035 [+0.006, +0.062] (9/12) | +0.070 [+0.042, +0.097] (11/12) |
| b − c − best other on its line | +0.022 [+0.002, +0.041] (8/12) | +0.028 [+0.004, +0.054] (9/12) |

- Family-level 97.5% CIs. Resampling templates, two cells include zero: v188 a − c [−0.010, +0.075] and v186 b − c
  [−0.001, +0.059].
- End3 (frozen ceiling): margins are larger, +0.088 to +0.152, every CI above zero in both studies.
- **On-line elevation** (9 non-partial answer-line forms vs 15 off-line forms): +0.060 [+0.052, +0.070] and +0.068
  [+0.059, +0.078] at end4, but only +0.004 [−0.007, +0.015] and +0.014 [+0.004, +0.023] at end3. This is relative
  only: at end3 both groups read about +0.19 (above the answer-line group's +0.105 at end4). Loop 3 mixes everything
  strongly and loop 4 lowers the off-line combinations most. (A first reading, "answer mixing appears only at loop 4",
  was wrong and was corrected after a side note.)
- With the carry-and-borrow ceiling the end4 cells stay positive (v188: a + b +0.048, a − c +0.041 [+0.009, +0.070],
  b − c +0.031 [+0.009, +0.051]).

**Two-number check (added after a side note; not part of the rule).** On each answer line the partial result is the
only two-number member, so a state that simply holds two-number mixtures better would also pass the line test.
Partial result minus the best of the 9 other two-number forms (`results_v188/line/v188_line_twonum.txt`):
- end4, both studies: a + b +0.125 / +0.087, a − c +0.065 / +0.109, b − c +0.033 / +0.037. Every family CI is
  above zero; b − c's template CI includes zero in v188.
- end3: a + b +0.138 / +0.125 (CIs above zero); a − c +0.039 [−0.026, +0.104] / +0.064 [+0.043, +0.085]; b − c
  −0.041 / −0.033, below b + c or a + c.
- So "simpler, two-number forms read better" does not explain a + b, explains part of a − c's edge at end3, and
  b − c is not distinguished from other two-number forms at end3.

**Normalization check on the mid/end alternation (2026-10-13; post hoc; reading rule fixed here before running).**
Script `v188_rms_check.py` (CPU kernels `astrocyte-v188-rms188`, `astrocyte-v188-rms186`).
- **Concern** (raised by a parallel session). At P1 the excess of a + b is higher at the end of every loop than in
  its middle (v188: mid1 +0.03, end1 +0.16, mid2 +0.13, end2 +0.38, mid3 +0.22, end3 +0.45, mid4 +0.19, end4 +0.22).
  But endT states are after Ouro's RMSNorm, which Ouro feeds into the next loop, while midT states are the raw residual
  stream after layer 12. Per-item scale differences in raw states could make mids harder to probe.
- **Check.** Re-probe the four mid states after dividing each item's state by its own root-mean-square (the per-item
  part of RMSNorm). The norm's learned per-dimension gain is irrelevant, because the probe standardizes each dimension.
  End states are unchanged.
  - Labels (P1, word problems, mod 5, frozen copy ceiling): a + b, a − c, b − c, the answer digit, answer + 2c, a + c,
    and the copy of b.
  - Raw mids are recomputed in the same run.
- **Statistic.** For a + b, the within-loop alternation D = mean over loops t = 1..4 of [acc(endT) − acc(midT)], per
  item (the ceiling cancels), with a family-level 97.5% CI. Computed with raw mids (D_raw) and normalized mids (D_rms).
- **Reading** (a + b, both studies):
  - **ARTIFACT** if D_rms < 0.5 · D_raw in both studies, or D_rms's CI includes 0 in both;
  - **REAL** if D_rms ≥ 0.75 · D_raw with CI lower bound > 0 in both;
  - otherwise **PARTIAL**.
  - The other labels' D values are reported to show whether any alternation is specific to a + b.

**Normalization check: results (2026-10-13).** `results_v188/rms/v188_rms_items_{188,186}.npz`, statistics
`results_v188/rms/v188_rms_stats.txt`. The raw by-state excesses reproduce the frozen analysis exactly.
**Reading: REAL.** a + b alternation, raw → normalized mids: v188 +0.162 → +0.161 [+0.147, +0.175]; v186 +0.170 →
+0.169 [+0.155, +0.182] (ratios 1.00 and 0.99). Per-item RMS normalization changes nothing.
- **Not specific to a + b.** Alternation D (end − mid, mean over loops), v188 / v186:
  - a − c +0.181 / +0.172;
  - a + c (unneeded) +0.186 / +0.193;
  - answer + 2c +0.123 / +0.121;
  - answer digit +0.144 / +0.161;
  - b − c +0.075 / +0.074;
  - copy of b **−0.070 / −0.067**.
  
  Every combination reads better at the end of a loop than in its middle; the copy reads better in the middle. In each
  loop, layers 13–24 make combinations of the numbers readable. This is a property of the loop's two halves, not
  evidence about a + b.
- **Timing (descriptive; bears on the "answer formed, c added back" rival).** After loop 1, a + b and the answer
  digit read about the same (v188 +0.158 / +0.155; v186 +0.126 / +0.154). After loop 2 the answer reads higher
  (+0.528 vs +0.383; +0.588 vs +0.392). No sign that a + b forms before the answer.
- **Unneeded combinations fade.** a + c peaks at end2 (+0.46 / +0.48) and falls to +0.09 / +0.08 by end4. a + b
  peaks at end3 and keeps +0.22 / +0.18 at end4.

**What this means for the note** (section 6). In word problems, the unwritten a + b is present at the moment it is
needed, beyond what the answer digit mixed with c explains. The margin is small (0.075) and passes only the
lowered bar chosen after v186's near-miss; any write-up states both bars. In bare addition, the "sum before writing"
reading is withdrawn: it is generic mixing of a and b.
