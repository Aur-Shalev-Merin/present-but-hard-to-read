# v235 pre-registration: a pretrained positive control for the clamp (a two-hop bridge)

Workstream paper3 (Ouro note, arXiv v2). Claimed 2026-11-06. Follows v230, v231, v233, v234 and `docs/v235_design.md`.
FROZEN 2026-11-08, before any clamp with a learned subspace.

## 1. Why

The clamp has shown power only in small models trained from scratch (v230, v231, v233). Every reading in a pretrained
model so far (Ouro; Qwen on Kudo et al.'s form) is NOT-USED, UNCLEAR or "cannot tell", and outside reviewers ask
whether the clamp can say "used" in a pretrained model at all. v235 tests it where an unwritten intermediate is known
to be used: two-hop recall without a chain ("The capital of the country where the city of X is located is"). The country
is never written. Pilot 2 established, without any learned subspace, that the country at the city's last token (e1) is
used and that the same value, readable at the prompt's last token, is not used there before layer 20. So v235 can test
both halves of the protocol: does the clamp detect a used value, and does it stay silent where the value is readable but
unused?

**Disclosures.**
1. Pilot 1 (`astrocyte-v235-pilot`): what the model knows about each hop (section 6).
2. Pilot 2 (`astrocyte-v235-pilot2`): whole-state patches from other-country cities, same-country cities, people and
   country names, at e1 and the last token, one layer at a time; a random 24-dim clamp; states for every known city
   (section 6).
3. Probe checks on pilot 2's states (`v235_pilot2_readout.py`): the country's nearest-class-mean readout per layer, the
   copy ceiling (the same readout from the name tokens' embeddings), and the share of the country subspace left after
   projecting off the copy (section 6). These use the class means that the clamp's subspace is built from, but no clamp
   with a learned subspace was run.
4. The band (hidden_states 8, 12, 16), the positions and the negative position were chosen from pilot 2's patches and
   readouts.
5. `astrocyte-v235-smoke`: the frozen code with every learned subspace replaced by a random one (mechanics and the
   whole-state patch only). Numbers in section 6.
6. All results of v215-v234 are known.

## 2. Design (`v235_clamp.py`; one GPU kernel)

- **Model:** Qwen2.5-7B base, fp16, Kaggle T4 x2 (account aurmerin, as pilot 2).
- **Items:** the 1,038 cities that pilot 1 marks as known (country right on the full continuation; two-hop capital right
  on the full continuation and on the first token; duplicate entries kept once), 25 countries (8-60 cities each), and
  the 99 known people (`v235_pilot2_items.json`). Prompt: pilot 1's two-hop template with three demonstrations from
  countries outside the 25. Halves within each country (seed 2352).
- **Positions:** e1 (the city's last token; primary) and last (the prompt's last token; the negative position).
- **States and bases:** for every city, hidden_states 8, 12, 16 at both positions, and name0 (the mean layer-0
  embedding of the city-name tokens: the copy's surface code). Per position, layer and half: U = the country's class
  means (centred on their mean; 24 dims); C = the same from name0; Uo = U projected off C (primary); R = random, Uo's
  dimension (seed 2354). Share of the country left = ||U - C C^T U||^2 / dim(U) (v216's definition). A pair uses the
  bases fitted on the half that does not contain x.
- **Pairs** (seed 2353), one per city x and type: *other* (x' a known city of another country), *same* (another known
  city of x's country; design check), *person* (a known person born in another country, from the person two-hop
  prompt, state at the person's last token; secondary, descriptive).
- **Conditions** at hidden_states 8, 12, 16, at the position: none; cU, cUo, cC, cR: h <- h - V V^T (h - h'); full:
  h <- h' (the whole state at the same layers, the ground truth at the band). Outcome: argmax over the 25 capitals'
  first tokens at the last position. hit = the capital of x''s country; changed = the answer is not x's capital.

## 3. Rules

- **Statistic** (type other, per position): hit(cUo) - hit(cR), cluster bootstrap over x's country (25 clusters),
  10,000 resamples, seed 2355, 97.5% interval. **USED** if >= 0.10 with lower > 0; **NOT-USED** if upper < 0.02;
  **UNCLEAR** otherwise (v215 / v234 rule).
- **Ground truth at the band:** the same label on hit(full) - hit(cR). e1 must be USED and last must be NOT-USED;
  otherwise the verdict is GROUND-TRUTH-FAILED.
- **Design check** (type same, e1): changed(cUo) - changed(cR), same bootstrap; DESIGN-FAILED if >= 0.10 with lower > 0
  (the clamp damages the answer instead of swapping a value).
- **Verdict:**
  - **PASSES:** e1 clamp USED and last clamp NOT-USED.
  - **DETECTS-NOT-SPECIFIC:** e1 clamp USED, last clamp not NOT-USED.
  - **MISSES:** e1 clamp not USED (with the ground truth in place).
  - **GROUND-TRUTH-FAILED** or **DESIGN-FAILED:** as above; no reading.
- Reported alongside, not part of the verdict: cU (no projection), cC (the copy's subspace), the share of the country
  left after the projection, the same-country rates at last, and type person (cUo - cR and full - cR with the same
  bootstrap).

## 4. Readings for the paper (fixed now)

- **PASSES:** in a pretrained model, the clamp finds an unwritten intermediate that whole-state patches show is used,
  and stays silent at a position where the same value reads at 0.98 but is not used. The paper says the protocol can
  return "used" in a pretrained model and separates readable from used there. The signal is strong (about 0.99 of the
  country left after the projection), so this does not show power at Ouro's or Kudo's strength, and the paper says so.
- **DETECTS-NOT-SPECIFIC:** the clamp finds the used value but also fires where the value is readable and unused. The
  paper says a USED from this clamp form needs a whole-state ground truth at the same position. Its NOT-USED readings
  are not weakened by this outcome.
- **MISSES:** the clamp form misses, at full strength, a used value in a pretrained model. The paper must say that its
  NOT-USED readings in pretrained models (Ouro, Qwen on Kudo's form) may be misses, and weaken them.
- **GROUND-TRUTH-FAILED / DESIGN-FAILED:** reported; the paper says the control was not established.

Predictions: PASSES 60%, DETECTS-NOT-SPECIFIC 15%, MISSES 15%, GROUND-TRUTH-FAILED or DESIGN-FAILED 10%.

## 5. Run plan

Freeze -> push as `v235-freeze` -> `V235_ME=aurmerin V235_MODE=clamp` (GPU) -> download to `results_v235/clamp/`.

A faint arm (degrading only the clamp's fitting data, as in v231, down to the share of value the Ouro and Kudo clamps
keep) is not part of this registration. If wanted, it gets its own.

## 6. Pre-freeze checks

- Pilot 1 (`astrocyte-v235-pilot`): country -> capital 25/25; city -> country 0.97 (1,083 cities); two-hop city ->
  capital 0.98 (0.995 among cities whose country the model gives right); first-token argmax agrees with
  full-continuation scoring on 0.995; the capitals' first tokens are distinct. Person -> country 0.93, person two-hop
  0.86 (122 people).
- Pilot 2 (`astrocyte-v235-pilot2`, 200 targets, single-layer whole-state patches): at e1, other-country city hit
  0.82-0.84 (layers 4-12), 0.72 (16), 0.58 (20), 0.00 (24); person 0.67-0.70 (4-14); country name 0.96-1.00;
  same-country city keeps the answer (0.99); all layers 1-27: 0.81 (city), 0.67 (person), 1.00 (country name). At last,
  every source 0.00-0.05 through layer 20 and 1.00 at layer 24. Random 24-dim clamp at e1: kept 1.000, hit 0.000.
- Readout (`v235_pilot2_readout.py`; nearest class mean, half-split): e1 0.32 (layer 0), 0.54 (4), 0.94-0.96 (8-27);
  last 0.03 (0), 0.25 (4), 0.81 (8), 0.92-1.00 (12-27). Copy ceiling (name tokens' embeddings): 0.38 (chance 0.04).
  Share of the e1 country subspace left after projecting off the copy: 0.99 (layers 4-27).
- `astrocyte-v235-smoke` (T4 x2, account aurmerin; 30 pairs per type, all from the first country, so the intervals are
  degenerate; every learned subspace replaced by a random one of the same dimension):
  - v1: random clamps change at most 3% of answers in every condition; the whole-state patch at the band moves the answer
    to x''s capital in 0.90 of pairs at e1 (city and person sources) and 0.00 at last; same-country pairs keep the
    answer. Share of the country left after the copy projection 0.988-0.990 (e1), 0.992-0.993 (last). Bug found: at
    last, some class-mean bases came out with 25 dims (float32 rounding on the last token's outlier dimensions left a
    spurious direction above the tolerance).
  - Fix (before any learned-subspace clamp): class means fitted in float64 and bases capped at 24 dims. v2 (the code
    to be frozen): every basis 24 dims; random clamps change at most 3%; whole-state patch 0.90 at e1 (city, person),
    0.00 at last; shares as v1.
- `v235_clamp.py --selftest` passes (labels, verdict, orthonormal bases, Uo orthogonal to C); a CPU run on a tiny random
  Qwen2 checks the mechanics end to end.

## 7. Hashes (sha256 of the LF-normalised file, first 16 hex)

- `v235_clamp.py` 089a2bdeba402573 (the code smoke v2 ran), `v235_pilot.py` 442760543b204add (prompts),
  `build_kaggle_v235.py` 28334dcefff36883.
- Data: `v235_cities.json` 5299759bc458e12e, `v235_pilot2_items.json` 1fae350dc4df2422 (built by
  `v235_pilot2_items.py` from pilot 1's output).

## 8. RESULT (2026-11-08, `aurmerin/astrocyte-v235-clamp` v1, frozen code; `results_v235/clamp/`)

**Verdict: PASSES.** Labels: e1 whole-state USED, last whole-state NOT-USED (ground truth holds); e1 clamp USED, last
clamp NOT-USED; design check ok at both positions.

| pairs (n = 1,038 each) | position | none | cU | cUo | cC | cR | full | cUo - cR [97.5% CI] | full - cR |
|---|---|---|---|---|---|---|---|---|---|
| other-country city | e1 | 0.000 | 0.793 | 0.789 | 0.000 | 0.000 | 0.849 | +0.789 [+0.732, +0.841] USED | +0.849 [+0.808, +0.886] USED |
| other-country city | last | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] NOT-USED | +0.000 NOT-USED |
| person (secondary) | e1 | 0.000 | 0.528 | 0.524 | 0.000 | 0.000 | 0.760 | +0.524 [+0.468, +0.577] USED | +0.760 [+0.706, +0.807] USED |
| person (secondary) | last | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 NOT-USED | +0.000 NOT-USED |

(hit = the answer is the source's capital.) Same-country pairs: changed(cUo) - changed(cR) +0.000 [-0.003, +0.003] at
e1, +0.001 [+0.000, +0.004] at last. Share of the country left after the copy projection 0.988-0.990 (e1),
0.992-0.993 (last).

- The clamp of the country's 24-dim subspace at e1 carries 0.79 / 0.85 = 93% of the whole-state patch's effect at the
  same layers; the copy's subspace (cC) and a random one do nothing.
- The subspace fitted on cities transfers a person's birth country: 0.52 of pairs (whole state 0.76).
- At the last token, where the country reads at 0.92-1.00 (layers 12-27), neither the clamp nor the whole state at
  layers 8-16 moves the answer.
- Reading (section 4, fixed): the protocol returns "used" in a pretrained model and separates readable from used there.
  The signal is strong (0.99 of the country left after the projection), so this does not show power at Ouro's or Kudo's
  strength.

Correction (2026-11-09, found in an outside review of the paper; no rule or label affected): section 6 gives the
person patch at e1 as 0.67-0.70 (layers 4-14); the pilot 2 log has 0.63 at layer 10, so the range is 0.63-0.70.
