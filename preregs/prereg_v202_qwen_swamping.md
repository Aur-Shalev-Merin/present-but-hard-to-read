# v202 pre-registration: is the copied digit "present but swamped" in a non-looped model? (Qwen2.5-1.5B)

*Drafted 2026-10-15. Status: **FROZEN 2026-10-15** (this commit, pushed to origin before the main run). No probe has
been fitted on real Qwen states; the pilot (section 5) ran the real model on 140 prompts for timing and tokenisation
only. Frozen files (sha256, first 16 hex chars, LF):
- `v202_extract.py` 1c68dc20dde2d0d8 (unchanged since the pilot); `v202_analyze.py` 46d9d8aca3bf0459 (the selftest v2 ran
  this exact file); `build_kaggle_v202.py` c6f1ae75facfb379;
- `v178_analyze.py` 90bbc8a848dc5544; `v181_swamp_check.py` 0a239557c1360a98; `v178_labels.json` 11f0d3852b766479.*

## 1. Why

The Ouro note's Sections 3.2–3.3 show that, at the last prompt token, a digit copied from the question is present
(a family-demeaned "minimal-pair" probe reads it at 0.92–0.97 on held-out templates) but swamped (a standard probe reads
it at 0.15–0.23, the best single-prompt linear decoder at 0.17–0.21, and more templates barely help). The note says
nothing in Sections 3 and 4 is specific to looped models, but only Ouro was tested. An outside review (2026-10-15) asked
for one or two standard models of similar size. The swamping-mechanism check (2026-10-15) found the crowding partly
generic (a colour word's subspace is crowded too), which makes a standard model the natural next test.

- **Question:** in a standard (non-looped) transformer of similar size, is the copied digit at the last prompt token
  present (minimal-pair probe) but swamped (standard probe and structured decoder at chance-like levels)?
- Either answer changes the note. Replication makes the probing half a general claim; failure makes the swamping a
  finding about Ouro (or looped models).

## 2. Design

**Model.** `Qwen/Qwen2.5-1.5B` (base; 28 layers, d = 1536, 1.5B parameters), fp32, frozen. Its tokenizer gives every
digit its own token, as Ouro's does (checked: all 1,370 v181-design and 3,080 v185-design prompts pass the minimal-pair
token check). Llama-3.2-1B is not used: it tokenizes numbers in chunks of up to three digits, so the units digit would
not have its own token.

**Prompt.** Unchanged from the Ouro studies: the first four GSM8K training problems, each followed directly by its final
answer, then "Question: <q>\nAnswer:".

**Items.** Rebuilt exactly as for Ouro from `v178_labels.json` (same seed, same functions):
- **v181 design (primary):** 137 families from 77 GSM-Symbolic templates; one integer's units digit set to 0–9.
  Slots +0 (the digit), +1, +3, sentence end, '?', last prompt token.
- **v185 design (secondary):** 77 base questions × {added sentence at the start, before the last sentence} × {vary a
  colour word, vary a digit} in "Unrelated fact: a nearby park has 4D <colour> benches." Slots +0, '?', last.

**States.** Hidden states after layers 0 (embeddings), 4, 7, 14, 21 and 28 (final, normed): h0, l4, l7, l14, l21, l28,
the analogues of Ouro's h0, mid1, end1–end4 (a quarter, half, three quarters and all of the depth). **Q = l7, l14, l21,
l28.** The last slot is also stored at all 29 depths (descriptive profile).

**Forward pass** (`v202_extract.py`). The shared 4-shot prefix is run once and its key-value cache reused (a fresh copy
per prompt). Checked against a full forward on the first prompt of each design: logits and every stored state must
agree to < 1e-3, or the run aborts.

**Probes and measurements** (`v202_analyze.py`): the Ouro code, unchanged.
- The v178 probe (StandardScaler → PCA 256 → logistic, inner grouped CV over C), 5 folds grouped by template.
- Standard probe on raw states; contrastive (minimal-pair) probe on family-demeaned states.
- `v181_swamp_check.py`: overlap O, signal/noise in the digit subspace, structured single-prompt decoder (shrinkage LDA),
  Gaussian-model ceiling, learning curve (25/50/75/100% of training templates).

## 3. Gates (any failure → UNSCORABLE)

- **G1:** minimal pairs verified in ≥ 99% of prompts in each design.
- **G2 (apparatus):** the standard probe reads the varied digit on its own token (+0 slot, l7) at ≥ 0.90.
- **G3:** the cached-vs-full forward check < 1e-3 (the extraction aborts otherwise).

## 4. Primary readout and labels (v181 design, last prompt token, over Q)

- c = mean contrastive accuracy over Q; s = max standard accuracy over Q; lda = max structured decoder over Q;
  gain = mean over Q of standard(100% of training templates) − standard(50%).
- **ABSENT:** contrastive < 0.30 at every Q layer (the digit does not reach the last token).
- **READABLE:** s ≥ 0.50 or lda ≥ 0.60 (not swamped: a single-prompt linear readout works).
- **REPLICATES:** c ≥ 0.80, s ≤ 0.30, lda ≤ 0.40 and gain ≤ 0.05 (present but swamped, as in Ouro).
- **MIXED:** otherwise.

For reference, Ouro: c 0.93–0.97, s 0.15–0.23, lda 0.17–0.21, gain +0.015.

**Secondary (descriptive, not labels):** Q1 within/between spread and Q2 standard/contrastive at every slot and state;
O and signal/noise; the last-token profile over all 29 depths; v185 design: standard and contrastive for a colour word
and an added digit, and O(colour)/O(digit) (Ouro: 5.37 / 9.75 = 0.551).

## 5. Pilot (timing and tokenisation only)

`astrocyte-v202-pilot` (Kaggle CPU): the extraction on 6 families and 2 base questions with the real model, the
tokenisation check over every item, and the analysis selftest (synthetic states on the real family structure, through
the real probe: swamped → REPLICATES, readable → READABLE, absent → ABSENT).

**5a. Pilot v1 result (2026-10-15; `results_v202/pilot_v1/`).**
- Real model on Kaggle CPU (torch 2.11, transformers 4.54.1): prefix 213 tokens; 2.6–2.9 s per prompt. The full run
  (4,450 prompts) takes about 3.5 hours on CPU, within the 12-hour limit, so no GPU is needed.
- Cached vs full forward: max |logit diff| 1.7e-5 / 2.4e-5, max |state diff| 7.6e-5 (bar 1e-3).
- Tokenisation check over every item: 0 failures in both designs; the pilot's 140 prompts are all valid minimal pairs.
- Selftest: the swamped world gave REPLICATES; the script then stopped on a bug in the selftest's own printing (it looked
  for a line starting with "PRIMARY", which starts with a newline). Fixed; the selftest is re-run alone (v2).
- No probe was fitted on real Qwen states.

**5b. Selftest v2 (2026-10-15; `results_v202/selftest_v2/v202_selftest.txt`, Kaggle CPU, the frozen analysis file):**
every world gives its required label.

| world | contrastive | standard max | decoder max | label |
|---|---|---|---|---|
| swamped | 1.000 | 0.126 | 0.125 | REPLICATES |
| readable | 1.000 | 1.000 | 0.996 | READABLE |
| absent | 0.104 | 0.101 | 0.099 | ABSENT |

## 6. Predictions (made before any target readout)

- Gates pass: P ≈ 0.95.
- **REPLICATES 55%, MIXED 20%, READABLE 20%, ABSENT 5%.**
  - The mechanism check found the crowding partly generic, which favours replication.
  - A non-looped model has no repeated block to re-mix the state, which might leave the digit less crowded at middle
    layers (READABLE at l7 or l14).

## 7. What each outcome means for the note

- **REPLICATES:** "Nothing in Sections 3 and 4 is specific to looped models" becomes a tested statement. One short
  paragraph and a row per measurement in Section 3 (or an appendix table).
- **READABLE:** swamping is not general; the note says it is found in Ouro and not in a standard model of similar size,
  and the probing half is scoped to Ouro (or looped models).
- **ABSENT:** the standard model does not carry the digit to the last token; the note says so and scopes the claim.
- **MIXED:** reported as measured, with no general claim.

## 8. Run plan and compute

1. Local code test on a tiny random Qwen2 (done: cached-vs-full |diff| ~1e-6; every real prompt passes the token check).
2. Pilot (Kaggle CPU) and selftest: done (section 5).
3. Freeze: hashes of `v202_extract.py`, `v202_analyze.py`, `v178_analyze.py`, `v181_swamp_check.py`,
   `v178_labels.json`, `build_kaggle_v202.py`; push to origin before the main run.
4. Main extraction `astrocyte-v202-qwen`: 4,450 prompts on Kaggle CPU, about 3.5 hours.
5. CPU analysis `astrocyte-v202-cpu`.

## 9. RESULT (2026-10-15; `results_v202/cpu/v202_analysis.txt`)

Run: extraction `aurshalevmerin/astrocyte-v202-qwen` (Kaggle CPU, 11,372 s, 2.6 s per prompt; cached-vs-full check
6.5e-5 and 8.4e-5), analysis `astrocyte-v202-cpu` (CPU). Every file ran at its frozen hash.

**Gates: all pass.** Minimal pairs 4,450/4,450; standard probe on the digit's own token at l7 1.000; check < 1e-3.

**PRIMARY: REPLICATES** (predicted 55%). At the last prompt token, over l7–l28:
- minimal-pair (contrastive) probe mean 0.876 (0.98 / 0.75 / 0.90 at l7 / l21 / l28; 0.99 at l4);
- standard probe at most 0.139;
- structured single-prompt decoder at most 0.143;
- standard-probe gain from 50% to 100% of templates +0.004.

| | Ouro (v181) | Qwen2.5-1.5B (v202) |
|---|---|---|
| minimal-pair probe, last token | 0.92–0.97 | 0.75–0.98 |
| standard probe | 0.15–0.23 | 0.12–0.14 |
| structured decoder | 0.17–0.21 | 0.12–0.14 |
| Gaussian ceiling | 0.75–0.92 | 0.53–0.73 |
| overlap O | 6.5–11.7 | 15–24 |
| gain 50→100% templates | +0.015 | +0.004 |
| trace (within / between spread) | 5–7% | 2–6% |

The copied digit is present but swamped in a standard transformer too, and more crowded than in Ouro.

**Secondary (descriptive).**
- **Depth profile at the last token:** the minimal-pair probe reads the digit at 0.98–1.00 from layer 1 to 9, dips
  to 0.58–0.63 at layers 18–20, and recovers to 0.88–0.91 from layer 23. The standard probe stays at 0.12–0.19 at every
  depth.
- **Colour word vs added digit (v185 design):** the minimal-pair probe reads both at 0.98–1.00. Their subspaces are
  equally crowded: O(colour)/O(digit) = 0.926 over l7–l28 (Ouro 0.551). The exception is the final layer, where the
  colour's crowding falls (O 4.5–6.1) and the standard probe reads the colour at 0.62–0.70, while the digit stays at
  0.15–0.18 (O 16–21).
- Slots before the last token behave as in Ouro: the minimal-pair readout fades with depth at +3 and the sentence end
  (0.42–0.50 at l28) and holds at '?'.

**Reading, per section 7 (REPLICATES).** "Present but swamped" is not specific to looped models: in Qwen2.5-1.5B the
copied digit is at the last prompt token in a code shared across templates, and no single-prompt linear decoder we
built reads it. The note's "nothing in Sections 3 and 4 is specific to looped models" becomes a tested statement for
Section 3. In Qwen the crowding is the same for a colour word and a digit, which supports the generic reading of the
mechanism check (any token's subspace is crowded), except at the final layer, where the colour becomes readable.
