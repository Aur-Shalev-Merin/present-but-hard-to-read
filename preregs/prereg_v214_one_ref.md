# v214 pre-registration: does one reference prompt do the work of ten?

Workstream paper3 (follow-up to the Ouro note, arXiv:2610.28498). Claimed 2026-10-22. Saved states only; no extraction.

## 1. Why

The note reads the copied digit by comparing the ten variants of a question that differ only in that digit
(subtracting the family mean): 0.79-0.96 at the last prompt token, against at most 0.17-0.36 for a probe that sees one
prompt at a time. Using that comparison needs all ten variants. If a single sibling (the same question with one fixed
other digit) does nearly as well, the presence check in the protocol costs one extra forward pass instead of nine.
This is a cost question about the check, not a better single-prompt reader: making the sibling needs the input edited,
so it applies where the protocol already edits inputs, and it is never pitched as a cheaper probe of unknown values.

## 2. Design

- Data (saved states, last prompt token, the note's Q layers): PRIMARY = the minimal-pair families built from GSM8K test
  questions in four models: Qwen2.5-1.5B (`astrocyte-v205-b3`, 150 questions), Ouro-1.4B (`astrocyte-v206-b3`, 100),
  Qwen2.5-7B and Ouro-2.6B (`astrocyte-v213-x-*`, `v213_b3`, the 100 GSM8K questions). SECONDARY = the Orca-Math
  families of v213 (100 per model) and the 137-family v181 design in Qwen2.5-1.5B (v202), Qwen2.5-7B and Ouro-2.6B
  (v213). Ouro-1.4B's v181 states are on the second account and are not used.
- Q: Qwen l7, l14, l21, l28; Ouro end1-end4 (loop ends).
- Modes, all with `v178_analyze.probe_predict` (PCA-256 + logistic, C by inner grouped 3-fold CV) and the same outer
  5 folds (grouped by question for B3 data, by template for v181 data):
  - **mp**: subtract the family's 10-variant mean (the note's minimal-pair contrast); labels 0-9, chance 0.10.
  - **one (r)**: subtract the family's variant with digit r and drop that variant; labels the other nine, chance 1/9.
    Run for every r = 0..9 and averaged.
  - **std**: no subtraction (single-prompt probe, reference only).
- Comparison on a chance-corrected scale k = (acc - c)/(1 - c), averaged over Q (and over r for one);
  D = k_one - k_mp.

## 3. Labels (per model, on its primary data)

- **UNSCORABLE:** mp < 0.50 (digit not shown present).
- **ONE-SUFFICES:** D >= -0.05.
- **NEEDS-FAMILY:** one < 0.50.
- **PARTIAL:** otherwise.

Secondary data are reported with the same numbers and labels, descriptively.

## 4. Predictions and what each outcome means

- ONE-SUFFICES in all four models 55%; at least one PARTIAL (none NEEDS-FAMILY) 35%; any NEEDS-FAMILY 10%.
- **ONE-SUFFICES:** the protocol section says a single reference variant reads the digit about
  as well as all ten, so the presence check costs one extra forward pass; numbers in an appendix table.
- **PARTIAL:** the paper says one reference recovers most of the signal and the full family reads better, with the gap.
- **NEEDS-FAMILY:** the paper says the comparison needs several variants: the digit's change depends on the context
  beyond a shared code plus one sibling's idiosyncrasy. Reported as a limit of the shortcut.
- Not a change to any existing claim of the note in any outcome. A placeholder-token reference (a neutral token in
  the digit's place, needing new extraction) is run only if wanted after this result.

## 5. Pre-freeze checks (no probe statistic on real states computed)

- Selftest (`python v214_one_ref.py --selftest`, local, 25 s): additive world (shared code + family context) ->
  ONE-SUFFICES (mp 1.000, one 1.000); item-noise world (large per-variant noise) -> PARTIAL (mp 0.984, one 0.856,
  D -0.144): the shortcut can fail and the rule detects it.
- Metadata (downloaded meta files only): every primary and secondary family has exactly ten variants, one per digit.

## 6. Run plan

`V214_MODE=main python build_kaggle_v214.py` -> `astrocyte-v214-main` (Kaggle CPU, kernel_sources = the five state
kernels above); freeze pushed before it runs; RESULT section here.

## 7. Frozen files (sha256, first 16 hex, LF)

- `v214_one_ref.py` 2202cec9acd23aeb
- `v178_analyze.py` 90bbc8a848dc5544
- `build_kaggle_v214.py` 60cd62787234d023

The kernel prints the sha of each file it runs.

## 8. RESULT (2026-10-22, `astrocyte-v214-main`; files ran at the frozen hashes)

**PARTIAL in all four models** (predicted: ONE-SUFFICES in all four 55%, at least one PARTIAL 35%). Last prompt
token, GSM8K test-question families, mean over Q (one: also over the reference digit r):

| model | std | mp (10 variants) | one (1 reference) | chance-corrected mp / one | D | label |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.14-0.21 | 0.912 | 0.807 | 0.902 / 0.783 | -0.118 | PARTIAL |
| Ouro-1.4B | 0.25-0.35 | 0.959 | 0.901 | 0.955 / 0.888 | -0.067 | PARTIAL |
| Qwen2.5-7B | 0.16-0.39 | 0.861 | 0.776 | 0.845 / 0.749 | -0.097 | PARTIAL |
| Ouro-2.6B | 0.17-0.20 | 0.787 | 0.647 | 0.763 / 0.603 | -0.160 | PARTIAL |

Secondary (Orca families, v181 design): PARTIAL in all five, D -0.108 to -0.163.

- One reference recovers most of the signal (chance-corrected 79-93% of the ten-variant contrast's level) but not
  all of it. The gap is small where the digit is strongest (Qwen l7: 0.96 vs 0.99; Ouro-1.4B end1: 0.92 vs 0.98)
  and largest where it is weak (Qwen-7B l21: 0.49 vs 0.63; Ouro-2.6B end4: 0.53 vs 0.69).
- The choice of reference digit matters: per-layer accuracy ranges over r by 0.05-0.25 (e.g. Ouro-2.6B end4
  0.35-0.59). A single sibling carries its own idiosyncratic change, which the ten-variant mean averages out, as in
  the selftest's item-noise world.
- Reading per section 4 (PARTIAL): one reference recovers most of the signal, the full family reads better. Untested:
  two or three references, or a neutral placeholder token.
