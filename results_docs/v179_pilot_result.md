# v179 control-only pilot (2026-10-11): where is a copied question digit readable in Ouro? — **only on its own token** (for a STANDARD probe)

> **Correction (v181, same day):** the reading below ("values stay at their token", "summary-token probing is dead") was wrong. A minimal-pair (family-demeaned) probe reads the copied digit at the last token at 0.92–0.97 on every loop. The digit is present but swamped by context, about 6% of the variation between questions. See `docs/v181_result.md`.

Exploratory pilot after v178 came out UNSCORABLE (`docs/prereg_v178_ouro_latent_intermediates.md`). Code:
`v179_extract.py`, `v179_analyze.py`, `build_kaggle_v179.py` (da85fa7). Output: `results_v179/v179_analysis.txt`.
Only control labels were probed: the digits written in the question. No intermediate-value label was read.

**Setup.**
- 840 GSM-Symbolic items (10 per v178 template); frozen Ouro-1.4B, fp32; v176/v177 4-shot answer-only prompt.
- Labels: the units digit of the first, the last and the largest question number.
- Positions: the digit's own token (+0), +1, +3, the end of its sentence, '?', and the last prompt token.
- States: h0, layer 12 of loop 1, and the end of loops 1–4.
- Probe: v178's probe (PCA 256 + logistic, inner CV over C or constant), with 5-fold CV grouped by template.
- Reading rule, fixed before the data: accuracy ≥ max(0.5, majority + 0.15).
- Check: the forward pass matched Ouro's own logits exactly (|diff| 0).

**Result.**
- **+0 (the digit's own token):** reads at every state, 0.97–1.00. This is the apparatus check, and it passed.
- **+1:** never reaches the bar. It rises only at loop 1's mid layer (0.48 / 0.45 / 0.50 against majorities
  0.30 / 0.20 / 0.40) and is back to majority by the end of loop 1.
- **+3, sentence end, '?' and last token:** at majority or below for every label and every state.

**Reading.** The units digit of a question number is linearly readable only on its own token. It is not carried
forward to later positions at any loop. The likely mechanism is that values stay at their token and are fetched by
attention when needed.
- So probing a summary token (v178's design) cannot work, and neither can probing the sentence end or '?'.
- A computed intermediate would also most likely sit on the token where it is computed, not be broadcast.
- Per the decision map set before the data, this is case 1: **summary-token probing is dead**. The remaining route
  to "does Ouro compute the middle step" is causal patching, which needs its own pre-registration.

**Caveat.** This is a linear probe with about 670 training items per fold and generalisation across templates. A
nonlinear or per-template code is not excluded.
