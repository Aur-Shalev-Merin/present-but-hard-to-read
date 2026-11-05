# v222 pre-registration: does the clamp detect a value that is stored at one token and read at a later one?

Workstream paper3 (Ouro note). Claimed 2026-10-26. Follows v221.

## 1. Why

v221 (positive control) found that v215's copy-orthogonalised clamp has power where a computed value is used at the
same token (inb, B is the next token: cBo - cR +0.607, USED), and that the operation swap has none. But the note's
NOT-USED verdicts (Ouro; Kudo et al.'s answer-only setting at eq) are about a different kind of use: a value held at
one token and read by a later one. v221 does not show the clamp can see that. This study runs the clamp, unchanged,
where such a use is certain: in Kudo et al.'s Simple chain " b = B , a = A" the model writes b's digit, then computes
a = k o2 b four tokens later (tokens ' ,', ' a', ' =', ' ', then the digit). B is stored at the digit token (digB) and
read from there, unless the model recomputes it from the prompt; a full-state patch tells which.

## 2. Task, items, positions

- v221's items unchanged (v215's 240 families, 12,000 prompts, Simple chain, teacher-forced prompt + " b = B , a = A");
  `v222_items.py` adds digB (the b digit token) and inA (the token before the a digit). ITEMS_SHA f917c3b7da3ddc87.
- Qwen2.5-7B fp16 over two T4s; states at digB x hidden_states 0, 4, ..., 28. correct = the b digit is the argmax at
  inb and the a digit is the argmax at inA (teacher forcing). Accuracy is reported, not gated (pairs use correct x).
- In families with k = 0 and o2 = '+' (one sixth), a is a copy of b; it is still a use of B and is reported by k.

## 3. Clamp (v215 amendment 1, unchanged except the position and the outcome)

- Plan: `v215_analyze.patch_plan` at digB: B subspace (class means of B, 9 dims), d2-copy subspace, B orthogonalised
  against it (cBo), random 9-dim subspace; fitted on the other half of the families (seed 2151 split, 2152 random);
  1,000 pairs, x correct, x' same family, version and d1, different d2 (so different B and A).
- For each pair, x''s layer outputs captured at digB; x run under none, cB, cBo, cD, cR (band: hidden_states 12, 16,
  20, 24), **full** (layer outputs 12..28; v215's positive control) and **fullall** (every layer output 1..28;
  descriptive, new: is A read from digB at all?). x's text is unchanged, so the prompt still gives x's B.
- Outcome: argmax over the ten digit tokens at inA; hit = x''s A, kept = x's A.
- Statistic: hit(cBo) - hit(cR), cluster bootstrap over x's family, 10,000 resamples, seed 2223, 97.5% CI.
  **UNSCORABLE** hit(full) < 0.30 | **USED** diff >= 0.10 and CI lower > 0 | **NOT-USED** CI upper < 0.02 | **UNCLEAR**.
- Descriptive: cB - cR, cD, kept rates, hits by k (a = k o2 b).

## 4. Predictions and readings

Predictions: full >= 0.30, 70%; USED, 55%; fullall >= 0.50, 80%.

- **USED:** the clamp detects a value stored at one token and read four tokens later. The note may say its NOT-USED
  verdicts come from a test shown to have power for this kind of use, in Qwen2.5-7B at this distance (the Kudo
  answer-only eq position is further from the answer; Ouro is a different model). Stated with those limits.
- **NOT-USED or UNCLEAR (full >= 0.30):** the full patch at the band layers moves a, so a is read from digB through
  those layers, yet the clamp does not detect it. The clamp then misses this kind of use, and the note's NOT-USED
  verdicts lose most of their force; the main text says so.
- **UNSCORABLE (full < 0.30):** patching the whole state at the band layers does not move a. Then fullall decides
  the description (no label): fullall >= 0.30 means a reads B from digB below layer 12, outside the clamp's band, and
  the note states that the clamp sees only use through the band layers (at the note's own eq tests the full patch did
  move the answer, 0.86-0.96, so that route exists there); fullall < 0.30 means a is recomputed from the prompt and
  this control is uninformative, which the note also states.

## 5. Pre-freeze checks

- Items (`python v222_items.py`): 12,000; tokens after the b digit ' ,', ' a', ' =', ' ', digit (Qwen tokenizer,
  local). The extraction asserts the b and a digit tokens and inA = the a digit token - 1 for every item.
- Selftest (`python v222_clamp.py --selftest`): see section 7.
- Smoke (`astrocyte-v222-smoke`, before the freeze; mechanics only, random bases, 20 pairs, no statistic): section 7.

## 6. Frozen files (sha256, first 16 hex, LF; 2026-10-26)

- `v222_items.py` b5813a4fb56ef4f8 (ITEMS_SHA f917c3b7da3ddc87)
- `v222_clamp.py` 25c7dccc72e25c7f
- `build_kaggle_v222.py` 7ca3428a5b384da1
- reused unchanged: `v221_items.py` 2b6bd412a702ccc4, `v215_analyze.py` e9d53523b1da41e2 (patch_plan), `v215_items.py`
  9249f0523d7219eb, `v215_pilot.py` d049a0b2d81fb80b, `v178_analyze.py` 90bbc8a848dc5544

Run order: `V222_MODE=run` (GPU, one kernel: extraction, plan, clamp, analysis) on the second Kaggle account
(aurmerin; V222_ACCOUNT), as v221 (execution note 1 there).

## 7. Selftest and smoke record

- Selftest (60 synthetic families, B coded at digB): plan 1,000 pairs, 28 bases, every pair differs in B and A; made-up
  results labelled correctly (cBo moves a -> USED; nothing moves -> NOT-USED; full fails -> UNSCORABLE).
- Smoke (`aurmerin/astrocyte-v222-smoke`, 2026-10-26; sections 1-4 were written before it finished): 40 items, b
  digit token ' 0' at 101, inA ' ' at 105, a digit next (assertions pass on all 40); 20 pairs from family 0 with random
  bases (pairs not filtered for correct items in the smoke): hook check (fullall at inA itself) 0.90, the misses being
  pairs whose x' the model gets wrong. The smoke prints the full / fullall hit rates at digB on these 20 pairs (0.90
  / 0.90), so the full-patch rate was seen on 20 real pairs before the freeze; no reading or threshold was changed.
- No statistic on real v222 states has been computed before this freeze.

## 8. Amendment 1 (2026-10-26, after astrocyte-v222-run was launched, before any of its output was seen)

Code, labels and thresholds unchanged; the readings in section 4 are revised. Review note: at digB, b is the written
token, so its embedding carries B into every layer of that token, while the computed b in the note's NOT-USED tests
(eq, pre) is never written. A miss here could come from that difference alone. The full-patch gate already covers
the case where a reads B only below the band (then UNSCORABLE, and fullall describes it); it does not cover a miss with
full >= 0.30, which could still be specific to a written value. Revised readings:

- **USED:** as before, with one more stated limit: the value clamped here is written; transfer to a computed, unwritten
  value (the note's eq and pre tests) is not shown.
- **NOT-USED or UNCLEAR (full >= 0.30):** the clamp misses a written value that is read four tokens later through the
  band layers. The note states that the clamp's power for stored-and-read-later use is not shown (and failed for a
  written value), and reports its NOT-USED verdicts as coming from a test of unshown power for that kind of use, in
  the main text. It does not claim, from this result alone, that the clamp would miss a computed value.
- **UNSCORABLE:** unchanged.

This softens the consequence of a miss (section 4 said the NOT-USED verdicts "lose most of their force"); it is made
before any v222 result, and the main-text statement above is still required.

## 9. RESULT (2026-10-27; frozen code with amendment 1, `aurmerin/astrocyte-v222-run`)

Accuracy (teacher forced): b 0.911, a 0.905, both 0.837 (plus 0.847, minus 0.827). Plan: 1,000 pairs; at the band
layers the orthogonalised B keeps 9 dims and 0.53-0.64 of B's share.

| condition | none | cB | **cBo** | cD | cR | full | fullall |
|---|---|---|---|---|---|---|---|
| hit (a moves to x''s A) | 0.000 | 0.826 | **0.806** | 0.070 | 0.000 | 0.888 | 0.895 |

**cBo - cR +0.806 [+0.765, +0.844] -> USED.** full 0.888 >= 0.30 (scorable); fullall 0.895, about the same as full, so
a reads B from digB through the band layers and later, hardly at all below layer 12. The d2-copy subspace alone (cD)
moves a in 7% of pairs.

By k (descriptive): a = k + b, k = 1, 2 and a = k - b, k = 7, 8, 9: cBo 0.83-0.87; a = 0 + b (a copies b): cBo 0.632,
full 0.726.

Predictions: full >= 0.30 (70%) right; USED (55%) right; fullall >= 0.50 (80%) right.

**Reading (section 4 with amendment 1):** the clamp detects a value stored at one token and read four tokens later,
with a wide margin, in Qwen2.5-7B. Limits the note states: the value here is written (its own token), the distance is
four tokens, and the model is Qwen2.5-7B; transfer to a computed, unwritten value (the note's eq and pre tests) and to
Ouro is not shown.

Note (2026-10-27, after the result; outside review): section 1 says the note's NOT-USED verdicts are "about a value
held at one token and read by a later one". That holds for the answer-only test at the end of b's equation, not for
Ouro's: its patch and clamp act at P1 and read the digit predicted at P1, the same token, which v221's inb control
matches. The paper maps each control to the verdicts it bears on. Also: at digB, B is the token itself, so this
control is close to built in, as inb was in v221.
