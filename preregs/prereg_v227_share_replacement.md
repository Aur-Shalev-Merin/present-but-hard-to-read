# v227 pre-registration: replacement positive control for the share swap (Mistral-7B-v0.1, the b step)

Workstream paper3 (Ouro note). Claimed 2026-10-27 (v225 and v226 were used for other work).
Follows v224.

## 1. Why

v224 froze a share-based operation swap and judged it on two fresh positive controls. Control 2 (Qwen2.5-7B, the a
step) was FOLLOWS (I / N 0.65), and no fresh null run was FOLLOWS (0 / 420). Control 1 (Qwen2.5-1.5B, the b step) was
UNSCORABLE: b correct 0.78 < 0.80. The smoke had sampled only the first families, where the 1.5B model was at 1.000,
so the gate failure was not foreseen. The verdict is PARTIAL, which allows no re-score of the note's targets.

This study replaces control 1 with a model expected to pass the gate, keeping everything else frozen. It is added
after v224's results were known; that is acceptable only because the statistic, TAU and the pass rule were frozen in
v224 and are not changed here, so this control cannot be tuned to pass. A failure here counts against the method.
Control 1's descriptive numbers (inb I / N 0.52) are not used.

## 2. Design (`v227_run.py`)

- Model: **Mistral-7B-v0.1** (32 layers, another model family), fp16 over two T4s. Not looked at with any statistic.
  The first choice, Qwen2.5-3B, failed the accuracy pilot (section 4) and was dropped before any state was saved.
  Mistral encodes each digit as a space token then the digit; the digit's id is the last token of its encoding, and
  the chain's tokens (letter, "=", space, digit) sit where the design expects (checked on 240 items, 0 mismatches).
- Items: v224's b set unchanged (= v222's, f917c3b7da3ddc87; 240 families, 12,000 prompts, Simple chain, teacher
  forced). Operands (d1, d2); version = b's operation.
- Statistic: `v224_share.analyse`, unchanged (TAU 0.30, accuracy gate 0.80 per version, labels as v224 section 2).
  The only adaptation: v224's layers {16, 20, 24, 28} of 28 map to the same relative depth in 32 layers,
  round(l x 32 / 28) = **{18, 23, 27, 32}**; only those states are saved.
- Primary: **inb**. Descriptive: eqb, letb, eq, last.

## 3. Reading

Prediction: inb FOLLOWS, 75%.

- **inb FOLLOWS:** with v224's control 2 and fresh nulls, the share swap is **VALID**. A separate pre-registration
  then re-scores the note's targets with it, unchanged, disclosing that those results were known under the old swap,
  that the short-chain eq / last positions were seen in v223 and v224's development run, and that this control was a
  replacement added after v224.
- **inb not FOLLOWS (UNCLEAR, DOES-NOT-FOLLOW, NO-SIGNAL):** the method stays PARTIAL, with a failed replacement
  reported; no re-score, and no further replacement.
- **UNSCORABLE** (gate fails despite the pilot): the method stays PARTIAL; no further replacement.

## 4. Pre-freeze checks

- Accuracy pilot (`astrocyte-v227-pilot`): 400 items drawn at random over all families (seed 2271), b step accuracy
  only, no states saved. Freeze only if both versions are >= 0.85 (a margin over the 0.80 gate); otherwise stop and
  report. Qwen2.5-3B (2026-10-27, `astrocyte-v227-pilot` v1): b correct 0.542 (plus, 203 items) and 0.589 (minus,
  197): failed, model dropped. Mistral-7B-v0.1 (`astrocyte-v227-pilot` v2): positions as expected ('t', '=', space token, digit), b correct
  0.951 (plus, 203 items) and 0.853 (minus, 197): passes the 0.85 freeze bar. The minus value is close: its rough 95%
  interval reaches 0.80, so the full set could still fail the gate (then UNSCORABLE, section 3).
- Mechanics: `v227_run.py analyse` on random states (12,000 x 5 x 4 x 16): runs, NO-SIGNAL at every position.

## 5. Frozen files (sha256, first 16 hex, LF; 2026-10-27)

- `v227_run.py` 628ae41f0f2ece2e
- `build_kaggle_v227.py` 0a1efff371b4c6d1
- reused unchanged (v224 freeze): `v224_share.py` 2b0bc8868d13b00e (TAU 0.30), `v224_items.py` 14087b6369b3ce8e,
  `v222_items.py` b5813a4fb56ef4f8, `v221_items.py` 2b6bd412a702ccc4, `v215_items.py` 9249f0523d7219eb, `v215_pilot.py`
  d049a0b2d81fb80b, `v178_analyze.py` 90bbc8a848dc5544

Run order (aurmerin): `extract` (GPU), then `analyse` (CPU). No statistic on Mistral's states before this freeze.

## 6. RESULT (2026-10-28; frozen code, `aurmerin/astrocyte-v227-x`, `-a`)

Gate: b correct 0.959 (plus) and 0.865 (minus), both >= 0.80.

**inb FOLLOWS:** N 0.150, D+ +0.047, D- +0.097, I +0.072 [+0.066, +0.077], I / N 0.48 (bar 0.30). Shares: plus S 0.148
vs M 0.101; minus M 0.152 vs S 0.055. Descriptive: eqb FOLLOWS too (I / N 0.48); letb, eq and last UNCLEAR, each with
D+ < 0 (a preference for the difference on plus problems, as in Qwen2.5-7B's development run).

Prediction (inb FOLLOWS, 75%) right.

**Reading (section 3): with v224's control 2 (Qwen2.5-7B, a step, FOLLOWS) and fresh null seeds (0 / 420 FOLLOWS),
the share swap is VALID.** It has now passed two positive controls it was never tuned on, in two model families and
two steps of the chain, and no null world. A separate pre-registration re-scores the note's targets with it,
unchanged, with the disclosures listed in section 3.
