# v207: input distance of the digit, word-swap and noise changes (exploratory, descriptive)

*2026-10-18. Not pre-registered: a seconds-scale inspection of the input-embedding matrices, suggested by an outside
review of note v8.3 to narrow the open question in Sec 3.4 (why a digit leaves a much smaller trace at the last token
than a word does, when token-sized noise leaves an equally small one). Script `v207_input_distance.py` (reads only the
needed embedding rows by HTTP range requests on the public safetensors files); output `results_v207/v207_input_distance.txt`.*

## What was measured

The spread between the ten variants of a family at the input, as mean pairwise Euclidean distance between the input
embeddings that differ across variants:
- digit family: the ten digit tokens `0`–`9`;
- word-swap family (v205 B1): the ten fixed swap words (seed 205);
- noise family (v205 B1): the ten noise vectors (seed 205, norm 1.032 = median embedding norm), added to the same digit.

## Result

| | Qwen2.5-1.5B | Ouro-1.4B |
|---|---|---|
| digit vs digit | 0.70 (0.59–0.79) | 0.88 (0.76–0.98) |
| swap word vs swap word | 1.42 | 1.52 |
| random word vs random word (60 words) | 1.37 | 1.54 |
| noise vector vs noise vector | 1.46 (1.42–1.51) | (no noise run) |
| median token norm | 1.02 | 1.10 |

Last-token signal in v205 B1 (Qwen, layers 7/14/21/28; signal-to-noise inside each change's own subspace, standard probe):
- digit: SNR 0.0009 / 0.0037 / 0.0031 / 0.0025; probe 0.12–0.14.
- word swap: SNR 0.065 / 0.038 / 0.011 / 0.205; probe 0.24–0.74.
- noise: SNR 0.0010 / 0.0006 / 0.0002 / 0.0001; probe 0.11–0.15.

## Reading (descriptive)

- Digits start close together: about half the distance between random words in both models (0.51× in Qwen, 0.58× in
  Ouro). Part of the digit's small trace is already present at the input.
- Input distance does not explain the word-vs-noise difference. Word-swap and noise variants start equally far apart
  (1.42 vs 1.46), but at the last token the word swap's SNR is 50–2,000 times the noise's, and only the word swap is
  readable. Real-token changes are carried to the last token; random directions of the same size are not.
- Beyond the input gap, digits are carried less than words at most depths. If SNR scaled with squared input distance,
  a word's SNR would be about 4× a digit's ((1.42/0.70)²); observed 3–80× (about 4× only at layer 21).
- Open: why digit differences are carried less well than word differences. The attention-transport check
  (last token's attention to the varied position) would be the next step; it needs a new extraction.
