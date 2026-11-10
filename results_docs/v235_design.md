# v235 design: a pretrained positive control for the clamp (a two-hop bridge)

Workstream paper3 (Ouro note, arXiv v2). Design notes, not a pre-registration. The prereg
(`docs/prereg_v235_*.md`) is drafted after pilot 2.

## Why

The clamp protocol has shown power only in small models trained from scratch (v230, v231, v233). Every
pretrained-model reading so far (Ouro, Qwen on Kudo's form) is NOT-USED, UNCLEAR or "cannot tell". A reader cannot tell
whether the protocol can say "used" in a pretrained model at all. v235 looks for a pretrained setting where an unwritten
intermediate is used for certain, and asks whether the clamp finds it there.

The setting is two-hop factual recall without a chain: "The capital of the country where the city of X is located is"
-> the capital. The country (the bridge) is never written. If the model answers through the country, the country's
code at the city's tokens is used.

## Model and data

- Qwen2.5-7B base, fp16, Kaggle T4 x2 (the same model as v216 and v234).
- 25 countries (`v235_data.py`, from Wikidata and Compositional Celebrities): up to 60 cities each by population, and
  122 people with a known birth country. Downloads sit in separate scratch folders.
- Prompts (`v235_pilot.py`): three demonstrations from countries outside the 25 (Sydney, Gothenburg, Mombasa; for people,
  Cate Blanchett, Greta Garbo, Wangari Maathai).

## Pilot 1 (`astrocyte-v235-pilot`, done 2026-11-07)

- Country -> capital 25/25. City -> country 0.97 (1,083 cities). Two-hop city -> capital 0.98, or 0.995 among cities
  whose country the model gives right; the first-token argmax over the 25 capitals agrees with full-continuation
  scoring on 0.995. The capitals' first tokens are all distinct.
- Person -> country 0.93; person two-hop 0.86 (people born in the United States 0.38).
- Known items (country right, two-hop right on both scorings, names in one country only, duplicates kept once):
  1,038 cities (Vietnam 8, Poland 15, France 18; most countries 50-60) and 99 people
  (`v235_pilot2_items.py` -> `v235_pilot2_items.json`).

## Ground truth (pilot 2, `astrocyte-v235-pilot2`, running)

Pilot 2 tests whether the country at the city's tokens is what the answer uses. It never touches a learned subspace.
For 200 target cities (8 per country), whole-state patches at one position replace one layer at a time
(hidden_states 2-24) or all layers 1-27. The positions are e1 (the city's last token) and last (the prompt's last
token). Sources:

- city_other: a known city of another country. The answer should move to that country's capital.
- city_same: another city of the same country. The answer should stay.
- person: a person born in another country, taken at the person's last token. The person and the city share only
  the country, so a move to the person's capital means the country (or something fixed by it) is what the downstream
  computation reads at e1.
- country: another country's name, from "The capital of {name} is".

It also runs a random 24-dim clamp at e1 (layers 8-20; it should do nothing). It saves states for all 1,038 cities at
hidden_states 0, 4, ..., 24 and 27 (e1 and last), plus the mean layer-0 embedding of the city-name tokens.

From these states, on CPU and before the prereg:

- Country readout per layer: nearest class mean, fitted on half the cities and tested on the other half. This is a
  probe check, allowed before the freeze.
- Copy ceiling: the same readout from the name tokens' layer-0 embeddings. It says how much of the country the
  surface of the name already carries.

## Planned treatment (to be fixed in the prereg)

- Clamp pairs: x = a city of country C, x' = a city of country C' != C, both known. Outcome: argmax over the capitals'
  first tokens. hit = cap(C'), kept = cap(C).
- U = the country's class-mean subspace at e1 (25 classes, 24 dims), fitted on the other half of the cities. Uo = U
  projected off the copy, meaning the country's class means in the name tokens' layer-0 embeddings. Applied at
  band layers chosen from pilot 2: the layers where the single-layer city_other and person patches move the answer.
  h <- h - U U^T (h - h'), as in v216 and v234.
- Comparisons: a random subspace of Uo's dimension (cR); same-country pairs (the clamp should leave the answer alone:
  a design check); a negative position (a demonstration token, where no country of x is held).
- Gate: the whole-state patch at the position must reach hit >= 0.30 (v234's rule), or the position is UNSCORABLE.
- Share of the country code left after the copy projection, reported with v216's definition.
- Optional faint arm: degrade only the clamp's fitting data (fewer cities per country, or noise), as v231 did, down to
  the share of value that the Ouro and Kudo clamps keep. This asks whether the clamp still finds the bridge when the
  bridge is faint.

## Readings (draft)

- The clamp detects the bridge: the protocol can say "used" in a pretrained model, at least at this signal strength.
  The paper's NOT-USED and cannot-tell readings in Ouro and Qwen then stop being "the method may be blind in pretrained
  models".
- The clamp misses a bridge that the whole-state patches show is used: the clamp form is blind in pretrained models, and
  the paper must say so about every pretrained reading.

The literature already reports that two-hop bridges in pretrained models are resolved at the subject's last tokens in
early to middle layers. So the bridge is likely a strong signal. That is why the faint arm matters.
