"""v213 items (docs/prereg_v213_scale_up.md): the many-question pool, rebuilt deterministically wherever it is needed.

GSM8K part: v205_extract.gsm_items, unchanged (every usable training question after the four shots, one random units
digit each, seed 205; a fixed random 1,000 test questions, permutation seed 2051; source 'gsm').
Orca part (microsoft/orca-math-word-problems-200k, train split):
  - usable: contains an integer with >= 2 digits (as GSM8K); the last such integer's units digit is set to a random
    digit (seed 213);
  - near-duplicates removed: 4-word shingles with numbers masked; a question is dropped if it shares >= 30% of its
    shingles (or the other question's, whichever set is smaller) with any GSM8K question (train or test), counting
    only shingles that occur in at most 50 GSM8K questions;
  - the remaining pool is shuffled (seed 2131); the first 1,000 form the Orca test set ('orca_test'); training
    candidates follow in that order, and a candidate is dropped if it is a near-duplicate (same rule) of an Orca
    test question, or a reworded twin of any test question (GSM8K or Orca): content-word Jaccard >= TWIN = 0.30
    (words of >= 4 letters outside a stop list; compared with every question of each test set that shares a word occurring
    in at most 30 questions of that test set); the
    first N_ORCA_TRAIN = 40,000 kept form the Orca training pool ('orca_train', rank = order).
  - test items (GSM8K and Orca) get twin = True if a GSM8K training item is a reworded twin by the same rule (the GSM8K
    pool is kept as in v205/v206; the analysis reports accuracy on the test items without a twin as a secondary measure).
Each item: source, split (train/test), rank (training order within its source), question (with the set digit), orig,
cu (char index of the units digit), digit, idx (a unique integer id), twin (test items only).
sha16() is the sha256 prefix of the canonical JSON of the item list; kernels abort if it differs from ITEMS_SHA.

  python v213_items.py      (prints counts and the hash)
"""
import collections, hashlib, json, re
import numpy as np

NUM = r'\d+(?:,\d{3})*(?:\.\d+)?'
N_ORCA_TEST, N_ORCA_TRAIN, TH, TWIN = 1000, 40000, 0.30, 0.30
ITEMS_SHA = 'db8f37a5f24b8c64'   # frozen 2026-10-21 (local x2 and Kaggle)


def _shingles(q, n=4):
    w = re.findall(r"[a-z#']+", re.sub(NUM, '#', q.lower()))
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}


class _Index:
    def __init__(self, qs, maxdf=50):
        self.sh = [_shingles(q) for q in qs]; self.ix = collections.defaultdict(list)
        for i, s in enumerate(self.sh):
            for t in s:
                self.ix[t].append(i)
        self.maxdf = maxdf

    def near(self, q, th=TH):
        s = _shingles(q)
        if not s:
            return False
        cnt = collections.Counter()
        for t in s:
            hits = self.ix.get(t, ())
            if len(hits) <= self.maxdf:
                for i in hits:
                    cnt[i] += 1
        return any(c / max(1, min(len(s), len(self.sh[i]))) >= th for i, c in cnt.items())


STOP = set('''that this with from have will were what when then than they their there these those your about after
again also because been being before between both could does doing each every first more most much once only other over
same some such them very which while would into many just like make made need sold left total year years days hours
minutes cost costs price buys bought spend spent gets give gave takes took find how'''.split())


def _words(q):
    return {w for w in re.findall(r'[a-z]+', q.lower()) if len(w) >= 4 and w not in STOP}


class _Twins:
    """Reworded-twin test: content-word Jaccard >= TWIN with any indexed question sharing a rare word."""
    def __init__(self, qs, maxdf=30):
        self.w = [_words(q) for q in qs]; self.ix = collections.defaultdict(list)
        for i, s in enumerate(self.w):
            for t in s:
                self.ix[t].append(i)
        self.maxdf = maxdf

    def twin(self, q, th=TWIN):
        s = _words(q)
        if not s:
            return False
        cnt = collections.Counter(i for t in s for i in self.ix.get(t, ()) if len(self.ix[t]) <= self.maxdf)
        return any(len(s & self.w[i]) / len(s | self.w[i]) >= th for i in cnt)


def _set_digit(q, rng):
    ms = [m for m in re.finditer(NUM, q) if '.' not in m.group() and len(m.group().replace(',', '')) >= 2]
    if not ms:
        return None
    cu = ms[-1].end() - 1; d = int(rng.integers(10))
    return dict(orig=q, question=q[:cu] + str(d) + q[cu + 1:], cu=cu, digit=d)


def build():
    from datasets import load_dataset
    import v205_extract as E5
    gtr = load_dataset('openai/gsm8k', 'main', split='train'); gte = load_dataset('openai/gsm8k', 'main', split='test')
    items = []
    for it in E5.gsm_items(gtr):
        items.append(dict(source='gsm', split=it['split'], rank=it['rank'], question=it['question'], orig=it['orig'],
                          cu=it['cu'], digit=it['digit'], idx=it['idx']))
    gidx = _Index([x['question'] for x in gtr] + [x['question'] for x in gte])
    orca = load_dataset('microsoft/orca-math-word-problems-200k', split='train')
    rng = np.random.default_rng(213); pool = []
    for oi, x in enumerate(orca):
        it = _set_digit(x['question'], rng)
        if it is not None and not gidx.near(x['question']):
            it['oi'] = oi; pool.append(it)
    perm = np.random.default_rng(2131).permutation(len(pool))
    test = [pool[k] for k in perm[:N_ORCA_TEST]]; tidx = _Index([t['orig'] for t in test]); train = []
    twins = [_Twins([it['orig'] for it in items if it['split'] == 'test']), _Twins([t['orig'] for t in test])]; dropped = 0
    for k in perm[N_ORCA_TEST:]:
        if len(train) == N_ORCA_TRAIN:
            break
        if tidx.near(pool[k]['orig']) or any(t.twin(pool[k]['orig']) for t in twins):
            dropped += 1; continue
        train.append(pool[k])
    gtw = _Twins([it['orig'] for it in items if it['split'] == 'train'])
    for it in items:
        if it['split'] == 'test':
            it['twin'] = bool(gtw.twin(it['orig']))
    for split, lst in (('test', test), ('train', train)):
        for r, it in enumerate(lst):
            items.append(dict(source='orca', split=split, rank=r, question=it['question'], orig=it['orig'], cu=it['cu'],
                              digit=it['digit'], idx=1000000 + it['oi'], **({'twin': bool(gtw.twin(it['orig']))} if split == 'test' else {})))
    return items, dict(orca_usable_clean=len(pool), orca_train_dropped=dropped,
                       test_with_gsm_twin={s_: sum(1 for it in items if it.get('twin') and it['source'] == s_) for s_ in ('gsm', 'orca')})


def sha16(items):
    return hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:16]


def load_checked():
    items, info = build(); h = sha16(items)
    if ITEMS_SHA is not None and h != ITEMS_SHA:
        raise SystemExit('v213 items hash %s != frozen %s' % (h, ITEMS_SHA))
    return items, h, info


if __name__ == '__main__':
    items, info = build()
    c = collections.Counter((i['source'], i['split']) for i in items)
    print(dict(c), info, 'sha16', sha16(items))
