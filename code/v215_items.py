"""v215 items (docs/prereg_v215_kudo_retest.md). Kudo et al.'s Level 3 task in their format (re-implemented from their
generator and preprocessor; see v215_pilot.py for the rendering).

  fam   240 minimal-pair families: a family fixes the letters (a, b), three same-level demonstrations and (k, o2) in
        KO, and contains every operand pair (d1, d2), d1 >= d2, valid in BOTH versions (o1 = + and o1 = -), each in
        both versions. Fields: fam, pair (fam*100 + 10*d1 + d2), ver ('plus'/'minus'), letters, numbers, B, A, src,
        prompt, target, cpos (char positions of eq / semi / last in the prompt), cB (char index of the computed b
        digit in prompt + target).
  fams  the same 240 families (same letters, numbers and demonstrations) in their Simple CoT format (App. A): target
        " b = B , a = A" (no restated equation), demonstrations in the same format; cB = the digit after the first
        "b = " of the target.
  rep   Part C replication set: 10,000 train + 2,000 test single problems over every valid (k, o2, d1, o1, d2), random
        letters and demonstrations; no test problem shares b's equation (d1, o1, d2) with a training problem
        (TEST_TRIPLES: 22 of the 110 valid triples, seed 2154).

  python v215_items.py      -> prints counts and sha16 (ITEMS_SHA)
"""
import hashlib, json, sys
import numpy as np

from v215_pilot import op, render, valid, sample, spaced

ITEMS_SHA = '8c08adec21d20ae3'   # frozen 2026-10-23 (fam + rep + fams)
KO = [(0, '+'), (1, '+'), (2, '+'), (9, '-'), (8, '-'), (7, '-')]
N_PER_KO = 40
N_REP_TRAIN, N_REP_TEST = 10000, 2000
LET = 'abcdefghijklmnopqrstuvwxyz'


def render_short(p):
    src, _ = render(p); B = op(p['d1'], p['o1'], p['d2']); A = op(p['k'], p['o2'], B)
    return src, ' ' + spaced('%s=%d , %s=%d' % (p['b'], B, p['a'], A)) + '\n'


def item(p, demos, fmt='full', **extra):
    R = render_short if fmt == 'short' else render
    src, tgt = R(p)
    prompt = ''.join(''.join(R(q)) for q in demos) + src
    base = len(prompt) - len(src)
    cpos = {'eq': base + src.index(';') - 2, 'semi': base + src.index(';'), 'last': len(prompt) - 1}
    full = prompt + tgt; b = p['b']
    occ = [i for i in range(len(prompt), len(full)) if full.startswith(b + ' = ', i)]
    cB = occ[0 if fmt == 'short' else 1] + len(b + ' = ')  # the digit in the chain's step "b = B"
    B = op(p['d1'], p['o1'], p['d2']); A = op(p['k'], p['o2'], B)
    assert full[cB] == str(B), (full, cB)
    return dict(p, B=B, A=A, src=src, prompt=prompt, target=tgt, cpos=cpos, cB=cB, **extra)


def twin_pairs(k, o2):
    return [(d1, d2) for d1 in range(10) for d2 in range(d1 + 1)
            if all(valid(dict(k=k, o2=o2, d1=d1, o1=o1, d2=d2)) for o1 in '+-')]


def families(rng, fmt='full'):
    out = []; f = 0
    for k, o2 in KO:
        for _ in range(N_PER_KO):
            a, b = (str(x) for x in rng.choice(list(LET), 2, replace=False)); demos = [sample(rng) for _ in range(3)]
            for d1, d2 in twin_pairs(k, o2):
                for ver, o1 in (('plus', '+'), ('minus', '-')):
                    out.append(item(dict(a=a, b=b, k=k, o2=o2, d1=d1, o1=o1, d2=d2), demos, fmt=fmt, fam=f, ver=ver,
                                    pair=f * 100 + 10 * d1 + d2))
            f += 1
    return out


def replication(rng):
    triples = sorted({(d1, o1, d2) for d1 in range(10) for o1 in '+-' for d2 in range(10)
                      if 0 <= op(d1, o1, d2) <= 9})
    test = set(triples[i] for i in np.random.default_rng(2154).permutation(len(triples))[:len(triples) // 5])
    out, seen = [], set()
    for split, n in (('train', N_REP_TRAIN), ('test', N_REP_TEST)):
        c = 0
        while c < n:
            p = sample(rng)
            if ((p['d1'], p['o1'], p['d2']) in test) != (split == 'test') or render(p)[0] in seen:
                continue
            seen.add(render(p)[0]); out.append(item(p, [sample(rng) for _ in range(3)], split=split)); c += 1
    return out, sorted(test)


def sha16(items):
    return hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:16]


def build_short():
    return families(np.random.default_rng(2155), 'short')


def all_sha():
    fam, rep, _ = build()
    return sha16(fam + rep + build_short())


def build():
    fam = families(np.random.default_rng(2155))
    rep, test = replication(np.random.default_rng(2156))
    info = dict(n_fam=len(set(i['fam'] for i in fam)), n_fam_items=len(fam), n_rep=len(rep),
                pairs_per_ko={'%d%s' % ko: len(twin_pairs(*ko)) for ko in KO}, test_triples=[list(t) for t in test])
    return fam, rep, info


if __name__ == '__main__':
    fam, rep, info = build(); s = all_sha()
    print(json.dumps(info)); print('ITEMS_SHA', s, '(frozen: %s)' % ITEMS_SHA)
    if ITEMS_SHA and s != ITEMS_SHA:
        sys.exit('items hash differs from the frozen one')
