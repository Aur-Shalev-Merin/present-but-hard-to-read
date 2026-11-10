"""v234 items (docs/prereg_v234_kudo_equal_answer.md): Kudo et al.'s Implicit format as in v216 ("a=k+b, b=d1+d2 ; a?",
target " a = A", three demonstrations in the same format), but k varies within a family so that the answer does not fix
b, and the clamp pairs share the answer.

A family fixes the letters, the three demonstrations and o2, and crosses k in KSET[o2] ({0, 1, 2} for +, {9, 8, 7} for -)
with every (d1, d2), d2 <= d1, d1 + d2 <= 7 (so B <= 7 and every k gives a one-digit answer for both o1), and o1 in
{+, -} (versions plus / minus): 3 x 20 x 2 = 120 items. 60 families per o2, 120 in all (14,400 items).
Fields as v216 (fam, ver, letters, numbers, B, A, src, prompt, target, cpos, cA).
Clamp pairs (built after extraction, from correct items): x and x' in the same family and version, same d1, different d2,
equal answer A, so k' = k o2-inverse (B - B'); the prediction under a used b is P = k o2 B' (x's k, x''s b).

  python v234_items.py      -> counts, pairs available, sha16
"""
import json, sys
import numpy as np

import v215_items as V5
from v215_pilot import op, sample
from v216_items import item

ITEMS_SHA = 'ac4121c22f88fe31'   # set at the freeze (2026-11-08)
KSET = {'+': (0, 1, 2), '-': (9, 8, 7)}
N_PER_O2 = 60
DPAIRS = [(d1, d2) for d1 in range(8) for d2 in range(d1 + 1) if d1 + d2 <= 7]


def build():
    rng = np.random.default_rng(2340); out = []; f = 0
    for o2 in ('+', '-'):
        for _ in range(N_PER_O2):
            a, b = (str(x) for x in rng.choice(list(V5.LET), 2, replace=False)); demos = [sample(rng) for _ in range(3)]
            for k in KSET[o2]:
                for d1, d2 in DPAIRS:
                    for ver, o1 in (('plus', '+'), ('minus', '-')):
                        out.append(item(dict(a=a, b=b, k=k, o2=o2, d1=d1, o1=o1, d2=d2), demos, fam=f, ver=ver))
            f += 1
    for i in out:
        assert 0 <= i['B'] <= 9 and 0 <= i['A'] <= 9
    return out


def partners(its, ok=None):
    """For each item index, the indices of its equal-answer partners (same family, version, d1; different d2)."""
    ok = np.ones(len(its), bool) if ok is None else np.asarray(ok)
    key = {}
    for j, i in enumerate(its):
        if ok[j]:
            key.setdefault((i['fam'], i['ver'], i['d1'], i['A']), []).append(j)
    return {j: [m for m in key.get((i['fam'], i['ver'], i['d1'], i['A']), []) if its[m]['d2'] != i['d2']]
            for j, i in enumerate(its) if ok[j]}


def prediction(x, x2):
    return op(x['k'], x['o2'], x2['B'])


if __name__ == '__main__':
    its = build(); s = V5.sha16(its); al = partners(its)
    n_with = sum(1 for v in al.values() if v)
    for j, v in al.items():
        for m in v:
            x, y = its[j], its[m]
            assert x['A'] == y['A'] and x['B'] != y['B'] and x['k'] != y['k'] and 0 <= prediction(x, y) <= 9 and prediction(x, y) != x['A']
    print(json.dumps(dict(n=len(its), n_fam=len(set(i['fam'] for i in its)), items_with_partner=n_with,
                          mean_partners=float(np.mean([len(v) for v in al.values() if v])))))
    print('ITEMS_SHA', s, '(frozen: %s)' % (ITEMS_SHA or 'not yet'))
    if ITEMS_SHA and s != ITEMS_SHA:
        sys.exit('items hash differs from the frozen one')
