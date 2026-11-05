"""v216 items (docs/prereg_v216_answer_only.md): v215's 240 families (same letters, numbers and demonstrations, same
seed) in Kudo et al.'s Implicit format: target " a = A", demonstrations in the same format.
Fields as v215 (fam, ver, pair, letters, numbers, B, A, src, prompt, target) plus cpos = char positions of eq (the d2
digit), last (the final '?') and pre (the space before the answer digit, in prompt + target), and cA = the answer digit.

  python v216_items.py      -> counts and sha16
"""
import json, sys
import numpy as np

import v215_items as V5
from v215_pilot import op, render, spaced, sample

ITEMS_SHA = '497db0bcb9748f30'   # frozen 2026-10-24


def render_implicit(p):
    src, _ = render(p); A = op(p['k'], p['o2'], op(p['d1'], p['o1'], p['d2']))
    return src, ' ' + spaced('%s=%d' % (p['a'], A)) + '\n'


def item(p, demos, **extra):
    src, tgt = render_implicit(p)
    prompt = ''.join(''.join(render_implicit(q)) for q in demos) + src
    base = len(prompt) - len(src); full = prompt + tgt
    B = op(p['d1'], p['o1'], p['d2']); A = op(p['k'], p['o2'], B)
    cA = len(prompt) + len(' ' + spaced(p['a'] + '=')) + 1
    assert full[cA] == str(A) and full[cA - 1] == ' ', (full, cA)
    cpos = {'eq': base + src.index(';') - 2, 'last': len(prompt) - 1, 'pre': cA - 1}
    return dict(p, B=B, A=A, src=src, prompt=prompt, target=tgt, cpos=cpos, cA=cA, **extra)


def build():
    rng = np.random.default_rng(2155); out = []; f = 0                  # v215's generator and seed
    for k, o2 in V5.KO:
        for _ in range(V5.N_PER_KO):
            a, b = (str(x) for x in rng.choice(list(V5.LET), 2, replace=False)); demos = [sample(rng) for _ in range(3)]
            for d1, d2 in V5.twin_pairs(k, o2):
                for ver, o1 in (('plus', '+'), ('minus', '-')):
                    out.append(item(dict(a=a, b=b, k=k, o2=o2, d1=d1, o1=o1, d2=d2), demos, fam=f, ver=ver,
                                    pair=f * 100 + 10 * d1 + d2))
            f += 1
    return out


if __name__ == '__main__':
    fam = build(); s = V5.sha16(fam)
    v5 = V5.build()[0]
    same = all(x['src'] == y['src'] and x['fam'] == y['fam'] for x, y in zip(fam, v5))
    print(json.dumps(dict(n=len(fam), n_fam=len(set(i['fam'] for i in fam)), same_problems_as_v215=same)))
    print('ITEMS_SHA', s, '(frozen: %s)' % ITEMS_SHA)
    if ITEMS_SHA and s != ITEMS_SHA:
        sys.exit('items hash differs from the frozen one')
