"""v224 items (docs/prereg_v224_share_swap.md). Two fresh positive-control sets, Kudo et al.'s Simple chain format
(target " b = B , a = A", teacher forced):

  build_b()  the b step: v222's items unchanged (v215's 240 families, 12,000 prompts; positions eq, last, letb, eqb,
             inb, digB, inA). Operands of the step: u = d1, v = d2; version = b's operation (o1).
  build_a()  the a step, new: 200 families (seed 2241), each fixing the letters, three demonstrations and b's equation
             type (even families: b = 0 + d2; odd: b = d1 - d2 with d1 in 4..9), and varying (k, B) over every pair
             with B in 0..4 and B <= k <= 9 - B (30 pairs), each in two versions that differ only in a's operation
             (o2 = + / -), so a = k + B or k - B is always a digit. Operands of the step: u = k, v = B (B is written in
             the chain, so it is a copy at the a step); version = o2. Positions: eq, last, digB, eqa (the "=" after a;
             A two tokens ahead), inA (the token before A's digit). 12,000 prompts.

  python v224_items.py      -> counts and sha16 of both sets
"""
import json, sys
import numpy as np

import v215_items as V5
import v222_items as V2
from v215_pilot import sample, LET

ITEMS_SHA_A = '670ce4ceed4c2ae1'   # frozen 2026-10-27
N_FAM_A = 200


def build_b():
    return V2.build()


def build_a():
    rng = np.random.default_rng(2241); out = []
    for f in range(N_FAM_A):
        a, b = (str(x) for x in rng.choice(list(LET), 2, replace=False)); demos = [sample(rng) for _ in range(3)]
        o1, d1 = ('+', 0) if f % 2 == 0 else ('-', int(rng.integers(4, 10)))
        for B in range(5):
            d2 = B if o1 == '+' else d1 - B
            for k in range(B, 10 - B):
                for ver, o2 in (('plus', '+'), ('minus', '-')):
                    it = V5.item(dict(a=a, b=b, k=k, o2=o2, d1=d1, o1=o1, d2=d2), demos, fmt='short', fam=f, ver=ver,
                                 pair=f * 100 + 10 * k + B)
                    full = it['prompt'] + it['target']; c = it['cB']
                    cA = full.index('%s = ' % a, c) + len(a + ' = ')
                    assert it['B'] == B and full[c:cA + 1] == '%d , %s = %d' % (B, a, it['A']), full
                    out.append(dict(it, cA=cA, cpos=dict(it['cpos'], digB=c, eqa=cA - 2, inA=cA - 1)))
    return out


if __name__ == '__main__':
    ib, ia = build_b(), build_a()
    print(json.dumps(dict(n_b=len(ib), n_a=len(ia), fam_a=len(set(i['fam'] for i in ia)), example_a=ia[0]['target'],
                          A_range=[min(i['A'] for i in ia), max(i['A'] for i in ia)])))
    print('b set sha %s (v222 frozen %s)' % (V5.sha16(ib), V2.ITEMS_SHA))
    s = V5.sha16(ia); print('ITEMS_SHA_A', s, '(frozen: %s)' % ITEMS_SHA_A)
    if ITEMS_SHA_A and s != ITEMS_SHA_A:
        sys.exit('items hash differs from the frozen one')
