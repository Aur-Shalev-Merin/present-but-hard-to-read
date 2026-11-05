"""v222 items (docs/prereg_v222_clamp_distance.md): v221's items, unchanged (v215's 240 families in Kudo et al.'s Simple
chain format, target " b = B , a = A"), with the positions this study needs:
  digB  the token of the written b digit (B is stored there; the model reads it later to compute a)
  inA   the token just before the a digit (where A is predicted)
cA is the char index of the a digit in prompt + target.

  python v222_items.py      -> counts and sha16
"""
import json, sys

import v215_items as V5
import v221_items as V1

ITEMS_SHA = 'f917c3b7da3ddc87'   # frozen 2026-10-26


def build():
    out = []
    for it in V1.build():
        full = it['prompt'] + it['target']
        c = it['cB']; cA = full.index('%s = ' % it['a'], c) + len(it['a'] + ' = ')
        assert full[c:cA + 1] == '%d , %s = %d' % (it['B'], it['a'], it['A']), full
        out.append(dict(it, cA=cA, cpos=dict(it['cpos'], digB=c, inA=cA - 1)))
    return out


if __name__ == '__main__':
    its = build(); s = V5.sha16(its)
    print(json.dumps(dict(n=len(its), n_fam=len(set(i['fam'] for i in its)), example=its[0]['target'])))
    print('ITEMS_SHA', s, '(frozen: %s)' % ITEMS_SHA)
    if ITEMS_SHA and s != ITEMS_SHA:
        sys.exit('items hash differs from the frozen one')
