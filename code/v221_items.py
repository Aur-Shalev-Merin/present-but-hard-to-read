"""v221 items (docs/prereg_v221_positive_control.md): v215's 240 families in Kudo et al.'s Simple chain format (target
" b = B , a = A", the short-chain arm of v215; v215_items.build_short, unchanged), with three more positions in the
chain's step " b = B" (side review: at inb, B is the next token, so a positive result there is close to built in):
  letb  the letter b (B is three tokens ahead)
  eqb   the "=" after it (B is two tokens ahead)
  inb   the token just before the computed b digit (the space after "b ="; the model predicts B there)
cpos keeps v215's eq / semi / last.

  python v221_items.py      -> counts and sha16
"""
import json, sys

import v215_items as V5

ITEMS_SHA = '50a50828de872803'   # frozen 2026-10-26


def build():
    out = []
    for it in V5.build_short():
        full = it['prompt'] + it['target']
        c = it['cB']
        assert full[c - 4:c + 1] == '%s = %d' % (it['b'], it['B']), full
        out.append(dict(it, cpos=dict(it['cpos'], letb=c - 4, eqb=c - 2, inb=c - 1)))
    return out


if __name__ == '__main__':
    its = build(); s = V5.sha16(its)
    print(json.dumps(dict(n=len(its), n_fam=len(set(i['fam'] for i in its)), example=its[0]['target'])))
    print('ITEMS_SHA', s, '(frozen: %s)' % ITEMS_SHA)
    if ITEMS_SHA and s != ITEMS_SHA:
        sys.exit('items hash differs from the frozen one')
