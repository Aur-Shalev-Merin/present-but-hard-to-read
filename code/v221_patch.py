"""v221 Part B (docs/prereg_v221_positive_control.md): v215_patch unchanged (conditions none, cB, cBo, cD, cR, full;
hooks at the band layers at token p; outcome = the b digit predicted at the chain's "b =" step, i.e. at inb; primary
statistic hit(cBo) - hit(cR), seed 2153; USED >= 0.10 with CI lower > 0, NOT-USED CI upper < 0.02, UNSCORABLE
hit(full) < 0.30), run on v221's items so that p can be eq, eqb or inb. v215_patch takes its items from
v215_items.build_short; here that function returns v221's items (the same problems with the extra positions), and the
v215 items hash check is replaced by v221's.

  GPU:  OUT_DIR=... python v221_patch.py extract <dir with v215_plan.npz, v215_pairs.json from v221_analyze | SMOKE>
  CPU:  python v221_patch.py analyse <dir with v215_patch.json>
"""
import sys

import v215_items as V5
import v221_items as I
import v215_patch as P

if __name__ == '__main__':
    its = I.build(); s = V5.sha16(its)
    print('v221 items sha %s (frozen %s)' % (s, I.ITEMS_SHA), flush=True)
    if I.ITEMS_SHA and s != I.ITEMS_SHA:
        sys.exit('items hash differs')
    V5.build_short = lambda: its; V5.ITEMS_SHA = ''
    {'extract': P.extract, 'analyse': P.analyse}[sys.argv[1]](sys.argv[2])
