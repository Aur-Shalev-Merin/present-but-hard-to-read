"""v188 sign-flip contrasts (post hoc; suggested by a reviewer; the last analysis of the saved v186/v188 states).

For each pair of operands, the sign the problem needs against its flipped twin, each as excess over its own copy ceiling:
a+b vs a-b, a-c vs a+c, b-c vs b+c, and on the answer line answer+c (= a+b) vs answer-c (= a+b-2c). A generic mixture of
two operands has no reason to prefer the needed sign. Family- and template-level 97.5% CIs, end4 and end3, frozen ceiling.

  python v188_sign_check.py results_v188/line/v188_line_items_188.npz results_v188/line/v188_line_items_186.npz
"""
import sys
import numpy as np
import v188_line_check as L

PAIRS = [('a+b', (1, 1, 0), 'a-b', (1, 4, 0)), ('a-c', (1, 0, 4), 'a+c', (1, 0, 1)), ('b-c', (0, 1, 4), 'b+c', (0, 1, 1)),
         ('ans+c (= a+b)', (1, 1, 0), 'ans-c', (1, 1, 3))]

for path in sys.argv[1:]:
    z = np.load(path); v = int(z['study']); fam, tid = z['fam'], z['tid']
    fi = {tuple(int(x) for x in f): i for i, f in enumerate(z['forms'])}
    for si, st in enumerate(('end4', 'end3')):
        ex = z['state'][si].astype(float) - z['ceil'][0]
        for need, fn, flip, ff in PAIRS:
            d = ex[fi[fn]] - ex[fi[ff]]; fc = L.cluster_ci(fam, d); tc = L.cluster_ci(tid, d)
            pos = sum(d[tid == t].mean() > 0 for t in np.unique(tid))
            print('v%d %s | %-14s %+.3f vs %-6s %+.3f | difference %+.3f family [%+.3f, %+.3f] template [%+.3f, %+.3f] | '
                  'templates positive %d/%d' % (v, st, need, ex[fi[fn]].mean(), flip, ex[fi[ff]].mean(), fc[0], fc[1], fc[2],
                                                tc[1], tc[2], pos, len(np.unique(tid))))
