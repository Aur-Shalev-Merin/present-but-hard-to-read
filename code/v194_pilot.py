"""v194 pilot analysis (pre-freeze; CONTROL LABELS ONLY, as allowed by the pre-registration discipline).

Pilot set: 12 templates x 2 versions x 1 family x 30 combinations (other names and tens than the main set).
Reports per version: answer-only accuracy by loop (gate G1 risk), tokenisation, and the copied digits a, b, c at P0 by
state with the minimal-pair probe (gate G2 risk). No target label (a+b, a-b) is computed.

  python v194_pilot.py <dir with v194_states.npy, v194_meta.json>
"""
import glob, json, os, sys
import numpy as np

import v178_analyze as V
import v186_analyze as A

f = sorted(glob.glob(os.path.join(sys.argv[1], '**', 'v194_states.npy'), recursive=True))[0]
meta = json.load(open(os.path.join(os.path.dirname(f), 'v194_meta.json'))); X = np.load(f, mmap_mode='r')
its = meta['items']; S = meta['states']
print('pilot items %d; tokenisation ok %d' % (len(its), sum(i['tok_ok'] for i in its)))
for op in ('plus', 'minus'):
    idx = np.array([k for k, i in enumerate(its) if i['op'] == op])
    acc = np.mean([its[k]['correct'] for k in idx], 0)
    print('%-5s answer-only accuracy by loop %s' % (op, np.round(acc, 3).tolist()))
    fam = np.array([its[k]['fam'] for k in idx]); tid = np.array([its[k]['tid'] for k in idx])
    for lab in ('a', 'b', 'c'):
        y = np.array([its[k]['labels'][lab] for k in idx])
        accs = [float((V.probe_predict(A.demean(np.asarray(X[idx, 0, st], dtype=np.float32), fam), y, tid,
                                       V.folds(tid)) == y).mean()) for st in range(len(S))]
        print('%-5s copy %s at P0 by state: %s | best %.3f' % (op, lab, ' '.join('%s %.3f' % (s, a) for s, a in zip(S, accs)),
                                                              max(accs)), flush=True)
