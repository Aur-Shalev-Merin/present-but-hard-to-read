"""v199 pilot analysis (pre-freeze; CONTROL LABELS ONLY, as allowed by the pre-registration discipline).

Pilot set: v194's 720 pilot items (12 templates x 2 versions x 1 family x 30 combinations; other names and tens digits).
Reports per version: answer-only accuracy by loop, and the copied digits a, b, c at Sb (every state) and at Sc (loop
ends) with the minimal-pair probe (gate G2 risk). No target label (first step, answer) is computed.

  python v199_pilot.py <dir with v199_states.npy, v199_meta.json>
"""
import glob, json, os, sys
import numpy as np

import v178_analyze as V
import v186_analyze as A

f = sorted(glob.glob(os.path.join(sys.argv[1], '**', 'v199_states.npy'), recursive=True))[0]
meta = json.load(open(os.path.join(os.path.dirname(f), 'v199_meta.json'))); X = np.load(f, mmap_mode='r')
its = meta['items']; S = meta['states']
print('pilot items %d' % len(its))
for op in ('plus', 'minus'):
    idx = np.array([k for k, i in enumerate(its) if i['op'] == op])
    print('%-5s answer-only accuracy by loop %s' % (op, np.round(np.mean([its[k]['correct'] for k in idx], 0), 3).tolist()))
    fam = np.array([its[k]['fam'] for k in idx]); tid = np.array([its[k]['tid'] for k in idx])
    for pos, sts in ((0, range(len(S))), (1, [S.index('end%d' % t) for t in range(1, 5)])):
        for lab in ('a', 'b', 'c'):
            y = np.array([its[k]['labels'][lab] for k in idx])
            accs = [float((V.probe_predict(A.demean(np.asarray(X[idx, pos, st], dtype=np.float32), fam), y, tid,
                                           V.folds(tid)) == y).mean()) for st in sts]
            print('%-5s copy %s at %s: %s | best %.3f' % (op, lab, ('Sb', 'Sc')[pos], ' '.join(
                '%s %.3f' % (S[st], a) for st, a in zip(sts, accs)), max(accs)), flush=True)
