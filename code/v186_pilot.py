"""v186 PRE-FREEZE pilot analysis: control labels only (no intermediate or answer-digit label is read).

  python v186_pilot.py <dir with v186_states.npy, v186_meta.json from V186_SETS=pilot>

Reports: (1) answer-only accuracy by loop (the gate: >= 0.30 after loop 4); (2) family-demeaned probe accuracy for
the copied units digits a, b, c at P0 (last prompt token), every state (positive control: >= 0.90 after loop 4).
Probe = v178 probe_predict, 5-fold CV grouped by template.
"""
import json, os, sys
import numpy as np

import v178_analyze as V


def main(root):
    meta = json.load(open(os.path.join(root, 'v186_meta.json'))); X = np.load(os.path.join(root, 'v186_states.npy'))
    its = meta['items']; assert all(i['set'] == 'pilot' for i in its), 'pilot reads pilot items only'
    fam = np.array([i['fam'] for i in its]); tid = np.array([i['tid'] for i in its]); fold = V.folds(tid)
    acc = np.mean([i['correct'] for i in its], 0)
    print('pilot items %d, families %d, tokenisation ok %d; check %s' % (len(its), len(set(fam)), sum(i['tok_ok'] for i in its),
                                                                       meta['check_logit']))
    print('answer-only accuracy by loop: %s   (gate: loop 4 >= 0.30 -> %s)' % (np.round(acc, 3).tolist(), acc[3] >= 0.30))
    for t in sorted(set(tid)):
        print('  template %2d: %s' % (t, np.round(np.mean([i['correct'] for i in its if i['tid'] == t], 0), 2).tolist()))
    print('\ncopy controls, family-demeaned probe at P0 (positive control: end4 >= 0.90)')
    print('%-4s %s' % ('', ' '.join('%6s' % s for s in meta['states'])))
    for lab in ('a', 'b', 'c'):
        y = np.array([i['labels'][lab] for i in its]); row = []
        for st in range(len(meta['states'])):
            Z = X[:, 0, st].astype(np.float32)
            for f in np.unique(fam):
                Z[fam == f] -= Z[fam == f].mean(0)
            row.append((V.probe_predict(Z, y, tid, fold) == y).mean())
        print('%-4s %s' % (lab, ' '.join('%6.3f' % r for r in row)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
