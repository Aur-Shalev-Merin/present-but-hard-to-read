"""v180 analysis (positive control; exploratory). Rules fixed before the data:

  python v180_analyze.py <dir with v180_states.npy, v180_meta.json>

Probe = v178's probe_predict (standardise -> PCA 256 -> logistic, C in {const, 0.01, 0.1, 1} by inner CV); 5-fold CV over
items (seeded; v178 folds() on item ids). Per op (add, mul) separately.
- Copy labels (units digit of a, of b) at qmark and last, every state.  READS if acc >= max(0.5, majority + 0.15).
- Result label (units digit of r) at qmark and last, every state, against an input baseline (same probe on one-hot
  units / tens / hundreds digits of a and b).  COMPUTED-READABLE if acc - baseline >= 0.10.
- Apparatus: a, b at their own +0 token at h0 >= 0.9.
Interpretation map:
  copies READ at last (some loop, either op) -> the probe can read operands at the last token in bare arithmetic, so
      v179's blindness in word problems is a property of word problems (claim 3 of the note stands as a contrast);
  copies never READ at last in either op -> v179's blindness may be the probe / this model at the last token (claim 3
      is cut or reduced to "this probe cannot").
"""
import json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V

G = {}


def init(root):
    meta = json.load(open(os.path.join(root, 'v180_meta.json')))
    G['meta'] = meta; G['X'] = np.load(os.path.join(root, 'v180_states.npy'), mmap_mode='r')


def digits(v):
    return [v % 10, v // 10 % 10, v // 100 % 10]


def job(args):
    op, lab, si, st = args
    its = G['meta']['items']; idx = np.array([i['k'] for i in its if i['op'] == op])
    y = np.array([its[k]['labels'][lab] for k in idx]); fold = V.folds(idx)
    if si == 'input':
        X = np.zeros((len(idx), 60))
        for r, k in enumerate(idx):
            for j, d in enumerate(digits(its[k]['a']) + digits(its[k]['b'])):
                X[r, 10 * j + d] = 1
    else:
        X = np.asarray(G['X'][idx, si, st], dtype=np.float32)
    return args, float((V.probe_predict(X, y, idx, fold) == y).mean())


def main(root, procs=4):
    init(root); meta = G['meta']; slots, states = meta['slots'], meta['states']; its = meta['items']
    print('items %d, tokenisation ok %d; check |logit diff| %s' % (len(its), sum(i['tok_ok'] for i in its), meta['check_logit']))
    for op in ('add', 'mul'):
        acc = np.mean([i['correct'] for i in its if i['op'] == op], 0)
        print('%s answer-only accuracy per loop: %s' % (op, np.round(acc, 3).tolist()))
    jobs = []
    for op in ('add', 'mul'):
        jobs += [(op, 'r', 'input', None)] + [(op, lab, slots.index(lab + '+0'), 0) for lab in ('a', 'b')]
        jobs += [(op, lab, slots.index(sl), st) for lab in ('a', 'b', 'r') for sl in ('qmark', 'last')
                 for st in range(len(states))]
    with Pool(procs, initializer=init, initargs=(root,)) as pool:
        res = dict(pool.map(job, jobs))
    copies_read_last = False; computed = []
    for op in ('add', 'mul'):
        sub = [i for i in its if i['op'] == op]; base = res[(op, 'r', 'input', None)]
        print('\n== %s | apparatus a+0 %.3f, b+0 %.3f at h0 | result-digit input baseline %.3f' % (
            op, res[(op, 'a', slots.index('a+0'), 0)], res[(op, 'b', slots.index('b+0'), 0)], base))
        print('%-14s %s' % ('label @ slot', ' '.join('%6s' % s for s in states)))
        for lab in ('a', 'b', 'r'):
            y = np.array([i['labels'][lab] for i in sub]); maj = np.bincount(y).max() / len(y); bar = max(0.5, maj + 0.15)
            for sl in ('qmark', 'last'):
                row = [res[(op, lab, slots.index(sl), st)] for st in range(len(states))]
                print('%-14s %s   (majority %.3f%s)' % ('%s @ %s' % (lab, sl), ' '.join('%6.3f' % a for a in row), maj,
                                                       ', bar %.3f' % bar if lab != 'r' else ', S bar +0.10'))
                if lab != 'r' and sl == 'last' and max(row[2:]) >= bar:
                    copies_read_last = True
                if lab == 'r':
                    computed += [(op, sl, states[st]) for st in range(len(states)) if row[st] - base >= 0.10]
    print('\ncopies READ at last token (some loop end, either op): %s' % copies_read_last)
    print('result digit COMPUTED-READABLE (S >= 0.10): %s' % (computed if computed else 'none'))


if __name__ == '__main__':
    main(sys.argv[1])
