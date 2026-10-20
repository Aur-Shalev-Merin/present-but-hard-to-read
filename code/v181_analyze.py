"""v181 analysis (minimal pairs; exploratory). Rules fixed before the data:

  python v181_analyze.py <dir with v181_states.npy, v181_meta.json>

Q1 (does the state move with the digit at all?): per slot x state,
  ratio = mean over families of the within-family spread (mean ||x_d - family mean||) / the between-family spread (mean
  ||family mean - grand mean||).  h0 at later slots is exactly 0 (same token); +0 is the varied token itself.
Q2 (can it be read?): v178's probe_predict, 5-fold CV grouped by template, label = the varied digit (balanced, chance 0.10)
  standard     = raw states;
  contrastive  = states minus their family mean (the family's 10 variants) -> context cancels; a readout that needs the
                 other variants of the same question.
  READS if accuracy >= 0.30.
Interpretation map:
  later slots: Q1 ratio ~ 0 and contrastive < 0.30  -> the digit is (near) absent there: values stay on their token;
  contrastive READS but standard does not          -> present but swamped by context: the minimal-pair readout works;
  Q1 ratio clearly > 0 but contrastive < 0.30       -> it moves the state, but not in a digit code shared across questions.
"""
import json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V

G = {}


def init(root):
    G['meta'] = json.load(open(os.path.join(root, 'v181_meta.json')))
    G['X'] = np.load(os.path.join(root, 'v181_states.npy'), mmap_mode='r')


def feats(si, st, contrastive):
    its = G['meta']['items']; X = np.asarray(G['X'][:, si, st], dtype=np.float32)
    if contrastive:
        fam = np.array([i['fam'] for i in its])
        for f in np.unique(fam):
            X[fam == f] -= X[fam == f].mean(0)
    return X


def job(args):
    si, st, contrastive = args; its = G['meta']['items']
    y = np.array([i['digit'] for i in its]); tids = np.array([i['tid'] for i in its])
    return args, float((V.probe_predict(feats(si, st, contrastive), y, tids, V.folds(tids)) == y).mean())


def main(root, procs=4):
    init(root); meta = G['meta']; its = meta['items']; slots, states = meta['slots'], meta['states']
    fam = np.array([i['fam'] for i in its])
    print('items %d, families %d, templates %d, minimal pairs ok %d; check |logit diff| %s' % (
        len(its), len(set(fam)), len({i['tid'] for i in its}), sum(i['pair_ok'] for i in its), meta['check_logit']))
    print('\nQ1 within-family spread / between-family spread')
    print('%-6s %s' % ('slot', ' '.join('%7s' % s for s in states)))
    for si, sl in enumerate(slots):
        row = []
        for st in range(len(states)):
            X = np.asarray(G['X'][:, si, st], dtype=np.float32)
            means = np.stack([X[fam == f].mean(0) for f in np.unique(fam)])
            w = np.mean([np.linalg.norm(X[fam == f] - means[k], axis=1).mean() for k, f in enumerate(np.unique(fam))])
            bsp = np.linalg.norm(means - means.mean(0), axis=1).mean()
            row.append(w / bsp)
        print('%-6s %s' % (sl, ' '.join('%7.4f' % r for r in row)), flush=True)
    jobs = [(si, st, c) for si in range(len(slots)) for st in range(len(states)) for c in (False, True)
            if not (st == 0 and si > 0)]
    with Pool(procs, initializer=init, initargs=(root,)) as pool:
        res = dict(pool.map(job, jobs))
    print('\nQ2 probe accuracy for the varied digit (chance 0.10, READS >= 0.30); standard / contrastive')
    print('%-6s %s' % ('slot', ' '.join('%13s' % s for s in states)))
    reads = {'standard': [], 'contrastive': []}
    for si, sl in enumerate(slots):
        cells = []
        for st in range(len(states)):
            if (si, st, False) not in res:
                cells.append('%13s' % '(same token)'); continue
            a, c = res[(si, st, False)], res[(si, st, True)]
            cells.append('  %.3f / %.3f' % (a, c))
            for nm, v in (('standard', a), ('contrastive', c)):
                if v >= 0.30 and si > 0:
                    reads[nm].append((sl, states[st]))
        print('%-6s %s' % (sl, ' '.join(cells)), flush=True)
    print('\nREADS at later slots, standard   : %s' % (reads['standard'] or 'none'))
    print('READS at later slots, contrastive: %s' % (reads['contrastive'] or 'none'))


if __name__ == '__main__':
    main(sys.argv[1])
