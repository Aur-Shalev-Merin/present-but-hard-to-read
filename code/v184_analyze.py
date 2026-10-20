"""v184 analysis (irrelevant-number control; exploratory). Rules fixed before the data:

  python v184_analyze.py <dir containing v184_states.npy and v184_meta.json (searched recursively)>

Family types: irr_start, irr_end (vary a number in an added unrelated sentence), rel_start, rel_end (vary a relevant
number on the same contexts). Per type:
  Q1  within-family / between-family spread at the last token, per state;
  Q2  contrastive (family-demeaned) probe for the varied digit at qmark and last, and the standard probe at last
      (v178 probe_predict, 5-fold grouped by template; chance 0.10; READS >= 0.30);
  ANS fraction of families in which Ouro's top first-answer token at the last prompt token differs across the 10
      variants, per loop (does the digit change the answer it starts to give?).
Transfer: contrastive probe trained on rel families (training-fold templates), tested on irr families (held-out
templates), same insertion position; fixed C = 0.1, PCA 256.
Interpretation map:
  irr READS at the last token at some loop end (either position) -> copied numbers reach the answer position whether
      or not they matter (v181's trace is not only the answer in progress);
  irr does not read but rel does -> the v181 trace follows the answer in progress;
  irr ANS > 0.10 at loop 4 -> the "irrelevant" number changes Ouro's answer, so it is not irrelevant to the model (flag).
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V

G = {}
TYPES = ['irr_start', 'irr_end', 'rel_start', 'rel_end']


def init(root):
    f = sorted(glob.glob(os.path.join(root, '**', 'v184_states.npy'), recursive=True))[0]; d = os.path.dirname(f)
    G['meta'] = json.load(open(os.path.join(d, 'v184_meta.json'))); G['X'] = np.load(f, mmap_mode='r')
    its = G['meta']['items']
    G['type'] = np.array(['%s_%s' % (i['kind'], i['where']) for i in its])
    G['fam'] = np.array([i['fam'] for i in its]); G['y'] = np.array([i['digit'] for i in its])
    G['tid'] = np.array([i['tid'] for i in its])


def feats(idx, si, st, contrastive=True):
    X = np.asarray(G['X'][idx, si, st], dtype=np.float32); fam = G['fam'][idx]
    if contrastive:
        for f in np.unique(fam):
            X[fam == f] -= X[fam == f].mean(0)
    return X


def job(args):
    kind, si, st, contrastive = args
    if kind.startswith('transfer'):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        where = kind.split('_')[1]
        tr_i = np.where(G['type'] == 'rel_' + where)[0]; te_i = np.where(G['type'] == 'irr_' + where)[0]
        Xtr, Xte = feats(tr_i, si, st), feats(te_i, si, st)
        fold_tr = V.folds(G['tid'][tr_i]); u = sorted(set(G['tid'][tr_i]))
        f_of = {t: fold_tr[list(G['tid'][tr_i]).index(t)] for t in u}
        fold_te = np.array([f_of.get(t, -1) for t in G['tid'][te_i]]); pred = np.full(len(te_i), -1)
        for f in np.unique(fold_tr):
            a, b = fold_tr != f, fold_te == f
            if not b.any():
                continue
            m = make_pipeline(StandardScaler(), PCA(min(256, a.sum() - 1, Xtr.shape[1]), random_state=0),
                              LogisticRegression(C=0.1, max_iter=500)).fit(Xtr[a], G['y'][tr_i][a])
            pred[b] = m.predict(Xte[b])
        ok = pred >= 0
        return args, float((pred[ok] == G['y'][te_i][ok]).mean())
    idx = np.where(G['type'] == kind)[0]; y = G['y'][idx]; tids = G['tid'][idx]
    return args, float((V.probe_predict(feats(idx, si, st, contrastive), y, tids, V.folds(tids)) == y).mean())


def main(root, procs=4):
    init(root); meta = G['meta']; its = meta['items']; slots, states = meta['slots'], meta['states']
    print('items %d, families %d, minimal pairs ok %d; check |logit diff| %s' % (
        len(its), len(set(G['fam'])), sum(i['pair_ok'] for i in its), meta['check_logit']))
    print('\nANS: share of families whose top first-answer token changes across the 10 variants (loops 1-4)')
    for kind in TYPES:
        fams = sorted({i['fam'] for i in its if '%s_%s' % (i['kind'], i['where']) == kind}); row = []
        for t in range(4):
            row.append(np.mean([len({i['top'][t] for i in its if i['fam'] == f}) > 1 for f in fams]))
        print('  %-10s %s' % (kind, ' '.join('%.3f' % r for r in row)))
    print('\nQ1 within/between spread at the last token')
    li = slots.index('last')
    for kind in TYPES:
        idx = np.where(G['type'] == kind)[0]; fam = G['fam'][idx]; row = []
        for st in range(1, len(states)):
            X = feats(idx, li, st, contrastive=False); us = np.unique(fam)
            means = np.stack([X[fam == f].mean(0) for f in us])
            w = np.mean([np.linalg.norm(X[fam == f] - means[k], axis=1).mean() for k, f in enumerate(us)])
            row.append(w / np.linalg.norm(means - means.mean(0), axis=1).mean())
        print('  %-10s %s' % (kind, ' '.join('%s %.4f' % (s, r) for s, r in zip(states[1:], row))))
    jobs = [(kind, slots.index(sl), st, True) for kind in TYPES for sl in ('qmark', 'last') for st in range(1, len(states))]
    jobs += [(kind, li, st, False) for kind in TYPES for st in range(1, len(states))]
    jobs += [('transfer_' + w, li, st, True) for w in ('start', 'end') for st in range(1, len(states))]
    jobs += [(kind, slots.index('+0'), 0, False) for kind in TYPES]
    with Pool(procs, initializer=init, initargs=(root,)) as pool:
        res = dict(pool.map(job, jobs))
    print('\napparatus: standard probe at +0, h0: %s' % ' '.join(
        '%s %.3f' % (k, res[(k, slots.index('+0'), 0, False)]) for k in TYPES))
    print('\nQ2 accuracy for the varied digit (chance 0.10, READS >= 0.30)')
    print('%-24s %s' % ('', ' '.join('%6s' % s for s in states[1:])))
    reads = {}
    for kind in TYPES:
        for nm, sl, c in (('contrastive @ last', 'last', True), ('contrastive @ qmark', 'qmark', True),
                          ('standard @ last', 'last', False)):
            row = [res[(kind, slots.index(sl), st, c)] for st in range(1, len(states))]
            print('%-10s %-20s %s' % (kind, nm, ' '.join('%6.3f' % a for a in row)))
            if nm == 'contrastive @ last':
                reads[kind] = max(row[1:]) >= 0.30
    for w in ('start', 'end'):
        row = [res[('transfer_' + w, li, st, True)] for st in range(1, len(states))]
        print('%-10s %-20s %s' % ('rel->irr', 'transfer (%s) @ last' % w, ' '.join('%6.3f' % a for a in row)))
    print('\nirr READS at the last token (start / end): %s / %s' % (reads['irr_start'], reads['irr_end']))
    print('rel READS at the last token (start / end): %s / %s' % (reads['rel_start'], reads['rel_end']))


if __name__ == '__main__':
    main(sys.argv[1])
