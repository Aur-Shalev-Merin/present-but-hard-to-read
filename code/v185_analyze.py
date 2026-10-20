"""v185 analysis (word control; exploratory). Derived from v184_analyze.py. Rules fixed before the data:

  python v185_analyze.py <dir containing v185_states.npy and v185_meta.json (searched recursively)>

Family types: col_start, col_end (vary a colour word in the unrelated sentence), dig_start, dig_end (vary its digit), on
identical contexts. Per type: Q1 within/between spread at the last token; Q2 contrastive probe at qmark and last and
standard probe at last (chance 0.10; READS >= 0.30); ANS as in v184.
Interpretation map (last token, mean over loop ends end1-end4, both insertion points pooled):
  colour READS and its contrastive accuracy is within 0.10 of the digit's -> the trace is not specific to numbers:
      the last token keeps a faint copy of individual earlier tokens (words and numbers alike);
  colour does not READ, or is more than 0.10 below the digit -> numbers are carried preferentially.
Amendment A1 (2026-10-11, committed after the GPU extraction was pushed but before any output was downloaded or any
analysis run; suggested by an outside review: probe accuracy near ceiling cannot show a difference in strength):
  SECONDARY, descriptive, not part of the verdict: r = Q1 spread(colour) / spread(digit) at the last token, mean of
  end1-end4 and both insertion points. If colour READS: r < 0.5 -> "numbers carried more strongly"; 0.5 <= r < 2/3 ->
  "intermediate"; 2/3 <= r <= 1.5 -> "comparable strength"; r > 1.5 -> "colour carried more strongly".
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V

G = {}
TYPES = ['col_start', 'col_end', 'dig_start', 'dig_end']


def init(root):
    f = sorted(glob.glob(os.path.join(root, '**', 'v185_states.npy'), recursive=True))[0]; d = os.path.dirname(f)
    G['meta'] = json.load(open(os.path.join(d, 'v185_meta.json'))); G['X'] = np.load(f, mmap_mode='r')
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
    li = slots.index('last'); spread = {}
    for kind in TYPES:
        idx = np.where(G['type'] == kind)[0]; fam = G['fam'][idx]; row = []
        for st in range(1, len(states)):
            X = feats(idx, li, st, contrastive=False); us = np.unique(fam)
            means = np.stack([X[fam == f].mean(0) for f in us])
            w = np.mean([np.linalg.norm(X[fam == f] - means[k], axis=1).mean() for k, f in enumerate(us)])
            row.append(w / np.linalg.norm(means - means.mean(0), axis=1).mean())
        spread[kind] = np.mean(row[1:])                    # end1-end4
        print('  %-10s %s' % (kind, ' '.join('%s %.4f' % (s, r) for s, r in zip(states[1:], row))))
    jobs = [(kind, slots.index(sl), st, True) for kind in TYPES for sl in ('qmark', 'last') for st in range(1, len(states))]
    jobs += [(kind, li, st, False) for kind in TYPES for st in range(1, len(states))]
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
    col = np.mean([res[(k, li, st, True)] for k in ('col_start', 'col_end') for st in range(2, len(states))])
    dig = np.mean([res[(k, li, st, True)] for k in ('dig_start', 'dig_end') for st in range(2, len(states))])
    print('\ncontrastive @ last, mean over end1-end4 and both positions: colour %.3f, digit %.3f' % (col, dig))
    print('colour READS (start / end): %s / %s' % (reads['col_start'], reads['col_end']))
    print('VERDICT: %s' % ('NOT NUMBER-SPECIFIC (colour within 0.10 of digit)' if (reads['col_start'] or reads['col_end'])
                           and col >= dig - 0.10 else 'NUMBERS PREFERRED'))
    # Amendment A1 (committed before any v185 output was seen): secondary, descriptive strength readout.
    r = np.mean([spread['col_start'], spread['col_end']]) / np.mean([spread['dig_start'], spread['dig_end']])
    strength = ('n/a (colour does not read)' if not (reads['col_start'] or reads['col_end']) else
                'numbers carried more strongly' if r < 0.5 else 'intermediate' if r < 2 / 3 else
                'comparable strength' if r <= 1.5 else 'colour carried more strongly')
    print('SECONDARY (A1): spread ratio colour/digit at the last token, end1-end4 = %.3f -> %s' % (r, strength))


if __name__ == '__main__':
    main(sys.argv[1])
