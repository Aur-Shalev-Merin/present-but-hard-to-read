"""v223 (EXPLORATORY, no labels): why does the operation swap fail where b is certainly computed (v221 inb)?

Hypothesis: the state codes the operand pair (d1, d2) jointly, so every function of the pair reads near ceiling and
the swap (S vs M accuracy) has no room. On v221's saved states, per version, at inb / eqb / eq / last and layers
16, 24, 28:
  accuracy  v215's minimal-pair probe (v178 probe on family-demeaned states, folds by family) and its copy ceiling,
            for S = (d1 + d2) mod 5, M = (d1 - d2) mod 5 and R0..R3, random 5-way functions of the pair (seed 2231)
  share     encoding view: family-demeaned states, copies regressed out (one-hot d1, d2, fitted on training folds),
            then the share of the held-out residual variance explained by the label's class means (fitted on training
            folds); for S, M, R0..R3 and B (0-9). A pair code gives S, M and R similar shares; an explicit code of B
            gives S (plus) or M (minus) the larger share.

  python v223_swapdiag.py <dir with v221_fam_states.npy>
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V

OUT = os.environ.get('OUT_DIR', '.')
LAYERS = list(range(0, 29, 4)); POSALL = ['eq', 'last', 'letb', 'eqb', 'inb']
POS = ['inb', 'eqb', 'eq', 'last']; LIS = [LAYERS.index(l) for l in (16, 24, 28)]
LABS = ['S', 'M', 'R0', 'R1', 'R2', 'R3']
G = {}


def demean(X, fam):
    X = np.array(X, dtype=np.float32)
    for f in np.unique(fam):
        X[fam == f] -= X[fam == f].mean(0)
    return X


def label_table(its):
    pairs = sorted({(i['d1'], i['d2']) for i in its}); rng = np.random.default_rng(2231)
    R = {}
    for r in range(4):
        perm = rng.permutation(len(pairs)); R['R%d' % r] = {pairs[j]: int(k % 5) for k, j in enumerate(perm)}
    out = {}
    for lab in LABS + ['B']:
        if lab == 'S':
            out[lab] = np.array([(i['d1'] + i['d2']) % 5 for i in its])
        elif lab == 'M':
            out[lab] = np.array([(i['d1'] - i['d2']) % 5 for i in its])
        elif lab == 'B':
            out[lab] = np.array([i['B'] for i in its])
        else:
            out[lab] = np.array([R[lab][(i['d1'], i['d2'])] for i in its])
    return out


def init(d):
    G.update(d)


def job(args):
    ver, p, li, lab, feat = args
    idx = G['idx_' + ver]; fam = G['fam'][idx]; y = G['y'][lab][idx]
    if feat == 'ceil':
        Z = np.zeros((len(idx), 20), np.float32); Z[np.arange(len(idx)), G['d1'][idx]] = 1; Z[np.arange(len(idx)), 10 + G['d2'][idx]] = 1
    else:
        Z = np.asarray(G['X'][idx, POSALL.index(p), li], np.float32)
    return args, float((V.probe_predict(demean(Z, fam), y, fam, V.folds(fam)) == y).mean())


def share(Z, fam, d1, d2, ys):
    """Held-out share of residual variance (after copies) explained by each label's class means; folds by family."""
    Z = demean(Z, fam); C = np.zeros((len(Z), 20), np.float32); C[np.arange(len(Z)), d1] = 1; C[np.arange(len(Z)), 10 + d2] = 1
    C = demean(C, fam); fold = V.folds(fam); num = {k: 0.0 for k in ys}; den = 0.0
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        W = np.linalg.lstsq(C[tr], Z[tr], rcond=None)[0]; Rtr = Z[tr] - C[tr] @ W; Rte = Z[te] - C[te] @ W
        den += float((Rte ** 2).sum())
        for k, y in ys.items():
            mu = {c: Rtr[y[tr] == c].mean(0) for c in np.unique(y[tr])}
            M = np.stack([mu.get(c, np.zeros(Z.shape[1], np.float32)) for c in y[te]])
            num[k] += float(((Rte - M) ** 2).sum())
    return {k: 1 - num[k] / den for k in ys}


def main(root, procs=4):
    import v215_items as V5, v221_items as I
    its = I.build(); s = V5.sha16(its); assert s == I.ITEMS_SHA, s
    f = glob.glob(os.path.join(root, '**', 'v221_fam_states.npy'), recursive=True)[0]
    X = np.load(f, mmap_mode='r'); y = label_table(its)
    fam = np.array([i['fam'] for i in its]); d1 = np.array([i['d1'] for i in its]); d2 = np.array([i['d2'] for i in its])
    idx = {v: np.array([k for k, i in enumerate(its) if i['ver'] == v]) for v in ('plus', 'minus')}
    for lab in LABS:
        print('label %s agrees with S on %.2f, with M on %.2f of items' % (lab, (y[lab] == y['S']).mean(), (y[lab] == y['M']).mean()), flush=True)
    jobs = [(v, p, li, lab, 'state') for v in idx for p in POS for li in LIS for lab in LABS]
    jobs += [(v, 'inb', 0, lab, 'ceil') for v in idx for lab in LABS]
    d = dict(X=X, y=y, fam=fam, d1=d1, d2=d2, idx_plus=idx['plus'], idx_minus=idx['minus'])
    with Pool(procs, initializer=init, initargs=(d,)) as pool:
        res = dict(pool.imap_unordered(job, jobs, chunksize=1))
    out = dict(acc={}, share={})
    print('copy ceilings: ' + ' | '.join('%s %s %.3f' % (v, lab, res[(v, 'inb', 0, lab, 'ceil')]) for v in idx for lab in LABS), flush=True)
    for p in POS:
        print('--- %s: accuracy (family-demeaned probe), layers 16 / 24 / 28 ---' % p, flush=True)
        for v in idx:
            print('  %-5s ' % v + ' | '.join('%s %s' % (lab, ' '.join('%.2f' % res[(v, p, li, lab, 'state')] for li in LIS)) for lab in LABS), flush=True)
            out['acc']['%s_%s' % (p, v)] = {lab: [res[(v, p, li, lab, 'state')] for li in LIS] for lab in LABS}
        print('--- %s: share of residual variance (after copies) explained by class means, layers 16 / 24 / 28 ---' % p, flush=True)
        for v in idx:
            sh = [share(np.asarray(X[idx[v], POSALL.index(p), li], np.float32), fam[idx[v]], d1[idx[v]], d2[idx[v]],
                        {k: y[k][idx[v]] for k in LABS + ['B']}) for li in LIS]
            print('  %-5s ' % v + ' | '.join('%s %s' % (k, ' '.join('%.3f' % s_[k] for s_ in sh)) for k in LABS + ['B']), flush=True)
            out['share']['%s_%s' % (p, v)] = {k: [s_[k] for s_ in sh] for k in LABS + ['B']}
    for k in ('ceil',):
        out[k] = {'%s_%s' % (v, lab): res[(v, 'inb', 0, lab, 'ceil')] for v in idx for lab in LABS}
    json.dump(out, open(os.path.join(OUT, 'v223_swapdiag.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
