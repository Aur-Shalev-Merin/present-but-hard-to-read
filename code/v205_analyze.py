"""v205 Part B analysis (docs/prereg_v205_swamping_robustness.md), Qwen2.5-1.5B, last prompt token, Q = l7, l14, l21, l28.

B1 random-perturbation null. For each kind (swap: 10 random word tokens at the digit's position; noise: 10 random
   embedding vectors added there), the same measurements as the digit families of v202 (same 137 contexts): overlap O
   (overlap_lda, unchanged), structured decoder, minimal-pair and standard probes. r = mean O(kind) / mean O(digit) over Q.
   Validity per kind: minimal-pair probe mean over Q >= 0.50 (the change reaches the last token), else UNINFORMATIVE.
   Descriptive, per kind and the digit: each O against its own nulls (labels shuffled within family; top-9 residual PCs),
   v205_saved.aniso.
     GENERIC        r >= 0.75 for both kinds (any change at that token lands in equally crowded directions)
     TOKEN-SPECIFIC r <= 0.50 for both kinds (the digit's directions are more crowded than a meaningless change's)
     MIXED          otherwise
B2 template-count curve. Every usable GSM8K training question (6,249), one prompt each with a random units digit; a fixed
   random 1,000 are the test set, the other 5,249 the training pool. Nested training subsets (one permutation, seed 205)
   of N = 60, 125, 250, 500, 1,000, 2,000, N_max/2, 4,000, N_max. Two probes, each scored on the same 1,000 test questions:
     pca   the v178 probe (StandardScaler -> PCA 256 -> logistic, C in {const, 0.01, 0.1, 1}), chosen C reported;
     full  StandardScaler -> logistic on all 1,536 dimensions, C in {const, 1e-4, 1e-3, 1e-2, 0.1, 1, 10}, chosen C reported.
   Effective rank of the training pool's states (participation ratio; components to 95% variance), so N can be stated
   as a multiple of it. Gate: the pca probe on the digit's own token (+0, l7) at N_max >= 0.90.
   level = max over Q and both probes of the accuracy at N_max; gain = max over Q and probes of acc(N_max) - acc(N_max/2).
     STAYS-SWAMPED  level <= 0.30 and gain <= 0.02
     CLIMBS         level >= 0.50 or gain >= 0.05
     PARTIAL        otherwise
   Descriptive: an MLP (v205_saved settings) at N_max.
B3 positive control for B2: minimal pairs on 150 of B2's test questions (10 digits each). The v178 probe, contrastive and
   standard, folds by question. If the contrastive mean over Q < 0.50, B2's label is reported as UNINFORMATIVE (the digit
   is not shown to be present in these prompts).

  python v205_analyze.py <dir with v205_b1_*, v205_b2_* and v202_v181_* files>
  python v205_analyze.py --selftest
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V
import v181_swamp_check as S

Q = ['l7', 'l14', 'l21', 'l28']
G = {}


def init(d):
    G.update(d)


def load(root, stem):
    f = max(glob.glob(os.path.join(root, '**', stem + '_states.npy'), recursive=True), key=os.path.getsize)
    meta = json.load(open(os.path.join(os.path.dirname(f), stem + '_meta.json')))
    print('states:', os.path.relpath(f, root), flush=True)
    return np.load(f, mmap_mode='r'), meta


def probe_job(args):
    key, st, contrastive = args
    X = np.asarray(G[key][:, st], dtype=np.float32); y, fam, tid = G[key + '_y'], G[key + '_fam'], G[key + '_tid']
    if contrastive:
        X = S.demean(X, fam)
    return args, float((V.probe_predict(X, y, tid, V.folds(tid)) == y).mean())


def fit_c(Xtr, ytr, gtr, Xte, pca, grid):
    """v178-style probe on one split; returns predictions and the chosen C."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    red = (make_pipeline(StandardScaler(), PCA(min(256, Xtr.shape[1], len(Xtr) - 1), random_state=0)) if pca
           else StandardScaler()).fit(Xtr)
    Ztr, Zte = red.transform(Xtr), red.transform(Xte); inner = V.folds(gtr, k=3, seed=1781); best, ba = None, -1
    for C in grid:
        acc = []
        for g in range(3):
            a, b = inner != g, inner == g
            if C == 'const':
                acc.append((ytr[b] == np.bincount(ytr[a], minlength=10).argmax()).mean())
            else:
                m = LogisticRegression(C=C, max_iter=300).fit(Ztr[a], ytr[a]); acc.append((m.predict(Ztr[b]) == ytr[b]).mean())
        if np.mean(acc) > ba:
            best, ba = C, np.mean(acc)
    if best == 'const':
        return np.full(len(Xte), np.bincount(ytr, minlength=10).argmax()), best
    return LogisticRegression(C=best, max_iter=300).fit(Ztr, ytr).predict(Zte), best


GRIDS = {'pca': ('const', 0.01, 0.1, 1.0), 'full': ('const', 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)}


def curve_job(args):
    sl, st, n, probe = args
    tr = G['order'][:n]; te = np.where(G['split'] == 'test')[0]; X = G['X'][:, sl, st]
    pred, C = fit_c(np.asarray(X[tr], np.float32), G['y'][tr], G['g'][tr], np.asarray(X[te], np.float32), probe == 'pca',
                    GRIDS[probe])
    return args, (float((pred == G['y'][te]).mean()), C)


def mlp_fit_predict(Xtr, ytr, gtr, Xte):
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    red = make_pipeline(StandardScaler(), PCA(min(256, Xtr.shape[1], len(Xtr) - 1), random_state=0)).fit(Xtr)
    Ztr, Zte = red.transform(Xtr), red.transform(Xte); inner = V.folds(gtr, k=3, seed=1781); best, ba = None, -1
    m = lambda a: MLPClassifier((256,), alpha=a, early_stopping=True, max_iter=300, random_state=0)
    for a in (1e-4, 1e-2, 1.0):
        acc = np.mean([(m(a).fit(Ztr[inner != g], ytr[inner != g]).predict(Ztr[inner == g]) == ytr[inner == g]).mean()
                       for g in range(3)])
        if acc > ba:
            best, ba = a, acc
    return m(best).fit(Ztr, ytr).predict(Zte)


def mlp_job(args):
    sl, st = args
    tr = G['order']; te = np.where(G['split'] == 'test')[0]; X = G['X'][:, sl, st]
    pred = mlp_fit_predict(np.asarray(X[tr], np.float32), G['y'][tr], G['g'][tr], np.asarray(X[te], np.float32))
    return args, float((pred == G['y'][te]).mean())


def b1(X1, m1, X0, m0, procs=4, say=print):
    states = m0['states']; si0 = m0['slots'].index('last'); si1 = m1['slots'].index('last')
    sts = [states.index(q) for q in Q]; arrs = {}
    its0 = m0['items']
    arrs.update(digit=X0[:, si0], digit_y=np.array([i['digit'] for i in its0]), digit_fam=np.array([i['fam'] for i in its0]),
                digit_tid=np.array([i['tid'] for i in its0]))
    its1 = m1['items']
    for kind in ('swap', 'noise'):
        idx = np.array([k for k, i in enumerate(its1) if i['kind'] == kind])
        arrs.update({kind: X1[idx][:, si1], kind + '_y': np.array([its1[k]['digit'] for k in idx]),
                     kind + '_fam': np.array([its1[k]['fam'] for k in idx]), kind + '_tid': np.array([its1[k]['tid'] for k in idx])})
    G.update(arrs)
    jobs = [(k, st, c) for k in ('digit', 'swap', 'noise') for st in sts for c in (False, True)]
    with Pool(procs, initializer=init, initargs=(arrs,)) as pool:
        res = dict(pool.map(probe_job, jobs))
    O = {}
    say('--- B1 random-perturbation null (last prompt token) ---')
    for k in ('digit', 'swap', 'noise'):
        for st in sts:
            r = S.overlap_lda(np.asarray(arrs[k][:, st], dtype=np.float64), arrs[k + '_y'], arrs[k + '_fam'], V.folds(arrs[k + '_tid']))
            O[(k, st)] = r
        say('%-6s %s' % (k, ' | '.join('%s O %.2f SNR %.4f decoder %.3f std %.3f mp %.3f' % (
            states[st], O[(k, st)]['O'], O[(k, st)]['snrU'], O[(k, st)]['lda'], res[(k, st, False)], res[(k, st, True)])
            for st in sts)))
    import v205_saved as SV
    for k in ('digit', 'swap', 'noise'):
        rr = [SV.aniso(np.asarray(arrs[k][:, st], dtype=np.float64), arrs[k + '_y'], arrs[k + '_fam'],
                       V.folds(arrs[k + '_tid'])) for st in sts]
        say('%-6s own-null check: %s' % (k, ' | '.join('%s O %.2f shuffled-label %.2f O/shuffled %.2f residual-PC %.2f' % (
            states[st], r['O'], r['shuf'], r['O'] / r['shuf'], r['resPC']) for st, r in zip(sts, rr))))
    od = np.mean([O[('digit', st)]['O'] for st in sts]); lab = {}
    for k in ('swap', 'noise'):
        mp = np.mean([res[(k, st, True)] for st in sts]); r = np.mean([O[(k, st)]['O'] for st in sts]) / od
        lab[k] = ('UNINFORMATIVE' if mp < 0.50 else 'G' if r >= 0.75 else 'T' if r <= 0.50 else 'M', r, mp)
        say('%s: minimal-pair mean %.3f, r = O(%s)/O(digit) = %.3f' % (k, mp, k, r))
    ls = [lab[k][0] for k in ('swap', 'noise')]
    out = ('UNINFORMATIVE' if 'UNINFORMATIVE' in ls else 'GENERIC' if ls == ['G', 'G'] else
           'TOKEN-SPECIFIC' if ls == ['T', 'T'] else 'MIXED')
    say('B1 -> %s' % out)
    return out


def b2(X2, m2, procs=4, say=print):
    its = m2['items']; states = m2['states']; si = m2['slots'].index('last')
    split = np.array([i['split'] for i in its]); trp = np.where(split == 'train')[0]
    order = trp[np.random.default_rng(205).permutation(len(trp))]; nmax = len(order)
    NS = sorted(set([n for n in (60, 125, 250, 500, 1000, 2000, 4000) if n < nmax] + [nmax // 2, nmax]))
    arrs = dict(X=np.asarray(X2), y=np.array([i['digit'] for i in its]), g=np.array([i['idx'] for i in its]), split=split,
                order=order)
    G.update(arrs); sts = [states.index(q) for q in Q]
    say('--- B2 template-count curve (%d training questions, %d test; chance 0.10) ---' % (nmax, (split == 'test').sum()))
    for st in sts:
        Xt = np.asarray(X2[order, si, st], dtype=np.float64); Xt -= Xt.mean(0)
        lam = np.linalg.svd(Xt, compute_uv=False) ** 2; c = np.cumsum(lam) / lam.sum()
        pr = lam.sum() ** 2 / (lam ** 2).sum(); n95 = int(np.searchsorted(c, 0.95) + 1)
        say('%-5s effective rank: participation ratio %.1f, components to 95%% variance %d -> N_max = %.1f x PR, %.1f x n95'
            % (states[st], pr, n95, nmax / pr, nmax / n95))
    jobs = [(si, st, n, p) for st in sts for n in NS for p in ('pca', 'full')] + [(0, states.index('l7'), nmax, 'pca')]
    with Pool(procs, initializer=init, initargs=(arrs,)) as pool:
        res = dict(pool.map(curve_job, jobs))
        mres = dict(pool.map(mlp_job, [(si, st) for st in sts]))
    say('N           ' + ' '.join('%15d' % n for n in NS))
    for st in sts:
        for p in ('pca', 'full'):
            say('%-5s %-5s ' % (states[st], p) + ' '.join('%6.3f (C %-6s)' % (res[(si, st, n, p)][0], res[(si, st, n, p)][1])
                                                     for n in NS))
    say('MLP at N_max: %s' % ' '.join('%s %.3f' % (states[st], mres[(si, st)]) for st in sts))
    g = res[(0, states.index('l7'), nmax, 'pca')][0]
    level = max(res[(si, st, nmax, p)][0] for st in sts for p in ('pca', 'full'))
    gain = max(res[(si, st, nmax, p)][0] - res[(si, st, nmax // 2, p)][0] for st in sts for p in ('pca', 'full'))
    out = ('UNSCORABLE' if g < 0.90 else 'CLIMBS' if level >= 0.50 or gain >= 0.05 else
           'STAYS-SWAMPED' if level <= 0.30 and gain <= 0.02 else 'PARTIAL')
    say("gate: pca probe on the digit's own token (+0, l7) at N_max: %.3f (>= 0.90)" % g)
    say('B2: level %.3f, gain over the last doubling %+.3f -> %s' % (level, gain, out))
    return out


def b3(X3, m3, procs=4, say=print):
    its = m3['items']; states = m3['states']; si = m3['slots'].index('last'); sts = [states.index(q) for q in Q]
    arrs = {'c': np.asarray(X3[:, si]), 'c_y': np.array([i['digit'] for i in its]), 'c_fam': np.array([i['idx'] for i in its]),
            'c_tid': np.array([i['idx'] for i in its])}
    G.update(arrs)
    with Pool(procs, initializer=init, initargs=(arrs,)) as pool:
        res = dict(pool.map(probe_job, [('c', st, c) for st in sts for c in (False, True)]))
    mp = float(np.mean([res[('c', st, True)] for st in sts]))
    say('--- B3 positive control: minimal pairs on %d B2 test questions ---' % len(set(arrs['c_fam'])))
    say(' '.join('%s standard %.3f / minimal-pair %.3f' % (states[st], res[('c', st, False)], res[('c', st, True)]) for st in sts))
    say('B3: minimal-pair mean over Q %.3f -> %s' % (mp, 'present' if mp >= 0.50 else 'NOT SHOWN PRESENT (B2 uninformative)'))
    return mp >= 0.50


def selftest():
    """Synthetic B2: readable world (digit code large vs context) -> CLIMBS; swamped world -> STAYS-SWAMPED."""
    rng = np.random.default_rng(0); d = 64; code = rng.standard_normal((10, d))
    for world, amp in (('readable', 2.0), ('swamped', 0.02)):
        n = 2400; y = rng.integers(0, 10, n)
        X = np.zeros((n, 3, 6, d), np.float32)
        ctx = rng.standard_normal((n, d)) * 3
        for st in range(6):
            X[:, 2, st] = ctx + amp * code[y]
            X[:, 0, st] = ctx * 0.1 + 3 * code[y]
        its = [dict(digit=int(y[k]), idx=k, split='train' if k < 1900 else 'test') for k in range(n)]
        m = dict(items=its, states=['h0', 'l4', 'l7', 'l14', 'l21', 'l28'], slots=['+0', 'qmark', 'last'])
        out = []
        print('selftest %-9s -> %s' % (world, b2(X, m, say=out.append)))


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        selftest()
    else:
        root = sys.argv[1]
        X1, m1 = load(root, 'v205_b1'); X0, m0 = load(root, 'v202_v181'); b1(X1, m1, X0, m0)
        parts = [load(root, 'v205_b2_s%d' % k) for k in range(3)]
        X2 = np.concatenate([np.asarray(p[0]) for p in parts]); m2 = dict(parts[0][1])
        m2['items'] = sum([p[1]['items'] for p in parts], [])
        lab2 = b2(X2, m2)
        X3, m3 = load(root, 'v205_b3'); ok3 = b3(X3, m3)
        print('B2 final -> %s' % (lab2 if ok3 or lab2 == 'UNSCORABLE' else 'UNINFORMATIVE (B3: digit not shown present)'))
