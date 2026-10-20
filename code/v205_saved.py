"""v205 Part A, on saved states only (docs/prereg_v205_swamping_robustness.md).

  A1 anisotropy baseline for the overlap O. Per fold (v178 folds by template, as overlap_lda): U = the 9-dim span of the
     family-centred class means fitted on the training families; F = the held-out families' mean states. O(U) = context
     variance per dimension inside U / per dimension overall (overlap_lda's O). Baseline: the same O for 50 random 9-dim
     subspaces drawn uniformly inside the top-k principal components of the TRAINING prompts' raw states (what the
     standard probe's PCA is fitted on), k = 16, 64, 256. R_k = O(U) / mean O(random in top-k), and the share of U inside
     the top-k PCs (||V_k^T U||^2 / 9). DESCRIPTIVE: the baseline depends on k (selftest), so the label for 'is the
     crowding special to the token?' comes from Part B's random-perturbation null.
  A2 MLP probe: StandardScaler -> PCA 256 -> MLP (one hidden layer of 256, ReLU, early stopping, alpha chosen by inner
     3-fold grouped CV over {1e-4, 1e-2, 1}), outer 5 folds grouped by template; raw (standard) and family-demeaned
     (contrastive, positive control) states at the last prompt token.
  A3 Study 3 at P0 with the carry and borrow in the copy ceiling (v194 states): ceiling features = one-hot(a_u, b_u, c_u)
     + first-step carry/borrow + second-step borrow + their product; otherwise v194's statistic (I, F, pair bootstrap,
     seed 194, 97.5% CI) unchanged. Also at P1 end4, and with the original ceiling to reproduce v194.

  python v205_saved.py ouro181 <dir>   (Ouro v181 states: A1, A2)
  python v205_saved.py qwen202 <dir>   (Qwen v202 states, both designs: A1, A2)
  python v205_saved.py ouro185 <dir>   (Ouro v185 states: A1 for colour and added digit)
  python v205_saved.py study3 <dir>    (v194 states: A3)
  python v205_saved.py selftest
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V
import v181_swamp_check as S

KS = (16, 64, 256)
TYPES = ['col_start', 'col_end', 'dig_start', 'dig_end']
G = {}


def init(d):
    G.update(d)


def right_sv(A):
    """Right singular vectors (rows, by decreasing singular value). Post-freeze numerical fix (2026-10-16): LAPACK's SVD
    failed to converge on Kaggle for one Ouro state; fall back to the same vectors from the eigendecomposition of A^T A."""
    try:
        return np.linalg.svd(A, full_matrices=False)[2]
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(A.T @ A); o = np.argsort(w)[::-1][:min(A.shape)]
        return v[:, o].T


def aniso(X, y, fam, fold, n_rand=50, seed=205):
    rng = np.random.default_rng(seed); rows = []
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        Xc = S.demean(X[tr], fam[tr]); mu = np.stack([Xc[y[tr] == k].mean(0) for k in range(10)]); mu -= mu.mean(0)
        U = right_sv(mu)[:9].T
        Xtr = X[tr] - X[tr].mean(0)                          # the standard probe's PCA is fitted on these prompts
        Vt = right_sv(Xtr)
        F = S.fam_means(X[te], fam[te]); F -= F.mean(0)
        d = X.shape[1]; ctxA = (F ** 2).sum() / d

        def O(B):
            return ((F @ B) ** 2).sum() / B.shape[1] / ctxA
        r = dict(O=O(U), n=int(te.sum()))
        os_ = []                                             # null: labels shuffled within each family (5 draws)
        for _ in range(5):
            ysh = y[tr].copy()
            for g in np.unique(fam[tr]):
                m = fam[tr] == g; ysh[m] = rng.permutation(ysh[m])
            mus = np.stack([Xc[ysh == k].mean(0) for k in range(10)]); mus -= mus.mean(0)
            os_.append(O(right_sv(mus)[:9].T))
        r['shuf'] = float(np.mean(os_))
        Rr = Xc - mu[y[tr]]                                  # null: top-9 PCs of the within-family residuals
        r['resPC'] = O(right_sv(Rr - Rr.mean(0))[:9].T)
        for k in KS:
            k_ = min(k, Vt.shape[0]); Vk = Vt[:k_].T
            r['rand%d' % k] = np.mean([O(Vk @ np.linalg.qr(rng.standard_normal((k_, 9)))[0]) for _ in range(n_rand)])
            r['share%d' % k] = float(((Vk.T @ U) ** 2).sum() / 9)
        rows.append(r)
    w = [r['n'] for r in rows]
    return {k: float(np.average([r[k] for r in rows], weights=w)) for k in rows[0] if k != 'n'}


def mlp_predict(X, y, groups, fold):
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pred = np.full(len(y), -1)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        red = make_pipeline(StandardScaler(), PCA(min(256, X.shape[1], tr.sum() - 1), random_state=0)).fit(X[tr])
        Ztr, Zte, ytr = red.transform(X[tr]), red.transform(X[te]), y[tr]
        inner = V.folds(groups[tr], k=3, seed=1781); best, ba = None, -1

        def m(a):
            return MLPClassifier((256,), alpha=a, early_stopping=True, max_iter=300, random_state=0)
        for a in (1e-4, 1e-2, 1.0):
            acc = np.mean([(m(a).fit(Ztr[inner != g], ytr[inner != g]).predict(Ztr[inner == g]) == ytr[inner == g]).mean()
                           for g in range(3)])
            if acc > ba:
                best, ba = a, acc
        pred[te] = m(best).fit(Ztr, ytr).predict(Zte)
    return pred


def mlp_job(args):
    st, contrastive = args
    X = np.asarray(G['X'][:, st], dtype=np.float32)
    if contrastive:
        X = S.demean(X, G['fam'])
    return args, float((mlp_predict(X, G['y'], G['tid'], V.folds(G['tid'])) == G['y']).mean())


def load(root, stem):
    f = max(glob.glob(os.path.join(root, '**', stem + '_states.npy'), recursive=True), key=os.path.getsize)
    meta = json.load(open(os.path.join(os.path.dirname(f), stem + '_meta.json')))
    print('states:', os.path.relpath(f, root), flush=True)
    return np.load(f, mmap_mode='r'), meta


def digit_design(X, meta, name, Q, procs=4):
    its = meta['items']; states = meta['states']; si = meta['slots'].index('last')
    y = np.array([i['digit'] for i in its]); fam = np.array([i['fam'] for i in its]); tid = np.array([i['tid'] for i in its])
    fold = V.folds(tid); sts = [k for k, s in enumerate(states) if s != 'h0']
    print('--- %s, last prompt token: A1 anisotropy baseline ---' % name)
    R = {}
    for st in sts:
        r = aniso(np.asarray(X[:, si, st], dtype=np.float64), y, fam, fold); R[states[st]] = r
        print('%-5s O(U) %6.2f | shuffled-label null %6.2f (O/null %.2f) | residual-PC null %6.2f | '
              'random in top-16/64/256 %6.2f %6.2f %6.2f | R256 %.2f | share of U in top-16/64/256 '
              '%.2f %.2f %.2f' % (states[st], r['O'], r['shuf'], r['O'] / r['shuf'], r['resPC'], r['rand16'],
                                  r['rand64'], r['rand256'], r['O'] / r['rand256'],
                                  r['share16'], r['share64'], r['share256']), flush=True)
    print('A1 %s (%s): mean O/shuffled-label null %.2f' % (name, '/'.join(Q), float(np.mean([R[q]['O'] / R[q]['shuf'] for q in Q]))))
    print('A1 %s (%s, descriptive): mean R16 %.2f, R64 %.2f, R256 %.2f' % (name, '/'.join(Q), *[
        float(np.mean([R[q]['O'] / R[q]['rand%d' % k] for q in Q])) for k in KS]))
    arrs = dict(X=np.asarray(X[:, si]), si=slice(None), y=y, fam=fam, tid=tid); G.update(arrs)
    jobs = [(st, c) for st in sts for c in (False, True)]
    with Pool(procs, initializer=init, initargs=(arrs,)) as pool:
        res = dict(pool.map(mlp_job, jobs))
    print('--- %s: A2 MLP probe, standard / contrastive ---' % name)
    print(' '.join('%s %.3f/%.3f' % (states[st], res[(st, False)], res[(st, True)]) for st in sts))
    qi = [states.index(q) for q in Q]; raw = max(res[(s, False)] for s in qi); ctr = max(res[(s, True)] for s in qi)
    lab2 = ('UNINFORMATIVE' if ctr < 0.70 else 'NONLINEAR-READABLE' if raw >= 0.40 else 'STILL-SWAMPED' if raw <= 0.30
            else 'PARTIAL')
    print('A2 %s (%s): MLP standard max %.3f, contrastive max %.3f -> %s' % (name, '/'.join(Q), raw, ctr, lab2))


def colour_design(X, items, states, name, Q):
    typ = np.array(['%s_%s' % (i['kind'], i['where']) for i in items]); si = 2
    print('--- %s, colour vs added digit: A1 anisotropy baseline (descriptive) ---' % name)
    for t in TYPES:
        idx = np.where(typ == t)[0]; y = np.array([items[i]['digit'] for i in idx]); fam = np.array([items[i]['fam'] for i in idx])
        tid = np.array([items[i]['tid'] for i in idx]); rr = []
        for q in Q:
            r = aniso(np.asarray(X[idx][:, si, states.index(q)], dtype=np.float64), y, fam, V.folds(tid)); rr.append(r)
        print('%-9s %s' % (t, ' '.join('%s O %.2f shuffled %.2f O/shuffled %.2f rand256 %.2f' % (q, r['O'], r['shuf'], r['O'] / r['shuf'], r['rand256'])
                                       for q, r in zip(Q, rr))), flush=True)


def study3(root):
    import v194_analyze as B
    import v186_analyze as A
    X, meta = load(root, 'v194'); its = meta['items']; S8 = meta['states']; E = S8.index('end4')
    sel = lambda op: [k for k, i in enumerate(its) if i['set'] == 'main' and i['op'] == op]
    out = {}
    for op in ('plus', 'minus'):
        idx = np.array(sel(op)); sub = [its[k] for k in idx]
        fam = np.array([i['fam'] for i in sub]); tid = np.array([i['tid'] for i in sub]); fold = V.folds(tid)
        L = [i['labels'] for i in sub]; au, bu, cu = (np.array([l[k] for l in L]) for k in 'abc')
        first = (au + bu) if op == 'plus' else (au - bu)
        c1 = ((au + bu) >= 10) if op == 'plus' else (au < bu); c2 = (first % 10) < cu
        Z0 = A.onehot(sub); Zcb = np.hstack([Z0, np.stack([c1, c2, c1 & c2], 1).astype(np.float32)])
        for f in (B.AB, B.AMB):
            y = np.array([B.form(i, f) for i in sub])
            for nm, Z in (('ceil', Z0), ('ceil_cb', Zcb)):
                out[(op, f, nm)] = (V.probe_predict(A.demean(Z, fam), y, tid, fold) == y).astype(float)
            for pos in (0, 1):
                Xs = np.asarray(X[idx, pos, E], dtype=np.float32)
                out[(op, f, pos)] = (V.probe_predict(A.demean(Xs, fam), y, tid, fold) == y).astype(float)
        out[(op, 'pair')] = np.array([i['pair'] for i in sub])
        print('%s: carry/borrow rates c1 %.3f c2 %.3f' % (op, c1.mean(), c2.mean()), flush=True)
    print('--- A3 Study 3, I with the original and the carry/borrow ceiling (after loop 4) ---')
    res = {}
    for pos, pn in ((0, 'P0'), (1, 'P1')):
        for nm in ('ceil', 'ceil_cb'):
            ex = {(op, f): out[(op, f, pos)] - out[(op, f, nm)] for op in ('plus', 'minus') for f in (B.AB, B.AMB)}
            dp = ex[('plus', B.AB)] - ex[('plus', B.AMB)]; dm = ex[('minus', B.AMB)] - ex[('minus', B.AB)]
            (I, lo, hi), (F, flo, fhi) = B.pair_ci(out[('plus', 'pair')], dp, out[('minus', 'pair')], dm)
            res[(pn, nm)] = (I, lo, hi)
            print('%s end4, %-7s | D+ %+.3f D- %+.3f | I %+.3f [%+.3f, %+.3f] | F %+.3f [%+.3f, %+.3f]' % (
                pn, nm, dp.mean(), dm.mean(), I, lo, hi, F, flo, fhi), flush=True)
    I, lo, hi = res[('P0', 'ceil_cb')]
    print('A3 P0 with carry/borrow: I %+.3f [%+.3f, %+.3f] -> %s' % (
        I, lo, hi, 'EXPLAINED by the carry/borrow (CI includes 0)' if lo <= 0 <= hi else
        'REMAINS above zero' if lo > 0 else 'REVERSES (below zero)'))


def selftest():
    """A1 on synthetic states: (a) the class code lies in random directions of a 64-dim anisotropic context ->
    R64 near 1; (b) the context is concentrated on the class code itself -> R64 about 2 or more."""
    rng = np.random.default_rng(0); d, nf, k_act = 512, 120, 64
    act = np.linalg.qr(rng.standard_normal((d, k_act)))[0]
    for world in ('anisotropy', 'excess'):
        code = act @ np.linalg.qr(rng.standard_normal((k_act, 10)))[0] if world == 'anisotropy' else \
            np.linalg.qr(rng.standard_normal((d, 10)))[0]
        code = code.T * 3; code -= code.mean(0)
        Xs, ys, fs, ts = [], [], [], []
        for f in range(nf):
            ctx = act @ (rng.standard_normal(k_act) * np.linspace(6, 1, k_act)) + 0.3 * rng.standard_normal(d)
            if world == 'excess':
                ctx += code.T @ rng.standard_normal(10) * 2
            for k in range(10):
                Xs.append(ctx + code[k] + 0.05 * rng.standard_normal(d)); ys.append(k); fs.append(f); ts.append(f // 2)
        X, y, fam, tid = np.array(Xs), np.array(ys), np.array(fs), np.array(ts)
        r = aniso(X, y, fam, V.folds(tid), n_rand=20)
        print('selftest %-10s O(U) %.2f | R16 %.2f R64 %.2f R256 %.2f | share64 %.2f' % (
            world, r['O'], r['O'] / r['rand16'], r['O'] / r['rand64'], r['O'] / r['rand256'], r['share64']))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'selftest':
        selftest()
    elif mode == 'ouro181':
        X, m = load(sys.argv[2], 'v181'); digit_design(X, m, 'Ouro v181', ['end1', 'end2', 'end3', 'end4'])
    elif mode == 'qwen202':
        X, m = load(sys.argv[2], 'v202_v181'); digit_design(X, m, 'Qwen v202', ['l7', 'l14', 'l21', 'l28'])
        X5, m5 = load(sys.argv[2], 'v202_v185'); colour_design(X5, m5['items'], m5['states'], 'Qwen v202', ['l7', 'l14', 'l21', 'l28'])
    elif mode == 'ouro185':
        X5, m5 = load(sys.argv[2], 'v185'); colour_design(X5, m5['items'], m5['states'], 'Ouro v185', ['end1', 'end2', 'end3', 'end4'])
    elif mode == 'study3':
        study3(sys.argv[2])
