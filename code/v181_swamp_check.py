"""v181 "swamped" measurement (post hoc; reading rule in docs/v181_result.md, fixed before running).

At the last prompt token, per loop state, with folds grouped by template (v178 folds; fit on training templates,
evaluate on held-out ones):
  1. overlap: context variance per dimension inside the 9-dim digit subspace / per dimension overall (O), and the
     signal-to-noise inside the subspace (SNR_U) vs over all dimensions (SNR_all);
  2. structured linear decoder: shrinkage LDA (digit means from family-centred training states, Ledoit-Wolf covariance
     of training items around their digit means) applied to raw single held-out prompts; plus the same rule on samples
     from the fitted Gaussian (model-based, optimistic);
  3. learning curve (needs sklearn): the frozen standard probe and the minimal-pair probe trained on 25/50/75/100% of
     each training fold's templates.

  python v181_swamp_check.py <dir with v181_states.npy, v181_meta.json> [--no-lc]
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

import v178_analyze as V

FRACS = (0.25, 0.5, 0.75, 1.0)
G = {}


def demean(X, fam):
    X = X.copy()
    for f in np.unique(fam):
        X[fam == f] -= X[fam == f].mean(0)
    return X


def fam_means(X, fam):
    return np.stack([X[fam == f].mean(0) for f in np.unique(fam)])


def ledoit_wolf(Z):
    """Ledoit-Wolf shrunk covariance of centred rows Z (n x p); same estimator as sklearn.covariance.LedoitWolf."""
    n, p = Z.shape
    S = Z.T @ Z / n
    mu = np.trace(S) / p
    d2 = ((S - mu * np.eye(p)) ** 2).sum()
    b2 = ((Z ** 2).sum(1) ** 2).sum() / n ** 2 - (S ** 2).sum() / n
    s = min(b2, d2) / d2
    return s * mu * np.eye(p) + (1 - s) * S, s


def overlap_lda(X, y, fam, fold, seed=181):
    rows = []
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        mu = np.stack([demean(X[tr], fam[tr])[y[tr] == k].mean(0) for k in range(10)]); mu -= mu.mean(0)
        U = np.linalg.svd(mu, full_matrices=False)[2][:9].T
        Xc_te = demean(X[te], fam[te])
        mu_te = np.stack([Xc_te[y[te] == k].mean(0) for k in range(10)]); mu_te -= mu_te.mean(0)
        F = fam_means(X[te], fam[te]); F -= F.mean(0)
        R = Xc_te - mu_te[y[te]]
        d = X.shape[1]
        ctxU, ctxA = ((F @ U) ** 2).sum() / (len(F) - 1) / 9, (F ** 2).sum() / (len(F) - 1) / d
        resU, resA = ((R @ U) ** 2).sum() / (len(R) - 1) / 9, (R ** 2).sum() / (len(R) - 1) / d
        sigU, sigA = ((mu_te @ U) ** 2).sum() / 10 / 9, (mu_te ** 2).sum() / 10 / d
        g = X[tr].mean(0); N = X[tr] - g - mu[y[tr]]; N = N - N.mean(0)
        S, shrink = ledoit_wolf(N)
        W = np.linalg.solve(S, mu.T); b = -0.5 * (mu * W.T).sum(1)
        lda = float((np.argmax((X[te] - g) @ W + b, 1) == y[te]).mean())
        rng = np.random.default_rng(seed + int(f)); Lc = np.linalg.cholesky(S)
        yy = rng.integers(0, 10, 4000); xs = mu[yy] + rng.standard_normal((4000, d)) @ Lc.T
        mc = float((np.argmax(xs @ W + b, 1) == yy).mean())
        rows.append(dict(O=ctxU / ctxA, snrU=sigU / (ctxU + resU), snrA=sigA / (ctxA + resA), lda=lda, mc=mc,
                         shrink=shrink, n_te=int(te.sum())))
    w = np.array([r['n_te'] for r in rows], float)
    return {k: float(np.average([r[k] for r in rows], weights=w)) for k in rows[0] if k != 'n_te'}


def fit_predict(Xtr, ytr, gtr, Xte):
    """The v178 probe (StandardScaler -> PCA 256 -> logistic, inner grouped CV over const/0.01/0.1/1) on one split."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    dim = min(256, Xtr.shape[1], len(Xtr) - 1)
    red = make_pipeline(StandardScaler(), PCA(dim, random_state=0)).fit(Xtr)
    Ztr, Zte = red.transform(Xtr), red.transform(Xte)
    inner = V.folds(gtr, k=3, seed=1781); best, best_acc = None, -1
    for C in ('const', 0.01, 0.1, 1.0):
        acc = []
        for g in range(3):
            a, b = inner != g, inner == g
            if C == 'const':
                acc.append((ytr[b] == np.bincount(ytr[a]).argmax()).mean())
            else:
                m = LogisticRegression(C=C, max_iter=300).fit(Ztr[a], ytr[a]); acc.append((m.predict(Ztr[b]) == ytr[b]).mean())
        if np.mean(acc) > best_acc:
            best, best_acc = C, np.mean(acc)
    if best == 'const':
        return np.full(len(Xte), np.bincount(ytr).argmax())
    return LogisticRegression(C=best, max_iter=300).fit(Ztr, ytr).predict(Zte)


def lc_job(args):
    st, f, frac, draw, kind = args
    X, y, fam, tid, fold = G['X'][:, G['si'], st].astype(np.float32), G['y'], G['fam'], G['tid'], G['fold']
    if kind == 'mp':
        X = demean(X, fam)
    tr, te = fold != f, fold == f
    tt = np.unique(tid[tr]); k = max(3, int(round(frac * len(tt))))
    keep = np.isin(tid, np.random.default_rng(1810 + 7 * draw + 101 * int(f)).choice(tt, k, replace=False)) & tr
    pred = fit_predict(X[keep], y[keep], tid[keep], X[te])
    return args, (int((pred == y[te]).sum()), int(te.sum()))


def init(Xall, si, y, fam, tid, fold):
    G.update(X=Xall, si=si, y=y, fam=fam, tid=tid, fold=fold)


def main(root, lc=True):
    f = sorted(glob.glob(os.path.join(root, '**', 'v181_states.npy'), recursive=True))[0]
    meta = json.load(open(os.path.join(os.path.dirname(f), 'v181_meta.json'))); Xall = np.load(f, mmap_mode='r')
    its = meta['items']; si = meta['slots'].index('last'); states = meta['states']
    y = np.array([i['digit'] for i in its]); fam = np.array([i['fam'] for i in its]); tid = np.array([i['tid'] for i in its])
    fold = V.folds(tid); sts = [k for k, s in enumerate(states) if s != 'h0']
    print('v181 last prompt token: %d items, %d families, %d templates, d = %d' % (
        len(its), len(set(fam)), len(set(tid)), Xall.shape[-1]), flush=True)
    res = {}
    for st in sts:
        r = overlap_lda(np.asarray(Xall[:, si, st], dtype=np.float64), y, fam, fold); res[st] = r
        print('%-5s overlap O %6.1f | SNR in digit subspace %.3f vs all dims %.4f | structured LDA on single prompts '
              '%.3f | Gaussian-model ceiling %.3f | LW shrinkage %.3f' % (
                  states[st], r['O'], r['snrU'], r['snrA'], r['lda'], r['mc'], r['shrink']), flush=True)
    if not lc:
        return
    from multiprocessing import Pool
    jobs = [(st, f_, fr, dr, kind) for st in sts for f_ in np.unique(fold) for fr in FRACS
            for dr in (range(3) if fr < 1 else range(1)) for kind in ('std', 'mp')]
    with Pool(4, initializer=init, initargs=(Xall, si, y, fam, tid, fold)) as pool:
        out = dict(pool.map(lc_job, jobs))
    curve = {}
    for st in sts:
        for kind in ('std', 'mp'):
            for fr in FRACS:
                draws = range(3) if fr < 1 else range(1)
                accs = [sum(out[(st, f_, fr, dr, kind)][0] for f_ in np.unique(fold)) / len(y) for dr in draws]
                curve[(st, kind, fr)] = float(np.mean(accs))
        print('%-5s learning curve (25/50/75/100%% of training templates): standard %s | minimal-pair %s' % (
            states[st], ' '.join('%.3f' % curve[(st, 'std', fr)] for fr in FRACS),
            ' '.join('%.3f' % curve[(st, 'mp', fr)] for fr in FRACS)), flush=True)
    ends = [st for st in sts if states[st].startswith('end')]
    gain = float(np.mean([curve[(st, 'std', 1.0)] - curve[(st, 'std', 0.5)] for st in ends]))
    lda_max = max(res[st]['lda'] for st in ends)
    if lda_max <= 0.40 and gain <= 0.05:
        lab = 'SWAMPED'
    elif gain >= 0.10 or lda_max >= 0.60:
        lab = 'SAMPLE-LIMITED'
    else:
        lab = 'MIXED'
    print('READING (rule fixed before running): structured LDA max over loop ends %.3f; standard-probe gain 50%%->100%% '
          'templates %+.3f -> %s' % (lda_max, gain, lab))


if __name__ == '__main__':
    main(sys.argv[1], lc='--no-lc' not in sys.argv)
