"""v181 referee check (post hoc; reading rule in docs/prereg_v188_ouro_intermediate_confirmation.md, section 7, item 5).

Does the standard probe miss the copied digit at the last prompt token only because of its PCA-256 step? At the last
token, for every state (mid1 .. end4), 5-fold CV grouped by template (v178 folds), label = the varied digit:
  frozen    the v178 probe on raw states (StandardScaler -> PCA 256 -> logistic, C in const/0.01/0.1/1)
  full      raw states, StandardScaler -> logistic on all dimensions, C in const/1e-4/1e-3/0.01/0.1/1/10
  pca_dm    PCA fitted on family-demeaned training states, applied to raw states; same wide C grid
  mp_raw    minimal-pair probe: train on family-demeaned training states, predict raw (not demeaned) held-out states
  mp        minimal-pair probe as in v181 (demeaned train and test), for reference

  python v181_pca_check.py <dir with v181_states.npy, v181_meta.json>
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V

CS = ('const', 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0)
G = {}


def demean(X, fam):
    X = X.copy()
    for f in np.unique(fam):
        X[fam == f] -= X[fam == f].mean(0)
    return X


def fit_predict(Ztr, ytr, gtr, Zte):
    """Inner grouped CV over CS (as in v178 probe_predict), then fit on all training items."""
    from sklearn.linear_model import LogisticRegression
    inner = V.folds(gtr, k=3, seed=1781); best, best_acc = None, -1
    for C in CS:
        acc = []
        for g in range(3):
            a, b = inner != g, inner == g
            if C == 'const':
                acc.append((ytr[b] == np.bincount(ytr[a]).argmax()).mean())
            else:
                m = LogisticRegression(C=C, max_iter=500).fit(Ztr[a], ytr[a]); acc.append((m.predict(Ztr[b]) == ytr[b]).mean())
        if np.mean(acc) > best_acc:
            best, best_acc = C, np.mean(acc)
    if best == 'const':
        return np.full(len(Zte), np.bincount(ytr).argmax()), best
    return LogisticRegression(C=best, max_iter=500).fit(Ztr, ytr).predict(Zte), best


def job(args):
    st, kind = args
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    its = G['meta']['items']; si = G['meta']['slots'].index('last')
    X = np.asarray(G['X'][:, si, st], dtype=np.float32)
    y = np.array([i['digit'] for i in its]); tid = np.array([i['tid'] for i in its]); fam = np.array([i['fam'] for i in its])
    fold = V.folds(tid)
    if kind == 'frozen':
        return args, (float((V.probe_predict(X, y, tid, fold) == y).mean()), None)
    if kind == 'mp':
        return args, (float((V.probe_predict(demean(X, fam), y, tid, fold) == y).mean()), None)
    pred = np.full(len(y), -1); Cs = []
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        if kind == 'full':
            sc = StandardScaler().fit(X[tr]); Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
        elif kind == 'pca_dm':
            D = demean(X[tr], fam[tr]); sc = StandardScaler().fit(D); p = PCA(256, random_state=0).fit(sc.transform(D))
            Ztr, Zte = p.transform(sc.transform(X[tr])), p.transform(sc.transform(X[te]))
        else:                                           # mp_raw: train demeaned, test raw
            D = demean(X[tr], fam[tr]); sc = StandardScaler().fit(D); p = PCA(256, random_state=0).fit(sc.transform(D))
            Ztr, Zte = p.transform(sc.transform(D)), p.transform(sc.transform(X[te]))
        pred[te], c = fit_predict(Ztr, y[tr], tid[tr], Zte); Cs.append(c)
    return args, (float((pred == y).mean()), Cs)


def init(X, meta):
    G['X'], G['meta'] = X, meta


if __name__ == '__main__':
    f = sorted(glob.glob(os.path.join(sys.argv[1], '**', 'v181_states.npy'), recursive=True))[0]
    meta = json.load(open(os.path.join(os.path.dirname(f), 'v181_meta.json'))); X = np.load(f, mmap_mode='r')
    states = meta['states']; sts = [k for k, s in enumerate(states) if s != 'h0']
    kinds = ('frozen', 'full', 'pca_dm', 'mp_raw', 'mp')
    with Pool(4, initializer=init, initargs=(X, meta)) as pool:
        res = dict(pool.map(job, [(st, k) for st in sts for k in kinds]))
    print('v181 last prompt token: %d items, %d families, %d templates, state dim %d (chance 0.10)' % (
        len(meta['items']), len({i['fam'] for i in meta['items']}), len({i['tid'] for i in meta['items']}), X.shape[-1]))
    print('%-8s %s' % ('probe', ' '.join('%7s' % states[st] for st in sts)))
    for k in kinds:
        print('%-8s %s' % (k, ' '.join('%7.3f' % res[(st, k)][0] for st in sts)))
    for k in ('full', 'pca_dm', 'mp_raw'):
        print('chosen C (%s): %s' % (k, {states[st]: res[(st, k)][1] for st in sts}))
    ends = [st for st in sts if states[st].startswith('end')]
    hi = max(max(res[(st, k)][0] for st in ends) for k in ('full', 'pca_dm'))
    lab = 'PCA ARTIFACT' if hi >= 0.50 else 'HOLDS' if hi <= 0.30 else 'PARTIAL'
    print('READING (rule fixed before running): best of full / pca_dm at loop ends %.3f -> %s' % (hi, lab))
