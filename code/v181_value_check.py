# EXPLORATORY follow-up on v181: is the contrastive code at the last token a 1-D number-value direction or a digit code?
import json, sys, numpy as np
sys.path.insert(0, '.')
import v178_analyze as V
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import glob, os
R = os.path.dirname(sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else 'results_v181', '**', 'v181_states.npy'), recursive=True))[0]) + '/'
print('states from', R, flush=True)
meta = json.load(open(R + 'v181_meta.json')); its = meta['items']
X_all = np.load(R + 'v181_states.npy', mmap_mode='r')
y = np.array([i['digit'] for i in its]); fam = np.array([i['fam'] for i in its]); tids = np.array([i['tid'] for i in its])
fold = V.folds(tids)
def demean(X):
    X = X.copy()
    for f in np.unique(fam):
        X[fam == f] -= X[fam == f].mean(0)
    return X
def cv_r2(X, t):
    pred = np.zeros_like(t, dtype=float)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        pc = PCA(min(256, tr.sum() - 1), random_state=0).fit(X[tr])
        pred[te] = Ridge(alpha=10.0).fit(pc.transform(X[tr]), t[tr]).predict(pc.transform(X[te]))
    return 1 - ((pred - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
for slot, si in (('last', 5), ('qmark', 4), ('+3', 2)):
    for st_name, st in (('end1', 2), ('end4', 5)):
        X = demean(np.asarray(X_all[:, si, st], dtype=np.float32))
        r2_lin = cv_r2(X, y.astype(float))
        r2_cos = np.mean([cv_r2(X, np.cos(2 * np.pi * y / 10)), cv_r2(X, np.sin(2 * np.pi * y / 10))])
        p = V.probe_predict(X, y, tids, fold); acc = (p == y).mean(); err = p != y
        dist = np.abs(p[err] - y[err]).mean() if err.any() else float('nan')
        # 10-class accuracy using only the top-k contrastive PCs (fit on training folds inside probe via PCA dim)
        acc_k = {}
        for k in (1, 2, 4):
            pk = np.full(len(y), -1)
            for f in np.unique(fold):
                tr, te = fold != f, fold == f
                pc = PCA(k, random_state=0).fit(X[tr]); Ztr, Zte = pc.transform(X[tr]), pc.transform(X[te])
                from sklearn.linear_model import LogisticRegression
                pk[te] = LogisticRegression(C=1.0, max_iter=1000).fit(Ztr, y[tr]).predict(Zte)
            acc_k[k] = (pk == y).mean()
        print('%-5s %s | 10-class acc %.3f, mean |err| among errors %.2f (uniform ~3.3) | R2 value(0-9) %.3f, R2 circular %.3f | acc with top-1/2/4 PCs %.3f/%.3f/%.3f' % (
            slot, st_name, acc, dist, r2_lin, r2_cos, acc_k[1], acc_k[2], acc_k[4]), flush=True)
