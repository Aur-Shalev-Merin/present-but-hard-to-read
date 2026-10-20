# EXPLORATORY diagnostics on v178 (primary already UNSCORABLE; nothing here changes the label)
import sys, json, time, numpy as np
sys.path.insert(0, '.')
import v178_analyze as V
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
R = 'results_v178/'
A = np.load(R + 'merged/v178_states_last.npy').astype(np.float32)
Qs = np.load(R + 'merged/v178_states_q.npy').astype(np.float32)
labels = json.load(open('v178_labels.json'))['items']
tids = np.array([i['tid'] for i in labels]); fold = V.folds(tids)
inst = np.array([i['inst'] for i in labels]); fold_w = V.folds(inst)
y = np.array([i['y_target'] for i in labels])
yc = np.array([-1 if i['y_copy'] is None else i['y_copy'] for i in labels]); okc = yc >= 0
def maj(v): return np.bincount(v).max() / len(v)
print('target majority %.3f | copy n=%d majority %.3f' % (maj(y), okc.sum(), maj(yc[okc])), flush=True)
def lr(X, y, fold, C=0.1):
    pred = np.full(len(y), -1)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        red = make_pipeline(StandardScaler(), PCA(min(256, tr.sum() - 1, X.shape[1]), random_state=0)).fit(X[tr])
        pred[te] = LogisticRegression(C=C, max_iter=300).fit(red.transform(X[tr]), y[tr]).predict(red.transform(X[te]))
    return (pred == y).mean()
O = V.ORDER
for nm, X in (('last', A), ('q', Qs)):
    for st in (2, 4, 6, 8):
        t0 = time.time()
        print('%s %-5s copy-cross %.3f | target-cross %.3f | target-within %.3f  (%.0fs)' % (
            nm, O[st], lr(X[okc, st], yc[okc], fold[okc]), lr(X[:, st], y, fold), lr(X[:, st], y, fold_w), time.time() - t0), flush=True)
Xin = V.input_features([i['qnums'] for i in labels])
u = {t: k for k, t in enumerate(np.unique(tids))}; T = np.eye(len(u))[[u[t] for t in tids]]
print('input feats: target-cross %.3f | target-within %.3f | template-onehot-only within %.3f | input+template within %.3f' % (
    lr(Xin, y, fold), lr(Xin, y, fold_w), lr(T, y, fold_w), lr(np.hstack([Xin, T]), y, fold_w)), flush=True)
