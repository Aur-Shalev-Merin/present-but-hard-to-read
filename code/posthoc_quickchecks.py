"""POST HOC quick checks after the fourth outside reviews (docs/posthoc_quickchecks_2026-11-06.md; rules recorded first).

  python posthoc_quickchecks.py [c1|c2|c3|all]   -> results_posthoc/quickchecks_<check>.json and printed summary

C1  standard vs minimal-pair probe for the computed value s in the v230 small models (new one-token families).
C2  share of b's basis left after projecting off a random subspace of the copy (+ answer) basis's dimension.
C3  minimal-pair probe on v181 (Ouro-1.4B copied digit, last prompt token) with test items contrasted against k siblings.
Descriptive only.
"""
import json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v178_analyze as V
OUT = os.path.join(HERE, 'results_posthoc'); os.makedirs(OUT, exist_ok=True)


def fit_predict(Xtr, ytr, gtr, tests, pca_dim=256):
    """v178's probe_predict for one outer fold: returns predictions for each test matrix in `tests`."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    dim = min(pca_dim, Xtr.shape[1], len(Xtr) - 1)
    red = make_pipeline(StandardScaler(), PCA(dim, random_state=0)).fit(Xtr)
    Ztr = red.transform(Xtr); inner = V.folds(gtr, k=3, seed=1781); best, best_acc = None, -1
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
        return [np.full(len(T), np.bincount(ytr).argmax()) for T in tests]
    m = LogisticRegression(C=best, max_iter=300).fit(Ztr, ytr)
    return [m.predict(red.transform(T)) for T in tests]


def demean(X, fam):
    X = np.array(X, dtype=np.float32)
    for f in np.unique(fam):
        X[fam == f] -= X[fam == f].mean(0)
    return X


def cv_acc(X, y, groups):
    fold = V.folds(groups); pred = np.full(len(y), -1)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        pred[te] = fit_predict(X[tr], y[tr], groups[tr], [X[te]])[0]
    return float((pred == y).mean())


def c1():
    import torch, v230_pilot as VP, v230_protocol as PR
    VP.DEV = 'cpu'; k = 6; P = PR.P; rng = np.random.default_rng(2330); its = []
    for f in range(200):
        x, w, op = int(rng.integers(P)), int(rng.integers(P)), int(rng.integers(2))
        for z in range(P):
            its.append(dict(fam=f, op=op, x=x, z=z, w=w, k=k, s=(x + z) % P if op == 0 else (x - z) % P))
    fam = np.array([i['fam'] for i in its]); s = np.array([i['s'] for i in its]); z = np.array([i['z'] for i in its])
    plist = VP.positions(k); out = {}
    for seed in range(5):
        task = VP.Task('pos'); net = VP.Net(VP.V, VP.L, 'std')
        net.load_state_dict(torch.load(os.path.join(HERE, 'results_v230', 'protocol', 'gpu0', 'v230_std_pos_s%d.pt' % seed),
                                       map_location='cpu', weights_only=True)); net.eval()
        X, ok, _ = PR.states(net, task, its, plist); li = PR.BAND.index(4)
        for name, pos in ((';', 9 - k), ('filler', plist[1])):
            Z = X[:, plist.index(pos), li]; row = {}
            for lab, y in (('s', s), ('z', z)):
                row[lab] = dict(standard=cv_acc(Z, y, fam), minimal_pairs=cv_acc(demean(Z, fam), y, fam))
            out['seed%d_%s' % (seed, name)] = row
            print('C1 seed %d %-6s (pos %d, layer 4; model accuracy %.3f): s std %.3f / mp %.3f | z std %.3f / mp %.3f' % (
                seed, name, pos, ok.mean(), row['s']['standard'], row['s']['minimal_pairs'], row['z']['standard'],
                row['z']['minimal_pairs']), flush=True)
    return out


def left(U, Q):
    Q, _ = np.linalg.qr(Q)
    return float(np.linalg.norm(U - Q @ (Q.T @ U)) ** 2 / U.shape[1])


def c2():
    rng = np.random.default_rng(2331); out = {}
    for study, f, offs in (('v215 chain', 'results_v221/a/v215_plan.npz', ('D',)),
                           ('v216 no chain', 'results_v216/a_v3/v216_plan.npz', ('D', 'An'))):
        p = np.load(os.path.join(HERE, f)); rows = []
        for key in sorted(k for k in p.files if k.startswith('U_')):
            _, pos, half, L = key.split('_'); U, _ = np.linalg.qr(p[key].astype(np.float64))
            Q = np.hstack([p['%s_%s_%s_%s' % (o, pos, half, L)] for o in offs]).astype(np.float64)
            rnd = [left(U, rng.standard_normal(Q.shape)) for _ in range(20)]
            rows.append(dict(pos=pos, half=int(half), layer=int(L), dims_U=U.shape[1], dims_off=Q.shape[1], left_copy=left(U, Q),
                             left_random=float(np.mean(rnd)), left_random_min=float(np.min(rnd))))
        out[study] = rows
        lc = [r['left_copy'] for r in rows]; lr = [r['left_random'] for r in rows]
        print('C2 %-14s dims b %d, off %d | left after copy%s projection %.2f-%.2f | after random of same dim %.4f-%.4f (min draw %.4f)' % (
            study, rows[0]['dims_U'], rows[0]['dims_off'], ' + answer' if 'An' in offs else '', min(lc), max(lc), min(lr), max(lr),
            min(r['left_random_min'] for r in rows)), flush=True)
    out['v233 analytic'] = {'z (22 of 192)': 1 - 22 / 192, 'z and x (44 of 192)': 1 - 44 / 192}
    print('C2 v233 small models, random subspace expectation: z %.3f, z+x %.3f (observed 0.55-0.67 and 0.26-0.44)' % (
        1 - 22 / 192, 1 - 44 / 192))
    return out


def c3():
    root = os.path.join(HERE, 'results_v181', 'merged'); meta = json.load(open(os.path.join(root, 'v181_meta.json')))
    A = np.load(os.path.join(root, 'v181_states.npy'), mmap_mode='r'); its = meta['items']
    slots, states = meta['slots'], meta['states']; si = slots.index('last')
    y = np.array([i['digit'] for i in its]); tids = np.array([i['tid'] for i in its]); fam = np.array([i['fam'] for i in its])
    fold = V.folds(tids); KS = (0, 1, 2, 4, 9); out = {}
    for st in ('end1', 'end2', 'end3', 'end4'):
        X = np.asarray(A[:, si, states.index(st)], dtype=np.float32); Xd = demean(X, fam)
        rng = np.random.default_rng(2332); tests = {}
        for kk in KS:
            T = X.copy() if kk == 0 else np.zeros_like(X)
            if kk:
                for n in range(len(X)):
                    sib = np.flatnonzero((fam == fam[n]) & (np.arange(len(X)) != n))
                    T[n] = X[n] - X[rng.choice(sib, kk, replace=False)].mean(0)
            tests[kk] = T
        pred = {kk: np.full(len(y), -1) for kk in KS}
        for f in np.unique(fold):
            tr, te = fold != f, fold == f
            ps = fit_predict(Xd[tr], y[tr], tids[tr], [tests[kk][te] for kk in KS] + [Xd[te]])
            for kk, pp in zip(KS, ps):
                pred[kk][te] = pp
            pred.setdefault('full', np.full(len(y), -1))[te] = ps[-1]
        out[st] = {str(kk): float((pred[kk] == y).mean()) for kk in list(KS) + ['full']}
        print('C3 last token %s: full family %.3f | k siblings: %s' % (
            st, out[st]['full'], '  '.join('k=%d %.3f' % (kk, out[st][str(kk)]) for kk in KS)), flush=True)
    return out


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for name, fn in (('c2', c2), ('c3', c3), ('c1', c1)):
        if which in (name, 'all'):
            json.dump(fn(), open(os.path.join(OUT, 'quickchecks_%s.json' % name), 'w'), indent=1)
