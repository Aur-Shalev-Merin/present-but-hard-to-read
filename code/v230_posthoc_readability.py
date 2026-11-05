"""POST HOC (after the v230 result and two cold reviews): is s linearly readable at the positions where the share swap
said FOLLOWS but DAS found no s (fillers, w's token, '=')? Independent of the swap: on family-demeaned states with the
copies (x, z, w, op, answer; one-hot) regressed out on the training folds, a ridge classifier decodes s on held-out
families (v178 folds). Controls: the same probe for four random functions of the operand pair (the swap's R0-R3), and
the negative-control models, which never need s. k = 6, layers 1-5, frozen v230 checkpoints. Descriptive only.

  python v230_posthoc_readability.py   -> results_v230/posthoc_readability.json
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v230_protocol as PR
import v178_analyze as V178


def ridge_acc(Z, fam, cops, y, lam=1.0):
    Z = PR.demean(Z, fam); C = np.zeros((len(Z), sum(n for _, n in cops)), np.float32); o = 0
    for c, n in cops:
        C[np.arange(len(Z)), o + c] = 1; o += n
    C = PR.demean(C, fam); fold = V178.folds(fam); hit = np.zeros(len(Z), bool); ncl = int(y.max()) + 1
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        W = np.linalg.lstsq(C[tr], Z[tr], rcond=None)[0]; Rtr = Z[tr] - C[tr] @ W; Rte = Z[te] - C[te] @ W
        mu, sd = Rtr.mean(0), Rtr.std(0) + 1e-6; Rtr = (Rtr - mu) / sd; Rte = (Rte - mu) / sd
        Y = np.eye(ncl, dtype=np.float32)[y[tr]]; Y -= Y.mean(0)
        B = np.linalg.solve(Rtr.T @ Rtr + lam * len(Rtr) * np.eye(Rtr.shape[1], dtype=np.float32), Rtr.T @ Y)
        hit[te] = (Rte @ B).argmax(1) == y[te]
    return float(hit.mean())


if __name__ == '__main__':
    import torch, v230_pilot as VP
    VP.DEV = 'cpu'; k = 6; its = PR.items(k); plist = VP.positions(k); fam = np.array([i['fam'] for i in its])
    g = lambda f: np.array([i[f] for i in its]); s = g('s')
    R = PR.labels(g('x'), g('z'), PR.P); rand = [R['R%d' % r] for r in range(4)]
    out = {}
    for kind in ('pos', 'neg'):
        for seed in range(5):
            task = VP.Task(kind); net = VP.Net(VP.V, VP.L, 'std')
            f = os.path.join(HERE, 'results_v230', 'protocol', 'gpu0' if kind == 'pos' else 'gpu1', 'v230_std_%s_s%d.pt' % (kind, seed))
            net.load_state_dict(torch.load(f, map_location='cpu', weights_only=True)); net.eval()
            X, ok, y = PR.states(net, task, its, plist)
            cops = [(g('x'), PR.P), (g('z'), PR.P), (g('w'), PR.P), (g('op'), 2), (y.astype(int), PR.P)]
            for pi, pos in enumerate(plist):
                row = {}
                for li, l in enumerate(PR.BAND):
                    Z = X[:, pi, li]
                    row[l] = dict(s=ridge_acc(Z, fam, cops, s), rand=float(np.mean([ridge_acc(Z, fam, cops, r) for r in rand])))
                best = max(row, key=lambda l: row[l]['s'])
                out['%s_%d_%d' % (kind, seed, pos)] = row
                print('%s seed %d pos %2d: s best %.3f (layer %d), random functions there %.3f | s by layer %s' % (
                    kind, seed, pos, row[best]['s'], best, row[best]['rand'], ' '.join('%.2f' % row[l]['s'] for l in PR.BAND)), flush=True)
    json.dump(out, open(os.path.join(HERE, 'results_v230', 'posthoc_readability.json'), 'w'), indent=1)
