"""v220 measurement at one training checkpoint (prereg docs/prereg_v220_training_order.md).

Probe set (fixed, seed 220):
  dig  per operation 8 families; a family fixes the tens digits (ta, tb, tc) and holds all 1,000 unit triples
       (ua, ub, uc), every one a valid problem (answer in 10..99). Labels mod 5 depend only on the units digits
       (10 = 0 mod 5), so within a family every label varies and the answer is not a function of a+b.
  mod  per operation 4,000 random (a, b, c) triples; labels mod 23.
Labels per operation (op 0: a+b-c, op 1: a-b-c):
  p1   the first partial, a op b          twin  the other sign (a-b on op 0, a+b on op 1)
  ac   a-c                                bc    b-c on op 0, b+c on op 1 (a-b-c = a-(b+c))
  ans  the answer
Positions: b (end of b), c (end of c), out (where the answer's last token is predicted: dig the answer's tens digit,
fed in; mod '=').
Probe: closed-form ridge classifier on standardised states (lambda = 0.01 n_train), one solve for all labels;
dig states are family-demeaned and folds are families (4 folds of 2 per operation); mod uses 5 random folds, no
demeaning. Copy ceiling: the same probe on one-hot copies of the digits visible at the position (dig: ua, ub [, uc]
[, the answer's tens digit]; mod: a, b [, c]). excess = probe accuracy on the state - accuracy on the copies.
Model output: dig the predicted units digit mod 5 at out; mod the predicted answer at out (accuracy vs ans).
"""
import numpy as np
import torch

LABELS = ['p1', 'twin', 'ac', 'bc', 'ans']
POS = {'dig': {'b': 4, 'c': 7, 'out': 9}, 'mod': {'b': 2, 'c': 4, 'out': 5}}
P = 23


def _labels(op, x, y, z, m):
    if op == 0:
        return np.stack([x + y, x - y, x - z, y - z, x + y - z], 1) % m
    return np.stack([x - y, x + y, x - z, y + z, x - y - z], 1) % m


def probe_set(fmt):
    """Returns dict(op -> dict(tok [n, T_in], lab [n, 5], fold [n], group [n] or None, copies {pos: [n, k] ints},
    K classes, ncls per copy))."""
    rng = np.random.default_rng(220); out = {}
    for op in (0, 1):
        if fmt == 'dig':
            ok = lambda k: 2 <= k <= 8 if op == 0 else 3 <= k <= 9
            trip = [(a, b, c) for a in range(1, 10) for b in range(1, 10) for c in range(1, 10)
                    if ok(a + b - c if op == 0 else a - b - c)]
            fam = [trip[i] for i in rng.choice(len(trip), 8, replace=False)]
            u = np.array([(ua, ub, uc) for ua in range(10) for ub in range(10) for uc in range(10)])
            rows = []
            for f, (ta, tb, tc) in enumerate(fam):
                a, b, c = 10 * ta + u[:, 0], 10 * tb + u[:, 1], 10 * tc + u[:, 2]
                r = a + b - c if op == 0 else a - b - c
                assert r.min() >= 10 and r.max() <= 99
                rows.append(np.stack([a, b, c, r, np.full(len(u), f)], 1))
            a, b, c, r, g = np.concatenate(rows).T
            tok = np.stack([a // 10, a % 10, np.full_like(a, 10 + op), b // 10, b % 10, np.full_like(a, 11),
                            c // 10, c % 10, np.full_like(a, 12), r // 10], 1)
            ua, ub, uc = a % 10, b % 10, c % 10
            out[op] = dict(tok=tok, lab=_labels(op, ua, ub, uc, 5), fold=g % 4, group=g, K=5,
                           copies={'b': np.stack([ua, ub], 1), 'c': np.stack([ua, ub, uc], 1),
                                   'out': np.stack([ua, ub, uc, r // 10], 1)}, ncopy=10)
        else:
            idx = rng.choice(P ** 3, 4000, replace=False)
            a, b, c = idx // (P * P), (idx // P) % P, idx % P
            tok = np.stack([a, np.full_like(a, 23 + op), b, np.full_like(a, 24), c, np.full_like(a, 25)], 1)
            out[op] = dict(tok=tok, lab=_labels(op, a, b, c, P), fold=rng.integers(0, 5, len(a)), group=None, K=P,
                           copies={'b': np.stack([a, b], 1), 'c': np.stack([a, b, c], 1),
                                   'out': np.stack([a, b, c], 1)}, ncopy=P)
    return out


def ridge_acc(X, lab, fold, group, K, lam=0.01):
    """Held-out accuracy per label column. X [n, d] float tensor; lab [n, L] ints. Family-demeaned if group given."""
    X = X.double()
    if group is not None:
        g = torch.as_tensor(group, device=X.device)
        mu = torch.zeros(int(g.max()) + 1, X.shape[1], dtype=X.dtype, device=X.device).index_add_(0, g, X)
        X = X - (mu / torch.bincount(g).unsqueeze(1).to(X.dtype))[g]
    lab_t = torch.as_tensor(lab, device=X.device).long(); L = lab.shape[1]
    Y = torch.zeros(len(X), L * K, dtype=X.dtype, device=X.device)
    Y.scatter_(1, lab_t + torch.arange(L, device=X.device) * K, 1.0)
    hit = torch.zeros(L, dtype=X.dtype, device=X.device); fold = np.asarray(fold)
    for f in np.unique(fold):
        tr = torch.as_tensor(fold != f, device=X.device); te = ~tr
        mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        Ytr = Y[tr] - Y[tr].mean(0)
        W = torch.linalg.solve(Xtr.T @ Xtr + lam * tr.sum() * torch.eye(X.shape[1], dtype=X.dtype, device=X.device),
                               Xtr.T @ Ytr)
        pred = (Xte @ W).view(-1, L, K).argmax(-1)
        hit += (pred == lab_t[te]).sum(0)
    return (hit / len(X)).cpu().numpy()


def onehot(cp, ncls):
    return torch.as_tensor(np.concatenate([np.eye(ncls)[cp[:, j]] for j in range(cp.shape[1])], 1))


def copy_ceilings(ps):
    """{pos: [2 ops, 5 labels]} accuracy of the probe on one-hot copies (input-only; computed once per format)."""
    return {pos: np.stack([ridge_acc(onehot(ps[op]['copies'][pos], ps[op]['ncopy']), ps[op]['lab'], ps[op]['fold'],
                                     ps[op]['group'], ps[op]['K']) for op in (0, 1)])
            for pos in ps[0]['copies']}


@torch.no_grad()
def states(net, idx, bs=4096):
    """Residual stream after every block application (list over depth of [n, T, d]) and the final logits."""
    hs, lg = [], []
    for i in range(0, len(idx), bs):
        x = net.emb(idx[i:i + bs]) + net.pos[:, :idx.shape[1]]; h = []
        for _ in range(net.loops):
            for b in net.blocks:
                x = b(x); h.append(x)
        hs.append(torch.stack(h)); lg.append(net.head(net.lnf(x)))
    return torch.cat(hs, 1), torch.cat(lg, 0)


def measure(net, ps, ceil, fmt, dev):
    """excess [3 pos, depth, 2 ops, 5 labels], model-output accuracy [2 ops] (raw; vs the answer label)."""
    was = net.training; net.eval(); pos = POS[fmt]; ex, mod_acc = [], []
    for op in (0, 1):
        d = ps[op]; H, lg = states(net, torch.as_tensor(d['tok'], device=dev))
        e_op = []
        for pn in ('b', 'c', 'out'):
            e_op.append([ridge_acc(H[l, :, pos[pn]].float(), d['lab'], d['fold'], d['group'], d['K']) - ceil[pn][op]
                         for l in range(H.shape[0])])
        ex.append(e_op)
        pr = lg[:, pos['out']].argmax(-1).cpu().numpy()
        acc = ((pr % 5) == d['lab'][:, 4]).mean() if fmt == 'dig' else (pr == d['lab'][:, 4]).mean()
        mod_acc.append(acc)
    net.train(was)
    return np.transpose(np.array(ex), (1, 2, 0, 3)), np.array(mod_acc)


# ---------------------------------------------------------------- selftest (synthetic states, no network)
def selftest():
    """Synthetic dig/mod states: copies only; copies + p1 code; copies + answer code; copies + answer mixed with c
    (answer-then-mix). Prints mean-over-ops excess per label at out and I = mean_op(e_p1 - e_twin)."""
    rng = np.random.default_rng(0)
    for fmt in ('dig', 'mod'):
        ps = probe_set(fmt); ceil = copy_ceilings(ps); K = ps[0]['K']; d = 64
        E = {k: rng.normal(size=(K, d)) for k in ('p1', 'ans', 'mix')}; C = rng.normal(size=(4, ps[0]['ncopy'], d))
        Emix = rng.normal(size=(K, K, d))
        for world in ('copies', 'p1', 'answer', 'answer+c mix'):
            res = []
            for op in (0, 1):
                q = ps[op]; cp = q['copies']['out']; n = len(cp)
                X = sum(C[j][cp[:, j]] for j in range(cp.shape[1])) + rng.normal(size=(n, d)) * 1.0
                if world == 'p1':
                    X += 1.0 * E['p1'][q['lab'][:, 0]]
                if world in ('answer', 'answer+c mix'):
                    X += 1.0 * E['ans'][q['lab'][:, 4]]
                if world == 'answer+c mix':
                    cval = cp[:, 2] % K
                    X += 1.0 * Emix[q['lab'][:, 4], cval]
                if q['group'] is not None:
                    X += rng.normal(size=(8, d))[q['group']] * 3
                res.append(ridge_acc(torch.as_tensor(X), q['lab'], q['fold'], q['group'], K) - ceil['out'][op])
            res = np.array(res); I = (res[:, 0] - res[:, 1]).mean()
            print('%s %-13s excess p1 %+.3f twin %+.3f ac %+.3f bc %+.3f ans %+.3f | I %+.3f'
                  % (fmt, world, *res.mean(0), I))
        print(fmt, 'copy ceilings at out (op0):', np.round(ceil['out'][0], 3), 'at b:', np.round(ceil['b'][0], 3))


if __name__ == '__main__':
    selftest()
