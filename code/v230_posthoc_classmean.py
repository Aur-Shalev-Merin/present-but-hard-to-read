"""POST HOC (after the v230 result): v215's original clamp codes (separate family-demeaned class means of s and of the
z copy; Uo kept at sv > 1e-3 * max; random subspace of P - 1 dims) on the five frozen v230 pos checkpoints, same pairs.
Question: would the clamp as used for Ouro / Kudo have detected s at ';'?"""
import sys, json, os
REPO = os.path.dirname(os.path.abspath(__file__))   # folder with v230_protocol.py and results_v230/ (an absolute local path in the original)
sys.path.insert(0, REPO)
import numpy as np
from v230_protocol import *
def clamp_mean(net, task, its, ok, X, plist, pos, var='s', say=log):
    """v215_patch at token pos. var 's': the computed value, pairs differ in z (main). var 'w': w's code, pairs differ
    in w only (known-answer check, pre-freeze)."""
    import torch, v230_pilot as VP
    pi = plist.index(pos); fam = np.array([i['fam'] for i in its])
    fams = sorted(set(fam.tolist())); half = {f: int(j % 2) for j, f in enumerate(np.random.default_rng(2151).permutation(fams))}
    hf = np.array([half[f] for f in fam]); rr = np.random.default_rng(2152); d = X.shape[-1]; B = {c: {0: {}, 1: {}} for c in ('cB', 'cBo', 'cD', 'cR')}
    geo = []
    # class codes taken jointly (least squares on one-hot s, x, z, w, op; family-demeaned), not as separate class means:
    # w is drawn per pair, so separate class means of z carry w's code through sampling imbalance (pre-freeze known-
    # answer check: z's subspace moved w at w's token in 40-97% of pairs). Each factor's coefficients, centred over its
    # classes, are its code adjusted for the others.
    # Rank: singular values above the largest from 5 shuffled-label fits (the pilot's copy-span rule): a weak factor's
    # full 22-dim code spans the highest-variance noise directions (known-answer check: z's code at w's token still
    # moved w in 11-50% of pairs), so only directions above the null are kept, for the subspace and for the copy.
    facs = ['s', 'x', 'z', 'w', 'op']; ncl = {'op': 2}; vals = {f: np.array([i[f] for i in its]) for f in facs}
    off = np.cumsum([0] + [ncl.get(f, P) for f in facs]); rs = np.random.default_rng(2154)

    def fit(Zm, m, shuffle=None):                                               # centred code of each factor
        vs = {g: (rs.permutation(vals[g][m]) if g == shuffle else vals[g][m]) for g in facs}
        H = demean(np.concatenate([np.eye(ncl.get(g, P), dtype=np.float32)[vs[g]] for g in facs], 1), fam[m])
        coef = np.linalg.lstsq(H, Zm, rcond=None)[0]
        return {g: coef[off[j]:off[j + 1]] - coef[off[j]:off[j + 1]].mean(0) for j, g in enumerate(facs)}

    def code(Zm, m, f):                                                         # POST HOC: v215's separate class means
        v = vals[f][m]; M = np.stack([Zm[v == c].mean(0) for c in np.unique(v)]); M -= M.mean(0)
        return np.linalg.svd(M, full_matrices=False)[2][:len(M) - 1].T
    for li, l in enumerate(BAND):
        Z = demean(X[:, pi, li], fam)
        for h in (0, 1):
            m = hf != h                                                         # fitted on the OTHER half
            U, Dd = code(Z[m], m, var), code(Z[m], m, 'z'); Uo = U
            if U.shape[1]:
                Uo, sv, _ = np.linalg.svd(U - Dd @ (Dd.T @ U), full_matrices=False); Uo = Uo[:, sv > 1e-3 * sv.max()]   # v215 rule
            R = np.linalg.qr(rr.standard_normal((d, P - 1)))[0]   # v215: class count - 1 dims          # random, Uo's dimension
            for c, A in (('cB', U), ('cBo', Uo), ('cD', Dd), ('cR', R)):
                B[c][h][l] = torch.tensor(A, dtype=torch.float32, device=VP.DEV)
            ov = float((np.linalg.svd(U.T @ Dd, compute_uv=False) ** 2).mean()) if U.shape[1] and Dd.shape[1] else 0.0
            geo.append(dict(layer=l, half=h, dims=U.shape[1], dims_copy=Dd.shape[1], dims_o=Uo.shape[1], overlap=ov))
    rng = np.random.default_rng(2151); cand = np.flatnonzero(ok); base = rng.permutation(cand)[:NCLAMP]
    if len(base) < 2:
        say('  clamp %s pos %2d: no correct base items -> UNSCORABLE' % (var, pos))
        return dict(pos=pos, var=var, n=len(base), label='UNSCORABLE', geometry=geo)
    bc = torch.tensor([[its[j]['op'], its[j]['x'], its[j]['z'], its[j]['w'], its[j]['k']] for j in base]); sc = bc.clone()
    col = 2 if var == 's' else 3
    sc[:, col] = torch.tensor([(int(v) + 1 + int(rng.integers(P - 1))) % P for v in bc[:, col]])     # a different value
    bseq, sseq = task.encode(bc), task.encode(sc); bi, si = bseq[:, :-1].to(VP.DEV), sseq[:, :-1].to(VP.DEV)
    hb = torch.tensor(hf[base], device=VP.DEV)

    def fwd(idx, src=None, U=None, full=False):
        x = net.emb(idx) + net.pos[:, :idx.shape[1]]; i = 0; cap = {}
        for blk in net.blocks:
            x = blk(x); i += 1; cap[i] = x[:, pos].clone()
            if full:
                x[:, pos] = src[i]
            elif U is not None and i in BAND:
                Ui = U(i); dd = x[:, pos] - src[i]
                x[:, pos] = x[:, pos] - torch.einsum('nd,nde,nfe->nf', dd, Ui, Ui)
        return net.head(net.lnf(x))[:, -1].argmax(-1), cap
    with torch.no_grad():
        _, src = fwd(si)
        out = {'none': fwd(bi)[0], 'full': fwd(bi, src, full=True)[0]}
        for c in ('cB', 'cBo', 'cD', 'cR'):
            def Uf(i, c=c):                                                     # per-pair subspace by the base's half
                A0, A1 = B[c][0][i], B[c][1][i]; m = max(A0.shape[1], A1.shape[1])
                pad = lambda A: torch.nn.functional.pad(A, (0, m - A.shape[1]))
                return torch.where(hb[:, None, None] == 0, pad(A0)[None], pad(A1)[None])
            out[c] = fwd(bi, src, Uf)[0]
    tgt, own = sseq[:, -1].to(VP.DEV), bseq[:, -1].to(VP.DEV)
    hit = {c: (v == tgt).float().cpu().numpy() for c, v in out.items()}; kept = {c: float((v == own).float().mean()) for c, v in out.items()}
    dlt = hit['cBo'] - hit['cR']; bf = fam[base]; u = np.unique(bf); ix = {f: j for j, f in enumerate(u)}
    S = np.zeros(len(u)); Nn = np.zeros(len(u)); np.add.at(S, [ix[f] for f in bf], dlt); np.add.at(Nn, [ix[f] for f in bf], 1)
    rg = np.random.default_rng(2153); bs = []
    for _ in range(NBOOT):
        q = rg.integers(0, len(u), len(u)); bs.append(S[q].sum() / Nn[q].sum())
    lo, hi = np.percentile(bs, (1.25, 98.75))
    lab_ = ('UNSCORABLE' if hit['full'].mean() < 0.30 else 'USED' if dlt.mean() >= 0.10 and lo > 0 else
            'NOT-USED' if hi < 0.02 else 'UNCLEAR')
    say('  clamp %s pos %2d (%d pairs): hit %s | kept %s | cBo - cR %+.3f [%+.3f, %+.3f] -> %s' % (
        var, pos, len(base), ' '.join('%s %.3f' % (c, hit[c].mean()) for c in hit), ' '.join('%s %.2f' % (c, kept[c]) for c in kept),
        dlt.mean(), lo, hi, lab_))
    return dict(pos=pos, var=var, n=len(base), hit={c: float(v.mean()) for c, v in hit.items()}, kept=kept, diff=float(dlt.mean()),
                ci=[float(lo), float(hi)], label=lab_, geometry=geo)



if __name__ == '__main__':
    import os, torch, v230_pilot as VP
    VP.DEV = 'cpu'; out = {}
    for seed in range(5):
        task = VP.Task('pos'); net = VP.Net(VP.V, VP.L, 'std')
        f = os.path.join(REPO, 'results_v230', 'protocol', 'gpu0', 'v230_std_pos_s%d.pt' % seed)
        net.load_state_dict(torch.load(f, map_location='cpu', weights_only=True)); net.eval()
        for p_ in net.parameters():
            p_.requires_grad_(False)
        for k in (6, 0):
            its = items(k); plist = VP.positions(k); X, ok, y = states(net, task, its, plist)
            for pos in plist:
                print('seed %d k %d' % (seed, k), end='', flush=True)
                r = clamp_mean(net, task, its, ok, X, plist, pos, say=print); r.pop('geometry', None)
                out['%d_%d_%d' % (seed, k, pos)] = r
    json.dump(out, open(os.path.join(REPO, 'results_v230', 'posthoc_classmean_clamp.json'), 'w'), indent=1)
