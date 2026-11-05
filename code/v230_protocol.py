"""v230 protocol arm (docs/prereg_v230_positive_control.md): does the note's protocol say yes where DAS says an
unwritten intermediate is computed and used?

Models: v230_pilot's std 'pos' models, seeds 0-4, retrained on their pilot schedules (training is bitwise
deterministic: each checkpoint's sha256 must equal the one DAS was measured on, else the run stops), and 'neg' models
(y = T1[x] + T2[z] op w; x op z never needed), seeds 0-4, 30k schedule, gate held-out accuracy >= 0.95.

Items per k in {6, 0}: 100 families x 60 (x, z) pairs (no repeats within a family) x both operations (twins share x, z
and w; w drawn per pair), 12,000 items, seed 2300 + k. Positions: v230_pilot.positions(k) (';', a middle filler, the
last pre-w token, w, '='). States: residual after blocks 1-5 (BAND; every block output a later position can read).

1. Share swap (v224_share.analyse, ported to mod 23): per version (op), position and layer in BAND, family-demeaned
   states; copies one-hot x, z, w and the answer y, family-demeaned, regressed out on training folds (5 by family,
   v178's seed); share_L = 1 - sum |r - mu_L|^2 / sum |r|^2 on held-out folds, averaged over BAND. Labels S = (x + z)
   mod 23, M = (x - z) mod 23, R0..R3 (S or M of a shuffled pair, seed 2231). N, D+, D-, I, cluster bootstrap over
   families (10,000, seed 2241, 97.5% CIs); UNSCORABLE (accuracy < 0.80 in a version), NO-SIGNAL, FOLLOWS (D+ > 0,
   D- > 0, I CI lower > 0, I >= BAR x N), DOES-NOT-FOLLOW (I CI upper < 0.10 N), UNCLEAR.
   BAR = max(TAU 0.30 (v224), TAU230), TAU230 from null worlds on this design (v224's rule, set before the freeze).
2. Clamp (v215_patch rules, with prereg disclosure 4): at position p, per BAND layer, codes fitted on the other half of
   the families (split seed 2151) by least squares on one-hot s, x, z, w, op (s the computed value: x + z on plus,
   x - z on minus) -> U = s's code, D = z's code, each keeping the directions above the largest singular value of 5
   shuffled-label fits (seed 2154); Uo = U orthogonalised against D; R random with Uo's dimension (seed 2152). Pairs (1,000 per position, seed 2151): base x correct,
   source the same op, x and w with a different z (so s and the answer differ, w does not). Conditions at token p:
   none, cB (U), cBo (Uo), cD (D), cR (R) at every BAND layer (h <- h - U U^T (h - h_src)), full (the whole state at
   every block output 1-6). Outcome: argmax at '='; hit = the source's answer. Statistic hit(cBo) - hit(cR), cluster
   bootstrap over the base's family, 10,000, seed 2153, 97.5% CI: UNSCORABLE hit(full) < 0.30 | USED >= 0.10 and CI
   lower > 0 | NOT-USED CI upper < 0.02 | UNCLEAR.

  python v230_protocol.py --selftest   synthetic share worlds; port check against v224_share on its own design
  python v230_protocol.py --nullbar    null worlds seeds 1-6 -> TAU230 (pre-freeze)
  python v230_protocol.py --known <ckpt> <seed>   clamp known answer on a real checkpoint: w at its own token (pre-freeze)
  python v230_protocol.py pos|neg      models + both arms (GPU; OUT_DIR)
  python v230_protocol.py null         fresh null seeds 7-12 (CPU)
  python v230_protocol.py verdict <root>
"""
import glob, hashlib, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

import v178_analyze as V178

OUT = os.environ.get('OUT_DIR', '.'); TINY = os.environ.get('TINY') == '1'
P = 23; BAND = [1, 2] if TINY else [1, 2, 3, 4, 5]; LABS = ['S', 'M', 'R0', 'R1', 'R2', 'R3']
TAU = 0.30          # v224
TAU230 = 0.35       # set 2026-11-01 by --nullbar (largest I/N among null runs with signal 0.225: fourier, lam 0.05)
NULL_KINDS = ('pair', 'fourier', 'mlp', 'paircode', 'paircode_op')
LAMS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0); DD = 192
NFAM, NPAIR, NCLAMP = (10, 30, 50) if TINY else (100, 60, 1000)
NBOOT = 200 if TINY else 10000
# checkpoints DAS was measured on (pilots 3-5; sha256 first 16 hex) and their schedules (STEPS)
SHA = {0: '423563683be2c8f9', 1: '0a46ffeb90ae782b', 2: '39aac1088343cf2d', 3: 'fe7a13c4e6aa5fd9', 4: '017e54b930608bae'}
STEPS_OF = {0: 30000, 1: 30000, 2: 60000, 3: 30000, 4: 30000}
# ground truth (prereg section 3; pilot DAS): positive ';' (S >= 0.70 at some layer, X and Z <= 0.15 at every layer),
# negative (S <= 0.15 at every layer), indeterminate otherwise (seed 0's '=': 0.13-0.18 at layer 5)
GT_POS = {6: 3, 0: 9}
GT_IND = {(0, 6): [11], (0, 0): [11]}


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v230_protocol_log.txt'), 'a').write(s + '\n')


def items(k):
    rng = np.random.default_rng(2300 + k); its = []
    for f in range(NFAM):
        for pz in rng.choice(P * P, NPAIR, replace=False):
            x, z = divmod(int(pz), P); w = int(rng.integers(P))
            for op in (0, 1):
                s = (x + z) % P if op == 0 else (x - z) % P
                its.append(dict(fam=f, ver=('plus', 'minus')[op], op=op, x=x, z=z, w=w, k=k, s=s, u=x, v=z))
    return its


def demean(X, fam):
    X = np.array(X, dtype=np.float32)
    for f in np.unique(fam):
        X[fam == f] -= X[fam == f].mean(0)
    return X


# ---------------------------------------------------------------- share swap (v224_share, general modulus and copies)

def labels(u, v, mod):
    pairs = sorted(set(zip(u.tolist(), v.tolist()))); rng = np.random.default_rng(2231)
    out = {'S': (u + v) % mod, 'M': (u - v) % mod}
    for r in range(4):
        perm = rng.permutation(len(pairs)); g = (lambda a, b: (a + b) % mod) if r < 2 else (lambda a, b: (a - b) % mod)
        mp = {pairs[j]: g(*pairs[perm[j]]) for j in range(len(pairs))}
        out['R%d' % r] = np.array([mp[p] for p in zip(u.tolist(), v.tolist())])
    return out


def contrib(Z, fam, cops, ys):
    """cops: list of (int array, n classes) copies, one-hot and regressed out. Held-out res and err_L per item."""
    Z = demean(Z, fam); C = np.zeros((len(Z), sum(n for _, n in cops)), np.float32); o = 0
    for c, n in cops:
        C[np.arange(len(Z)), o + c] = 1; o += n
    C = demean(C, fam); fold = V178.folds(fam); res = np.zeros(len(Z)); err = {k: np.zeros(len(Z)) for k in ys}
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        W = np.linalg.lstsq(C[tr], Z[tr], rcond=None)[0]; Rtr = Z[tr] - C[tr] @ W; Rte = Z[te] - C[te] @ W
        res[te] = (Rte ** 2).sum(1)
        for k, y in ys.items():
            mu = {c: Rtr[y[tr] == c].mean(0) for c in np.unique(y[tr])}
            M = np.stack([mu.get(c, np.zeros(Z.shape[1], np.float32)) for c in y[te]])
            err[k][te] = ((Rte - M) ** 2).sum(1)
    return res, err


def analyse(get, its, ok, layers, cop_keys, mod=P, ncls=P, nboot=NBOOT, seed=2241, bar=None):
    """v224_share.analyse with the modulus, the copy set and the layers as arguments. get(idx, layer) -> states."""
    fams = sorted(set(i['fam'] for i in its)); fx = {f: j for j, f in enumerate(fams)}
    vers = ('plus', 'minus'); F = len(fams); keys = LABS
    Sres = np.zeros((2, len(layers), F)); Serr = np.zeros((2, len(layers), len(keys), F)); g1 = {}
    for vi, ver in enumerate(vers):
        idx = np.array([k for k, i in enumerate(its) if i['ver'] == ver]); g1[ver] = float(np.mean([ok[k] for k in idx]))
        fam = np.array([its[k]['fam'] for k in idx]); u = np.array([its[k]['u'] for k in idx]); v = np.array([its[k]['v'] for k in idx])
        cops = [(np.array([its[k][c] for k in idx]), ncls) for c in cop_keys]
        ys = labels(u, v, mod); fi = np.array([fx[f] for f in fam])
        for li, l in enumerate(layers):
            res, err = contrib(np.asarray(get(idx, l), np.float32), fam, cops, ys)
            np.add.at(Sres[vi, li], fi, res)
            for ki, k in enumerate(keys):
                np.add.at(Serr[vi, li, ki], fi, err[k])

    def stat(w, per_layer=False):
        sh = 1 - np.einsum('bf,vlkf->bvlk', w, Serr) / np.einsum('bf,vlf->bvl', w, Sres)[..., None]
        if per_layer:
            return sh
        sh = sh.mean(2)
        S_p, M_p, S_m, M_m = sh[:, 0, 0], sh[:, 0, 1], sh[:, 1, 0], sh[:, 1, 1]
        N = (S_p + M_m) / 2; Rb = sh[:, :, 2:].mean((1, 2)); Dp, Dm = S_p - M_p, M_m - S_m
        return dict(N=N, Rbar=Rb, lift=N - Rb, Dp=Dp, Dm=Dm, I=(Dp + Dm) / 2, sh=sh)
    pt = stat(np.ones((1, F))); rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, F, F), minlength=F) for _ in range(nboot)]).astype(float)
    bs = stat(W); ci = {k: np.percentile(bs[k], (1.25, 98.75)) for k in ('N', 'lift', 'I', 'Dp', 'Dm')}
    r = {k: float(pt[k][0]) for k in ('N', 'Rbar', 'lift', 'Dp', 'Dm', 'I')}
    shl = stat(np.ones((1, F)), True)[0]
    r.update(ci={k: [float(x) for x in c] for k, c in ci.items()}, I_rel=r['I'] / r['N'] if r['N'] > 0 else float('nan'), g1=g1,
             shares={'%s_%s' % (ver, k): float(pt['sh'][0, vi, ki]) for vi, ver in enumerate(vers) for ki, k in enumerate(keys)},
             shares_by_layer={'%s_%s' % (ver, k): [float(x) for x in shl[vi, :, ki]] for vi, ver in enumerate(vers) for k in ('S', 'M')
                              for ki in [keys.index(k)]})
    if min(g1.values()) < 0.80:
        lab = 'UNSCORABLE'
    elif ci['N'][0] <= 0:
        lab = 'NO-SIGNAL'
    elif bar is not None and r['Dp'] > 0 and r['Dm'] > 0 and ci['I'][0] > 0 and r['I'] >= bar * r['N']:
        lab = 'FOLLOWS'
    elif ci['I'][1] < 0.10 * r['N']:
        lab = 'DOES-NOT-FOLLOW'
    else:
        lab = 'UNCLEAR'
    r['label'] = lab
    return r


def fmt(name, r):
    return ('%-8s N %.3f  Rbar %.3f  lift %+.3f  D+ %+.3f  D- %+.3f  I %+.3f [%+.3f, %+.3f]  I/N %+.2f  -> %s' % (
        name, r['N'], r['Rbar'], r['lift'], r['Dp'], r['Dm'], r['I'], *r['ci']['I'], r['I_rel'], r['label']))


def bar():
    return max(TAU, TAU230) if TAU230 is not None else None


# ---------------------------------------------------------------- null worlds on this design

def answer(its, kind='pos'):
    import v230_pilot as VP
    import torch
    t = lambda key: torch.tensor([i[key] for i in its])
    return VP.Task(kind).answer(t('op'), t('x'), t('z'), t('w')).numpy()


def world(its, kind, lam, seed, comp=0.0):
    """Copies U1[x] + U2[z] + U3[w] + U4[y] (y = the pos answer), one mixture of (x, z, op) at lam x the copies' RMS,
    family offsets, N(0, 1) noise per layer, d 192. comp > 0 adds a code of s = x op z (the computed value) at
    comp x the copies' RMS (positive worlds, selftest only)."""
    rng = np.random.default_rng(seed); n = len(its)
    x, z, w, o = (np.array([i[c] for i in its]) for c in ('x', 'z', 'w', 'op')); y = answer(its)
    U = rng.standard_normal((4, P, DD)); cop = U[0][x] + U[1][z] + U[2][w] + U[3][y]; rms = np.sqrt((cop ** 2).sum(1).mean())
    sc = lambda a: a * rms / np.sqrt((a ** 2).sum(1).mean())
    if kind == 'pair':
        mix = sum(np.tanh(2 * rng.standard_normal((m1, DD))[a] * rng.standard_normal((m2, DD))[b])
                  for a, b, m1, m2 in ((x, z, P, P), (x, o, P, 2), (z, o, P, 2)))
    elif kind == 'fourier':
        mix = np.zeros((n, DD))
        for f in (1, 2, 5):
            c1, s1, c2, s2 = (g(2 * np.pi * f * a / P) for a in (x, z) for g in (np.cos, np.sin))
            Fm = np.stack([c1 * c2, s1 * s2, c1 * s2, s1 * c2], 1); R = rng.standard_normal((2, 4, DD))
            mix += np.einsum('nk,nkd->nd', Fm, R[o])
    elif kind == 'mlp':
        H = np.concatenate([np.eye(P)[x], np.eye(P)[z], np.eye(2)[o]], 1)
        mix = np.maximum(H @ rng.standard_normal((2 * P + 2, 256)) + rng.standard_normal(256), 0) @ rng.standard_normal((256, DD))
    elif kind == 'paircode':
        mix = rng.standard_normal((P, P, DD))[x, z]
    elif kind == 'paircode_op':
        mix = rng.standard_normal((2, P, P, DD))[o, x, z]
    elif kind == 'copies':
        mix = np.zeros((n, DD))
    else:
        raise ValueError(kind)
    sig = cop + (lam * sc(mix) if mix.any() else 0)
    if comp:
        s = np.array([i['s'] for i in its]); sig = sig + comp * sc(rng.standard_normal((P, DD))[s])
    offs = {f: rng.standard_normal(DD) * 3 for f in set(i['fam'] for i in its)}
    sig = sig + np.stack([offs[i['fam']] for i in its])
    return {l: (sig + rng.standard_normal((n, DD))).astype(np.float32) for l in BAND}


def run_world(its, kind, lam, seed, comp=0.0, nboot=1000, b=None):
    Xw = world(its, kind, lam, seed, comp); y = answer(its); its = [dict(i, y=int(a)) for i, a in zip(its, y)]
    return analyse(lambda idx, l: Xw[l][idx], its, [True] * len(its), BAND, ('x', 'z', 'w', 'y'), nboot=nboot, bar=b)


def _null_job(a):
    kind, lam, seed = a; r = run_world(items(6), kind, lam, seed, b=bar())
    return dict(kind=kind, lam=lam, seed=seed, spec=bool(r['ci']['N'][0] > 0), I=r['I'], N=r['N'], I_rel=r['I_rel'], label=r['label'])


def null_grid(seeds, say=print, procs=4):
    from multiprocessing import Pool
    jobs = [(kind, lam, seed) for kind in NULL_KINDS for lam in LAMS for seed in seeds]
    with Pool(procs) as pool:
        rows = pool.map(_null_job, jobs, chunksize=1)
    for kind in NULL_KINDS:
        ks = [q for q in rows if q['kind'] == kind]; sp = [q['I_rel'] for q in ks if q['spec'] and np.isfinite(q['I_rel'])]
        say('null %-11s: signal in %2d/%d runs; I/N max among them %s; FOLLOWS %d' % (
            kind, len(sp), len(ks), '%+.3f' % max(sp) if sp else '-', sum(q['label'] == 'FOLLOWS' for q in ks)))
    return rows


def nullbar(say=print):
    rows = null_grid(range(1, 7), say)
    sp = [q['I_rel'] for q in rows if q['spec'] and np.isfinite(q['I_rel'])]; worst = max(sp) if sp else 0.0
    tau = max(0.20, float(np.ceil(1.5 * worst / 0.05 - 1e-9) * 0.05))
    say('largest I/N among null runs with signal %.3f -> TAU230 = %.2f; FOLLOWS bar max(0.30, TAU230) = %.2f' % (worst, tau, max(TAU, tau)))
    return tau, rows


def selftest():
    import v224_share as V224
    ok = True
    # port check: on v224's own b design and world, this analyse (mod 5, copies u, v with 10 classes) must equal v224's
    its = V224.design('b'); Xw = V224.world(its, 'pair', 1.0, 1, 0.5)
    a = V224.analyse(lambda idx, l: Xw[l][idx], its, [True] * len(its), nboot=200, tau=0.30)
    b = analyse(lambda idx, l: Xw[l][idx], its, [True] * len(its), V224.SL, ('u', 'v'), mod=5, ncls=10, nboot=200, bar=0.30)
    same = all(abs(a[k] - b[k]) < 1e-9 for k in ('N', 'Dp', 'Dm', 'I')) and a['ci'] == b['ci'] and a['label'] == b['label']
    ok &= same; print('port check vs v224_share (b design, computed c 0.5): %s  v224 %s | here %s' % (
        'OK' if same else 'FAIL', fmt('', a), fmt('', b)), flush=True)
    its = items(6)
    for name, kind, lam, comp, want in (('computed c=0.25', 'pair', 1.0, 0.25, 'FOLLOWS'), ('computed c=1', 'pair', 1.0, 1.0, 'FOLLOWS'),
                                        ('copies only', 'copies', 0.0, 0.0, 'not FOLLOWS'), ('paircode lam 2', 'paircode', 2.0, 0.0, 'not FOLLOWS'),
                                        ('paircode_op lam 2', 'paircode_op', 2.0, 0.0, 'not FOLLOWS'), ('fourier lam 2', 'fourier', 2.0, 0.0, 'not FOLLOWS')):
        r = run_world(its, kind, lam, 1, comp, nboot=200, b=0.30)
        good = (r['label'] == want) if want != 'not FOLLOWS' else r['label'] != 'FOLLOWS'; ok &= good
        print('selftest %-18s %s %s' % (name, fmt('', r), 'OK' if good else 'FAIL'), flush=True)
    return ok


# ---------------------------------------------------------------- models, states, clamp

def model(kind, seed):
    import torch, v230_pilot as VP
    VP.OUT = OUT; VP.STEPS = 40 if TINY else STEPS_OF[seed] if kind == 'pos' else 30000
    net, task, tr = VP.train('std', kind, seed)
    f = os.path.join(OUT, 'v230_std_%s_s%d.pt' % (kind, seed)); sha = hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]
    log('model %s seed %d: sha %s (expected %s) | held-out %.4f' % (kind, seed, sha, SHA[seed] if kind == 'pos' else '-', tr['final_acc']))
    if kind == 'pos' and not TINY and sha != SHA[seed]:
        sys.exit('checkpoint differs from the one DAS measured: stop (prereg section 7)')
    for p_ in net.parameters():
        p_.requires_grad_(False)
    return net, task, dict(sha=sha, **{k: tr[k] for k in ('step_to_0p99', 'final_acc', 'steps')})


def states(net, task, its, plist):
    import torch, v230_pilot as VP
    c = torch.tensor([[i['op'], i['x'], i['z'], i['w'], i['k']] for i in its]); seq = task.encode(c)
    X = np.zeros((len(its), len(plist), len(BAND), net.emb.weight.shape[1]), np.float32); ok = np.zeros(len(its), bool)
    with torch.no_grad():
        for b in range(0, len(its), 2048):
            idx = seq[b:b + 2048, :-1].to(VP.DEV); x = net.emb(idx) + net.pos[:, :idx.shape[1]]; i = 0
            for blk in net.blocks:
                x = blk(x); i += 1
                if i in BAND:
                    X[b:b + 2048, :, BAND.index(i)] = x[:, plist].float().cpu().numpy()
            ok[b:b + 2048] = (net.head(net.lnf(x))[:, -1].argmax(-1).cpu() == seq[b:b + 2048, -1]).numpy()
    return X, ok, seq[:, -1].numpy()


def clamp(net, task, its, ok, X, plist, pos, var='s', say=log):
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

    def code(Zm, m, f):
        _, S, Vt = np.linalg.svd(fit(Zm, m)[f], full_matrices=False)
        null = max(float(np.linalg.svd(fit(Zm, m, f)[f], compute_uv=False)[0]) for _ in range(5))
        return Vt[:int((S > null).sum())].T
    for li, l in enumerate(BAND):
        Z = demean(X[:, pi, li], fam)
        for h in (0, 1):
            m = hf != h                                                         # fitted on the OTHER half
            U, Dd = code(Z[m], m, var), code(Z[m], m, 'z'); Uo = U
            if U.shape[1]:
                Uo, sv, _ = np.linalg.svd(U - Dd @ (Dd.T @ U), full_matrices=False); Uo = Uo[:, sv > 1e-3]   # U orthonormal
            R = np.linalg.qr(rr.standard_normal((d, P - 1)))[0][:, :Uo.shape[1]]          # random, Uo's dimension
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


def run_models(kind):
    import v230_pilot as VP
    if TAU230 is None and not TINY:
        sys.exit('TAU230 not set: the null bar must be frozen into this file first')
    log('v230 protocol | %s models | device %s | TINY %s | FOLLOWS bar %s' % (kind, VP.DEV, TINY, bar()))
    for seed in range(5):
        t0 = time.time(); net, task, info = model(kind, seed); res = dict(kind=kind, seed=seed, model=info, cells={})
        if kind == 'neg' and not TINY and info['final_acc'] < 0.95:
            log('  neg seed %d below the 0.95 gate: UNSCORABLE' % seed); res['unscorable'] = True
            json.dump(res, open(os.path.join(OUT, 'v230_protocol_%s_s%d.json' % (kind, seed)), 'w'), indent=1); continue
        for k in (6, 0):
            its = items(k); plist = VP.positions(k); X, ok, y = states(net, task, its, plist)
            its = [dict(i, y=int(a)) for i, a in zip(its, y)]
            log('  seed %d k %d: %d items, correct %.4f' % (seed, k, len(its), ok.mean()))
            for pos in plist:
                pi = plist.index(pos)
                r = analyse(lambda idx, l: X[idx, pi, BAND.index(l)], its, ok, BAND, ('x', 'z', 'w', 'y'), bar=bar())
                log('  share ' + fmt('k%d p%d' % (k, pos), r))
                c = clamp(net, task, its, ok, X, plist, pos)
                res['cells']['%d_%d' % (k, pos)] = dict(share=r, clamp=c)
        log('  seed %d done in %d s' % (seed, time.time() - t0))
        json.dump(res, open(os.path.join(OUT, 'v230_protocol_%s_s%d.json' % (kind, seed)), 'w'), indent=1)


def known(ckpt, seed):
    """Pre-freeze known-answer check of the clamp: w's own code at w's token (pairs differ in w only), k 6."""
    import torch, v230_pilot as VP
    task = VP.Task('pos'); net = VP.Net(VP.V, VP.L, 'std').to(VP.DEV)
    net.load_state_dict(torch.load(ckpt, map_location=VP.DEV, weights_only=True)); net.eval()
    for p_ in net.parameters():
        p_.requires_grad_(False)
    its = items(6); plist = VP.positions(6); X, ok, y = states(net, task, its, plist)
    print('seed %s: correct %.4f' % (seed, ok.mean()), flush=True)
    return clamp(net, task, its, ok, X, plist, 10, var='w', say=print)


def verdict(root):
    R = {}
    for f in glob.glob(os.path.join(root, '**', 'v230_protocol_*_s*.json'), recursive=True):
        r = json.load(open(f)); R[(r['kind'], r['seed'])] = r
    nulls = json.load(open(glob.glob(os.path.join(root, '**', 'v230_null_fresh.json'), recursive=True)[0]))
    nf = sum(q['label'] == 'FOLLOWS' for q in nulls)
    want = len(NULL_KINDS) * len(LAMS) * 6                                    # 5 + 5 models, 270 null runs
    got = (sum(kd == 'pos' for kd, _ in R), sum(kd == 'neg' for kd, _ in R), len(nulls))
    log('verdict inputs: %d pos, %d neg models, %d fresh null runs (expected 5, 5, %d)' % (*got, want))
    if got != (5, 5, want) and not TINY:
        sys.exit('missing outputs: no verdict')
    out = dict(null_follows=nf, arms={})
    for arm, yes in (('share', 'FOLLOWS'), ('clamp', 'USED')):
        neg_fp = [(s, c) for (kd, s), r in R.items() if kd == 'neg' for c, v in r['cells'].items() if v[arm]['label'] == yes]
        gt_fp = [(s, c) for (kd, s), r in R.items() if kd == 'pos' for c, v in r['cells'].items()
                 if int(c.split('_')[1]) != GT_POS[int(c.split('_')[0])] and int(c.split('_')[1]) not in GT_IND.get((s, int(c.split('_')[0])), [])
                 and v[arm]['label'] == yes] if arm == 'clamp' else []
        specific = not neg_fp and not gt_fp and (nf == 0 or arm == 'clamp')
        a = dict(neg_false_pos=neg_fp, gt_negative_false_pos=gt_fp, specific=specific)
        for k in (6, 0):
            labs = {s: R[('pos', s)]['cells']['%d_%d' % (k, GT_POS[k])][arm]['label'] for s in range(5) if ('pos', s) in R}
            n = sum(v == yes for v in labs.values())
            a['k%d' % k] = dict(labels=labs, n_yes=n, power='DETECTS' if n >= 4 else 'SEED-DEPENDENT' if n >= 2 else 'MISSES')
        out['arms'][arm] = a
        log('%s: k6 %s (%d/5) | k0 %s (%d/5) | specific %s (neg-model yes %s, DAS-negative yes %s%s)' % (
            arm, a['k6']['power'], a['k6']['n_yes'], a['k0']['power'], a['k0']['n_yes'], specific, neg_fp, gt_fp,
            ', fresh null FOLLOWS %d' % nf if arm == 'share' else ''))
    json.dump(out, open(os.path.join(OUT, 'v230_verdict.json'), 'w'), indent=1)


if __name__ == '__main__':
    a = sys.argv[1]
    if a == '--selftest':
        sys.exit(0 if selftest() else 1)
    elif a == '--nullbar':
        t, rows = nullbar(); json.dump(dict(tau230=t, rows=rows), open(os.path.join(OUT, 'v230_nullbar.json'), 'w'), indent=1)
    elif a == '--known':
        known(sys.argv[2], sys.argv[3])
    elif a in ('pos', 'neg'):
        run_models(a)
    elif a == 'null':
        if TAU230 is None and not TINY:
            sys.exit('TAU230 not set: the null bar must be frozen into this file first')
        rows = null_grid(range(7, 13), log); json.dump(rows, open(os.path.join(OUT, 'v230_null_fresh.json'), 'w'), indent=1)
    elif a == 'verdict':
        verdict(sys.argv[2])
