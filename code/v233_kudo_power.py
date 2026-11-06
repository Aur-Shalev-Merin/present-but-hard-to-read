"""v233: clamp power for the form used in the re-test of Kudo et al. (docs/prereg_v233_kudo_power.md).

v231's design, unchanged except for the clamp form. The five v230 pos models (v231's saved checkpoints, sha-checked
against v230), at ';' with k 6, where DAS puts s and the clamp is USED in 5/5. Only the clamp's FITTING states are
degraded (v231's noises(): same seeds, doses, shapes and calibration); the forward pass is untouched.

  form kudo   as the clamps of v215 / v216: separate family-centred class means of s (the candidate, B there), and of
              z (the copy that differs within a pair, d2 there), fitted on the other half of the families; s's basis
              projected off z's (v216: left singular vectors of U - D D^T U with singular value > 1e-3); statistic
              hit(cBo) - hit(cR), cR a random subspace of cBo's dimension. Descriptive: hit(cB) (unprojected, = v231's
              Ouro form), hit(cD) (the copy alone), and the share of s's basis left after the projection
              (||U - D D^T U||^2 / dims; Kudo's clamps kept 0.35-0.64).
  sample      native (all of the fitting half, ~260 items per class) and kudo (782 items, 34 per class: d / n_per_class
              192 / 34 = 5.6; Kudo's 3584 / 600 = 6.0; the same subset as v231's 'ouro' sample).
Pairs, bootstrap and the USED / NOT-USED / UNSCORABLE rule are v230's (v215's).

  python v233_kudo_power.py known <ckpt dir>   CPU: cB equals v231's Ouro-form cB; cBo orthogonal to cD
  python v233_kudo_power.py run <seeds> <ckpt root>   loads v230_std_pos_s<seed>.pt, sha check, grid -> v233_power_s<seed>.json
  python v233_kudo_power.py verdict <root>
"""
import glob, hashlib, json, os, sys, time
import numpy as np

import v230_protocol as PR
import v231_power as V1
from v230_protocol import P, BAND, NCLAMP, NBOOT, demean, items, states

OUT = PR.OUT
DOSES, SHAPES = V1.DOSES, V1.SHAPES
SAMPLES = {'native': None, 'kudo': 782}
POS, K = V1.POS, V1.K
PROJS = ['zx', 'z']      # primary: z and x projected (share of s left matched to Kudo's); secondary: z only (Kudo's procedure)


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v233_log.txt'), 'a').write(s + '\n')


def clamp_k(net, task, its, ok, X, plist, pos, noise=None, sub=None, proj='zx', say=log):
    """v231_power.clamp_g with the kudo form (same random streams: rr draws one random matrix per layer and half).
    proj 'zx' (primary): s projected off z's and x's class means together, which leaves about as much of s as Kudo's
    clamps kept (0.35-0.64; 0.51 on seed 0); 'z' (secondary): off z's alone, Kudo's procedure (0.74 left on seed 0)."""
    import torch, v230_pilot as VP
    pi = plist.index(pos); fam = np.array([i['fam'] for i in its])
    fams = sorted(set(fam.tolist())); half = {f: int(j % 2) for j, f in enumerate(np.random.default_rng(2151).permutation(fams))}
    hf = np.array([half[f] for f in fam]); rr = np.random.default_rng(2152); d = X.shape[-1]
    conds = ('cB', 'cBo', 'cD', 'cR')
    B = {c: {0: {}, 1: {}} for c in conds}; geo = []
    vals = {f: np.array([i[f] for i in its]) for f in ('s', 'z', 'x')}
    pick = {}
    for h in (0, 1):
        m = hf != h
        if sub is not None:
            keep = np.random.default_rng([2311, h]).choice(np.flatnonzero(m), sub, replace=False); m = np.zeros(len(its), bool); m[keep] = True
        pick[h] = m

    def means(Zm, m, f):
        v = vals[f][m]; M = np.stack([Zm[v == c].mean(0) for c in np.unique(v)]); M -= M.mean(0)
        return np.linalg.svd(M, full_matrices=False)[2][:len(M) - 1].T
    for li, l in enumerate(BAND):
        Zall = X[:, pi, li] if noise is None else X[:, pi, li] + noise[l]
        for h in (0, 1):
            m = pick[h]; Z = demean(Zall[m], fam[m])
            U, Dd = means(Z, m, 's'), means(Z, m, 'z')
            if proj == 'zx':
                Dd = np.linalg.qr(np.concatenate([Dd, means(Z, m, 'x')], 1))[0]
            M = U - Dd @ (Dd.T @ U); Uo, sv, _ = np.linalg.svd(M, full_matrices=False); Uo = Uo[:, sv > 1e-3]
            R = np.linalg.qr(rr.standard_normal((d, P - 1)))[0][:, :Uo.shape[1]]
            for c, A in (('cB', U), ('cBo', Uo), ('cD', Dd), ('cR', R)):
                B[c][h][l] = torch.tensor(A, dtype=torch.float32, device=VP.DEV)
            geo.append(dict(layer=l, half=h, dims=U.shape[1], dims_o=Uo.shape[1], n_fit=int(m.sum()),
                            left=float(np.linalg.norm(M) ** 2 / U.shape[1]),
                            orth=float(np.abs(Uo.T @ Dd).max()) if Uo.shape[1] else 0.0))
    rng = np.random.default_rng(2151); cand = np.flatnonzero(ok); base = rng.permutation(cand)[:NCLAMP]
    bc = torch.tensor([[its[j]['op'], its[j]['x'], its[j]['z'], its[j]['w'], its[j]['k']] for j in base]); sc = bc.clone()
    sc[:, 2] = torch.tensor([(int(v) + 1 + int(rng.integers(P - 1))) % P for v in bc[:, 2]])
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
        for c in conds:
            def Uf(i, c=c):
                A0, A1 = B[c][0][i], B[c][1][i]; mm = max(A0.shape[1], A1.shape[1])
                pad = lambda A: torch.nn.functional.pad(A, (0, mm - A.shape[1]))
                return torch.where(hb[:, None, None] == 0, pad(A0)[None], pad(A1)[None])
            out[c] = fwd(bi, src, Uf)[0]
    tgt, own = sseq[:, -1].to(VP.DEV), bseq[:, -1].to(VP.DEV)
    hit = {c: (v == tgt).float().cpu().numpy() for c, v in out.items()}; kept = {c: float((v == own).float().mean()) for c, v in out.items()}
    bf = fam[base]; u = np.unique(bf); ix = np.array([{f: j for j, f in enumerate(u)}[f] for f in bf]); Nn = np.bincount(ix, minlength=len(u))
    sums = {c: np.bincount(ix, hit[c], minlength=len(u)) for c in hit}
    stat = lambda q: (sums['cBo'][q].sum() - sums['cR'][q].sum()) / Nn[q].sum()
    dlt = stat(np.arange(len(u))); rg = np.random.default_rng(2153)
    bs = [stat(rg.integers(0, len(u), len(u))) for _ in range(NBOOT)]
    lo, hi = np.percentile(bs, (1.25, 98.75))
    lab = ('UNSCORABLE' if hit['full'].mean() < 0.30 else 'USED' if dlt >= 0.10 and lo > 0 else 'NOT-USED' if hi < 0.02 else 'UNCLEAR')
    left = float(np.mean([g['left'] for g in geo]))
    say('  clamp kudo-%s pos %d (%d pairs): hit %s | left %.2f | stat %+.3f [%+.3f, %+.3f] -> %s' % (
        proj, pos, len(base), ' '.join('%s %.3f' % (c, hit[c].mean()) for c in hit), left, dlt, lo, hi, lab))
    return dict(pos=pos, form='kudo', proj=proj, n=len(base), hit={c: float(v.mean()) for c, v in hit.items()}, kept=kept, diff=float(dlt),
                ci=[float(lo), float(hi)], label=lab, left=left, geometry=geo)


def load(seed, root):
    import torch, v230_pilot as VP
    f = sorted(glob.glob(os.path.join(root, '**', 'v230_std_pos_s%d.pt' % seed), recursive=True))[0]
    sha = hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]
    if sha != PR.SHA[seed]:
        sys.exit('seed %d checkpoint %s differs from v230 (%s vs %s)' % (seed, f, sha, PR.SHA[seed]))
    task = VP.Task('pos'); net = VP.Net(VP.V, VP.L, 'std')
    net.load_state_dict(torch.load(f, map_location=VP.DEV, weights_only=True)); net.to(VP.DEV).eval()
    for p_ in net.parameters():
        p_.requires_grad_(False)
    return net, task, dict(sha=sha, file=os.path.basename(f))


def grid(net, task, seed, say=log):
    import v230_pilot as VP
    its = items(K); plist = VP.positions(K); X, ok, y = states(net, task, its, plist)
    its = [dict(i, y=int(a)) for i, a in zip(its, y)]; pi = plist.index(POS)
    say('seed %d: %d items, correct %.4f' % (seed, len(its), ok.mean()))
    nz = V1.noises(X[:, pi], its, seed); res = dict(seed=seed, cells={}, calib={})
    fam = np.array([i['fam'] for i in its])
    for dose in DOSES:
        for sh in (SHAPES if dose is not None else [None]):
            noise = None
            if dose is not None:
                noise = {l: nz[(l, sh)]['alpha'][dose] * nz[(l, sh)]['E'] for l in BAND}
                real = [V1.share_pop(demean(X[:, pi, li] + noise[l], fam), its) for li, l in enumerate(BAND)]
                res['calib']['%s_%s' % (dose, sh)] = dict(target=dose, population=real)
                say('dose %s %s: realised share by layer %s' % (dose, sh, ' '.join('%.4f' % r for r in real)))
            for sname, sub in SAMPLES.items():
                for pj in PROJS:
                    r = clamp_k(net, task, its, ok, X, plist, POS, noise, sub, pj,
                                say=lambda s, d=dose, h=sh, n=sname: say('dose %-5s %-5s %-6s' % (d, h, n) + s))
                    res['cells']['%s|%s|%s|%s' % (dose, sh, sname, pj)] = r
    return res


def known(ckdir):
    """Pre-freeze known answers (local CPU, seed 0): (1) the kudo form's unprojected cB and the whole-state patch equal
    v231's Ouro-form cB and full exactly (same bases, same pairs); (2) cBo is orthogonal to cD at every layer and half."""
    import v230_pilot as VP
    VP.DEV = 'cpu'; net, task, info = load(0, ckdir)
    its = items(K); plist = VP.positions(K); X, ok, y = states(net, task, its, plist); its = [dict(i, y=int(a)) for i, a in zip(its, y)]
    a = V1.clamp_g(net, task, its, ok, X, plist, POS, 'ouro', say=print); b = clamp_k(net, task, its, ok, X, plist, POS, proj='z', say=print)
    ok1 = abs(a['hit']['cB'] - b['hit']['cB']) < 1e-9 and abs(a['hit']['full'] - b['hit']['full']) < 1e-9
    print('KNOWN 1 (kudo cB = v231 ouro cB: %.3f vs %.3f; full %.3f vs %.3f): %s' % (
        b['hit']['cB'], a['hit']['cB'], b['hit']['full'], a['hit']['full'], 'PASS' if ok1 else 'FAIL'))
    for c in (b, clamp_k(net, task, its, ok, X, plist, POS, proj='zx', say=print)):
        orth = max(g['orth'] for g in c['geometry'])
        print('KNOWN 2 (%s: cBo orthogonal to cD: max |cos| %.1e; dims kept %s; share of s left %.2f): %s' % (
            c['proj'], orth, sorted({g['dims_o'] for g in c['geometry']}), c['left'], 'PASS' if orth < 1e-5 else 'FAIL'))


def verdict(root):
    """Per cell DETECTS (USED in >= 4/5), SEED-DEPENDENT (2-3), MISSES (<= 1); detection limit per (shape, sample);
    the Kudo reading in Kudo's regime (shape state, sample kudo)."""
    rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, '**', 'v233_power_s*.json'), recursive=True))]
    if sorted(r['seed'] for r in rs) != list(range(5)):
        sys.exit('expected the five seeds, got %s' % sorted(r['seed'] for r in rs))
    for r in rs:
        if r['model']['sha'] != PR.SHA[r['seed']]:
            sys.exit('seed %d checkpoint differs from v230' % r['seed'])
    miss = set()
    for r in rs:
        for ck, c in r['calib'].items():
            bad = [l for l, x in zip(BAND, c['population']) if abs(x - c['target']) > max(0.10 * c['target'], 0.0005)]
            if bad:
                miss.add((r['seed'], ck)); print('seed %d dose/shape %s missed its target at layers %s: its cells count as not USED' % (r['seed'], ck, bad))
    used = lambda r, key: r['cells'][key]['label'] == 'USED' and (r['seed'], '_'.join(key.split('|')[:2])) not in miss
    lab = {}
    for key in rs[0]['cells']:
        n = sum(used(r, key) for r in rs)
        lab[key] = 'DETECTS' if n >= 4 else 'SEED-DEPENDENT' if n >= 2 else 'MISSES'
        print('%-26s USED %d/5 -> %-14s stat %s | left %s' % (key, n, lab[key], ' '.join('%+.2f' % r['cells'][key]['diff'] for r in rs),
                                                                ' '.join('%.2f' % r['cells'][key]['left'] for r in rs)))
    lim = {}
    for sh in SHAPES:
        for sname in SAMPLES:
            for pj in PROJS:
                ok = lab['None|None|%s|%s' % (sname, pj)] == 'DETECTS'; best = 'native' if ok else None
                for dose in DOSES[1:]:
                    ok = ok and lab['%s|%s|%s|%s' % (dose, sh, sname, pj)] == 'DETECTS'
                    best = dose if ok else best
                lim['%s|%s|%s' % (sh, sname, pj)] = best
                print('detection limit %-5s %-6s %-2s: %s' % (sh, sname, pj, best))
    rd = lambda L: 'POWER-AT-KUDO-LEVEL' if L in (0.01, 0.005, 0.002) else 'AT-THE-LIMIT' if L == 0.03 else 'UNINFORMATIVE'
    reading, second = rd(lim['state|kudo|zx']), rd(lim['state|kudo|z'])
    print('Kudo reading (state-shaped, Kudo-matched sample; primary, z and x projected): limit %s -> %s' % (lim['state|kudo|zx'], reading))
    print("secondary (z projected, Kudo's procedure): limit %s -> %s" % (lim['state|kudo|z'], second))
    return dict(labels=lab, limits=lim, reading=reading, secondary=second, calibration_misses=sorted(map(list, miss)))


if __name__ == '__main__':
    a = sys.argv[1]
    if a == 'known':
        known(sys.argv[2])
    elif a == 'verdict':
        verdict(sys.argv[2])
    elif a == 'run':
        for seed in [int(s) for s in sys.argv[2].split(',')]:
            t0 = time.time(); net, task, info = load(seed, sys.argv[3]); r = grid(net, task, seed); r['model'] = info
            json.dump(r, open(os.path.join(OUT, 'v233_power_s%d.json' % seed), 'w'), indent=1)
            log('seed %d done in %d s' % (seed, time.time() - t0))
