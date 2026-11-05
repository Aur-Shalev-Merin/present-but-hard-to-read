"""v231: clamp power for a faint unwritten value (docs/v231_design.md; prereg docs/prereg_v231_clamp_power.md when frozen).

The five v230 pos models (retrained on a Kaggle T4 and checked against their sha256), at ';' with k 6, where DAS
puts s and the v230 clamp is USED in 5/5. The forward pass is untouched, so the model still uses s; only the clamp's
FITTING states are degraded, to the regime the Ouro clamp faced:
  dose    s's share of the residual (the share swap's N: held-out class means of s after the copies x, z, w and the
          answer are regressed out, family-demeaned, averaged over the two versions), lowered by adding variance
          drawn per item: native, 0.1, 0.03, 0.01, 0.005, 0.002 (Ouro's a+b ~0.007). Calibrated per layer.
  shape   iso: isotropic Gaussian; state: Gaussian with the covariance of the model's own residual at that layer and
          position (states minus the joint codes of s, x, z, w and the operation), i.e. more of the model's own
          non-s variation.
  sample  native: the whole fitting half (~260 items per class); ouro: 782 of them (34 per class; d / n_per_class =
          192 / 34 = 5.6, Ouro 2048 / 360 = 5.7).
  form    ouro: as the Ouro clamp (v194 / v209): separate family-centred class means of s, no copy projection,
          statistic hit(cB) - hit(cR), cR a random subspace of the same dimension: would s's own subspace move the
          answer? Ouro's own comparison, hit(cB) - max(hit(cX), hit(cR)) with cX the class means of the other sign
          (x - z on plus items, x + z on minus; Ouro's a - b), is kept as descriptive (pre-freeze known answer: cX
          moves the answer in 78% of pairs even at full strength, so it caps that comparison near +0.16).
          v230: the frozen v230 clamp (joint null-ranked codes, copy projected, cBo - cR).
Pairs, bases, bootstrap and the USED / NOT-USED / UNSCORABLE rule are v230's (v215's).

  python v231_power.py known <ckpt dir>   pre-freeze known answers on local checkpoints (CPU): native cells reproduce
                                          v230's clamp; doses hit their targets
  python v231_power.py run <seeds>        Kaggle GPU: retrain, sha check, the full grid -> v231_power_s<seed>.json
  python v231_power.py verdict <root>     aggregate (local)
"""
import glob, json, os, sys, time
import numpy as np

import v230_protocol as PR
from v230_protocol import P, BAND, NCLAMP, NBOOT, demean, items, states, labels, contrib

OUT = PR.OUT
DOSES = [None, 0.1, 0.03, 0.01, 0.005, 0.002]
SHAPES = ['iso', 'state']
SAMPLES = {'native': None, 'ouro': 782}
FORMS = ['ouro', 'v230']
POS, K = 3, 6
OURO_N = 0.007


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v231_log.txt'), 'a').write(s + '\n')


def share_s(Z, its):
    """The share swap's N for s at one layer: needed label's held-out share, copies x, z, w, y regressed, by version."""
    out = []
    for ver, lab in (('plus', 'S'), ('minus', 'M')):
        idx = np.array([k for k, i in enumerate(its) if i['ver'] == ver]); g = lambda f: np.array([its[k][f] for k in idx])
        res, err = contrib(Z[idx], g('fam'), [(g(c), P) for c in ('x', 'z', 'w', 'y')], {lab: labels(g('u'), g('v'), P)[lab]})
        out.append(1 - err[lab].sum() / res.sum())
    return float(np.mean(out))


def share_in(Z, its):
    """In-sample version of share_s (class means and copy fit on all items): biased up by about as much as the held-out
    one is biased down (~1 / items per class), so their mean estimates the population share."""
    out = []
    for ver, lab in (('plus', 'S'), ('minus', 'M')):
        idx = np.array([k for k, i in enumerate(its) if i['ver'] == ver]); g = lambda f: np.array([its[k][f] for k in idx])
        fam = g('fam'); Zv = demean(Z[idx], fam); C = np.zeros((len(idx), 4 * P), np.float32)
        for j, c in enumerate(('x', 'z', 'w', 'y')):
            C[np.arange(len(idx)), j * P + g(c)] = 1
        C = demean(C, fam); R = Zv - C @ np.linalg.lstsq(C, Zv, rcond=None)[0]; y = labels(g('u'), g('v'), P)[lab]
        M = np.stack([R[y == c].mean(0) for c in range(P)])[y]
        out.append(1 - ((R - M) ** 2).sum() / (R ** 2).sum())
    return float(np.mean(out))


def share_pop(Z, its):
    return (share_s(Z, its) + share_in(Z, its)) / 2


def joint_resid(Z, its, fam):
    """States minus the joint least-squares codes of s, x, z, w and the operation (family-demeaned)."""
    facs = ['s', 'x', 'z', 'w', 'op']; ncl = {'op': 2}
    H = demean(np.concatenate([np.eye(ncl.get(f, P), dtype=np.float32)[np.array([i[f] for i in its])] for f in facs], 1), fam)
    return Z - H @ np.linalg.lstsq(H, Z, rcond=None)[0]


def noises(X, its, seed):
    """Per layer and shape: unit-scale noise E (n, d), and per dose the scale alpha at which s's population share
    (share_pop) is the target. Population share ~ S / (res0 + a^2 rn); alpha from that, refined five times on share_pop."""
    fam = np.array([i['fam'] for i in its]); out = {}; cops = [(np.array([i[c] for i in its]), P) for c in ('x', 'z', 'w', 'y')]
    for li, l in enumerate(BAND):
        Z = demean(X[:, li], fam); N0 = share_pop(Z, its); R = joint_resid(Z, its, fam)
        lam, V = np.linalg.eigh(np.cov(R.T)); lam = np.clip(lam, 0, None)
        res0 = contrib(Z, fam, cops, {})[0].sum()
        for sh in SHAPES:
            G = np.random.default_rng([2310, seed, l, SHAPES.index(sh)]).standard_normal(Z.shape).astype(np.float32)
            E = demean(G if sh == 'iso' else (G * np.sqrt(lam)[None]) @ V.T.astype(np.float32), fam)
            rn = contrib(E, fam, cops, {})[0].sum(); alpha = {}
            for t in DOSES[1:]:
                S = N0 * res0; a2 = max(S / t - res0, 0) / rn
                for _ in range(5):
                    got = share_pop(Z + np.sqrt(a2) * E, its); S = got * (res0 + a2 * rn); a2 = max(S / t - res0, 0) / rn
                alpha[t] = float(np.sqrt(a2))
            out[(l, sh)] = dict(E=E, N0=N0, alpha=alpha)
    return out


def clamp_g(net, task, its, ok, X, plist, pos, form, noise=None, sub=None, say=log):
    """v230_protocol.clamp generalised. noise: {layer: (n, d) array} added to the FITTING states only. sub: number of
    fitting items per half (random subset, rng 2311) or None. form 'v230' with noise=None, sub=None reproduces
    v230_protocol.clamp exactly (same random streams)."""
    import torch, v230_pilot as VP
    pi = plist.index(pos); fam = np.array([i['fam'] for i in its])
    fams = sorted(set(fam.tolist())); half = {f: int(j % 2) for j, f in enumerate(np.random.default_rng(2151).permutation(fams))}
    hf = np.array([half[f] for f in fam]); rr = np.random.default_rng(2152); d = X.shape[-1]
    conds = ('cB', 'cBo', 'cD', 'cR') if form == 'v230' else ('cB', 'cX', 'cR')
    B = {c: {0: {}, 1: {}} for c in conds}; geo = []
    facs = ['s', 'x', 'z', 'w', 'op']; ncl = {'op': 2}; vals = {f: np.array([i[f] for i in its]) for f in facs}
    # the Ouro control: Ouro compared a+b with a-b, the other sign of the same operands; here x - z on plus items and
    # x + z on minus items
    vals['tw'] = np.array([(i['x'] - i['z']) % P if i['op'] == 0 else (i['x'] + i['z']) % P for i in its])
    off = np.cumsum([0] + [ncl.get(f, P) for f in facs]); rs = np.random.default_rng(2154)
    pick = {}
    for h in (0, 1):
        m = hf != h
        if sub is not None:
            keep = np.random.default_rng([2311, h]).choice(np.flatnonzero(m), sub, replace=False); m = np.zeros(len(its), bool); m[keep] = True
        pick[h] = m

    def fit(Zm, m, shuffle=None):
        vs = {g: (rs.permutation(vals[g][m]) if g == shuffle else vals[g][m]) for g in facs}
        H = demean(np.concatenate([np.eye(ncl.get(g, P), dtype=np.float32)[vs[g]] for g in facs], 1), fam[m])
        coef = np.linalg.lstsq(H, Zm, rcond=None)[0]
        return {g: coef[off[j]:off[j + 1]] - coef[off[j]:off[j + 1]].mean(0) for j, g in enumerate(facs)}

    def code(Zm, m, f):                                                         # v230 form: joint, null-ranked
        _, S, Vt = np.linalg.svd(fit(Zm, m)[f], full_matrices=False)
        null = max(float(np.linalg.svd(fit(Zm, m, f)[f], compute_uv=False)[0]) for _ in range(5))
        return Vt[:int((S > null).sum())].T

    def means(Zm, m, f):                                                        # Ouro form: separate class means
        v = vals[f][m]; M = np.stack([Zm[v == c].mean(0) for c in np.unique(v)]); M -= M.mean(0)
        return np.linalg.svd(M, full_matrices=False)[2][:len(M) - 1].T
    for li, l in enumerate(BAND):
        Zall = X[:, pi, li] if noise is None else X[:, pi, li] + noise[l]
        for h in (0, 1):
            m = pick[h]; Z = demean(Zall[m], fam[m])
            if form == 'v230':
                U, Dd = code(Z, m, 's'), code(Z, m, 'z'); Uo = U
                if U.shape[1]:
                    Uo, sv, _ = np.linalg.svd(U - Dd @ (Dd.T @ U), full_matrices=False); Uo = Uo[:, sv > 1e-3]
                R = np.linalg.qr(rr.standard_normal((d, P - 1)))[0][:, :Uo.shape[1]]
                mats = (('cB', U), ('cBo', Uo), ('cD', Dd), ('cR', R))
            else:
                U, Ux = means(Z, m, 's'), means(Z, m, 'tw'); R = np.linalg.qr(rr.standard_normal((d, P - 1)))[0][:, :U.shape[1]]
                mats = (('cB', U), ('cX', Ux), ('cR', R))
            for c, A in mats:
                B[c][h][l] = torch.tensor(A, dtype=torch.float32, device=VP.DEV)
            geo.append(dict(layer=l, half=h, dims=U.shape[1], n_fit=int(m.sum())))
    rng = np.random.default_rng(2151); cand = np.flatnonzero(ok); base = rng.permutation(cand)[:NCLAMP]
    if len(base) < 2:
        return dict(pos=pos, form=form, n=len(base), label='UNSCORABLE', diff=float('nan'), ci=[float('nan')] * 2, hit={}, geometry=geo)
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
    if form == 'v230':
        stat = lambda q: (sums['cBo'][q].sum() - sums['cR'][q].sum()) / Nn[q].sum()
    else:                                                                       # Ouro form: s's own subspace against the random one
        stat = lambda q: (sums['cB'][q].sum() - sums['cR'][q].sum()) / Nn[q].sum()
    dlt = stat(np.arange(len(u))); rg = np.random.default_rng(2153)
    bs = [stat(rg.integers(0, len(u), len(u))) for _ in range(NBOOT)]
    lo, hi = np.percentile(bs, (1.25, 98.75))
    lab = ('UNSCORABLE' if hit['full'].mean() < 0.30 else 'USED' if dlt >= 0.10 and lo > 0 else 'NOT-USED' if hi < 0.02 else 'UNCLEAR')
    say('  clamp %-4s pos %d (%d pairs): hit %s | stat %+.3f [%+.3f, %+.3f] -> %s' % (
        form, pos, len(base), ' '.join('%s %.3f' % (c, hit[c].mean()) for c in hit), dlt, lo, hi, lab))
    extra = {}
    if form == 'ouro':                                                          # descriptive: Ouro's own comparison, cB - max(cX, cR)
        s2 = lambda q: (sums['cB'][q].sum() - max(sums['cX'][q].sum(), sums['cR'][q].sum())) / Nn[q].sum(); rg2 = np.random.default_rng(2153)
        b2 = [s2(rg2.integers(0, len(u), len(u))) for _ in range(NBOOT)]
        extra = dict(diff_vs_control=float(s2(np.arange(len(u)))), ci_vs_control=[float(x) for x in np.percentile(b2, (1.25, 98.75))])
    return dict(pos=pos, form=form, n=len(base), hit={c: float(v.mean()) for c, v in hit.items()}, kept=kept, diff=float(dlt),
                ci=[float(lo), float(hi)], label=lab, geometry=geo, **extra)


def grid(net, task, seed, say=log):
    import v230_pilot as VP
    its = items(K); plist = VP.positions(K); X, ok, y = states(net, task, its, plist)
    its = [dict(i, y=int(a)) for i, a in zip(its, y)]; pi = plist.index(POS)
    say('seed %d: %d items, correct %.4f' % (seed, len(its), ok.mean()))
    nz = noises(X[:, pi], its, seed); res = dict(seed=seed, cells={}, calib={})
    fam = np.array([i['fam'] for i in its])
    for dose in DOSES:
        for sh in (SHAPES if dose is not None else [None]):
            noise = None
            if dose is not None:
                noise = {l: nz[(l, sh)]['alpha'][dose] * nz[(l, sh)]['E'] for l in BAND}
                Zs = [demean(X[:, pi, li] + noise[l], fam) for li, l in enumerate(BAND)]
                real = [share_pop(Z, its) for Z in Zs]; held = [share_s(Z, its) for Z in Zs]
                res['calib']['%s_%s' % (dose, sh)] = dict(target=dose, population=real, held_out=held, N0=[nz[(l, sh)]['N0'] for l in BAND])
                say('dose %s %s: realised share by layer %s (native %s)' % (dose, sh, ' '.join('%.4f' % r for r in real),
                    ' '.join('%.3f' % nz[(l, sh)]['N0'] for l in BAND)))
            for sname, sub in SAMPLES.items():
                for form in FORMS:
                    r = clamp_g(net, task, its, ok, X, plist, POS, form, noise, sub,
                                say=lambda s, d=dose, h=sh, n=sname: say('dose %-5s %-5s %-6s' % (d, h, n) + s))
                    res['cells']['%s|%s|%s|%s' % (dose, sh, sname, form)] = r
    return res


def known(ckdir):
    """Pre-freeze known answers (local CPU, seed 0 checkpoint): (1) form v230 with no noise and the native sample equals
    v230_protocol.clamp; (2) form ouro's cB and cR equal the post hoc class-mean clamp; (3) realised doses within 10% of the target or 0.0005, whichever is larger."""
    import torch, v230_pilot as VP
    VP.DEV = 'cpu'; seed = 0; task = VP.Task('pos'); net = VP.Net(VP.V, VP.L, 'std')
    net.load_state_dict(torch.load(os.path.join(ckdir, 'v230_std_pos_s%d.pt' % seed), map_location='cpu', weights_only=True)); net.eval()
    its = items(K); plist = VP.positions(K); X, ok, y = states(net, task, its, plist); its = [dict(i, y=int(a)) for i, a in zip(its, y)]
    a = PR.clamp(net, task, its, ok, X, plist, POS, say=print); b = clamp_g(net, task, its, ok, X, plist, POS, 'v230', say=print)
    ok1 = a['hit'] == b['hit'] and abs(a['diff'] - b['diff']) < 1e-6 and max(abs(p - q) for p, q in zip(a['ci'], b['ci'])) < 1e-6
    print('KNOWN 1 (v230 form = frozen clamp): %s | hits equal %s, diff %r vs %r, ci %r vs %r' % (
        'PASS' if ok1 else 'FAIL', a['hit'] == b['hit'], a['diff'], b['diff'], a['ci'], b['ci']))
    c = clamp_g(net, task, its, ok, X, plist, POS, 'ouro', say=print)
    ph = json.load(open(os.path.join(os.path.dirname(os.path.dirname(ckdir)), 'posthoc_classmean_clamp.json')))['0_6_3']
    ok2 = abs(c['hit']['cB'] - ph['hit']['cB']) < 1e-9 and abs(c['hit']['full'] - ph['hit']['full']) < 1e-9
    print('KNOWN 2 (ouro form cB = post hoc class means: %.3f vs %.3f; cR %.3f vs %.3f): %s' % (
        c['hit']['cB'], ph['hit']['cB'], c['hit']['cR'], ph['hit']['cR'], 'PASS' if ok2 else 'FAIL'))
    pi = plist.index(POS); nz = noises(X[:, pi], its, seed); fam = np.array([i['fam'] for i in its]); worst = 0
    for sh in SHAPES:
        for t in DOSES[1:]:
            real = [share_pop(demean(X[:, pi, li] + nz[(l, sh)]['alpha'][t] * nz[(l, sh)]['E'], fam), its) for li, l in enumerate(BAND)]
            worst = max(worst, max(abs(r - t) / max(0.10 * t, 0.0005) for r in real))
            print('  %-5s target %.3f realised %s' % (sh, t, ' '.join('%.4f' % r for r in real)))
    print('KNOWN 3 (doses within 10%% or 0.0005; worst %.2f of the tolerance): %s' % (worst, 'PASS' if worst <= 1 else 'FAIL'))


def verdict(root):
    """Per cell DETECTS (USED in >= 4/5), SEED-DEPENDENT (2-3), MISSES (<= 1); detection limit per (shape, sample,
    form); the Ouro reading in Ouro's regime (form ouro, shape state, sample ouro)."""
    rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, '**', 'v231_power_s*.json'), recursive=True))]
    if sorted(r['seed'] for r in rs) != list(range(5)):
        sys.exit('expected the five seeds, got %s' % sorted(r['seed'] for r in rs))
    for r in rs:
        if r['model']['sha'] != PR.SHA[r['seed']]:
            sys.exit('seed %d checkpoint differs from v230' % r['seed'])
    miss = set()                                                                # calibration on every seed and layer
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
        print('%-34s USED %d/5 -> %-14s stat %s' % (key, n, lab[key], ' '.join('%+.2f' % r['cells'][key]['diff'] for r in rs)))
    lim = {}
    for sh in SHAPES:
        for sname in SAMPLES:
            for form in FORMS:
                ok = lab['None|None|%s|%s' % (sname, form)] == 'DETECTS'; best = 'native' if ok else None
                for dose in DOSES[1:]:
                    ok = ok and lab['%s|%s|%s|%s' % (dose, sh, sname, form)] == 'DETECTS'
                    best = dose if ok else best
                lim['%s|%s|%s' % (sh, sname, form)] = best
                print('detection limit %-5s %-6s %-4s: %s' % (sh, sname, form, best))
    g = lambda d: lab['%s|state|ouro|ouro' % d]
    reading = ('POWER-AT-OURO-LEVEL' if g(0.005) == 'DETECTS' and lim['state|ouro|ouro'] not in (None, 'native', 0.1, 0.03, 0.01)
               else 'UNINFORMATIVE' if g(0.01) != 'DETECTS' or lim['state|ouro|ouro'] in (None, 'native', 0.1, 0.03)
               else 'AT-THE-LIMIT')
    print('Ouro reading (form ouro, state-shaped, Ouro-matched sample): %s' % reading)
    return dict(labels=lab, limits=lim, reading=reading, calibration_misses=sorted(map(list, miss)))


if __name__ == '__main__':
    a = sys.argv[1]
    if a == 'known':
        known(sys.argv[2])
    elif a == 'verdict':
        verdict(sys.argv[2])
    elif a == 'run':
        for seed in [int(s) for s in sys.argv[2].split(',')]:
            t0 = time.time(); net, task, info = PR.model('pos', seed); r = grid(net, task, seed); r['model'] = info
            json.dump(r, open(os.path.join(OUT, 'v231_power_s%d.json' % seed), 'w'), indent=1)
            log('seed %d done in %d s' % (seed, time.time() - t0))
