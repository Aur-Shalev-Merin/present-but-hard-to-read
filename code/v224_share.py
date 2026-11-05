"""v224: the share-based operation swap (docs/prereg_v224_share_swap.md).

Why: v221's positive control showed that the accuracy swap saturates (v223: random functions of the operand pair read
0.99-1.00 where b is about to be written), while the share of residual variance a label explains still follows the
operation there. This module defines that statistic, its labels and its null worlds.

Per version (the step's operation: plus / minus), position and layer l in {16, 20, 24, 28}:
  1. family-demeaned states Z; copies C = one-hot(u) + one-hot(v), family-demeaned (u, v: the step's operands)
  2. folds by family (5, v178's seed); on the training folds, Z ~ C by least squares -> residuals R
  3. per label L, class means of R on the training folds; per held-out item, err_L = |r - mu_L(y)|^2, res = |r|^2
  4. share_L = 1 - sum err_L / sum res (pooled), averaged over the four layers
Labels: S = (u + v) mod 5, M = (u - v) mod 5, R0..R3 = S (R0, R1) or M (R2, R3) of a randomly re-assigned pair
  (seed 2231): random functions of (u, v) with exactly the class sizes of S and M.
  N = (share_S on plus + share_M on minus) / 2;   Rbar = mean share of R0..R3 over both versions;   lift = N - Rbar
  D+ = share_S - share_M on plus;  D- = share_M - share_S on minus;  I = (D+ + D-) / 2;  I_rel = I / N
Cluster bootstrap over families (both versions together), 10,000 resamples (1,000 in null worlds), 97.5% CIs.
  UNSCORABLE    accuracy gate fails (G1 < 0.80 in a version)
  NO-SIGNAL     N CI lower <= 0 (the needed label explains nothing beyond the copies)
  FOLLOWS       D+ > 0, D- > 0, I CI lower > 0 and I >= TAU * N
  DOES-NOT-FOLLOW  I CI upper < 0.10 * N
  UNCLEAR       otherwise
TAU is set before the freeze from the null worlds below: max(0.20, 1.5 x the largest I_rel of any null run with
signal, rounded up to 0.05). lift (N - Rbar) is descriptive only: a pure pair code already gives S and M a small lift
over the shuffled functions (+0.04 in the selftest), so it is not a test of specificity.

Synthetic worlds (family offsets, copies U1[u] + U2[v], one mixture scaled to lam x the copies' RMS, N(0,1) noise per
layer, d = 256; the real family design):
  null:  pair (tanh products of codes for (u,v), (u,op), (v,op)), fourier (same-frequency products of periodic digit
         features, map per op), mlp (random ReLU net of one-hot u, v, op), paircode (a random vector per (u, v)),
         paircode_op (a random vector per (u, v, op)) -- none holds the step's value
  positive: computed (a code of the value u op v at strength c x the copies' RMS, plus pair mixing at lam 1)

  python v224_share.py --selftest          positive worlds FOLLOWS; copies-only, paircode and paircode_op not
  python v224_share.py --nullbar           the null grid -> TAU (pre-freeze)
  python v224_share.py dev <root>          development: v221's states (inb, eqb, letb, eq, last), already seen
  python v224_share.py main <root>         the frozen analysis of the fresh controls
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

import v178_analyze as V

OUT = os.environ.get('OUT_DIR', '.')
LAYERS = list(range(0, 29, 4)); SL = [16, 20, 24, 28]; LABS = ['S', 'M', 'R0', 'R1', 'R2', 'R3']
TAU = 0.30   # set 2026-10-27 by --nullbar (largest I/N among null runs with signal 0.187)
NULL_KINDS = ('pair', 'fourier', 'mlp', 'paircode', 'paircode_op'); LAMS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0); DD = 256


def demean(X, fam):
    X = np.array(X, dtype=np.float32)
    for f in np.unique(fam):
        X[fam == f] -= X[fam == f].mean(0)
    return X


def labels(u, v):
    pairs = sorted(set(zip(u.tolist(), v.tolist()))); rng = np.random.default_rng(2231)
    out = {'S': (u + v) % 5, 'M': (u - v) % 5}
    for r in range(4):                                    # S (R0, R1) or M (R2, R3) of a shuffled pair: same class sizes
        perm = rng.permutation(len(pairs)); g = (lambda a, b: (a + b) % 5) if r < 2 else (lambda a, b: (a - b) % 5)
        mp = {pairs[j]: g(*pairs[perm[j]]) for j in range(len(pairs))}
        out['R%d' % r] = np.array([mp[p] for p in zip(u.tolist(), v.tolist())])
    return out


def contrib(Z, fam, u, v, ys):
    """Held-out per-item res and err_L (folds by family)."""
    Z = demean(Z, fam); C = np.zeros((len(Z), 20), np.float32); C[np.arange(len(Z)), u] = 1; C[np.arange(len(Z)), 10 + v] = 1
    C = demean(C, fam); fold = V.folds(fam); res = np.zeros(len(Z)); err = {k: np.zeros(len(Z)) for k in ys}
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        W = np.linalg.lstsq(C[tr], Z[tr], rcond=None)[0]; Rtr = Z[tr] - C[tr] @ W; Rte = Z[te] - C[te] @ W
        res[te] = (Rte ** 2).sum(1)
        for k, y in ys.items():
            mu = {c: Rtr[y[tr] == c].mean(0) for c in np.unique(y[tr])}
            M = np.stack([mu.get(c, np.zeros(Z.shape[1], np.float32)) for c in y[te]])
            err[k][te] = ((Rte - M) ** 2).sum(1)
    return res, err


def analyse(get, its, ok, nboot=10000, seed=2241, tau=None):
    """get(idx, layer) -> states (len(idx), d) at one position; its: items with fam, ver, u, v; ok: per-item correct."""
    tau = TAU if tau is None else tau
    fams = sorted(set(i['fam'] for i in its)); fx = {f: j for j, f in enumerate(fams)}
    vers = ('plus', 'minus'); F = len(fams); keys = LABS
    Sres = np.zeros((2, len(SL), F)); Serr = np.zeros((2, len(SL), len(keys), F))
    g1 = {}
    for vi, ver in enumerate(vers):
        idx = np.array([k for k, i in enumerate(its) if i['ver'] == ver]); g1[ver] = float(np.mean([ok[k] for k in idx]))
        fam = np.array([its[k]['fam'] for k in idx]); u = np.array([its[k]['u'] for k in idx]); v = np.array([its[k]['v'] for k in idx])
        ys = labels(u, v); fi = np.array([fx[f] for f in fam])
        for li, l in enumerate(SL):
            res, err = contrib(np.asarray(get(idx, l), np.float32), fam, u, v, ys)
            np.add.at(Sres[vi, li], fi, res)
            for ki, k in enumerate(keys):
                np.add.at(Serr[vi, li, ki], fi, err[k])

    def stat(w):                                          # w: (B, F) family weights
        sh = 1 - np.einsum('bf,vlkf->bvlk', w, Serr) / np.einsum('bf,vlf->bvl', w, Sres)[..., None]
        sh = sh.mean(2)                                   # over layers -> (B, 2, labels)
        S_p, M_p, S_m, M_m = sh[:, 0, 0], sh[:, 0, 1], sh[:, 1, 0], sh[:, 1, 1]
        N = (S_p + M_m) / 2; Rb = sh[:, :, 2:].mean((1, 2)); Dp, Dm = S_p - M_p, M_m - S_m
        return dict(N=N, Rbar=Rb, lift=N - Rb, Dp=Dp, Dm=Dm, I=(Dp + Dm) / 2, sh=sh)
    pt = stat(np.ones((1, F))); rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, F, F), minlength=F) for _ in range(nboot)]).astype(float)
    bs = stat(W); ci = {k: np.percentile(bs[k], (1.25, 98.75)) for k in ('N', 'lift', 'I', 'Dp', 'Dm')}
    r = {k: float(pt[k][0]) for k in ('N', 'Rbar', 'lift', 'Dp', 'Dm', 'I')}
    r.update(ci={k: [float(x) for x in c] for k, c in ci.items()}, I_rel=r['I'] / r['N'] if r['N'] > 0 else float('nan'), g1=g1,
             shares={'%s_%s' % (ver, k): float(pt['sh'][0, vi, ki]) for vi, ver in enumerate(vers) for ki, k in enumerate(keys)})
    if min(g1.values()) < 0.80:
        lab = 'UNSCORABLE'
    elif ci['N'][0] <= 0:
        lab = 'NO-SIGNAL'
    elif tau is not None and r['Dp'] > 0 and r['Dm'] > 0 and ci['I'][0] > 0 and r['I'] >= tau * r['N']:
        lab = 'FOLLOWS'
    elif ci['I'][1] < 0.10 * r['N']:
        lab = 'DOES-NOT-FOLLOW'
    else:
        lab = 'UNCLEAR'
    r['label'] = lab
    return r


def fmt(name, r):
    return ('%-6s N %.3f  Rbar %.3f  lift %+.3f [%+.3f, %+.3f]  D+ %+.3f  D- %+.3f  I %+.3f [%+.3f, %+.3f]  I/N %+.2f  -> %s' % (
        name, r['N'], r['Rbar'], r['lift'], *r['ci']['lift'], r['Dp'], r['Dm'], r['I'], *r['ci']['I'], r['I_rel'], r['label']))


# ---------------------------------------------------------------- synthetic worlds

def design(which):
    import v224_items as I
    if which == 'b':
        its = [dict(i, u=i['d1'], v=i['d2']) for i in I.build_b()]
    else:
        its = [dict(i, u=i['k'], v=i['B']) for i in I.build_a()]
    keep = set(sorted(set(i['fam'] for i in its))[::2])
    return [i for i in its if i['fam'] in keep]


def world(its, kind, lam, seed, comp=0.0):
    rng = np.random.default_rng(seed); n = len(its)
    u = np.array([i['u'] for i in its]); v = np.array([i['v'] for i in its]); o = np.array([i['ver'] == 'minus' for i in its], int)
    U1, U2 = rng.standard_normal((10, DD)), rng.standard_normal((10, DD)); cop = U1[u] + U2[v]; rms = np.sqrt((cop ** 2).sum(1).mean())
    sc = lambda x: x * rms / np.sqrt((x ** 2).sum(1).mean())
    if kind == 'pair':
        mix = sum(np.tanh(2 * rng.standard_normal((10, DD))[x] * rng.standard_normal((10, DD))[y]) for x, y in ((u, v), (u, o), (v, o)))
    elif kind == 'fourier':
        mix = np.zeros((n, DD))
        for f in (1, 2, 5):
            c1, s1, c2, s2 = (g(2 * np.pi * f * x / 10) for x in (u, v) for g in (np.cos, np.sin))
            Fm = np.stack([c1 * c2, s1 * s2, c1 * s2, s1 * c2], 1); R = rng.standard_normal((2, 4, DD))
            mix += np.einsum('nk,nkd->nd', Fm, R[o])
    elif kind == 'mlp':
        H = np.concatenate([np.eye(10)[u], np.eye(10)[v], np.eye(2)[o]], 1)
        mix = np.maximum(H @ rng.standard_normal((22, 256)) + rng.standard_normal(256), 0) @ rng.standard_normal((256, DD))
    elif kind == 'paircode':
        P = rng.standard_normal((10, 10, DD)); mix = P[u, v]
    elif kind == 'paircode_op':
        P = rng.standard_normal((2, 10, 10, DD)); mix = P[o, u, v]
    elif kind == 'copies':
        mix = np.zeros((n, DD))
    else:
        raise ValueError(kind)
    sig = cop + (lam * sc(mix) if mix.any() else 0)
    if comp:
        val = np.where(o == 0, u + v, u - v) + 9; sig = sig + comp * sc(rng.standard_normal((19, DD))[val])
    offs = {f: rng.standard_normal(DD) * 3 for f in set(i['fam'] for i in its)}
    sig = sig + np.stack([offs[i['fam']] for i in its])
    return {l: (sig + rng.standard_normal((n, DD))).astype(np.float32) for l in SL}


def run_world(its, kind, lam, seed, comp=0.0, nboot=1000, tau=None):
    Xw = world(its, kind, lam, seed, comp)
    return analyse(lambda idx, l: Xw[l][idx], its, [True] * len(its), nboot=nboot, tau=tau)


def nullbar(say=print):
    worst = 0.0; rows = []
    for which in ('b', 'a'):
        its = design(which)
        for kind in NULL_KINDS:
            for lam in LAMS:
                for seed in range(1, 7):
                    r = run_world(its, kind, lam, seed); spec = r['ci']['N'][0] > 0
                    rows.append(dict(set=which, kind=kind, lam=lam, seed=seed, spec=spec, I=r['I'], N=r['N'], I_rel=r['I_rel']))
                    if spec and np.isfinite(r['I_rel']):
                        worst = max(worst, r['I_rel'])
                ks = [x for x in rows if x['set'] == which and x['kind'] == kind]
                say('null %s %-11s: signal in %2d/%d runs; I/N max %+.3f (runs with signal %s)' % (
                    which, kind, sum(x['spec'] for x in ks), len(ks), max(x['I_rel'] for x in ks),
                    '%+.3f' % max([x['I_rel'] for x in ks if x['spec']]) if any(x['spec'] for x in ks) else '-'))
    tau = max(0.20, np.ceil(1.5 * worst / 0.05) * 0.05)
    say('largest I/N among null runs with signal %.3f -> TAU = %.2f' % (worst, tau))
    return tau, rows


def selftest(tau=0.20):
    ok = True
    for which in ('b', 'a'):
        its = design(which)
        for name, kind, lam, comp, want in (('computed c=0.5', 'pair', 1.0, 0.5, 'FOLLOWS'), ('computed c=1', 'pair', 1.0, 1.0, 'FOLLOWS'),
                                            ('copies only', 'copies', 0.0, 0.0, 'not FOLLOWS'), ('paircode lam 2', 'paircode', 2.0, 0.0, 'not FOLLOWS'),
                                            ('paircode_op lam 2', 'paircode_op', 2.0, 0.0, 'not FOLLOWS')):
            r = run_world(its, kind, lam, 1, comp, tau=tau)
            good = (r['label'] == want) if want != 'not FOLLOWS' else r['label'] != 'FOLLOWS'; ok &= good
            print('selftest %s %-18s %s %s' % (which, name, fmt('', r), 'OK' if good else 'FAIL'), flush=True)
    return ok


# ---------------------------------------------------------------- real states

def run_states(root, tag, its, positions, okkey, say=print, tau=None):
    f = glob.glob(os.path.join(root, '**', tag + '_states.npy'), recursive=True)[0]
    X = np.load(f, mmap_mode='r'); m = json.load(open(f.replace('_states.npy', '_meta.json')))
    assert len(m['items']) == len(its)
    POS = m['positions']; ok = [x[okkey] for x in m['items']]
    out = {}
    for p in positions:
        r = analyse(lambda idx, l: X[idx, POS.index(p), LAYERS.index(l)], its, ok, tau=tau); out[p] = r
        say(fmt(p, r)); say('       shares: ' + ' '.join('%s %.3f' % (k, s) for k, s in r['shares'].items()) + ' | G1 ' + str(r['g1']))
    return out


def dev(root):
    """Development (already seen data): v221's states, b step, with v221's correct flag."""
    import v221_items as V1, v215_items as V5
    its = [dict(i, u=i['d1'], v=i['d2']) for i in V1.build()]
    f = glob.glob(os.path.join(root, '**', 'v221_fam_states.npy'), recursive=True)[0]
    X = np.load(f, mmap_mode='r'); m = json.load(open(f.replace('_states.npy', '_meta.json'))); ok = [x['correct'] for x in m['items']]
    out = {}
    for p in ['inb', 'eqb', 'letb', 'eq', 'last']:
        r = analyse(lambda idx, l: X[idx, m['positions'].index(p), LAYERS.index(l)], its, ok); out[p] = r
        print(fmt(p, r), flush=True); print('       shares: ' + ' '.join('%s %.3f' % (k, s) for k, s in r['shares'].items()), flush=True)
    json.dump(out, open(os.path.join(OUT, 'v224_dev.json'), 'w'), indent=1)


def main(root):
    import v224_items as I
    ib = [dict(i, u=i['d1'], v=i['d2']) for i in I.build_b()]; ia = [dict(i, u=i['k'], v=i['B']) for i in I.build_a()]
    res = {}
    print('=== fresh control 1: Qwen2.5-1.5B, b step (primary inb) ===', flush=True)
    res['b_1.5b'] = run_states(root, 'v224_b_1.5b', ib, ['inb', 'eqb', 'letb', 'eq', 'last'], 'okB', say=lambda s: print(s, flush=True))
    print('=== fresh control 2: Qwen2.5-7B, a step (primary inA) ===', flush=True)
    res['a_7b'] = run_states(root, 'v224_a_7b', ia, ['inA', 'eqa', 'digB', 'eq', 'last'], 'okA', say=lambda s: print(s, flush=True))
    print('=== null worlds, fresh seeds 7-12 at every lam (no run may be FOLLOWS) ===', flush=True)
    nf = 0
    for which in ('b', 'a'):
        its = design(which)
        for kind in NULL_KINDS:
            labs = [run_world(its, kind, lam, s)['label'] for lam in LAMS for s in range(7, 13)]
            nf += labs.count('FOLLOWS'); print('  %s %-11s FOLLOWS in %d/%d' % (which, kind, labs.count('FOLLOWS'), len(labs)), flush=True)
    res['null_follows'] = nf
    p1, p2 = res['b_1.5b']['inb']['label'], res['a_7b']['inA']['label']
    verdict = 'VALID' if p1 == 'FOLLOWS' and p2 == 'FOLLOWS' and nf == 0 else 'PARTIAL' if 'FOLLOWS' in (p1, p2) and nf == 0 else 'FAILS'
    res['verdict'] = verdict
    print('primary: control 1 inb %s, control 2 inA %s, null FOLLOWS %d -> method %s' % (p1, p2, nf, verdict), flush=True)
    json.dump(res, open(os.path.join(OUT, 'v224_result.json'), 'w'), indent=1)


if __name__ == '__main__':
    a = sys.argv[1]
    if a == '--selftest':
        sys.exit(0 if selftest() else 1)
    elif a == '--nullbar':
        t, rows = nullbar(); json.dump(dict(tau=t, rows=rows), open(os.path.join(OUT, 'v224_nullbar.json'), 'w'), indent=1)
    elif a == 'dev':
        dev(sys.argv[2])
    else:
        main(sys.argv[2])
