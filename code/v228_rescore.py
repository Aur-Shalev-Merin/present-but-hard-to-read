"""v228: re-score of the note's targets with the share swap validated in v224 + v227 (docs/prereg_v228_rescore.md).

The statistic is v224_share.analyse with two generalisations, fixed before any target state is touched:
  copies   one-hot of EVERY digit that varies in the prompt (Kudo: d1, d2; Ouro Study 3: a, b, c units digits);
           in v224 / v227 these were the step's two operands, which were the only varied digits
  answer   where the answer is about to be written or fixes b within a family (Kudo answer-only; Ouro at P1), its
           digit's one-hot is regressed out with the copies (primary); the copies-only version is descriptive
and the bootstrap cluster is whatever links a plus family to its minus twin (Kudo: fam; Ouro: pair). Labels S, M,
R0..R3 from the step's operands (Kudo: d1, d2; Ouro: a, b), TAU 0.30, labels FOLLOWS / DOES-NOT-FOLLOW / UNCLEAR /
NO-SIGNAL exactly as v224 section 2. Accuracy gates: each original study's own (all passed there), not re-applied.

Targets (states saved by the original extractions; aurshalevmerin):
  T1 Kudo, their chain format (v215 fam states): eq (primary), last; Qwen2.5-7B layers 16, 20, 24, 28
  T2 Kudo, answer only (v216 fam states): eq, pre, last (all primary); answer regressed; layers 16, 20, 24, 28
  T3 Ouro-1.4B Study 3 (v194 states, main set): P1 (primary), P0; answer regressed; the last two loops' states
     mid3, end3, mid4, end4 (the upper half of the effective depth, as v224's layers are of Qwen's)

  python v228_rescore.py --selftest     the generalised code reproduces v224_share.analyse; answer-only worlds
                                        are not FOLLOWS once the answer is regressed
  python v228_rescore.py <root>         the re-score
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

import v178_analyze as V
import v224_share as SH

OUT = os.environ.get('OUT_DIR', '.'); LAB = SH.LABS


def contrib(Z, fam, cops, ys):
    """As v224_share.contrib with any number of one-hot copy variables (each 0..9)."""
    Z = SH.demean(Z, fam); C = np.zeros((len(Z), 10 * len(cops)), np.float32)
    for j, c in enumerate(cops):
        C[np.arange(len(Z)), 10 * j + c] = 1
    C = SH.demean(C, fam); fold = V.folds(fam); res = np.zeros(len(Z)); err = {k: np.zeros(len(Z)) for k in ys}
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        W = np.linalg.lstsq(C[tr], Z[tr], rcond=None)[0]; Rtr = Z[tr] - C[tr] @ W; Rte = Z[te] - C[te] @ W
        res[te] = (Rte ** 2).sum(1)
        for k, y in ys.items():
            mu = {c: Rtr[y[tr] == c].mean(0) for c in np.unique(y[tr])}
            M = np.stack([mu.get(c, np.zeros(Z.shape[1], np.float32)) for c in y[te]])
            err[k][te] = ((Rte - M) ** 2).sum(1)
    return res, err


def analyse(get, its, copy_keys, cluster='fam', layers=None, nboot=10000, seed=2241, tau=None):
    """get(idx, layer) -> states; its: items with fam, ver, u, v, the copy keys and the cluster key."""
    tau = SH.TAU if tau is None else tau; layers = layers or SH.SL
    cl = sorted(set(i[cluster] for i in its)); cx = {f: j for j, f in enumerate(cl)}; F = len(cl)
    Sres = np.zeros((2, len(layers), F)); Serr = np.zeros((2, len(layers), len(LAB), F))
    for vi, ver in enumerate(('plus', 'minus')):
        idx = np.array([k for k, i in enumerate(its) if i['ver'] == ver])
        fam = np.array([its[k]['fam'] for k in idx]); u = np.array([its[k]['u'] for k in idx]); v = np.array([its[k]['v'] for k in idx])
        cops = [np.array([its[k][c] for k in idx]) for c in copy_keys]; ys = SH.labels(u, v); ci = np.array([cx[its[k][cluster]] for k in idx])
        for li, l in enumerate(layers):
            res, err = contrib(np.asarray(get(idx, l), np.float32), fam, cops, ys)
            np.add.at(Sres[vi, li], ci, res)
            for ki, k in enumerate(LAB):
                np.add.at(Serr[vi, li, ki], ci, err[k])

    def stat(w):
        sh = (1 - np.einsum('bf,vlkf->bvlk', w, Serr) / np.einsum('bf,vlf->bvl', w, Sres)[..., None]).mean(2)
        S_p, M_p, S_m, M_m = sh[:, 0, 0], sh[:, 0, 1], sh[:, 1, 0], sh[:, 1, 1]
        N = (S_p + M_m) / 2; Rb = sh[:, :, 2:].mean((1, 2)); Dp, Dm = S_p - M_p, M_m - S_m
        return dict(N=N, Rbar=Rb, lift=N - Rb, Dp=Dp, Dm=Dm, I=(Dp + Dm) / 2, sh=sh)
    pt = stat(np.ones((1, F))); rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, F, F), minlength=F) for _ in range(nboot)]).astype(float)
    bs = stat(W); ci = {k: np.percentile(bs[k], (1.25, 98.75)) for k in ('N', 'lift', 'I', 'Dp', 'Dm')}
    r = {k: float(pt[k][0]) for k in ('N', 'Rbar', 'lift', 'Dp', 'Dm', 'I')}
    r.update(ci={k: [float(x) for x in c] for k, c in ci.items()}, I_rel=r['I'] / r['N'] if r['N'] > 0 else float('nan'), g1={'plus': 1.0, 'minus': 1.0},
             shares={'%s_%s' % (ver, k): float(pt['sh'][0, vi, ki]) for vi, ver in enumerate(('plus', 'minus')) for ki, k in enumerate(LAB)})
    if ci['N'][0] <= 0:
        lab = 'NO-SIGNAL'
    elif r['Dp'] > 0 and r['Dm'] > 0 and ci['I'][0] > 0 and r['I'] >= tau * r['N']:
        lab = 'FOLLOWS'
    elif ci['I'][1] < 0.10 * r['N']:
        lab = 'DOES-NOT-FOLLOW'
    else:
        lab = 'UNCLEAR'
    r['label'] = lab
    return r


def say_r(name, r):
    print(SH.fmt(name, r), flush=True)
    print('       shares: ' + ' '.join('%s %.3f' % (k, s) for k, s in r['shares'].items()), flush=True)


def selftest():
    ok = True
    its = [dict(i, d1=i['u'], d2=i['v']) for i in SH.design('b')]
    Xw = SH.world(its, 'pair', 1.0, 1, comp=0.5); get = lambda idx, l: Xw[l][idx]
    a = SH.analyse(get, its, [True] * len(its), nboot=1000); b = analyse(get, its, ['u', 'v'], nboot=1000)
    same = all(abs(a[k] - b[k]) < 1e-9 for k in ('N', 'Dp', 'Dm', 'I')) and a['label'] == b['label'] and np.allclose(a['ci']['I'], b['ci']['I'])
    print('selftest reproduce v224: v224 %s I %+.5f | v228 %s I %+.5f -> %s' % (a['label'], a['I'], b['label'], b['I'], 'OK' if same else 'FAIL'), flush=True)
    ok &= same
    rng = np.random.default_rng(5); n = len(its)                      # answer-only world: a code of A = u op v + k
    u = np.array([i['u'] for i in its]); v = np.array([i['v'] for i in its]); o = np.array([i['ver'] == 'minus' for i in its])
    k = np.array([i['fam'] % 3 for i in its]); A = (np.where(o, u - v, u + v) + k) % 10
    for i, x in zip(its, A):
        i['A'] = int(x)
    U1, U2, UA = (rng.standard_normal((10, 256)) for _ in range(3)); offs = {f: rng.standard_normal(256) * 3 for f in set(i['fam'] for i in its)}
    base = U1[u] + U2[v] + 1.0 * UA[A] + np.stack([offs[i['fam']] for i in its])
    Xa = {l: (base + rng.standard_normal(base.shape)).astype(np.float32) for l in SH.SL}; get = lambda idx, l: Xa[l][idx]
    r0 = analyse(get, its, ['u', 'v'], nboot=1000); r1 = analyse(get, its, ['u', 'v', 'A'], nboot=1000)
    good = r1['label'] != 'FOLLOWS'; ok &= good
    print('selftest answer-only world: copies only %s (I/N %+.2f) | answer regressed %s (I/N %+.2f) -> %s' % (
        r0['label'], r0['I_rel'], r1['label'], r1['I_rel'], 'OK' if good else 'FAIL'), flush=True)
    Xb = {l: (base + 0.8 * SH.world(its, 'copies', 0, 2, comp=1.0)[l] + rng.standard_normal(base.shape)).astype(np.float32) for l in SH.SL}
    r2 = analyse(lambda idx, l: Xb[l][idx], its, ['u', 'v', 'A'], nboot=1000)
    good = r2['label'] == 'FOLLOWS'; ok &= good
    print('selftest answer + computed value: answer regressed %s (I/N %+.2f) -> %s' % (r2['label'], r2['I_rel'], 'OK' if good else 'FAIL'), flush=True)
    return ok


def load(root, stem):
    f = glob.glob(os.path.join(root, '**', stem + '_states.npy'), recursive=True)
    f = [x for x in f if 'gpu0' not in x and 'gpu1' not in x][0]
    m = json.load(open(f.replace('_states.npy', '_meta.json'), encoding='utf-8'))
    return np.load(f, mmap_mode='r'), m


def main(root):
    import v215_items as V5, v216_items as V6
    out = {}
    LI = list(range(0, 29, 4))
    print('=== T1 Kudo et al., their chain format (v215 fam states, Qwen2.5-7B) ===', flush=True)
    X, m = load(root, 'v215_fam'); fam, _, _ = V5.build(); assert m['items_sha'] == V5.all_sha() and len(m['items']) == len(fam)
    its = [dict(i, u=i['d1'], v=i['d2']) for i in fam]; P = m['positions']
    for p in ('eq', 'last'):
        r = analyse(lambda idx, l: X[idx, P.index(p), LI.index(l)], its, ['d1', 'd2']); say_r('T1 ' + p, r); out['T1_' + p] = r
        r = analyse(lambda idx, l: X[idx, P.index(p), LI.index(l)], its, ['d1', 'd2', 'A']); say_r('   (answer regressed too, descriptive) ' + p, r); out['T1a_' + p] = r
    print('=== T2 Kudo et al., answer only (v216 fam states, Qwen2.5-7B); primary: copies and answer regressed ===', flush=True)
    X, m = load(root, 'v216_fam'); its6 = V6.build(); assert len(m['items']) == len(its6) and V5.sha16(its6) == m['items_sha']
    its = [dict(i, u=i['d1'], v=i['d2']) for i in its6]; P = m['positions']
    for p in ('eq', 'pre', 'last'):
        r = analyse(lambda idx, l: X[idx, P.index(p), LI.index(l)], its, ['d1', 'd2', 'A']); say_r('T2 ' + p, r); out['T2_' + p] = r
        r = analyse(lambda idx, l: X[idx, P.index(p), LI.index(l)], its, ['d1', 'd2']); say_r('   (copies only, descriptive) ' + p, r); out['T2c_' + p] = r
    print('=== T3 Ouro-1.4B Study 3 (v194 states, main set); primary: copies a, b, c and the answer regressed; states mid3, end3, mid4, end4 ===', flush=True)
    X, m = load(root, 'v194'); ST = m['states']; sel = [k for k, i in enumerate(m['items']) if i['set'] == 'main']
    its = [dict(fam=i['fam'], pair=i['pair'], ver=i['op'], u=i['labels']['a'], v=i['labels']['b'], a=i['labels']['a'],
                b=i['labels']['b'], c=i['labels']['c'], ans=i['labels']['ans']) for i in (m['items'][k] for k in sel)]
    sel = np.array(sel); STS = ['mid3', 'end3', 'mid4', 'end4']
    for pi, p in ((1, 'P1'), (0, 'P0')):
        get = lambda idx, l, pi=pi: X[sel[idx], pi, ST.index(l)]
        r = analyse(get, its, ['a', 'b', 'c', 'ans'], cluster='pair', layers=STS); say_r('T3 ' + p, r); out['T3_' + p] = r
        r = analyse(get, its, ['a', 'b', 'c'], cluster='pair', layers=STS); say_r('   (copies only, descriptive) ' + p, r); out['T3c_' + p] = r
    json.dump(out, open(os.path.join(OUT, 'v228_result.json'), 'w'), indent=1)
    print('primary: T1 eq %s | T2 eq %s, pre %s, last %s | T3 P1 %s' % (out['T1_eq']['label'], out['T2_eq']['label'], out['T2_pre']['label'],
                                                                      out['T2_last']['label'], out['T3_P1']['label']), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        sys.exit(0 if selftest() else 1)
    main(sys.argv[1])
