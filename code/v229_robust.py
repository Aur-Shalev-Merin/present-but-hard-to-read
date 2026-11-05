"""v229: Ouro Study 3 P1 under the share swap, robustness (docs/prereg_v229_ouro_p1_robust.md).

Two gaps in v228's FOLLOWS for Ouro P1 (I +0.004, I / N 0.63, N 0.007), raised by an outside review:
  A  carry / borrow   P1 is the answer's tens digit. Within a family the tens digit is fixed by the first-step carry
                      (plus) or borrow (minus) and the second-step borrow (checked on all 72 families: 0 exceptions),
                      both sign-specific. v228 regressed the units digits of a, b, c and the answer, not this.
                      Part A adds cb = c1 + 2 c2 (one-hot, 4 classes = span of c1, c2, c1 c2 within a family) to the
                      regressed copies; otherwise v228's analyse unchanged (TAU 0.30, pair cluster, mid3-end4).
  B  design-matched null   TAU 0.30 came from Qwen's two-operand designs. Part B builds v224's five null kinds on
                      Study 3's own items (main set, 14,400): copies of a, b, c (units), a strong answer code, a
                      carry / borrow code, and one generic mixture of (a, b, op) with no a op b term, with lam
                      extended down to 0.05 so that some worlds reach Ouro's small N. Each world is scored with Part A's
                      statistic. TAU3 = max(0.20, 1.5 x the largest I / N of any run with signal, rounded up to 0.05),
                      v224's rule. No answer x c mixture is included: on units digits it contains a op b (the
                      alternative the paper already states cannot be separated).

  python v229_robust.py --selftest     carry-only world: not FOLLOWS once cb is regressed; computed world FOLLOWS
  python v229_robust.py <root>         Part A on the v194 states, then Part B
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v224_share as SH
import v228_rescore as R

OUT = os.environ.get('OUT_DIR', '.'); STS = ['mid3', 'end3', 'mid4', 'end4']; DD = SH.DD
KEYS = ['a', 'b', 'c', 'ans', 'cb']                       # Part A's regressed copies
LAMS3 = (0.05, 0.1) + SH.LAMS                              # v224's grid, extended down
SEEDS = range(1, 7)


def items3(meta_items):
    """Study 3 main-set items with the fields R.analyse needs, plus cb."""
    out = []
    for i in meta_items:
        if i['set'] != 'main':
            continue
        L = i['labels']; au, bu, cu = L['a'], L['b'], L['c']; plus = i['op'] == 'plus'
        first = au + bu if plus else au - bu
        c1 = (au + bu >= 10) if plus else (au < bu); c2 = (first % 10) < cu
        out.append(dict(fam=i['fam'], pair=i['pair'], ver=i['op'], u=au, v=bu, a=au, b=bu, c=cu, ans=L['ans'], cb=int(c1) + 2 * int(c2)))
    return out


def world3(its, kind, lam, seed, comp=0.0, cbs=1.0):
    """v224_share.world on Study 3's design: copies a, b, c; answer and carry codes at the copies' RMS; one mixture of
    (a, b, op) at lam x RMS; optional computed a op b code at comp x RMS; family offsets; N(0, 1) noise per state."""
    rng = np.random.default_rng(seed); n = len(its)
    u, v, c, A, cb = (np.array([i[k] for i in its]) for k in ('a', 'b', 'c', 'ans', 'cb'))
    o = np.array([i['ver'] == 'minus' for i in its], int)
    U1, U2, U3 = (rng.standard_normal((10, DD)) for _ in range(3)); cop = U1[u] + U2[v] + U3[c]
    rms = np.sqrt((cop ** 2).sum(1).mean()); sc = lambda x: x * rms / np.sqrt((x ** 2).sum(1).mean())
    if kind == 'pair':
        mix = sum(np.tanh(2 * rng.standard_normal((10, DD))[x] * rng.standard_normal((10, DD))[y]) for x, y in ((u, v), (u, o), (v, o)))
    elif kind == 'fourier':
        mix = np.zeros((n, DD))
        for f in (1, 2, 5):
            c1, s1, c2, s2 = (g(2 * np.pi * f * x / 10) for x in (u, v) for g in (np.cos, np.sin))
            Fm = np.stack([c1 * c2, s1 * s2, c1 * s2, s1 * c2], 1); Rm = rng.standard_normal((2, 4, DD))
            mix += np.einsum('nk,nkd->nd', Fm, Rm[o])
    elif kind == 'mlp':
        H = np.concatenate([np.eye(10)[u], np.eye(10)[v], np.eye(2)[o]], 1)
        mix = np.maximum(H @ rng.standard_normal((22, 256)) + rng.standard_normal(256), 0) @ rng.standard_normal((256, DD))
    elif kind == 'paircode':
        mix = rng.standard_normal((10, 10, DD))[u, v]
    elif kind == 'paircode_op':
        mix = rng.standard_normal((2, 10, 10, DD))[o, u, v]
    elif kind == 'none':
        mix = np.zeros((n, DD))
    else:
        raise ValueError(kind)
    sig = cop + sc(rng.standard_normal((10, DD))[A]) + cbs * sc(rng.standard_normal((2, 4, DD))[o, cb])
    if mix.any():
        sig = sig + lam * sc(mix)
    if comp:
        val = np.where(o == 0, u + v, u - v) % 10; sig = sig + comp * sc(rng.standard_normal((10, DD))[val])   # units digit of a op b
    offs = {f: rng.standard_normal(DD) * 3 for f in set(i['fam'] for i in its)}
    sig = sig + np.stack([offs[i['fam']] for i in its])
    return {l: (sig + rng.standard_normal((n, DD))).astype(np.float32) for l in STS}


def score(its, Xw, keys=KEYS, nboot=1000, tau=None):
    return R.analyse(lambda idx, l: Xw[l][idx], its, keys, cluster='pair', layers=STS, nboot=nboot, tau=tau)


def _null(args):
    its, kind, lam, seed = args
    r = score(its, world3(its, kind, lam, seed))
    return dict(kind=kind, lam=lam, seed=seed, N=r['N'], NCIlo=r['ci']['N'][0], I=r['I'], I_rel=r['I_rel'], Dp=r['Dp'], Dm=r['Dm'], label=r['label'])


def nullgrid(its, procs=4):
    jobs = [(its, k, lam, s) for k in SH.NULL_KINDS for lam in LAMS3 for s in SEEDS]
    with Pool(procs) as p:
        rows = p.map(_null, jobs)
    sig = [x for x in rows if x['NCIlo'] > 0 and np.isfinite(x['I_rel'])]
    worst = max([x['I_rel'] for x in sig], default=0.0)
    tau3 = float(max(0.20, np.ceil(1.5 * worst / 0.05 - 1e-9) * 0.05))
    return rows, worst, tau3


def selftest():
    its = items3(json.load(open('v194_items.json'))['items']); ok = True
    Xc = world3(its, 'none', 0, 1)                                     # carry / borrow + answer + copies, no a op b
    r0 = score(its, Xc, ['a', 'b', 'c', 'ans']); r1 = score(its, Xc)
    good = r1['label'] != 'FOLLOWS'; ok &= good
    print('selftest carry-only world: v228 copies %s (I/N %+.2f) | cb regressed %s (I/N %+.2f) -> %s' % (
        r0['label'], r0['I_rel'], r1['label'], r1['I_rel'], 'OK' if good else 'FAIL'), flush=True)
    for comp in (0.5, 1.0):
        r2 = score(its, world3(its, 'pair', 1.0, 2, comp=comp))
        good = r2['label'] == 'FOLLOWS'; ok &= good
        print('selftest computed world comp %.1f: cb regressed %s (I/N %+.2f) -> %s' % (comp, r2['label'], r2['I_rel'], 'OK' if good else 'FAIL'), flush=True)
    return ok


def main(root):
    X, m = R.load(root, 'v194'); ST = m['states']
    sel = np.array([k for k, i in enumerate(m['items']) if i['set'] == 'main']); its = items3(m['items'])
    assert len(its) == len(sel) == 14400
    out = {}
    print('=== Part A: Ouro Study 3, copies a, b, c, answer and carry / borrow regressed; states mid3-end4 ===', flush=True)
    for pi, p in ((1, 'P1'), (0, 'P0')):
        get = lambda idx, l, pi=pi: X[sel[idx], pi, ST.index(l)]
        r = R.analyse(get, its, KEYS, cluster='pair', layers=STS); R.say_r('A ' + p, r); out['A_' + p] = r
    r = R.analyse(lambda idx, l: X[sel[idx], 1, ST.index(l)], its, ['a', 'b', 'c', 'ans'], cluster='pair', layers=STS)
    R.say_r('   (v228 copies, reproduces v228) P1', r); out['v228_P1'] = r
    print('=== Part B: design-matched null (5 kinds x %d lam x seeds 1-6), Part A statistic ===' % len(LAMS3), flush=True)
    rows, worst, tau3 = nullgrid(its)
    for k in SH.NULL_KINDS:
        ks = [x for x in rows if x['kind'] == k]; sg = [x for x in ks if x['NCIlo'] > 0]
        print('  %-11s signal %2d/%d; FOLLOWS %d; I/N max with signal %s; N range with signal %s' % (
            k, len(sg), len(ks), sum(x['label'] == 'FOLLOWS' for x in ks), '%+.3f' % max(x['I_rel'] for x in sg) if sg else '-',
            '%.4f-%.4f' % (min(x['N'] for x in sg), max(x['N'] for x in sg)) if sg else '-'), flush=True)
    out.update(null_rows=rows, null_worst=worst, tau3=tau3)
    p1 = out['A_P1']; f30 = p1['label'] == 'FOLLOWS'
    f3 = f30 and p1['I'] >= tau3 * p1['N']
    verdict = 'STANDS' if f3 else 'NOT-BEYOND-DESIGN-MIXING' if f30 else 'NOT-SHOWN'
    out['verdict'] = verdict
    print('largest null I/N with signal %.3f -> TAU3 %.2f | P1 (cb regressed) %s, I/N %.2f -> %s' % (worst, tau3, p1['label'], p1['I_rel'], verdict), flush=True)
    json.dump(out, open(os.path.join(OUT, 'v229_result.json'), 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        sys.exit(0 if selftest() else 1)
    main(sys.argv[1])
