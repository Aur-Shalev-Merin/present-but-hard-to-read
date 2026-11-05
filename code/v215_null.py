"""v215 mixing null (CPU; docs/prereg_v215_kudo_retest.md section 3; v212's procedure rebuilt on v215's items).

Synthetic states on the real family design (every second family, 120): a family offset, copies of d1 and d2, and one
generic nonlinear mixture of (d1, d2, o1), with NO computed value (neither B nor A):
  pair     sum over the pairs (d1, d2), (d1, o1), (d2, o1) of tanh(2 U[x] * V[y])        (random codes per pair)
  fourier  same-frequency products of cos/sin(2 pi f d / 10) of d1 and d2, f in {1, 2, 5}, through a random map per
           operation (so the mixture may depend on o1)
  mlp      R2 relu(R1 [onehot d1; onehot d2; onehot o1] + r0), 256 hidden units
scaled to lam times the RMS norm of the copy code, plus N(0, 1) noise (independent per position and layer), d = 256.
Statistics: v215_analyze.part_a (lite: band layers, S, M, d1, d2), position eq.
  calib  seed 100, LAMS -> ex_N(lam); lam* by log-linear interpolation to the observed ex_N; a family whose largest
         ex_N on the grid is below the observed one is UNCALIBRATED
  run    seeds 1..6 at lam* -> I
  BEYOND-MIXING if the observed I exceeds every calibrated world's I, else NOT-SEPARABLE (per COMBINED position)

  python v215_null.py <dir with v215_partA.json>       (positions labelled COMBINED)
  python v215_null.py --selftest                       (an op-independent pair world gives I near 0)
"""
import glob, json, os, sys
import numpy as np

import v215_analyze as A

LAMS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0); SEEDS = range(1, 7); KINDS = ('pair', 'fourier', 'mlp'); DD = 256


def world(its, kind, lam, seed):
    rng = np.random.default_rng(seed); n = len(its)
    d1 = np.array([i['d1'] for i in its]); d2 = np.array([i['d2'] for i in its]); o = np.array([i['o1'] == '-' for i in its], int)
    U1, U2 = rng.standard_normal((10, DD)), rng.standard_normal((10, DD)); cop = U1[d1] + U2[d2]
    if kind == 'pair':
        mix = sum(np.tanh(2 * rng.standard_normal((10, DD))[x] * rng.standard_normal((10, DD))[y])
                  for x, y in ((d1, d2), (d1, o), (d2, o)))
    elif kind == 'fourier':
        mix = np.zeros((n, DD))
        for f in (1, 2, 5):
            c1, s1, c2, s2 = (g(2 * np.pi * f * x / 10) for x in (d1, d2) for g in (np.cos, np.sin))
            F = np.stack([c1 * c2, s1 * s2, c1 * s2, s1 * c2], 1); R = rng.standard_normal((2, 4, DD))
            mix += np.einsum('nk,nkd->nd', F, R[o])
    else:
        H = np.concatenate([np.eye(10)[d1], np.eye(10)[d2], np.eye(2)[o]], 1)
        mix = np.maximum(H @ rng.standard_normal((22, 256)) + rng.standard_normal(256), 0) @ rng.standard_normal((256, DD))
    mix *= lam * np.sqrt((cop ** 2).sum(1).mean() / (mix ** 2).sum(1).mean())
    offs = {f: rng.standard_normal(DD) * 3 for f in set(i['fam'] for i in its)}
    sig = np.stack([offs[i['fam']] for i in its]) + cop + mix
    X = np.zeros((n, 3, len(A.LAYERS), DD), np.float32)
    X[:, 0] = sig[:, None, :] + rng.standard_normal((n, len(A.LAYERS), DD))
    return X


def stats(its, kind, lam, seed):
    out = A.part_a(world(its, kind, lam, seed), its, [dict(correct=True)] * len(its), say=lambda s: None,
                   positions=['eq'], plan=False, lite=True)['eq']
    return out['exN'][0], out['I'][0]


def items():
    import v215_items as I
    fam, _, _ = I.build(); keep = set(sorted(set(i['fam'] for i in fam))[::2])
    return [i for i in fam if i['fam'] in keep]


def null(obs_exN, its, say=print):
    worlds = {}
    for kind in KINDS:
        ex = np.array([stats(its, kind, lam, 100)[0] for lam in LAMS])
        say('calib %-7s ex_N over lam %s: %s' % (kind, LAMS, ' '.join('%+.3f' % e for e in ex)))
        if ex.max() < obs_exN:
            say('  %s UNCALIBRATED (max %.3f < observed %.3f)' % (kind, ex.max(), obs_exN)); continue
        j = int(np.argmax(ex >= obs_exN))
        lam = LAMS[0] if j == 0 else float(np.exp(np.interp(obs_exN, ex[j - 1:j + 1], np.log(LAMS[j - 1:j + 1]))))
        worlds[kind] = (lam, [stats(its, kind, lam, s) for s in SEEDS])
        say('  %s lam* %.3f: ex_N %s | I %s' % (kind, lam, ' '.join('%+.3f' % e for e, _ in worlds[kind][1]),
                                              ' '.join('%+.3f' % i for _, i in worlds[kind][1])))
    return worlds


def main(root):
    its = items(); obs = {}
    for tag, arm in (('', 'main'), ('_short', 'short')):
        f = glob.glob(os.path.join(root, '**', 'v215_partA%s.json' % tag), recursive=True)
        if f:
            obs.update({'%s/%s' % (arm, p): r for p, r in json.load(open(f[0])).items()})
    for pos, r in obs.items():
        if r['label'] != 'COMBINED':
            print('%s: %s, no null needed' % (pos, r['label'])); continue
        print('--- %s: observed ex_N %+.4f, I %+.4f ---' % (pos, r['exN'][0], r['I'][0]), flush=True)
        w = null(r['exN'][0], its, say=lambda s: print(s, flush=True))
        Is = [i for _, v in w.values() for _, i in v]
        lab = 'UNCALIBRATED (no family reached the observed ex_N)' if not Is else (
            'BEYOND-MIXING' if r['I'][0] > max(Is) else 'NOT-SEPARABLE')
        print('%s: null I max %s vs observed %+.4f -> %s' % (pos, '%+.4f' % max(Is) if Is else '-', r['I'][0], lab), flush=True)


def selftest():
    its = items()[:2400]
    ex, I = stats(its, 'pair', 2.0, 1)
    ok = abs(I) < 0.03
    print('selftest: pair world (no o1-specific computation), lam 2: ex_N %+.3f, I %+.3f -> %s' % (ex, I, 'OK' if ok else 'FAIL'))
    return ok


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        sys.exit(0 if selftest() else 1)
    main(sys.argv[1])
