"""v216 answer-mixing null (CPU; docs/prereg_v216_answer_only.md section 3; v212's design on v216's items).

Synthetic states on the real family design (every second family, 120): a family offset, copies of d1 and d2, a strong
answer code (A's value; RMS norm equal to the copy code's), and one generic nonlinear mixture of (d1, d2, o1, A), with
NO b term:
  pair     sum over the pairs of (d1, d2, o1, A) of tanh(2 U[x] * V[y])           (6 pairs, random codes per pair)
  fourier  same-frequency products of cos/sin(2 pi f x / 10) over the pairs of (d1, d2, A), f in {1, 2, 5}, through a
           random map per operation
  mlp      R2 relu(R1 [onehot d1; d2; o1; A] + r0), 256 hidden units
scaled to lam times the copy code's RMS norm, plus N(0, 1) noise (independent per position and layer), d = 256.
Statistics: v215_analyze.part_a (lite), position eq. Calibration as v215_null (seed 100, LAMS, lam* by log-linear
interpolation to the observed ex_N; UNCALIBRATED if never reached), then seeds 1..6 at lam*.
BEYOND-ANSWER-MIXING if the observed I exceeds every calibrated world's I, else NOT-SEPARABLE.

  python v216_null.py <dir with v216_partA.json>
  python v216_null.py --selftest
"""
import glob, json, os, sys
import numpy as np

import v216_analyze                                     # sets v215_analyze.POS to v216's positions
import v215_analyze as A

LAMS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0); SEEDS = range(1, 7); KINDS = ('pair', 'fourier', 'mlp'); DD = 256


def world(its, kind, lam, seed):
    rng = np.random.default_rng(seed); n = len(its)
    v = {k: np.array([i[k] for i in its]) for k in ('d1', 'd2', 'A')}; v['o1'] = np.array([i['o1'] == '-' for i in its], int)
    U1, U2, UA = (rng.standard_normal((10, DD)) for _ in range(3)); cop = U1[v['d1']] + U2[v['d2']]
    ans = UA[v['A']]; ans *= np.sqrt((cop ** 2).sum(1).mean() / (ans ** 2).sum(1).mean())
    if kind == 'pair':
        keys = ['d1', 'd2', 'o1', 'A']
        mix = sum(np.tanh(2 * rng.standard_normal((10, DD))[v[x]] * rng.standard_normal((10, DD))[v[y]])
                  for i, x in enumerate(keys) for y in keys[i + 1:])
    elif kind == 'fourier':
        mix = np.zeros((n, DD))
        for f in (1, 2, 5):
            for x, y in (('d1', 'd2'), ('d1', 'A'), ('d2', 'A')):
                cx, sx, cy, sy = (g(2 * np.pi * f * v[z] / 10) for z in (x, y) for g in (np.cos, np.sin))
                F = np.stack([cx * cy, sx * sy, cx * sy, sx * cy], 1); R = rng.standard_normal((2, 4, DD))
                mix += np.einsum('nk,nkd->nd', F, R[v['o1']])
    else:
        H = np.concatenate([np.eye(10)[v['d1']], np.eye(10)[v['d2']], np.eye(2)[v['o1']], np.eye(10)[v['A']]], 1)
        mix = np.maximum(H @ rng.standard_normal((32, 256)) + rng.standard_normal(256), 0) @ rng.standard_normal((256, DD))
    mix *= lam * np.sqrt((cop ** 2).sum(1).mean() / (mix ** 2).sum(1).mean())
    offs = {f: rng.standard_normal(DD) * 3 for f in set(i['fam'] for i in its)}
    sig = np.stack([offs[i['fam']] for i in its]) + cop + ans + mix
    X = np.zeros((n, 3, len(A.LAYERS), DD), np.float32)
    X[:, 0] = sig[:, None, :] + rng.standard_normal((n, len(A.LAYERS), DD))
    return X


def stats(its, kind, lam, seed):
    out = A.part_a(world(its, kind, lam, seed), its, [dict(correct=True)] * len(its), say=lambda s: None,
                   positions=['eq'], plan=False, lite=True)['eq']
    return out['exN'][0], out['I'][0]


def items():
    import v216_items as I
    fam = I.build(); keep = set(sorted(set(i['fam'] for i in fam))[::2])
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
    obs = json.load(open(glob.glob(os.path.join(root, '**', 'v216_partA.json'), recursive=True)[0])); its = items()
    for pos, r in obs.items():
        if r['label'] != 'COMBINED':
            print('%s: %s, no null needed' % (pos, r['label'])); continue
        print('--- %s: observed ex_N %+.4f, I %+.4f ---' % (pos, r['exN'][0], r['I'][0]), flush=True)
        w = null(r['exN'][0], its, say=lambda s: print(s, flush=True))
        Is = [i for _, v in w.values() for _, i in v]
        lab = 'UNCALIBRATED' if not Is else ('BEYOND-ANSWER-MIXING' if r['I'][0] > max(Is) else 'NOT-SEPARABLE')
        print('%s: null I max %s vs observed %+.4f -> %s' % (pos, '%+.4f' % max(Is) if Is else '-', r['I'][0], lab), flush=True)


def selftest():
    its = items()[:2400]
    for kind in KINDS:
        ex, I = stats(its, kind, 1.0, 1)
        print('selftest: %s world with the answer code, lam 1: ex_N %+.3f, I %+.3f (recorded: what answer mixing gives)' % (kind, ex, I), flush=True)
    return True


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        sys.exit(0 if selftest() else 1)
    main(sys.argv[1])
