"""v177 analysis (written before any v177 data; rule in docs/v177_plan.md).

  python v177_analyze.py <dir with v176_symbolic_ouro_*.json and v176_symbolic_orig_ouro_*.json>
  python v177_analyze.py --selftest

PRIMARY: Ouro answer-only loop gain on GSM-Symbolic (500 number/name-perturbed variants, 100 templates):
  G = acc(loop 4) - acc(loop 1); 95% CI by cluster bootstrap over templates (10,000 resamples, seed 177).
  REASONING-SUPPORTED  G >= 0.0825 (half the v176 GSM8K gain of 0.165) and CI lower > 0
  MEMORY-SUSPECT       CI upper < 0.0825  -> the v176 GO is withdrawn
  UNCLEAR              otherwise
SECONDARY (reported): accuracy per loop on variants and on their 100 originals; the original-minus-variant drop at loop 4
paired by template (cluster CI).
"""
import glob, json, math, os, sys
import numpy as np

HALF_GAIN = 0.0825


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(root, data):
    P = []
    for f in sorted(glob.glob(os.path.join(root, '**', 'v176_%s_ouro_*.json' % data), recursive=True)):
        P += json.load(open(f))['problems']
    return P


def cluster_boot(tids, vals, n=10000, seed=177):
    """vals: per-item values; mean over items, resampling templates."""
    tids = np.asarray(tids); vals = np.asarray(vals, float); u = np.unique(tids)
    groups = [vals[tids == t] for t in u]; sums = np.array([g.sum() for g in groups]); cnts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(u), (n, len(u)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analyse(S, O, verbose=True):
    out = []; say = out.append
    for name, P in (('variants', S), ('originals', O)):
        if not P:
            continue
        say('%s: %d items, %d templates' % (name, len(P), len({p['tid'] for p in P})))
        for t in sorted(int(k) for k in P[0]['res']):
            k = sum(p['res'][str(t)]['correct'] for p in P); lo, hi = wilson(k, len(P))
            say('  loop %d: acc %.3f [%.3f, %.3f]' % (t, k / len(P), lo, hi))
    tids = [p['tid'] for p in S]
    g = [float(p['res']['4']['correct']) - float(p['res']['1']['correct']) for p in S]
    G = cluster_boot(tids, g)
    label = 'REASONING-SUPPORTED' if (G[0] >= HALF_GAIN and G[1] > 0) else ('MEMORY-SUSPECT' if G[2] < HALF_GAIN else 'UNCLEAR')
    say('PRIMARY loop-4 minus loop-1 gain on variants: %.3f [%.3f, %.3f] (bar %.4f) -> %s' % (G + (HALF_GAIN, label)))
    if O:
        orig = {p['tid']: float(p['res']['4']['correct']) for p in O}
        d = [orig[p['tid']] - float(p['res']['4']['correct']) for p in S if p['tid'] in orig]
        say('secondary: original minus variant accuracy at loop 4 (paired by template): %.3f [%.3f, %.3f]' % cluster_boot(
            [p['tid'] for p in S if p['tid'] in orig], d))
    if verbose:
        print('\n'.join(out))
    return label, G


def selftest():
    rng = np.random.default_rng(0)

    def world(p1, p4):
        S = []
        for t in range(100):
            q = rng.uniform(0.5, 1.5)                       # per-template difficulty
            for k in range(5):
                S.append(dict(tid=t, res={'1': dict(correct=bool(rng.random() < min(1, p1 * q))),
                                          '4': dict(correct=bool(rng.random() < min(1, p4 * q)))}))
        return S
    want = [((0.05, 0.20), 'REASONING-SUPPORTED'), ((0.05, 0.07), 'MEMORY-SUSPECT'), ((0.05, 0.11), 'UNCLEAR')]
    ok = True
    for (p1, p4), w in want:
        lab, G = analyse(world(p1, p4), [], verbose=False)
        print('selftest p1 %.2f p4 %.2f -> %-20s (want %s) G %.3f [%.3f, %.3f]' % ((p1, p4, lab, w) + G)); ok &= lab == w
    print('SELFTEST', 'OK' if ok else 'FAILED'); return ok


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        sys.exit(0 if selftest() else 1)
    root = sys.argv[1]
    analyse(load(root, 'symbolic'), load(root, 'symbolic_orig'))
