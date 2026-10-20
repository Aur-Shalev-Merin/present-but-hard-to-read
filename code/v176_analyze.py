"""v176 analysis (written before any v176 data; rule in docs/v176_plan.md).

  python v176_analyze.py results_v176
Reads results_v176/gpu*/v176_{huginn,ouro}_*.json. Per model and loop: accuracy with a Wilson 95% CI, mean gold logp.
Ouro GO: best-loop accuracy >= 0.10 and (best - loop 1) >= 0.05 with its paired-bootstrap 95% CI excluding 0.
Huginn: 'consistent with Lu et al.' if the 32-loop accuracy is within the Wilson 95% CI of 0.049 at the same n.
"""
import glob, json, math, os, sys
import numpy as np


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(root, model):
    probs, checks = [], []
    for f in sorted(glob.glob(os.path.join(root, 'gpu*', 'v176_%s_*.json' % model))):
        r = json.load(open(f)); probs += r['problems']; checks.append(r['check'])
    probs.sort(key=lambda p: p['i'])
    return probs, checks


def main(root):
    for model in ('huginn', 'ouro'):
        P, checks = load(root, model)
        if not P:
            print(model, 'no results'); continue
        n = len(P); ck = sorted(int(t) for t in P[0]['res'])
        print('== %s: %d problems; checks %s; tokenisation ok %d/%d; answer appears in question %.3f' % (
            model, n, checks, sum(p['tok_ok'] for p in P), n, np.mean([p['in_question'] for p in P])))
        C = {t: np.array([p['res'][str(t)]['correct'] for p in P], float) for t in ck}
        for t in ck:
            lo, hi = wilson(C[t].sum(), n)
            print('  loops %2d: acc %.3f [%.3f, %.3f]  mean gold logp %.2f' % (
                t, C[t].mean(), lo, hi, np.mean([p['res'][str(t)]['logp'] for p in P])))
        if model == 'huginn' and 32 in C:
            lo, hi = wilson(round(0.049 * n), n)
            print('  replication of Lu et al. (4.9%% at 32 loops): ours %.3f, band [%.3f, %.3f] -> %s' % (
                C[32].mean(), lo, hi, 'CONSISTENT' if lo <= C[32].mean() <= hi else 'NOT CONSISTENT'))
        if model == 'ouro':
            best = max(ck, key=lambda t: C[t].mean()); d = C[best] - C[ck[0]]
            rng = np.random.default_rng(176); bs = d[rng.integers(0, n, (10000, n))].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            go = C[best].mean() >= 0.10 and d.mean() >= 0.05 and lo > 0
            print('  best loop %d: acc %.3f; best - loop 1 = %.3f [%.3f, %.3f] -> %s' % (
                best, C[best].mean(), d.mean(), lo, hi, 'GO' if go else 'NO-GO'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results_v176')
