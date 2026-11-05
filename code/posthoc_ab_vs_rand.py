"""Post hoc: hit(ab) - hit(rand) alone, same cluster bootstrap and interval as v194 / v209.

Run after the v231 power study (not pre-registered). Output: results/posthoc_ab_vs_rand.txt
    python code/posthoc_ab_vs_rand.py > results/posthoc_ab_vs_rand.txt
"""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')   # the repository's results/ folder


def load(pat):
    rows = []
    for f in sorted(glob.glob(os.path.join(R, pat), recursive=True)):
        rows += json.load(open(f))['rows']
    return rows


def score(rows, kind, t, Q, seed):
    rs = [r for r in rows if r['type'] == kind]
    if not rs or ('ab@%s' % t) not in rs[0]:
        return
    fam = np.array([r['fam'] for r in rs])
    hit = {c: np.array([r['%s@%s' % (c, t)] == str(r['target']) for r in rs], float) for c in ('ab', 'amb', 'rand')}
    u = np.unique(fam); idx = {f: np.where(fam == f)[0] for f in u}
    rng = np.random.default_rng(seed); bs = []
    for _ in range(10000):
        p = np.concatenate([idx[f] for f in rng.choice(u, len(u))])
        bs.append(hit['ab'][p].mean() - hit['rand'][p].mean())
    d = hit['ab'].mean() - hit['rand'].mean(); lo, hi = np.percentile(bs, Q)
    lab = 'USED' if d >= 0.10 and lo > 0 else 'NOT-USED' if hi < 0.02 else 'UNCLEAR'
    print('%-3s %-3s n %4d | ab %.3f amb %.3f rand %.3f | ab - rand %+.3f [%+.3f, %+.3f] -> %s' % (
        kind, t, len(rs), hit['ab'].mean(), hit['amb'].mean(), hit['rand'].mean(), d, lo, hi, lab))


print('v194 patching at P1 (99.375%)')
v194 = load('results_v194/patch/**/v194_patch_part*.json')
for k in ('B1', 'B2'):
    for t in ('e3', 'e2'):
        score(v194, k, t, (0.3125, 99.6875), 1942)
print('v209 (99.17%)')
v209 = load('results_v209/main_all/v209_patch_part*.json')
for k in ('B1', 'B2'):
    for t in ('e1', 'cl', 'ab3'):
        score(v209, k, t, (100 * 0.05 / 12, 100 - 100 * 0.05 / 12), 2091)
