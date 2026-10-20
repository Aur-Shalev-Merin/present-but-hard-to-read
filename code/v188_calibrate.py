"""v188 bar calibration (pre-freeze; exploratory re-analysis of the FINISHED v186 run, old templates).

v188 moved its primary labels from mod 10 to mod 5 (parity confound). Its bars had been set from v186's mod-10
estimates, so they must be re-derived on the mod-5 scale. This re-runs v186's states (P0/P1, end4) with mod-5 labels
and v188's control sets, and prints the excess and D_XOR estimates the v188 bars are set from (at about half).

  python v188_calibrate.py <dir containing v186_states.npy, v186_meta.json (searched recursively)>
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v186_analyze as A

RND5 = np.random.default_rng(1882).integers(0, 5, (10, 10))


def add_mod5(L):
    a, b, c = L['a'], L['b'], L['c']
    return dict(L, m5=(a + b) % 5, amb5=(a - b) % 5, apc5=(a + c) % 5, ap2b5=(a + 2 * b) % 5, tapb5=(2 * a + b) % 5,
                abmc5=(a + b - 2 * c) % 5, abc5=(a + b + c) % 5, rnd5=int(RND5[a, b]))


def main(root):
    f = sorted(glob.glob(os.path.join(root, '**', 'v186_states.npy'), recursive=True))[0]
    meta = json.load(open(os.path.join(os.path.dirname(f), 'v186_meta.json')))
    for i in meta['items']:
        i['labels'] = add_mod5(i['labels'])
    X = np.load(f, mmap_mode='r'); E = meta['states'].index('end4'); its = meta['items']
    H = {'H1': ('bare', 0, ('amb5', 'ap2b5', 'tapb5')), 'H2': ('main', 1, ('amb5', 'apc5', 'abmc5', 'abc5'))}
    jobs = [(sp, p, E, l, ft) for sp, p, xor in H.values() for l in ('m5', 'rnd5') + xor for ft in ('state', 'ceil')]
    with Pool(4, initializer=A.init, initargs=(X, meta)) as pool:
        res = dict(pool.map(A.job, jobs))
    for name, (sp, p, xor) in H.items():
        fam = np.array([i['fam'] for i in its if i['set'] == sp])
        ex = {l: res[(sp, p, E, l, 'state')].astype(float) - res[(sp, p, E, l, 'ceil')] for l in ('m5', 'rnd5') + xor}
        bx = max(xor, key=lambda l: ex[l].mean())
        dc = A.cluster_ci(fam, ex['m5'], seed=188); dx = A.cluster_ci(fam, ex['m5'] - ex[bx], seed=188)
        dr = A.cluster_ci(fam, ex['m5'] - ex['rnd5'], seed=188)
        print('%s v186 %s P%d end4 mod 5 | raw %s' % (name, sp, p, ' '.join('%s %.3f/%.3f' % (
            l, res[(sp, p, E, l, 'state')].mean(), res[(sp, p, E, l, 'ceil')].mean()) for l in ('m5', 'rnd5') + xor)))
        print('%s excess %s | D_ceil %+.3f [%+.3f, %+.3f] | D_XOR vs %s %+.3f [%+.3f, %+.3f] | D_rnd %+.3f [%+.3f, %+.3f]' % (
            (name, ' '.join('%s %+.3f' % (l, ex[l].mean()) for l in ('m5', 'rnd5') + xor)) + dc + (bx,) + dx + dr), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
