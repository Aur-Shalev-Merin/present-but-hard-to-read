"""v206 analysis (docs/prereg_v206_ouro_template_curve.md): v205's B2 and B3 rules and code, unchanged, on Ouro's states.
Q = end1, end2, end3, end4 (Ouro's loop ends; v205 used Qwen layers 7-28). Labels as v205:
  B2  STAYS-SWAMPED (level <= 0.30 and gain <= 0.02) / CLIMBS (level >= 0.50 or gain >= 0.05) / PARTIAL; gate +0 at
      end1 (the analogue of v205's l7) >= 0.90; level and gain over Q and both probes (pca, full) at N_max.
  B3  minimal pairs on the first 100 test questions; contrastive mean over Q < 0.50 -> B2 UNINFORMATIVE.

  python v206_analyze.py <dir with v206_b2_s*_ and v206_b3_ files>
"""
import glob, os, sys
import numpy as np

import v205_analyze as A

A.Q = ['end1', 'end2', 'end3', 'end4']


def main(root):
    shards = sorted({os.path.basename(f).split('_states')[0] for f in glob.glob(os.path.join(root, '**', 'v206_b2_s*_states.npy'),
                                                                                 recursive=True)})
    print('shards:', shards, flush=True)
    parts = [A.load(root, s) for s in shards]
    X2 = np.concatenate([np.asarray(p[0]) for p in parts]); m2 = dict(parts[0][1])
    m2['items'] = sum([p[1]['items'] for p in parts], [])
    lab2 = b2_ouro(X2, m2)
    X3, m3 = A.load(root, 'v206_b3'); ok3 = A.b3(X3, m3)
    print('B2 final -> %s' % (lab2 if ok3 or lab2 == 'UNSCORABLE' else 'UNINFORMATIVE (B3: digit not shown present)'))


def b2_ouro(X2, m2):
    """v205's b2 with the apparatus gate read at end1 instead of l7 (Ouro has no l7)."""
    m2 = dict(m2); st = list(m2['states'])
    m2['states'] = [('l7' if s == 'end1' and 'l7' not in st else s) for s in st]   # gate reads 'l7' -> Ouro's end1
    A.Q = [('l7' if q == 'end1' else q) for q in ['end1', 'end2', 'end3', 'end4']]
    out = A.b2(X2, m2)
    A.Q = ['end1', 'end2', 'end3', 'end4']
    print('(in the B2 table above, "l7" is Ouro\'s end1)')
    return out


def selftest():
    """Synthetic Ouro-named states: readable -> CLIMBS, swamped -> STAYS-SWAMPED (through b2_ouro)."""
    rng = np.random.default_rng(0); d = 64; code = rng.standard_normal((10, d))
    for world, amp in (('readable', 2.0), ('swamped', 0.02)):
        n = 2400; y = rng.integers(0, 10, n); X = np.zeros((n, 3, 6, d), np.float32); ctx = rng.standard_normal((n, d)) * 3
        for st in range(6):
            X[:, 2, st] = ctx + amp * code[y]; X[:, 0, st] = ctx * 0.1 + 3 * code[y]
        its = [dict(digit=int(y[k]), idx=k, split='train' if k < 1900 else 'test') for k in range(n)]
        m = dict(items=its, states=['h0', 'mid1', 'end1', 'end2', 'end3', 'end4'], slots=['+0', 'qmark', 'last'])
        print('selftest %-9s -> %s' % (world, b2_ouro(X, m)), flush=True)


if __name__ == '__main__':
    selftest() if sys.argv[1] == '--selftest' else main(sys.argv[1])
