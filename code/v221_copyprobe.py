"""v221 part 2 (post hoc, descriptive; docs/prereg_v221_positive_control.md section 6): Kudo et al.'s standard probe
on the copied operands d1 and d2, in their own task (v215's replication set, their chain format, states saved by
astrocyte-v215-x at eq / semi / last x hidden_states 0, 4, ..., 28). The operands are in the prompt, so they are present
(v215's minimal pairs read them at 1.000 at every position). Probe: v215_analyze.cjob's settings (standardised states,
logistic C = 0.01), train 10,000 / test 2,000 (no shared b equation). Reported per label, position and layer, the
maximum per position, and whether it reaches their 90% threshold.

  python v221_copyprobe.py <dir with v215_rep_states.npy>
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

OUT = os.environ.get('OUT_DIR', '.'); LAYERS = list(range(0, 29, 4)); POS = ['eq', 'semi', 'last']
G = {}


def init(d):
    G.update(d)


def job(args):
    lab, pi, li = args
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y, tr = G['X'], G[lab], G['tr']
    Z = np.asarray(X[:, pi, li], np.float32); sc = StandardScaler().fit(Z[tr])
    m = LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(Z[tr]), y[tr])
    return args, float((m.predict(sc.transform(Z[~tr])) == y[~tr]).mean())


def main(root, procs=4):
    import v215_items as V5
    _, rep, _ = V5.build(); s = V5.all_sha()
    print('v215 items sha %s (frozen %s)' % (s, V5.ITEMS_SHA), flush=True)
    f = glob.glob(os.path.join(root, '**', 'v215_rep_states.npy'), recursive=True)[0]
    X = np.load(f, mmap_mode='r'); m = json.load(open(f.replace('_states.npy', '_meta.json')))
    assert m['items_sha'] == s and len(m['items']) == len(rep)
    tr = np.array([i['split'] == 'train' for i in rep])
    d = dict(X=X, tr=tr, d1=np.array([i['d1'] for i in rep]), d2=np.array([i['d2'] for i in rep]))
    jobs = [(lab, pi, li) for lab in ('d1', 'd2') for pi in range(len(POS)) for li in range(len(LAYERS))]
    with Pool(procs, initializer=init, initargs=(d,)) as pool:
        res = dict(pool.imap_unordered(job, jobs, chunksize=1))
    out = {}
    for lab in ('d1', 'd2'):
        y = d[lab][~tr]
        print('%s (majority %.3f), test accuracy by layer %s:' % (lab, np.bincount(y).max() / len(y), LAYERS), flush=True)
        for pi, p in enumerate(POS):
            row = [res[(lab, pi, li)] for li in range(len(LAYERS))]
            print('  %-4s %s | max %.3f -> %s their 90%% threshold' % (p, ' '.join('%.2f' % r for r in row), max(row),
                                                                     'reaches' if max(row) >= 0.90 else 'below'), flush=True)
            out['%s_%s' % (lab, p)] = dict(max=max(row), layers=row)
    json.dump(out, open(os.path.join(OUT, 'v221_copyprobe.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
