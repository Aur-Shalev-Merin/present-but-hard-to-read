"""v215 pilot, probe part (CPU; exploratory). Kudo et al.'s standard probe, small, on the pilot's saved states: one
logistic probe per (position, layer) for b's value, a's value and the copies d1, d2, trained on single prompts.
Train/test split by arithmetic key (k, op2, d1, op1, d2): test problems never share their arithmetic with training
problems (their no-overlap rule). No minimal-pair statistic.

  python v215_pilot_probe.py <dir with v215_pilot_states.npy and v215_pilot_items.json>
"""
import glob, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

P = ['d2', 'semi', 'last']; LAYERS = list(range(0, 29, 4)); G = {}


def op(x, o, y):
    return x + y if o == '+' else x - y


def init(d):
    G.update(d)


def job(args):
    name, a_, b_ = args
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y, tr = G['X'], G[name], G['tr']
    Z = np.asarray(X[:, a_, b_], np.float32); sc = StandardScaler().fit(Z[tr])
    m = LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(Z[tr]), y[tr])
    return args, float((m.predict(sc.transform(Z[~tr])) == y[~tr]).mean())


def main(root):
    t0 = time.time()
    f = glob.glob(os.path.join(root, '**', 'v215_pilot_states.npy'), recursive=True)[0]
    X = np.load(f); its = json.load(open(f.replace('_states.npy', '_items.json')))
    split = os.environ.get('V215_SPLIT', 'key')           # key: whole arithmetic; triple: b's equation (d1, op1, d2)
    keys = [((p['d1'], p['o1'], p['d2']) if split == 'triple' else (p['k'], p['o2'], p['d1'], p['o1'], p['d2'])) for p in its]
    uk = sorted(set(keys)); rng = np.random.default_rng(2150); test_keys = set(map(tuple, [uk[i] for i in rng.permutation(len(uk))[:len(uk) // 5]]))
    tr = np.array([k not in test_keys for k in keys])
    B = np.array([op(p['d1'], p['o1'], p['d2']) for p in its]); A = np.array([op(p['k'], p['o2'], b) for p, b in zip(its, B)])
    d = dict(X=X, tr=tr, b=B, a=A, d1=np.array([p['d1'] for p in its]), d2=np.array([p['d2'] for p in its]))
    print('split by %s | ' % split + '%d problems, %d arithmetic keys (%d test keys): train %d / test %d' % (len(its), len(uk), len(test_keys), tr.sum(), (~tr).sum()), flush=True)
    jobs = [(n, a_, b_) for n in ('b', 'a', 'd1', 'd2') for a_ in range(len(P)) for b_ in range(len(LAYERS))]
    res = {}
    with Pool(4, initializer=init, initargs=(d,)) as pool:
        for args, acc in pool.imap_unordered(job, jobs):
            res[args] = acc
            if len(res) % 12 == 0:
                print('  %d/%d fits (%.0fs)' % (len(res), len(jobs), time.time() - t0), flush=True)
    for n in ('b', 'a', 'd1', 'd2'):
        y = d[n][~tr]
        print('%-2s majority %.3f | ' % (n, np.bincount(y).max() / len(y)) + ' | '.join(
            '%s best %.3f (layer %d)' % (P[a_], max(res[(n, a_, b_)] for b_ in range(len(LAYERS))),
                                         LAYERS[int(np.argmax([res[(n, a_, b_)] for b_ in range(len(LAYERS))]))]) for a_ in range(len(P))), flush=True)
    print('by layer, last token: ' + ' | '.join('%s %s' % (n, ' '.join('%.2f' % res[(n, 2, b_)] for b_ in range(len(LAYERS))))
                                                for n in ('b', 'a', 'd1', 'd2')), flush=True)
    print('done (%.0fs)' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
