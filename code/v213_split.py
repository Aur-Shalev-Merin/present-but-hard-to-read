"""v213 B2x split across kernels (execution only; docs/prereg_v213_scale_up.md, execution note 1). Runs the frozen
v213_analyze.b2x unchanged, with its process pool swapped out:

  part K   fits only layer K's jobs (the curve fits for both probes at every size, layer K's MLP at N_max, and the gate
           on part 0), largest first, one job per task, saving every result as it finishes. The effective-rank SVDs
           are skipped here (combine prints them).
  combine  feeds the saved results of parts 0-3 back into b2x, which prints the same tables, labels and amendment-1
           intervals as a single run (every fit is deterministic: fixed folds, PCA and MLP seeds).

  python v213_split.py <dir> qwen7b|ouro26 part <K>
  python v213_split.py <dir> qwen7b|ouro26 combine
  python v213_split.py --selftest      (synthetic world: split output must equal a single run's, line for line)
"""
import glob, json, os, pickle, sys, time
from multiprocessing import Pool as RealPool
import numpy as np

import v213_analyze as A

NPART = 4


class Stop(Exception):
    pass


def run(task):
    name, args = task
    return task, getattr(A, name)(args)[1]


class PartPool:
    """Stands in for multiprocessing.Pool in b2x. On b2x's first map (the curve jobs) it runs all of part K's jobs
    (curve, MLP, gate) together on a real pool, then stops b2x before the report."""

    def __init__(self, k, path):
        self.k, self.path, self.done = k, path, {}

    def __call__(self, procs, initializer=None, initargs=()):
        self.real = RealPool(procs, initializer=initializer, initargs=initargs)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.real.terminate()
        return False

    def map(self, fn, jobs):
        assert fn is A.curve_job
        tasks = ([('mlp_job', self.k)] + ([('gate_job', 0)] if self.k == 0 else [])
                 + [('curve_job', j) for j in jobs if j[0] == self.k])          # jobs arrive largest first
        t0 = time.time()
        for task, r in self.real.imap_unordered(run, tasks, chunksize=1):
            self.done[task] = r
            pickle.dump(self.done, open(self.path + '.tmp', 'wb')); os.replace(self.path + '.tmp', self.path)
            print('%-9s %-22s done %3d/%d (%.0fs)' % (task[0], task[1], len(self.done), len(tasks), time.time() - t0), flush=True)
        raise Stop


class RestPool(PartPool):
    """Part K's jobs that an earlier (time-limited) part run did not finish, run in this process one at a time (BLAS
    may use several threads), each result merged into the saved results as it finishes."""

    def __init__(self, k, path, done):
        super().__init__(k, path); self.done = dict(done)

    def __call__(self, procs, initializer=None, initargs=()):
        initializer(*initargs)
        return self

    def __exit__(self, *e):
        return False

    def map(self, fn, jobs):
        assert fn is A.curve_job
        tasks = ([('mlp_job', self.k)] + ([('gate_job', 0)] if self.k == 0 else [])
                 + [('curve_job', j) for j in jobs if j[0] == self.k])
        todo = [t for t in tasks if t not in self.done]
        print('part %d: %d of %d tasks saved, running %s' % (self.k, len(tasks) - len(todo), len(tasks), todo), flush=True)
        t0 = time.time()
        for t in todo:
            task, r = run(t); self.done[task] = r
            pickle.dump(self.done, open(self.path + '.tmp', 'wb')); os.replace(self.path + '.tmp', self.path)
            print('%-9s %-22s done (%.0fs)' % (task[0], task[1], time.time() - t0), flush=True)
        raise Stop


def log_fits():
    """Print each logistic fit's size, C, time and iterations (does not change any fit)."""
    from sklearn.linear_model import LogisticRegression
    fit = LogisticRegression.fit

    def timed(self, X, y, *a, **k):
        t = time.time(); out = fit(self, X, y, *a, **k)
        print('  fit n=%d d=%d C=%s: %.0fs, n_iter %s' % (X.shape[0], X.shape[1], self.C, time.time() - t,
                                                       getattr(self, 'n_iter_', None)), flush=True)
        return out
    LogisticRegression.fit = timed


class SavedPool:
    def __init__(self, done):
        self.done = done

    def __call__(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def map(self, fn, jobs):
        return [(j, self.done[(fn.__name__, j)]) for j in jobs]


class NoSVD:
    """numpy for b2x in part mode, without the effective-rank SVDs."""
    linalg = type('linalg', (), {'svd': staticmethod(lambda *a, **k: np.ones(2))})

    def __getattr__(self, n):
        return getattr(np, n)


def load_b2(root, model):                                 # as v213_analyze.main, part B2
    fs = sorted(glob.glob(os.path.join(root, '**', 'v213_b2_s*_states.npy'), recursive=True))
    Xs, items = [], []
    for f in fs:
        Xs.append(np.load(f, mmap_mode='r')); mt = json.load(open(f.replace('_states.npy', '_meta.json'))); items += mt['items']
    m2 = A.rename(dict(mt, items=items), model); print('B2x shards:', len(fs))
    if len(Xs) == 1:                                      # one shard: keep it memory-mapped (no 6 GB copy)
        return Xs[0], m2
    return np.concatenate([np.asarray(x) for x in Xs]), m2


def part(X, m2, k, path, say=lambda s: print('[part, not used] ' + s)):
    A.Pool, A.np = PartPool(k, path), NoSVD()
    try:
        A.b2x(X, m2, say=say)
    except Stop:
        pass
    finally:
        A.Pool, A.np = RealPool, np


def combine(X, m2, paths, say=print, **kw):
    done = {}
    for p in paths:
        done.update(pickle.load(open(p, 'rb')))
    A.Pool = SavedPool(done)
    try:
        return A.b2x(X, m2, say=say, **kw)
    finally:
        A.Pool = RealPool


def selftest():
    import tempfile
    rng = np.random.default_rng(0); d = 256; code = rng.standard_normal((10, d)); A.A5.Q = ['l7', 'l14', 'l21', 'l28']
    ng, no, nt = 600, 3000, 1000; n = ng + no + 2 * nt
    y = rng.integers(0, 10, n); ctx = rng.standard_normal((n, d)) * 3
    X = np.zeros((n, 3, 6, d), np.float32)
    for st in range(6):
        X[:, 2, st] = ctx + (0.1 + 0.1 * st) * code[y]; X[:, 0, st] = ctx * 0.1 + 3 * code[y]   # layers differ
    its = [dict(digit=int(y[k]), idx=k, rank=k, **dict(zip(('source', 'split'), (
        ('gsm', 'train') if k < ng else ('orca', 'train') if k < ng + no else ('gsm', 'test') if k < ng + no + nt
        else ('orca', 'test'))))) for k in range(n)]
    m = dict(items=its, states=['h0', 'l4', 'l7', 'l14', 'l21', 'l28'], slots=['+0', 'qmark', 'last'])
    kw = dict(ns=(60, 125, 250, 500), n_ext=(1000, 2000))
    one, two = [], []
    lab1 = A.b2x(X, m, say=one.append, **kw)
    tmp = tempfile.mkdtemp(); paths = [os.path.join(tmp, 'p%d.pkl' % k) for k in range(NPART)]
    for k in range(NPART):
        A.Pool, A.np = PartPool(k, paths[k]), NoSVD()
        try:
            A.b2x(X, m, say=lambda s: None, **kw)
        except Stop:
            pass
        finally:
            A.Pool, A.np = RealPool, np
    lab2 = combine(X, m, paths, say=two.append, **kw)
    ok = lab1 == lab2 and one == two
    print('\n'.join(two)); print('split selftest: single %s, split %s, %d lines, identical %s -> %s' % (
        lab1, lab2, len(one), one == two, 'OK' if ok else 'FAIL'), flush=True)
    return ok


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        sys.exit(0 if selftest() else 1)
    root, model, mode = sys.argv[1:4]
    A.set_q(model)
    X, m2 = load_b2(root, model)
    out = os.environ.get('OUT_DIR', '.')
    if mode == 'part':
        k = int(sys.argv[4])
        part(X, m2, k, os.path.join(out, 'v213_b2part_%s_%d.pkl' % (model, k)))
    elif mode == 'rest':
        k = int(sys.argv[4]); log_fits()
        prev = glob.glob(os.path.join(root, '**', 'v213_b2part_%s_%d.pkl' % (model, k)), recursive=True)
        print('saved part results:', prev); done = {}
        for p in prev:
            done.update(pickle.load(open(p, 'rb')))
        A.Pool, A.np = RestPool(k, os.path.join(out, 'v213_b2part_%s_%d.pkl' % (model, k)), done), NoSVD()
        try:
            A.b2x(X, m2, say=lambda s: None)
        except Stop:
            pass
    else:
        paths = sorted(glob.glob(os.path.join(root, '**', 'v213_b2part_%s_*.pkl' % model), recursive=True))
        print('parts:', len(paths), paths)          # a rest run's file holds its part's earlier results too
        assert len(paths) >= NPART, 'need all %d parts' % NPART
        print('B2x:', combine(X, m2, paths))
