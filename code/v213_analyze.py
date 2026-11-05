"""v213 analysis (docs/prereg_v213_scale_up.md). Frozen v202/v205 functions on the new models' states, plus the extended
many-question curve.

  A   v202_analyze.analyse on the v181 design (minimal pairs vs single-prompt decoders at tens of templates) -> v202's
      label (REPLICATES / READABLE / MIXED / ABSENT / UNSCORABLE).
  B1  v205_analyze.b1 (word swap and embedding noise vs the digit) -> GENERIC / TOKEN-SPECIFIC / MIXED / UNINFORMATIVE.
  B2x the many-question curve. Training order: the GSM8K training pool in item order (v213_items) shuffled with seed 205
      (the same kind of random subset as v205/v206, independent of sharding, not the identical subsets), then the Orca
      training pool by rank. Grid: v205's sizes up to the GSM8K maximum (60, 125, 250, 500, 1000, 2000, 2624, 4000,
      5249) and then 10000, 20000, N_max/2, N_max (N_max = 45249). Probes as v205 (PCA-256 and full-state logistic, C
      chosen by inner 3-fold CV grouped by question) and an MLP at N_max; each fit is scored on the GSM8K test set
      (primary) and the Orca test set (secondary), and on each test set without the questions that have a reworded twin
      in the GSM8K training pool (secondary; v213_items twin flag). Labels at N_max on the GSM8K test set, over Q and
      both probes:
        level = max accuracy at N_max; gain = max over Q and probes of acc(N_max) - acc(N_max/2)
        CLOSES          level >= 0.80
        STILL-CLIMBING  level < 0.80 and gain >= 0.05
        PLATEAUS-BELOW  level <= 0.50 and gain <= 0.02, AND on the Orca test set level <= 0.50 and gain <= 0.02
        SHIFT-LIMITED   flat and low on the GSM8K test set but not on the Orca test set (the added data helps its own
                        distribution, not GSM8K; no claim that more data fails to help)
        PARTIAL         otherwise
      The GSM8K-only part of the curve (N <= 5249) is also labelled with v205's rule, for comparison with v205/v206.
      Gate: the PCA probe on the digit's own token (+0 slot, gate state) at N_max >= 0.90, else UNSCORABLE.
  B3  v205_analyze.b3 separately on the GSM8K and the Orca test questions (minimal pairs; >= 0.50 = present).
State names: qwen7b as v202 (Q = l7, l14, l21, l28); ouro26 h0, mid1, end1-end4, with end1 renamed 'l7' for the frozen
code (as v206), so Q = end1-end4. Gate G3 of v202 (cached-vs-full check < 1e-3) is applied to Ouro's absolute check;
for Qwen-7B in fp16 the extraction's relative check (< 1e-2, else it aborts) stands in for it.

  python v213_analyze.py <dir> qwen7b|ouro26 [A B1 B2 B3]
  python v213_analyze.py --selftest
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v202_analyze as A2
import v205_analyze as A5

GSM_NS = (60, 125, 250, 500, 1000, 2000, 2624, 4000, 5249)
G = {}


def rename(meta, model):
    m = dict(meta)
    if model == 'ouro26':
        m['states'] = [('l7' if s == 'end1' else s) for s in meta['states']]
    if meta.get('check_kind') == 'relative':
        m['check'] = 0.0 if meta['check'] < 1e-2 else meta['check']
    return m


def set_q(model):
    q = ['l7', 'l14', 'l21', 'l28'] if model == 'qwen7b' else ['l7', 'end2', 'end3', 'end4']
    A2.Q = list(q); A5.Q = list(q)
    return q


def init(d):
    G.update(d)


def curve_job(args):
    qi, n, probe = args
    tr = G['order'][:n]; X = G['X'][:, qi]
    pred, C = A5.fit_c(np.asarray(X[tr], np.float32), G['y'][tr], G['g'][tr],
                       np.asarray(X[G['te_all']], np.float32), probe == 'pca', A5.GRIDS[probe])
    yt = G['y'][G['te_all']]; src = G['src'][G['te_all']]; tw = G['twin'][G['te_all']]
    acc = {s: float((pred[src == s] == yt[src == s]).mean()) for s in ('gsm', 'orca')}
    acc.update({s + '_clean': float((pred[(src == s) & ~tw] == yt[(src == s) & ~tw]).mean()) for s in ('gsm', 'orca')})
    acc['_hit'] = (pred == yt)                      # per-test-question correctness (amendment 1: mean-gain CI)
    return args, (acc, C)


def gate_job(args):
    tr = G['order']; pred, _ = A5.fit_c(np.asarray(G['X0'][tr], np.float32), G['y'][tr], G['g'][tr],
                                        np.asarray(G['X0'][G['te_gsm']], np.float32), True, A5.GRIDS['pca'])
    return args, float((pred == G['y'][G['te_gsm']]).mean())


def mlp_job(args):
    qi = args; tr = G['order']; X = G['X'][:, qi]
    pred = A5.mlp_fit_predict(np.asarray(X[tr], np.float32), G['y'][tr], G['g'][tr], np.asarray(X[G['te_all']], np.float32))
    yt = G['y'][G['te_all']]; src = G['src'][G['te_all']]
    return args, {s: float((pred[src == s] == yt[src == s]).mean()) for s in ('gsm', 'orca')}


def label_ext(level, gain, olevel, ogain):
    """GSM8K test (primary) level/gain, Orca test level/gain. A flat GSM8K curve counts as PLATEAUS-BELOW only if the
    Orca test curve (the added data's own distribution) is flat and low too; otherwise SHIFT-LIMITED."""
    if level >= 0.80:
        return 'CLOSES'
    if gain >= 0.05:
        return 'STILL-CLIMBING'
    if level <= 0.50 and gain <= 0.02:
        return 'PLATEAUS-BELOW' if olevel <= 0.50 and ogain <= 0.02 else 'SHIFT-LIMITED'
    return 'PARTIAL'


def b2x(X2, m2, procs=4, say=print, ns=GSM_NS, n_ext=(10000, 20000)):
    its = m2['items']; states = m2['states']; si = m2['slots'].index('last'); q = A5.Q
    src = np.array([i['source'] for i in its]); split = np.array([i['split'] for i in its]); rank = np.array([i['rank'] for i in its])
    gi = np.array([i.get('gi', i['idx']) for i in its])
    gtr = np.where((src == 'gsm') & (split == 'train'))[0]
    gtr = gtr[np.argsort(gi[gtr])]; gtr = gtr[np.random.default_rng(205).permutation(len(gtr))]   # item order, then seed 205
    otr = np.where((src == 'orca') & (split == 'train'))[0]; otr = otr[np.argsort(rank[otr])]
    order = np.concatenate([gtr, otr]); nmax = len(order); ng = len(gtr)
    NS = sorted(set([n for n in ns if n <= ng] + [ng] + [n for n in n_ext if n < nmax] + [nmax // 2, nmax]))
    te_all = np.where(split == 'test')[0]
    qi_states = [states.index(s) for s in q]
    arrs = dict(X=np.asarray(X2[:, si][:, qi_states]), X0=np.asarray(X2[:, 0, states.index('l7')]),
                y=np.array([i['digit'] for i in its]), g=np.array([i['idx'] for i in its]), src=src, order=order,
                te_all=te_all, te_gsm=np.where((split == 'test') & (src == 'gsm'))[0],
                twin=np.array([bool(i.get('twin', False)) for i in its]))
    say('--- B2x many-question curve: %d GSM8K + %d Orca training questions; test %d GSM8K, %d Orca (chance 0.10) ---' % (
        ng, nmax - ng, (src[te_all] == 'gsm').sum(), (src[te_all] == 'orca').sum()))
    for k, s in enumerate(q):
        Xt = np.asarray(arrs['X'][order, k], dtype=np.float64); Xt -= Xt.mean(0)
        lam = np.linalg.svd(Xt, compute_uv=False) ** 2; c = np.cumsum(lam) / lam.sum()
        pr = lam.sum() ** 2 / (lam ** 2).sum(); n95 = int(np.searchsorted(c, 0.95) + 1)
        say('%-5s effective rank: participation ratio %.1f, components to 95%% variance %d -> N_max = %.1f x n95' % (s, pr, n95, nmax / n95))
    jobs = [(k, n, p) for k in range(len(q)) for n in NS for p in ('pca', 'full')]
    with Pool(procs, initializer=init, initargs=(arrs,)) as pool:
        res = dict(pool.map(curve_job, sorted(jobs, key=lambda j: -j[1])))
        mres = dict(pool.map(mlp_job, range(len(q))))
        gate = dict(pool.map(gate_job, [0]))[0]
    for test in ('gsm', 'orca', 'gsm_clean', 'orca_clean'):
        say('[%s test] N ' % test + ' '.join('%8d' % n for n in NS))
        for k, s in enumerate(q):
            for p in ('pca', 'full'):
                say('%-5s %-5s %s' % (s, p, ' '.join('%8.3f' % res[(k, n, p)][0][test] for n in NS)))
        if test in mres[0]:
            say('MLP at N_max: %s' % ' '.join('%s %.3f' % (s, mres[k][test]) for k, s in enumerate(q)))
    say('chosen C at N_max: %s' % ' '.join('%s/%s %s' % (s, p, res[(k, nmax, p)][1]) for k, s in enumerate(q) for p in ('pca', 'full')))
    acc = lambda k, n, p, t='gsm': res[(k, n, p)][0][t]
    level = max(acc(k, nmax, p) for k in range(len(q)) for p in ('pca', 'full'))
    gain = max(acc(k, nmax, p) - acc(k, nmax // 2, p) for k in range(len(q)) for p in ('pca', 'full'))
    olevel = max(acc(k, nmax, p, 'orca') for k in range(len(q)) for p in ('pca', 'full'))
    ogain = max(acc(k, nmax, p, 'orca') - acc(k, nmax // 2, p, 'orca') for k in range(len(q)) for p in ('pca', 'full'))
    lab = 'UNSCORABLE' if gate < 0.90 else label_ext(level, gain, olevel, ogain)
    lg = max(acc(k, ng, p) for k in range(len(q)) for p in ('pca', 'full'))
    hg = sorted(n for n in NS if n <= ng)[-2] if len([n for n in NS if n <= ng]) > 1 else ng
    gg = max(acc(k, ng, p) - acc(k, ng // 2 if ng // 2 in NS else hg, p) for k in range(len(q)) for p in ('pca', 'full'))
    v205lab = ('CLIMBS' if lg >= 0.50 or gg >= 0.05 else 'STAYS-SWAMPED' if lg <= 0.30 and gg <= 0.02 else 'PARTIAL')
    say("gate: PCA probe on the digit's own token (+0, l7) at N_max: %.3f (>= 0.90)" % gate)
    say('GSM8K-only part (N = %d): level %.3f, gain %+.3f -> %s (v205 rule)' % (ng, lg, gg, v205lab))
    say('B2x: GSM8K test level %.3f, gain over the last doubling %+.3f | Orca test level %.3f, gain %+.3f -> %s' % (
        level, gain, olevel, ogain, lab))
    # Amendment 1 (descriptive, no label): the MEAN gain over the Q x probe combinations from N_max/2 to N_max, with a
    # 95% bootstrap CI over test questions (paired: the same resampled questions for both sizes), per test set.
    srcte = src[te_all]
    for t in ('gsm', 'orca'):
        m = srcte == t
        D = np.mean([res[(k, nmax, p)][0]['_hit'][m].astype(float) - res[(k, nmax // 2, p)][0]['_hit'][m].astype(float)
                     for k in range(len(q)) for p in ('pca', 'full')], axis=0)
        rng = np.random.default_rng(2132); bs = [D[rng.integers(0, len(D), len(D))].mean() for _ in range(10000)]
        say('mean gain over %d Q x probe combinations, N_max/2 -> N_max, %s test: %+.4f [%+.4f, %+.4f] (descriptive)' % (
            2 * len(q), t, D.mean(), *np.percentile(bs, [2.5, 97.5])))
    return lab


def b3x(X3, m3, say=print):
    its = m3['items']; out = {}
    for s in ('gsm', 'orca'):
        idx = [k for k, i in enumerate(its) if i.get('source') == s]
        mm = dict(m3, items=[its[k] for k in idx]); say('[%s test questions]' % s)
        out[s] = A5.b3(np.asarray(X3[idx]), mm, say=say)
    return out


def main(root, model, parts):
    q = set_q(model)
    if 'A' in parts:
        X, meta = A5.load(root, 'v213_v181')
        import glob
        XL = np.load(sorted(glob.glob(os.path.join(root, '**', 'v213_v181_last_all.npy'), recursive=True))[0], mmap_mode='r')
        print('A:', A2.analyse(X, XL, rename(meta, model)))
    if 'B1' in parts:
        X1, m1 = A5.load(root, 'v213_b1'); X0, m0 = A5.load(root, 'v213_v181')
        print('B1:', A5.b1(X1, rename(m1, model), X0, rename(m0, model)))
    if 'B2' in parts:
        import glob, json
        fs = sorted(glob.glob(os.path.join(root, '**', 'v213_b2_s*_states.npy'), recursive=True))
        Xs, items = [], []
        for f in fs:
            Xs.append(np.load(f, mmap_mode='r')); mt = json.load(open(f.replace('_states.npy', '_meta.json'))); items += mt['items']
        m2 = rename(dict(mt, items=items), model); print('B2x shards:', len(fs))
        print('B2x:', b2x(np.concatenate([np.asarray(x) for x in Xs]), m2))
    if 'B3' in parts:
        X3, m3 = A5.load(root, 'v213_b3'); print('B3:', b3x(X3, rename(m3, model)))


def selftest():
    """Synthetic many-question worlds through b2x: readable -> CLOSES; data-hungry -> STILL-CLIMBING; swamped ->
    PLATEAUS-BELOW; shift (digit unreadable in the GSM8K-like items, readable but data-hungry in the Orca-like items,
    with its own code) -> SHIFT-LIMITED. Small sizes (the grid is scaled down) so it runs in minutes."""
    rng = np.random.default_rng(0); d = 1024; code = rng.standard_normal((10, d)); code2 = rng.standard_normal((10, d)); ok = True
    A5.Q = ['l7', 'l14', 'l21', 'l28']
    for world, amp, want in (('readable', 2.0, 'CLOSES'), ('hungry', 0.20, 'STILL-CLIMBING'), ('swamped', 0.01, 'PLATEAUS-BELOW'),
                             ('shift', None, 'SHIFT-LIMITED')):
        ng, no, nt = 600, 3000, 1000; n = ng + no + 2 * nt
        y = rng.integers(0, 10, n); ctx = rng.standard_normal((n, d)) * 3
        orc = np.array([ng <= k < ng + no or k >= ng + no + nt for k in range(n)])
        sig = (amp * code[y] if amp is not None else
               np.where(orc[:, None], 0.20 * code2[y], 0.01 * code[y]))
        X = np.zeros((n, 3, 6, d), np.float32)
        for st in range(6):
            X[:, 2, st] = ctx + sig; X[:, 0, st] = ctx * 0.1 + 3 * code[y]
        its = []
        for k in range(n):
            s, sp, r = (('gsm', 'train', k) if k < ng else ('orca', 'train', k - ng) if k < ng + no else
                        ('gsm', 'test', k) if k < ng + no + nt else ('orca', 'test', k))
            its.append(dict(digit=int(y[k]), idx=k, source=s, split=sp, rank=r))
        m = dict(items=its, states=['h0', 'l4', 'l7', 'l14', 'l21', 'l28'], slots=['+0', 'qmark', 'last'])
        lab = b2x(X, m, say=lambda *a: None, ns=(60, 125, 250, 500), n_ext=(1000, 2000))
        print('selftest %-9s -> %-15s %s' % (world, lab, 'OK' if lab == want else 'FAIL (want %s)' % want), flush=True)
        ok &= lab == want
    return ok


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        sys.exit(0 if selftest() else 1)
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or ['A', 'B1', 'B2', 'B3'])
