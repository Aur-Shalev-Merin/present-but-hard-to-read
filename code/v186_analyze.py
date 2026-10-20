"""v186 analysis (written and selftested BEFORE any main-set state exists; docs/prereg_v186_ouro_unwritten_intermediate.md).

  python v186_analyze.py <dir containing v186_states.npy, v186_meta.json (searched recursively)>
  python v186_analyze.py --selftest

Main set: two-step problems, answer = a + b - c, intermediate m = a + b never written. Families fix template, name and
tens digits; the units digits vary. All probes: family-mean subtracted states (or features), v178 probe_predict
(standardise -> PCA 256 -> multinomial logistic, C in {const, 0.01, 0.1, 1} by inner CV), 5-fold CV grouped by
template. Labels (units digits): T1 m=(a+b)%10 [test]; T2 ans [control]; T3 amb=(a-b)%10, apc=(a+c)%10, rnd=random
function of (a,b) [unneeded combinations]; T4 a, b, c [copying, positive control].
Copy ceiling: the same probe on one-hot codes of (a_u, b_u, c_u) (what separate copies allow linearly).

PRIMARY, at two positions, state end4: P0 = last prompt token; P1 = the answer token just before the units digit
(Ouro writes the leading digit first, so m's units digit may only be needed at P1). Excess of a label =
acc(label, state) - acc(label, its own copy ceiling), per item (the random function rnd has main effects, so its
ceiling is far above chance). Per position:
  D_ceil = excess(T1);  D_T3max = excess(T1) - max over T3 of excess(T3);  D_T3mean = excess(T1) - mean T3 excess;
  paired cluster bootstrap over families (10,000, seed 186), Bonferroni over the 2 positions -> 97.5% CIs.
  position COMPUTED: D_ceil >= 0.15 and D_T3max >= 0.15, both CI lower bounds > 0
  position NULL:     the D_T3mean CI lies entirely within [-0.05, +0.05] (equivalence)
  COMPUTED      if either position is COMPUTED (report which)
  NOT-DETECTED  if both positions are NULL
  UNCLEAR       otherwise
Gates (any failure -> UNSCORABLE): G1 main-set answer-only accuracy after loop 4 >= 0.30;
  G2 T4 a, b, c each >= 0.75 at P0 in its best state (pilot: b peaked at 0.81, so 0.90 at end4 was not attainable);
  G3 bare set: the sum's units digit at P1 (just before Ouro writes it) end4 >= 0.90.
Secondary (reported, not labels): T1 at P0 for every state (does it rise after loop 1?); raw accuracies at P1 end4;
  T1 at P0 end4 on items Ouro gets right vs wrong after loop 4; bare set: the sum's units digit at P0 vs its ceiling.
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V

T1, T2, T3, T4 = 'm', 'ans', ('amb', 'apc', 'rnd'), ('a', 'b', 'c')
BAR, NULL = 0.15, 0.05
G = {}


def demean(X, fam):
    X = X.astype(np.float32).copy()
    for f in np.unique(fam):
        X[fam == f] -= X[fam == f].mean(0)
    return X


def onehot(its, keys=('a', 'b', 'c')):
    Z = np.zeros((len(its), 10 * len(keys)), np.float32)
    for r, i in enumerate(its):
        for j, k in enumerate(keys):
            Z[r, 10 * j + i['labels'][k]] = 1
    return Z


def init(X, meta):
    G['X'], G['its'] = X, meta['items']


def job(args):
    split, pos, st, lab, feat = args
    idx = np.array([k for k, i in enumerate(G['its']) if i['set'] == split]); its = [G['its'][k] for k in idx]
    fam = np.array([i['fam'] for i in its]); tid = np.array([i['tid'] for i in its])
    y = np.array([i['labels'][lab] for i in its])
    Z = onehot(its) if feat == 'ceil' else np.asarray(G['X'][idx, pos, st], dtype=np.float32)
    return args, (V.probe_predict(demean(Z, fam), y, tid, V.folds(tid)) == y).astype(np.int8)


def cluster_ci(fam, d, n=10000, seed=186, q=(2.5, 97.5)):
    rng = np.random.default_rng(seed); u = np.unique(fam); by = {f: d[fam == f] for f in u}
    bs = []
    for _ in range(n):
        pick = rng.choice(u, len(u))
        bs.append(np.concatenate([by[f] for f in pick]).mean())
    return float(d.mean()), float(np.percentile(bs, q[0])), float(np.percentile(bs, q[1]))


def analyse(X, meta, procs=4, secondary=True, verbose=True):
    out = []; say = out.append; its = meta['items']; E = meta['states'].index('end4')
    main = [i for i in its if i['set'] == 'main']; fam = np.array([i['fam'] for i in main])
    acc = np.mean([i['correct'] for i in main], 0)
    say('main items %d, families %d, templates %d; answer-only accuracy by loop %s' % (
        len(main), len(set(fam)), len({i['tid'] for i in main}), np.round(acc, 3).tolist()))
    jobs = [('main', p, E, lab, 'state') for p in (0, 1) for lab in (T1, T2) + T3] + [('main', 0, E, T1, 'ceil')]
    jobs += [('main', 0, st, lab, 'state') for lab in T4 for st in range(len(meta['states']))]
    jobs += [('main', 0, E, lab, 'ceil') for lab in (T2,) + T3]
    has_bare = any(i['set'] == 'bare' for i in its)
    if has_bare:
        jobs += [('bare', 1, E, 'ans', 'state')]
    if secondary:
        jobs += [('main', 0, st, T1, 'state') for st in range(len(meta['states'])) if st != E]
        if has_bare:
            jobs += [('bare', 0, E, 'ans', 'state'), ('bare', 0, E, 'ans', 'ceil')]
    if procs > 1:
        with Pool(procs, initializer=init, initargs=(X, meta)) as pool:
            res = dict(pool.map(job, jobs))
    else:
        init(X, meta); res = dict(job(j) for j in jobs)
    a = lambda *k: res[k].mean()
    g1 = acc[3] >= 0.30
    bestT4 = {l: max(a('main', 0, s, l, 'state') for s in range(len(meta['states']))) for l in T4}
    g2 = all(v >= 0.75 for v in bestT4.values())
    g3 = a('bare', 1, E, 'ans', 'state') >= 0.90 if has_bare else False
    say('G1 accuracy after loop 4 %.3f (>= 0.30: %s) | G2 copies at P0, best state %s (each >= 0.75: %s); end4 %s | '
        'G3 bare sum digit at P1 end4 %s (>= 0.90: %s)' % (
            acc[3], g1, ' '.join('%s %.3f' % kv for kv in bestT4.items()), g2,
            ' '.join('%s %.3f' % (l, a('main', 0, E, l, 'state')) for l in T4),
            '%.3f' % a('bare', 1, E, 'ans', 'state') if has_bare else 'n/a', g3))
    say('P0 end4: T1 m %.3f | ceiling(m) %.3f | T2 ans %.3f (ceiling %.3f) | T3 %s' % (
        a('main', 0, E, T1, 'state'), a('main', 0, E, T1, 'ceil'), a('main', 0, E, T2, 'state'), a('main', 0, E, T2, 'ceil'),
        ' '.join('%s %.3f (ceiling %.3f)' % (l, a('main', 0, E, l, 'state'), a('main', 0, E, l, 'ceil')) for l in T3)))
    # excess of each label over its OWN copy ceiling (per item; the random function has main effects, so its ceiling is
    # far above chance). Two positions (P0, P1) -> Bonferroni: 97.5% CIs.
    Q = (1.25, 98.75); verdict = {}
    for p, pname in ((0, 'P0'), (1, 'P1')):
        ex = {l: res[('main', p, E, l, 'state')].astype(float) - res[('main', 0, E, l, 'ceil')] for l in (T1,) + T3}
        # rnd becomes partly readable whenever m is present (any function of (a, b) is easier given a, b and a+b), so
        # the 0.15 gap is required only against the XOR-like controls amb and apc; rnd must merely be beaten.
        best = max(('amb', 'apc'), key=lambda l: ex[l].mean())
        dc = cluster_ci(fam, ex[T1], q=Q); dt = cluster_ci(fam, ex[T1] - ex[best], q=Q)
        dr = cluster_ci(fam, ex[T1] - ex['rnd'], q=Q)
        dm = cluster_ci(fam, ex[T1] - np.mean([ex[l] for l in T3], 0), q=Q)
        say('%s end4 excess over own ceiling: %s' % (pname, ' '.join('%s %+.3f' % (l, ex[l].mean()) for l in (T1,) + T3)))
        say('%s D_ceil %+.3f [%+.3f, %+.3f] | D_XOR (vs %s) %+.3f [%+.3f, %+.3f] | D_rnd %+.3f [%+.3f, %+.3f] | '
            'D_T3mean %+.3f [%+.3f, %+.3f]' % ((pname,) + dc + (best,) + dt + dr + dm))
        verdict[pname] = ('COMPUTED' if dc[0] >= BAR and dc[1] > 0 and dt[0] >= BAR and dt[1] > 0 and dr[1] > 0 else
                          'NULL' if -NULL <= dm[1] and dm[2] <= NULL else 'MIXED')    # equivalence: whole CI within +-0.05
    t1 = res[('main', 0, E, T1, 'state')].astype(float)
    say('per position: %s' % verdict)
    if not (g1 and g2 and g3):
        label = 'UNSCORABLE'
    elif 'COMPUTED' in verdict.values():
        label = 'COMPUTED (%s)' % ', '.join(k for k, v in verdict.items() if v == 'COMPUTED')
    elif all(v == 'NULL' for v in verdict.values()):
        label = 'NOT-DETECTED'
    else:
        label = 'UNCLEAR'
    say('PRIMARY -> %s' % label)
    if secondary:
        say('--- secondary ---')
        say('T1 at P0 by state: %s' % ' '.join('%s %.3f' % (s, a('main', 0, k, T1, 'state')) for k, s in enumerate(meta['states'])))
        say('P1 end4 raw: %s' % ' '.join('%s %.3f' % (l, a('main', 1, E, l, 'state')) for l in (T1, T2) + T3))
        c4 = np.array([i['correct'][3] for i in main])
        say('T1 at P0 end4 on correct@4 (n=%d) %.3f | wrong@4 (n=%d) %.3f' % (c4.sum(), t1[c4].mean() if c4.any() else np.nan,
                                                                          (~c4).sum(), t1[~c4].mean() if (~c4).any() else np.nan))
        if has_bare:
            say('bare: sum units digit at P0 end4 %.3f vs ceiling %.3f' % (a('bare', 0, E, 'ans', 'state'), a('bare', 0, E, 'ans', 'ceil')))
    if verbose:
        print('\n'.join(out), flush=True)
    return label


def selftest():
    """Synthetic worlds on the real item structure; family offsets dominate, as in v181-v185."""
    its = json.load(open('v186_items.json'))['items']; its = [i for i in its if i['set'] in ('main', 'bare')]
    rng = np.random.default_rng(0); d = 64; N = len(its)
    W = {k: rng.normal(size=(10, d)) for k in ('a', 'b', 'c', 'm', 'ans')}
    U, Vv = rng.normal(size=(10, d)), rng.normal(size=(10, d))
    fam = np.array([i['fam'] for i in its]); off = 20 * rng.normal(size=(fam.max() + 1, d))

    def world(kind):
        X = np.zeros((N, 2, 8, d), np.float32)
        for r, i in enumerate(its):
            L = i['labels']; base = off[i['fam']] + 0.05 * rng.normal(size=d)
            v = base.copy()
            if kind != 'broken':
                v = v + 1.0 * (W['a'][L['a']] + W['b'][L['b']] + W['c'][L['c']])
            if kind == 'compute' or (kind.startswith('compute_on') and rng.random() < float(kind.split('_on')[1])):
                v = v + 1.0 * W['m'][L['m']]           # compute_onF: m is computed on a fraction F of items only
            if kind == 'answer':
                v = v + 1.0 * W['ans'][L['ans']]
            if kind == 'mixing':                       # generic nonlinear interaction of a and b, not the sum
                v = v + 1.0 * np.tanh(U[L['a']] * Vv[L['b']] * 2)
            p1 = v + (3.0 * W['ans'][L['ans']] if i['set'] == 'bare' or kind != 'broken' else 0)
            X[r, 0, :] = v; X[r, 1, :] = p1
        return X

    meta = dict(states=['mid1', 'end1', 'mid2', 'end2', 'mid3', 'end3', 'mid4', 'end4'],
                items=[dict(i, correct=[True] * 4) for i in its])
    want = {'compute': {'COMPUTED (P0, P1)'}, 'compute_on0.4': {'COMPUTED (P0, P1)'},
            'compute_on0.15': None, 'copy': {'NOT-DETECTED'}, 'answer': {'NOT-DETECTED'},
            'mixing': {'NOT-DETECTED', 'UNCLEAR'}, 'broken': {'UNSCORABLE'}}
    ok = True
    for k, w in want.items():
        print('== world %s' % k, flush=True)
        lab = analyse(world(k), meta, secondary=False, verbose=True)
        print('selftest %-14s -> %-17s %s' % (k, lab, 'recorded (sensitivity)' if w is None else
                                              'OK' if lab in w else 'FAIL (want %s)' % sorted(w)), flush=True)
        ok &= w is None or lab in w
    return ok


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        sys.exit(0 if selftest() else 1)
    f = sorted(glob.glob(os.path.join(sys.argv[1], '**', 'v186_states.npy'), recursive=True))[0]
    meta = json.load(open(os.path.join(os.path.dirname(f), 'v186_meta.json')))
    assert {i['set'] for i in meta['items']} >= {'main', 'bare'}, 'main run needs the main and bare sets'
    analyse(np.load(f, mmap_mode='r'), meta)
