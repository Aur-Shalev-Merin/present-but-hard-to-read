"""v188 normalization check on the mid/end alternation (post hoc; reading rule in
docs/prereg_v188_ouro_intermediate_confirmation.md, section 7, "Normalization check", fixed before running).

endT states are after Ouro's RMSNorm (fed into the next loop); midT states are the raw residual after layer 12. At P1
(word problems), re-probe the mid states after dividing each item's state by its own RMS (the per-item part of RMSNorm;
the learned per-dimension gain is removed anyway by the probe's per-dimension standardization). Labels (mod 5): a+b,
a-c, b-c, answer, answer+2c, a+c, copy of b. Alternation D = mean over loops of [acc(endT) - acc(midT)] per item.

  python v188_rms_check.py <dir with states/meta> <188|186>     (Kaggle: probes + statistics)
  python v188_rms_check.py --stats <npz> [<npz> ...]            (statistics and the reading from saved items)
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

LABELS = {'a+b': (1, 1, 0), 'a-c': (1, 0, 4), 'b-c': (0, 1, 4), 'ans': (1, 1, 4), 'ans+2c': (1, 1, 1), 'a+c': (1, 0, 1),
          'b': (0, 1, 0)}
Q = (1.25, 98.75)
G = {}


def cluster_ci(cl, d, n=10000, seed=188):
    """Same draws and result as v186_analyze.cluster_ci."""
    rng = np.random.default_rng(seed); u = np.unique(cl)
    S = np.array([d[cl == c].sum() for c in u]); N = np.array([(cl == c).sum() for c in u])
    bs = []
    for _ in range(n):
        p = rng.choice(len(u), len(u)); bs.append(S[p].sum() / N[p].sum())
    return float(d.mean()), float(np.percentile(bs, Q[0])), float(np.percentile(bs, Q[1]))


def init(X, its):
    G['X'], G['its'] = X, its


def job(args):
    import v186_analyze as A
    import v178_analyze as V
    import v188_shown_digit_check as S
    st, lab, ver = args; f = LABELS[lab]
    idx = np.array([k for k, i in enumerate(G['its']) if i['set'] == 'main']); its = [G['its'][k] for k in idx]
    fam = np.array([i['fam'] for i in its]); tid = np.array([i['tid'] for i in its])
    y = np.array([(f[0] * i['a'] + f[1] * i['b'] + f[2] * i['c']) % 5 for i in its])
    if ver == 'ceil':
        Z = S.feats(its, 'base')
    else:
        Z = np.asarray(G['X'][idx, 1, st], dtype=np.float32)
        if ver == 'rms':
            Z = Z / np.sqrt((Z.astype(np.float64) ** 2).mean(1, keepdims=True)).astype(np.float32)
    return args, (V.probe_predict(A.demean(Z, fam), y, tid, V.folds(tid)) == y).astype(np.int8)


def stats(path):
    z = np.load(path); v = int(z['study']); fam = z['fam']; labs = list(z['labels'])
    states = list(z['states']); mids = [k for k, s in enumerate(states) if s.startswith('mid')]
    ends = [k for k, s in enumerate(states) if s.startswith('end')]
    out = {}
    for li, lab in enumerate(labs):
        ceil = z['ceil'][li].astype(float)
        for ver in ('raw', 'rms'):
            acc = {s: z[ver][li, k].astype(float) if k in mids else z['raw'][li, k].astype(float)
                   for k, s in enumerate(states)}
            d = np.mean([acc[states[e]] - acc[states[m]] for m, e in zip(mids, ends)], 0)
            ci = cluster_ci(fam, d); out[(v, lab, ver)] = ci
            print('v%d P1 %-6s mids=%s | excess by state: %s | alternation D %+.3f [%+.3f, %+.3f]' % (
                v, lab, ver, ' '.join('%s %+.3f' % (s, (acc[s] - ceil).mean()) for s in states), ci[0], ci[1], ci[2]))
    sys.stdout.flush()
    return out


def reading(outs):
    o = {k: v for d in outs for k, v in d.items()}
    vs = sorted({k[0] for k in o})
    raw = {v: o[(v, 'a+b', 'raw')] for v in vs}; rms = {v: o[(v, 'a+b', 'rms')] for v in vs}
    art = all(rms[v][0] < 0.5 * raw[v][0] for v in vs) or all(rms[v][1] <= 0 for v in vs)
    real = all(rms[v][0] >= 0.75 * raw[v][0] and rms[v][1] > 0 for v in vs)
    lab = 'ARTIFACT' if art else 'REAL' if real else 'PARTIAL'
    print('READING (rule fixed before running): a+b alternation %s -> %s' % (
        '; '.join('v%d D_raw %+.3f, D_rms %+.3f [%+.3f, %+.3f] (ratio %.2f)' % (
            v, raw[v][0], rms[v][0], rms[v][1], rms[v][2], rms[v][0] / raw[v][0] if raw[v][0] else float('nan')) for v in vs),
        lab))
    return lab


def run(root, v):
    from multiprocessing import Pool
    import v188_shown_digit_check as S
    X, meta = S.load(root, v)
    states = meta['states']; its = meta['items']; main = [i for i in its if i['set'] == 'main']
    mids = [k for k, s in enumerate(states) if s.startswith('mid')]; ends = [k for k, s in enumerate(states) if s.startswith('end')]
    jobs = [(k, lab, ver) for lab in LABELS for k in mids for ver in ('raw', 'rms')] + \
           [(k, lab, 'raw') for lab in LABELS for k in ends] + [(0, lab, 'ceil') for lab in LABELS]
    with Pool(4, initializer=init, initargs=(X, its)) as pool:
        res = dict(pool.map(job, jobs))
    n = len(main); L = list(LABELS)
    raw = np.zeros((len(L), len(states), n), np.int8); rms = np.zeros_like(raw)
    for li, lab in enumerate(L):
        for k in range(len(states)):
            raw[li, k] = res[(k, lab, 'raw')]
            rms[li, k] = res[(k, lab, 'rms')] if k in mids else res[(k, lab, 'raw')]
    path = 'v188_rms_items_%d.npz' % v
    np.savez_compressed(path, study=v, labels=np.array(L), states=np.array(states), fam=np.array([i['fam'] for i in main]),
                        tid=np.array([i['tid'] for i in main]), raw=raw, rms=rms,
                        ceil=np.stack([res[(0, lab, 'ceil')] for lab in L]))
    print('v%d: %d items; saved %s' % (v, n, path), flush=True)
    stats(path)


if __name__ == '__main__':
    if sys.argv[1] == '--stats':
        reading([stats(p) for p in sys.argv[2:]])
    else:
        run(sys.argv[1], int(sys.argv[2]))
