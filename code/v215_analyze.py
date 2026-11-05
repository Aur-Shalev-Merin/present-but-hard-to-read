"""v215 analysis (CPU; docs/prereg_v215_kudo_retest.md).

Part A (fam states): v178 probe on family-demeaned states, outer 5 folds grouped by family, separately per version.
  Labels mod 5: S = (d1 + d2) mod 5, M = (d1 - d2) mod 5, needed N = S on plus / M on minus; copy ceiling = the same
  probe on one-hot (d1, d2). Per item excess(L) = mean over the band {12, 16, 20, 24} of hit(L, state) - hit(L, ceil).
  ex_N = (mean excess(N) on plus + on minus) / 2;  I = (D+ + D-)/2, D+ = ex(S) - ex(M) on plus, D- = ex(M) - ex(S) on
  minus. Cluster bootstrap over families, 10,000, seed 215, 97.5% CI, per position (eq, last).
  UNSCORABLE (G1 correct < 0.90 in a version, or G2 copies d1 / d2 best layer < 0.50 at the position) | COMBINED
  ex_N >= 0.05, CI lower > 0 | NOT-COMBINED CI upper < 0.02 | UNCLEAR.  Operation swap: FOLLOWS (I >= 0.05, CI lower
  > 0, D+ > 0, D- > 0) | DOES-NOT-FOLLOW (CI upper < 0.02) | UNCLEAR.
Part B plan (short-chain arm): for COMBINED positions (or all with V215_PLAN_ALL=1): per half of the families and band
  layer, the 9-dim basis of family-centred class means of B (0-9) fitted on the OTHER half; a random 9-dim basis per
  layer (seed 2152); 1,000 pairs (x correct, x' same family and version, same d1, different d2; seed 2151).
Short-chain arm G1: the chain's b step (predB = B) instead of the exact chain (exact chain reported).
Part C (rep states): standard probe (standardised states, logistic C = 0.01) for b's and a's values, train 10,000 /
  test 2,000 (no shared b equation), each position and layer; Acc_preCoT = max.

  python v215_analyze.py <root> [A] [S] [C]  -> A: main format (v215_partA.json); S: the short-chain arm (fams states,
                                               v215_partA_short.json, and the Part B plan from it); C: Part C
  python v215_analyze.py --selftest
"""
import glob, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V

LAYERS = list(range(0, 29, 4)); POS = ['eq', 'semi', 'last']; APOS = ['eq', 'last']
BAND = [LAYERS.index(l) for l in (12, 16, 20, 24)]
Q = (1.25, 98.75); OUT = os.environ.get('OUT_DIR', '.')
G = {}


def demean(X, fam):
    X = np.array(X, dtype=np.float32)
    for f in np.unique(fam):
        X[fam == f] -= X[fam == f].mean(0)
    return X


def labels(it):
    return dict(S=(it['d1'] + it['d2']) % 5, M=(it['d1'] - it['d2']) % 5, d1=it['d1'], d2=it['d2'], A5=it['A'] % 5,
                B=it['B'])


def init(d):
    G.update(d)


def job(args):
    ver, pos, li, lab, feat = args
    idx = G['idx_' + ver]; its = [G['its'][k] for k in idx]; fam = np.array([i['fam'] for i in its])
    y = np.array([labels(i)[lab] for i in its])
    if feat == 'ceil':
        Z = np.zeros((len(its), 20), np.float32); Z[np.arange(len(its)), [i['d1'] for i in its]] = 1
        Z[np.arange(len(its)), [10 + i['d2'] for i in its]] = 1
    else:
        Z = np.asarray(G['X'][idx, pos, li], np.float32)
    return args, (V.probe_predict(demean(Z, fam), y, fam, V.folds(fam)) == y).astype(np.int8)


def cluster_ci(fam_p, vp, fam_m, vm, n=10000, seed=215):
    """(mean vp + mean vm)/2 with a cluster bootstrap over families (both versions resampled together)."""
    u = np.unique(np.concatenate([fam_p, fam_m])); ix = {f: j for j, f in enumerate(u)}
    S = np.zeros((2, len(u))); N = np.zeros((2, len(u)))
    for r, (fa, v) in enumerate(((fam_p, vp), (fam_m, vm))):
        np.add.at(S[r], [ix[f] for f in fa], v); np.add.at(N[r], [ix[f] for f in fa], 1)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        q = rng.integers(0, len(u), len(u)); bs.append((S[0, q].sum() / N[0, q].sum() + S[1, q].sum() / N[1, q].sum()) / 2)
    return (vp.mean() + vm.mean()) / 2, *np.percentile(bs, Q)


def part_a(X, its, meta, procs=4, say=print, positions=APOS, plan=True, plan_all=False, outdir=OUT, lite=False, tag=''):
    """lite (mixing null): band layers and S, M, d1, d2 only."""
    lis = BAND if lite else list(range(len(LAYERS))); labs = ('S', 'M', 'd1', 'd2') if lite else ('S', 'M', 'd1', 'd2', 'A5')
    idx = {v: np.array([k for k, i in enumerate(its) if i['ver'] == v]) for v in ('plus', 'minus')}
    corr = {v: np.mean([meta[k]['correct'] for k in idx[v]]) for v in idx}
    say('Part A: %d families, plus %d / minus %d items | G1 correct (exact chain): plus %.3f, minus %.3f' % (
        len(set(i['fam'] for i in its)), len(idx['plus']), len(idx['minus']), corr['plus'], corr['minus']))
    pis = [POS.index(p) for p in positions]
    jobs = [(v, p, li, lab, 'state') for v in idx for p in pis for li in lis for lab in labs]
    jobs += [(v, 0, 0, lab, 'ceil') for v in idx for lab in labs if lab in ('S', 'M', 'A5')]
    d = dict(X=X, its=its, idx_plus=idx['plus'], idx_minus=idx['minus'])
    if procs > 1:
        with Pool(procs, initializer=init, initargs=(d,)) as pool:
            res = dict(pool.imap_unordered(job, jobs, chunksize=1))
    else:
        init(d); res = dict(job(j) for j in jobs)
    fam = {v: np.array([its[k]['fam'] for k in idx[v]]) for v in idx}
    elabs = [l for l in labs if l in ('S', 'M', 'A5')]
    ceil = {(v, lab): res[(v, 0, 0, lab, 'ceil')].mean() for v in idx for lab in elabs}
    say('copy ceilings (one-hot d1, d2): ' + ' | '.join('%s %s %.3f' % (v, lab, c) for (v, lab), c in ceil.items()))
    out = {}
    for p in pis:
        name = POS[p]
        cop = {(v, lab): max(res[(v, p, li, lab, 'state')].mean() for li in lis) for v in idx for lab in ('d1', 'd2')}
        g1 = all(c >= 0.90 for c in corr.values()); g2 = all(c >= 0.50 for c in cop.values())
        ex = {(v, lab): np.mean([res[(v, p, li, lab, 'state')] for li in BAND], 0) - res[(v, 0, 0, lab, 'ceil')]
              for v in idx for lab in elabs}
        need = {'plus': 'S', 'minus': 'M'}
        exN = cluster_ci(fam['plus'], ex[('plus', 'S')], fam['minus'], ex[('minus', 'M')])
        dp = ex[('plus', 'S')] - ex[('plus', 'M')]; dm = ex[('minus', 'M')] - ex[('minus', 'S')]
        I = cluster_ci(fam['plus'], dp, fam['minus'], dm, seed=2150)
        lab = ('UNSCORABLE' if not (g1 and g2) else 'COMBINED' if exN[0] >= 0.05 and exN[1] > 0 else
               'NOT-COMBINED' if exN[2] < 0.02 else 'UNCLEAR')
        swap = ('UNSCORABLE' if not (g1 and g2) else
                'FOLLOWS' if I[0] >= 0.05 and I[1] > 0 and dp.mean() > 0 and dm.mean() > 0 else
                'DOES-NOT-FOLLOW' if I[2] < 0.02 else 'UNCLEAR')
        say('--- position %s ---' % name)
        say('G2 copies (family-demeaned, best layer): %s (each >= 0.50: %s)' % (
            ' '.join('%s/%s %.3f' % (v, l, c) for (v, l), c in cop.items()), g2))
        for v in (() if lite else idx):
            say('  %-5s by layer  S %s | M %s | A5 %s' % (v, *(' '.join('%.2f' % res[(v, p, li, l, 'state')].mean()
                                                                    for li in range(len(LAYERS))) for l in ('S', 'M', 'A5'))))
        say('  excess (band): plus S %+.3f M %+.3f -> D+ %+.3f | minus M %+.3f S %+.3f -> D- %+.3f' % (
            ex[('plus', 'S')].mean(), ex[('plus', 'M')].mean(), dp.mean(), ex[('minus', 'M')].mean(),
            ex[('minus', 'S')].mean(), dm.mean()) + ('' if lite else ' | A5 plus %+.3f minus %+.3f' % (
            ex[('plus', 'A5')].mean(), ex[('minus', 'A5')].mean())))
        say('  ex_N %+.4f [%+.4f, %+.4f] -> %s | I %+.4f [%+.4f, %+.4f] -> %s' % (*exN, lab, *I, swap))
        out[name] = dict(label=lab, swap=swap, exN=list(exN), I=list(I), Dp=float(dp.mean()), Dm=float(dm.mean()),
                         g1=bool(g1), g2=bool(g2))
    if not lite:
        json.dump(out, open(os.path.join(outdir, 'v215_partA%s.json' % tag), 'w'), indent=1)
    if plan:
        todo = [n for n in out if plan_all or out[n]['label'] == 'COMBINED']
        say('Part B plan for positions: %s' % (todo or 'none (no COMBINED position)'))
        if todo:
            patch_plan(X, its, meta, todo, outdir, say)
    return out


def patch_plan(X, its, meta, positions, outdir, say, per_pos=1000):
    fams = sorted(set(i['fam'] for i in its)); half = {f: int(j % 2) for j, f in enumerate(np.random.default_rng(2151).permutation(fams))}
    fam = np.array([i['fam'] for i in its]); B = np.array([i['B'] for i in its]); arr = {}
    rr = np.random.default_rng(2152)
    for p in positions:
        pi = POS.index(p)
        for li in BAND:
            Z = demean(np.asarray(X[:, pi, li], np.float32), fam)
            D2 = np.array([i['d2'] for i in its])
            for h in (0, 1):
                m = np.array([half[f] != h for f in fam])         # fitted on the OTHER half
                M = np.stack([Z[m & (B == b)].mean(0) for b in range(10) if (m & (B == b)).any()]); M -= M.mean(0)
                U = np.linalg.svd(M, full_matrices=False)[2][:len(M) - 1].T
                Md = np.stack([Z[m & (D2 == v)].mean(0) for v in range(10) if (m & (D2 == v)).any()]); Md -= Md.mean(0)
                Dd = np.linalg.svd(Md, full_matrices=False)[2][:len(Md) - 1].T   # amendment 1: the d2-copy subspace (d2 <= 4: 4 dims)
                Uo, sv, _ = np.linalg.svd(U - Dd @ (Dd.T @ U), full_matrices=False)
                Uo = Uo[:, sv > 1e-3 * sv.max()]                          # B orthogonalised against the copy
                arr['U_%s_%d_%d' % (p, h, LAYERS[li])] = U.astype(np.float32)
                arr['D_%s_%d_%d' % (p, h, LAYERS[li])] = Dd.astype(np.float32)
                arr['Uo_%s_%d_%d' % (p, h, LAYERS[li])] = Uo.astype(np.float32)
                c2 = np.linalg.svd(U.T @ Dd, compute_uv=False) ** 2
                say('  %s layer %d half %d: B vs d2-copy subspace mean cos^2 %.3f (max %.3f); orthogonalised B keeps %d dims, '
                    'share of B left %.3f' % (p, LAYERS[li], h, c2.mean(), c2.max(), Uo.shape[1], np.mean(1 - c2)))
            arr['R_%s_%d' % (p, LAYERS[li])] = np.linalg.qr(rr.standard_normal((Z.shape[1], 9)))[0].astype(np.float32)
    rng = np.random.default_rng(2151); pairs = []
    alts = {k: [j for j, o in enumerate(its) if o['fam'] == i['fam'] and o['ver'] == i['ver'] and o['d1'] == i['d1'] and o['d2'] != i['d2']]
            for k, i in enumerate(its)}                   # x' differs from x only in d2 (so also in B)
    for p in positions:
        cand = [k for k, i in enumerate(its) if meta[k]['correct'] and alts[k]]
        for k in rng.permutation(cand)[:per_pos]:
            pairs.append(dict(pos=p, x=int(k), x2=int(rng.choice(alts[k])), half=half[its[k]['fam']]))
    np.savez(os.path.join(outdir, 'v215_plan.npz'), **arr)
    json.dump(dict(positions=positions, band=[LAYERS[li] for li in BAND], pairs=pairs), open(os.path.join(outdir, 'v215_pairs.json'), 'w'))
    say('plan: %d bases, %d pairs' % (len(arr), len(pairs)))


def cjob(args):
    lab, pi, li = args
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y, tr = G['X'], G[lab], G['tr']
    Z = np.asarray(X[:, pi, li], np.float32); sc = StandardScaler().fit(Z[tr])
    m = LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(Z[tr]), y[tr])
    return args, float((m.predict(sc.transform(Z[~tr])) == y[~tr]).mean())


def part_c(X, its, procs=4, say=print):
    tr = np.array([i['split'] == 'train' for i in its])
    d = dict(X=X, tr=tr, b=np.array([i['B'] for i in its]), a=np.array([i['A'] for i in its]))
    jobs = [(lab, pi, li) for lab in ('b', 'a') for pi in range(len(POS)) for li in range(len(LAYERS))]
    with Pool(procs, initializer=init, initargs=(d,)) as pool:
        res = dict(pool.imap_unordered(cjob, jobs, chunksize=1))
    say('Part C: their standard probe, train %d / test %d (no shared b equation)' % (tr.sum(), (~tr).sum()))
    for lab in ('b', 'a'):
        y = d[lab][~tr]
        say('  %s (majority %.3f): ' % (lab, np.bincount(y).max() / len(y)) + ' | '.join(
            '%s %s' % (POS[pi], ' '.join('%.2f' % res[(lab, pi, li)] for li in range(len(LAYERS)))) for pi in range(len(POS))))
        say('  %s Acc_preCoT (max over eq / semi / last and layers) %.3f  (Kudo et al. Table 2, Qwen2.5-7B L3: %s)' % (
            lab, max(res.values() if False else [res[(lab, pi, li)] for pi in range(len(POS)) for li in range(len(LAYERS))]),
            '50.5%' if lab == 'b' else '17.9%'))
    return res


def load(root, part):
    f = glob.glob(os.path.join(root, '**', 'v215_%s_states.npy' % part), recursive=True)[0]
    return np.load(f, mmap_mode='r'), json.load(open(f.replace('_states.npy', '_meta.json')))


def main(root, parts):
    import v215_items as I
    fam, rep, info = I.build(); s = I.all_sha()
    print('items sha %s (frozen %s)' % (s, I.ITEMS_SHA), flush=True)
    if I.ITEMS_SHA and s != I.ITEMS_SHA:
        sys.exit('items hash differs')
    if 'A' in parts:
        X, m = load(root, 'fam'); assert m['items_sha'] == s and len(m['items']) == len(fam)
        print('=== Part A, main format (their simple_backtracking chain) ===', flush=True)
        part_a(X, fam, m['items'], plan=False)
    if 'S' in parts:
        fams = I.build_short(); X, m = load(root, 'fams'); assert m['items_sha'] == s and len(m['items']) == len(fams)
        print('=== Part A, short-chain arm (their Simple chain: "b = B , a = A"); Part B plan from here ===', flush=True)
        print('short arm: exact chain %.4f, b step %.4f (G1 uses the b step)' % (np.mean([x['correct'] for x in m['items']]),
              np.mean([x['predB'] == fams[x['n']]['B'] for x in m['items']])), flush=True)
        mi = [dict(x, correct=bool(x['predB'] == fams[x['n']]['B'])) for x in m['items']]
        part_a(X, fams, mi, plan_all=os.environ.get('V215_PLAN_ALL') == '1', tag='_short')
    if 'C' in parts:
        X, m = load(root, 'rep'); assert m['items_sha'] == s and len(m['items']) == len(rep)
        print('Part C: replication items correct (exact chain) %.4f' % np.mean([x['correct'] for x in m['items']]), flush=True)
        part_c(X, rep)


def selftest():
    """Synthetic states on the real family design (60 families): computed (a shared code of B's value) -> COMBINED +
    FOLLOWS; copies only -> NOT-COMBINED + DOES-NOT-FOLLOW; op-independent pair mixing of (d1, d2) -> not FOLLOWS."""
    import tempfile, v215_items as I
    fam, _, _ = I.build(); keep = set(sorted(set(i['fam'] for i in fam))[::4]); its = [i for i in fam if i['fam'] in keep]
    meta = [dict(correct=True) for _ in its]; rng = np.random.default_rng(0); dd = 64; ok = True
    U1, U2, CB = (rng.standard_normal((10, dd)) for _ in range(3)); Ux, Vy = rng.standard_normal((10, dd)), rng.standard_normal((10, dd))
    off = {f: rng.standard_normal(dd) * 3 for f in keep}
    for world, want, wswap in (('computed', 'COMBINED', 'FOLLOWS'), ('copies', 'NOT-COMBINED', 'DOES-NOT-FOLLOW'),
                               ('mixing', None, 'not FOLLOWS')):
        X = np.zeros((len(its), 3, len(LAYERS), dd), np.float32)
        for k, i in enumerate(its):
            v = off[i['fam']] + U1[i['d1']] + U2[i['d2']] + rng.standard_normal((3, len(LAYERS), dd)) * 1.0
            if world == 'computed':
                v = v + 0.8 * CB[i['B']]
            if world == 'mixing':
                v = v + 1.5 * np.tanh(2 * Ux[i['d1']] * Vy[i['d2']])
            X[k] = v
        out = part_a(X, its, meta, procs=4, say=lambda s: None, plan=world == 'computed', plan_all=True, outdir=tempfile.mkdtemp())
        for p, r in out.items():
            good = (r['label'] == want if want else True) and (r['swap'] == wswap if wswap != 'not FOLLOWS' else r['swap'] != 'FOLLOWS')
            print('selftest %-8s %-4s ex_N %+.3f [%+.3f, %+.3f] -> %-12s I %+.3f -> %-15s %s' % (
                world, p, *r['exN'], r['label'], r['I'][0], r['swap'], 'OK' if good else 'FAIL'), flush=True)
            ok &= good
    return ok


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        sys.exit(0 if selftest() else 1)
    main(sys.argv[1], sys.argv[2:] or ['A', 'S', 'C'])
