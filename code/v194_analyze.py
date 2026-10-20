"""v194 analysis (docs/prereg_v194_ouro_operation_swap.md). Operation swap: does the readable sign follow the problem?

Items: plus versions (first step a + b) and minus twins (first step a - b), identical numbers. States at P0 (last prompt
token) and P1 (answer token before the units digit), mid/end of loops 1-4. Probe: v178 probe on family-demeaned states,
folds grouped by template, separately per version. Labels mod 5: AB = (a+b) mod 5, AMB = (a-b) mod 5 (units digits).
excess(L) = acc(L, state) - acc(L, copy ceiling = same probe on one-hot (a_u, b_u, c_u)), per item.

PRIMARY (P1, end4):
  D+ = excess(AB) - excess(AMB) on plus items;  D- = excess(AMB) - excess(AB) on minus items
  I  = (mean D+ + mean D-) / 2       (a fixed preference for either sign cancels; it is F = (mean D- - mean D+) / 2)
  paired cluster bootstrap over pair ids (a plus family and its minus twin), 10,000 resamples, seed 194, 97.5% CI.
  FOLLOWS          I >= 0.05, CI lower > 0, mean D+ > 0 and mean D- > 0
  DOES-NOT-FOLLOW  CI upper of I < 0.02
  UNCLEAR          otherwise
Gates (any failure -> UNSCORABLE): G1 answer-only accuracy after loop 4 >= 0.30 in EACH version; G2 copies at P0,
best state, in each version: a and c >= 0.75, b >= 0.60 (b fades; pilot 0.72-0.77, set before the freeze); G3 bare sum units digit at P1 end4 >= 0.90.
Secondary: the same at end3 and at P0; the line test (needed form vs the other forms on answer + k*c) per version;
by-state excess of AB and AMB per version.
Part B patch plan (frozen here, used by v194_patch.py): templates split into halves {0-5}, {6-11}; for each half, 9-dim
subspaces from the OTHER half's plus items at P1 end3 (family-centred class means of (a+b) mod 10, (a-b) mod 10, answer
mod 10) and a random orthonormal 9-dim subspace (seed 1941); pairs (x, x') of plus items in the same family with the same
c_u and the same parity of a_u + b_u but a different (a_u + b_u) mod 10, x correct after loop 4; 1,000 pairs per half.
Amendment 2026-10-14 (before any analysis): B2 answer-fixed pairs (x, x'') with the same answer as x, a different
(a_u + b_u) mod 10 of the same parity (c compensates), x correct; 1,000 per half, separate random stream (seed 1943).
Amendment 2026-10-14: second patch point at the end of loop 2; its subspaces (U_*_h_e2) are fitted on end2 states.

  python v194_analyze.py <dir with v194_states.npy, v194_meta.json>     (writes v194_patch_plan.npz, v194_patch_pairs.json)
  python v194_analyze.py --selftest
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V
import v186_analyze as A

AB, AMB = (1, 1, 0), (1, 4, 0)
LINE = {'plus': [(1, 1, k) for k in range(4)], 'minus': [(1, 4, k) for k in range(4)]}   # answer + k*c, needed first
NEED = {'plus': AB, 'minus': AMB}
Q = (1.25, 98.75)
G = {}


def form(i, f):
    L = i['labels']
    return (f[0] * L['a'] + f[1] * L['b'] + f[2] * L['c']) % 5


def init(X, its):
    G['X'], G['its'] = X, its


def job(args):
    split, op, pos, st, lab, feat = args
    idx = np.array([k for k, i in enumerate(G['its']) if i['set'] == split and i['op'] == op])
    its = [G['its'][k] for k in idx]
    fam = np.array([i['fam'] for i in its]); tid = np.array([i['tid'] for i in its])
    if isinstance(lab, tuple):
        y = np.array([form(i, lab) for i in its])
    else:
        y = np.array([i['labels'][lab] for i in its])
    Z = A.onehot(its) if feat == 'ceil' else np.asarray(G['X'][idx, pos, st], dtype=np.float32)
    return args, (V.probe_predict(A.demean(Z, fam), y, tid, V.folds(tid)) == y).astype(np.int8)


def pair_ci(pair_p, dp, pair_m, dm, n=10000, seed=194):
    """Paired cluster bootstrap over pair ids for I = (mean dp + mean dm)/2 and F = (mean dm - mean dp)/2."""
    u = np.unique(np.concatenate([pair_p, pair_m])); ix = {p: j for j, p in enumerate(u)}
    Sp, Np, Sm, Nm = (np.zeros(len(u)) for _ in range(4))
    np.add.at(Sp, [ix[p] for p in pair_p], dp); np.add.at(Np, [ix[p] for p in pair_p], 1)
    np.add.at(Sm, [ix[p] for p in pair_m], dm); np.add.at(Nm, [ix[p] for p in pair_m], 1)
    rng = np.random.default_rng(seed); Is, Fs = [], []
    for _ in range(n):
        q = rng.choice(len(u), len(u)); a, b = Sp[q].sum() / Np[q].sum(), Sm[q].sum() / Nm[q].sum()
        Is.append((a + b) / 2); Fs.append((b - a) / 2)
    I, F = (dp.mean() + dm.mean()) / 2, (dm.mean() - dp.mean()) / 2
    return (I, *np.percentile(Is, Q)), (F, *np.percentile(Fs, Q))


def analyse(X, meta, procs=4, secondary=True, verbose=True, plan_dir=None):
    out = []; say = out.append; its = meta['items']; S = meta['states']; E, E3 = S.index('end4'), S.index('end3')
    sel = lambda sp, op: [i for i in its if i['set'] == sp and i['op'] == op]
    acc = {op: np.mean([i['correct'] for i in sel('main', op)], 0) for op in ('plus', 'minus')}
    say('items: plus %d, minus %d, bare %d' % (len(sel('main', 'plus')), len(sel('main', 'minus')), len(sel('bare', 'plus'))))
    say('answer-only accuracy by loop: plus %s | minus %s' % (np.round(acc['plus'], 3).tolist(), np.round(acc['minus'], 3).tolist()))
    sts = list(range(len(S))) if secondary else [E]
    jobs = [('main', op, 0, st, l, 'state') for op in ('plus', 'minus') for l in ('a', 'b', 'c') for st in range(len(S))]
    jobs += [('bare', 'plus', 1, E, 'ans', 'state')]
    for op in ('plus', 'minus'):
        for f in (AB, AMB):
            jobs += [('main', op, 1, st, f, 'state') for st in sts] + [('main', op, 1, E, f, 'ceil')]
            if secondary:
                jobs += [('main', op, 0, E, f, 'state')]
        if secondary:
            jobs += [('main', op, 1, st, f, x) for f in LINE[op][1:] for st, x in ((E, 'state'), (E3, 'state'), (E, 'ceil'))]
    jobs = list(dict.fromkeys(jobs))
    if procs > 1:
        with Pool(procs, initializer=init, initargs=(X, its)) as pool:
            res = dict(pool.map(job, jobs))
    else:
        init(X, its); res = dict(job(j) for j in jobs)
    a = lambda *k: res[k].mean()
    best = {op: {l: max(a('main', op, 0, st, l, 'state') for st in range(len(S))) for l in 'abc'} for op in ('plus', 'minus')}
    g1 = all(acc[op][3] >= 0.30 for op in acc)
    g2 = all(best[op]['a'] >= 0.75 and best[op]['b'] >= 0.60 and best[op]['c'] >= 0.75 for op in best)
    g3 = a('bare', 'plus', 1, E, 'ans', 'state') >= 0.90
    say('G1 accuracy after loop 4 plus %.3f minus %.3f (each >= 0.30: %s) | G2 copies best state %s (a, c >= 0.75, b >= 0.60: %s) | '
        'G3 bare sum digit at P1 end4 %.3f (>= 0.90: %s)' % (acc['plus'][3], acc['minus'][3], g1, ' '.join(
            '%s %s' % (op, ' '.join('%s %.3f' % kv for kv in best[op].items())) for op in best), g2,
            a('bare', 'plus', 1, E, 'ans', 'state'), g3))

    def stat(pos, st, tag):
        ex = {(op, f): res[('main', op, pos, st, f, 'state')].astype(float) - res[('main', op, 1, E, f, 'ceil')]
              for op in ('plus', 'minus') for f in (AB, AMB)}
        dp = ex[('plus', AB)] - ex[('plus', AMB)]; dm = ex[('minus', AMB)] - ex[('minus', AB)]
        pp = np.array([i['pair'] for i in sel('main', 'plus')]); pm = np.array([i['pair'] for i in sel('main', 'minus')])
        (I, Ilo, Ihi), (F, Flo, Fhi) = pair_ci(pp, dp, pm, dm)
        say('%s | plus: a+b %+.3f a-b %+.3f -> D+ %+.3f | minus: a-b %+.3f a+b %+.3f -> D- %+.3f | I %+.3f [%+.3f, %+.3f] | '
            'fixed preference for a-b F %+.3f [%+.3f, %+.3f]' % (
                tag, ex[('plus', AB)].mean(), ex[('plus', AMB)].mean(), dp.mean(), ex[('minus', AMB)].mean(),
                ex[('minus', AB)].mean(), dm.mean(), I, Ilo, Ihi, F, Flo, Fhi))
        if I >= 0.05 and Ilo > 0 and dp.mean() > 0 and dm.mean() > 0:
            return 'FOLLOWS'
        return 'DOES-NOT-FOLLOW' if Ihi < 0.02 else 'UNCLEAR'

    label = stat(1, E, 'PRIMARY P1 end4')
    if not (g1 and g2 and g3):
        label = 'UNSCORABLE'
    say('PRIMARY -> %s' % label)
    if secondary:
        say('--- secondary ---')
        say('P1 end3: %s' % stat(1, E3, 'P1 end3'))
        say('P0 end4: %s' % stat(0, E, 'P0 end4'))
        for op in ('plus', 'minus'):
            ceil = {f: res[('main', op, 1, E, f, 'ceil')] for f in LINE[op]}
            for st, sn in ((E, 'end4'), (E3, 'end3')):
                ex = {f: res[('main', op, 1, st, f, 'state')].astype(float) - ceil[f] for f in LINE[op]}
                bo = max(LINE[op][1:], key=lambda f: ex[f].mean()); d = ex[LINE[op][0]] - ex[bo]
                fam = np.array([i['fam'] for i in sel('main', op)])
                say('line test %s %s: needed %+.3f, line %s -> needed - best other %+.3f [%+.3f, %+.3f]' % (
                    op, sn, ex[LINE[op][0]].mean(), ' '.join('%s %+.3f' % (f, ex[f].mean()) for f in LINE[op][1:]),
                    *A.cluster_ci(fam, d, seed=194, q=Q)))
            for f in (AB, AMB):
                say('%s %s excess by state: %s' % (op, 'a+b' if f == AB else 'a-b', ' '.join(
                    '%s %+.3f' % (s, a('main', op, 1, k, f, 'state') - a('main', op, 1, E, f, 'ceil')) for k, s in enumerate(S))))
    if plan_dir is not None:
        patch_plan(X, its, S.index('end3'), plan_dir, say, E2=S.index('end2'))
    if verbose:
        print('\n'.join(out), flush=True)
    return label


def patch_plan(X, its, E3, outdir, say, per_half=1000, E2=None):
    """Part B plan: cross-fitted subspaces and pairs (frozen procedure)."""
    plus = [k for k, i in enumerate(its) if i['set'] == 'main' and i['op'] == 'plus']
    half = {k: int(its[k]['tid'] >= 6) for k in plus}
    rng = np.random.default_rng(1941); d = X.shape[-1]; plan = {}; pairs = []
    rng2 = np.random.default_rng(1943); pairs_fixed = []

    def basis(ks, key, st=None):
        st = E3 if st is None else st
        Z = np.asarray(X[np.array(ks), 1, st], dtype=np.float64); fam = np.array([its[k]['fam'] for k in ks])
        Z = A.demean(Z, fam); y = np.array([key(its[k]) for k in ks])
        mu = np.stack([Z[y == v].mean(0) for v in range(10)]); mu -= mu.mean(0)
        return np.linalg.svd(mu, full_matrices=False)[2][:9].T.astype(np.float32)

    keys = {'ab': lambda i: (i['labels']['a'] + i['labels']['b']) % 10, 'amb': lambda i: (i['labels']['a'] - i['labels']['b']) % 10,
            'ans': lambda i: i['labels']['ans']}
    for h in (0, 1):
        other = [k for k in plus if half[k] != h]
        for nm, key in keys.items():
            plan['U_%s_%d' % (nm, h)] = basis(other, key)
        plan['U_rand_%d' % h] = np.linalg.qr(rng.normal(size=(d, 9)))[0].astype(np.float32)
        if E2 is not None:   # amendment 2026-10-14: second patch point, end of loop 2 (subspaces fitted on end2 states)
            for nm, key in keys.items():
                plan['U_%s_%d_e2' % (nm, h)] = basis(other, key, E2)
        cand = []
        mine = [k for k in plus if half[k] == h and its[k]['correct'][3]]
        byfam = {}
        for k in [k for k in plus if half[k] == h]:
            byfam.setdefault(its[k]['fam'], []).append(k)
        for k in mine:
            L = its[k]['labels']; s = (L['a'] + L['b']) % 10
            opts = [j for j in byfam[its[k]['fam']] if its[j]['labels']['c'] == L['c']
                    and (its[j]['labels']['a'] + its[j]['labels']['b']) % 2 == s % 2
                    and (its[j]['labels']['a'] + its[j]['labels']['b']) % 10 != s]
            if opts:
                cand.append((k, int(rng.choice(opts))))
        take = rng.choice(len(cand), min(per_half, len(cand)), replace=False)
        pairs += [dict(half=h, k=cand[t][0], kp=cand[t][1]) for t in sorted(take)]
        say('patch plan half %d: subspaces from %d plus items of the other half; %d candidate pairs, %d taken' % (
            h, len(other), len(cand), min(per_half, len(cand))))
        # amendment 2026-10-14 (B2, answer-fixed pairs): x'' has the SAME answer as x but a different (a+b) mod 10 of the
        # same parity (c shifted to compensate); separate random stream so the B1 pairs above are unchanged.
        cand2 = []
        for k in mine:
            L = its[k]['labels']; s = (L['a'] + L['b']) % 10
            opts = [j for j in byfam[its[k]['fam']] if its[j]['answer'] == its[k]['answer']
                    and (its[j]['labels']['a'] + its[j]['labels']['b']) % 2 == s % 2
                    and (its[j]['labels']['a'] + its[j]['labels']['b']) % 10 != s]
            if opts:
                cand2.append((k, int(rng2.choice(opts))))
        take2 = rng2.choice(len(cand2), min(per_half, len(cand2)), replace=False)
        pairs_fixed += [dict(half=h, k=cand2[t][0], kp=cand2[t][1]) for t in sorted(take2)]
        say('patch plan half %d (B2, answer fixed): %d candidate pairs, %d taken' % (h, len(cand2), min(per_half, len(cand2))))
    np.savez_compressed(os.path.join(outdir, 'v194_patch_plan.npz'), **plan)
    # k, kp index the main+bare items of v194_items.json in file order (the extraction's order)
    json.dump(dict(pairs=pairs, pairs_fixed=pairs_fixed), open(os.path.join(outdir, 'v194_patch_pairs.json'), 'w'))


def selftest():
    its = [i for i in json.load(open('v194_items.json'))['items'] if i['set'] in ('main', 'bare')]
    N = len(its); d = 64; rng = np.random.default_rng(0)
    W = {k: rng.normal(size=(10, d)) for k in ('a', 'b', 'c', 'ab', 'amb', 'ans')}
    U, Vv = rng.normal(size=(10, d)), rng.normal(size=(10, d))
    fam = np.array([i['fam'] for i in its]); off = 20 * rng.normal(size=(fam.max() + 1, d))

    def world(kind):
        X = np.zeros((N, 2, 8, d), np.float32)
        for r, i in enumerate(its):
            L = i['labels']; v = off[i['fam']] + 0.05 * rng.normal(size=d)
            if kind != 'broken':
                v = v + W['a'][L['a']] + W['b'][L['b']] + (W['c'][L['c']] if i['set'] == 'main' else 0)
            p1 = v + 3.0 * W['ans'][L['ans']]
            ab, amb = (L['a'] + L['b']) % 10, (L['a'] - L['b']) % 10
            if i['set'] == 'main':
                on = kind == 'follows' or (kind.startswith('partial') and rng.random() < float(kind.split('_')[1]))
                if on:
                    p1 = p1 + (W['ab'][ab] if i['op'] == 'plus' else W['amb'][amb])
                if kind == 'fixed_plus':
                    p1 = p1 + W['ab'][ab]
                if kind == 'fixed_difference':
                    p1 = p1 + W['amb'][amb]
                if kind == 'generic_mixing':
                    p1 = p1 + np.tanh(U[L['a']] * Vv[L['b']] * 2)
            X[r, 0, :] = v; X[r, 1, :] = p1
        return X

    meta = dict(states=['mid1', 'end1', 'mid2', 'end2', 'mid3', 'end3', 'mid4', 'end4'],
                items=[dict(i, correct=[True] * 4) for i in its])
    want = {'follows': {'FOLLOWS'}, 'partial_0.4': None, 'fixed_plus': {'DOES-NOT-FOLLOW'},
            'fixed_difference': {'DOES-NOT-FOLLOW'}, 'generic_mixing': {'DOES-NOT-FOLLOW', 'UNCLEAR'}, 'broken': {'UNSCORABLE'}}
    ok = True
    for k, w in want.items():
        print('== world %s' % k, flush=True)
        lab = analyse(world(k), meta, secondary=False, verbose=True)
        good = w is None or lab in w
        print('selftest %-17s -> %-16s %s' % (k, lab, 'recorded (sensitivity)' if w is None else 'OK' if good else
                                              'FAIL (want %s)' % sorted(w)), flush=True)
        ok &= good
    return ok


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        sys.exit(0 if selftest() else 1)
    f = sorted(glob.glob(os.path.join(sys.argv[1], '**', 'v194_states.npy'), recursive=True))[0]
    meta = json.load(open(os.path.join(os.path.dirname(f), 'v194_meta.json')))
    assert {i['set'] for i in meta['items']} == {'main', 'bare'}
    analyse(np.load(f, mmap_mode='r'), meta, plan_dir='.')
