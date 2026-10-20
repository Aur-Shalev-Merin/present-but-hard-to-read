"""v199 analysis (docs/prereg_v199_ouro_running_total.md). Is the running total held at the story's sentence ends?

Items: v194 main set (plus: first step a + b; minus twins: a - b; identical numbers). States at Sb (the period ending the
sentence that gives b), Sc (the period ending the sentence that gives c) and the four tokens before Sb (position index
1 + k for Sb-k), mid/end of loops 1-4. The sentence's tail after b is Sb-k for k < tail = Sb - bU (2-5 tokens). Probe: the v178 probe
on family-demeaned states, folds grouped by template, separately per version. Labels mod 5 on units digits.
excess(L) = acc(L, state) - acc(L, copy ceiling = same probe on one-hot (a_u, b_u, c_u)), per item.

PRIMARY (Part A, Sb, each loop end end1..end4):
  D+ = excess(a+b) - excess(a-b) on plus items;  D- = excess(a-b) - excess(a+b) on minus items
  I  = (mean D+ + mean D-) / 2      (a fixed preference for either sign cancels; F = (mean D- - mean D+) / 2)
  paired cluster bootstrap over pair ids, 10,000 resamples, seed 199, 99.375% CI (Bonferroni: 2.5% for Part A split
  over the four loop ends; Part B gets the other 2.5%).
  FOLLOWS          at one or more loop ends: I >= 0.05, CI lower > 0, mean D+ > 0 and mean D- > 0
  DOES-NOT-FOLLOW  at every loop end: CI upper of I < 0.02
  UNCLEAR          otherwise
Gates (any failure -> UNSCORABLE): G1 answer-only accuracy after loop 4 >= 0.30 in each version; G2 copies a and b at
Sb, best state, >= 0.75 in each version.
Part B gate (not a label): Part B runs if Part A is FOLLOWS, or if the same FOLLOWS rule holds at Sb-1 (the token
before the period, inside every template's tail) at some loop end.
Secondary (reported, not labels): I at the Sb mid states; I at Sb-1 at the loop ends; at Sc, I for the first step at
the loop ends and I for the answer (a+b-c against a-b-c) at every state; copies at Sc.
Part B plan (written here whatever the label): templates split into halves {0-5}, {6-11}; for each version, half and
tail offset k = 0..4 (Sb-k), 9-dim subspaces from the OTHER half's items of that version whose tail contains Sb-k:
family-centred class means of the needed first step N ((a+b) mod 10 plus, (a-b) mod 10 minus) and of its sign-flipped
twin F, at each patch point mid1, end1, mid2, end2, mid3, end3, mid4 (an offset that the other half lacks uses the
nearest smaller offset's subspace); a random orthonormal 9-dim subspace per half (seed 1991). Pairs (x, x') in
the same family: same c_u, N_x' != N_x (mod 10) with the same parity, x correct after loop 4; 500 per version and half
(seed 1992).

  python v199_analyze.py <dir with v199_states.npy, v199_meta.json>   (writes v199_patch_plan.npz, v199_patch_pairs.json)
  python v199_analyze.py --selftest
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V
import v186_analyze as A

AB, AMB = (1, 1, 0), (1, 4, 0)            # first step, mod 5
ANSP, ANSM = (1, 1, 4), (1, 4, 4)         # answer a+b-c, a-b-c, mod 5
FIRST = (AB, AMB); ANSWER = (ANSP, ANSM)  # (needed in plus, needed in minus)
Q = (0.3125, 99.6875)
POINTS = ['mid1', 'end1', 'mid2', 'end2', 'mid3', 'end3', 'mid4']   # Part B patch points (end4 cannot reach P1)
G = {}


def form(i, f):
    L = i['labels']
    return (f[0] * L['a'] + f[1] * L['b'] + f[2] * L['c']) % 5


def need10(i, flip=False):
    L = i['labels']; minus = (i['op'] == 'minus') != flip
    return (L['a'] - L['b']) % 10 if minus else (L['a'] + L['b']) % 10


def init(X, its):
    G['X'], G['its'] = X, its


def job(args):
    op, pos, st, lab, feat = args
    idx = np.array([k for k, i in enumerate(G['its']) if i['op'] == op])
    its = [G['its'][k] for k in idx]
    fam = np.array([i['fam'] for i in its]); tid = np.array([i['tid'] for i in its])
    y = np.array([form(i, lab) for i in its]) if isinstance(lab, tuple) else np.array([i['labels'][lab] for i in its])
    Z = A.onehot(its) if feat == 'ceil' else np.asarray(G['X'][idx, pos, st], dtype=np.float32)
    return args, (V.probe_predict(A.demean(Z, fam), y, tid, V.folds(tid)) == y).astype(np.int8)


def pair_ci(pair_p, dp, pair_m, dm, n=10000, seed=199):
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
    out = []; say = out.append; its = meta['items']; S = meta['states']
    ENDS = [S.index('end%d' % t) for t in range(1, 5)]
    sel = lambda op: [i for i in its if i['op'] == op]
    acc = {op: np.mean([i['correct'] for i in sel(op)], 0) for op in ('plus', 'minus')}
    say('items: plus %d, minus %d' % (len(sel('plus')), len(sel('minus'))))
    say('answer-only accuracy by loop: plus %s | minus %s' % (np.round(acc['plus'], 3).tolist(), np.round(acc['minus'], 3).tolist()))
    allst = list(range(len(S)))
    jobs = [(op, 0, st, l, 'state') for op in ('plus', 'minus') for l in ('a', 'b') for st in allst]
    jobs += [(op, 0, st, f, 'state') for op in ('plus', 'minus') for f in FIRST for st in (allst if secondary else ENDS)]
    jobs += [(op, 1, E, f, 'ceil') for op in ('plus', 'minus') for f in FIRST + (ANSWER if secondary else ()) for E in ENDS[-1:]]
    jobs += [(op, 2, st, f, 'state') for op in ('plus', 'minus') for f in FIRST for st in ENDS]       # Sb-1, Part B gate
    if secondary:
        jobs += [(op, 0, st, 'c', 'state') for op in ('plus', 'minus') for st in allst]
        jobs += [(op, 1, st, l, 'state') for op in ('plus', 'minus') for l in ('a', 'b', 'c') for st in ENDS]
        jobs += [(op, 1, st, f, 'state') for op in ('plus', 'minus') for f in FIRST for st in ENDS]
        jobs += [(op, 1, st, f, 'state') for op in ('plus', 'minus') for f in ANSWER for st in allst]
    jobs = list(dict.fromkeys(jobs))
    if procs > 1:
        with Pool(procs, initializer=init, initargs=(X, its)) as pool:
            res = dict(pool.map(job, jobs))
    else:
        init(X, its); res = dict(job(j) for j in jobs)
    a = lambda *k: res[k].mean()
    best = {op: {l: max(a(op, 0, st, l, 'state') for st in allst) for l in 'ab'} for op in ('plus', 'minus')}
    g1 = all(acc[op][3] >= 0.30 for op in acc)
    g2 = all(best[op]['a'] >= 0.75 and best[op]['b'] >= 0.75 for op in best)
    say('G1 accuracy after loop 4 plus %.3f minus %.3f (each >= 0.30: %s) | G2 copies at Sb, best state: %s (a, b >= 0.75: %s)' % (
        acc['plus'][3], acc['minus'][3], g1, ' | '.join('%s %s' % (op, ' '.join('%s %.3f' % kv for kv in best[op].items()))
                                                       for op in best), g2))
    pp = np.array([i['pair'] for i in sel('plus')]); pm = np.array([i['pair'] for i in sel('minus')])

    def stat(pos, st, forms, tag):
        fp, fm = forms
        ex = {(op, f): res[(op, pos, st, f, 'state')].astype(float) - res[(op, 1, ENDS[-1], f, 'ceil')]
              for op in ('plus', 'minus') for f in forms}
        dp = ex[('plus', fp)] - ex[('plus', fm)]; dm = ex[('minus', fm)] - ex[('minus', fp)]
        (I, Ilo, Ihi), (F, Flo, Fhi) = pair_ci(pp, dp, pm, dm)
        say('%s | plus: needed %+.3f flipped %+.3f -> D+ %+.3f | minus: needed %+.3f flipped %+.3f -> D- %+.3f | '
            'I %+.3f [%+.3f, %+.3f] | F %+.3f [%+.3f, %+.3f]' % (
                tag, ex[('plus', fp)].mean(), ex[('plus', fm)].mean(), dp.mean(), ex[('minus', fm)].mean(),
                ex[('minus', fp)].mean(), dm.mean(), I, Ilo, Ihi, F, Flo, Fhi))
        return dict(I=I, lo=Ilo, hi=Ihi, follows=bool(I >= 0.05 and Ilo > 0 and dp.mean() > 0 and dm.mean() > 0))

    r = [stat(0, E, FIRST, 'PRIMARY Sb %s first step' % S[E]) for E in ENDS]
    label = 'FOLLOWS' if any(x['follows'] for x in r) else 'DOES-NOT-FOLLOW' if all(x['hi'] < 0.02 for x in r) else 'UNCLEAR'
    if not (g1 and g2):
        label = 'UNSCORABLE'
    say('PRIMARY -> %s (passing loop ends: %s)' % (label, ', '.join(S[E] for E, x in zip(ENDS, r) if x['follows']) or 'none'))
    r1 = [stat(2, E, FIRST, 'GATE Sb-1 %s first step' % S[E]) for E in ENDS]
    run_b = label == 'FOLLOWS' or (label != 'UNSCORABLE' and any(x['follows'] for x in r1)); G['run_b'] = run_b
    say('PART B GATE -> %s (Sb-1 passing loop ends: %s)' % ('RUN' if run_b else 'SKIP', ', '.join(
        S[E] for E, x in zip(ENDS, r1) if x['follows']) or 'none'))
    if secondary:
        say('--- secondary (not labels) ---')
        for st in allst:
            if st not in ENDS:
                stat(0, st, FIRST, 'Sb %s first step' % S[st])
        for E in ENDS:
            stat(1, E, FIRST, 'Sc %s first step' % S[E])
        for st in allst:
            stat(1, st, ANSWER, 'Sc %s answer' % S[st])
        for op in ('plus', 'minus'):
            say('%s copies at Sb by state: %s' % (op, ' | '.join('%s %s' % (l, ' '.join('%.2f' % a(op, 0, st, l, 'state') for st in allst))
                                                                for l in 'abc')))
            say('%s copies at Sc at loop ends: %s' % (op, ' | '.join('%s %s' % (l, ' '.join('%.2f' % a(op, 1, E, l, 'state') for E in ENDS))
                                                                   for l in 'abc')))
    if plan_dir is not None:
        patch_plan(X, its, S, plan_dir, say)
    if verbose:
        print('\n'.join(out), flush=True)
    return label


def patch_plan(X, its, S, outdir, say, per=500):
    """Part B plan: cross-fitted subspaces at each tail offset and pairs (frozen procedure)."""
    rng = np.random.default_rng(1991); rng2 = np.random.default_rng(1992); d = X.shape[-1]; plan = {}; pairs = []
    half = lambda k: int(its[k]['tid'] >= 6)

    def basis(ks, flip, st, off=0):
        Z = np.asarray(X[np.array(ks), 0 if off == 0 else 1 + off, st], dtype=np.float64); fam = np.array([its[k]['fam'] for k in ks])
        Z = A.demean(Z, fam); y = np.array([need10(its[k], flip) for k in ks])
        mu = np.stack([Z[y == v].mean(0) for v in range(10)]); mu -= mu.mean(0)
        return np.linalg.svd(mu, full_matrices=False)[2][:9].T.astype(np.float32)

    for h in (0, 1):
        plan['U_rand_%d' % h] = np.linalg.qr(rng.normal(size=(d, 9)))[0].astype(np.float32)
    for op in ('plus', 'minus'):
        ks = [k for k, i in enumerate(its) if i['op'] == op]
        for h in (0, 1):
            other = [k for k in ks if half(k) != h]
            for o in range(5):
                fit = [k for k in other if its[k]['pos']['tail'] > o]
                src = o if fit else max(q for q in range(o) if any(its[k]['pos']['tail'] > q for k in other))
                for p in POINTS:
                    for nm, flip in (('N', False), ('F', True)):
                        key = 'U_%s_%s_%d_%s_o%d' % (nm, op, h, p, o)
                        plan[key] = basis(fit, flip, S.index(p), o) if fit else plan['U_%s_%s_%d_%s_o%d' % (nm, op, h, p, src)]
                say('patch plan %s half %d offset %d: %s' % (op, h, o, 'fitted on %d items' % len(fit) if fit else
                                                            'none in the other half; uses offset %d' % src))
            byfam = {}
            for k in [k for k in ks if half(k) == h]:
                byfam.setdefault(its[k]['fam'], []).append(k)
            cand = []
            for k in [k for k in ks if half(k) == h and its[k]['correct'][3]]:
                n = need10(its[k])
                opts = [j for j in byfam[its[k]['fam']] if its[j]['labels']['c'] == its[k]['labels']['c']
                        and need10(its[j]) % 2 == n % 2 and need10(its[j]) != n]
                if opts:
                    cand.append((k, int(rng2.choice(opts))))
            take = rng2.choice(len(cand), min(per, len(cand)), replace=False)
            pairs += [dict(op=op, half=h, k=its[cand[t][0]]['k'], kp=its[cand[t][1]]['k'], pos=its[cand[t][0]]['pos'])
                      for t in sorted(take)]
            say('patch plan %s half %d: subspaces from %d items of the other half; %d candidate pairs, %d taken' % (
                op, h, len(other), len(cand), min(per, len(cand))))
    np.savez_compressed(os.path.join(outdir, 'v199_patch_plan.npz'), **plan)
    # k, kp index the main items of v194_items.json in file order (the extraction's order)
    json.dump(dict(pairs=pairs), open(os.path.join(outdir, 'v199_patch_pairs.json'), 'w'))


def selftest():
    its = [i for i in json.load(open('v194_items.json'))['items'] if i['set'] == 'main']
    N = len(its); d = 64; rng = np.random.default_rng(0)
    W = {k: rng.normal(size=(10, d)) for k in ('a', 'b', 'c', 'ab', 'amb')}
    U, Vv = rng.normal(size=(10, d)), rng.normal(size=(10, d))
    fam = np.array([i['fam'] for i in its]); off = 20 * rng.normal(size=(fam.max() + 1, d))

    def world(kind):
        X = np.zeros((N, 6, 8, d), np.float32)
        for r, i in enumerate(its):
            L = i['labels']; v = off[i['fam']] + 0.05 * rng.normal(size=d)
            if kind != 'broken':
                v = v + W['a'][L['a']] + W['b'][L['b']]
            ab, amb = (L['a'] + L['b']) % 10, (L['a'] - L['b']) % 10
            on = kind == 'follows' or (kind.startswith('partial') and rng.random() < float(kind.split('_')[1]))
            v2 = v.copy()                                    # Sb; the other tail positions hold copies only
            if on:
                v2 = v2 + (W['ab'][ab] if i['op'] == 'plus' else W['amb'][amb])
            if kind == 'fixed_difference':
                v2 = v2 + W['amb'][amb]
            if kind == 'generic_mixing':
                v2 = v2 + np.tanh(U[L['a']] * Vv[L['b']] * 2)
            X[r, :, :] = v; X[r, 0, :] = v2; X[r, 1, :] = v + W['c'][L['c']]
            if kind == 'tail_only':                          # running total one token before the period, not at it
                X[r, 2, :] = v + (W['ab'][ab] if i['op'] == 'plus' else W['amb'][amb])
        return X

    meta = dict(states=['mid1', 'end1', 'mid2', 'end2', 'mid3', 'end3', 'mid4', 'end4'],
                items=[dict(i, k=k, correct=[True] * 4, pos=dict(Sb=0, Sc=0, bU=0, P1=0, tail=[3, 4, 5, 5, 3, 2, 3, 4, 3, 2, 3, 4][i['tid']]))
                       for k, i in enumerate(its)])
    want = {'follows': {'FOLLOWS'}, 'partial_0.4': None, 'copies_only': {'DOES-NOT-FOLLOW'},
            'fixed_difference': {'DOES-NOT-FOLLOW'}, 'generic_mixing': {'DOES-NOT-FOLLOW', 'UNCLEAR'}, 'broken': {'UNSCORABLE'},
            'tail_only': {'DOES-NOT-FOLLOW', 'UNCLEAR'}}
    gate = {'follows': True, 'copies_only': False, 'fixed_difference': False, 'generic_mixing': False, 'broken': False,
            'tail_only': True}
    ok = True
    for k, w in want.items():
        print('== world %s' % k, flush=True)
        X = world(k)
        lab = analyse(X, meta, secondary=False, verbose=True, plan_dir='.' if k == 'follows' else None)
        good = (w is None or lab in w) and (k not in gate or G['run_b'] == gate[k])
        print('selftest %-17s -> %-16s Part B %-4s %s' % (k, lab, 'RUN' if G['run_b'] else 'SKIP', 'recorded (sensitivity)'
                                                          if w is None else 'OK' if good else 'FAIL (want %s, Part B %s)' % (
                                                              sorted(w), gate.get(k))), flush=True)
        ok &= good
    pj = json.load(open('v199_patch_pairs.json'))['pairs']; pl = np.load('v199_patch_plan.npz')
    for p in pj:
        x, xp = its[p['k']], its[p['kp']]
        assert x['fam'] == xp['fam'] and x['labels']['c'] == xp['labels']['c'] and need10(x) != need10(xp)
        assert need10(x) % 2 == need10(xp) % 2 and int(x['tid'] >= 6) == p['half'] and x['op'] == p['op']
    print('selftest plan: %d pairs, rule checked; %d subspaces, shapes %s' % (len(pj), len(pl.files), {pl[f].shape for f in pl.files}))
    return ok


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        sys.exit(0 if selftest() else 1)
    f = sorted(glob.glob(os.path.join(sys.argv[1], '**', 'v199_states.npy'), recursive=True))[0]
    meta = json.load(open(os.path.join(os.path.dirname(f), 'v199_meta.json')))
    assert {i['set'] for i in meta['items']} == {'main'}
    analyse(np.load(f, mmap_mode='r'), meta, plan_dir='.')
