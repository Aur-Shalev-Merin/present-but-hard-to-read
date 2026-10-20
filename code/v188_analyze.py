"""v188 analysis: confirmation of v186 on NEW templates (docs/prereg_v188_ouro_intermediate_confirmation.md).
Written and selftested BEFORE any v188 state exists.

  python v188_analyze.py <dir containing v188_states.npy, v188_meta.json (searched recursively)>
  python v188_analyze.py --selftest

Probe machinery = frozen v186_analyze.py (family-demeaned states; v178 probe_predict; 5-fold CV grouped by template, or
by family on the bare set; copy ceiling = the same probe on one-hot (a_u, b_u, c_u)). excess(L) = acc(L, state) -
acc(L, copy ceiling), per item. Paired cluster bootstrap over families (10,000, seed 188). Two hypotheses ->
Bonferroni: 97.5% CIs.

PRIMARY LABELS ARE MOD 5 (chance 0.2). Mod 10 every combination of a, b, c shares its parity with a+b or with the
answer digit (a-b with a+b; a+b+2c with a+b; a+b+c = ans+2c with ans), which hands controls free accuracy when the
state holds m or the answer; 5 is prime, so combinations not proportional to (a+b) or (a+b-c) are clean controls.
H1 (bare addition, P0 = last prompt token, end4): target m5 = (a+b) mod 5; XOR-like controls (a-b), (a+2b), (2a+b)
   mod 5; rnd5 (random function of (a,b) into 0..4) must only be beaten.
   REPLICATED     D_ceil >= 0.15 and D_XOR >= 0.15 (both CI lower > 0) and D_rnd CI lower > 0
   NOT-REPLICATED D_ceil or D_XOR has its CI upper bound below 0.15
   (a-priori bars; the v186 re-scoring gave D_XOR -0.111, i.e. no specificity, so no reason to lower them)
H2 (word problems, P1 = answer token just before the units digit, end4): target m5; controls (a-b), (a+c), and the
   answer-digit-and-c combinations (a+b-2c) = (ans-c) and (a+b+c) = (ans+2c), all mod 5 (m = ans + c, so a state that
   merely mixes the answer digit with c raises these too); rnd5 must only be beaten.
   Bars = about half of v186's estimates re-scored on the mod-5 scale with these controls (v188_calibrate.py on the
   finished v186 states: excess +0.176, D_XOR +0.048 [0.032, 0.063]). NOTE: chosen after v186's near-miss, so a
   REPLICATED means a small edge holds on new problems, not that v186's original 0.15 bars were cleared.
   REPLICATED     D_ceil >= 0.09 and D_XOR >= 0.025 (both CI lower > 0) and D_rnd CI lower > 0
   NOT-REPLICATED D_ceil CI upper < 0.09 or D_XOR CI upper < 0.025
   Otherwise UNCLEAR (each hypothesis).
Gates (any failure -> UNSCORABLE for both): G1 word-problem answer-only accuracy after loop 4 >= 0.30; G2 copies a, b, c
   each >= 0.75 at P0 in their best state; G3 bare: the sum's units digit at P1 end4 >= 0.90.
Secondary: each target's excess by state; word P0 excess; H2 against v186's original 0.15 / 0.15 bars; the same tests
   on mod-10 labels (v186-style).
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v186_analyze as A

Q = (1.25, 98.75)
HYP = {'H1': dict(split='bare', pos=0, t='m5', rnd='rnd5', xor=('amb5', 'ap2b5', 'tapb5'), bar_c=0.15, bar_x=0.15),
       'H2': dict(split='main', pos=1, t='m5', rnd='rnd5', xor=('amb5', 'apc5', 'abmc5', 'abc5'), bar_c=0.09, bar_x=0.025)}
HYP10 = {'H1': dict(HYP['H1'], t='m', rnd='rnd', xor=('amb', 'ap2b', 'tapb')),          # secondary, v186-style mod 10
         'H2': dict(HYP['H2'], t='m', rnd='rnd', xor=('amb', 'apc', 'abmc', 'ab2c'))}


def ci(fam, d):
    return A.cluster_ci(fam, d, seed=188, q=Q)


def test(name, h, res, its, E, a, say, bar_c=None, bar_x=None):
    sp, p, t, r = h['split'], h['pos'], h['t'], h['rnd']
    bar_c = h['bar_c'] if bar_c is None else bar_c; bar_x = h['bar_x'] if bar_x is None else bar_x
    fam = np.array([i['fam'] for i in its if i['set'] == sp]); labs = (t, r) + h['xor']
    ex = {l: res[(sp, p, E, l, 'state')].astype(float) - res[(sp, p, E, l, 'ceil')] for l in labs}
    bx = max(h['xor'], key=lambda l: ex[l].mean())
    dc, dx, dr = ci(fam, ex[t]), ci(fam, ex[t] - ex[bx]), ci(fam, ex[t] - ex[r])
    say('%s (%s, P%d, end4) raw: %s' % (name, sp, p, ' '.join('%s %.3f (ceil %.3f)' % (
        l, a(sp, p, E, l, 'state'), a(sp, p, E, l, 'ceil')) for l in labs)))
    say('%s excess: %s' % (name, ' '.join('%s %+.3f' % (l, ex[l].mean()) for l in labs)))
    say('%s D_ceil %+.3f [%+.3f, %+.3f] (bar %.2f) | D_XOR vs %s %+.3f [%+.3f, %+.3f] (bar %.2f) | D_rnd %+.3f [%+.3f, %+.3f]'
        % ((name,) + dc + (bar_c, bx) + dx + (bar_x,) + dr))
    if dc[0] >= bar_c and dc[1] > 0 and dx[0] >= bar_x and dx[1] > 0 and dr[1] > 0:
        return 'REPLICATED'
    return 'NOT-REPLICATED' if dc[2] < bar_c or dx[2] < bar_x else 'UNCLEAR'


def analyse(X, meta, procs=4, secondary=True, verbose=True):
    out = []; say = out.append; its = meta['items']; S = meta['states']; E = S.index('end4')
    main = [i for i in its if i['set'] == 'main']; bare = [i for i in its if i['set'] == 'bare']
    acc = np.mean([i['correct'] for i in main], 0); accb = np.mean([i['correct'] for i in bare], 0)
    say('word items %d (families %d, templates %d), bare items %d (families %d)' % (
        len(main), len({i['fam'] for i in main}), len({i['tid'] for i in main}), len(bare), len({i['fam'] for i in bare})))
    say('answer-only accuracy by loop: word %s | bare %s' % (np.round(acc, 3).tolist(), np.round(accb, 3).tolist()))
    jobs = [('main', 0, st, l, 'state') for l in ('a', 'b', 'c') for st in range(len(S))] + [('bare', 1, E, 'ans', 'state')]
    hs = list(HYP.values()) + (list(HYP10.values()) if secondary else [])
    for h in hs:
        jobs += [(h['split'], h['pos'], E, l, f) for l in (h['t'], h['rnd']) + h['xor'] for f in ('state', 'ceil')]
    if secondary:
        for h in HYP.values():
            jobs += [(h['split'], h['pos'], st, h['t'], 'state') for st in range(len(S)) if st != E]
        jobs += [('main', 0, E, 'm5', 'state')]
    jobs = list(dict.fromkeys(jobs))
    if procs > 1:
        with Pool(procs, initializer=A.init, initargs=(X, meta)) as pool:
            res = dict(pool.map(A.job, jobs))
    else:
        A.init(X, meta); res = dict(A.job(j) for j in jobs)
    a = lambda *k: res[k].mean()
    g1 = acc[3] >= 0.30
    best = {l: max(a('main', 0, s, l, 'state') for s in range(len(S))) for l in ('a', 'b', 'c')}
    g2 = all(v >= 0.75 for v in best.values()); g3 = a('bare', 1, E, 'ans', 'state') >= 0.90
    say('G1 word accuracy after loop 4 %.3f (>= 0.30: %s) | G2 copies best state %s (>= 0.75: %s) | G3 bare sum digit at '
        'P1 end4 %.3f (>= 0.90: %s)' % (acc[3], g1, ' '.join('%s %.3f' % kv for kv in best.items()), g2,
                                        a('bare', 1, E, 'ans', 'state'), g3))
    labels = {n: test(n, h, res, its, E, a, say) for n, h in HYP.items()}
    if not (g1 and g2 and g3):
        labels = {n: 'UNSCORABLE' for n in labels}
    say('PRIMARY -> H1 (bare, before writing) %s | H2 (word problems, before the units digit) %s' % (labels['H1'], labels['H2']))
    if secondary:
        say('--- secondary ---')
        for n, h in HYP.items():
            sp, p, t = h['split'], h['pos'], h['t']
            say('%s target excess by state: %s' % (n, ' '.join('%s %+.3f' % (s, a(sp, p, k, t, 'state') - a(sp, p, E, t, 'ceil'))
                                                               for k, s in enumerate(S))))
        say('word P0 end4: m5 excess %+.3f' % (a('main', 0, E, 'm5', 'state') - a('main', 1, E, 'm5', 'ceil')))
        say("H2 at v186's ORIGINAL bars (0.15 / 0.15; not a label) -> %s"
            % test('H2@0.15', HYP['H2'], res, its, E, a, say, 0.15, 0.15))
        for n, h in HYP10.items():
            say('%s mod-10 version (v186-style; not a label) -> %s' % (n, test(n + '-mod10', h, res, its, E, a, say)))
    if verbose:
        print('\n'.join(out), flush=True)
    return labels


def selftest():
    its = json.load(open('v188_items.json'))['items']; N = len(its); d = 64
    rng = np.random.default_rng(0)
    W = {k: rng.normal(size=(10, d)) for k in ('a', 'b', 'c', 'm', 'ans')}
    U, Vv = rng.normal(size=(10, d)), rng.normal(size=(10, d))
    fam = np.array([i['fam'] for i in its]); off = 20 * rng.normal(size=(fam.max() + 1, d))

    def world(kind):
        X = np.zeros((N, 2, 8, d), np.float32)
        for r, i in enumerate(its):
            L = i['labels']; v = off[i['fam']] + 0.05 * rng.normal(size=d)
            if kind != 'broken':
                v = v + W['a'][L['a']] + W['b'][L['b']] + (W['c'][L['c']] if i['set'] == 'main' else 0)
            p0, p1 = v.copy(), v + 3.0 * W['ans'][L['ans']]
            on = kind == 'compute' or (kind.startswith('partial') and rng.random() < float(kind.split('_')[1]))
            if on and i['set'] == 'bare':
                p0 = p0 + W['m'][L['m']]
            if on and i['set'] == 'main':
                p1 = p1 + W['m'][L['m']]
            if kind == 'ans_c_mixing' and i['set'] == 'main':   # generic interaction of the answer digit and c at P1
                p1 = p1 + np.tanh(U[L['ans']] * Vv[L['c']] * 2)
            if kind == 'bare_mixing' and i['set'] == 'bare':    # generic interaction of a and b before writing
                p0 = p0 + np.tanh(U[L['a']] * Vv[L['b']] * 2)
            X[r, 0, :] = p0; X[r, 1, :] = p1
        return X

    meta = dict(states=['mid1', 'end1', 'mid2', 'end2', 'mid3', 'end3', 'mid4', 'end4'],
                items=[dict(i, correct=[True] * 4) for i in its])
    want = {'compute': ({'REPLICATED'}, {'REPLICATED'}), 'partial_0.4': ({'REPLICATED'}, {'REPLICATED'}),
            'partial_0.15': None, 'copy': ({'NOT-REPLICATED'}, {'NOT-REPLICATED'}),
            'ans_c_mixing': ({'NOT-REPLICATED'}, {'NOT-REPLICATED', 'UNCLEAR'}),
            'bare_mixing': ({'NOT-REPLICATED', 'UNCLEAR'}, {'NOT-REPLICATED'}), 'broken': ({'UNSCORABLE'}, {'UNSCORABLE'})}
    ok = True
    for k, w in want.items():
        print('== world %s' % k, flush=True)
        lab = analyse(world(k), meta, secondary=False, verbose=True)
        good = w is None or (lab['H1'] in w[0] and lab['H2'] in w[1])
        print('selftest %-13s -> H1 %-15s H2 %-15s %s' % (k, lab['H1'], lab['H2'], 'recorded (sensitivity)' if w is None
                                                          else 'OK' if good else 'FAIL (want %s)' % (w,)), flush=True)
        ok &= good
    return ok


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        sys.exit(0 if selftest() else 1)
    f = sorted(glob.glob(os.path.join(sys.argv[1], '**', 'v188_states.npy'), recursive=True))[0]
    meta = json.load(open(os.path.join(os.path.dirname(f), 'v188_meta.json')))
    assert {i['set'] for i in meta['items']} == {'main', 'bare'}
    analyse(np.load(f, mmap_mode='r'), meta)
