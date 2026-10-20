"""v202 analysis (docs/prereg_v202_qwen_swamping.md): Sections 3.2-3.3 of the Ouro note in Qwen2.5-1.5B.

Everything reuses the Ouro code unchanged: the v178 probe (V.probe_predict; folds grouped by template, V.folds), the
v181 contrastive (family-demeaned) readout, and v181_swamp_check (overlap_lda, lc_job: structured decoder, Gaussian
ceiling, learning curve). States: h0, l4, l7, l14, l21, l28 (Ouro's h0, mid1, end1-end4). Q = l7, l14, l21, l28.

Gates (any failure -> UNSCORABLE): G1 minimal pairs ok in >= 99% of prompts in each design; G2 the standard probe reads
the varied digit on its own token (+0 slot, l7) at >= 0.90 in the v181 design; G3 the extraction's cached-vs-full
forward check < 1e-3 (the extraction aborts otherwise).
PRIMARY (v181 design, last prompt token, over Q): c = mean contrastive accuracy, s = max standard accuracy, lda = max
structured single-prompt decoder, gain = mean over Q of standard(100% of training templates) - standard(50%).
  ABSENT      contrastive < 0.30 at every Q layer
  READABLE    s >= 0.50 or lda >= 0.60                         (not swamped in a standard model)
  REPLICATES  c >= 0.80 and s <= 0.30 and lda <= 0.40 and gain <= 0.05   (present but swamped, as in Ouro)
  MIXED       otherwise
Secondary (descriptive): Q1 within/between spread and Q2 standard/contrastive at every slot and state; overlap O and
signal/noise; the last-token profile over all 29 depths; v185 design: standard and contrastive for a colour word and an
added digit, and O(colour)/O(digit).

  python v202_analyze.py <dir with v202_* files>
  python v202_analyze.py --selftest
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V
import v181_swamp_check as S

Q = ['l7', 'l14', 'l21', 'l28']
FRACS = (0.25, 0.5, 0.75, 1.0)
G = {}


def demean(X, fam):
    return S.demean(X, fam)


def init(arrs):
    G.update(arrs)


def probe_job(args):
    key, sl, contrastive = args          # sl: tuple index into G[key]
    X = np.asarray(G[key][(slice(None),) + sl], dtype=np.float32)
    y, fam, tid = G[key + '_y'], G[key + '_fam'], G[key + '_tid']
    if contrastive:
        X = demean(X, fam)
    return args, float((V.probe_predict(X, y, tid, V.folds(tid)) == y).mean())


def spread(X, fam):
    u = np.unique(fam); M = np.stack([X[fam == f].mean(0) for f in u]); g = M.mean(0)
    within = np.mean([np.linalg.norm(X[fam == f] - M[i], axis=1).mean() for i, f in enumerate(u)])
    return float(within / np.linalg.norm(M - g, axis=1).mean())


def load(root, design):
    f = sorted(glob.glob(os.path.join(root, '**', 'v202_%s_states.npy' % design), recursive=True))[0]; d = os.path.dirname(f)
    meta = json.load(open(os.path.join(d, 'v202_%s_meta.json' % design)))
    return np.load(f, mmap_mode='r'), np.load(os.path.join(d, 'v202_%s_last_all.npy' % design), mmap_mode='r'), meta


def label(c_by, s_by, lda_by, gain):
    c = np.mean([c_by[q] for q in Q]); s = max(s_by[q] for q in Q); lda = max(lda_by[q] for q in Q)
    if all(c_by[q] < 0.30 for q in Q):
        lab = 'ABSENT'
    elif s >= 0.50 or lda >= 0.60:
        lab = 'READABLE'
    elif c >= 0.80 and s <= 0.30 and lda <= 0.40 and gain <= 0.05:
        lab = 'REPLICATES'
    else:
        lab = 'MIXED'
    return lab, c, s, lda


def analyse(X, XL, meta, M185=None, procs=4, say=print):
    its = meta['items']; states = meta['states']; slots = meta['slots']; si = slots.index('last')
    y = np.array([i['digit'] for i in its]); fam = np.array([i['fam'] for i in its]); tid = np.array([i['tid'] for i in its])
    ok = np.mean([i['pair_ok'] for i in its]); sts = [k for k, s in enumerate(states) if s != 'h0']
    say('v181 design: %d prompts, %d families, %d templates | minimal pairs ok %.4f | cached-vs-full check %.2e' % (
        len(its), len(set(fam)), len(set(tid)), ok, meta.get('check', 0)))
    arrs = dict(a=X, a_y=y, a_fam=fam, a_tid=tid, b=XL, b_y=y, b_fam=fam, b_tid=tid)
    jobs = [('a', (sl, st), c) for sl in range(len(slots)) for st in range(len(states)) for c in (False, True)
            if not (st == 0 and slots[sl] != '+0')]
    jobs += [('a', (0, states.index('l7')), False)]
    jobs += [('b', (li,), c) for li in range(XL.shape[1]) for c in (False, True)]
    if M185 is not None:
        X5, its5 = M185
        typ = np.array(['%s_%s' % (i['kind'], i['where']) for i in its5]); s5 = 2
        for t in ('col_start', 'col_end', 'dig_start', 'dig_end'):
            idx = np.where(typ == t)[0]
            arrs.update({t: X5[idx][:, s5], t + '_y': np.array([its5[i]['digit'] for i in idx]),
                         t + '_fam': np.array([its5[i]['fam'] for i in idx]), t + '_tid': np.array([its5[i]['tid'] for i in idx])})
            jobs += [(t, (st,), c) for st in sts for c in (False, True)]
    jobs = list(dict.fromkeys(jobs))
    with Pool(procs, initializer=init, initargs=(arrs,)) as pool:
        res = dict(pool.map(probe_job, jobs))
    fold = V.folds(tid)
    lcj = [(st, f_, fr, dr, kind) for st in sts for f_ in np.unique(fold) for fr in FRACS
           for dr in (range(3) if fr < 1 else range(1)) for kind in ('std', 'mp')]
    with Pool(procs, initializer=S.init, initargs=(X, si, y, fam, tid, fold)) as pool:
        lco = dict(pool.map(S.lc_job, lcj))
    curve = {(st, kind, fr): float(np.mean([sum(lco[(st, f_, fr, dr, kind)][0] for f_ in np.unique(fold)) / len(y)
                                            for dr in (range(3) if fr < 1 else range(1))]))
             for st in sts for kind in ('std', 'mp') for fr in FRACS}
    ov = {st: S.overlap_lda(np.asarray(X[:, si, st], dtype=np.float64), y, fam, fold) for st in sts}
    g2 = res[('a', (0, states.index('l7')), False)]
    say('G1 minimal pairs ok %.4f (>= 0.99: %s) | G2 standard probe on the digit\'s own token at l7 %.3f (>= 0.90: %s)' % (
        ok, ok >= 0.99, g2, g2 >= 0.90))
    say('\nQ1 within-family / between-family spread\nslot   ' + ' '.join('%8s' % s for s in states))
    for sl, name in enumerate(slots):
        say('%-6s ' % name + ' '.join('%8.4f' % spread(np.asarray(X[:, sl, st], dtype=np.float32), fam) if not (
            st == 0 and name != '+0') else '%8s' % '-' for st in range(len(states))))
    say('\nQ2 probe accuracy for the varied digit (chance 0.10): standard / contrastive\nslot   ' +
        ' '.join('%15s' % s for s in states))
    for sl, name in enumerate(slots):
        say('%-6s ' % name + ' '.join('%6.3f / %6.3f' % (res[('a', (sl, st), False)], res[('a', (sl, st), True)])
                                       if ('a', (sl, st), False) in res else '%15s' % '(same token)' for st in range(len(states))))
    say('\nSwamping at the last prompt token (as Table 2 of the note)')
    say('state  overlap  SNR-in-subspace  structured-decoder  Gaussian-ceiling  standard 25/50/75/100%  minimal-pair 25/50/75/100%')
    for st in sts:
        r = ov[st]
        say('%-6s %7.2f  %15.4f  %18.3f  %16.3f  %s  %s' % (states[st], r['O'], r['snrU'], r['lda'], r['mc'],
            ' '.join('%.3f' % curve[(st, 'std', fr)] for fr in FRACS), ' '.join('%.3f' % curve[(st, 'mp', fr)] for fr in FRACS)))
    say('\nLast prompt token over all depths (layer: standard / contrastive)')
    say(' '.join('%d: %.2f/%.2f' % (li, res[('b', (li,), False)], res[('b', (li,), True)]) for li in range(XL.shape[1])))
    if M185 is not None:
        say('\nv185 design, last prompt token: standard / contrastive, and overlap O')
        Ov = {}
        for t in ('col_start', 'col_end', 'dig_start', 'dig_end'):
            Xt = np.asarray(arrs[t], dtype=np.float64)
            for st in sts:
                Ov[(t, st)] = S.overlap_lda(Xt[:, st], arrs[t + '_y'], arrs[t + '_fam'], V.folds(arrs[t + '_tid']))['O']
            say('%-9s ' % t + ' '.join('%s %.3f/%.3f O %.2f' % (states[st], res[(t, (st,), False)], res[(t, (st,), True)],
                                                                 Ov[(t, st)]) for st in sts))
        oc = np.mean([Ov[(t, states.index(q))] for t in ('col_start', 'col_end') for q in Q])
        od = np.mean([Ov[(t, states.index(q))] for t in ('dig_start', 'dig_end') for q in Q])
        say('O colour %.2f, digit %.2f, ratio %.3f (Ouro: 5.37, 9.75, 0.551)' % (oc, od, oc / od))
    qi = {q: states.index(q) for q in Q}
    c_by = {q: res[('a', (si, qi[q]), True)] for q in Q}; s_by = {q: res[('a', (si, qi[q]), False)] for q in Q}
    lda_by = {q: ov[qi[q]]['lda'] for q in Q}
    gain = float(np.mean([curve[(qi[q], 'std', 1.0)] - curve[(qi[q], 'std', 0.5)] for q in Q]))
    lab, c, s, lda = label(c_by, s_by, lda_by, gain)
    if not (ok >= 0.99 and g2 >= 0.90 and meta.get('check', 0) < 1e-3):
        lab = 'UNSCORABLE'
    say('\nPRIMARY (last prompt token, %s): contrastive mean %.3f | standard max %.3f | structured decoder max %.3f | '
        'standard gain 50->100%% templates %+.3f -> %s' % ('/'.join(Q), c, s, lda, gain, lab))
    return lab


def selftest():
    """Synthetic states on the real v181 family/template structure, through the real probe and swamp code."""
    import re
    items = json.load(open('v178_labels.json'))['items']
    sys.path.insert(0, '.')
    from v202_extract import families
    fams = families(items); rng = np.random.default_rng(0); d = 96
    code = rng.standard_normal((10, d)); code -= code.mean(0)
    want = {'swamped': 'REPLICATES', 'readable': 'READABLE', 'absent': 'ABSENT'}
    for world, (amp, ctx) in {'swamped': (0.25, 4.0), 'readable': (3.0, 1.0), 'absent': (0.0, 4.0)}.items():
        its, rows = [], []
        tmpl = {t: rng.standard_normal((4, d)) for t in {f['tid'] for f in fams}}   # template-specific directions
        for fi, f in enumerate(fams):
            base = ctx * (tmpl[f['tid']].T @ rng.standard_normal(4)) + ctx * 0.5 * rng.standard_normal(d)
            base += (ctx / 4) * (rng.standard_normal(10) @ code)          # context also varies along the digit's own code
            for dg in range(10):
                v = base + amp * code[dg] + 0.05 * rng.standard_normal(d)
                rows.append(np.stack([np.stack([v + (5 * code[dg] if sl == 0 else 0)] * 6) for sl in range(6)]))
                its.append(dict(fam=fi, tid=f['tid'], digit=dg, pair_ok=True))
        X = np.array(rows, np.float32); XL = X[:, 5, :, :][:, [0, 1, 2, 3, 4, 5]]
        meta = dict(items=its, states=['h0', 'l4', 'l7', 'l14', 'l21', 'l28'], slots=['+0', '+1', '+3', 'sent', 'qmark', 'last'],
                    check=0.0)
        out = []
        lab = analyse(X, XL, meta, procs=4, say=out.append)
        print('selftest %-9s -> %-10s (want %s) %s' % (world, lab, want[world], 'OK' if lab == want[world] else 'FAIL'))
        print('   ' + [l.strip() for l in out if 'PRIMARY' in l][0])


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        selftest()
    else:
        root = sys.argv[1]
        X, XL, meta = load(root, 'v181')
        M5 = None
        if glob.glob(os.path.join(root, '**', 'v202_v185_states.npy'), recursive=True):
            X5, _, m5 = load(root, 'v185'); M5 = (X5, m5['items'])
        analyse(X, XL, meta, M5)
