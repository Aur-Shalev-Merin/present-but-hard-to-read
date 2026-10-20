"""v212: a calibrated null for the protocol's mixture tests (docs/prereg_v212_mixture_null.md; outside review W3).

Synthetic P1 states on Study 3's real items (v194_items.json), built as v194's selftest builds them but with NO
a+b term: a family offset, copies of a, b, c and a strong answer code (the state Ouro is about to write), plus one
generic nonlinear mixture of (a, b, c, answer) units digits:
  pair    sum over the 6 pairs (x, y) of tanh(2 U_x[x] * V_y[y])               (elementwise products of random codes)
  mlp     R2 relu(R1 [onehot a; b; c; ans] + r0), 256 hidden units             (a random one-layer network)
  fourier R . (same-frequency products of cos/sin(2 pi f x / 10) across pairs), f in {1, 2, 5}
The mixture is scaled to lam times the RMS norm of the answer code. Statistics use v194's probe jobs (v178 probe on
family-demeaned states, mod-5 labels, copy ceiling = one-hot a, b, c):
  needed    ex(needed form) (a+b on plus, a-b on minus), mean of the two versions
  line      ex(needed) - max over the other three forms on its line answer + k c (per version)
  I         (D+ + D-)/2, D+ = ex(a+b) - ex(a-b) on plus items, D- = ex(a-b) - ex(a+b) on minus items

Two stages (prereg section 2):
  calib  seed 100, LAMS_CAL, needed only -> per family the matched lam* (log-linear interpolation to OBS['needed'])
  run    seeds SEEDS at lam*, full statistics

  python v212_mixture_null.py calib <family> [<lam> ...]            -> v212cal_<family>_<lam>.json in OUT_DIR
  python v212_mixture_null.py match <dir with v212cal_*.json>        -> lam* per family (frozen rule)
  python v212_mixture_null.py run <family> <lam> <seed> [<seed> ...]  -> v212_<family>_<lam>_<seed>.json
  python v212_mixture_null.py analyse <dir>                          -> the pre-registered reading (section 3)
  python v212_mixture_null.py selftest
"""
import glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np

import v194_analyze as S3

D = 64
FAMILIES = ('pair', 'mlp', 'fourier')
LAMS_CAL = (0.005, 0.01, 0.02, 0.04, 0.08)
CAL_SEED = 100
SEEDS = (0, 1, 2, 3, 4, 5)
OBS = dict(needed=(0.207 + 0.267) / 2, line=(0.036, 0.045), I=0.157)   # Study 3, P1 end4 (prereg_v194 sections 10-11)
CEIL = {}


def items():
    return [i for i in json.load(open('v194_items.json'))['items'] if i['set'] in ('main', 'bare')]


def world(its, family, lam, seed, extra=None):
    """N x 2 x 1 x D states (slot 1 = P1). extra='follows' adds v194's explicit needed-form code (selftest only)."""
    rng = np.random.default_rng(seed)
    W = {k: rng.normal(size=(10, D)) for k in ('a', 'b', 'c', 'ans', 'ab', 'amb')}
    fam = np.array([i['fam'] for i in its]); off = 20 * rng.normal(size=(fam.max() + 1, D))
    keys = ('a', 'b', 'c', 'ans'); pairs = [(x, y) for n, x in enumerate(keys) for y in keys[n + 1:]]
    if family == 'pair':
        UV = {p: (rng.normal(size=(10, D)), rng.normal(size=(10, D))) for p in pairs}
    elif family == 'mlp':
        R1, r0, R2 = rng.normal(size=(256, 40)), rng.normal(size=256), rng.normal(size=(D, 256)) / 16
    elif family == 'fourier':
        nf = len(pairs) * 3 * 4; R = rng.normal(size=(D, nf)) / np.sqrt(nf)
    X = np.zeros((len(its), 2, 1, D), np.float32)
    lab = lambda i, k: i['labels'][k]
    M = np.zeros((len(its), D))
    for r, i in enumerate(its):
        if i['set'] != 'main':
            continue
        if family == 'pair':
            M[r] = sum(np.tanh(2 * UV[p][0][lab(i, p[0])] * UV[p][1][lab(i, p[1])]) for p in pairs)
        elif family == 'mlp':
            z = np.zeros(40)
            for j, k in enumerate(keys):
                z[10 * j + lab(i, k)] = 1
            M[r] = R2 @ np.maximum(R1 @ z + r0, 0)
        elif family == 'fourier':
            ph = {k: {f: (np.cos(2 * np.pi * f * lab(i, k) / 10), np.sin(2 * np.pi * f * lab(i, k) / 10)) for f in (1, 2, 5)}
                  for k in keys}
            feats = [u * v for (x, y) in pairs for f in (1, 2, 5) for u in ph[x][f] for v in ph[y][f]]
            M[r] = R @ np.array(feats)
    main = np.array([i['set'] == 'main' for i in its])
    rms_ans = np.sqrt(((3.0 * W['ans']) ** 2).sum(1).mean())
    rms_m = np.sqrt((M[main] ** 2).sum(1).mean()) if family in FAMILIES else 1.0
    M *= lam * rms_ans / max(rms_m, 1e-12)
    for r, i in enumerate(its):
        L = i['labels']; v = off[i['fam']] + 0.05 * rng.normal(size=D)
        v = v + W['a'][L['a']] + W['b'][L['b']] + (W['c'][L['c']] if i['set'] == 'main' else 0)
        p1 = v + 3.0 * W['ans'][L['ans']] + M[r]
        if extra == 'follows' and i['set'] == 'main':
            p1 = p1 + lam * (W['ab'][(L['a'] + L['b']) % 10] if i['op'] == 'plus' else W['amb'][(L['a'] - L['b']) % 10])
        X[r, 0, 0] = v; X[r, 1, 0] = p1
    return X


def stats(X, its, full=True, procs=4):
    if full:
        forms = {op: list(dict.fromkeys(S3.LINE[op] + [S3.AB, S3.AMB])) for op in ('plus', 'minus')}
    else:
        forms = {'plus': [S3.AB], 'minus': [S3.AMB]}
    want = [('main', op, 1, 0, f, x) for op in forms for f in forms[op] for x in ('state', 'ceil')]
    jobs = [j for j in want if not (j[-1] == 'ceil' and j in CEIL)]     # the copy ceiling does not depend on the world
    with Pool(procs, initializer=S3.init, initargs=(X, its)) as pool:
        res = dict(pool.map(S3.job, jobs))
    for j in want:
        if j[-1] == 'ceil':
            CEIL.setdefault(j, res.get(j))
            res[j] = CEIL[j]
    ex = {(op, f): res[('main', op, 1, 0, f, 'state')].astype(float) - res[('main', op, 1, 0, f, 'ceil')]
          for op in forms for f in forms[op]}
    out = dict(needed_plus=ex[('plus', S3.AB)].mean(), needed_minus=ex[('minus', S3.AMB)].mean())
    out['needed'] = (out['needed_plus'] + out['needed_minus']) / 2
    if not full:
        return out
    sel = lambda op: [i for i in its if i['set'] == 'main' and i['op'] == op]
    dp = ex[('plus', S3.AB)] - ex[('plus', S3.AMB)]; dm = ex[('minus', S3.AMB)] - ex[('minus', S3.AB)]
    pp = np.array([i['pair'] for i in sel('plus')]); pm = np.array([i['pair'] for i in sel('minus')])
    (I, Ilo, Ihi), (F, _, _) = S3.pair_ci(pp, dp, pm, dm, n=2000)
    out.update(I=I, I_lo=Ilo, I_hi=Ihi, F=F)
    for op in ('plus', 'minus'):
        L = S3.LINE[op]; m = {f: ex[(op, f)].mean() for f in L}
        out['line_' + op] = m[L[0]] - max(m[f] for f in L[1:])
        out['line_forms_' + op] = [round(m[f], 4) for f in L]
    return out


def match(rows):
    """Frozen rule: lam* = log-linear interpolation of the calibration curve (needed vs log lam, one seed) to
    OBS['needed'] between the two bracketing grid points; None if the curve never brackets it."""
    lam = np.array([r['lam'] for r in rows]); nd = np.array([r['needed'] for r in rows]); o = np.argsort(lam)
    lam, nd = lam[o], nd[o]
    for k in range(len(lam) - 1):
        lo, hi = sorted((nd[k], nd[k + 1]))
        if lo <= OBS['needed'] <= hi and nd[k + 1] != nd[k]:
            t = (OBS['needed'] - nd[k]) / (nd[k + 1] - nd[k])
            return float('%.3g' % np.exp(np.log(lam[k]) + t * (np.log(lam[k + 1]) - np.log(lam[k]))))
    return None


def calib(family, lams):
    its = items(); od = os.environ.get('OUT_DIR', '.')
    for lam in lams:
        t0 = time.time(); r = stats(world(its, family, lam, CAL_SEED), its, full=False)
        r.update(family=family, lam=lam, seed=CAL_SEED, wall_s=round(time.time() - t0))
        json.dump(r, open(os.path.join(od, 'v212cal_%s_%s.json' % (family, lam)), 'w'))
        print('calib %s lam %s | needed %+.3f (plus %+.3f, minus %+.3f) | %ds' % (
            family, lam, r['needed'], r['needed_plus'], r['needed_minus'], r['wall_s']), flush=True)


def run(family, lam, seeds):
    its = items(); od = os.environ.get('OUT_DIR', '.')
    for sd in seeds:
        t0 = time.time(); r = stats(world(its, family, lam, sd), its)
        r.update(family=family, lam=lam, seed=sd, wall_s=round(time.time() - t0))
        json.dump(r, open(os.path.join(od, 'v212_%s_%s_%d.json' % (family, lam, sd)), 'w'))
        print('%s lam %s seed %d | needed %+.3f | line %+.3f / %+.3f | I %+.3f | %ds' % (
            family, lam, sd, r['needed'], r['line_plus'], r['line_minus'], r['I'], r['wall_s']), flush=True)


def analyse(rdir, say=print):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(rdir, '**', 'v212_*.json'), recursive=True)]
    labels = {}
    for fam in FAMILIES:
        g = [r for r in rows if r['family'] == fam]
        if not g:
            continue
        null = [r['line_plus'] for r in g] + [r['line_minus'] for r in g]
        if min(OBS['line']) > max(null):
            lab = 'SEPARABLE'
        elif np.mean(null) >= np.mean(OBS['line']):
            lab = 'NOT-SEPARABLE'
        else:
            lab = 'PARTLY'
        labels[fam] = lab
        say('%-7s lam* %s, %d worlds | needed %+.3f (observed %+.3f) | null line margins %s | mean %+.3f max %+.3f vs '
            'observed %+.3f / %+.3f -> %s | null I mean %+.3f max %+.3f vs observed %+.3f' % (
                fam, sorted({r['lam'] for r in g}), len(g), np.mean([r['needed'] for r in g]), OBS['needed'],
                ' '.join('%+.3f' % v for v in null), np.mean(null), max(null), *OBS['line'], lab,
                np.mean([r['I'] for r in g]), max(r['I'] for r in g), OBS['I']))
    if labels:
        order = ['NOT-SEPARABLE', 'PARTLY', 'SEPARABLE']
        say('v212 line test against generic mixing -> %s (worst family)' % min(labels.values(), key=order.index))
    return labels


def selftest():
    """A world with v194's explicit needed-form code must give a clearly positive line margin; pure Fourier mixing
    (the reviewer's alternative) must not; the matching rule must interpolate a toy curve correctly."""
    its = items(); ok = True
    for name, fam, lam, extra in (('follows', 'none', 1.0, 'follows'), ('fourier', 'fourier', 0.8, None)):
        r = stats(world(its, fam, lam, 99, extra=extra), its)   # seed 99: neither a calibration nor a run seed
        line = (r['line_plus'] + r['line_minus']) / 2
        good = line > 0.05 if name == 'follows' else line < 0.03
        print('selftest %-8s needed %+.3f | line %+.3f / %+.3f | I %+.3f -> %s' % (
            name, r['needed'], r['line_plus'], r['line_minus'], r['I'], 'OK' if good else 'FAIL'), flush=True)
        ok &= good
    toy = [dict(lam=l, needed=n) for l, n in ((0.01, 0.05), (0.02, 0.137), (0.04, 0.337), (0.08, 0.7))]
    m = match(toy); good = m is not None and abs(m - 0.02 * 2 ** 0.5) < 1e-3
    print('selftest match   toy curve -> lam* %s (want %.4f) %s' % (m, 0.02 * 2 ** 0.5, 'OK' if good else 'FAIL'), flush=True)
    return ok and good


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'calib':
        calib(sys.argv[2], [float(x) for x in sys.argv[3:]] or LAMS_CAL)
    elif cmd == 'match':
        rows = [json.load(open(f)) for f in glob.glob(os.path.join(sys.argv[2], '**', 'v212cal_*.json'), recursive=True)]
        print(json.dumps({fam: match([r for r in rows if r['family'] == fam]) for fam in FAMILIES}))
    elif cmd == 'run':
        run(sys.argv[2], float(sys.argv[3]), [int(s) for s in sys.argv[4:]])
    elif cmd == 'selftest':
        sys.exit(0 if selftest() else 1)
    else:
        analyse(sys.argv[2])
