"""v208 Part B (docs/prereg_v208_input_sensitivity.md): does projecting the network's amplified output directions out of
the states help a single-prompt probe read the digit?

Amplified output directions A_m (per model and decision state): the top-m right singular vectors of the stacked linear
responses J u to v208's 64 isotropic nudges at the last slot, over all 24 v208 prompts (uncentred; a randomized range
finder for the input-embedding-to-last-token Jacobians at the varied digit). m = 16 primary; m = 4 and 64 descriptive.
Baselines with the same m: PC = the top-m principal components of the B2 training pool's states (centred, training
questions only); RND = a random m-dim subspace (seed 208).
Probe: v205's 'full' probe (StandardScaler -> logistic on all dimensions, C in {const, 1e-4 ... 10}, inner 3-fold by
question; v205_analyze.fit_c, unchanged), trained on the whole B2 training pool (5,249 questions) and scored on B2's fixed
1,000 test questions (Qwen: v205 B2; Ouro: v206 B2), on the last-token states with A's span removed, h - A A^T h.
D_X = acc(projected by X) - acc(raw), paired bootstrap over test questions (2,000 resamples, seed 208), 95% interval.
Labels per model over the decision states (qwen l7-l28; ouro end1-end4):
  HELPS    D_J >= 0.05 with lower bound > 0 at >= 3 of 4 states; then SPECIFIC if D_J - D_PC >= 0.03 with lower bound
           > 0 at >= 3 of 4, else GENERIC
  NO-HELP  D_J <= 0.02 with upper bound < 0.05 at >= 3 of 4
  MIXED    otherwise
Amendment 1 (2026-10-18, before any Part B statistic): SPECIFIC additionally requires D_J - D_PCm >= 0.03 with lower
bound > 0 at >= 3 of 4, where PCm = the top-16 PCs of as many training states as there are nudge responses (a random
subsample, seed 208), so the two subspaces are estimated from the same number of vectors. Descriptive: D_J for J
estimated from each half of the prompts, and the split-half stability of both subspaces.
Descriptive: share of B3's digit-mean energy (family-demeaned class means) inside A_16 against 16/d; the B3 minimal-pair
readout after removing A_16; D_J at m = 4 and 64.

  python v208_method.py <model> <dir> [<dir> ...]      model = qwen | ouro; dirs hold v208 outputs and the B2/B3 states
  python v208_method.py --selftest
"""
import glob, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

DECISION = {'qwen': ['l7', 'l14', 'l21', 'l28'], 'ouro': ['end1', 'end2', 'end3', 'end4']}
STEMS = {'qwen': (['v205_b2_s%d' % k for k in range(3)], 'v205_b3'), 'ouro': (['v206_b2_s%d' % k for k in range(4)], 'v206_b3')}
M_PRIMARY, M_DESC, B, SEED = 16, (4, 64), 2000, 208
G = {}


def init(d):
    G.update(d)


def proj_out(X, A):
    return X - (X @ A) @ A.T if A is not None else X


def job(args):
    import v205_analyze as A5
    st, cond = args; X = G['X%d' % st]; A = G['A'].get((st, cond)); tr, te = G['tr'], G['te']
    Xp = proj_out(X, A)
    pred, C = A5.fit_c(Xp[tr], G['y'][tr], G['g'][tr], Xp[te], False, A5.GRIDS['full'])
    return args, ((pred == G['y'][te]).astype(np.int8), C)


def amplified(Y, m):
    return np.linalg.svd(np.asarray(Y, np.float64), full_matrices=False)[2][:m].T.astype(np.float32)


def top_pcs(X, m):
    Xc = np.asarray(X, np.float64) - X.mean(0)
    return np.linalg.svd(Xc, full_matrices=False)[2][:m].T.astype(np.float32)


def rnd(d, m):
    return np.linalg.qr(np.random.default_rng(SEED).standard_normal((d, m)))[0].astype(np.float32)


def boot(a, b, rng):
    diff = a.astype(float) - b.astype(float); n = len(diff)
    bs = diff[rng.integers(0, n, (B, n))].mean(1)
    return float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def label(rows_j, rows_jpc, rows_jpcm=None):
    """Amendment 1: SPECIFIC also needs J16 to beat PC16m (PCs from as many training states as there are responses)."""
    if sum(m >= 0.05 and lo > 0 for m, lo, hi in rows_j) >= 3:
        spec = all(sum(m >= 0.03 and lo > 0 for m, lo, hi in rr) >= 3 for rr in ([rows_jpc] + ([rows_jpcm] if rows_jpcm else [])))
        return 'HELPS (%s)' % ('SPECIFIC' if spec else 'GENERIC')
    if sum(m <= 0.02 and hi < 0.05 for m, lo, hi in rows_j) >= 3:
        return 'NO-HELP'
    return 'MIXED'


def run(model, Yiso, X2, y, g, split, states2, X3=None, y3=None, fam3=None, states3=None, procs=4, say=print):
    """Yiso: {decision state: (n_resp x d)}; X2: items x states2 x d (last slot); split: 'train'/'test' per item."""
    rng = np.random.default_rng(SEED); dec = DECISION[model]; d = X2.shape[-1]
    tr = np.where(split == 'train')[0]; te = np.where(split == 'test')[0]
    arrs = dict(y=y, g=g, tr=tr, te=te, A={}); conds = ['raw', 'J16', 'PC16', 'PC16m', 'RND16', 'J4', 'J64', 'J16a', 'J16b']
    sub = np.random.default_rng(SEED).permutation(tr); stab = {}
    for st in dec:
        k = states2.index(st); X = np.ascontiguousarray(X2[:, k], dtype=np.float32); arrs['X%d' % k] = X
        arrs['A'][(k, 'J16')] = amplified(Yiso[st], 16); arrs['A'][(k, 'J4')] = amplified(Yiso[st], 4)
        arrs['A'][(k, 'J64')] = amplified(Yiso[st], 64); arrs['A'][(k, 'PC16')] = top_pcs(X[tr], 16)
        arrs['A'][(k, 'RND16')] = rnd(d, 16)
        nr = len(Yiso[st]); h = nr // 2                  # amendment 1: matched-count PCs; split-half J (by prompt order)
        arrs['A'][(k, 'PC16m')] = top_pcs(X[sub[:nr]], 16)
        arrs['A'][(k, 'J16a')] = amplified(Yiso[st][:h], 16); arrs['A'][(k, 'J16b')] = amplified(Yiso[st][h:], 16)
        ov = lambda P, Q: float(((P.T @ Q) ** 2).sum() / P.shape[1])
        stab[st] = (ov(arrs['A'][(k, 'J16a')], arrs['A'][(k, 'J16b')]),
                    ov(top_pcs(X[sub[:h]], 16), top_pcs(X[sub[h:nr]], 16)))
    G.update(arrs); jobs = [(states2.index(st), c) for st in dec for c in conds]
    with Pool(procs, initializer=init, initargs=(arrs,)) as pool:
        res = dict(pool.map(job, jobs))
    say('== %s Part B: %d training, %d test questions; full probe; chance 0.10' % (model, len(tr), len(te)))
    rows = {c: [] for c in conds[1:]}; rows['J16-PC16'] = []; rows['J16-PC16m'] = []
    for st in dec:
        k = states2.index(st); raw = res[(k, 'raw')][0]
        line = '%-5s raw %.3f (C %s)' % (st, raw.mean(), res[(k, 'raw')][1])
        for c in conds[1:]:
            r = boot(res[(k, c)][0], raw, rng); rows[c].append(r)
            line += ' | %s %.3f D %+.3f [%+.3f, %+.3f]' % (c, res[(k, c)][0].mean(), *r)
        rows['J16-PC16'].append(boot(res[(k, 'J16')][0], res[(k, 'PC16')][0], rng))
        rows['J16-PC16m'].append(boot(res[(k, 'J16')][0], res[(k, 'PC16m')][0], rng))
        say(line)
    for nm in ('J16-PC16', 'J16-PC16m'):
        say('%-10s ' % nm + '  '.join('%s %+.3f [%+.3f, %+.3f]' % (st, *r) for st, r in zip(dec, rows[nm])))
    say('subspace stability (split halves, mean squared cosine of 16-dim spans): ' +
        '  '.join('%s J %.3f / PC %.3f' % (st, *stab[st]) for st in dec))
    lab = label(rows['J16'], rows['J16-PC16'], rows['J16-PC16m'])
    if X3 is not None:                                   # descriptive: digit energy in A_16, minimal pairs after removal
        import v178_analyze as V
        import v181_swamp_check as S
        for st in dec:
            k2, k3 = states2.index(st), states3.index(st); A = arrs['A'][(k2, 'J16')]
            Xd = S.demean(np.asarray(X3[:, k3], np.float32), fam3)
            M = np.stack([Xd[y3 == c].mean(0) for c in range(10)]); share = float(((M @ A) ** 2).sum() / (M ** 2).sum())
            accs = [float((V.probe_predict(S.demean(proj_out(np.asarray(X3[:, k3], np.float32), AA), fam3), y3, fam3,
                                           V.folds(fam3)) == y3).mean()) for AA in (None, A)]
            say('%-5s B3 digit-mean energy in A_16 %.3f (random %.3f) | minimal-pair readout %.3f -> %.3f after removing A_16'
                % (st, share, 16 / d, *accs))
    say('Part B label (%s): %s' % (model, lab))
    return lab


def main(model, dirs):
    import json
    import v205_analyze as A5
    import v208_analyze as A8
    data = A8.load(dirs)[model]; LIN, FUL, norms, meta = data
    iso = A8.class_slices(meta['classes'])['ISO']; si = meta['slots'].index('last')
    Yiso = {st: LIN[:, iso, si, meta['states'].index(st)].reshape(-1, LIN.shape[-1]) for st in DECISION[model]}
    root = dirs[0]; b2s, b3s = STEMS[model]
    parts = []
    for dd in dirs:
        for s in b2s:
            if glob.glob(os.path.join(dd, '**', s + '_states.npy'), recursive=True):
                parts.append(A5.load(dd, s))
    assert len(parts) == len(b2s), 'missing B2 shards'
    m2 = dict(parts[0][1]); m2['items'] = sum([p[1]['items'] for p in parts], [])
    si2 = m2['slots'].index('last'); X2 = np.concatenate([np.asarray(p[0][:, si2]) for p in parts])
    its = m2['items']; y = np.array([i['digit'] for i in its]); g = np.array([i['idx'] for i in its])
    split = np.array([i['split'] for i in its])
    X3 = y3 = fam3 = st3 = None
    for dd in dirs:
        if glob.glob(os.path.join(dd, '**', b3s + '_states.npy'), recursive=True):
            X3a, m3 = A5.load(dd, b3s); si3 = m3['slots'].index('last'); X3 = np.asarray(X3a[:, si3])
            y3 = np.array([i['digit'] for i in m3['items']]); fam3 = np.array([i['idx'] for i in m3['items']]); st3 = m3['states']
    run(model, Yiso, X2, y, g, split, m2['states'], X3, y3, fam3, st3)


def selftest():
    """Synthetic, with the shift that defeats single-prompt probes: training contexts vary along U1 only, test contexts
    along U1 and U2 (directions the training set never shows); the digit code lies in C, orthogonal to both.
      'specific': the nudge responses span U1 + U2 -> removing them helps, and beats the training-set PCs (which only
                  see U1) -> HELPS (SPECIFIC).
      'nohelp':   the responses span the digit code -> removing them deletes the signal -> NO-HELP.
    The GENERIC and MIXED branches of the label rule are checked on fixed rows."""
    rng = np.random.default_rng(0); d, n = 32, 1500; ok = True
    for world, want in (('specific', 'HELPS (SPECIFIC)'), ('nohelp', 'NO-HELP')):
        Q = np.linalg.qr(rng.standard_normal((d, d)))[0]; U1, U2, C = Q[:, :8], Q[:, 8:16], Q[:, 16:26]
        y = rng.integers(0, 10, n); code = rng.standard_normal((10, 10)) @ C.T; te = np.arange(n) >= 1200
        ctx = (rng.standard_normal((n, 8)) * 3) @ U1.T + (rng.standard_normal((n, 8)) * 8 * te[:, None]) @ U2.T
        X = ctx + 0.6 * code[y] + 0.5 * rng.standard_normal((n, d))
        X2 = np.repeat(X[:, None], 4, 1).astype(np.float32)
        span = {'specific': np.hstack([U1, U2]), 'nohelp': C}[world]
        Yiso = {st: (rng.standard_normal((300, span.shape[1])) @ span.T) for st in DECISION['qwen']}
        split = np.where(te, 'test', 'train')
        got = run('qwen', Yiso, X2, y, np.arange(n), split, DECISION['qwen'], say=lambda *a: None, procs=2)
        good = got == want; ok &= good
        print('selftest world %-6s want %s got %s %s' % (world, want, got, 'OK' if good else 'FAIL'))
    hi_, lo_ = [(0.08, 0.04, 0.12)] * 4, [(0.01, -0.02, 0.04)] * 4
    for rows_j, rows_jpc, rows_jpcm, want in ((hi_, lo_, lo_, 'HELPS (GENERIC)'),
                                              (hi_, hi_, lo_, 'HELPS (GENERIC)'),     # beats PC16 only: not SPECIFIC
                                              (hi_, hi_, hi_, 'HELPS (SPECIFIC)'),
                                              ([(0.03, 0.0, 0.06)] * 4, lo_, lo_, 'MIXED'),
                                              (hi_[:2] + [(0.0, -0.03, 0.03)] * 2, lo_, lo_, 'MIXED')):
        got = label(rows_j, rows_jpc, rows_jpcm); good = got == want; ok &= good
        print('selftest label rule want %s got %s %s' % (want, got, 'OK' if good else 'FAIL'))
    print('SELFTEST', 'ALL OK' if ok else 'FAILED')
    return ok


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        sys.exit(0 if selftest() else 1)
    main(sys.argv[1], sys.argv[2:])
