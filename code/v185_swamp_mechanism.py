"""Why is the copied digit swamped? Mechanism check (post hoc; reading rule in docs/v181_result.md, fixed before running).

  M1 overlap by token type (v185): O (v181_swamp_check.overlap_lda, unchanged) for the colour and digit families on the
     same 77 contexts, last prompt token. r = mean O(colour) / mean O(digit) over end1-end4 and both insertion points.
  M2 regression (v185 digit families, both insertion points pooled; v181): per fold, the held-out families' mean states
     projected on the digit subspace U fitted on the training families, regressed (ridge, alpha 1, fitted on the
     training families) on the counts of each units digit over the question's other integers (primary) or over all its
     digit characters except the varied one (secondary). Out-of-sample R^2 pooled over the 9 dimensions, training mean
     as baseline, fold average weighted by size. Controls: 20 random orthonormal 9-dim subspaces (seed 185), and (v185)
     the colour subspace fitted the same way from the colour families.
  M3 (descriptive) Study 3 copies a, b, c at P0, every state, per version (v194_analyze.job, unchanged).

  python v185_swamp_mechanism.py v185 <dir with v185_states.npy, v185_meta.json>     (needs v178_labels.json in cwd)
  python v185_swamp_mechanism.py v181 <dir with v181_states.npy, v181_meta.json>     (needs v178_labels.json in cwd)
  python v185_swamp_mechanism.py v194 <dir with v194_states.npy, v194_meta.json>
  python v185_swamp_mechanism.py selftest
"""
import glob, json, os, re, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

import v178_analyze as V
import v181_swamp_check as S

NUM = r'\d+(?:,\d{3})*(?:\.\d+)?'
TYPES = ['col_start', 'col_end', 'dig_start', 'dig_end']


def families(items, seed=181):
    """v181_extract.families, copied (that module imports torch at load)."""
    rng = np.random.default_rng(seed); out = []
    for t in sorted({i['tid'] for i in items}):
        its = [i for i in items if i['tid'] == t]; it = its[int(rng.integers(len(its)))]; q = it['question']
        ms = [m for m in re.finditer(NUM, q) if '.' not in m.group() and len(m.group().replace(',', '')) >= 2]
        if not ms:
            continue
        for nm, m in (('first', ms[0]), ('last', ms[-1])):
            if nm == 'last' and m.start() == ms[0].start():
                continue
            out.append(dict(tid=t, inst=it['inst'], anchor=nm, question=q, cu=m.end() - 1))
    return out


def bases(items):
    """v185_extract.bases, copied: one question per template, preferring the 'last' anchor."""
    out = {}
    for f in families(items):
        if f['tid'] not in out or f['anchor'] == 'last':
            out[f['tid']] = f
    return [out[t] for t in sorted(out)]


def unit_counts(q, skip_cu=None):
    """Count of each units digit over the integers in q (integer part's last digit), leaving out the number ending at
    skip_cu."""
    c = np.zeros(10)
    for m in re.finditer(NUM, q):
        ip = m.group().split('.')[0].replace(',', '')
        if skip_cu is not None and m.start() + len(m.group().split('.')[0]) - 1 == skip_cu:
            continue
        c[int(ip[-1])] += 1
    return c


def digit_counts(q, skip_cu=None):
    c = np.zeros(10)
    for j, ch in enumerate(q):
        if ch.isdigit() and j != skip_cu:
            c[int(ch)] += 1
    return c


def fit_U(X, y, fam):
    Xc = S.demean(X, fam); mu = np.stack([Xc[y == k].mean(0) for k in range(10)]); mu -= mu.mean(0)
    return np.linalg.svd(mu, full_matrices=False)[2][:9].T


def fam_table(X, y, fam):
    """Per family: mean state (rows ordered by np.unique(fam))."""
    u = np.unique(fam)
    return u, np.stack([X[fam == f].mean(0) for f in u])


def r2_job(F, Z, ftid, fold_of_fam, U_of_fold, alpha=1.0):
    """Out-of-sample R^2 of the projection of family means on U (per fold), from features Z; weighted fold average."""
    r2, w = [], []
    for f in sorted(U_of_fold):
        tr, te = fold_of_fam != f, fold_of_fam == f
        U = U_of_fold[f]; m = F[tr].mean(0)
        Ptr, Pte = (F[tr] - m) @ U, (F[te] - m) @ U
        zm = Z[tr].mean(0); A = Z[tr] - zm
        W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (Ptr - Ptr.mean(0)))
        pred = (Z[te] - zm) @ W + Ptr.mean(0)
        r2.append(1 - ((Pte - pred) ** 2).sum() / ((Pte - Ptr.mean(0)) ** 2).sum()); w.append(te.sum())
    return float(np.average(r2, weights=w))


def m2(Xd, yd, famd, tidd, Zs, Xc=None, yc=None, famc=None, tidc=None, n_rand=20, seed=185):
    """Xd: digit families' states at one slot/state. Zs: {feature set: (n_fam, 10)} rows ordered by np.unique(famd).
    Returns R^2 for U (digit), the random subspaces and, if colour states are given, the colour subspace."""
    fold = V.folds(tidd); u, F = fam_table(Xd, yd, famd)
    ftid = np.array([tidd[famd == f][0] for f in u]); fold_of_fam = np.array([fold[famd == f][0] for f in u])
    Ud = {f: fit_U(Xd[fold != f], yd[fold != f], famd[fold != f]) for f in np.unique(fold)}
    rng = np.random.default_rng(seed); d = Xd.shape[1]
    R = [np.linalg.qr(rng.standard_normal((d, 9)))[0] for _ in range(n_rand)]
    out = {}
    for name, Z in Zs.items():
        out[(name, 'U')] = r2_job(F, Z, ftid, fold_of_fam, Ud)
        rr = [r2_job(F, Z, ftid, fold_of_fam, {f: Rk for f in Ud}) for Rk in R]
        out[(name, 'rand_mean')], out[(name, 'rand_p95')] = float(np.mean(rr)), float(np.percentile(rr, 95))
        if Xc is not None:
            foldc = V.folds(tidc)
            Uc = {f: fit_U(Xc[foldc != f], yc[foldc != f], famc[foldc != f]) for f in np.unique(foldc)}
            out[(name, 'col')] = r2_job(F, Z, ftid, fold_of_fam, Uc)
    return out


def load(root, stem):
    f = max(glob.glob(os.path.join(root, '**', stem + '_states.npy'), recursive=True), key=os.path.getsize)   # merged set
    meta = json.load(open(os.path.join(os.path.dirname(f), stem + '_meta.json')))
    X = np.load(f, mmap_mode='r'); assert len(X) == len(meta['items']), (f, X.shape)
    print('states:', os.path.relpath(f, root), X.shape, flush=True)
    return X, meta


def ends_mean(res, sts, states, key):
    return float(np.mean([res[st][key] for st in sts if states[st].startswith('end')]))


def run_v185(root):
    X, meta = load(root, 'v185'); its = meta['items']; states = meta['states']; si = meta['slots'].index('last')
    typ = np.array(['%s_%s' % (i['kind'], i['where']) for i in its]); y = np.array([i['digit'] for i in its])
    fam = np.array([i['fam'] for i in its]); tid = np.array([i['tid'] for i in its])
    sts = [k for k, s in enumerate(states) if s != 'h0']
    bq = {b['tid']: b['question'] for b in bases(json.load(open('v178_labels.json'))['items'])}
    assert set(tid) == set(bq), 'base questions do not match the v185 templates'
    print('v185 last prompt token: %d items, %d families, %d templates, d = %d' % (len(its), len(set(fam)), len(set(tid)),
                                                                                 X.shape[-1]), flush=True)
    print('--- M1: overlap O by token type (colour vs digit, same contexts) ---')
    O = {}
    for t in TYPES:
        idx = np.where(typ == t)[0]; fold = V.folds(tid[idx])
        for st in sts:
            r = S.overlap_lda(np.asarray(X[idx, si, st], dtype=np.float64), y[idx], fam[idx], fold); O[(t, st)] = r
            print('%-9s %-5s O %6.2f | SNR in subspace %.3f | structured LDA %.3f | Gaussian ceiling %.3f' % (
                t, states[st], r['O'], r['snrU'], r['lda'], r['mc']), flush=True)
    E = [st for st in sts if states[st].startswith('end')]
    oc = float(np.mean([O[(t, st)]['O'] for t in ('col_start', 'col_end') for st in E]))
    od = float(np.mean([O[(t, st)]['O'] for t in ('dig_start', 'dig_end') for st in E]))
    r = oc / od
    lab = 'UNINFORMATIVE' if od < 3 else 'NUMBER-SPECIFIC' if r <= 0.5 else 'GENERIC' if r >= 0.75 else 'MIXED'
    print('M1 (end1-end4, both insertion points): mean O colour %.2f, digit %.2f, r = %.3f -> %s' % (oc, od, r, lab))
    print('--- M2: regression of the digit-subspace projection on the other numbers\' digits (v185 digit families) ---')
    di, ci = np.where(np.char.startswith(typ, 'dig'))[0], np.where(np.char.startswith(typ, 'col'))[0]
    ud = np.unique(fam[di]); ftid = [tid[di][fam[di] == f][0] for f in ud]
    Zs = {'units': np.stack([unit_counts(bq[t]) for t in ftid]), 'alldigits': np.stack([digit_counts(bq[t]) for t in ftid])}
    res = {}
    for st in sts:
        res[st] = m2(np.asarray(X[di, si, st], dtype=np.float64), y[di], fam[di], tid[di], Zs,
                     np.asarray(X[ci, si, st], dtype=np.float64), y[ci], fam[ci], tid[ci])
        for name in Zs:
            q = res[st]
            print('%-5s %-9s R2 digit subspace %+.3f | random mean %+.3f p95 %+.3f | colour subspace %+.3f' % (
                states[st], name, q[(name, 'U')], q[(name, 'rand_mean')], q[(name, 'rand_p95')], q[(name, 'col')]),
                flush=True)
    for name in Zs:
        rU = float(np.mean([res[st][(name, 'U')] for st in E]))
        dR = float(np.mean([res[st][(name, 'U')] - max(res[st][(name, 'rand_p95')], res[st][(name, 'col')]) for st in E]))
        print('M2 v185 %s (end1-end4): R2(U) %+.3f, delta R2 %+.3f%s' % (name, rU, dR, '   <- primary' if name == 'units' else ''))


def run_v181(root):
    X, meta = load(root, 'v181'); its = meta['items']; states = meta['states']; si = meta['slots'].index('last')
    y = np.array([i['digit'] for i in its]); fam = np.array([i['fam'] for i in its]); tid = np.array([i['tid'] for i in its])
    sts = [k for k, s in enumerate(states) if s != 'h0']; E = [st for st in sts if states[st].startswith('end')]
    fq = {(f['tid'], f['anchor']): f for f in families(json.load(open('v178_labels.json'))['items'])}
    u = np.unique(fam); key = [(its[int(np.where(fam == f)[0][0])]['tid'], its[int(np.where(fam == f)[0][0])]['anchor']) for f in u]
    assert all(k in fq for k in key), 'families do not match the v181 meta'
    print('v181 last prompt token: %d items, %d families, %d templates' % (len(its), len(u), len(set(tid))), flush=True)
    Zs = {'units': np.stack([unit_counts(fq[k]['question'], fq[k]['cu']) for k in key]),
          'alldigits': np.stack([digit_counts(fq[k]['question'], fq[k]['cu']) for k in key])}
    print('features: other integers per question, mean %.2f (min %d, max %d)' % (
        Zs['units'].sum(1).mean(), Zs['units'].sum(1).min(), Zs['units'].sum(1).max()))
    res = {}
    for st in sts:
        res[st] = m2(np.asarray(X[:, si, st], dtype=np.float64), y, fam, tid, Zs)
        for name in Zs:
            q = res[st]
            print('%-5s %-9s R2 digit subspace %+.3f | random mean %+.3f p95 %+.3f' % (
                states[st], name, q[(name, 'U')], q[(name, 'rand_mean')], q[(name, 'rand_p95')]), flush=True)
    for name in Zs:
        rU = float(np.mean([res[st][(name, 'U')] for st in E]))
        dR = float(np.mean([res[st][(name, 'U')] - res[st][(name, 'rand_p95')] for st in E]))
        print('M2 v181 %s (end1-end4): R2(U) %+.3f, delta R2 %+.3f%s' % (name, rU, dR, '   <- primary' if name == 'units' else ''))


def run_v194(root):
    from multiprocessing import Pool
    import v194_analyze as B
    X, meta = load(root, 'v194'); its = meta['items']; S8 = meta['states']
    jobs = [('main', op, 0, st, l, 'state') for op in ('plus', 'minus') for l in 'abc' for st in range(len(S8))]
    with Pool(4, initializer=B.init, initargs=(X, its)) as pool:
        res = dict(pool.map(B.job, jobs))
    print('--- M3: Study 3 copies at P0 by state (%s) ---' % ' '.join(S8))
    for op in ('plus', 'minus'):
        for l in 'abc':
            print('%-5s copy %s at P0: %s' % (op, l, ' '.join('%.3f' % res[('main', op, 0, st, l, 'state')].mean()
                                                          for st in range(len(S8)))))
    e4 = S8.index('end4')
    bb = [res[('main', op, 0, e4, 'b', 'state')].mean() for op in ('plus', 'minus')]
    print('M3: copy of b at P0 after loop 4: plus %.3f, minus %.3f -> %s' % (
        bb[0], bb[1], 'strong (>= 0.85 in both)' if min(bb) >= 0.85 else 'not strong in both; state the value'))


def selftest():
    """Synthetic: other numbers' digits write the same codes into the context (interference world) vs not."""
    rng = np.random.default_rng(0); d, nt = 64, 40
    code = rng.standard_normal((10, d)); col = rng.standard_normal((10, d))
    for world in ('interference', 'none'):
        Xs, ys, fs, ts, Z = [], [], [], [], []
        for t in range(nt):
            others = rng.integers(0, 10, rng.integers(2, 6)); z = np.bincount(others, minlength=10).astype(float)
            ctx = rng.standard_normal(d) * 3 + (code[others].sum(0) * 2 if world == 'interference' else 0)
            for k in range(10):
                Xs.append(ctx + code[k] * 0.5 + rng.standard_normal(d) * 0.1); ys.append(k); fs.append(t); ts.append(t)
            Z.append(z)
        out = m2(np.array(Xs), np.array(ys), np.array(fs), np.array(ts), {'units': np.array(Z)}, n_rand=5)
        print('selftest %-12s R2(U) %+.3f, random p95 %+.3f' % (world, out[('units', 'U')], out[('units', 'rand_p95')]))
    assert unit_counts('He has 1,205 apples and 3.5 kg, then 40.', skip_cu=None).tolist() == [1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
    q = 'Tom has 37 cards and 12 more.'
    assert unit_counts(q, skip_cu=q.index('37') + 1).tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert digit_counts(q, skip_cu=q.index('37') + 1).tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    print('selftest feature counts ok')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'selftest':
        selftest()
    else:
        {'v185': run_v185, 'v181': run_v181, 'v194': run_v194}[mode](sys.argv[2])
