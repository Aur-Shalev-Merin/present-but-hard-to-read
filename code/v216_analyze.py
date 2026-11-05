"""v216 analysis (CPU; docs/prereg_v216_answer_only.md). Part A uses v215_analyze.part_a unchanged (probe, labels S, M,
copy ceiling, band, ex_N, I, bootstrap, labels) with v216's positions eq / pre / last; secondary: the answer line,
excess of (A + j k) mod 5 for j = -2..2 per position (band); Part B plan: per position, band layer and half of the
families, the B, d2-copy, answer (class means of A) and random subspaces, and B orthogonalised against copy + answer.

  python v216_analyze.py <root>        -> reading; v216_partA.json, v216_plan.npz, v216_pairs.json
  python v216_analyze.py --selftest
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v215_analyze as A

A.POS = ['eq', 'pre', 'last']; POS = A.POS; OUT = os.environ.get('OUT_DIR', '.')
LINE = (-2, -1, 0, 1, 2)
_labels = A.labels


def labels(it):
    d = _labels(it)
    d.update({'L%+d' % j: (it['A'] + j * it['k']) % 5 for j in LINE})
    return d


A.labels = labels                                       # adds the answer-line labels for A.job


def answer_line(X, its, say=print, procs=4):
    idx = {v: np.array([k for k, i in enumerate(its) if i['ver'] == v]) for v in ('plus', 'minus')}
    labs = ['L%+d' % j for j in LINE]
    jobs = [(v, p, li, lab, 'state') for v in idx for p in range(len(POS)) for li in A.BAND for lab in labs]
    jobs += [(v, 0, 0, lab, 'ceil') for v in idx for lab in labs]
    d = dict(X=X, its=its, idx_plus=idx['plus'], idx_minus=idx['minus'])
    with Pool(procs, initializer=A.init, initargs=(d,)) as pool:
        res = dict(pool.imap_unordered(A.job, jobs, chunksize=1))
    say('Answer line, excess over the copy ceiling (band), (A + j k) mod 5; the needed form of b is j = -1 on o2 = + and '
        '(by symmetry of mod 5) the negation on o2 = -:')
    for p in range(len(POS)):
        say('  %-4s ' % POS[p] + ' | '.join('%s %s' % (v, ' '.join('%s %+.3f' % (lab, np.mean([res[(v, p, li, lab, 'state')].mean()
                                                                                          for li in A.BAND]) - res[(v, 0, 0, lab, 'ceil')].mean())
                                                                for lab in labs)) for v in idx))
    return res


def by_correct(X, its, meta, say=print, procs=4):
    """Part A split by whether the model answers the item correctly (pre-registered, secondary): the per-item excess
    of the needed and flipped forms (band) from the same family-demeaned probes, averaged within correct and wrong
    items; ex_N and I with the family-cluster bootstrap (seed 2164). Wrong answers concentrate in minus-minus items,
    so the split is partly an operation split; it is reported per version too."""
    idx = {v: np.array([k for k, i in enumerate(its) if i['ver'] == v]) for v in ('plus', 'minus')}
    jobs = [(v, p, li, lab, 'state') for v in idx for p in range(len(POS)) for li in A.BAND for lab in ('S', 'M')]
    jobs += [(v, 0, 0, lab, 'ceil') for v in idx for lab in ('S', 'M')]
    d = dict(X=X, its=its, idx_plus=idx['plus'], idx_minus=idx['minus'])
    with Pool(procs, initializer=A.init, initargs=(d,)) as pool:
        res = dict(pool.imap_unordered(A.job, jobs, chunksize=1))
    corr = {v: np.array([meta[k]['correct'] for k in idx[v]]) for v in idx}
    fam = {v: np.array([its[k]['fam'] for k in idx[v]]) for v in idx}
    out = {}
    for p in range(len(POS)):
        ex = {(v, lab): np.mean([res[(v, p, li, lab, 'state')] for li in A.BAND], 0) - res[(v, 0, 0, lab, 'ceil')]
              for v in idx for lab in ('S', 'M')}
        for name, sel in (('correct', True), ('wrong', False)):
            mp, mm = corr['plus'] == sel, corr['minus'] == sel
            if mp.sum() < 50 or mm.sum() < 50:
                say('  %-4s %-7s too few items (plus %d, minus %d)' % (POS[p], name, mp.sum(), mm.sum())); continue
            exN = A.cluster_ci(fam['plus'][mp], ex[('plus', 'S')][mp], fam['minus'][mm], ex[('minus', 'M')][mm], seed=2164)
            dp = ex[('plus', 'S')][mp] - ex[('plus', 'M')][mp]; dm = ex[('minus', 'M')][mm] - ex[('minus', 'S')][mm]
            I = A.cluster_ci(fam['plus'][mp], dp, fam['minus'][mm], dm, seed=2165)
            say('  %-4s %-7s (plus %4d, minus %4d): ex_N %+.4f [%+.4f, %+.4f] | I %+.4f [%+.4f, %+.4f] (D+ %+.3f, D- %+.3f)' % (
                POS[p], name, mp.sum(), mm.sum(), *exN, *I, dp.mean(), dm.mean()))
            out['%s/%s' % (POS[p], name)] = dict(exN=list(exN), I=list(I), n=[int(mp.sum()), int(mm.sum())])
    return out


def patch_plan(X, its, meta, positions, outdir, say, per_pos=1000):
    fams = sorted(set(i['fam'] for i in its)); half = {f: int(j % 2) for j, f in enumerate(np.random.default_rng(2161).permutation(fams))}
    fam = np.array([i['fam'] for i in its]); arr = {}; rr = np.random.default_rng(2162)
    lab = {k: np.array([i[k] for i in its]) for k in ('B', 'A', 'd2')}

    def basis(Z, m, y):
        M = np.stack([Z[m & (y == v)].mean(0) for v in range(10) if (m & (y == v)).any()]); M -= M.mean(0)
        return np.linalg.svd(M, full_matrices=False)[2][:len(M) - 1].T
    for p in positions:
        pi = POS.index(p)
        for li in A.BAND:
            Z = A.demean(np.asarray(X[:, pi, li], np.float32), fam); L = A.LAYERS[li]
            for h in (0, 1):
                m = np.array([half[f] != h for f in fam])         # fitted on the OTHER half
                U, D, An = basis(Z, m, lab['B']), basis(Z, m, lab['d2']), basis(Z, m, lab['A'])
                Q = np.linalg.qr(np.concatenate([D, An], 1))[0]
                Uo, sv, _ = np.linalg.svd(U - Q @ (Q.T @ U), full_matrices=False); Uo = Uo[:, sv > 1e-3 * sv.max()]
                for k, v in (('U', U), ('D', D), ('An', An), ('Uo', Uo)):
                    arr['%s_%s_%d_%d' % (k, p, h, L)] = v.astype(np.float32)
                cd, ca = (np.linalg.svd(U.T @ W, compute_uv=False) ** 2 for W in (D, An))
                left = np.linalg.norm(U - Q @ (Q.T @ U)) ** 2 / U.shape[1]
                say('  %s layer %d half %d: B vs copy mean cos^2 %.3f, vs answer %.3f; share of B left after both %.3f' % (
                    p, L, h, cd.mean(), ca.mean(), left))
            arr['R_%s_%d' % (p, L)] = np.linalg.qr(rr.standard_normal((Z.shape[1], 9)))[0].astype(np.float32)
    rng = np.random.default_rng(2161); pairs = []
    alts = {k: [j for j, o in enumerate(its) if o['fam'] == i['fam'] and o['ver'] == i['ver'] and o['d1'] == i['d1'] and o['d2'] != i['d2']]
            for k, i in enumerate(its)}
    for p in positions:
        cand = [k for k, i in enumerate(its) if meta[k]['correct'] and alts[k]]
        for k in rng.permutation(cand)[:per_pos]:
            pairs.append(dict(pos=p, x=int(k), x2=int(rng.choice(alts[k])), half=half[its[k]['fam']]))
    np.savez(os.path.join(outdir, 'v216_plan.npz'), **arr)
    json.dump(dict(positions=positions, band=[A.LAYERS[li] for li in A.BAND], pairs=pairs), open(os.path.join(outdir, 'v216_pairs.json'), 'w'))
    say('plan: %d bases, %d pairs' % (len(arr), len(pairs)))


def relabel(out, its, meta, say=print):
    """Execution note 1 (prereg section 9): part_a gates on v215's G1 (correct >= 0.90 in each version); v216's
    registered G1 is 0.50. Labels recomputed from the same statistics with v215's rules and G1 at 0.50."""
    corr = [np.mean([meta[k]['correct'] for k, i in enumerate(its) if i['ver'] == v]) for v in ('plus', 'minus')]
    g1 = all(c >= 0.50 for c in corr)
    for p, r in out.items():
        ok = g1 and r['g2']; exN, I = r['exN'], r['I']
        r['label'] = ('UNSCORABLE' if not ok else 'COMBINED' if exN[0] >= 0.05 and exN[1] > 0 else
                      'NOT-COMBINED' if exN[2] < 0.02 else 'UNCLEAR')
        r['swap'] = ('UNSCORABLE' if not ok else 'FOLLOWS' if I[0] >= 0.05 and I[1] > 0 and r['Dp'] > 0 and r['Dm'] > 0 else
                     'DOES-NOT-FOLLOW' if I[2] < 0.02 else 'UNCLEAR')
        r['g1'] = bool(g1)
        say('%s, G1 at 0.50 (correct plus %.3f, minus %.3f; execution note 1): %s | %s' % (p, *corr, r['label'], r['swap']))


def analyse(X, its, meta, say=print, outdir=OUT, plan_all=False):
    for v in ('plus', 'minus'):
        for o2 in '+-':
            s = [meta[k]['correct'] for k, i in enumerate(its) if i['ver'] == v and i['o2'] == o2]
            say('answer accuracy %s, o2 %s: %.3f (n %d)' % (v, o2, np.mean(s), len(s)))
    out = A.part_a(X, its, meta, say=say, positions=POS, plan=False, outdir=outdir, tag='')
    os.replace(os.path.join(outdir, 'v215_partA.json'), os.path.join(outdir, 'v216_partA.json'))
    relabel(out, its, meta, say=say)
    json.dump(out, open(os.path.join(outdir, 'v216_partA.json'), 'w'), indent=1)
    answer_line(X, its, say=say)
    say('Split by whether the model answers correctly (pre-registered, secondary):')
    split = by_correct(X, its, meta, say=say)
    json.dump(split, open(os.path.join(outdir, 'v216_partA_split.json'), 'w'), indent=1)
    todo = [p for p in POS if plan_all or out[p]['label'] == 'COMBINED']
    say('Part B plan for positions: %s' % (todo or 'none'))
    if todo:
        patch_plan(X, its, meta, todo, outdir, say)
    return out


def main(root):
    import v215_items as V5, v216_items as I
    its = I.build(); s = V5.sha16(its)
    print('items sha %s (frozen %s)' % (s, I.ITEMS_SHA), flush=True)
    if I.ITEMS_SHA and s != I.ITEMS_SHA:
        sys.exit('items hash differs')
    f = glob.glob(os.path.join(root, '**', 'v216_fam_states.npy'), recursive=True)[0]
    X = np.load(f, mmap_mode='r'); m = json.load(open(f.replace('_states.npy', '_meta.json')))
    assert m['items_sha'] == s and len(m['items']) == len(its)
    analyse(X, its, m['items'], plan_all=os.environ.get('V216_PLAN_ALL') == '1')


def selftest():
    """60 synthetic families on the real design: b computed -> COMBINED + FOLLOWS; copies only -> NOT-COMBINED, not
    FOLLOWS; the answer only (a code of A, no b) -> recorded (first run: NOT-COMBINED, ex_N -0.11, but I +0.02-0.04:
    an answer code alone gives a small sign-following signal, the confound the answer-mixing null addresses)."""
    import tempfile, v216_items as I
    fam = I.build(); keep = set(sorted(set(i['fam'] for i in fam))[::4]); its = [i for i in fam if i['fam'] in keep]
    meta = [dict(correct=True) for _ in its]; rng = np.random.default_rng(0); dd = 64; ok = True
    U1, U2, CB, CA = (rng.standard_normal((10, dd)) for _ in range(4)); off = {f: rng.standard_normal(dd) * 3 for f in keep}
    for world, want, wswap in (('computed', 'COMBINED', 'FOLLOWS'), ('copies', 'NOT-COMBINED', 'not FOLLOWS'), ('answer', None, None)):
        X = np.zeros((len(its), 3, len(A.LAYERS), dd), np.float32)
        for k, i in enumerate(its):
            v = off[i['fam']] + U1[i['d1']] + U2[i['d2']] + rng.standard_normal((3, len(A.LAYERS), dd))
            if world == 'computed':
                v = v + 0.8 * CB[i['B']]
            if world == 'answer':
                v = v + 0.8 * CA[i['A']]
            X[k] = v
        out = A.part_a(X, its, meta, say=lambda s: None, positions=POS, plan=False, outdir=tempfile.mkdtemp())
        for p, r in out.items():
            good = want is None or (r['label'] == want and (r['swap'] != 'FOLLOWS' if wswap == 'not FOLLOWS' else r['swap'] == wswap))
            print('selftest %-8s %-4s ex_N %+.3f [%+.3f, %+.3f] -> %-12s I %+.3f -> %-15s %s' % (
                world, p, *r['exN'], r['label'], r['I'][0], r['swap'], ('OK' if good else 'FAIL') if want else '(recorded)'), flush=True)
            ok &= good
        if world == 'computed':
            patch_plan(X, its, meta, ['pre'], tempfile.mkdtemp(), lambda s: None)
    return ok


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        sys.exit(0 if selftest() else 1)
    main(sys.argv[1])
