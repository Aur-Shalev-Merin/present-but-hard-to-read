"""v232: Ouro Study 3 at the end of the sentence that gives b, before c is seen (docs/prereg_v232_ouro_sb.md).

At Sb (the period ending "Her friend gave her 13 more stickers.") a and b are in the prompt and c is not, so a units
code of a op b there cannot be the answer with c added back, the alternative no label separates at P1.

Part A  share swap at Sb (primary) and Sb-1 (secondary) on the saved v199 states: v228_rescore.analyse unchanged
        (labels from a, b; cluster pair; states mid3, end3, mid4, end4; 10,000 resamples; 97.5% CIs) with the copies
        of a and b regressed (the only varied digits before Sb). The bar TAU_SB comes from null worlds on this design,
        scored blind before any real state: v224's five mixture kinds on Study 3's items with copies of a and b only
        (no answer, carry or c, none of which can be at Sb), one identical state per (family, a, b) as in the real
        causal states, lam 0.05-16, seeds 1-6; TAU_SB = max(0.20, 1.5 x the largest I / N among runs with signal,
        rounded up to 0.05) (v224's rule). A FOLLOWS also needs N >= N_FLOOR 0.002, else TOO-FAINT.
Part B  v199's frozen clamped patching over the sentence's tail (v199_patch.py, byte for byte), with a new plan: the
        needed step's and its twin's 9-dim subspaces (v199's frozen fit, reproduced and checked) projected off the
        copies of a and b (class means at the same point, offset and fit set; v216's projection). Statistic
        hit(N) - hit(rand) (v231: the twin comparison is weak), cluster bootstrap over x's family, 10,000 resamples,
        seed 2321, 97.5% CI; USED >= 0.10 with CI lower > 0, NOT-USED CI upper < 0.02, UNSCORABLE hit(btok) < 0.30.

  python v232_sb.py --selftest      synthetic checks (Study 3's items; no states)
  python v232_sb.py cpu <root>      Part A (null first, then the states) and the Part B plan (needs v199 states + plan)
  python v232_sb.py b <rdir>        Part B analysis of v199_patch_part*.json
"""
import glob, json, os, shutil, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v186_analyze as A
import v224_share as SH
import v228_rescore as R

OUT = os.environ.get('OUT_DIR', '.'); DD = SH.DD
STS = ['mid3', 'end3', 'mid4', 'end4']; EARLY = ['mid1', 'end1', 'mid2', 'end2']
KEYS = ['a', 'b']; N_FLOOR = 0.002
LAMS3 = (0.05, 0.1) + SH.LAMS; SEEDS = range(1, 7)
if os.environ.get('V232_TINY'):                                          # smoke test of the code path only
    LAMS3, SEEDS = (1.0,), range(1, 2)
POINTS = ['mid1', 'end1', 'mid2', 'end2', 'mid3', 'end3', 'mid4']          # v199's patch points
QB = (1.25, 98.75)


def items(meta_items):
    return [dict(fam=i['fam'], pair=i['pair'], ver=i['op'], u=i['labels']['a'], v=i['labels']['b'], a=i['labels']['a'],
                 b=i['labels']['b']) for i in meta_items]


def worldSb(its, kind, lam, seed, comp=0.0):
    """Null world on Sb's design: copies of a and b; one mixture of (a, b, op) at lam x the copies' RMS; optional
    units code of a op b at comp x RMS; family offsets; N(0, 1) noise per (family, a, b, op), so items that differ
    only in c share one state, as the causal states at Sb do."""
    rng = np.random.default_rng(seed); n = len(its)
    u, v = (np.array([i[k] for i in its]) for k in ('a', 'b')); o = np.array([i['ver'] == 'minus' for i in its], int)
    U1, U2 = (rng.standard_normal((10, DD)) for _ in range(2)); cop = U1[u] + U2[v]
    rms = np.sqrt((cop ** 2).sum(1).mean()); sc = lambda x: x * rms / np.sqrt((x ** 2).sum(1).mean())
    if kind == 'pair':
        mix = sum(np.tanh(2 * rng.standard_normal((10, DD))[x] * rng.standard_normal((10, DD))[y]) for x, y in ((u, v), (u, o), (v, o)))
    elif kind == 'fourier':
        mix = np.zeros((n, DD))
        for f in (1, 2, 5):
            c1, s1, c2, s2 = (g(2 * np.pi * f * x / 10) for x in (u, v) for g in (np.cos, np.sin))
            Fm = np.stack([c1 * c2, s1 * s2, c1 * s2, s1 * c2], 1); Rm = rng.standard_normal((2, 4, DD))
            mix += np.einsum('nk,nkd->nd', Fm, Rm[o])
    elif kind == 'mlp':
        H = np.concatenate([np.eye(10)[u], np.eye(10)[v], np.eye(2)[o]], 1)
        mix = np.maximum(H @ rng.standard_normal((22, 256)) + rng.standard_normal(256), 0) @ rng.standard_normal((256, DD))
    elif kind == 'paircode':
        mix = rng.standard_normal((10, 10, DD))[u, v]
    elif kind == 'paircode_op':
        mix = rng.standard_normal((2, 10, 10, DD))[o, u, v]
    elif kind == 'none':
        mix = np.zeros((n, DD))
    else:
        raise ValueError(kind)
    sig = cop + (lam * sc(mix) if mix.any() else 0)
    if comp:
        val = np.where(o == 0, u + v, u - v) % 10; sig = sig + comp * sc(rng.standard_normal((10, DD))[val])
    fams = sorted(set(i['fam'] for i in its)); fx = {f: j for j, f in enumerate(fams)}
    offs = rng.standard_normal((len(fams), DD)) * 3; fi = np.array([fx[i['fam']] for i in its])
    sig = sig + offs[fi]
    _, inv = np.unique(np.stack([fi, u, v, o], 1), axis=0, return_inverse=True); inv = inv.ravel(); m = inv.max() + 1
    return {l: (sig + rng.standard_normal((m, DD))[inv]).astype(np.float32) for l in STS}


def score(its, Xw, nboot=1000, tau=None):
    return R.analyse(lambda idx, l: Xw[l][idx], its, KEYS, cluster='pair', layers=STS, nboot=nboot, tau=tau)


def _null(args):
    its, kind, lam, seed = args
    r = score(its, worldSb(its, kind, lam, seed))
    return dict(kind=kind, lam=lam, seed=seed, N=r['N'], NCIlo=r['ci']['N'][0], I=r['I'], I_rel=r['I_rel'], label=r['label'])


def nullgrid(its, procs=4):
    jobs = [(its, k, lam, s) for k in SH.NULL_KINDS for lam in LAMS3 for s in SEEDS]
    with Pool(procs) as p:
        rows = p.map(_null, jobs)
    sig = [x for x in rows if x['NCIlo'] > 0 and np.isfinite(x['I_rel'])]
    worst = max([x['I_rel'] for x in sig], default=0.0)
    return rows, worst, float(max(0.20, np.ceil(1.5 * worst / 0.05 - 1e-9) * 0.05))


COMPS = (0.05, 0.08, 0.12, 0.2, 0.3, 0.5)


def power(its, n_obs, rows, tau):
    """Descriptive power read at Ouro's observed mixing: per kind, the null lam whose N (seed 1) is nearest Ouro's N at
    Sb; plant a op b at each comp (seed 101) and report the smallest comp labelled FOLLOWS above the floor."""
    out = {}
    for k in SH.NULL_KINDS:
        ks = [x for x in rows if x['kind'] == k and x['seed'] == min(SEEDS)]
        lam = min(ks, key=lambda x: abs(x['N'] - n_obs))['lam']; got = None; res = []
        for comp in COMPS:
            r = score(its, worldSb(its, k, lam, 101, comp=comp), tau=tau); lab = floor_label(r)
            res.append(dict(comp=comp, N=r['N'], I=r['I'], I_rel=r['I_rel'], label=lab))
            if lab == 'FOLLOWS' and got is None:
                got = res[-1]
        out[k] = dict(lam=lam, rows=res, first=got)
        print('power %-11s at lam %.2f (null N nearest Ouro\'s %.4f): smallest FOLLOWS %s' % (
            k, lam, n_obs, 'comp %.2f (I %+.4f, N %.4f)' % (got['comp'], got['I'], got['N']) if got else 'none up to comp %.2f' % COMPS[-1]), flush=True)
    return out


def floor_label(r):
    return 'TOO-FAINT' if r['label'] == 'FOLLOWS' and r['N'] < N_FLOOR else r['label']


# ---------------------------------------------------------------- Part B plan

def need10(i, flip=False):
    L = i['labels']; minus = (i['op'] == 'minus') != flip
    return (L['a'] - L['b']) % 10 if minus else (L['a'] + L['b']) % 10


def project(U, Qs):
    """v216: U's span with the copies' spans removed; returns (Uo, share of U left)."""
    Q = np.linalg.qr(np.concatenate(Qs, 1))[0]; M = U - Q @ (Q.T @ U)
    Uo, sv, _ = np.linalg.svd(M, full_matrices=False)
    return Uo[:, sv > 1e-3 * sv.max()], float(np.linalg.norm(M) ** 2 / U.shape[1])


def plan(X, its, S, plan0, say):
    """Reproduce v199's frozen subspaces (checked against its plan), then project them off a's and b's copies."""
    half = lambda k: int(its[k]['tid'] >= 6); out = {'U_rand_%d' % h: plan0['U_rand_%d' % h] for h in (0, 1)}
    worst = 0.0; left = {}

    def top(ks, y, st, off, r=9):
        Z = np.asarray(X[np.array(ks), 0 if off == 0 else 1 + off, st], dtype=np.float64); fam = np.array([its[k]['fam'] for k in ks])
        Z = A.demean(Z, fam); mu = np.stack([Z[y == v].mean(0) for v in range(10)]); mu -= mu.mean(0)
        return np.linalg.svd(mu, full_matrices=False)[2][:r].T
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
                        if not fit:
                            out[key] = out['U_%s_%s_%d_%s_o%d' % (nm, op, h, p, src)]; continue
                        st = S.index(p)
                        U = top(fit, np.array([need10(its[k], flip) for k in fit]), st, o)
                        U0 = plan0[key].astype(np.float64)
                        worst = max(worst, float(np.linalg.norm(U @ U.T - U0 @ U0.T)))
                        Qa = top(fit, np.array([its[k]['labels']['a'] for k in fit]), st, o)
                        Qb = top(fit, np.array([its[k]['labels']['b'] for k in fit]), st, o)
                        out[key], left[key] = project(U0, [Qa, Qb])
                        out[key] = out[key].astype(np.float32)
                say('plan %s half %d offset %d: %s' % (op, h, o, 'fitted on %d items' % len(fit) if fit else 'uses offset %d' % src))
    return out, worst, left


def cpu(root):
    f = [x for x in glob.glob(os.path.join(root, '**', 'v199_states.npy'), recursive=True) if 'gpu0' not in x and 'gpu1' not in x][0]
    meta = json.load(open(os.path.join(os.path.dirname(f), 'v199_meta.json'), encoding='utf-8'))
    X = np.load(f, mmap_mode='r'); S = meta['states']; P = meta['positions']; mi = meta['items']
    assert {i['set'] for i in mi} == {'main'} and len(mi) == X.shape[0] == 14400, (len(mi), X.shape)
    its = items(mi); out = {}
    print('=== Part A null: 5 kinds x %d lam x seeds 1-6 on Sb\'s design (copies a, b), scored before any state ===' % len(LAMS3), flush=True)
    rows, worst, tau = nullgrid(its)
    for k in SH.NULL_KINDS:
        ks = [x for x in rows if x['kind'] == k]; sg = [x for x in ks if x['NCIlo'] > 0]
        print('  %-11s signal %2d/%d; FOLLOWS at 0.30 %d; I/N max with signal %s; N with signal %s' % (
            k, len(sg), len(ks), sum(x['label'] == 'FOLLOWS' for x in ks), '%+.3f' % max(x['I_rel'] for x in sg) if sg else '-',
            '%.4f-%.4f' % (min(x['N'] for x in sg), max(x['N'] for x in sg)) if sg else '-'), flush=True)
    print('largest null I/N with signal %.3f -> TAU_SB %.2f; null runs with I >= TAU_SB x N and N >= %.3f: %d of %d' % (
        worst, tau, N_FLOOR, sum(x['N'] >= N_FLOOR and x['I'] >= tau * x['N'] and x['label'] == 'FOLLOWS' for x in rows), len(rows)), flush=True)
    out.update(null_rows=rows, null_worst=worst, tau_sb=tau)
    print('=== Part A: Ouro Study 3 at Sb (primary) and Sb-1 (secondary); copies a, b regressed; TAU_SB %.2f ===' % tau, flush=True)
    for p in ('Sb', 'Sb-1'):
        pi = P.index(p)
        r = R.analyse(lambda idx, l, pi=pi: X[idx, pi, S.index(l)], its, KEYS, cluster='pair', layers=STS, tau=tau)
        r['label_floor'] = floor_label(r); R.say_r('A ' + p, r); print('   -> %s' % r['label_floor'], flush=True); out['A_' + p] = r
        r = R.analyse(lambda idx, l, pi=pi: X[idx, pi, S.index(l)], its, KEYS, cluster='pair', layers=EARLY, tau=tau, nboot=1000)
        R.say_r('   (loops 1-2, descriptive) ' + p, r); out['A_%s_early' % p] = r
    print('PART A (Sb, primary): %s' % out['A_Sb']['label_floor'], flush=True)
    out['power'] = power(its, out['A_Sb']['N'], rows, tau)
    print('=== Part B plan: v199\'s frozen subspaces, projected off the copies of a and b ===', flush=True)
    pf = sorted(glob.glob(os.path.join(root, '**', 'v199_patch_plan.npz'), recursive=True))[0]
    pj = sorted(glob.glob(os.path.join(root, '**', 'v199_patch_pairs.json'), recursive=True))[0]
    plan0 = dict(np.load(pf)); pl, worst_p, left = plan(X, mi, S, plan0, print)
    print('reproduction of v199\'s frozen subspaces: max ||UU^T - U0U0^T|| %.2e (must be < 1e-3)' % worst_p, flush=True)
    out['plan_repro'] = worst_p; out['plan_left'] = left
    lv = np.array(list(left.values()))
    print('share of the needed step\'s directions left after the projection: median %.3f, range %.3f-%.3f; dims kept %s' % (
        np.median(lv), lv.min(), lv.max(), sorted({pl[k].shape[1] for k in left})), flush=True)
    json.dump(out, open(os.path.join(OUT, 'v232_result.json'), 'w'), indent=1, default=float)
    if worst_p >= 1e-3:
        print('PLAN CHECK FAILED: no plan written, Part B not run', flush=True); sys.exit(1)
    os.makedirs(os.path.join(OUT, 'plan'), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, 'plan', 'v199_patch_plan.npz'), **pl)
    shutil.copy(pj, os.path.join(OUT, 'plan', 'v199_patch_pairs.json'))
    print('plan written (%d subspaces) with v199\'s pairs' % len(pl), flush=True)


# ---------------------------------------------------------------- Part B analysis

def label_b(rows):
    bd = [r for r in rows if r['b_differs']]
    pos = float(np.mean([r['btok'] == str(r['target_b']) for r in bd]))
    hit = {c: np.array([r[c] == str(r['target']) for r in rows], float) for c in ('N', 'F', 'rand', 'full', 'Nsb', 'fullsb')}
    fam = np.array([r['fam'] for r in rows]); u = np.unique(fam); idx = {f: np.where(fam == f)[0] for f in u}
    rng = np.random.default_rng(2321); bs, bf = [], []
    for _ in range(10000):
        p = np.concatenate([idx[f] for f in rng.choice(u, len(u))])
        bs.append(hit['N'][p].mean() - hit['rand'][p].mean())
        bf.append(hit['N'][p].mean() - max(hit['F'][p].mean(), hit['rand'][p].mean()))
    d = hit['N'].mean() - hit['rand'].mean(); lo, hi = np.percentile(bs, QB)
    lab = 'UNSCORABLE' if pos < 0.30 else 'USED' if d >= 0.10 and lo > 0 else 'NOT-USED' if hi < 0.02 else 'UNCLEAR'
    df = hit['N'].mean() - max(hit['F'].mean(), hit['rand'].mean())
    return dict(btok=pos, hits={c: float(h.mean()) for c, h in hit.items()}, d=float(d), ci=[float(lo), float(hi)], label=lab,
                frozen=float(df), frozen_ci=[float(x) for x in np.percentile(bf, QB)])


def analyse_b(rdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(rdir, '**', 'v199_patch_part*.json'), recursive=True)):
        rows += json.load(open(f))['rows']
    r = label_b(rows)
    print('pairs %d, families %d | unpatched own digit %.3f | method control btok hit %.3f (UNSCORABLE below 0.30)' % (
        len(rows), len({x['fam'] for x in rows}), np.mean([x['none'] == str(x['own']) for x in rows]), r['btok']))
    print('hits: %s' % ' '.join('%s %.3f' % kv for kv in r['hits'].items()))
    print('descriptive, v199\'s frozen statistic hit(N) - max(hit(F), hit(rand)): %+.3f [%+.3f, %+.3f]' % (r['frozen'], *r['frozen_ci']))
    print('PART B: hit(N, copies projected) - hit(rand) %+.3f [%+.3f, %+.3f] (97.5%%) -> %s' % (r['d'], *r['ci'], r['label']))
    json.dump(r, open(os.path.join(OUT, 'v232_partB.json'), 'w'), indent=1)


def selftest():
    its = items([i for i in json.load(open('v194_items.json'))['items'] if i['set'] == 'main']); ok = True
    for comp, kind, want in ((0.5, 'pair', True), (1.0, 'none', True), (0.0, 'none', False), (0.0, 'pair', False)):
        r = score(its, worldSb(its, kind, 1.0, 3, comp=comp)); got = r['label'] == 'FOLLOWS'
        good = got == want; ok &= good
        print('selftest world %-5s comp %.1f: %s (I/N %+.2f, N %.4f) -> %s' % (kind, comp, r['label'], r['I_rel'], r['N'], 'OK' if good else 'FAIL'), flush=True)
    rng = np.random.default_rng(0); d = 64
    U = np.linalg.qr(rng.standard_normal((d, 9)))[0]; Qa = np.linalg.qr(rng.standard_normal((d, 9)))[0]
    Qb = np.linalg.qr(0.5 * U + rng.standard_normal((d, 9)))[0]
    Uo, left = project(U, [Qa, Qb]); orth = max(np.abs(Uo.T @ Qa).max(), np.abs(Uo.T @ Qb).max())
    good = orth < 1e-6 and np.allclose(Uo.T @ Uo, np.eye(Uo.shape[1]), atol=1e-6) and 0 < left < 1; ok &= good
    print('selftest projection: max |cos| with the copies %.1e, orthonormal, share left %.2f -> %s' % (orth, left, 'OK' if good else 'FAIL'))
    for eff, want in ((0.30, 'USED'), (0.0, 'NOT-USED')):
        rows = []
        for k in range(2000):
            hitN = rng.random() < 0.01 + eff
            rows.append(dict(fam=k % 60, b_differs=True, target=1, target_b=2, own=0, none='0', btok='2' if rng.random() < 0.6 else '0',
                             N='1' if hitN else '0', F='1' if rng.random() < 0.01 else '0', rand='1' if rng.random() < 0.01 else '0',
                             full='0', Nsb='0', fullsb='0'))
        r = label_b(rows); good = r['label'] == want; ok &= good
        print('selftest Part B planted %.2f: %+.3f [%+.3f, %+.3f] %s -> %s' % (eff, r['d'], *r['ci'], r['label'], 'OK' if good else 'FAIL'))
    return ok


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        sys.exit(0 if selftest() else 1)
    elif sys.argv[1] == 'cpu':
        cpu(sys.argv[2])
    elif sys.argv[1] == 'b':
        analyse_b(sys.argv[2])
