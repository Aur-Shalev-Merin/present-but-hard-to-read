"""v188 within-line check (post hoc; reading rule in docs/prereg_v188_ouro_intermediate_confirmation.md, section 7,
"Within-line check", fixed before running).

Mod 5, every form other than the answer (a+b-c) lies on at most one line {answer + k*x, k = 1..4}, x a copy in {a, b, c}.
Each partial result is one member of a line: a+b = answer + c, a-c = answer - b, b-c = answer - a. A generic interaction
of the answer digit with one copied number predicts the four forms on a line read about equally well. At P1 (word
problems), for one study per run: all 31 forms at end4 and end3 (state probes), frozen and carry-and-borrow ceilings;
per-item correctness saved to v188_line_items_<study>.npz; within-line contrasts, the on-line elevation and per-template
sign counts printed.

  python v188_line_check.py <dir with states/meta> <188|186>       (Kaggle: probes + statistics)
  python v188_line_check.py --stats <npz> [<npz> ...]              (statistics and the reading from saved items)
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np


def norm(v):
    v = [x % 5 for x in v]; i = next(j for j in range(3) if v[j]); inv = pow(v[i], -1, 5)
    return tuple((x * inv) % 5 for x in v)


FORMS = [f for f in __import__('itertools').product(range(5), repeat=3) if any(f) and f[next(i for i in range(3) if f[i])] == 1]
ANS = (1, 1, 4)
COPIES = {(1, 0, 0): 'a', (0, 1, 0): 'b', (0, 0, 1): 'c'}
LINES = {x: [(k, norm([ANS[j] + (k if j == xi else 0) for j in range(3)])) for k in range(1, 5)]
         for xi, x in enumerate('abc')}
PARTIAL = {'c': (1, 1, 0), 'b': (1, 0, 4), 'a': (0, 1, 4)}
PNAME = {'c': 'a+b', 'b': 'a-c', 'a': 'b-c'}
ON = {f for l in LINES.values() for _, f in l}
OFF = [f for f in FORMS if f not in ON and f != ANS and f not in COPIES]
assert len(FORMS) == 31 and len(ON) == 12 and len(OFF) == 15
assert all(PARTIAL[x] in {f for _, f in LINES[x]} for x in 'abc') and not ON & set(COPIES) and ANS not in ON
FROZEN = {(1, 4, 0): 'a-b', (1, 0, 1): 'a+c', (1, 1, 3): 'ans-c', (1, 1, 1): 'ans+2c'}
Q = (1.25, 98.75)


def lname(f):
    for x, l in LINES.items():
        for k, g in l:
            if g == f:
                return 'ans%s%s' % ({1: '+', 2: '+2', 3: '-2', 4: '-'}[k], x)
    return '%da+%db+%dc' % f


def cluster_ci(cl, d, n=10000, seed=188):
    """Same draws and result as v186_analyze.cluster_ci (resampled mean = sum of picked cluster sums / their sizes)."""
    rng = np.random.default_rng(seed); u = np.unique(cl)
    S = np.array([d[cl == c].sum() for c in u]); N = np.array([(cl == c).sum() for c in u])
    bs = []
    for _ in range(n):
        p = rng.choice(len(u), len(u)); bs.append(S[p].sum() / N[p].sum())
    return float(d.mean()), float(np.percentile(bs, Q[0])), float(np.percentile(bs, Q[1]))


def stats(path):
    z = np.load(path); v = int(z['study']); fam, tid, ok = z['fam'], z['tid'], z['ok'].astype(bool)
    fi = {tuple(f): i for i, f in enumerate(z['forms'])}
    out = {}
    for si, st in enumerate(('end4', 'end3')):
        for ci, c in enumerate(('base', 'cb')):
            ex = z['state'][si].astype(float) - z['ceil'][ci]
            E = lambda f: ex[fi[f]]
            tag = 'v%d P1 %s ceiling=%s' % (v, st, c)
            for x in 'cba':
                others = [g for _, g in LINES[x] if g != PARTIAL[x]]
                best = max(others, key=lambda g: E(g).mean())
                d = E(PARTIAL[x]) - E(best); dm = E(PARTIAL[x]) - np.mean([E(g) for g in others], 0)
                fc, tc = cluster_ci(fam, d), cluster_ci(tid, d)
                pos = sum(d[tid == t].mean() > 0 for t in np.unique(tid))
                out[(v, st, c, x)] = (fc, tc, pos)
                print('%s | %s-line: %s %+.3f | %s | %s - best other (%s): %+.3f family [%+.3f, %+.3f] template [%+.3f, %+.3f] '
                      '| templates positive %d/%d | %s - mean of others %+.3f' % (
                          tag, x, PNAME[x], E(PARTIAL[x]).mean(), ' '.join('%s %+.3f' % (lname(g), E(g).mean()) for g in others),
                          PNAME[x], lname(best), fc[0], fc[1], fc[2], tc[1], tc[2], pos, len(np.unique(tid)), PNAME[x], dm.mean()))
            on_np = [g for g in ON if g not in PARTIAL.values()]
            e = np.mean([E(g) for g in on_np], 0) - np.mean([E(g) for g in OFF], 0)
            fc = cluster_ci(fam, e)
            print('%s | on-line elevation: 9 non-partial answer-line forms %+.3f vs 15 off-line forms %+.3f -> %+.3f '
                  'family [%+.3f, %+.3f]' % (tag, np.mean([E(g).mean() for g in on_np]), np.mean([E(g).mean() for g in OFF]),
                                             fc[0], fc[1], fc[2]))
            m5 = (1, 1, 0); nonans = [g for g in FORMS if g not in (ANS, m5)]
            ru = max(nonans, key=lambda g: E(g).mean()); bf = max(FROZEN, key=lambda g: E(g).mean())
            for lab, g in (('runner-up %s' % lname(ru), ru), ('best frozen control %s' % FROZEN[bf], bf)):
                d = E(m5) - E(g); pos = sum(d[tid == t].mean() > 0 for t in np.unique(tid))
                print('%s | a+b - %s %+.3f | templates positive %d/%d | correct-only %+.3f family [%+.3f, %+.3f]' % (
                    (tag, lab, d.mean(), pos, len(np.unique(tid))) + cluster_ci(fam[ok], d[ok])))
    return out


TWO = [f for f in FORMS if sum(x != 0 for x in f) == 2]                 # forms using exactly two of a, b, c
assert len(TWO) == 12 and all(p in TWO for p in PARTIAL.values())


def twonum(path):
    """Added after a side note (not part of the reading rule above): on each answer line the partial result is the only
    two-number member, so compare each partial result with the best of the 9 other two-number forms."""
    z = np.load(path); v = int(z['study']); fam, tid = z['fam'], z['tid']
    fi = {tuple(f): i for i, f in enumerate(z['forms'])}
    others = [f for f in TWO if f not in PARTIAL.values()]
    for si, st in enumerate(('end4', 'end3')):
        for ci, c in enumerate(('base', 'cb')):
            ex = z['state'][si].astype(float) - z['ceil'][ci]
            E = lambda f: ex[fi[f]]
            best = max(others, key=lambda g: E(g).mean())
            print('v%d P1 %s ceiling=%s | two-number forms: best non-partial %s %+.3f | %s' % (
                v, st, c, lname(best), E(best).mean(), ' | '.join(
                    '%s %+.3f, minus best %+.3f family [%+.3f, %+.3f] template [%+.3f, %+.3f]' % (
                        (PNAME[x], E(PARTIAL[x]).mean()) + cluster_ci(fam, E(PARTIAL[x]) - E(best))
                        + cluster_ci(tid, E(PARTIAL[x]) - E(best))[1:]) for x in 'cba')))


def reading(outs):
    cells = [(k, v) for o in outs for k, v in o.items() if k[1] == 'end4' and k[2] == 'base']
    n = sum(v[0][1] > 0 for _, v in cells)
    lab = 'SPECIFIC' if n >= 5 else 'MIXED' if n >= 3 else 'GENERIC'
    print('READING (rule fixed before running): end4, frozen ceiling, cells with family CI lower > 0: %d of %d -> %s'
          % (n, len(cells), lab))
    return lab


def run(root, v):
    from multiprocessing import Pool
    import v188_referee_checks as R
    assert R.FORMS == FORMS
    X, meta = R.S.load(root, v)
    St = meta['states']; E4, E3 = St.index('end4'), St.index('end3'); its = meta['items']
    main = [i for i in its if i['set'] == 'main']
    jobs = [(st, f, 'state') for st in (E4, E3) for f in FORMS] + [(E4, f, c) for f in FORMS for c in ('base', 'cb')]
    with Pool(4, initializer=R.init, initargs=(X, its)) as pool:
        res = dict(pool.map(R.job, jobs))
    path = 'v188_line_items_%d.npz' % v
    np.savez_compressed(path, study=v, forms=np.array(FORMS), fam=np.array([i['fam'] for i in main]),
                        tid=np.array([i['tid'] for i in main]), ok=np.array([i['correct'][3] for i in main], np.int8),
                        state=np.stack([[res[(st, f, 'state')] for f in FORMS] for st in (E4, E3)]),
                        ceil=np.stack([[res[(E4, f, c)] for f in FORMS] for c in ('base', 'cb')]))
    print('v%d: %d items; saved %s' % (v, len(main), path), flush=True)
    stats(path)


if __name__ == '__main__':
    if sys.argv[1] == '--stats':
        reading([stats(p) for p in sys.argv[2:]])
    elif sys.argv[1] == '--twonum':
        for p in sys.argv[2:]:
            twonum(p)
    else:
        run(sys.argv[1], int(sys.argv[2]))
