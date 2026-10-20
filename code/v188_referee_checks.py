"""v188 referee checks (post hoc; reading rules in docs/prereg_v188_ouro_intermediate_confirmation.md, section 7).

At P1 (word problems), for v188 and v186 states:
  - all 31 mod-5 linear forms x*a + y*b + z*c (first nonzero coefficient 1), each as excess over its own copy ceiling
    (frozen ceiling: one-hot a_u, b_u, c_u) and over the carry-and-borrow ceiling; end4 and end3;
  - m5 = (1,1,0) ranked among the 30 forms other than the answer form (1,1,4) = a + b - c;
  - margins over the runner-up and over the four frozen controls, with family- and template-level bootstraps (97.5%);
  - the same margins on items Ouro answers correctly after loop 4 (scored on the subset; probe trained on all items).

  python v188_referee_checks.py <dir with v188_* and v186_* states/meta>
"""
import itertools, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v186_analyze as A
import v178_analyze as V
import v188_shown_digit_check as S

FORMS = [f for f in itertools.product(range(5), repeat=3) if any(f) and f[next(i for i in range(3) if f[i])] == 1]
assert len(FORMS) == 31
M, ANS = (1, 1, 0), (1, 1, 4)
FROZEN = {(1, 4, 0): 'a-b', (1, 0, 1): 'a+c', (1, 1, 3): 'ans-c', (1, 1, 1): 'ans+2c'}
G = {}


def name(f):
    return FROZEN.get(f) or {M: 'a+b', ANS: 'ans'}.get(f) or '%da+%db+%dc' % f


def init(X, its):
    G['X'], G['its'] = X, its


def job(args):
    st, f, feat = args
    idx = np.array([k for k, i in enumerate(G['its']) if i['set'] == 'main']); its = [G['its'][k] for k in idx]
    fam = np.array([i['fam'] for i in its]); tid = np.array([i['tid'] for i in its])
    y = np.array([(f[0] * i['a'] + f[1] * i['b'] + f[2] * i['c']) % 5 for i in its])
    Z = np.asarray(G['X'][idx, 1, st], dtype=np.float32) if feat == 'state' else S.feats(its, feat)
    return args, (V.probe_predict(A.demean(Z, fam), y, tid, V.folds(tid)) == y).astype(np.int8)


def ci(cl, d, mask=None):
    m = np.ones(len(d), bool) if mask is None else mask
    return A.cluster_ci(cl[m], d[m], seed=188, q=(1.25, 98.75))


def run(v, X, meta):
    St = meta['states']; E, E3 = St.index('end4'), St.index('end3'); its = meta['items']
    main = [i for i in its if i['set'] == 'main']
    fam = np.array([i['fam'] for i in main]); tid = np.array([i['tid'] for i in main])
    ok = np.array([i['correct'][3] for i in main], bool)
    # label sanity: forms reproduce the frozen mod-5 labels
    for f, l in ((M, 'm5'), ((1, 4, 0), 'amb5'), ((1, 0, 1), 'apc5'), ((1, 1, 3), 'abmc5'), ((1, 1, 1), 'abc5')):
        assert all((f[0] * i['a'] + f[1] * i['b'] + f[2] * i['c']) % 5 == i['labels'][l] for i in main), l
    jobs = [(st, f, 'state') for st in (E, E3) for f in FORMS] + [(E, f, c) for f in FORMS for c in ('base', 'cb')]
    with Pool(4, initializer=init, initargs=(X, its)) as pool:
        res = dict(pool.map(job, jobs))
    print('v%d: %d items, %d templates, %d families, correct after loop 4 %.3f' % (
        v, len(main), len(set(tid)), len(set(fam)), ok.mean()), flush=True)
    for st in (E, E3):
        for c in ('base', 'cb'):
            ex = {f: res[(st, f, 'state')].astype(float) - res[(E, f, c)] for f in FORMS}
            order = sorted([f for f in FORMS if f != ANS], key=lambda f: -ex[f].mean())
            rank = order.index(M) + 1; ru = order[1] if order[0] == M else order[0]
            best4 = max(FROZEN, key=lambda f: ex[f].mean())
            tag = 'v%d P1 %s ceiling=%s' % (v, St[st], c)
            print('%s | m5 rank %d of 30 (answer form excluded; answer excess %+.3f) | top 6: %s' % (
                tag, rank, ex[ANS].mean(), ', '.join('%s %+.3f' % (name(f), ex[f].mean()) for f in order[:6])))
            for lab, comp in (('runner-up %s' % name(ru), ru), ('best frozen control %s' % name(best4), best4)):
                d = ex[M] - ex[comp]
                print('%s | m5 - %s: %+.3f | family CI [%+.3f, %+.3f] | template CI [%+.3f, %+.3f] | correct-only %+.3f '
                      'template CI [%+.3f, %+.3f] (n=%d) | wrong-only %+.3f' % (
                          (tag, lab, d.mean()) + ci(fam, d)[1:] + ci(tid, d)[1:] + ci(tid, d, ok) + (ok.sum(),
                                                                                                    d[~ok].mean())))
            print('%s | all 31 (excess): %s' % (tag, ' '.join('%s %+.3f' % (name(f), ex[f].mean())
                                                               for f in sorted(FORMS, key=lambda f: -ex[f].mean()))))
            if st == E and c == 'base':
                G.setdefault('verdict', {})[v] = (rank, ci(tid, ex[M] - ex[ru])[1])
        sys.stdout.flush()


if __name__ == '__main__':
    for v in (188, 186):
        run(v, *S.load(sys.argv[1], v))
    r188, lo188 = G['verdict'][188]; r186, _ = G['verdict'][186]
    if r188 != 1:
        lab = 'FAILS'
    elif r186 == 1 and lo188 > 0:
        lab = 'SURVIVES'
    else:
        lab = 'WEAKENED'
    print('READING (rule fixed before running): m5 rank v188 %d, v186 %d; v188 template CI lower vs runner-up %+.3f -> %s'
          % (r188, r186, lo188, lab))
