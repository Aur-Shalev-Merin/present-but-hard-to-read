"""v188 post-hoc check (exploratory; does not change the frozen labels): the shown answer digit in the copy ceiling.

At P1 the gold answer's tens digit has already been fed in, and P1 is that token. Within a family the tens digit is
base + carry - borrow (carry: a_u + b_u >= 10; borrow: (a+b)_u < c_u), so it carries information about a + b. The frozen
copy ceiling holds only one-hot (a_u, b_u, c_u). Here the H2 statistics (word problems, P1, mod 5) are recomputed with:
  base   one-hot (a_u, b_u, c_u)                      [reproduces the frozen numbers]
  shown  base + one-hot of the shown tens digit       [everything visible as copied tokens at P1]
  cb     base + carry + borrow + carry*borrow          [stricter: more than is visible]
Run on v188 (primary) and v186 (old templates, mod 5 as in v188_calibrate.py). Same probe, folds and bootstrap.

  python v188_shown_digit_check.py <dir with v188_states.npy/v188_meta.json and v186_states.npy/v186_meta.json>
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v186_analyze as A
import v178_analyze as V
import v188_analyze as B
from v188_calibrate import add_mod5

LABS = ('m5', 'rnd5', 'amb5', 'apc5', 'abmc5', 'abc5')
G = {}


def init(X, its, E):
    G['X'], G['its'], G['E'] = X, its, E


def feats(its, kind):
    Z = A.onehot(its)
    if kind == 'base':
        return Z
    au, bu, cu = (np.array([i[k] % 10 for i in its]) for k in ('a', 'b', 'c'))
    carry = (au + bu >= 10).astype(np.float32); borrow = ((au + bu) % 10 < cu).astype(np.float32)
    if kind == 'shown':
        t = np.zeros((len(its), 10), np.float32); t[np.arange(len(its)), [i['answer'] // 10 for i in its]] = 1
        return np.hstack([Z, t])
    return np.hstack([Z, np.stack([carry, borrow, carry * borrow], 1)])


def job(args):
    st, lab, feat = args
    idx = np.array([k for k, i in enumerate(G['its']) if i['set'] == 'main']); its = [G['its'][k] for k in idx]
    fam = np.array([i['fam'] for i in its]); tid = np.array([i['tid'] for i in its])
    y = np.array([i['labels'][lab] for i in its])
    Z = np.asarray(G['X'][idx, 1, st], dtype=np.float32) if feat == 'state' else feats(its, feat)
    return args, (V.probe_predict(A.demean(Z, fam), y, tid, V.folds(tid)) == y).astype(np.int8)


def load(root, v):
    f = sorted(glob.glob(os.path.join(root, '**', 'v%d_states.npy' % v), recursive=True))[0]
    meta = json.load(open(os.path.join(os.path.dirname(f), 'v%d_meta.json' % v)))
    items = [i for i in json.load(open('v%d_items.json' % v))['items'] if i['set'] in ('main', 'bare')]
    assert len(items) == len(meta['items'])
    for m, i in zip(meta['items'], items):                       # attach the full numbers; check the order matches
        assert (m['set'], m['fam'], m['tid'], m['labels']['m']) == (i['set'], i['fam'], i['tid'], i['labels']['m'])
        m.update(a=i['a'], b=i['b'], c=i['c'], answer=i['answer'])
        m['labels'] = add_mod5(m['labels']) if 'm5' not in m['labels'] else m['labels']
    return np.load(f, mmap_mode='r'), meta


def run(v, X, meta):
    S = meta['states']; E = S.index('end4'); its = meta['items']
    main = [i for i in its if i['set'] == 'main']; fam = np.array([i['fam'] for i in main])
    au, bu, cu = (np.array([i[k] % 10 for i in main]) for k in ('a', 'b', 'c'))
    print('v%d: carry rate %.3f, borrow rate %.3f, shown tens digits per family %.2f' % (
        v, np.mean(au + bu >= 10), np.mean((au + bu) % 10 < cu),
        np.mean([len({i['answer'] // 10 for i in main if i['fam'] == f}) for f in np.unique(fam)])), flush=True)
    sts = [E, S.index('end3')]
    jobs = [(st, l, 'state') for st in sts for l in LABS] + [(E, l, f) for l in LABS for f in ('base', 'shown', 'cb')]
    with Pool(4, initializer=init, initargs=(X, its, E)) as pool:
        res = dict(pool.map(job, jobs))
    for st in sts:
        for ft in ('base', 'shown', 'cb'):
            ex = {l: res[(st, l, 'state')].astype(float) - res[(E, l, ft)] for l in LABS}
            bx = max(('amb5', 'apc5', 'abmc5', 'abc5'), key=lambda l: ex[l].mean())
            dc, dx, dr = B.ci(fam, ex['m5']), B.ci(fam, ex['m5'] - ex[bx]), B.ci(fam, ex['m5'] - ex['rnd5'])
            lab = ('REPLICATED' if dc[0] >= 0.09 and dc[1] > 0 and dx[0] >= 0.025 and dx[1] > 0 and dr[1] > 0 else
                   'NOT-REPLICATED' if dc[2] < 0.09 or dx[2] < 0.025 else 'UNCLEAR')
            print('v%d P1 %s ceiling=%-5s | ceil %s | excess %s' % (v, S[st], ft, ' '.join(
                '%s %.3f' % (l, res[(E, l, ft)].mean()) for l in LABS), ' '.join('%s %+.3f' % (l, ex[l].mean()) for l in LABS)))
            print('v%d P1 %s ceiling=%-5s | D_ceil %+.3f [%+.3f, %+.3f] | D_XOR vs %s %+.3f [%+.3f, %+.3f] | D_rnd %+.3f '
                  '[%+.3f, %+.3f] | at v188 bars (0.09/0.025): %s' % ((v, S[st], ft) + dc + (bx,) + dx + dr + (lab,)),
                  flush=True)


if __name__ == '__main__':
    for v in (188, 186):
        run(v, *load(sys.argv[1], v))
