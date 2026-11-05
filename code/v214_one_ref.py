"""v214 one-reference contrast (docs/prereg_v214_one_ref.md). Does ONE sibling prompt (the same question with a fixed
other digit r) read the copied digit nearly as well as the 10-variant minimal-pair contrast?

For each family (one question, ten variants differing only in the copied digit), at the last prompt token:
  mp    the minimal-pair contrast: subtract the family's 10-variant mean (as v181/v202/v205), labels 0-9 (chance 0.10)
  one   subtract the family's variant with digit r, drop that variant, labels the other nine (chance 1/9); run for each
        r = 0..9 and averaged over r
  std   no subtraction (the single-prompt probe, for reference)
Probe: v178_analyze.probe_predict (PCA-256 + logistic, C by inner grouped 3-fold CV), outer 5 folds grouped by
question (B3 data) or template (v181 data), the same folds for every mode. Saved states only (no extraction).

Label per model on its PRIMARY data (GSM8K test-question families), chance-corrected accuracy k = (acc - c)/(1 - c),
averaged over Q layers (and over r for one); D = k_one - k_mp:
  UNSCORABLE    mp < 0.50 (the digit is not shown present)
  ONE-SUFFICES  D >= -0.05
  NEEDS-FAMILY  one < 0.50
  PARTIAL       otherwise

  python v214_one_ref.py <root>
  python v214_one_ref.py --selftest
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v178_analyze as V

QQ = ['l7', 'l14', 'l21', 'l28']
QO = ['end1', 'end2', 'end3', 'end4']
# name, kernel slug, file stem, Q, family/group keys, source filter, role
DATA = [('qwen1.5b', 'astrocyte-v205-b3', 'v205_b3', QQ, ('idx', 'idx'), None, 'primary'),
        ('ouro1.4b', 'astrocyte-v206-b3', 'v206_b3', QO, ('idx', 'idx'), None, 'primary'),
        ('qwen7b', 'astrocyte-v213-x-qwen7b', 'v213_b3', QQ, ('idx', 'idx'), 'gsm', 'primary'),
        ('ouro2.6b', 'astrocyte-v213-x-ouro26', 'v213_b3', QO, ('idx', 'idx'), 'gsm', 'primary'),
        ('qwen7b-orca', 'astrocyte-v213-x-qwen7b', 'v213_b3', QQ, ('idx', 'idx'), 'orca', 'secondary'),
        ('ouro2.6b-orca', 'astrocyte-v213-x-ouro26', 'v213_b3', QO, ('idx', 'idx'), 'orca', 'secondary'),
        ('qwen1.5b-v181', 'astrocyte-v202-qwen', 'v202_v181', QQ, ('fam', 'tid'), None, 'secondary'),
        ('qwen7b-v181', 'astrocyte-v213-x-qwen7b', 'v213_v181', QQ, ('fam', 'tid'), None, 'secondary'),
        ('ouro2.6b-v181', 'astrocyte-v213-x-ouro26', 'v213_v181', QO, ('fam', 'tid'), None, 'secondary')]
MODES = ['std', 'mp'] + ['r%d' % r for r in range(10)]
G = {}


def init(d):
    G.update(d)


def chance(mode):
    return 1 / 9 if mode.startswith('r') else 0.10


def job(args):
    name, qi, mode = args
    X = np.asarray(G[name + '_X'][:, qi], np.float32); y, fam, grp = G[name + '_y'], G[name + '_fam'], G[name + '_grp']
    fold = V.folds(grp)
    if mode == 'mp':
        X = X.copy()
        for f in np.unique(fam):
            X[fam == f] -= X[fam == f].mean(0)
    elif mode.startswith('r'):
        r = int(mode[1:]); keep = y != r
        ref = {}
        for f in np.unique(fam):
            m = (fam == f) & (y == r); assert m.sum() == 1, (name, f, r, m.sum()); ref[f] = X[m][0]
        X = X[keep] - np.stack([ref[f] for f in fam[keep]]); y, grp, fold = y[keep], grp[keep], fold[keep]
    return args, float((V.probe_predict(X, y, grp, fold) == y).mean())


def label(mp, one, kmp, kone):
    D = kone - kmp
    return ('UNSCORABLE' if mp < 0.50 else 'ONE-SUFFICES' if D >= -0.05 else 'NEEDS-FAMILY' if one < 0.50 else 'PARTIAL'), D


def run(data, procs=4, say=print):
    """data: list of (name, X [n, nQ, d], y, fam, grp, role). Returns {name: label}."""
    arrs = {}
    for name, X, y, fam, grp, role in data:
        arrs.update({name + '_X': X, name + '_y': y, name + '_fam': fam, name + '_grp': grp})
    jobs = [(name, qi, mode) for name, X, *_ in data for qi in range(X.shape[1]) for mode in MODES]
    with Pool(procs, initializer=init, initargs=(arrs,)) as pool:
        res = dict(pool.map(job, jobs, chunksize=1))
    out = {}
    say('%-15s %-5s %6s %6s %6s %6s  (std / mp / one, mean over r / one min-max over r)' % ('data', 'layer', 'std', 'mp', 'one', ''))
    for name, X, y, fam, grp, role in data:
        for qi in range(X.shape[1]):
            o = [res[(name, qi, 'r%d' % r)] for r in range(10)]
            say('%-15s %-5d %6.3f %6.3f %6.3f  %.3f-%.3f' % (name, qi, res[(name, qi, 'std')], res[(name, qi, 'mp')], np.mean(o), min(o), max(o)))
        nq = X.shape[1]
        mp = np.mean([res[(name, qi, 'mp')] for qi in range(nq)])
        one = np.mean([res[(name, qi, 'r%d' % r)] for qi in range(nq) for r in range(10)])
        kmp, kone = (mp - 0.10) / 0.90, (one - 1 / 9) / (8 / 9)
        lab, D = label(mp, one, kmp, kone)
        say('%-15s [%s] mp %.3f, one %.3f (chance-corrected %.3f vs %.3f, D %+.3f) -> %s' % (name, role, mp, one, kmp, kone, D, lab))
        out[name] = lab
    prim = [out[n] for n, *_, role in data if role == 'primary']
    say('PRIMARY: %s' % ', '.join('%s %s' % (n, out[n]) for n, *_, role in data if role == 'primary'))
    return out


def load(root, name, slug, stem, q, keys, source, role):
    fs = [f for f in glob.glob(os.path.join(root, '**', stem + '_states.npy'), recursive=True) if slug in f.replace('\\', '/')]
    assert len(fs) == 1, (name, fs)
    meta = json.load(open(fs[0].replace('_states.npy', '_meta.json'))); its = meta['items']
    X = np.load(fs[0], mmap_mode='r'); si = meta['slots'].index('last')
    st = [meta['states'].index(s if s in meta['states'] else {'end1': 'l7'}.get(s, s)) for s in q]
    keep = np.array([source is None or i.get('source') == source for i in its])
    idx = np.where(keep)[0]
    Xq = np.stack([np.asarray(X[idx, si, s], np.float32) for s in st], 1)
    sel = [its[k] for k in idx]
    print('%-15s %s: %d prompts, %d families, states %s' % (name, os.path.relpath(fs[0], root), len(idx),
          len(set(i[keys[0]] for i in sel)), [meta['states'][s] for s in st]), flush=True)
    return (name, Xq, np.array([i['digit'] for i in sel]), np.array([i[keys[0]] for i in sel]),
            np.array([i[keys[1]] for i in sel]), role)


def selftest():
    """additive world (a shared digit code plus a family context) -> ONE-SUFFICES; item-noise world (each variant has
    its own large noise, which one reference carries in full and a 10-variant mean mostly averages out) -> not."""
    rng = np.random.default_rng(0); d = 64; code = rng.standard_normal((10, d)); ok = True
    for world, noise, want in (('additive', 0.1, 'ONE-SUFFICES'), ('item-noise', 2.0, None)):
        nf = 150; fam = np.repeat(np.arange(nf), 10); y = np.tile(np.arange(10), nf)
        ctx = rng.standard_normal((nf, d)) * 5
        X = (ctx[fam] + code[y] + noise * rng.standard_normal((len(y), d)))[:, None, :].repeat(2, 1).astype(np.float32)
        lab = run([(world, X, y, fam, fam, 'primary')], say=lambda s: print('  ' + s))[world]
        good = lab == want if want else lab != 'ONE-SUFFICES'
        print('selftest %-10s -> %-13s %s' % (world, lab, 'OK' if good else 'FAIL'), flush=True); ok &= good
    return ok


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        sys.exit(0 if selftest() else 1)
    run([load(sys.argv[1], *spec) for spec in DATA])
