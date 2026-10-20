"""v209 patch plan (docs/prereg_v209_ouro_early_patch.md): the inputs v209_patch.py needs beyond v194's frozen plan.

Reads v194's Part A states (v194_states.npy, N x 2 slots x 8 states x d; slot 1 = P1) and v194's frozen patch plan, and
writes v209_patch_plan.npz with
  U_{ab,amb,ans}_{h}_e1   end-of-loop-1 subspaces, fitted exactly as v194_analyze.patch_plan fits its end3/end2 ones
                          (family-demeaned P1 states of the OTHER template half, class means of the label mod 10, top 9
                          right singular vectors);
  mu_e3_<fam>             each plus family's mean P1 state at the end of loop 3 (for ablating the answer subspace);
and copies every array of v194_patch_plan.npz unchanged (U_*_h at end3, U_*_h_e2 at end2, U_rand_h).
Only control quantities are computed here (subspaces and family means); no patching result is read or produced.

  python v209_plan.py <dir containing v194_states.npy and v194_meta.json> <dir containing v194_patch_plan.npz>
"""
import glob, json, os, sys
import numpy as np

import v186_analyze as A


def basis(X, its, ks, key, st):
    """v194_analyze.patch_plan.basis, verbatim apart from the arguments."""
    Z = np.asarray(X[np.array(ks), 1, st], dtype=np.float64); fam = np.array([its[k]['fam'] for k in ks])
    Z = A.demean(Z, fam); y = np.array([key(its[k]) for k in ks])
    mu = np.stack([Z[y == v].mean(0) for v in range(10)]); mu -= mu.mean(0)
    return np.linalg.svd(mu, full_matrices=False)[2][:9].T.astype(np.float32)


KEYS = {'ab': lambda i: (i['labels']['a'] + i['labels']['b']) % 10,
        'amb': lambda i: (i['labels']['a'] - i['labels']['b']) % 10,
        'ans': lambda i: i['labels']['ans']}


def build(X, meta, old):
    its = meta['items']; S = meta['states']; E1, E3 = S.index('end1'), S.index('end3')
    plus = [k for k, i in enumerate(its) if i['set'] == 'main' and i['op'] == 'plus']
    half = {k: int(its[k]['tid'] >= 6) for k in plus}
    plan = {k: old[k] for k in old.files}
    for h in (0, 1):
        other = [k for k in plus if half[k] != h]
        for nm, key in KEYS.items():
            plan['U_%s_%d_e1' % (nm, h)] = basis(X, its, other, key, E1)
    fams = sorted({its[k]['fam'] for k in plus})
    for f in fams:
        ks = np.array([k for k in plus if its[k]['fam'] == f])
        plan['mu_e3_%s' % f] = np.asarray(X[ks, 1, E3], dtype=np.float64).mean(0).astype(np.float32)
    return plan, len(plus), len(fams)


if __name__ == '__main__':
    f = sorted(glob.glob(os.path.join(sys.argv[1], '**', 'v194_states.npy'), recursive=True))[0]
    meta = json.load(open(os.path.join(os.path.dirname(f), 'v194_meta.json')))
    old = np.load(sorted(glob.glob(os.path.join(sys.argv[2], '**', 'v194_patch_plan.npz'), recursive=True))[0])
    plan, n, nf = build(np.load(f, mmap_mode='r'), meta, old)
    out = os.environ.get('OUT_DIR', '.')
    np.savez_compressed(os.path.join(out, 'v209_patch_plan.npz'), **plan)
    print('v209 plan: %d plus items, %d families; arrays %d (%d from v194)' % (n, nf, len(plan), len(old.files)))
    for h in (0, 1):
        for nm in KEYS:
            a, b = plan['U_%s_%d_e1' % (nm, h)], plan['U_%s_%d' % (nm, h)]
            sv = np.linalg.svd(a.T @ b, compute_uv=False)
            print('half %d %-3s: end1 vs end3 subspace, mean cos^2 of principal angles %.3f' % (h, nm, float((sv ** 2).mean())))
        for nm in ('amb', 'ans'):
            sv = np.linalg.svd(plan['U_ab_%d_e1' % h].T @ plan['U_%s_%d_e1' % (nm, h)], compute_uv=False)
            print('half %d end1: ab vs %-3s mean cos^2 %.3f' % (h, nm, float((sv ** 2).mean())))
