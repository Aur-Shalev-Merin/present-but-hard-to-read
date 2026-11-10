"""v234 clamp plan (CPU; docs/prereg_v234_kudo_equal_answer.md). From the extracted states: per position (eq, pre, last),
band layer (hidden_states 12, 16, 20, 24) and half of the families, family-centred class means fitted on the OTHER half
(v216's basis()): U = b (classes 0-7, 7 dims), D = the d2 copy, K = the k copy, An = the answer; Uo = U projected off
D and K (left singular vectors of U - QQ^T U with singular value > 1e-3 of the largest; v216's procedure). d1 and the
answer are shared within a pair, so their codes carry no difference and are not projected out. R = a random subspace
with Uo's dimension. Reports the share of U left after the projection, ||U - QQ^T U||^2 / dim(U) (v216 / v233's
definition), and the overlaps. Pairs: up to 1,000 per position, base and partner both answered correctly, from
v234_items.partners (same family, version and d1; different d2; equal answer), seed 2341.

  python v234_plan.py <dir with v234_fam_states.npy, v234_fam_meta.json>   -> v234_plan.npz, v234_pairs.json
"""
import glob, json, os, sys
import numpy as np

import v234_items as I

OUT = os.environ.get('OUT_DIR', '.'); LAYERS = list(range(0, 29, 4)); BAND = [12, 16, 20, 24]; POS = ['eq', 'pre', 'last']


def demean(X, fam):
    X = np.array(X, dtype=np.float32)
    for f in np.unique(fam):
        X[fam == f] -= X[fam == f].mean(0)
    return X


def basis(Z, m, y):
    M = np.stack([Z[m & (y == v)].mean(0) for v in range(10) if (m & (y == v)).any()]); M -= M.mean(0)
    return np.linalg.svd(M, full_matrices=False)[2][:len(M) - 1].T


def plan(root, say=print, per_pos=1000):
    X = np.load(glob.glob(os.path.join(root, '**', 'v234_fam_states.npy'), recursive=True)[0], mmap_mode='r')
    meta = json.load(open(glob.glob(os.path.join(root, '**', 'v234_fam_meta.json'), recursive=True)[0]))
    assert not meta.get('pilot'), 'pilot states'
    its = I.build(); assert len(its) == X.shape[0]
    ok = np.array([m['correct'] for m in meta['items']]); fam = np.array([i['fam'] for i in its])
    say('items %d, answered correctly %.3f (plus %.3f, minus %.3f; o2 + %.3f, o2 - %.3f)' % (
        len(its), ok.mean(), *[ok[[i['ver'] == v for i in its]].mean() for v in ('plus', 'minus')],
        *[ok[[i['o2'] == o for i in its]].mean() for o in '+-']))
    fams = sorted(set(fam.tolist())); half = {f: int(j % 2) for j, f in enumerate(np.random.default_rng(2341).permutation(fams))}
    lab = {k: np.array([i[k] for i in its]) for k in ('B', 'A', 'd2', 'k')}
    rr = np.random.default_rng(2342); arr = {}; geo = []
    for p in POS:
        pi = POS.index(p)
        for L in BAND:
            Z = demean(np.asarray(X[:, pi, LAYERS.index(L)], np.float32), fam)
            for h in (0, 1):
                m = np.array([half[f] != h for f in fam])
                U, D, K, An = (basis(Z, m, lab[k]) for k in ('B', 'd2', 'k', 'A'))
                Q = np.linalg.qr(np.concatenate([D, K], 1))[0]
                Uo, sv, _ = np.linalg.svd(U - Q @ (Q.T @ U), full_matrices=False); Uo = Uo[:, sv > 1e-3 * sv.max()]
                for k, v in (('U', U), ('D', D), ('K', K), ('An', An), ('Uo', Uo)):
                    arr['%s_%s_%d_%d' % (k, p, h, L)] = v.astype(np.float32)
                left = float(np.linalg.norm(U - Q @ (Q.T @ U)) ** 2 / U.shape[1])
                cd, ck, ca = (float((np.linalg.svd(U.T @ W, compute_uv=False) ** 2).sum() / U.shape[1]) for W in (D, K, An))
                geo.append(dict(pos=p, layer=L, half=h, dims=U.shape[1], dims_o=Uo.shape[1], left=left, in_d2=cd, in_k=ck, in_A=ca))
                say('  %-4s layer %d half %d: b %d dims, kept %d after projection; share of b left %.3f (in d2 copy %.3f, in k copy %.3f, '
                    'in answer %.3f)' % (p, L, h, U.shape[1], Uo.shape[1], left, cd, ck, ca))
            arr['R_%s_%d' % (p, L)] = np.linalg.qr(rr.standard_normal((Z.shape[1], arr['Uo_%s_0_%d' % (p, L)].shape[1])))[0].astype(np.float32)
    al = I.partners(its, ok); rng = np.random.default_rng(2341); pairs = []
    cand = [j for j in sorted(al) if al[j]]
    for p in POS:
        for j in rng.permutation(cand)[:per_pos]:
            j2 = int(rng.choice(al[j])); pairs.append(dict(pos=p, x=int(j), x2=j2, half=half[its[j]['fam']], P=I.prediction(its[j], its[j2])))
    np.savez(os.path.join(OUT, 'v234_plan.npz'), **arr)
    json.dump(dict(positions=POS, band=BAND, geometry=geo, pairs=pairs, n_candidates=len(cand)), open(os.path.join(OUT, 'v234_pairs.json'), 'w'))
    for p in POS:
        say('%s: mean share of b left %.3f' % (p, np.mean([g['left'] for g in geo if g['pos'] == p])))
    say('plan: %d bases, %d pairs (%d base items with a correct equal-answer partner)' % (len(arr), len(pairs), len(cand)))


if __name__ == '__main__':
    plan(sys.argv[1])
