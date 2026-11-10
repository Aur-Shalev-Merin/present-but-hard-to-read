"""v235 pilot 2 readout checks (CPU; docs/v235_design.md). Probe checks only, no clamp.

From v235_states.npz (1,038 known cities): cities split in two halves within each country (seed 2352); nearest class mean
(centred class means of the country) fitted on one half, tested on the other, both ways.
  - country readout at e1 and last, per saved layer
  - copy ceiling: the same readout from the mean layer-0 embedding of the city-name tokens (name0)
  - share of the country subspace left after projecting off the copy: U = orthonormal basis of the centred class means at
    e1 (24 dims), C = the same from name0; share = ||U - C C^T U||^2 / 24 (v216's definition)

  python v235_pilot2_readout.py <dir with v235_states.npz>
"""
import glob, json, os, sys
import numpy as np


def halves(y, seed=2352):
    rng = np.random.default_rng(seed); h = np.zeros(len(y), int)
    for c in np.unique(y):
        ix = np.where(y == c)[0]; rng.shuffle(ix); h[ix[len(ix) // 2:]] = 1
    return h


def means(X, y, K):
    mu = X.mean(0); return np.stack([X[y == c].mean(0) for c in range(K)]) - mu, mu


def ncm(X, y, h, K):
    acc = []
    for a in (0, 1):
        M, mu = means(X[h == a], y[h == a], K); T = X[h != a] - mu
        pred = np.argmin(((T[:, None, :] - M[None]) ** 2).sum(-1), 1); acc.append(np.mean(pred == y[h != a]))
    return float(np.mean(acc))


def basis(X, y, K):
    M, _ = means(X, y, K); u, s, _ = np.linalg.svd(M.T, full_matrices=False); return u[:, :K - 1]


def main(root):
    z = np.load(glob.glob(os.path.join(root, '**', 'v235_states.npz'), recursive=True)[0])
    y = z['y']; K = int(y.max()) + 1; h = halves(y); L = [int(l) for l in z['save_layers']]
    e1, last, n0 = z['e1'].astype(np.float32), z['last'].astype(np.float32), z['name0'].astype(np.float32)
    out = {'n': int(len(y)), 'K': K, 'chance': 1 / K, 'copy_ceiling_name0': ncm(n0, y, h, K), 'layers': L,
           'e1': {}, 'last': {}, 'share_left_e1': {}}
    print('%d cities, %d countries, chance %.3f | copy ceiling (name tokens, layer-0 embedding mean) %.3f' % (
        len(y), K, 1 / K, out['copy_ceiling_name0']))
    for j, l in enumerate(L):
        out['e1'][l] = ncm(e1[:, j], y, h, K); out['last'][l] = ncm(last[:, j], y, h, K)
        sh = []
        for a in (0, 1):
            U = basis(e1[h == a, j], y[h == a], K); C = basis(n0[h == a], y[h == a], K)
            sh.append(float(np.linalg.norm(U - C @ (C.T @ U)) ** 2 / U.shape[1]))
        out['share_left_e1'][l] = float(np.mean(sh))
        print('layer %2d: readout e1 %.3f  last %.3f | share of e1 country subspace left off the copy %.3f' % (
            l, out['e1'][l], out['last'][l], out['share_left_e1'][l]))
    json.dump(out, open(os.path.join(root, 'v235_pilot2_readout.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
