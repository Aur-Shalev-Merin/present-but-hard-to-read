"""v208 analysis (docs/prereg_v208_input_sensitivity.md).
  python v208_analyze.py <dir> [<dir> ...]     every v208_<model>_s<k>_{lin,full}.npy / _meta.json found below the dirs
  python v208_analyze.py --selftest            synthetic worlds with known answers
Per prompt, slot and state: g2[X] = mean over the directions of class X of |r(u)|^2 (linear response, squared gain).
  R_tok = g2[PAIR] / g2[ISO]   (selectivity for directions in which token embeddings differ; 1 = no selectivity)
  D     = g2[DIG]  / g2[PAIR]  (digit-difference directions against word-difference directions, per unit input)
Medians over prompts, 95% percentile bootstrap intervals over prompts (2,000 resamples, seed 208).
Labels at the last slot over the four decision states (qwen l7-l28, ouro end1-end4), per model:
  Q1  TUNED   R_tok >= 3 and lower bound > 1.5 at >= 3 of 4 states;  FLAT  R_tok <= 1.5 and upper bound < 2 at >= 3 of 4;
      else MIXED.
  Q2  LOW-GAIN  D <= 0.5 and upper bound < 0.7 at >= 3 of 4;  DISTANCE-ONLY  D >= 0.8 and lower bound > 0.6 at >= 3 of 4;
      else MIXED.
  Linearity (bearing on the finite changes): median over prompts and changes of cos(dh, |delta| r(u)) at the last slot;
      LINEAR if >= 0.7 for SWAP and NOISE at >= 3 of 4 states, else LOCAL-ONLY.
Descriptive: the same ratios at slots j and qmark; g2[SWAPD] / g2[NOISED]; full-size squared gains |dh|^2 / |delta|^2.
"""
import glob, json, os, sys
import numpy as np

DECISION = {'qwen': ['l7', 'l14', 'l21', 'l28'], 'ouro': ['end1', 'end2', 'end3', 'end4']}
B, SEED = 2000, 208


def class_slices(classes):
    out, a = {}, 0
    for name, n in classes:
        out[name] = slice(a, a + n); a += n
    return out


def g2(LIN, sl):
    """LIN: n x dirs x slots x states x d -> {class: n x slots x states} mean squared gain."""
    sq = (LIN.astype(np.float64) ** 2).sum(-1)
    return {c: sq[:, s].mean(1) for c, s in sl.items()}


def med_ci(x, rng):
    x = np.asarray(x, float); m = float(np.median(x))
    bs = np.median(x[rng.integers(0, len(x), (B, len(x)))], axis=1)
    return m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def label_q1(rows):
    if sum(m >= 3 and lo > 1.5 for m, lo, hi in rows) >= 3:
        return 'TUNED'
    if sum(m <= 1.5 and hi < 2 for m, lo, hi in rows) >= 3:
        return 'FLAT'
    return 'MIXED'


def label_q2(rows):
    if sum(m <= 0.5 and hi < 0.7 for m, lo, hi in rows) >= 3:
        return 'LOW-GAIN'
    if sum(m >= 0.8 and lo > 0.6 for m, lo, hi in rows) >= 3:
        return 'DISTANCE-ONLY'
    return 'MIXED'


def linearity(LIN, FUL, norms, sl, fsl):
    """cos(dh, |delta| r(u)) per prompt and full change; DIG uses DIG, SWAP uses SWAPD, NOISE uses NOISED (same order)."""
    pair = {'DIG': 'DIG', 'SWAP': 'SWAPD', 'NOISE': 'NOISED'}; out = {}
    for fc, s in fsl.items():
        r = LIN[:, sl[pair[fc]]].astype(np.float64); dh = FUL[:, s].astype(np.float64)
        nrm = norms[:, s][..., None, None, None]; pred = r * nrm
        cos = (pred * dh).sum(-1) / (np.linalg.norm(pred, axis=-1) * np.linalg.norm(dh, axis=-1) + 1e-30)
        ratio = np.linalg.norm(dh, axis=-1) / (np.linalg.norm(pred, axis=-1) + 1e-30)
        out[fc] = (cos, ratio)                                   # each n x changes x slots x states
    return out


def analyze(LIN, FUL, norms, meta, out=print):
    rng = np.random.default_rng(SEED); model = meta['model']; states = meta['states']; slots = meta['slots']
    sl = class_slices(meta['classes']); fsl = class_slices(meta['full']); G = g2(LIN, sl)
    dec = [states.index(s) for s in DECISION[model]]; res = {}
    out('== %s: %d prompts | states %s | step %.4f' % (model, LIN.shape[0], states, meta['step']))
    for si, slot in enumerate(slots):
        R = G['PAIR'][:, si] / G['ISO'][:, si]; D = G['DIG'][:, si] / G['PAIR'][:, si]
        SN = G['SWAPD'][:, si] / G['NOISED'][:, si]
        rows = {'R_tok': [med_ci(R[:, k], rng) for k in range(len(states))],
                'D': [med_ci(D[:, k], rng) for k in range(len(states))],
                'SWAPD/NOISED': [med_ci(SN[:, k], rng) for k in range(len(states))]}
        out('-- slot %s%s' % (slot, '  (decision slot)' if slot == 'last' else '  (descriptive)'))
        for name, rr in rows.items():
            out('   %-13s ' % name + '  '.join('%s %.3g [%.3g, %.3g]' % (states[k], *rr[k]) for k in range(len(states))))
        res[slot] = rows
    last = res['last']
    q1 = label_q1([last['R_tok'][k] for k in dec]); q2 = label_q2([last['D'][k] for k in dec])
    L = linearity(LIN, FUL, norms, sl, fsl); si = slots.index('last'); lin_rows = {}
    for fc, (cos, ratio) in L.items():
        lin_rows[fc] = [(float(np.median(cos[:, :, si, k])), float(np.median(ratio[:, :, si, k]))) for k in range(len(states))]
        out('   linearity %-5s cos / |dh| over prediction: ' % fc +
            '  '.join('%s %.2f / %.2f' % (states[k], *lin_rows[fc][k]) for k in range(len(states))))
    linear = 'LINEAR' if all(sum(lin_rows[fc][k][0] >= 0.7 for k in dec) >= 3 for fc in ('SWAP', 'NOISE')) else 'LOCAL-ONLY'
    fg = {}
    for fc, s in fsl.items():
        dh2 = (FUL[:, s].astype(np.float64) ** 2).sum(-1)[:, :, si]          # n x changes x states
        fg[fc] = (dh2 / (norms[:, s][..., None] ** 2)).mean(1)               # n x states
    for a_, b_ in (('SWAP', 'NOISE'), ('DIG', 'SWAP')):
        rr = [med_ci(fg[a_][:, k] / fg[b_][:, k], rng) for k in range(len(states))]
        out('   full-size squared gain %s/%s: ' % (a_, b_) + '  '.join('%s %.3g [%.3g, %.3g]' % (states[k], *rr[k])
                                                                       for k in range(len(states))))
    out('Q1 (selectivity, last slot, %s): %s' % ('/'.join(DECISION[model]), q1))
    out('Q2 (digit vs word directions, last slot): %s' % q2)
    out('Linearity (bearing on the finite changes): %s' % linear)
    return dict(q1=q1, q2=q2, linear=linear, rows=res, lin=lin_rows)


def load(dirs):
    by = {}
    for dd in dirs:
        for f in glob.glob(os.path.join(dd, '**', 'v208_*_s*_meta.json'), recursive=True):
            m = json.load(open(f)); base = f[:-len('_meta.json')]
            by.setdefault(m['model'], []).append((m, np.load(base + '_lin.npy'), np.load(base + '_full.npy')))
    out = {}
    for model, parts in by.items():
        parts.sort(key=lambda p: p[0]['items'][0]['k'] if p[0]['items'] else 0)
        meta = dict(parts[0][0]); meta['items'] = sum((p[0]['items'] for p in parts), [])
        assert all(p[0]['step'] == meta['step'] and p[0]['classes'] == meta['classes'] for p in parts)
        LIN = np.concatenate([p[1] for p in parts]); FUL = np.concatenate([p[2] for p in parts])
        norms = np.array([it['full_input_norms'] for it in meta['items']])
        out[model] = (LIN, FUL, norms, meta)
    return out


def selftest():
    """Worlds: r(u) = J u at every slot and state, J = U diag(sig) V^T per prompt; token embeddings live mainly in a
    k-dim subspace E. TUNED: J's large singular directions span E. FLAT: J isotropic. LOW: digit differences lie in a
    subspace J nearly ignores. Full changes = J delta (exactly linear) or with a nonlinear term (LOCAL-ONLY)."""
    rng = np.random.default_rng(0); d, n, k = 48, 24, 8
    classes = [['DIG', 9], ['PAIR', 30], ['ISO', 64], ['SWAPD', 10], ['NOISED', 10]]
    full = [['DIG', 9], ['SWAP', 10], ['NOISE', 10]]
    ok = True
    for world, want in (('tuned', ('TUNED', 'DISTANCE-ONLY', 'LINEAR')), ('flat', ('FLAT', 'DISTANCE-ONLY', 'LINEAR')),
                        ('low', ('TUNED', 'LOW-GAIN', 'LINEAR')), ('nonlin', ('TUNED', 'DISTANCE-ONLY', 'LOCAL-ONLY'))):
        Eb = np.linalg.qr(rng.standard_normal((d, d)))[0]; E = Eb[:, :k]; Dg = Eb[:, k:k + 3]
        def tokdiff(m, digit=False):
            if digit and world == 'low':
                v = rng.standard_normal((m, 3)) @ Dg.T
            else:
                v = rng.standard_normal((m, k)) @ E.T + 0.05 * rng.standard_normal((m, d))
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        LIN = np.zeros((n, 123, 3, 5, d), np.float32); FUL = np.zeros((n, 29, 3, 5, d), np.float32); norms = np.zeros((n, 29))
        for p in range(n):
            if world == 'flat':
                J = rng.standard_normal((d, d)) / np.sqrt(d)
            else:
                sig = np.full(d, 0.1); sig[:k] = 1.0
                if world == 'low':
                    sig[k:k + 3] = 0.05
                J = np.linalg.qr(rng.standard_normal((d, d)))[0] @ np.diag(sig) @ Eb.T
            dirs = np.concatenate([tokdiff(9, True), tokdiff(30), rng.standard_normal((64, d)), tokdiff(10),
                                   rng.standard_normal((10, d))])
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            r = dirs @ J.T; LIN[p] = r[:, None, None, :]
            nr = np.concatenate([np.full(9, 0.7), np.full(10, 1.4), np.full(10, 1.0)]); norms[p] = nr
            fd = np.concatenate([dirs[:9], dirs[103:113], dirs[113:123]]) * nr[:, None]; dh = fd @ J.T
            if world == 'nonlin':
                dh = dh + 3 * np.linalg.norm(dh, axis=1, keepdims=True) * rng.standard_normal(dh.shape) / np.sqrt(d)
            FUL[p] = dh[:, None, None, :]
        meta = dict(model='qwen', states=['l4', 'l7', 'l14', 'l21', 'l28'], slots=['j', 'qmark', 'last'], classes=classes,
                    full=full, step=0.02)
        res = analyze(LIN, FUL, norms, meta, out=lambda *a: None)
        got = (res['q1'], res['q2'], res['linear']); good = got == want; ok &= good
        print('selftest world %-7s want %s got %s %s' % (world, want, got, 'OK' if good else 'FAIL'))
    print('SELFTEST', 'ALL OK' if ok else 'FAILED')
    return ok


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        sys.exit(0 if selftest() else 1)
    data = load(sys.argv[1:])
    for model in ('qwen', 'ouro'):
        if model in data:
            analyze(*data[model])
