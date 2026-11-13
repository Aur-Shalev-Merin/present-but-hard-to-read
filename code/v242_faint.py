"""v242: the two-hop clamp at a faint signal (docs/prereg_v242_twohop_faint.md). v235's clamp, unchanged in form, with
only the states its subspaces are fitted on degraded, as in v231: the forward pass is untouched, so the model still uses
the country at the city's last token (e1).

Items, prompts, halves, pairs (type 'other', one per city, seed 2353), band (hidden_states 8, 12, 16), position e1,
outcome (argmax over the 25 capitals' first tokens) and the clamp h <- h - V V^T (h - h') are v235's (helpers imported
from the frozen v235_clamp.py).

Dose: the country's population share of the state's variance at e1, with the copy's subspace C (from the city name's
token embeddings) projected out, per band layer and fitting half: sb / (sb + msw) from the ANOVA variance components
(between-class component over between plus within). Native 0.15-0.24 (pilot 2 states). Targets 0.1, 0.03, 0.01, 0.005,
0.002 (Ouro's a+b: about 0.008-0.010). Added variance is drawn once per item; its scale s is set so that the population
share of the degraded fitting states, sb / (sb + msw + s^2 Wn) with Wn the drawn noise's variance, equals the target
(the in-sample estimate on the degraded states, which fluctuates by about 0.001, is reported):
  state (primary): Gaussian with the covariance of the half's within-country residual at that layer;
  iso: isotropic Gaussian.
Cells: native (no noise) + 5 doses x 2 shapes. In each cell the bases are fitted as in v235 on the degraded states of
the half that does not contain x: U (country class means, <= 24 dims), C (copy, from name0, not degraded), Uo = U off C;
R random with Uo's dimension (seed 2423 + cell index).
Rows per pair: none, full (x''s whole state at the band; ground truth), and per cell cUo and cR.

Analysis (per cell): stat = hit(cUo) - hit(cR), hit = x''s capital; cluster bootstrap over x's country, 10,000
resamples, seed 2425, 97.5% interval; v235's label (USED >= 0.10 with lower > 0; NOT-USED upper < 0.02; else UNCLEAR).
Ground truth: hit(full) - hit(cR, native) must be USED, else GROUND-TRUTH-FAILED.
Calibration: a (dose, shape) whose realised share misses its target at any band layer or half by more than 10% (or
0.0005) counts as not USED, and is reported.
Detection limit per shape: the faintest dose at which the cell and every stronger dose, native included, are USED;
none if native is not. Reading (shape state): POWER-AT-OURO-LEVEL if the limit is 0.005 or fainter; AT-THE-LIMIT if
0.01; UNINFORMATIVE if 0.03 or stronger, or none.

  GPU:  OUT_DIR=... python v242_faint.py run         (V242_SMOKE=1: every learned subspace random; mechanics only)
  CPU:  python v242_faint.py analyse <dir>;  python v242_faint.py --selftest;  python v242_faint.py calib <states.npz>
"""
import glob, json, os, sys, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v235_clamp import halves, orth, cmeans, pairs, label, boot, BAND   # frozen in v235

SMOKE = os.environ.get('V242_SMOKE') == '1'
DOSES = [0.1, 0.03, 0.01, 0.005, 0.002]; SHAPES = ['state', 'iso']
CELLS = [('native', None)] + [(s, d) for s in SHAPES for d in DOSES]
CNAME = ['native'] + ['%s_%g' % (s, d) for s in SHAPES for d in DOSES]
ROWS = ['none', 'full'] + [k + ':' + c for c in CNAME for k in ('cUo', 'cR')]


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v242_log.txt'), 'a').write(s + '\n')


def anova(X, y, K):
    """ANOVA variance components summed over dimensions: between-class component sb and within-class mean square msw."""
    n = len(y); mu = X.mean(0); ns = np.array([(y == c).sum() for c in range(K)])
    M = np.stack([X[y == c].mean(0) for c in range(K)])
    ssb = float((ns[:, None] * (M - mu) ** 2).sum()); ssw = float(sum(((X[y == c] - M[c]) ** 2).sum() for c in range(K)))
    msb, msw = ssb / (K - 1), ssw / (n - K); n0 = (n - (ns ** 2).sum() / n) / (K - 1)
    return max((msb - msw) / n0, 0.0), msw


def pop_share(X, y, K):
    """ANOVA estimate of the population share of variance between class means (all dimensions pooled)."""
    sb, msw = anova(X, y, K); return sb / (sb + msw)


def degrade(X, y, K, C, target, shape, rng):
    """X [n, d] float64 states of one fitting half. Noise drawn once per item; its scale is set so that the population
    share (copy projected out) is the target: sb / (sb + msw + s^2 Wn), with sb, msw the clean states' ANOVA components
    and Wn the drawn noise's variance. Returns (X + s E, population share, in-sample estimate on the degraded states)."""
    n, d = X.shape
    if shape == 'state':
        Rw = X - np.stack([X[y == c].mean(0) for c in range(K)])[y]; Rw = Rw - Rw.mean(0)
        E = rng.standard_normal((n, n)) @ Rw / np.sqrt(n - 1)
    else:
        E = rng.standard_normal((n, d)) * np.sqrt((X - X.mean(0)).var(0).mean())
    proj = lambda Z: Z - (Z @ C) @ C.T
    Xp, Ep = proj(X), proj(E)
    sb, msw = anova(Xp, y, K); Wn = float(Ep.var(0, ddof=1).sum())
    s2 = max((sb / target - sb - msw) / Wn, 0.0)   # target above the native share: no noise, and calib_ok() flags the miss
    sc = float(np.sqrt(s2)); popv = sb / (sb + msw + s2 * Wn)
    return X + sc * E, popv, pop_share(Xp + sc * Ep, y, K)


def fit_cells(Sx, N0, y, h, K):
    """Sx: layer -> states [n, d] (e1). Returns bases {(cell, layer, half): {cUo, cR}} and realised shares."""
    B, real = {}, {}
    for half in (0, 1):
        m = h == half; C = orth(cmeans(N0[m], y[m], K).T, cap=K - 1)
        for l in BAND:
            X = np.asarray(Sx[l][m], np.float64)
            for ci, (shape, dose) in enumerate(CELLS):
                if shape == 'native':
                    Xd = X; v = pop_share(X - (X @ C) @ C.T, y[m], K); real[(ci, l, half)] = (v, v)
                else:
                    Xd, pv, ev = degrade(X, y[m], K, C, dose, shape, np.random.default_rng([2421, ci, l, half]))
                    real[(ci, l, half)] = (pv, ev)
                U = orth(cmeans(Xd, y[m], K).T, cap=K - 1); Uo = orth(U - C @ (C.T @ U), cap=U.shape[1])
                R = orth(np.random.default_rng([2423, ci, l, half]).standard_normal((X.shape[1], Uo.shape[1])))
                B[(ci, l, half)] = dict(cUo=Uo, cR=R)
    return B, real


def calib_ok(real):
    bad = []
    for ci, (shape, dose) in enumerate(CELLS):
        if shape == 'native':
            continue
        for (cj, l, half), (v, _) in real.items():
            if cj == ci and abs(v - dose) > max(0.1 * dose, 0.0005):
                bad.append((CNAME[ci], l, half, v))
    return bad


def run():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from v235_pilot import prompt
    torch.set_grad_enabled(False); t0 = time.time()
    D = json.load(open(os.path.join(HERE, 'v235_cities.json'), encoding='utf-8'))
    S = json.load(open(os.path.join(HERE, 'v235_pilot2_items.json'), encoding='utf-8'))
    keys = S['keys']; K = len(keys); caps = [D['countries'][k]['capital'] for k in keys]
    cities, persons = S['all_cities'], S['persons']; y = np.array([keys.index(c['ckey']) for c in cities]); h = halves(y)
    repo = os.environ.get('V242_REPO', 'Qwen/Qwen2.5-7B'); tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    dev = m.get_input_embeddings().weight.device; layers = m.model.layers
    first = [tok.encode(' ' + c, add_special_tokens=False)[0] for c in caps]; assert len(set(first)) == K

    def enc(x):
        p = prompt('city_cap', x); e = tok(p, return_offsets_mapping=True); off = e['offset_mapping']
        st, end = p.rfind(x), p.rfind(x) + len(x) - 1
        t = next(j for j, (u, v) in enumerate(off) if u <= end < v)
        return e['input_ids'], t, [j for j, (u, v) in enumerate(off) if v > st and u <= end]

    def hs(ids):
        return m(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states

    Sx = {l: [] for l in BAND}; N0 = []
    for c in cities:
        ids, t, name = enc(c['city']); H = hs(ids)
        for l in BAND:
            Sx[l].append(H[l][0, t].float().cpu().numpy())
        N0.append(H[0][0, name].float().mean(0).cpu().numpy())
    Sx = {l: np.stack(v) for l, v in Sx.items()}; N0 = np.stack(N0).astype(np.float64)
    log('v242 faint two-hop clamp | smoke %s | %d cities | states in %.0fs' % (SMOKE, len(cities), time.time() - t0))
    B, real = fit_cells(Sx, N0, y, h, K)
    for ci, c in enumerate(CNAME):
        log('  %-12s population share / in-sample estimate by (layer, half): %s' % (c, ' '.join('%d/%d %.4f/%.4f' % (l, hf, *real[(ci, l, hf)]) for l in BAND for hf in (0, 1))))
    bad = calib_ok(real); log('calibration misses: %s' % (bad or 'none'))
    if SMOKE:
        rr = np.random.default_rng(0); B = {k: {c: orth(rr.standard_normal(v[c].shape)) for c in v} for k, v in B.items()}
    Vt = {k: {c: torch.tensor(v[c], dtype=torch.float32) for c in v} for k, v in B.items()}
    P = [q for q in pairs(cities, persons, keys, np.random.default_rng(2353)) if q['type'] == 'other']
    if SMOKE:
        P = P[:int(os.environ.get('V242_N', '30'))]
    st = {'on': False}

    def hook(li):
        hi = li + 1

        def f(mod, args, out):
            hh = out[0] if isinstance(out, tuple) else out
            if st['on'] and hi in BAND:
                t = st['t']; src = st['src'][hi].to(hh.device).float()
                for j, r in enumerate(ROWS):
                    if r == 'full':
                        hh[j, t] = src.to(hh.dtype)
                    elif ':' in r:
                        kind, c = r.split(':'); V = st['V'][(CNAME.index(c), hi)][kind].to(hh.device); v = hh[j, t].float()
                        hh[j, t] = (v - V @ (V.T @ (v - src))).to(hh.dtype)
            return out
        return f
    for li in range(len(layers)):
        layers[li].register_forward_hook(hook(li))
    res = []; t1 = time.time()
    for n, q in enumerate(P):
        ids, tx, _ = enc(cities[q['x']]['city']); sids, ts, _ = enc(cities[q['src']]['city'])
        Hs = hs(sids); half = int(h[q['x']])
        st.update(on=True, t=tx, src={l: Hs[l][0, ts] for l in BAND},
                  V={(ci, l): Vt[(ci, l, 1 - half)] for ci in range(len(CELLS)) for l in BAND})
        out = m(input_ids=torch.tensor([ids] * len(ROWS), device=dev)).logits[:, -1, first].float().argmax(-1).tolist()
        st['on'] = False
        res.append(dict(x=q['x'], src=q['src'], y=int(y[q['x']]), ys=int(y[q['src']]), half=half, out=dict(zip(ROWS, out))))
        if n + 1 in (1, 30, 100) or (n + 1) % 250 == 0 or n + 1 == len(P):
            log('%d/%d pairs (%.2f s/pair)' % (n + 1, len(P), (time.time() - t1) / (n + 1)))
    json.dump(dict(smoke=SMOKE, cells=CNAME, real={'%d_%d_%d' % k: list(v) for k, v in real.items()}, calib_bad=bad, results=res),
              open(os.path.join(OUT, 'v242_faint.json'), 'w'))
    log('done (%.0fs)' % (time.time() - t0))
    analyse(OUT, say=log)


def verdict(labels, gt, bad_cells, shape='state'):
    if gt != 'USED':
        return 'GROUND-TRUTH-FAILED', None
    used = lambda c: labels[c] == 'USED' and c not in bad_cells
    if not used('native'):
        return 'UNINFORMATIVE', None
    limit = None
    for d in DOSES:
        if used('%s_%g' % (shape, d)):
            limit = d
        else:
            break
    if limit is not None and limit <= 0.005:
        return 'POWER-AT-OURO-LEVEL', limit
    if limit == 0.01:
        return 'AT-THE-LIMIT', limit
    return 'UNINFORMATIVE', limit


def analyse(root, say=print):
    R = json.load(open(glob.glob(os.path.join(root, '**', 'v242_faint.json'), recursive=True)[0]))
    rs = R['results']; cl = np.array([r['y'] for r in rs])
    hit = {k: np.array([r['out'][k] == r['ys'] for r in rs], float) for k in R['results'][0]['out']}
    d, lo, hi = boot(hit['full'] - hit['cR:native'], cl, seed=2425); gt = label(d, lo, hi)
    say('\nground truth: full %.3f, full - cR(native) %+.3f [%+.3f, %+.3f] -> %s' % (hit['full'].mean(), d, lo, hi, gt))
    bad_cells = sorted({b[0] for b in R['calib_bad']}); labels = {}; out = dict(smoke=R['smoke'], n=len(rs), ground_truth=[d, lo, hi, gt], cells={})
    for c in R['cells']:
        d, lo, hi = boot(hit['cUo:' + c] - hit['cR:' + c], cl, seed=2425); labels[c] = label(d, lo, hi)
        out['cells'][c] = dict(cUo=float(hit['cUo:' + c].mean()), cR=float(hit['cR:' + c].mean()), stat=[d, lo, hi], label=labels[c],
                               calibration_miss=c in bad_cells)
        say('  %-12s cUo %.3f  cR %.3f  cUo - cR %+.3f [%+.3f, %+.3f] -> %s%s' % (c, hit['cUo:' + c].mean(), hit['cR:' + c].mean(),
            d, lo, hi, labels[c], '  (calibration miss: counts as not USED)' if c in bad_cells else ''))
    for shape in SHAPES:
        v, lim = verdict(labels, gt, bad_cells, shape)
        out['verdict_' + shape] = [v, lim]
        say('shape %s: detection limit %s -> %s%s' % (shape, lim, v, '  (primary)' if shape == 'state' else ''))
    out['verdict'] = out['verdict_state'][0]
    json.dump(out, open(os.path.join(OUT, 'v242_result.json'), 'w'), indent=1)
    return out


def calib(path):
    z = np.load(path); y = z['y']; K = int(y.max()) + 1; h = halves(y); li = z['save_layers'].tolist()
    Sx = {l: z['e1'][:, li.index(l)].astype(np.float64) for l in BAND}
    B, real = fit_cells(Sx, z['name0'].astype(np.float64), y, h, K)
    for ci, c in enumerate(CNAME):
        print('%-12s %s' % (c, ' '.join('%d/%d %.4f/%.4f' % (l, hf, *real[(ci, l, hf)]) for l in BAND for hf in (0, 1))))
    print('calibration misses:', calib_ok(real) or 'none')
    print('dims (cUo) native layer 8 half 0:', B[(0, 8, 0)]['cUo'].shape, '| state 0.002:', B[(CNAME.index('state_0.002'), 8, 0)]['cUo'].shape)


def selftest():
    rng = np.random.default_rng(0); K = 5; y = np.repeat(np.arange(K), 40); X = rng.standard_normal((200, 60)) + 3 * np.eye(K, 60)[y]
    C = orth(rng.standard_normal((60, 4)))
    for shape in SHAPES:
        for t in (0.1, 0.01, 0.002):
            Xd, r, e = degrade(X, y, K, C, t, shape, np.random.default_rng(1))
            assert abs(r - t) < 1e-9 and Xd.shape == X.shape, (shape, t, r)
    lab = {c: 'USED' for c in CNAME}
    assert verdict(lab, 'USED', []) == ('POWER-AT-OURO-LEVEL', 0.002)
    assert verdict(dict(lab, **{'state_0.005': 'UNCLEAR'}), 'USED', []) == ('AT-THE-LIMIT', 0.01)
    assert verdict(dict(lab, **{'state_0.01': 'UNCLEAR'}), 'USED', []) == ('UNINFORMATIVE', 0.03)
    assert verdict(lab, 'USED', ['state_0.005']) == ('AT-THE-LIMIT', 0.01)
    assert verdict(dict(lab, native='UNCLEAR'), 'USED', []) == ('UNINFORMATIVE', None)
    assert verdict(lab, 'UNCLEAR', [])[0] == 'GROUND-TRUTH-FAILED'
    assert verdict(dict(lab, **{'state_0.002': 'NOT-USED'}), 'USED', []) == ('POWER-AT-OURO-LEVEL', 0.005)
    assert len(ROWS) == 2 + 2 * len(CNAME)
    print('selftest ok')


if __name__ == '__main__':
    a = sys.argv[1]
    selftest() if a == '--selftest' else run() if a == 'run' else calib(sys.argv[2]) if a == 'calib' else analyse(sys.argv[2])
