"""v235 clamp: a pretrained positive control for the clamp protocol (docs/prereg_v235_two_hop_control.md). One GPU kernel:
extraction, bases, clamp, analysis.

Items: the 1,038 known cities of pilot 2 (v235_pilot2_items.json), two-hop prompt "The capital of the country where the
city of {X} is located is" (three demonstrations, v235_pilot.prompt); halves within each country (seed 2352, as the
pilot 2 readout). Positions: e1 (the city's last token), last (the prompt's last token). BAND = hidden_states 8, 12, 16
(outputs of decoder layers 7, 11, 15).

1. States: for every city, hidden_states at BAND at e1 and last, and name0 (the mean layer-0 embedding of the city-name
   tokens: the copy's surface code).
2. Bases per (position, band layer, half h), fitted on the cities of half h: U = the country's centred class means
   (25 classes, centred on the mean of the class means, float64: at most 24 dims, orthonormalised); C = the same from name0 (the copy); Uo = U projected off C, orthonormalised
   (primary); R = random, Uo's dimension (seed 2354). Share of the country left = ||U - C C^T U||^2 / dim(U).
   A pair whose x is in half a uses the bases fitted on half 1 - a.
3. Pairs (seed 2353), one per city x for each type:
     other    x' = a known city of another country           (hit = cap(C'), kept = cap(C))
     same     x' = another known city of x's country         (design check: the answer should not change)
     person   x' = a known person born in another country, person two-hop prompt, x''s state at the person's last
              token (secondary, descriptive)
   Conditions at BAND, at the position (one batched pass per pair and position): none; cU, cUo, cC, cR:
   h <- h - V V^T (h - h'); full: h <- h' (the whole state at the same band layers: the clamp's ceiling).
   Outcome: argmax over the 25 capitals' first tokens at the last position.
4. Analysis (pair type other): per position, stat = hit(cUo) - hit(cR), cluster bootstrap over x's country, 10,000
   resamples, seed 2355, 97.5% interval. Labels (v215 / v234 rule): USED >= 0.10 with lower > 0; NOT-USED upper < 0.02;
   UNCLEAR otherwise. The whole-state patch gets the same label on hit(full) - hit(cR) (the ground truth at the band).
   Design check (pair type same): changed(cUo) - changed(cR), changed = answer != cap(C); DESIGN-FAILED if >= 0.10 with
   lower > 0.
   Verdict:
     GROUND-TRUTH-FAILED  e1's whole-state patch not USED, or last's whole-state patch not NOT-USED
     DESIGN-FAILED        design check fails at e1
     PASSES               e1 clamp USED and last clamp NOT-USED
     DETECTS-NOT-SPECIFIC e1 clamp USED, last clamp not NOT-USED
     MISSES               e1 clamp not USED

  GPU:  OUT_DIR=... python v235_clamp.py run      (V235_SMOKE=1: 30 cities per pair type, every learned
        subspace replaced by a random one: mechanics and the whole-state patch only, no verdict)
  CPU:  python v235_clamp.py analyse <dir with v235_clamp.json>
        python v235_clamp.py --selftest
"""
import glob, json, os, sys, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.'); HERE = os.path.dirname(os.path.abspath(__file__)); SMOKE = os.environ.get('V235_SMOKE') == '1'
sys.path.insert(0, HERE)
BAND = [8, 12, 16]; POS = ['e1', 'last']; CONDS = ['none', 'cU', 'cUo', 'cC', 'cR', 'full']; SUB = ['cU', 'cUo', 'cC', 'cR']
TYPES = ['other', 'same', 'person']


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v235_clamp_log.txt'), 'a').write(s + '\n')


def halves(y, seed=2352):
    rng = np.random.default_rng(seed); h = np.zeros(len(y), int)
    for c in np.unique(y):
        ix = np.where(y == c)[0]; rng.shuffle(ix); h[ix[len(ix) // 2:]] = 1
    return h


def orth(M, tol=1e-6, cap=None):
    u, s, _ = np.linalg.svd(np.asarray(M, np.float64), full_matrices=False); k = int((s > tol * s.max()).sum())
    return u[:, :k if cap is None else min(k, cap)]


def cmeans(X, y, K):
    M = np.stack([np.asarray(X[y == c], np.float64).mean(0) for c in range(K)]); return M - M.mean(0)   # rank <= K - 1


def bases(S, N0, y, h, K, rng):
    """S: dict (pos, layer) -> states [n, d]; N0: name0 [n, d]. Returns {(pos, layer, half): dict(U, Uo, C, R)}, shares."""
    out, share = {}, {}
    for half in (0, 1):
        m = h == half; C = orth(cmeans(N0[m], y[m], K).T, cap=K - 1)
        for (pos, l), X in S.items():
            U = orth(cmeans(X[m], y[m], K).T, cap=K - 1); P = U - C @ (C.T @ U); Uo = orth(P, cap=U.shape[1])
            R = orth(rng.standard_normal((X.shape[1], Uo.shape[1])))
            out[(pos, l, half)] = dict(cU=U, cUo=Uo, cC=C, cR=R)
            share[(pos, l, half)] = float(np.linalg.norm(P) ** 2 / U.shape[1])
    return out, share


def pairs(cities, persons, keys, rng):
    by = {k: [i for i, c in enumerate(cities) if c['ckey'] == k] for k in keys}; out = []
    for i, c in enumerate(cities):
        oth = [j for j, d in enumerate(cities) if d['ckey'] != c['ckey']]; same = [j for j in by[c['ckey']] if j != i]
        po = [j for j, p in enumerate(persons) if p['ckey'] != c['ckey']]
        out.append(dict(type='other', x=i, src=int(oth[rng.integers(len(oth))])))
        out.append(dict(type='same', x=i, src=int(same[rng.integers(len(same))])))
        out.append(dict(type='person', x=i, src=int(po[rng.integers(len(po))])))
    return out


def run():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from v235_pilot import prompt
    torch.set_grad_enabled(False); t0 = time.time()
    D = json.load(open(os.path.join(HERE, 'v235_cities.json'), encoding='utf-8'))
    S = json.load(open(os.path.join(HERE, 'v235_pilot2_items.json'), encoding='utf-8'))
    keys = S['keys']; K = len(keys); caps = [D['countries'][k]['capital'] for k in keys]
    cities, persons = S['all_cities'], S['persons']; y = np.array([keys.index(c['ckey']) for c in cities]); h = halves(y)
    repo = 'Qwen/Qwen2.5-7B'; tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    dev = m.get_input_embeddings().weight.device; layers = m.model.layers
    first = [tok.encode(' ' + c, add_special_tokens=False)[0] for c in caps]; assert len(set(first)) == K

    def enc(kind, x):
        p = prompt(kind, x); e = tok(p, return_offsets_mapping=True); off = e['offset_mapping']
        st, end = p.rfind(x), p.rfind(x) + len(x) - 1
        t = next(j for j, (u, v) in enumerate(off) if u <= end < v)
        return e['input_ids'], {'e1': t, 'last': len(e['input_ids']) - 1}, [j for j, (u, v) in enumerate(off) if v > st and u <= end]

    def hs(ids):
        return m(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states

    # 1) states
    Sx = {(p, l): [] for p in POS for l in BAND}; N0 = []; t1 = time.time()
    for c in cities:
        ids, t, name = enc('city_cap', c['city']); H = hs(ids)
        for p in POS:
            for l in BAND:
                Sx[(p, l)].append(H[l][0, t[p]].float().cpu().numpy())
        N0.append(H[0][0, name].float().mean(0).cpu().numpy())
    Sx = {k: np.stack(v) for k, v in Sx.items()}; N0 = np.stack(N0)
    log('v235 clamp | smoke %s | %d cities, %d people | states in %.0fs (load %.0fs)' % (SMOKE, len(cities), len(persons),
                                                                                       time.time() - t1, t1 - t0))
    # 2) bases
    B, share = bases(Sx, N0, y, h, K, np.random.default_rng(2354))
    for k in sorted(share):
        log('  %-4s layer %2d half %d: U %d dims, Uo %d, C %d; share of the country left after the copy projection %.3f' % (
            k[0], k[1], k[2], B[k]['cU'].shape[1], B[k]['cUo'].shape[1], B[k]['cC'].shape[1], share[k]))
    # 3) pairs
    P = pairs(cities, persons, keys, np.random.default_rng(2353))
    if SMOKE:
        P = [q for t in TYPES for q in [r for r in P if r['type'] == t][:30]]
    st = {'mode': None}

    def hook(li):
        hi = li + 1

        def f(mod, args, out):
            hh = out[0] if isinstance(out, tuple) else out
            if st['mode'] == 'clamp' and hi in BAND:
                t = st['t']; src = st['src'][hi].to(hh.device).float()
                for j, c in enumerate(CONDS):
                    if c == 'full':
                        hh[j, t] = src.to(hh.dtype)
                    elif c in SUB:
                        V = st['V'][(c, hi)].to(hh.device); v = hh[j, t].float()
                        hh[j, t] = (v - V @ (V.T @ (v - src))).to(hh.dtype)
            return out
        return f
    for li in range(len(layers)):
        layers[li].register_forward_hook(hook(li))
    if SMOKE:   # mechanics only: every learned subspace replaced by a random one of the same dimension
        rr = np.random.default_rng(0); B = {k: {c: orth(rr.standard_normal(v[c].shape)) for c in SUB} for k, v in B.items()}
    Vt = {k: {c: torch.tensor(v[c], dtype=torch.float32) for c in SUB} for k, v in B.items()}
    res = []; t1 = time.time()
    for n, q in enumerate(P):
        x = cities[q['x']]; ids, tx, _ = enc('city_cap', x['city'])
        if q['type'] == 'person':
            sp = persons[q['src']]; sids, ts, _ = enc('person_cap', sp['person']); ys = keys.index(sp['ckey'])
        else:
            sc = cities[q['src']]; sids, ts, _ = enc('city_cap', sc['city']); ys = keys.index(sc['ckey'])
        Hs = hs(sids); r = dict(q, y=int(y[q['x']]), ys=ys, half=int(h[q['x']]))
        for p in POS:
            st.update(mode='clamp', t=tx[p], src={l: Hs[l][0, ts[p]] for l in BAND},
                      V={(c, l): Vt[(p, l, 1 - r['half'])][c] for c in SUB for l in BAND})
            out = m(input_ids=torch.tensor([ids] * len(CONDS), device=dev)).logits[:, -1, first].float().argmax(-1).tolist()
            st['mode'] = None; r[p] = dict(zip(CONDS, out))
        res.append(r)
        if n + 1 in (1, 30, 100) or (n + 1) % 500 == 0 or n + 1 == len(P):
            o = [q for q in res if q['type'] == 'other']
            log('%d/%d pairs (%.2f s/pair) | type other, e1 hit so far: %s' % (n + 1, len(P), (time.time() - t1) / (n + 1),
                ' '.join('%s %.3f' % (c, np.mean([q['e1'][c] == q['ys'] for q in o])) for c in CONDS) if o else '-'))
    json.dump(dict(smoke=SMOKE, keys=keys, share={'%s_%d_%d' % k: v for k, v in share.items()}, results=res),
              open(os.path.join(OUT, 'v235_clamp.json'), 'w'))
    log('done (%.0fs)' % (time.time() - t0))


def label(d, lo, hi):
    return 'USED' if d >= 0.10 and lo > 0 else 'NOT-USED' if hi < 0.02 else 'UNCLEAR'


def verdict(L):
    """L: dict with e1_full, last_full, e1_clamp, last_clamp, design_e1 (labels)."""
    if L['e1_full'] != 'USED' or L['last_full'] != 'NOT-USED':
        return 'GROUND-TRUTH-FAILED'
    if L['design_e1'] == 'DESIGN-FAILED':
        return 'DESIGN-FAILED'
    if L['e1_clamp'] != 'USED':
        return 'MISSES'
    return 'PASSES' if L['last_clamp'] == 'NOT-USED' else 'DETECTS-NOT-SPECIFIC'


def boot(d, cl, seed=2355, n=10000):
    u = np.unique(cl); ix = np.searchsorted(u, cl); S = np.zeros(len(u)); N = np.zeros(len(u))
    np.add.at(S, ix, d); np.add.at(N, ix, 1); rng = np.random.default_rng(seed); bs = np.empty(n)
    for i in range(n):
        q = rng.integers(0, len(u), len(u)); bs[i] = S[q].sum() / N[q].sum()
    lo, hi = np.percentile(bs, (1.25, 98.75)); return float(d.mean()), float(lo), float(hi)


def analyse(root, say=print):
    R = json.load(open(glob.glob(os.path.join(root, '**', 'v235_clamp.json'), recursive=True)[0]))
    out = {'smoke': R['smoke'], 'share': R['share']}; L = {}
    for t in TYPES:
        rs = [r for r in R['results'] if r['type'] == t]; cl = np.array([r['y'] for r in rs]); out[t] = {}
        for p in POS:
            hit = {c: np.array([r[p][c] == r['ys'] for r in rs], float) for c in CONDS}
            chg = {c: np.array([r[p][c] != r['y'] for r in rs], float) for c in CONDS}
            e = dict(n=len(rs), hit={c: float(hit[c].mean()) for c in CONDS}, changed={c: float(chg[c].mean()) for c in CONDS})
            if t == 'same':
                d, lo, hi = boot(chg['cUo'] - chg['cR'], cl); e['design'] = [d, lo, hi]
                e['design_label'] = 'DESIGN-FAILED' if d >= 0.10 and lo > 0 else 'ok'
                L['design_' + p] = e['design_label']
            else:
                for c in ('cUo', 'cU', 'cC', 'full'):
                    d, lo, hi = boot(hit[c] - hit['cR'], cl); e[c + '-cR'] = [d, lo, hi, label(d, lo, hi)]
                if t == 'other':
                    L[p + '_clamp'] = e['cUo-cR'][3]; L[p + '_full'] = e['full-cR'][3]
            out[t][p] = e
            say('%-6s %-4s n %d | hit %s | changed %s%s' % (t, p, len(rs), ' '.join('%s %.3f' % (c, hit[c].mean()) for c in CONDS),
                ' '.join('%s %.3f' % (c, chg[c].mean()) for c in CONDS),
                (' | design %+.3f [%+.3f, %+.3f] %s' % (*e['design'], e['design_label'])) if t == 'same' else
                ''.join(' | %s-cR %+.3f [%+.3f, %+.3f] %s' % (c, *e[c + '-cR']) for c in ('cUo', 'full'))))
    out['labels'] = L; out['verdict'] = None if R['smoke'] else verdict(L)
    say('labels %s -> verdict %s' % (L, out['verdict']))
    json.dump(out, open(os.path.join(OUT, 'v235_result.json'), 'w'), indent=1)
    return out


def selftest():
    assert label(0.12, 0.05, 0.2) == 'USED' and label(0.0, -0.01, 0.015) == 'NOT-USED' and label(0.09, 0.05, 0.13) == 'UNCLEAR'
    base = dict(e1_full='USED', last_full='NOT-USED', e1_clamp='USED', last_clamp='NOT-USED', design_e1='ok')
    assert verdict(base) == 'PASSES'
    assert verdict(dict(base, last_clamp='UNCLEAR')) == 'DETECTS-NOT-SPECIFIC'
    assert verdict(dict(base, e1_clamp='UNCLEAR')) == 'MISSES' and verdict(dict(base, e1_clamp='NOT-USED')) == 'MISSES'
    assert verdict(dict(base, last_full='UNCLEAR')) == 'GROUND-TRUTH-FAILED'
    assert verdict(dict(base, e1_full='UNCLEAR')) == 'GROUND-TRUTH-FAILED'
    assert verdict(dict(base, design_e1='DESIGN-FAILED')) == 'DESIGN-FAILED'
    rng = np.random.default_rng(0); y = rng.integers(0, 5, 200); X = rng.standard_normal((200, 30)) + np.eye(5, 30)[y] * 3
    N0 = rng.standard_normal((200, 30)); hv = halves(y); Bs, sh = bases({('e1', 8): X}, N0, y, hv, 5, rng)
    b = Bs[('e1', 8, 0)]
    for c in SUB:
        assert np.allclose(b[c].T @ b[c], np.eye(b[c].shape[1]), atol=1e-6)
    assert np.abs(b['cC'].T @ b['cUo']).max() < 1e-6 and 0 < sh[('e1', 8, 0)] <= 1
    print('selftest ok')


if __name__ == '__main__':
    a = sys.argv[1]
    selftest() if a == '--selftest' else run() if a == 'run' else analyse(sys.argv[2])
