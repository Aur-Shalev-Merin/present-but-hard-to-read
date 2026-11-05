"""v222: does v215's clamp detect a value that is stored at one token and read at a later one?
(docs/prereg_v222_clamp_distance.md)

Kudo et al.'s Simple chain (v221's items, target " b = B , a = A", Qwen2.5-7B fp16 over two T4s). The model writes b's
digit, then computes a = k o2 b a few tokens later, so B is stored at the digit token (digB) and read later to get A.
The clamp, unchanged from v215 (amendment 1), is applied at digB; the outcome is the a digit predicted at inA.

  extract  one teacher-forced pass per item: states at digB x hidden_states 0, 4, ..., 28; correct = the b digit is the
           argmax at inb AND the a digit is the argmax at inA                       -> v222_states.npy, v222_meta.json
  plan     v215_analyze.patch_plan at digB (B subspace, d2-copy subspace, B orthogonalised against it, random; fitted on
           the other half of the families; 1,000 pairs x correct, x' same family, version and d1, different d2)
                                                                                   -> v215_plan.npz, v215_pairs.json
  patch    for each pair, x' captured at digB, then x run under: none, cB, cBo, cD, cR (band layers 12, 16, 20, 24), full
           (layer outputs 12..28, v215's positive control) and fullall (every layer output 1..28; descriptive: is A read
           from digB at all?); outcome = argmax over the ten digit tokens at inA; hit = x''s A  -> v222_patch.json
  analyse  hit(cBo) - hit(cR), cluster bootstrap over x's family, 10,000 resamples, seed 2223, 97.5% CI:
           UNSCORABLE hit(full) < 0.30 | USED diff >= 0.10, CI lower > 0 | NOT-USED CI upper < 0.02 | UNCLEAR

  GPU:  OUT_DIR=... python v222_clamp.py all        (extract, plan, patch, analyse)
        OUT_DIR=... V222_SMOKE=1 python v222_clamp.py all   (40 items; random bases, 20 pairs from family 0; mechanics)
  CPU:  python v222_clamp.py --selftest
"""
import json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np

import v215_items as V5
import v222_items as I

OUT = os.environ.get('OUT_DIR', '.'); SMOKE = os.environ.get('V222_SMOKE') == '1'
LAYERS = list(range(0, 29, 4)); BAND = [12, 16, 20, 24]; FULL = list(range(12, 29)); FULLALL = list(range(1, 29))
CONDS = ['none', 'cB', 'cBo', 'cD', 'cR', 'full', 'fullall']; POS = 'digB'


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v222_log.txt'), 'a').write(s + '\n')


def items():
    its = I.build(); s = V5.sha16(its)
    log('v222 items sha %s (frozen %s)' % (s, I.ITEMS_SHA))
    if I.ITEMS_SHA and s != I.ITEMS_SHA:
        sys.exit('items hash differs from the frozen one')
    return its, s


def load():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False); t0 = time.time()
    repo = 'Qwen/Qwen2.5-7B'; tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    log('loaded %s in %.0fs' % (repo, time.time() - t0))
    return torch, tok, m, m.get_input_embeddings().weight.device, [tok.encode(str(k), add_special_tokens=False)[0] for k in range(10)]


def toks(tok, it, digit):
    enc = tok(it['prompt'] + it['target'], return_offsets_mapping=True); ids, off = enc['input_ids'], enc['offset_mapping']
    cover = lambda c: next(t for t, (u, v) in enumerate(off) if u <= c < v)
    tB, tA = cover(it['cB']), cover(it['cA'])
    assert ids[tB] == digit[it['B']] and ids[tA] == digit[it['A']] and cover(it['cpos']['inA']) == tA - 1, (it['target'], tB, tA)
    return ids, tB, tA


def extract(its, s, torch, tok, m, dev, digit):
    from numpy.lib.format import open_memmap
    X = open_memmap(os.path.join(OUT, 'v222_states.npy'), mode='w+', dtype=np.float16, shape=(len(its), 1, len(LAYERS), m.config.hidden_size))
    meta = []; t1 = time.time()
    for n, it in enumerate(its):
        ids, tB, tA = toks(tok, it, digit)
        out = m(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True); hs, lg = out.hidden_states, out.logits[0]
        for b, l in enumerate(LAYERS):
            X[n, 0, b] = hs[l][0, tB].float().cpu().numpy()
        pb, pa = (int(lg[t - 1, digit].float().argmax()) for t in (tB, tA))
        meta.append(dict(n=n, okB=pb == it['B'], okA=pa == it['A'], correct=bool(pb == it['B'] and pa == it['A'])))
        if n == 0:
            log('digB token %d %r, inA token %d %r, a digit %r' % (tB, tok.decode([ids[tB]]), tA - 1, tok.decode([ids[tA - 1]]), tok.decode([ids[tA]])))
        if n + 1 in (1, 100) or (n + 1) % 2000 == 0 or n + 1 == len(its):
            log('item %5d/%d (%.3f s/item) | b correct %.3f, a correct %.3f, both %.3f' % (
                n + 1, len(its), (time.time() - t1) / (n + 1), *(np.mean([x[k] for x in meta]) for k in ('okB', 'okA', 'correct'))))
    X.flush()
    json.dump(dict(items_sha=s, layers=LAYERS, positions=[POS], items=meta), open(os.path.join(OUT, 'v222_meta.json'), 'w'))
    for v in ('plus', 'minus'):
        log('%s: b correct %.3f, a correct %.3f, both %.3f' % (v, *(np.mean([meta[k][q] for k, i in enumerate(its) if i['ver'] == v])
                                                                   for q in ('okB', 'okA', 'correct'))))
    return X, meta


def plan(X, its, meta, outdir=OUT, say=log):
    import v215_analyze as A
    A.POS = [POS]; A.patch_plan(X, its, meta, [POS], outdir, say)


def patch(its, torch, tok, m, dev, digit, root=OUT):
    if SMOKE:                                             # mechanics only: random bases, pairs from the first family
        rr = np.random.default_rng(0); P = {'R_%s_%d' % (POS, h): np.linalg.qr(rr.standard_normal((m.config.hidden_size, 9)))[0].astype(np.float32) for h in BAND}
        f0 = [k for k, i in enumerate(its) if i['fam'] == 0]
        alt = lambda k: [j for j in f0 if its[j]['ver'] == its[k]['ver'] and its[j]['d1'] == its[k]['d1'] and its[j]['d2'] != its[k]['d2']]
        pairs = [dict(pos=POS, x=k, x2=int(rr.choice(alt(k))), half=0) for k in [k for k in f0 if alt(k)][:20]]
    else:
        P = np.load(os.path.join(root, 'v215_plan.npz')); pairs = json.load(open(os.path.join(root, 'v215_pairs.json')))['pairs']
    layers = m.model.layers; st = {'t': None, 'mode': None, 'cap': {}, 'src': {}, 'U': {}}

    def hook(li):
        hi = li + 1                                       # hidden_states index of this layer's output

        def f(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out; t = st['t']
            if st['mode'] == 'cap':
                st['cap'][hi] = h[0, t].detach().clone()
            elif st['mode'] in ('cB', 'cBo', 'cD', 'cR') and hi in BAND:
                U = st['U'][hi].to(h.device, h.dtype); d = h[0, t] - st['src'][hi].to(h.device)
                h[0, t] = h[0, t] - U @ (U.T @ d)
            elif (st['mode'] == 'full' and hi in FULL) or (st['mode'] == 'fullall' and hi in FULLALL):
                h[0, t] = st['src'][hi].to(h.device)
            return out
        return f
    hs = [layers[li].register_forward_hook(hook(li)) for li in range(len(layers))]
    res = []; t1 = time.time()
    for n, p in enumerate(pairs):
        x, x2 = its[p['x']], its[p['x2']]
        ids, tB, tA = toks(tok, x, digit); ids2, tB2, tA2 = toks(tok, x2, digit); assert (len(ids2), tB2, tA2) == (len(ids), tB, tA)
        st['t'] = tB; st['mode'] = 'cap'; st['cap'] = {}; m(input_ids=torch.tensor([ids2], device=dev)); st['src'] = dict(st['cap'])
        r = dict(p, A=x['A'], A2=x2['A'], fam=x['fam'], k='%d%s' % (x['k'], x['o2']))
        if SMOKE:                                         # hook check: the full patch at inA itself must give x''s A
            st['t'] = tA - 1; st['mode'] = 'cap'; st['cap'] = {}; m(input_ids=torch.tensor([ids2], device=dev))
            src_p = st['src']; st['src'] = dict(st['cap']); st['mode'] = 'fullall'
            r['mech'] = int(m(input_ids=torch.tensor([ids], device=dev)).logits[0, tA - 1, digit].float().argmax())
            st['t'] = tB; st['src'] = src_p
        for c in CONDS:
            st['mode'] = c
            if c in ('cB', 'cBo', 'cD', 'cR'):
                key = {'cB': 'U_%s_%d_%d', 'cBo': 'Uo_%s_%d_%d', 'cD': 'D_%s_%d_%d'}
                st['U'] = {hi: torch.tensor(P[('R_%s_%d' % (POS, hi)) if (c == 'cR' or SMOKE) else (key[c] % (POS, p['half'], hi))]) for hi in BAND}
            r[c] = int(m(input_ids=torch.tensor([ids], device=dev)).logits[0, tA - 1, digit].float().argmax())
        st['mode'] = None; res.append(r)
        if n + 1 in (1, 20, 100) or (n + 1) % 250 == 0 or n + 1 == len(pairs):
            log('%d/%d pairs (%.2f s/pair) | hit so far: %s' % (n + 1, len(pairs), (time.time() - t1) / (n + 1), ' '.join(
                '%s %.3f' % (c, np.mean([q[c] == q['A2'] for q in res])) for c in CONDS + (['mech'] if SMOKE else []))))
    for h in hs:
        h.remove()
    json.dump(dict(smoke=SMOKE, results=res), open(os.path.join(OUT, 'v222_patch.json'), 'w'))


def analyse(root=OUT, say=log):
    R = json.load(open(os.path.join(root, 'v222_patch.json'))); rs = R['results']; fam = np.array([r['fam'] for r in rs])
    hit = {c: np.array([r[c] == r['A2'] for r in rs], float) for c in CONDS}
    kept = {c: np.mean([r[c] == r['A'] for r in rs]) for c in CONDS}
    d = hit['cBo'] - hit['cR']; u = np.unique(fam); ix = {f: j for j, f in enumerate(u)}
    S = np.zeros(len(u)); N = np.zeros(len(u)); np.add.at(S, [ix[f] for f in fam], d); np.add.at(N, [ix[f] for f in fam], 1)
    rng = np.random.default_rng(2223); bs = []
    for _ in range(10000):
        q = rng.integers(0, len(u), len(u)); bs.append(S[q].sum() / N[q].sum())
    lo, hi = np.percentile(bs, (1.25, 98.75))
    lab = ('UNSCORABLE' if hit['full'].mean() < 0.30 else 'USED' if d.mean() >= 0.10 and lo > 0 else
           'NOT-USED' if hi < 0.02 else 'UNCLEAR')
    say('%s -> a (%d pairs%s): hit %s | kept own A %s | cBo - cR %+.4f [%+.4f, %+.4f] -> %s (cB - cR %+.4f, descriptive)' % (
        POS, len(rs), ', SMOKE' if R['smoke'] else '', ' '.join('%s %.3f' % (c, hit[c].mean()) for c in CONDS),
        ' '.join('%s %.3f' % (c, kept[c]) for c in CONDS), d.mean(), lo, hi, lab, (hit['cB'] - hit['cR']).mean()))
    for k in sorted(set(r.get('k', '') for r in rs)):          # descriptive: k = 0 with '+' makes a a copy of b
        w = np.array([r.get('k', '') == k for r in rs])
        say('  a = %s b (%d pairs): hit %s' % (k, w.sum(), ' '.join('%s %.3f' % (c, hit[c][w].mean()) for c in CONDS)))
    json.dump(dict(label=lab, diff=[float(d.mean()), float(lo), float(hi)], hit={c: float(v.mean()) for c, v in hit.items()},
                   kept={c: float(v) for c, v in kept.items()}, n=len(rs)), open(os.path.join(OUT, 'v222_result.json'), 'w'), indent=1)
    return lab


def main():
    its, s = items(); torch, tok, m, dev, digit = load()
    if SMOKE:
        its_x = its[:40]; extract(its_x, s, torch, tok, m, dev, digit)
    else:
        X, meta = extract(its, s, torch, tok, m, dev, digit); plan(X, its, meta)
    patch(its, torch, tok, m, dev, digit); analyse()


def selftest():
    """Synthetic: the plan runs at digB on the real design (B coded in the states); analyse labels made-up results
    correctly (cBo moves A -> USED; nothing moves -> NOT-USED; full fails -> UNSCORABLE)."""
    import tempfile
    global OUT
    its = I.build(); keep = set(sorted(set(i['fam'] for i in its))[::4]); its = [i for i in its if i['fam'] in keep]
    meta = [dict(correct=True) for _ in its]; rng = np.random.default_rng(0); dd = 64
    U1, U2, CB = (rng.standard_normal((10, dd)) for _ in range(3)); off = {f: rng.standard_normal(dd) * 3 for f in keep}
    X = np.stack([off[i['fam']] + U1[i['d1']] + U2[i['d2']] + 0.8 * CB[i['B']] + rng.standard_normal((1, len(LAYERS), dd)) for i in its]).astype(np.float32)
    OUT = tempfile.mkdtemp(); plan(X, its, meta, OUT, lambda s: None)
    pj = json.load(open(os.path.join(OUT, 'v215_pairs.json'))); P = np.load(os.path.join(OUT, 'v215_plan.npz'))
    ok = pj['positions'] == [POS] and len(pj['pairs']) == 1000 and all(
        its[p['x']]['d1'] == its[p['x2']]['d1'] and its[p['x']]['B'] != its[p['x2']]['B'] and its[p['x']]['A'] != its[p['x2']]['A'] for p in pj['pairs'])
    print('selftest plan: %d pairs, %d bases (%s), pairs differ in B and A: %s' % (len(pj['pairs']), len(P.files), sorted(P.files)[:3], ok), flush=True)
    for world, want in (('used', 'USED'), ('not used', 'NOT-USED'), ('no control', 'UNSCORABLE')):
        res = []
        for k, p in enumerate(pj['pairs']):
            x, x2 = its[p['x']], its[p['x2']]; r = dict(p, A=x['A'], A2=x2['A'], fam=x['fam'])
            for c in CONDS:
                moved = (c in ('full', 'fullall') and world != 'no control') or (c in ('cB', 'cBo') and world == 'used' and k % 2 == 0)
                r[c] = x2['A'] if moved else x['A']
            res.append(r)
        json.dump(dict(smoke=False, results=res), open(os.path.join(OUT, 'v222_patch.json'), 'w'))
        lab = analyse(OUT, say=lambda s: None); good = lab == want; ok &= good
        print('selftest %-10s -> %-10s %s' % (world, lab, 'OK' if good else 'FAIL'), flush=True)
    return ok


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        sys.exit(0 if selftest() else 1)
    main()
