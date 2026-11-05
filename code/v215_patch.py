"""v215 Part B: is the value used by the chain? (docs/prereg_v215_kudo_retest.md section 4)

For each planned pair (x, x') at a position p (same family and version, different B, x correct):
  capture x''s decoder-layer outputs at p (layers 11..27, i.e. hidden_states 12..28 before the final norm), then run x
  (prompt + its own target, teacher forcing) under each condition, hooks on the decoder-layer outputs at token p:
    none   no change
    cB     at the band layers (hidden_states 12, 16, 20, 24 = outputs of layers 11, 15, 19, 23): replace x's component
           in the B subspace (fitted on the other half of the families) by x''s:  h <- h - U U^T (h - h')
    cR     the same with the random 9-dim subspace
    full   positive control: replace x's whole state at p by x''s at every layer output 11..27
  outcome: argmax over the ten digit tokens at the chain's step "b =" (the token before the computed digit);
  hit = x''s B, kept = x's B.
  Amendment 1 (2026-10-23, before any analysis): cBo = cB with the B subspace orthogonalised against the d2-copy
  subspace (class means of d2, fitted the same way), cD = the d2-copy subspace itself (reference); the primary statistic
  is hit(cBo) - hit(cR) (cB - cR is descriptive: at the d2 token the B subspace overlaps the copied digit).
  Statistic per position: hit(cBo) - hit(cR), cluster bootstrap over x's family, 10,000 resamples, seed 2153, 97.5% CI.
    UNSCORABLE hit(full) < 0.30 | USED diff >= 0.10, CI lower > 0 | NOT-USED CI upper < 0.02 | UNCLEAR

  GPU:  OUT_DIR=... python v215_patch.py extract <dir with v215_plan.npz, v215_pairs.json | SMOKE>   (V215_SMOKE=1: 20
        pairs per position, the random subspace in place of the B subspace too: mechanics only; SMOKE needs no plan)
  CPU:  python v215_patch.py analyse <dir with v215_patch.json>
"""
import glob, json, os, sys, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.'); SMOKE = os.environ.get('V215_SMOKE') == '1'
BAND = [12, 16, 20, 24]; FULL = list(range(12, 29)); CONDS = ['none', 'cB', 'cBo', 'cD', 'cR', 'full']


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v215_patch_log.txt'), 'a').write(s + '\n')


def extract(root):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import v215_items as I
    torch.set_grad_enabled(False); t0 = time.time()
    _, _, _ = I.build(); s = I.all_sha(); fam = I.build_short()    # Part B runs in the short-chain arm
    if I.ITEMS_SHA and s != I.ITEMS_SHA:
        sys.exit('items hash differs')
    if root == 'SMOKE':                                   # mechanics only: random bases, pairs from the first family
        rr = np.random.default_rng(0); plan = {'R_%s_%d' % (p, h): np.linalg.qr(rr.standard_normal((3584, 9)))[0].astype(np.float32)
                                              for p in ('eq', 'last') for h in BAND}
        f0 = [k for k, i in enumerate(fam) if i['fam'] == 0]
        alt = lambda k: [j for j in f0 if fam[j]['ver'] == fam[k]['ver'] and fam[j]['d1'] == fam[k]['d1'] and fam[j]['d2'] != fam[k]['d2']]
        pairs = [dict(pos=p, x=k, x2=int(rr.choice(alt(k))), half=0) for p in ('eq', 'last') for k in [k for k in f0 if alt(k)][:20]]
    else:
        plan = np.load(glob.glob(os.path.join(root, '**', 'v215_plan.npz'), recursive=True)[0])
        pj = json.load(open(glob.glob(os.path.join(root, '**', 'v215_pairs.json'), recursive=True)[0]))
        pairs = pj['pairs']
        if SMOKE:
            pairs = [p for pos in pj['positions'] for p in [q for q in pairs if q['pos'] == pos][:20]]
    repo = 'Qwen/Qwen2.5-7B'; tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    dev = m.get_input_embeddings().weight.device; layers = m.model.layers; digit = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(10)]
    log('v215 patch | %d pairs | smoke %s | items sha %s | loaded in %.0fs' % (len(pairs), SMOKE, s, time.time() - t0))
    st = {'t': None, 'mode': None, 'cap': {}, 'src': {}, 'U': {}}

    def hook(li):
        hi = li + 1                                       # hidden_states index of this layer's output

        def f(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out; t = st['t']
            if st['mode'] == 'cap':
                st['cap'][hi] = h[0, t].detach().clone()
            elif st['mode'] in ('cB', 'cBo', 'cD', 'cR') and hi in BAND:
                U = st['U'][hi].to(h.device, h.dtype); d = h[0, t] - st['src'][hi].to(h.device)
                h[0, t] = h[0, t] - U @ (U.T @ d)
            elif st['mode'] == 'full' and hi in FULL:
                h[0, t] = st['src'][hi].to(h.device)
            return out
        return f
    for li in range(len(layers)):
        layers[li].register_forward_hook(hook(li))
    res = []; t1 = time.time()
    for n, p in enumerate(pairs):
        x, x2 = fam[p['x']], fam[p['x2']]
        ids = tok(x['prompt'] + x['target'], return_offsets_mapping=True); off = ids['offset_mapping']; ids = ids['input_ids']
        ids2 = tok(x2['prompt'] + x2['target']).input_ids; assert len(ids2) == len(ids)
        cover = lambda c: next(t for t, (u, v) in enumerate(off) if u <= c < v)
        t = cover(x['cpos'][p['pos']]); tB = cover(x['cB']); st['t'] = t
        st['mode'] = 'cap'; st['cap'] = {}; m(input_ids=torch.tensor([ids2], device=dev)); st['src'] = dict(st['cap'])
        r = dict(p, B=x['B'], B2=x2['B'], fam=x['fam'])
        if SMOKE:                                         # hook check: the full patch at the chain's own "b =" token
            st['t'] = tB - 1; st['mode'] = 'cap'; st['cap'] = {}; m(input_ids=torch.tensor([ids2], device=dev))
            src_p = st['src']; st['src'] = dict(st['cap']); st['mode'] = 'full'
            r['mech'] = int(m(input_ids=torch.tensor([ids], device=dev)).logits[0, tB - 1, digit].float().argmax())
            st['t'] = t; st['src'] = src_p
        for c in CONDS:
            st['mode'] = c
            if c in ('cB', 'cBo', 'cD', 'cR'):
                key = {'cB': 'U_%s_%d_%d', 'cBo': 'Uo_%s_%d_%d', 'cD': 'D_%s_%d_%d'}
                st['U'] = {hi: torch.tensor(plan[('R_%s_%d' % (p['pos'], hi)) if (c == 'cR' or SMOKE) else
                                                 (key[c] % (p['pos'], p['half'], hi))]) for hi in BAND}
            lg = m(input_ids=torch.tensor([ids], device=dev)).logits[0, tB - 1, digit].float()
            r[c] = int(lg.argmax())
        st['mode'] = None; res.append(r)
        if n + 1 in (1, 20, 100) or (n + 1) % 250 == 0 or n + 1 == len(pairs):
            log('%d/%d pairs (%.2f s/pair) | hit so far: %s' % (n + 1, len(pairs), (time.time() - t1) / (n + 1), ' '.join(
                '%s %.3f' % (c, np.mean([q[c] == q['B2'] for q in res])) for c in CONDS + (['mech'] if SMOKE else []))))
    json.dump(dict(smoke=SMOKE, items_sha=s, results=res), open(os.path.join(OUT, 'v215_patch.json'), 'w'))
    log('done (%.0fs)' % (time.time() - t0))


def analyse(root, say=print):
    R = json.load(open(glob.glob(os.path.join(root, '**', 'v215_patch.json'), recursive=True)[0]))
    out = {}
    for pos in sorted(set(r['pos'] for r in R['results'])):
        rs = [r for r in R['results'] if r['pos'] == pos]; fam = np.array([r['fam'] for r in rs])
        hit = {c: np.array([r[c] == r['B2'] for r in rs], float) for c in CONDS}
        kept = {c: np.mean([r[c] == r['B'] for r in rs]) for c in CONDS}
        d = hit['cBo'] - hit['cR']; u = np.unique(fam); ix = {f: j for j, f in enumerate(u)}   # amendment 1: cBo primary
        S = np.zeros(len(u)); N = np.zeros(len(u)); np.add.at(S, [ix[f] for f in fam], d); np.add.at(N, [ix[f] for f in fam], 1)
        rng = np.random.default_rng(2153); bs = []
        for _ in range(10000):
            q = rng.integers(0, len(u), len(u)); bs.append(S[q].sum() / N[q].sum())
        lo, hi = np.percentile(bs, (1.25, 98.75))
        lab = ('UNSCORABLE' if hit['full'].mean() < 0.30 else 'USED' if d.mean() >= 0.10 and lo > 0 else
               'NOT-USED' if hi < 0.02 else 'UNCLEAR')
        say('%s (%d pairs%s): hit %s | kept own B %s | cBo - cR %+.4f [%+.4f, %+.4f] -> %s (cB - cR %+.4f, descriptive)' % (
            pos, len(rs), ', SMOKE' if R['smoke'] else '', ' '.join('%s %.3f' % (c, hit[c].mean()) for c in CONDS),
            ' '.join('%s %.3f' % (c, kept[c]) for c in CONDS), d.mean(), lo, hi, lab, (hit['cB'] - hit['cR']).mean()))
        out[pos] = lab
    return out


if __name__ == '__main__':
    {'extract': extract, 'analyse': analyse}[sys.argv[1]](sys.argv[2])
