"""v216 Part B: is b used by the answer, without a chain? (docs/prereg_v216_answer_only.md section 4; v215_patch adapted)

For each planned pair (x, x') at a position p (same family, version and d1, different d2, x correct): capture x''s
decoder-layer outputs at p, then run x (prompt + " a = A", teacher forcing) under each condition, hooks on the
decoder-layer outputs at token p:
  none   no change
  cB / cBo / cA / cD / cR   at the band layers (hidden_states 12, 16, 20, 24 = outputs of layers 11, 15, 19, 23):
         h <- h - U U^T (h - h'), with U = the B subspace / B orthogonalised against the d2-copy and answer subspaces
         (primary) / the answer subspace / the d2-copy subspace / random 9 dims (bases fitted on the other half)
  full   positive control: x''s whole state at p at every layer output 11-27
  outcome: argmax over the ten digit tokens at pre (the token before the answer digit); hit = x''s answer, kept = x's.
  Per position: hit(cBo) - hit(cR), cluster bootstrap over x's family, 10,000 resamples, seed 2163, 97.5% CI.
    UNSCORABLE hit(full) < 0.30 | USED >= 0.10 with CI lower > 0 | NOT-USED CI upper < 0.02 | UNCLEAR

  GPU:  OUT_DIR=... python v216_patch.py extract <dir with v216_plan.npz, v216_pairs.json | SMOKE>   (V216_SMOKE=1:
        20 pairs per position, the random subspace in place of every learned one: mechanics only)
  CPU:  python v216_patch.py analyse <dir with v216_patch.json>
"""
import glob, json, os, sys, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.'); SMOKE = os.environ.get('V216_SMOKE') == '1'
BAND = [12, 16, 20, 24]; FULL = list(range(12, 29)); CONDS = ['none', 'cB', 'cBo', 'cA', 'cD', 'cR', 'full']
KEY = {'cB': 'U', 'cBo': 'Uo', 'cA': 'An', 'cD': 'D'}


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v216_patch_log.txt'), 'a').write(s + '\n')


def extract(root):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import v215_items as V5, v216_items as I
    torch.set_grad_enabled(False); t0 = time.time()
    fam = I.build(); s = V5.sha16(fam)
    if I.ITEMS_SHA and s != I.ITEMS_SHA:
        sys.exit('items hash differs')
    if root == 'SMOKE':
        rr = np.random.default_rng(0); plan = {'R_%s_%d' % (p, h): np.linalg.qr(rr.standard_normal((3584, 9)))[0].astype(np.float32)
                                              for p in ('eq', 'pre', 'last') for h in BAND}
        f0 = [k for k, i in enumerate(fam) if i['fam'] == 0]
        alt = lambda k: [j for j in f0 if fam[j]['ver'] == fam[k]['ver'] and fam[j]['d1'] == fam[k]['d1'] and fam[j]['d2'] != fam[k]['d2']]
        pairs = [dict(pos=p, x=k, x2=int(rr.choice(alt(k))), half=0) for p in ('eq', 'pre', 'last') for k in [k for k in f0 if alt(k)][:20]]
    else:
        plan = np.load(glob.glob(os.path.join(root, '**', 'v216_plan.npz'), recursive=True)[0])
        pj = json.load(open(glob.glob(os.path.join(root, '**', 'v216_pairs.json'), recursive=True)[0]))
        pairs = pj['pairs']
        if SMOKE:
            pairs = [p for pos in pj['positions'] for p in [q for q in pairs if q['pos'] == pos][:20]]
    repo = 'Qwen/Qwen2.5-7B'; tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    dev = m.get_input_embeddings().weight.device; layers = m.model.layers
    digit = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(10)]
    log('v216 patch | %d pairs | smoke %s | items sha %s | loaded in %.0fs' % (len(pairs), SMOKE, s, time.time() - t0))
    st = {'t': None, 'mode': None, 'cap': {}, 'src': {}, 'U': {}}

    def hook(li):
        hi = li + 1

        def f(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out; t = st['t']
            if st['mode'] == 'cap':
                st['cap'][hi] = h[0, t].detach().clone()
            elif st['mode'] in KEY or st['mode'] == 'cR':
                if hi in BAND:
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
        enc = tok(x['prompt'] + x['target'], return_offsets_mapping=True); off = enc['offset_mapping']; ids = enc['input_ids']
        ids2 = tok(x2['prompt'] + x2['target']).input_ids; assert len(ids2) == len(ids)
        cover = lambda c: next(t for t, (u, v) in enumerate(off) if u <= c < v)
        t = cover(x['cpos'][p['pos']]); tA = cover(x['cA']); st['t'] = t
        st['mode'] = 'cap'; st['cap'] = {}; m(input_ids=torch.tensor([ids2], device=dev)); st['src'] = dict(st['cap'])
        r = dict(p, A=x['A'], A2=x2['A'], fam=x['fam'])
        for c in CONDS:
            st['mode'] = c
            if c in KEY or c == 'cR':
                st['U'] = {hi: torch.tensor(plan[('R_%s_%d' % (p['pos'], hi)) if (c == 'cR' or SMOKE) else
                                                 ('%s_%s_%d_%d' % (KEY[c], p['pos'], p['half'], hi))]) for hi in BAND}
            r[c] = int(m(input_ids=torch.tensor([ids], device=dev)).logits[0, tA - 1, digit].float().argmax())
        st['mode'] = None; res.append(r)
        if n + 1 in (1, 20, 100) or (n + 1) % 250 == 0 or n + 1 == len(pairs):
            log('%d/%d pairs (%.2f s/pair) | hit so far: %s' % (n + 1, len(pairs), (time.time() - t1) / (n + 1), ' '.join(
                '%s %.3f' % (c, np.mean([q[c] == q['A2'] for q in res])) for c in CONDS)))
    json.dump(dict(smoke=SMOKE, items_sha=s, results=res), open(os.path.join(OUT, 'v216_patch.json'), 'w'))
    log('done (%.0fs)' % (time.time() - t0))


def analyse(root, say=print):
    R = json.load(open(glob.glob(os.path.join(root, '**', 'v216_patch.json'), recursive=True)[0]))
    out = {}
    for pos in sorted(set(r['pos'] for r in R['results'])):
        rs = [r for r in R['results'] if r['pos'] == pos]; fam = np.array([r['fam'] for r in rs])
        hit = {c: np.array([r[c] == r['A2'] for r in rs], float) for c in CONDS}
        kept = {c: np.mean([r[c] == r['A'] for r in rs]) for c in CONDS}
        d = hit['cBo'] - hit['cR']; u = np.unique(fam); ix = {f: j for j, f in enumerate(u)}
        S = np.zeros(len(u)); N = np.zeros(len(u)); np.add.at(S, [ix[f] for f in fam], d); np.add.at(N, [ix[f] for f in fam], 1)
        rng = np.random.default_rng(2163); bs = []
        for _ in range(10000):
            q = rng.integers(0, len(u), len(u)); bs.append(S[q].sum() / N[q].sum())
        lo, hi = np.percentile(bs, (1.25, 98.75))
        lab = ('UNSCORABLE' if hit['full'].mean() < 0.30 else 'USED' if d.mean() >= 0.10 and lo > 0 else
               'NOT-USED' if hi < 0.02 else 'UNCLEAR')
        say('%s (%d pairs%s): hit %s | kept own answer %s | cBo - cR %+.4f [%+.4f, %+.4f] -> %s' % (
            pos, len(rs), ', SMOKE' if R['smoke'] else '', ' '.join('%s %.3f' % (c, hit[c].mean()) for c in CONDS),
            ' '.join('%s %.3f' % (c, kept[c]) for c in CONDS), d.mean(), lo, hi, lab))
        out[pos] = lab
    return out


if __name__ == '__main__':
    {'extract': extract, 'analyse': analyse}[sys.argv[1]](sys.argv[2])
