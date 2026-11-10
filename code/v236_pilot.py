"""v236 pilot (GPU; pre-design, whole-state patches only, no learned subspace): where does b-relevant information that
the answer uses sit, in Kudo et al.'s Implicit format without a chain?

Items: v234's frozen items (v234_items.build(), ITEMS_SHA ac4121c22f88fe31). Pairs share family and version and have
equal answers A (k varies), both answered correctly (argmax over the ten digits at pre):
  D1  same d2, different d1   (new: at eq, the d2 token, the two prompts have the same token)
  D2  same d1, different d2   (v234's pairs)
Prediction under a used b: P = k o2 b' (x's k with x''s b). hit = P, kept = A.
Positions (token indices found from character offsets): d1, o1 (the operator after d1), sp (the space before d2),
eq (d2), semi (' ;'), qa (the question's letter), last ('?'), ta (the target's letter), teq (' ='), pre (the space
before the answer digit, where the answer is read).
Patches: x''s state replaces x's at one position, at one layer (hidden_states l = output of decoder layer l - 1, l in
LAYERS), or at all layers 1-27 ('all'); one batched pass per pair and position.
Pairs: up to N_PER per type (seed 2360), candidates checked for correctness in the same pass.

  OUT_DIR=... python v236_pilot.py
"""
import json, os, sys, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24]; N_PER = int(os.environ.get('V236_N', '300'))
POS = ['d1', 'o1', 'sp', 'eq', 'semi', 'qa', 'last', 'ta', 'teq', 'pre']; TYPES = ['D1', 'D2']


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v236_pilot_log.txt'), 'a').write(s + '\n')


def candidates(its, rng):
    out = {}
    for t, same, diff in (('D1', 'd2', 'd1'), ('D2', 'd1', 'd2')):
        key = {}
        for j, i in enumerate(its):
            key.setdefault((i['fam'], i['ver'], i[same], i['A']), []).append(j)
        c = [(j, m) for g in key.values() for j in g for m in g if its[j][diff] != its[m][diff]]
        out[t] = [c[q] for q in rng.permutation(len(c))]
    return out


def positions(x, off):
    cover = lambda c: next(t for t, (u, v) in enumerate(off) if u <= c < v)
    eq = cover(x['cpos']['eq']); pre = cover(x['cpos']['pre']); last = cover(x['cpos']['last'])
    return dict(d1=eq - 3, o1=eq - 2, sp=eq - 1, eq=eq, semi=eq + 1, qa=last - 1, last=last, ta=last + 1, teq=pre - 1, pre=pre)


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import v215_items as V5, v234_items as I
    from v215_pilot import op
    torch.set_grad_enabled(False); t0 = time.time()
    its = I.build(); s = V5.sha16(its); assert s == I.ITEMS_SHA, s
    repo = 'Qwen/Qwen2.5-7B'; tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    dev = m.get_input_embeddings().weight.device; layers = m.model.layers; NL = len(layers)
    digit = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(10)]
    st = {'mode': None}

    def hook(li):
        hi = li + 1

        def f(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if st['mode'] == 'patch':
                for j, ls in enumerate(st['rows']):
                    if hi in ls:
                        h[j, st['t']] = st['src'][hi].to(h.device, h.dtype)
            return out
        return f
    for li in range(NL):
        layers[li].register_forward_hook(hook(li))
    rows = [[l] for l in LAYERS] + [list(range(1, NL))]; rlab = ['L%d' % l for l in LAYERS] + ['all']

    def enc(x):
        e = tok(x['prompt'] + x['target'], return_offsets_mapping=True); P = positions(x, e['offset_mapping'])
        return e['input_ids'], P

    x0 = its[5]; ids, P = enc(x0)
    log('loaded in %.0fs | items sha %s | positions on item 5: %s' % (time.time() - t0, s, {k: (v, tok.decode([ids[v]])) for k, v in P.items()}))
    cand = candidates(its, np.random.default_rng(2360)); res = []; t1 = time.time(); seen = {}
    for t in TYPES:
        n_ok = n_tried = 0
        for j, mm in cand[t]:
            if n_ok >= N_PER:
                break
            x, x2 = its[j], its[mm]; n_tried += 1
            ids, P = enc(x); ids2, P2 = enc(x2); assert len(ids) == len(ids2) and P == P2
            H = m(input_ids=torch.tensor([ids2], device=dev), output_hidden_states=True)
            ok2 = int(H.logits[0, P['pre'], digit].float().argmax()) == x2['A']
            ok1 = int(m(input_ids=torch.tensor([ids], device=dev)).logits[0, P['pre'], digit].float().argmax()) == x['A']
            if not (ok1 and ok2):
                continue
            hs = H.hidden_states; Pp = op(x['k'], x['o2'], x2['B'])
            r = dict(type=t, x=j, x2=mm, A=x['A'], P=Pp, fam=x['fam'])
            for p in POS:
                st.update(mode='patch', rows=rows, t=P[p], src={l: hs[l][0, P[p]] for l in range(1, NL)})
                out = m(input_ids=torch.tensor([ids] * len(rows), device=dev)).logits[:, P['pre'], digit].float().argmax(-1).tolist()
                st['mode'] = None; r[p] = dict(zip(rlab, out))
            res.append(r); n_ok += 1
            if n_ok in (1, 20, 100) or n_ok % 100 == 0:
                log('%s: %d pairs (%d tried; %.2f s/pair)' % (t, n_ok, n_tried, (time.time() - t1) / len(res)))
        log('%s: %d pairs from %d candidates tried' % (t, n_ok, n_tried))
    json.dump(dict(items_sha=s, rows=rlab, positions=POS, results=res), open(os.path.join(OUT, 'v236_pilot.json'), 'w'))
    for t in TYPES:
        rs = [r for r in res if r['type'] == t]
        log('\n%s (%d pairs): hit P (kept A) by position and layer' % (t, len(rs)))
        log('%-5s ' % 'pos' + ' '.join('%11s' % k for k in rlab))
        for p in POS:
            log('%-5s ' % p + ' '.join('%5.2f(%4.2f)' % (np.mean([r[p][k] == r['P'] for r in rs]), np.mean([r[p][k] == r['A'] for r in rs]))
                                      for k in rlab))
    log('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
