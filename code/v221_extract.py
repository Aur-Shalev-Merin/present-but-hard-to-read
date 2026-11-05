"""v221 extraction (GPU; docs/prereg_v221_positive_control.md). Qwen2.5-7B fp16 over two T4s. One teacher-forced
forward pass per item over prompt + " b = B , a = A". Saved: states at eq / last / letb / eqb / inb (in that order) x hidden_states
0, 4, ..., 28; correct = the b digit is the argmax at inb (v215's G1 for the short arm); predB and pB at inb.
V221_PILOT=1: 40 items. Outputs: v221_fam_states.npy, v221_fam_meta.json, v221_log.txt.
"""
import json, os, sys, time
import numpy as np

import v215_items as V5
import v221_items as I

OUT = os.environ.get('OUT_DIR', '.'); PILOT = os.environ.get('V221_PILOT') == '1'
LAYERS = list(range(0, 29, 4)); POS = ['eq', 'last', 'letb', 'eqb', 'inb']


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v221_log.txt'), 'a').write(s + '\n')


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False); t0 = time.time()
    its = I.build(); s = V5.sha16(its)
    log('v221 extract | pilot %s | items sha %s (frozen %s)' % (PILOT, s, I.ITEMS_SHA))
    if I.ITEMS_SHA and s != I.ITEMS_SHA:
        sys.exit('items hash differs from the frozen one')
    if PILOT:
        its = its[:40]
    repo = 'Qwen/Qwen2.5-7B'; tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    dev = m.get_input_embeddings().weight.device; d = m.config.hidden_size
    digit = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(10)]
    log('loaded %s in %.0fs | %d items' % (repo, time.time() - t0, len(its)))
    from numpy.lib.format import open_memmap
    X = open_memmap(os.path.join(OUT, 'v221_fam_states.npy'), mode='w+', dtype=np.float16, shape=(len(its), len(POS), len(LAYERS), d))
    meta = []; t1 = time.time()
    for n, it in enumerate(its):
        full = it['prompt'] + it['target']
        enc = tok(full, return_offsets_mapping=True); ids, off = enc['input_ids'], enc['offset_mapping']
        cover = lambda c: next(t for t, (u, v) in enumerate(off) if u <= c < v)
        tix = {k: cover(it['cpos'][k]) for k in POS}; tB = cover(it['cB'])
        assert ids[tB] == digit[it['B']] and (tix['letb'], tix['eqb'], tix['inb']) == (tB - 3, tB - 2, tB - 1), (full, tB, tix)
        out = m(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True)
        hs, lg = out.hidden_states, out.logits[0]
        for a, k in enumerate(POS):
            for b, l in enumerate(LAYERS):
                X[n, a, b] = hs[l][0, tix[k]].float().cpu().numpy()
        dl = lg[tB - 1, digit].float().softmax(-1).cpu().numpy()
        meta.append(dict(n=n, correct=bool(dl.argmax() == it['B']), predB=int(dl.argmax()), pB=float(dl[it['B']]), tok=tix, T=len(ids)))
        if n == 0:
            log('positions %s -> %s | b digit token %d %r' % (tix, {k: tok.decode([ids[t]]) for k, t in tix.items()}, tB, tok.decode([ids[tB]])))
        if n + 1 in (1, 100, 500) or (n + 1) % 2000 == 0 or n + 1 == len(its):
            log('fam %5d/%d (%.3f s/item) | b step correct so far %.3f' % (n + 1, len(its), (time.time() - t1) / (n + 1),
                                                                          np.mean([x['correct'] for x in meta])))
    X.flush()
    json.dump(dict(items_sha=s, layers=LAYERS, positions=POS, items=meta), open(os.path.join(OUT, 'v221_fam_meta.json'), 'w'))
    log('done: %d items, b step correct %.4f (%.0fs)' % (len(its), np.mean([x['correct'] for x in meta]), time.time() - t0))


if __name__ == '__main__':
    main()
