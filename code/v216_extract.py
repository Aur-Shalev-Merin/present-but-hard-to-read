"""v216 extraction (GPU; docs/prereg_v216_answer_only.md). Qwen2.5-7B fp16 over two T4s. One teacher-forced forward pass
per item over prompt + " a = A". Saved: states at eq / pre / last (in that order) x hidden_states 0, 4, ..., 28;
correct = the answer digit is the argmax at pre; predA and pA (probability of A) at pre.
V216_PILOT=1: 40 items. Outputs: v216_fam_states.npy, v216_fam_meta.json, v216_log.txt.
"""
import json, os, sys, time
import numpy as np

import v215_items as V5
import v216_items as I

OUT = os.environ.get('OUT_DIR', '.'); PILOT = os.environ.get('V216_PILOT') == '1'
LAYERS = list(range(0, 29, 4)); POS = ['eq', 'pre', 'last']


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v216_log.txt'), 'a').write(s + '\n')


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False); t0 = time.time()
    its = I.build(); s = V5.sha16(its)
    log('v216 extract | pilot %s | items sha %s (frozen %s)' % (PILOT, s, I.ITEMS_SHA))
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
    X = open_memmap(os.path.join(OUT, 'v216_fam_states.npy'), mode='w+', dtype=np.float16, shape=(len(its), 3, len(LAYERS), d))
    meta = []; t1 = time.time()
    for n, it in enumerate(its):
        full = it['prompt'] + it['target']
        enc = tok(full, return_offsets_mapping=True); ids, off = enc['input_ids'], enc['offset_mapping']
        cover = lambda c: next(t for t, (u, v) in enumerate(off) if u <= c < v)
        tix = {k: cover(c) for k, c in it['cpos'].items()}; tA = cover(it['cA'])
        assert ids[tA] == digit[it['A']] and tix['pre'] == tA - 1, (full, tA, tix)
        out = m(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True)
        hs, lg = out.hidden_states, out.logits[0]
        for a, k in enumerate(POS):
            for b, l in enumerate(LAYERS):
                X[n, a, b] = hs[l][0, tix[k]].float().cpu().numpy()
        dl = lg[tA - 1, digit].float().softmax(-1).cpu().numpy()
        meta.append(dict(n=n, correct=bool(dl.argmax() == it['A']), predA=int(dl.argmax()), pA=float(dl[it['A']]), tok=tix, T=len(ids)))
        if n == 0:
            log('positions %s -> %s | answer token %d %r' % (tix, {k: tok.decode([ids[t]]) for k, t in tix.items()}, tA, tok.decode([ids[tA]])))
        if n + 1 in (1, 100, 500) or (n + 1) % 2000 == 0 or n + 1 == len(its):
            log('fam %5d/%d (%.3f s/item) | answer correct so far %.3f' % (n + 1, len(its), (time.time() - t1) / (n + 1),
                                                                         np.mean([x['correct'] for x in meta])))
    X.flush()
    json.dump(dict(items_sha=s, layers=LAYERS, positions=POS, items=meta), open(os.path.join(OUT, 'v216_fam_meta.json'), 'w'))
    log('done: %d items, answer correct %.4f (%.0fs)' % (len(its), np.mean([x['correct'] for x in meta]), time.time() - t0))


if __name__ == '__main__':
    main()
