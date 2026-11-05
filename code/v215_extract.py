"""v215 extraction (GPU; docs/prereg_v215_kudo_retest.md). Qwen2.5-7B fp16 over two T4s.

One forward pass per item over prompt + target (teacher forcing). Saved per item:
  states   (3 positions eq / semi / last of the prompt) x (8 layers: hidden_states 0, 4, ..., 28) x d, float16
           (causal model: states at prompt positions do not depend on the target)
  correct  every target token is the argmax of the logits before it (= greedy decoding writes exactly this chain)
  predB    argmax over the ten digit tokens at the chain's step "b =" (the token before the computed digit), and pB
           the probability of the true B digit there
Parts: V215_PART = fam | fams (the same families in their Simple CoT format) | rep. V215_PILOT=1: 40 items per part.
Outputs (OUT_DIR): v215_<part>_states.npy, v215_<part>_meta.json, v215_log.txt.
"""
import json, os, sys, time
import numpy as np

import v215_items as I

OUT = os.environ.get('OUT_DIR', '.'); PART = os.environ.get('V215_PART', 'fam'); PILOT = os.environ.get('V215_PILOT') == '1'
LAYERS = list(range(0, 29, 4)); POS = ['eq', 'semi', 'last']


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v215_log.txt'), 'a').write(s + '\n')


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False); t0 = time.time()
    fam, rep, info = I.build(); s = I.all_sha()
    log('v215 extract | part %s | pilot %s | items sha %s (frozen %s)' % (PART, PILOT, s, I.ITEMS_SHA))
    if I.ITEMS_SHA and s != I.ITEMS_SHA:
        sys.exit('items hash differs from the frozen one')
    its = {'fam': fam, 'fams': I.build_short(), 'rep': rep}[PART]
    if PILOT:
        its = its[:40]
    repo = 'Qwen/Qwen2.5-7B'; tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    dev = m.get_input_embeddings().weight.device; d = m.config.hidden_size
    digit = [tok.encode(str(k), add_special_tokens=False) for k in range(10)]
    assert all(len(x) == 1 for x in digit), digit
    digit = [x[0] for x in digit]
    log('loaded %s in %.0fs | %d items | digit ids %s' % (repo, time.time() - t0, len(its), digit))
    from numpy.lib.format import open_memmap
    X = open_memmap(os.path.join(OUT, 'v215_%s_states.npy' % PART), mode='w+', dtype=np.float16, shape=(len(its), 3, len(LAYERS), d))
    meta = []; t1 = time.time(); bad_boundary = 0
    for n, it in enumerate(its):
        full = it['prompt'] + it['target']
        enc = tok(full, return_offsets_mapping=True); ids, off = enc['input_ids'], enc['offset_mapping']
        cover = lambda c: next(t for t, (u, v) in enumerate(off) if u <= c < v)
        tix = {k: cover(c) for k, c in it['cpos'].items()}
        t_first = next(t for t, (u, v) in enumerate(off) if u >= len(it['prompt']))
        bad_boundary += off[t_first][0] != len(it['prompt']) or tix['last'] != t_first - 1
        tB = cover(it['cB']); assert ids[tB] == digit[it['B']], (full, tB)
        out = m(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True)
        hs, lg = out.hidden_states, out.logits[0]
        for a, k in enumerate(POS):
            for b, l in enumerate(LAYERS):
                X[n, a, b] = hs[l][0, tix[k]].float().cpu().numpy()
        pred = lg[t_first - 1:len(ids) - 1].argmax(-1).cpu().numpy()
        correct = bool((pred == np.array(ids[t_first:])).all())
        dl = lg[tB - 1, digit].float().softmax(-1).cpu().numpy()
        meta.append(dict(n=n, correct=correct, predB=int(dl.argmax()), pB=float(dl[it['B']]), tok=tix, tB=tB, T=len(ids)))
        if n == 0:
            log('positions %s -> %s | b digit token %d %r | target starts at token %d' % (
                tix, {k: tok.decode([ids[t]]) for k, t in tix.items()}, tB, tok.decode([ids[tB]]), t_first))
        if n + 1 in (1, 100, 500) or (n + 1) % 2000 == 0 or n + 1 == len(its):
            log('%s %5d/%d (%.3f s/item) | correct so far %.3f, predB = B %.3f' % (
                PART, n + 1, len(its), (time.time() - t1) / (n + 1), np.mean([x['correct'] for x in meta]),
                np.mean([x['predB'] == its[x['n']]['B'] for x in meta])))
    X.flush()
    json.dump(dict(part=PART, items_sha=s, layers=LAYERS, positions=POS, bad_boundary=bad_boundary, items=meta),
              open(os.path.join(OUT, 'v215_%s_meta.json' % PART), 'w'))
    log('%s done: %d items, boundary mismatches %d, correct %.4f (%.0fs)' % (
        PART, len(its), bad_boundary, np.mean([x['correct'] for x in meta]), time.time() - t0))


if __name__ == '__main__':
    main()
