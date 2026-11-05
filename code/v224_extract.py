"""v224 extraction (GPU; docs/prereg_v224_share_swap.md). One teacher-forced pass per item over prompt + " b = B , a = A",
fp16 over two T4s; states at the set's positions x hidden_states 0, 4, ..., 28.
  V224_SET=b  build_b(), positions eq, last, letb, eqb, inb      (the b step)
  V224_SET=a  build_a(), positions eq, last, digB, eqa, inA      (the a step)
  V224_MODEL=1.5b | 7b    Qwen/Qwen2.5-1.5B | Qwen/Qwen2.5-7B (both 28 layers)
  V224_N=<n>  only the first n items (smoke)
correct: okB = the b digit is the argmax at inb, okA = the a digit is the argmax at inA.
Outputs: v224_<set>_<model>_states.npy, v224_<set>_<model>_meta.json, v224_log.txt.
"""
import json, os, sys, time
import numpy as np

import v215_items as V5
import v224_items as I

OUT = os.environ.get('OUT_DIR', '.'); SET = os.environ.get('V224_SET', 'b'); MODEL = os.environ.get('V224_MODEL', '7b')
NMAX = int(os.environ.get('V224_N', '0')); LAYERS = list(range(0, 29, 4))
POS = {'b': ['eq', 'last', 'letb', 'eqb', 'inb'], 'a': ['eq', 'last', 'digB', 'eqa', 'inA']}[SET]
REPO = {'1.5b': 'Qwen/Qwen2.5-1.5B', '7b': 'Qwen/Qwen2.5-7B'}[MODEL]


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v224_log.txt'), 'a').write(s + '\n')


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False); t0 = time.time()
    its = I.build_b() if SET == 'b' else I.build_a(); s = V5.sha16(its)
    want = '' if SET == 'b' else I.ITEMS_SHA_A
    if SET == 'b':
        import v222_items as V2; want = V2.ITEMS_SHA
    log('v224 extract | set %s | model %s | items sha %s (frozen %s) | n %d' % (SET, REPO, s, want, NMAX or len(its)))
    if want and s != want:
        sys.exit('items hash differs from the frozen one')
    its = its[:NMAX] if NMAX else its
    tok = AutoTokenizer.from_pretrained(REPO)
    m = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float16, device_map='auto').eval()
    assert m.config.num_hidden_layers == 28, m.config.num_hidden_layers
    dev = m.get_input_embeddings().weight.device; d = m.config.hidden_size
    digit = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(10)]
    log('loaded in %.0fs | hidden %d' % (time.time() - t0, d))
    from numpy.lib.format import open_memmap
    tag = 'v224_%s_%s' % (SET, MODEL)
    X = open_memmap(os.path.join(OUT, tag + '_states.npy'), mode='w+', dtype=np.float16, shape=(len(its), len(POS), len(LAYERS), d))
    meta = []; t1 = time.time()
    for n, it in enumerate(its):
        enc = tok(it['prompt'] + it['target'], return_offsets_mapping=True); ids, off = enc['input_ids'], enc['offset_mapping']
        cover = lambda c: next(t for t, (u, v) in enumerate(off) if u <= c < v)
        tix = {k: cover(it['cpos'][k]) for k in POS}; tB, tA = cover(it['cB']), cover(it['cA'])
        assert ids[tB] == digit[it['B']] and ids[tA] == digit[it['A']], (it['target'], tB, tA)
        if SET == 'b':
            assert (tix['letb'], tix['eqb'], tix['inb']) == (tB - 3, tB - 2, tB - 1), tix
        else:
            assert (tix['digB'], tix['eqa'], tix['inA']) == (tB, tA - 2, tA - 1), tix
        out = m(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True); hs, lg = out.hidden_states, out.logits[0]
        for a, k in enumerate(POS):
            for b, l in enumerate(LAYERS):
                X[n, a, b] = hs[l][0, tix[k]].float().cpu().numpy()
        pb, pa = (int(lg[t - 1, digit].float().argmax()) for t in (tB, tA))
        meta.append(dict(n=n, okB=pb == it['B'], okA=pa == it['A']))
        if n == 0:
            log('positions %s -> %s | b digit %d, a digit %d' % (tix, {k: tok.decode([ids[t]]) for k, t in tix.items()}, tB, tA))
        if n + 1 in (1, 100) or (n + 1) % 2000 == 0 or n + 1 == len(its):
            log('item %5d/%d (%.3f s/item) | b correct %.3f, a correct %.3f' % (
                n + 1, len(its), (time.time() - t1) / (n + 1), np.mean([x['okB'] for x in meta]), np.mean([x['okA'] for x in meta])))
    X.flush()
    json.dump(dict(items_sha=s, set=SET, model=REPO, layers=LAYERS, positions=POS, n=len(its), items=meta),
              open(os.path.join(OUT, tag + '_meta.json'), 'w'))
    for v in ('plus', 'minus'):
        k = [j for j, i in enumerate(its) if i['ver'] == v]
        log('%s: b correct %.3f, a correct %.3f (%d items)' % (v, np.mean([meta[j]['okB'] for j in k]), np.mean([meta[j]['okA'] for j in k]), len(k)))
    log('done (%.0fs)' % (time.time() - t0))


if __name__ == '__main__':
    main()
