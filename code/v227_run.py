"""v227: replacement positive control for v224's share swap (docs/prereg_v227_share_replacement.md).

v224's control 1 (Qwen2.5-1.5B, b step) failed its accuracy gate. This runs the same control with Mistral-7B-v0.1
(a Qwen2.5-3B pilot failed first: b correct 0.54 / 0.59): v224's b-set items (= v222's), positions eq, last, letb, eqb,
inb, and v224_share.analyse unchanged (TAU 0.30, gate 0.80). Mistral-7B has 32 layers; v224's layers {16, 20, 24, 28}
of 28 map to the same relative depth, round(l * 32 / 28) = {18, 23, 27, 32}, the only states saved. Mistral encodes a
digit as a separate space token then the digit, so the digit's id is the last token of its encoding.

  pilot    GPU: 400 items drawn at random over all families (seed 2271), b step accuracy only, no states
  extract  GPU: all 12,000 items -> v227_states.npy, v227_meta.json
  analyse  CPU: v224_share.analyse at inb (primary), eqb, letb, eq, last -> v227_result.json
"""
import glob, json, os, sys, time
import numpy as np

import v215_items as V5
import v224_items as I

OUT = os.environ.get('OUT_DIR', '.'); REPO = 'mistralai/Mistral-7B-v0.1'; NL = 32
POS = ['eq', 'last', 'letb', 'eqb', 'inb']; MAP = {l: round(l * NL / 28) for l in (16, 20, 24, 28)}; SAVE = sorted(MAP.values())


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v227_log.txt'), 'a').write(s + '\n')


def items():
    import v222_items as V2
    its = I.build_b(); s = V5.sha16(its); assert s == V2.ITEMS_SHA, s
    return its, s


def gpu(pilot):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False); t0 = time.time()
    its, s = items(); sel = np.sort(np.random.default_rng(2271).choice(len(its), 400, replace=False)) if pilot else np.arange(len(its))
    tok = AutoTokenizer.from_pretrained(REPO)
    m = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float16, device_map='auto').eval()
    assert m.config.num_hidden_layers == NL, m.config.num_hidden_layers
    dev = m.get_input_embeddings().weight.device; d = m.config.hidden_size
    digit = [tok.encode(str(k), add_special_tokens=False)[-1] for k in range(10)]
    log('v227 %s | %s | items sha %s | %d items | layer map %s | loaded in %.0fs' % ('pilot' if pilot else 'extract', REPO, s, len(sel), MAP, time.time() - t0))
    if not pilot:
        from numpy.lib.format import open_memmap
        X = open_memmap(os.path.join(OUT, 'v227_states.npy'), mode='w+', dtype=np.float16, shape=(len(its), len(POS), len(SAVE), d))
    meta = []; t1 = time.time()
    for n, j in enumerate(sel):
        it = its[j]; enc = tok(it['prompt'] + it['target'], return_offsets_mapping=True); ids, off = enc['input_ids'], enc['offset_mapping']
        cover = lambda c: next(t for t, (u, v) in enumerate(off) if u <= c < v)
        tix = {k: cover(it['cpos'][k]) for k in POS}; tB = cover(it['cB'])
        assert ids[tB] == digit[it['B']] and (tix['letb'], tix['eqb'], tix['inb']) == (tB - 3, tB - 2, tB - 1), (it['target'], tix)
        out = m(input_ids=torch.tensor([ids], device=dev), output_hidden_states=not pilot); lg = out.logits[0]
        if not pilot:
            for a, k in enumerate(POS):
                for b, l in enumerate(SAVE):
                    X[n, a, b] = out.hidden_states[l][0, tix[k]].float().cpu().numpy()
        meta.append(dict(n=int(j), ver=it['ver'], okB=int(lg[tB - 1, digit].float().argmax()) == it['B']))
        if n == 0:
            log('positions %s -> %s' % (tix, {k: tok.decode([ids[t]]) for k, t in tix.items()}))
        if n + 1 in (1, 100) or (n + 1) % 2000 == 0 or n + 1 == len(sel):
            log('item %5d/%d (%.3f s/item) | b correct %.3f' % (n + 1, len(sel), (time.time() - t1) / (n + 1), np.mean([x['okB'] for x in meta])))
    for v in ('plus', 'minus'):
        log('%s: b correct %.3f (%d items)' % (v, np.mean([x['okB'] for x in meta if x['ver'] == v]), sum(x['ver'] == v for x in meta)))
    if not pilot:
        X.flush()
        json.dump(dict(items_sha=s, model=REPO, layers=SAVE, positions=POS, items=meta), open(os.path.join(OUT, 'v227_meta.json'), 'w'))
    log('done (%.0fs)' % (time.time() - t0))


def analyse(root):
    import v224_share as SH
    its, _ = items(); its = [dict(i, u=i['d1'], v=i['d2']) for i in its]
    f = glob.glob(os.path.join(root, '**', 'v227_states.npy'), recursive=True)[0]
    X = np.load(f, mmap_mode='r'); m = json.load(open(f.replace('_states.npy', '_meta.json')))
    assert len(m['items']) == len(its) and m['layers'] == SAVE; ok = [x['okB'] for x in m['items']]
    print('TAU %.2f (v224, frozen); layer map %s' % (SH.TAU, MAP), flush=True)
    out = {}
    for p in ['inb', 'eqb', 'letb', 'eq', 'last']:
        r = SH.analyse(lambda idx, l: X[idx, POS.index(p), SAVE.index(MAP[l])], its, ok); out[p] = r
        print(SH.fmt(p, r), flush=True)
        print('       shares: ' + ' '.join('%s %.3f' % (k, s) for k, s in r['shares'].items()) + ' | G1 ' + str(r['g1']), flush=True)
    lab = out['inb']['label']
    out['verdict'] = 'VALID' if lab == 'FOLLOWS' else 'PARTIAL'
    print('primary inb %s -> share swap %s (with v224: control 2 FOLLOWS, fresh nulls 0/420)' % (lab, out['verdict']), flush=True)
    json.dump(out, open(os.path.join(OUT, 'v227_result.json'), 'w'), indent=1)


if __name__ == '__main__':
    a = sys.argv[1]
    if a in ('pilot', 'extract'):
        gpu(a == 'pilot')
    else:
        analyse(sys.argv[2])
