"""v216 pilot (exploratory; accuracy only, no probe). Kudo et al.'s Level 3 task in their Implicit format: the model
writes only the answer (" a = A"), with three demonstrations in the same format. Qwen2.5-7B fp16 over two T4s.

Measured by one teacher-forced forward pass per problem (= greedy decoding writes exactly " a = A"):
  answer accuracy overall, by operation pair (o1, o2) and by Level (3: a = k o2 b , b = d1 o1 d2 ; a?; 2-in-order:
  b = d1 o1 d2 , a = k o2 b ; a?), against the majority-answer rate; N_PER problems per level.
"""
import json, os, time
import numpy as np

from v215_pilot import op, spaced, sample, render

OUT = os.environ.get('OUT_DIR', '.'); N_PER = int(os.environ.get('V216_N', '1000'))


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v216_pilot_log.txt'), 'a').write(s + '\n')


def render_implicit(p, level):
    B = op(p['d1'], p['o1'], p['d2']); A = op(p['k'], p['o2'], B)
    e_a = '%s=%d%s%s' % (p['a'], p['k'], p['o2'], p['b']); e_b = '%s=%d%s%d' % (p['b'], p['d1'], p['o1'], p['d2'])
    passage = '%s, %s' % ((e_a, e_b) if level == 3 else (e_b, e_a))
    return '%s ; %s?' % (spaced(passage), spaced(p['a'])), ' ' + spaced('%s=%d' % (p['a'], A)) + '\n', A


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False); t0 = time.time(); repo = 'Qwen/Qwen2.5-7B'
    tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    dev = m.get_input_embeddings().weight.device
    log('v216 pilot | %s | %.0fs to load | %d problems per level' % (repo, time.time() - t0, N_PER))
    rng = np.random.default_rng(216)
    for level in (3, 2):
        res = []; t1 = time.time(); seen = set()
        while len(res) < N_PER:
            p = sample(rng); src, tgt, A = render_implicit(p, level)
            if src in seen:
                continue
            seen.add(src)
            prompt = ''.join(''.join(render_implicit(q, level)[:2]) for q in [sample(rng) for _ in range(3)]) + src
            if not res:
                log('level %d example:\n%s%s' % (level, prompt, tgt.rstrip()))
            enc = tok(prompt + tgt, return_offsets_mapping=True); ids, off = enc['input_ids'], enc['offset_mapping']
            tf = next(t for t, (u, v) in enumerate(off) if u >= len(prompt))
            pred = m(input_ids=torch.tensor([ids], device=dev)).logits[0, tf - 1:len(ids) - 1].argmax(-1).cpu().numpy()
            res.append(dict(ok=bool((pred == np.array(ids[tf:])).all()), A=A, o1=p['o1'], o2=p['o2'],
                            B=op(p['d1'], p['o1'], p['d2'])))
            if len(res) in (1, 100, 500) or len(res) % 1000 == 0:
                log('level %d %4d/%d (%.3f s/problem) | correct so far %.3f' % (
                    level, len(res), N_PER, (time.time() - t1) / len(res), np.mean([r['ok'] for r in res])))
        acc = np.mean([r['ok'] for r in res]); maj = np.bincount([r['A'] for r in res]).max() / len(res)
        log('LEVEL %d: answer-only accuracy %.3f (majority answer %.3f)' % (level, acc, maj))
        for o1 in '+-':
            for o2 in '+-':
                s = [r['ok'] for r in res if r['o1'] == o1 and r['o2'] == o2]
                log('  b op %s, a op %s: %.3f (n %d)' % (o1, o2, np.mean(s), len(s)))
        json.dump(res, open(os.path.join(OUT, 'v216_pilot_level%d.json' % level), 'w'))
    log('done (%.0fs)' % (time.time() - t0))


if __name__ == '__main__':
    main()
