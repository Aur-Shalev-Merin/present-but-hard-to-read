"""v176 diagnostic (after the v176 result, NOT part of the frozen test): why does Huginn score 0/200 answer-only?
On 20 GSM8K test problems at 32 loops, for two prompt formats:
  A  the v176 plain 4-shot format ('Question: ...\nAnswer: <n>')
  B  Huginn's chat template: system 'answer with only the final number', 4 shots as user/assistant turns
print the top-5 next tokens at the first answer position and score exact match on the number tokens (teacher forced).
"""
import os
os.environ.setdefault('LOOP_MODEL', 'huginn')
import torch
import torch.nn.functional as Fn

import loop_grad_probe as P
import v176_answer_only as V

R = 4 if P.TINY else 32
NP = 3 if P.TINY else 20
SYS = 'You are a careful math solver. Reply with only the final numerical answer, no words, no working.'


def main():
    tok, model = P.load(); ad = P.Huginn(model)
    if P.TINY:
        shots = [('What is %d plus %d?' % (i, i + 1), str(2 * i + 1)) for i in range(4)]
        tests = [('What is %d plus 7?' % i, str(i + 7)) for i in range(NP)]
    else:
        from datasets import load_dataset
        tr = load_dataset('openai/gsm8k', 'main', split='train'); te = load_dataset('openai/gsm8k', 'main', split='test')
        shots = [(tr[i]['question'], V.gold(tr[i]['answer'])) for i in range(4)]
        tests = [(te[i]['question'], V.gold(te[i]['answer'])) for i in range(NP)]
    head = ''.join('Question: %s\nAnswer: %s\n\n' % s for s in shots)

    def fmt_a(q, a):
        p = tok(head + 'Question: %s\nAnswer:' % q).input_ids
        return p, tok(head + 'Question: %s\nAnswer: %s' % (q, a)).input_ids

    def fmt_b(q, a):
        msgs = [dict(role='system', content=SYS)]
        for sq, sa in shots:
            msgs += [dict(role='user', content=sq), dict(role='assistant', content=sa)]
        msgs.append(dict(role='user', content=q))
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return tok(p, add_special_tokens=False).input_ids, tok(p + a, add_special_tokens=False).input_ids

    for name, fmt in (('A plain few-shot', fmt_a), ('B chat template', fmt_b)):
        hits = 0
        for k, (q, a) in enumerate(tests):
            p, f = fmt(q, a)
            assert f[:len(p)] == p, 'prefix mismatch'
            ids = torch.tensor(f, device=P.DEV).unsqueeze(0)
            with torch.no_grad():
                h, ctx = ad.prepare(ids)
                for t in range(R):
                    h = ad.F(h, t, ctx)
                lg = ad.readout(h, ctx)[0].float()
            gpos = torch.arange(len(p) - 1, len(f) - 1, device=P.DEV); tgt = ids[0, len(p):]
            ok = bool((lg[gpos].argmax(-1) == tgt).all()); hits += ok
            if k < 6:
                top = Fn.softmax(lg[len(p) - 1], -1).topk(5)
                print('%s | gold %r -> %s | top5 %s' % (name[:1], a, 'OK' if ok else 'x',
                      [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)]), flush=True)
        print('== %s: %d/%d exact at %d loops' % (name, hits, len(tests), R), flush=True)


if __name__ == '__main__':
    main()
