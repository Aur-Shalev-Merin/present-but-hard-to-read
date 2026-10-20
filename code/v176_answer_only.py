"""v176: answer-only GSM8K accuracy vs loop count on frozen looped LMs (plan + go/no-go in docs/v176_plan.md).

Model from LOOP_MODEL: 'huginn' (huginn-0125, read out after R in CK_HUGINN loops of ONE 64-loop run; the recursion does
not depend on the total R, so the readout after t loops equals an R = t run) or 'ouro' (Ouro-1.4B, read out after each
of its 4 loops).
Prompt: 4-shot, answer only (fixed GSM8K train examples 0-3), 'Question: ...\nAnswer: <n>\n\n' ... 'Answer:'.
Scoring (teacher forced, exact; v176b): correct = every token of ' <n>' is the argmax AND the argmax after the number is
not a continuation (does not start with a digit, '.' or ','). v176 required a single '\n' after the number, which
Huginn's tokenizer never emits after a few-shot answer (merged '\n\n' token) -> Huginn 0/200 by artefact.
Also the summed log-probability of the number tokens.
Check (logged): adapter vs the model's own forward on problem 0 (argmax agreement, max logit diff).
Data (V176_DATA): 'gsm8k' (first V176_N test problems, default 200); v177 memorization check: 'symbolic' (apple/GSM-Symbolic
main: the first V176_SYM_K instances, default 5, of each of its 100 templates) or 'symbolic_orig' (the 100 original GSM8K
test questions those templates come from). Same 4 GSM8K train shots throughout.
Env: V176_SHARD 'i/n', TINY=1 (local pipeline test), OUT_DIR.
"""
import json, os, re, time
import numpy as np
import torch
import torch.nn.functional as Fn

import loop_grad_probe as P

N = int(os.environ.get('V176_N', 3 if P.TINY else 200))
CK = {'huginn': [1, 2, 4, 8, 16, 32], 'ouro': [1, 2, 3, 4]}[P.FAMILY]      # v176b: Huginn to 32 (v176 went to 64)
if P.TINY and P.FAMILY == 'huginn':
    CK = [1, 2, 4]
OUT = P.OUT
DATA = os.environ.get('V176_DATA', 'gsm8k')
SYM_K = int(os.environ.get('V176_SYM_K', 5))
TAG = '%s_%s' % (P.MODEL, os.environ.get('V176_SHARD', 'all').replace('/', 'of'))
if DATA != 'gsm8k':
    TAG = '%s_%s' % (DATA, TAG)


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v176_%s_log.txt' % TAG), 'a').write(s + '\n')


def gold(ans):
    return ans.split('####')[-1].strip().replace(',', '')


def problems(tok):
    if P.TINY:
        shots = [('What is %d plus %d?' % (i, i + 1), str(2 * i + 1)) for i in range(4)]
        tests = [('What is %d plus %d?' % (i, 7), str(i + 7), i % 2) for i in range(N)]
    else:
        from datasets import load_dataset
        tr = load_dataset('openai/gsm8k', 'main', split='train')
        shots = [(tr[i]['question'], gold(tr[i]['answer'])) for i in range(4)]
        if DATA == 'gsm8k':
            te = load_dataset('openai/gsm8k', 'main', split='test')
            tests = [(te[i]['question'], gold(te[i]['answer']), i) for i in range(N)]
        else:
            sym = load_dataset('apple/GSM-Symbolic', 'main', split='test')
            rows = sorted(sym, key=lambda r: (r['original_id'], r['instance']))
            if DATA == 'symbolic':
                tests = [(r['question'], gold(r['answer']), r['original_id']) for r in rows if r['instance'] < SYM_K]
            else:                                          # 'symbolic_orig': the 100 originals, once each
                seen = {}
                for r in rows:
                    seen.setdefault(r['original_id'], (r['original_question'], gold(r['original_answer']), r['original_id']))
                tests = list(seen.values())
    head = ''.join('Question: %s\nAnswer: %s\n\n' % s for s in shots)
    out = []
    for q, a, tid in tests:
        prompt = head + 'Question: %s\nAnswer:' % q
        p_ids = tok(prompt).input_ids; f_ids = tok(prompt + ' ' + a).input_ids      # number only (v176b fix)
        ok = f_ids[:len(p_ids)] == p_ids
        out.append(dict(q=q, a=a, tid=tid, ids=torch.tensor(f_ids), n_prompt=len(p_ids), tok_ok=ok,
                        in_question=a in re.findall(r'\d+(?:\.\d+)?', q.replace(',', ''))))
    return out


def main():
    t0 = time.time()
    global tok
    tok, model = P.load()
    ad = {'ouro': P.Ouro, 'huginn': P.Huginn}[P.FAMILY](model)
    probs = problems(tok)
    shard = os.environ.get('V176_SHARD')
    keep = list(range(len(probs)))[int(shard.split('/')[0])::int(shard.split('/')[1])] if shard else list(range(len(probs)))
    log('v176 %s | dtype %s | %d problems (this shard %d) | readouts after %s loops' % (P.REPO, P.DTYPE, len(probs), len(keep), CK))

    @torch.no_grad()
    def run(pr, seed):
        ids = pr['ids'].unsqueeze(0).to(P.DEV); n0 = pr['n_prompt']
        if P.FAMILY == 'ouro':
            h, ctx = ad.prepare(ids)
        else:
            h, ctx = ad.prepare(ids)                     # Huginn: prepare seeds 0 before its random initial state
        gpos = torch.arange(n0 - 1, ids.shape[1], device=P.DEV); tgt = ids[0, n0:]       # last position: token after the number
        res = {}
        for t in range(max(CK)):
            h = ad.F(h, t, ctx)
            if t + 1 in CK:
                lg = ad.readout(h, ctx)[0, gpos].float()
                lp = Fn.log_softmax(lg[:-1], -1); am = lg.argmax(-1)
                nxt = tok.decode([int(am[-1])])
                stops = not (nxt[:1].isdigit() or nxt[:1] in '.,')
                res[t + 1] = dict(correct=bool((am[:-1] == tgt).all()) and stops, digits_ok=bool((am[:-1] == tgt).all()),
                                  next=nxt, logp=float(lp.gather(1, tgt[:, None]).sum()))
        return res

    # check: adapter vs the model's own forward on problem 0
    pr = probs[0]; ids = pr['ids'].unsqueeze(0).to(P.DEV)
    with torch.no_grad():
        h, ctx = ad.prepare(ids)
        for t in range(max(CK)):
            h = ad.F(h, t, ctx)
        mine = ad.readout(h, ctx)[0].float()
        if P.FAMILY == 'ouro':
            o = ad.b(input_ids=ids, use_cache=False); ref = model.lm_head(o[1][-1])[0].float()
        else:
            torch.manual_seed(0); ref = model(input_ids=ids, num_steps=max(CK)).logits[0].float()
    agree = float((mine.argmax(-1) == ref.argmax(-1)).float().mean()); dmax = float((mine - ref).abs().max())
    log('check vs model forward: argmax agreement %.4f, max |logit diff| %.4f' % (agree, dmax))
    rep = dict(model=P.REPO, data=DATA, dtype=str(P.DTYPE), ck=CK, n=len(probs), check=dict(argmax_agree=agree, max_logit_diff=dmax),
               gpu=(torch.cuda.get_device_name(0) if P.DEV == 'cuda' else 'cpu'), problems=[])
    for k, i in enumerate(keep):
        pr = probs[i]; ts = time.time(); r = run(pr, i)
        rep['problems'].append(dict(i=i, a=pr['a'], tid=pr['tid'], tok_ok=pr['tok_ok'], in_question=pr['in_question'],
                                    res={str(t): v for t, v in r.items()}))
        if k % 10 == 0 or k == len(keep) - 1:
            acc = {t: np.mean([p['res'][str(t)]['correct'] for p in rep['problems']]) for t in CK}
            log('%3d/%d  acc so far %s  (%.1fs/problem)' % (k + 1, len(keep), {t: round(float(v), 3) for t, v in acc.items()},
                                                           time.time() - ts))
            json.dump(rep, open(os.path.join(OUT, 'v176_%s.json' % TAG), 'w'))
    rep['wall_s'] = round(time.time() - t0)
    json.dump(rep, open(os.path.join(OUT, 'v176_%s.json' % TAG), 'w'))
    log('done', rep['wall_s'], 's')


if __name__ == '__main__':
    main()
