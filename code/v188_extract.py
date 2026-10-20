"""v188 state extraction (identical to the frozen v186_extract.py except file names; docs/prereg_v188_*.md).

For each item of v188_items.json in the selected sets: the v176-v185 4-shot answer-only prompt; ONE forward pass over
prompt + ' <answer>' (teacher-forced). States at two positions:
  P0 = the last prompt token (after 'Answer:'), before any answer token;
  P1 = the answer token just before the answer's units digit (its next-token prediction is the units digit);
in order [mid1, end1, mid2, end2, mid3, end3, mid4, end4] (midT = residual after layer 12 of loop T; endT = normed
state after loop T). Also answer-only correctness after each loop (v176b scorer).
Check (logged, abort on failure): this forward's final logits == Ouro's own forward on item 0.
Outputs (OUT_DIR): v188_states.npy (float16, N x 2 x 8 x d), v188_meta.json.
Env: V188_SETS (comma list, default 'main,bare'), V188_PART=0/1 (half of the selected items), TINY=1, OUT_DIR.
"""
import json, os, time
os.environ.setdefault('LOOP_MODEL', 'ouro')
import numpy as np
import torch

import loop_grad_probe as P

MID = 12
OUT = P.OUT
STATES = ['mid1', 'end1', 'mid2', 'end2', 'mid3', 'end3', 'mid4', 'end4']


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v188_log.txt'), 'a').write(s + '\n')


def gold(ans):
    return ans.split('####')[-1].strip().replace(',', '')


def main():
    t0 = time.time()
    sets = os.environ.get('V188_SETS', 'main,bare').split(',')
    items = [i for i in json.load(open('v188_items.json'))['items'] if i['set'] in sets]
    if P.TINY:
        items = items[:6] + items[-6:]
    part = os.environ.get('V188_PART')
    if part is not None:
        h = (len(items) + 1) // 2; items = items[:h] if part == '0' else items[h:]
    tok, model = P.load(); ad = P.Ouro(model); b_ = ad.b
    L = b_.config.num_hidden_layers; mid = min(MID, L) - 1
    if P.TINY:
        shots = [('What is %d plus %d?' % (i, i + 1), str(2 * i + 1)) for i in range(4)]
    else:
        from datasets import load_dataset
        tr = load_dataset('openai/gsm8k', 'main', split='train')
        shots = [(tr[i]['question'], gold(tr[i]['answer'])) for i in range(4)]
    head = ''.join('Question: %s\nAnswer: %s\n\n' % s for s in shots)
    log('v188 extract | %s | sets %s | %d items (part %s) | dtype %s' % (P.REPO, sets, len(items), part, P.DTYPE))

    @torch.no_grad()
    def forward(ids):
        h, ctx = ad.prepare(ids); S = []; logits = []
        for t in range(4):
            for li, layer in enumerate(b_.layers[:L]):
                h = layer(h, attention_mask=ctx['mask'], position_ids=ctx['pos'], past_key_value=None, use_cache=False,
                          cache_position=ctx['cp'], position_embeddings=ctx['pe'], current_ut=t)
                h = h[0] if isinstance(h, tuple) else h
                if li == mid:
                    S.append(h)
            h = b_.norm(h); S.append(h); logits.append(ad.readout(h, ctx))
        return S, logits

    p = head + 'Question: %s\nAnswer:' % items[0]['question']
    ids = torch.tensor(tok(p + ' %d' % items[0]['answer']).input_ids, device=P.DEV).unsqueeze(0)
    S, lg = forward(ids)
    with torch.no_grad():
        o = b_(input_ids=ids, use_cache=False)
    dl = float((lg[-1][0].float() - model.lm_head(o[1][-1])[0].float()).abs().max())
    log('check vs Ouro forward: max |logit diff| %.2e' % dl)
    if not dl < 1e-3:
        raise SystemExit('v188 CHECK FAILED')

    d = b_.config.hidden_size; N = len(items)
    X = np.zeros((N, 2, len(STATES), d), np.float16); meta = []
    for k, it in enumerate(items):
        p = head + 'Question: %s\nAnswer:' % it['question']
        p_ids = tok(p).input_ids; f_ids = tok(p + ' %d' % it['answer']).input_ids; n0 = len(p_ids)
        ok = f_ids[:n0] == p_ids and tok.decode([f_ids[-1]]) == str(it['answer'] % 10)
        pos = [n0 - 1, len(f_ids) - 2]
        ids = torch.tensor(f_ids, device=P.DEV).unsqueeze(0)
        S, lg = forward(ids)
        X[k] = torch.stack([torch.stack([s[0, j] for s in S]) for j in pos]).float().cpu().numpy()
        tgt = ids[0, n0:]; corr = []
        for t in range(4):
            am = lg[t][0, n0 - 1:].float().argmax(-1)
            nxt = tok.decode([int(am[-1])]) if len(am) > len(tgt) else ''
            corr.append(bool((am[:len(tgt)] == tgt).all()) and not (nxt[:1].isdigit() or nxt[:1] in '.,'))
        meta.append(dict(k=k, set=it['set'], fam=it['fam'], tid=it['tid'], labels=it['labels'], pos=pos, tok_ok=bool(ok),
                         correct=corr))
        if k % 250 == 0 or k == N - 1:
            log('%5d/%d acc so far %s (%.0fs)' % (k + 1, N, np.round(np.mean([m['correct'] for m in meta], 0), 3).tolist(),
                                                  time.time() - t0))
    np.save(os.path.join(OUT, 'v188_states.npy'), X)
    json.dump(dict(model=P.REPO, n=N, d=d, mid_layer=mid + 1, check_logit=dl, positions=['P0_last_prompt', 'P1_before_units'],
                   states=STATES, items=meta, wall_s=round(time.time() - t0)), open(os.path.join(OUT, 'v188_meta.json'), 'w'))
    log('done %ds; tokenisation ok %d/%d' % (time.time() - t0, sum(m['tok_ok'] for m in meta), N))


if __name__ == '__main__':
    main()
