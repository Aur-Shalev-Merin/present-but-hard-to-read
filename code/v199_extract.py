"""v199 state extraction (v194_extract.py at two story positions; docs/prereg_v199_ouro_running_total.md).

Items: v194_items.json (unchanged), sets main or pilot. Same 4-shot answer-only prompt and ONE teacher-forced forward
pass over prompt + ' <answer>' as v194. States at six positions inside the question:
  Sb = the period ending the sentence that gives the second number (b): "Her friend gave her 13 more stickers."
  Sc = the period ending the sentence that gives the third number (c): "Then she used 15 stickers on a card."
  Sb-1 .. Sb-4 = the four tokens before Sb (the sentence's tail after b is Sb-k for k < Sb - bU, 2-5 tokens; positions
  at or before b's units digit are stored but never used)
in order [mid1, end1, mid2, end2, mid3, end3, mid4, end4] (midT = residual after layer 12 of loop T; endT = normed
state after loop T). Also answer-only correctness after each loop (v176b scorer) and the token positions Sb, Sc, bU
(b's units digit) and P1 (the answer token before the units digit), which v199_patch.py uses.
Check (logged, abort on failure): this forward's final logits == Ouro's own forward on item 0.
Outputs (OUT_DIR): v199_states.npy (float16, N x 6 x 8 x d; positions Sb, Sc, Sb-1, Sb-2, Sb-3, Sb-4), v199_meta.json.
Env: V199_SETS (main or pilot; default main), V199_PART=0/1 (half of the selected items), TINY=1, OUT_DIR.
"""
import json, os, time
os.environ.setdefault('LOOP_MODEL', 'ouro')
import numpy as np

MID = 12
STATES = ['mid1', 'end1', 'mid2', 'end2', 'mid3', 'end3', 'mid4', 'end4']
POS = ['Sb', 'Sc', 'Sb-1', 'Sb-2', 'Sb-3', 'Sb-4']


def gold(ans):
    return ans.split('####')[-1].strip().replace(',', '')


def head_text(tiny=False):
    if tiny:
        shots = [('What is %d plus %d?' % (i, i + 1), str(2 * i + 1)) for i in range(4)]
    else:
        from datasets import load_dataset
        tr = load_dataset('openai/gsm8k', 'main', split='train')
        shots = [(tr[i]['question'], gold(tr[i]['answer'])) for i in range(4)]
    return ''.join('Question: %s\nAnswer: %s\n\n' % s for s in shots)


def positions(tok, head, it):
    """Token positions of Sb, Sc, bU and P1 in prompt + ' <answer>', found by tokenising prefixes (asserted)."""
    p = head + 'Question: %s\nAnswer:' % it['question']
    f_ids = tok(p + ' %d' % it['answer']).input_ids
    q0 = len(head) + len('Question: '); q = it['question']
    dots = [q0 + j for j, ch in enumerate(q) if ch == '.']
    assert len(dots) == 3, q
    bs = q0 + q.index(str(it['b']), dots[0] - q0)
    assert dots[0] < bs < dots[1], q

    def last_tok(ci):
        ids = tok(p[:ci + 1]).input_ids
        assert ids == f_ids[:len(ids)], (q, ci)
        return len(ids) - 1
    pos = dict(Sb=last_tok(dots[1]), Sc=last_tok(dots[2]), bU=last_tok(bs + len(str(it['b'])) - 1), P1=len(f_ids) - 2)
    pos['tail'] = pos['Sb'] - pos['bU']          # tokens after b's units digit up to and including the period
    assert tok.decode([f_ids[pos['Sb']]]) == '.' and tok.decode([f_ids[pos['Sc']]]) == '.', q
    assert tok.decode([f_ids[pos['bU']]]) == str(it['b'] % 10), q
    assert tok.decode([f_ids[-1]]) == str(it['answer'] % 10), q
    return pos, f_ids, len(tok(p).input_ids)


def main():
    import torch
    import loop_grad_probe as P
    OUT = P.OUT

    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True)
        open(os.path.join(OUT, 'v199_log.txt'), 'a').write(s + '\n')

    t0 = time.time()
    sets = os.environ.get('V199_SETS', 'main').split(',')
    items = [i for i in json.load(open('v194_items.json'))['items'] if i['set'] in sets]
    if P.TINY:
        items = items[:6] + items[-6:]
    part = os.environ.get('V199_PART')
    if part is not None:
        h = (len(items) + 1) // 2; items = items[:h] if part == '0' else items[h:]
    tok, model = P.load(); ad = P.Ouro(model); b_ = ad.b
    L = b_.config.num_hidden_layers; mid = min(MID, L) - 1
    head = head_text(P.TINY)
    log('v199 extract | %s | sets %s | %d items (part %s) | dtype %s' % (P.REPO, sets, len(items), part, P.DTYPE))

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

    _, f0, _ = positions(tok, head, items[0])
    ids = torch.tensor(f0, device=P.DEV).unsqueeze(0)
    S, lg = forward(ids)
    with torch.no_grad():
        o = b_(input_ids=ids, use_cache=False)
    dl = float((lg[-1][0].float() - model.lm_head(o[1][-1])[0].float()).abs().max())
    log('check vs Ouro forward: max |logit diff| %.2e' % dl)
    if not dl < 1e-3:
        raise SystemExit('v199 CHECK FAILED')

    d = b_.config.hidden_size; N = len(items)
    X = np.zeros((N, len(POS), len(STATES), d), np.float16); meta = []
    for k, it in enumerate(items):
        pos, f_ids, n0 = positions(tok, head, it)
        ids = torch.tensor(f_ids, device=P.DEV).unsqueeze(0)
        S, lg = forward(ids)
        at = [pos['Sb'], pos['Sc']] + [pos['Sb'] - j for j in range(1, 5)]
        X[k] = torch.stack([torch.stack([s[0, j] for s in S]) for j in at]).float().cpu().numpy()
        tgt = ids[0, n0:]; corr = []
        for t in range(4):
            am = lg[t][0, n0 - 1:].float().argmax(-1)
            nxt = tok.decode([int(am[-1])]) if len(am) > len(tgt) else ''
            corr.append(bool((am[:len(tgt)] == tgt).all()) and not (nxt[:1].isdigit() or nxt[:1] in '.,'))
        meta.append(dict(k=k, set=it['set'], fam=it['fam'], pair=it['pair'], tid=it['tid'], op=it['op'], a=it['a'], b=it['b'],
                         c=it['c'], answer=it['answer'], labels=it['labels'], pos=pos, correct=corr))
        if k % 250 == 0 or k == N - 1:
            log('%5d/%d acc so far %s (%.0fs)' % (k + 1, N, np.round(np.mean([m['correct'] for m in meta], 0), 3).tolist(),
                                                  time.time() - t0))
    np.save(os.path.join(OUT, 'v199_states.npy'), X)
    json.dump(dict(model=P.REPO, n=N, d=d, mid_layer=mid + 1, check_logit=dl, positions=POS, states=STATES,
                   items=meta, wall_s=round(time.time() - t0)), open(os.path.join(OUT, 'v199_meta.json'), 'w'))
    log('done %ds' % (time.time() - t0))


if __name__ == '__main__':
    main()
