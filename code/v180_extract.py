"""v180 positive control for v179 (exploratory, no prereg): does the same probe read operand digits, and the computed
result digit, at the LAST token when the prompt is bare arithmetic instead of a word problem?

v179: in GSM-Symbolic word problems a copied question digit is linearly readable only on its own token (never at '?' or
the last prompt token, any loop). Bare-arithmetic interpretability work finds operands moved to the last token. If the
same probe reads them here, v179's blindness is about word problems; if not, it is about the probe.

Items (seed 180): 1000 'add' (a, b uniform in [10, 999], 'What is a plus b?') and 1000 'mul' (a in [2, 9], b in [10, 99],
'What is a times b?'); same 4-shot GSM8K head and answer-only format as v176-v179; one forward pass over prompt + ' r'.
Slots: a+0, b+0 (each operand's units-digit token), qmark ('?'), last (last prompt token).
States: [h0, mid1, end1, end2, end3, end4]. Labels: units digit of a, b and of the result r.
Also answer-only correctness per loop (v176b scorer). Check vs Ouro's own forward on item 0.
Outputs (OUT_DIR): v180_states.npy (float16, N x 4 x 6 x d), v180_meta.json.
Env: TINY=1 (tiny random Ouro, 12 items), V180_PART=0/1 (half of the items), OUT_DIR.
"""
import json, os, time
os.environ.setdefault('LOOP_MODEL', 'ouro')
import numpy as np
import torch

import loop_grad_probe as P

MID = 12
OUT = P.OUT
SLOTS = ['a+0', 'b+0', 'qmark', 'last']
STATES = ['h0', 'mid1', 'end1', 'end2', 'end3', 'end4']


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v180_log.txt'), 'a').write(s + '\n')


def gold(ans):
    return ans.split('####')[-1].strip().replace(',', '')


def make_items(n=1000, seed=180):
    rng = np.random.default_rng(seed); out = []
    for k in range(n):
        a, b = (int(x) for x in rng.integers(10, 1000, 2))
        out.append(dict(op='add', a=a, b=b, r=a + b, question='What is %d plus %d?' % (a, b)))
    for k in range(n):
        a, b = int(rng.integers(2, 10)), int(rng.integers(10, 100))
        out.append(dict(op='mul', a=a, b=b, r=a * b, question='What is %d times %d?' % (a, b)))
    return out


def main():
    t0 = time.time()
    items = make_items()
    if P.TINY:
        items = items[:6] + items[-6:]
    part = os.environ.get('V180_PART')
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
    log('v180 extract | %s | %d items (part %s) | layers/loop %d, mid layer %d | dtype %s' % (
        P.REPO, len(items), part, L, mid + 1, P.DTYPE))

    @torch.no_grad()
    def forward(ids):
        h, ctx = ad.prepare(ids); S = [h]; logits = []
        for t in range(4):
            for li, layer in enumerate(b_.layers[:L]):
                h = layer(h, attention_mask=ctx['mask'], position_ids=ctx['pos'], past_key_value=None, use_cache=False,
                          cache_position=ctx['cp'], position_embeddings=ctx['pe'], current_ut=t)
                h = h[0] if isinstance(h, tuple) else h
                if li == mid and t == 0:
                    S.append(h)
            h = b_.norm(h); S.append(h); logits.append(ad.readout(h, ctx))
        return S, logits

    p = head + 'Question: %s\nAnswer:' % items[0]['question']
    ids = torch.tensor(tok(p + ' %d' % items[0]['r']).input_ids, device=P.DEV).unsqueeze(0)
    S, lg = forward(ids)
    with torch.no_grad():
        o = b_(input_ids=ids, use_cache=False)
    dl = float((lg[-1][0].float() - model.lm_head(o[1][-1])[0].float()).abs().max())
    log('check vs Ouro forward: max |logit diff| %.2e' % dl)
    if not dl < 1e-3:
        raise SystemExit('v180 CHECK FAILED')

    d = b_.config.hidden_size; N = len(items)
    X = np.zeros((N, len(SLOTS), len(STATES), d), np.float16); meta = []
    for k, it in enumerate(items):
        q = it['question']; c0 = len(head) + len('Question: ')
        p = head + 'Question: %s\nAnswer:' % q
        e = tok(p, return_offsets_mapping=True); p_ids = e.input_ids; off = e.offset_mapping
        f_ids = tok(p + ' %d' % it['r']).input_ids; n0 = len(p_ids)
        q_end = len(tok(head + 'Question: %s' % q).input_ids) - 1

        def tok_at(c):
            c += c0
            return next(j for j, (s_, e_) in enumerate(off) if s_ <= c < e_)
        ca = q.index(str(it['a'])) + len(str(it['a'])) - 1
        cb = q.index(str(it['b']), ca + 1) + len(str(it['b'])) - 1
        pos = [tok_at(ca), tok_at(cb), q_end, n0 - 1]
        ok = (tok.decode([p_ids[pos[0]]]).strip() == q[ca] and tok.decode([p_ids[pos[1]]]).strip() == q[cb]
              and f_ids[:n0] == p_ids and tok.decode([p_ids[q_end]]).strip().endswith('?'))
        ids = torch.tensor(f_ids, device=P.DEV).unsqueeze(0)
        S, lg = forward(ids)
        X[k] = torch.stack([torch.stack([s[0, j] for s in S]) for j in pos]).float().cpu().numpy()
        tgt = ids[0, n0:]; corr = []
        for t in range(4):
            am = lg[t][0, n0 - 1:].float().argmax(-1)
            nxt = tok.decode([int(am[-1])]) if len(am) > len(tgt) else ''
            corr.append(bool((am[:len(tgt)] == tgt).all()) and not (nxt[:1].isdigit() or nxt[:1] in '.,'))
        meta.append(dict(k=k, op=it['op'], a=it['a'], b=it['b'], r=it['r'],
                         labels=dict(a=it['a'] % 10, b=it['b'] % 10, r=it['r'] % 10), pos=pos, tok_ok=bool(ok),
                         correct=corr))
        if k % 100 == 0 or k == N - 1:
            log('%4d/%d acc so far %s (%.0fs)' % (k + 1, N, np.round(np.mean([m['correct'] for m in meta], 0), 3).tolist(),
                                                 time.time() - t0))
    np.save(os.path.join(OUT, 'v180_states.npy'), X)
    json.dump(dict(model=P.REPO, n=N, d=d, mid_layer=mid + 1, check_logit=dl, slots=SLOTS, states=STATES, items=meta,
                   wall_s=round(time.time() - t0)), open(os.path.join(OUT, 'v180_meta.json'), 'w'))
    log('done %ds; tokenisation ok %d/%d' % (time.time() - t0, sum(m['tok_ok'] for m in meta), N))


if __name__ == '__main__':
    main()
