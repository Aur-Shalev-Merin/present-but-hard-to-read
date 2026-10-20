"""v179 control-only pilot (exploratory, no prereg): where in the prompt is a COPIED question digit linearly readable?

v178 was UNSCORABLE: at the last prompt token (and at '?') a linear probe could not read the units digit even of a number
copied from the question. Before any follow-up is designed, this pilot maps readability by position and loop.
It touches CONTROL labels only (digits that are in the question); no intermediate-value label is read or probed.

Items: 10 seeded instances per template from v178_labels.json (84 templates -> 840 items); same 4-shot answer-only prompt
and one forward pass over the prompt (frozen Ouro-1.4B, fp32). Ouro tokenises digits one per token.
Anchors (integers only): the first, the last and the largest question number. Per anchor, 4 slots:
  +0 = its units-digit token, +1, +3 tokens after it (clipped to the question), and the end of its sentence ('.?!').
Plus 2 global slots: '?' = question's final token, 'last' = last prompt token (after 'Answer:').
States per slot: [h0, mid1, end1, end2, end3, end4] (mid1 = layer 12 of loop 1; endT = normed state after loop T).
Outputs (OUT_DIR): v179_states.npy (float16, N x 14 x 6 x d), v179_meta.json (per item: tid, inst, labels, slot positions).
Env: TINY=1 (tiny random Ouro + 12 items), V179_PART=0/1 (which half of the selected items; unset = all), OUT_DIR.
"""
import json, os, re, time
os.environ.setdefault('LOOP_MODEL', 'ouro')
import numpy as np
import torch

import loop_grad_probe as P

MID = 12
NUM = r'\d+(?:,\d{3})*(?:\.\d+)?'
OUT = P.OUT
SLOTS = ['first+0', 'first+1', 'first+3', 'first_sent', 'last+0', 'last+1', 'last+3', 'last_sent',
         'big+0', 'big+1', 'big+3', 'big_sent', 'qmark', 'last']
STATES = ['h0', 'mid1', 'end1', 'end2', 'end3', 'end4']


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v179_log.txt'), 'a').write(s + '\n')


def gold(ans):
    return ans.split('####')[-1].strip().replace(',', '')


def select(items, per=10, seed=179):
    rng = np.random.default_rng(seed); out = []
    for t in sorted({i['tid'] for i in items}):
        its = [i for i in items if i['tid'] == t]
        out += [its[k] for k in sorted(rng.choice(len(its), per, replace=False))]
    return out


def anchors(q):
    """(char index of the units digit, value, char index of its sentence end) for first, last, largest number."""
    ms = list(re.finditer(NUM, q))
    vals = [float(m.group().replace(',', '')) for m in ms]
    big = vals.index(max(vals))
    res = {}
    for nm, k in (('first', 0), ('last', len(ms) - 1), ('big', big)):
        m, v = ms[k], vals[k]
        se = len(q) - 1
        for c in range(m.end(), len(q)):
            if q[c] in '.?!' and (c + 1 == len(q) or q[c + 1] in ' \n'):
                se = c; break
        res[nm] = (m.end() - 1, v, se)
    return res


def main():
    t0 = time.time()
    items = select(json.load(open('v178_labels.json'))['items'])
    if P.TINY:
        items = items[:12]
    part = os.environ.get('V179_PART')
    if part is not None:
        h = (len(items) + 1) // 2; items = items[:h] if part == '0' else items[h:]
    tok, model = P.load(); ad = P.Ouro(model); b = ad.b
    L = b.config.num_hidden_layers; mid = min(MID, L) - 1
    if P.TINY:
        shots = [('What is %d plus %d?' % (i, i + 1), str(2 * i + 1)) for i in range(4)]
    else:
        from datasets import load_dataset
        tr = load_dataset('openai/gsm8k', 'main', split='train')
        shots = [(tr[i]['question'], gold(tr[i]['answer'])) for i in range(4)]
    head = ''.join('Question: %s\nAnswer: %s\n\n' % s for s in shots)
    log('v179 extract | %s | %d items (part %s) | layers/loop %d, mid layer %d | dtype %s' % (
        P.REPO, len(items), part, L, mid + 1, P.DTYPE))

    @torch.no_grad()
    def forward(ids):
        h, ctx = ad.prepare(ids); S = [h]; logits = []
        for t in range(4):
            for li, layer in enumerate(b.layers[:L]):
                h = layer(h, attention_mask=ctx['mask'], position_ids=ctx['pos'], past_key_value=None, use_cache=False,
                          cache_position=ctx['cp'], position_embeddings=ctx['pe'], current_ut=t)
                h = h[0] if isinstance(h, tuple) else h
                if li == mid and t == 0:
                    S.append(h)
            h = b.norm(h); S.append(h); logits.append(ad.readout(h, ctx))
        return S, logits

    # check vs Ouro's own forward (final-loop logits)
    p = head + 'Question: %s\nAnswer:' % items[0]['question']
    ids = torch.tensor(tok(p).input_ids, device=P.DEV).unsqueeze(0)
    S, lg = forward(ids)
    with torch.no_grad():
        o = b(input_ids=ids, use_cache=False)
    dl = float((lg[-1][0].float() - model.lm_head(o[1][-1])[0].float()).abs().max())
    log('check vs Ouro forward: max |logit diff| %.2e' % dl)
    if not dl < 1e-3:
        raise SystemExit('v179 CHECK FAILED')

    d = b.config.hidden_size; N = len(items)
    X = np.zeros((N, len(SLOTS), len(STATES), d), np.float16); meta = []
    for k, it in enumerate(items):
        q = it['question']; c0 = len(head) + len('Question: ')
        p = head + 'Question: %s\nAnswer:' % q
        e = tok(p, return_offsets_mapping=True); ids_l = e.input_ids; off = e.offset_mapping
        q_end = len(tok(head + 'Question: %s' % q).input_ids) - 1

        def tok_at(c):
            c += c0
            return next(j for j, (s_, e_) in enumerate(off) if s_ <= c < e_)
        an = anchors(q); pos = []; lab = {}; ok = True
        for nm in ('first', 'last', 'big'):
            cd, v, se = an[nm]; j = tok_at(cd)
            ok &= tok.decode([ids_l[j]]).strip() == q[cd]
            pos += [j, min(j + 1, q_end), min(j + 3, q_end), tok_at(se)]
            lab[nm] = int(v) % 10 if v == int(v) else None
        pos += [q_end, len(ids_l) - 1]
        ids = torch.tensor(ids_l, device=P.DEV).unsqueeze(0)
        S, _ = forward(ids)
        X[k] = torch.stack([torch.stack([s[0, j] for s in S]) for j in pos]).float().cpu().numpy()
        meta.append(dict(k=k, tid=it['tid'], inst=it['inst'], labels=lab, pos=pos, digit_tok_ok=bool(ok)))
        if k % 100 == 0 or k == N - 1:
            log('%4d/%d (%.0fs)' % (k + 1, N, time.time() - t0))
    np.save(os.path.join(OUT, 'v179_states.npy'), X)
    json.dump(dict(model=P.REPO, n=N, d=d, mid_layer=mid + 1, check_logit=dl, slots=SLOTS, states=STATES, items=meta,
                   wall_s=round(time.time() - t0)), open(os.path.join(OUT, 'v179_meta.json'), 'w'))
    log('done %ds; digit tokens ok %d/%d' % (time.time() - t0, sum(m['digit_tok_ok'] for m in meta), N))


if __name__ == '__main__':
    main()
