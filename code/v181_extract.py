"""v181 minimal-pair readout (exploratory, no prereg): is a copied question digit present at later positions at all?

v179: a standard linear probe reads a copied digit only on its own token. That cannot separate "absent" from "present
but swamped by context". Here each family = one GSM-Symbolic question with ONE units digit set to 0..9 (10 variants,
token-for-token identical otherwise; Ouro tokenises digits one per token). Any change of the state at a later position is
caused by that digit only.

Families (seed 181): one instance per v178 template (84); anchors = the first and the last integer question number with
>= 2 digits (one family each; one if they coincide) -> up to 168 families x 10 variants.
Slots: +0 (the varied digit), +1, +3 (clipped to the question), sentence end ('.?!'), qmark, last (prompt's last token).
States: [h0, mid1, end1, end2, end3, end4]. Same 4-shot answer-only prompt as v176-v180, frozen Ouro-1.4B fp32.
Outputs (OUT_DIR): v181_states.npy (float16, N x 6 x 6 x d), v181_meta.json.
Env: TINY=1 (tiny random Ouro, 2 families), V181_PART=0/1 (half of the families), OUT_DIR.
"""
import json, os, re, time
os.environ.setdefault('LOOP_MODEL', 'ouro')
import numpy as np
import torch

import loop_grad_probe as P

MID = 12
NUM = r'\d+(?:,\d{3})*(?:\.\d+)?'
OUT = P.OUT
SLOTS = ['+0', '+1', '+3', 'sent', 'qmark', 'last']
STATES = ['h0', 'mid1', 'end1', 'end2', 'end3', 'end4']


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v181_log.txt'), 'a').write(s + '\n')


def gold(ans):
    return ans.split('####')[-1].strip().replace(',', '')


def families(items, seed=181):
    rng = np.random.default_rng(seed); out = []
    for t in sorted({i['tid'] for i in items}):
        its = [i for i in items if i['tid'] == t]; it = its[int(rng.integers(len(its)))]; q = it['question']
        ms = [m for m in re.finditer(NUM, q) if '.' not in m.group() and len(m.group().replace(',', '')) >= 2]
        if not ms:
            continue
        for nm, m in (('first', ms[0]), ('last', ms[-1])):
            if nm == 'last' and m.start() == ms[0].start():
                continue
            cu = m.end() - 1; se = len(q) - 1
            for c in range(m.end(), len(q)):
                if q[c] in '.?!' and (c + 1 == len(q) or q[c + 1] in ' \n'):
                    se = c; break
            out.append(dict(tid=t, inst=it['inst'], anchor=nm, question=q, cu=cu, se=se, orig_digit=int(q[cu])))
    return out


def main():
    t0 = time.time()
    fams = families(json.load(open('v178_labels.json'))['items'])
    if P.TINY:
        fams = fams[:2]
    part = os.environ.get('V181_PART')
    if part is not None:
        h = (len(fams) + 1) // 2; fams = fams[:h] if part == '0' else fams[h:]
    tok, model = P.load(); ad = P.Ouro(model); b_ = ad.b
    L = b_.config.num_hidden_layers; mid = min(MID, L) - 1
    if P.TINY:
        shots = [('What is %d plus %d?' % (i, i + 1), str(2 * i + 1)) for i in range(4)]
    else:
        from datasets import load_dataset
        tr = load_dataset('openai/gsm8k', 'main', split='train')
        shots = [(tr[i]['question'], gold(tr[i]['answer'])) for i in range(4)]
    head = ''.join('Question: %s\nAnswer: %s\n\n' % s for s in shots)
    log('v181 extract | %s | %d families x 10 (part %s) | layers/loop %d, mid layer %d | dtype %s' % (
        P.REPO, len(fams), part, L, mid + 1, P.DTYPE))

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

    p = head + 'Question: %s\nAnswer:' % fams[0]['question']
    ids = torch.tensor(tok(p).input_ids, device=P.DEV).unsqueeze(0)
    S, lg = forward(ids)
    with torch.no_grad():
        o = b_(input_ids=ids, use_cache=False)
    dl = float((lg[-1][0].float() - model.lm_head(o[1][-1])[0].float()).abs().max())
    log('check vs Ouro forward: max |logit diff| %.2e' % dl)
    if not dl < 1e-3:
        raise SystemExit('v181 CHECK FAILED')

    d = b_.config.hidden_size; N = 10 * len(fams); c0 = len(head) + len('Question: ')
    X = np.zeros((N, len(SLOTS), len(STATES), d), np.float16); meta = []; k = 0
    for fi, f in enumerate(fams):
        ref = None
        for dg in range(10):
            q = f['question'][:f['cu']] + str(dg) + f['question'][f['cu'] + 1:]
            p = head + 'Question: %s\nAnswer:' % q
            e = tok(p, return_offsets_mapping=True); ids_l = e.input_ids; off = e.offset_mapping
            q_end = len(tok(head + 'Question: %s' % q).input_ids) - 1

            def tok_at(c):
                c += c0
                return next(j for j, (s_, e_) in enumerate(off) if s_ <= c < e_)
            j = tok_at(f['cu'])
            pos = [j, min(j + 1, q_end), min(j + 3, q_end), tok_at(f['se']), q_end, len(ids_l) - 1]
            if ref is None:
                ref = (ids_l, pos)
            diff = [i for i, (x, y) in enumerate(zip(ids_l, ref[0])) if x != y]
            ok = len(ids_l) == len(ref[0]) and pos == ref[1] and set(diff) <= {j} and tok.decode([ids_l[j]]).strip() == str(dg)
            S, _ = forward(torch.tensor(ids_l, device=P.DEV).unsqueeze(0))
            X[k] = torch.stack([torch.stack([s[0, jj] for s in S]) for jj in pos]).float().cpu().numpy()
            meta.append(dict(k=k, fam=fi, tid=f['tid'], inst=f['inst'], anchor=f['anchor'], digit=dg, pos=pos, pair_ok=bool(ok)))
            k += 1
        if fi % 10 == 0 or fi == len(fams) - 1:
            log('%3d/%d families (%.0fs)' % (fi + 1, len(fams), time.time() - t0))
    np.save(os.path.join(OUT, 'v181_states.npy'), X)
    json.dump(dict(model=P.REPO, n=N, d=d, mid_layer=mid + 1, check_logit=dl, slots=SLOTS, states=STATES, items=meta,
                   wall_s=round(time.time() - t0)), open(os.path.join(OUT, 'v181_meta.json'), 'w'))
    log('done %ds; minimal pairs ok %d/%d' % (time.time() - t0, sum(m['pair_ok'] for m in meta), N))


if __name__ == '__main__':
    main()
