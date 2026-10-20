"""v184 irrelevant-number control for v181 (exploratory, no prereg).

v181: varying one units digit of a question number moves Ouro's last-prompt-token state in a digit-identity-like code
that a family-demeaned probe reads at 0.92-0.97. But the digit also changes the correct answer, so the trace could be
the copied number OR the answer in progress. Here the number cannot matter: an added sentence
  IRR = "Unrelated fact: a nearby park has 4D benches."
is inserted either at the start of the question ('start') or just before its final sentence ('end'). On identical
contexts we build two family types per base question:
  irr = vary D (0..9) with the relevant numbers fixed;
  rel = keep D = 3, vary the units digit of a relevant number (v181's anchor: last multi-digit integer, else first).
Base questions = v181's (one instance per template, seed 181). 77 templates x 2 positions x 2 types x 10 = 3080 forwards.
Slots: +0 (the varied digit), qmark, last. States [h0, mid1, end1, end2, end3, end4].
Also per loop: Ouro's top next token at the last prompt token (its first answer token) and its probability, so we can
see whether the "irrelevant" digit changes the answer it is starting to give.
Outputs (OUT_DIR): v184_states.npy (float16, N x 3 x 6 x d), v184_meta.json.
Env: TINY=1, V184_PART=0/1, OUT_DIR.
"""
import json, os, time
os.environ.setdefault('LOOP_MODEL', 'ouro')
import numpy as np
import torch

import loop_grad_probe as P
import v181_extract as V181

MID = 12
OUT = P.OUT
SLOTS = ['+0', 'qmark', 'last']
STATES = ['h0', 'mid1', 'end1', 'end2', 'end3', 'end4']
IRR = 'Unrelated fact: a nearby park has 4%d benches. '


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v184_log.txt'), 'a').write(s + '\n')


def bases(items):
    fams = V181.families(items); out = {}
    for f in fams:                                    # prefer the 'last' anchor per template
        if f['tid'] not in out or f['anchor'] == 'last':
            out[f['tid']] = f
    return [out[t] for t in sorted(out)]


def variants(f, where, kind):
    """10 (question, char index of the varied digit) pairs."""
    q = f['question']
    ins = 0 if where == 'start' else q.rfind('. ', 0, len(q) - 1) + 2
    if where == 'end' and ins == 1:                   # no earlier sentence -> insert at start
        ins = 0
    out = []
    for dg in range(10):
        irr_d, rel_d = (dg, 3) if kind == 'irr' else (3, dg)
        s = IRR % irr_d
        qq = q[:f['cu']] + str(rel_d) + q[f['cu'] + 1:]
        qq = qq[:ins] + s + qq[ins:]
        if kind == 'irr':
            c = ins + s.index('benches') - 2
        else:
            c = f['cu'] + (len(s) if f['cu'] >= ins else 0)
        assert qq[c] == str(dg), (qq, c, dg)
        out.append((qq, c))
    return out


def main():
    t0 = time.time()
    bs = bases(json.load(open('v178_labels.json'))['items'])
    if P.TINY:
        bs = bs[:2]
    part = os.environ.get('V184_PART')
    if part is not None:
        h = (len(bs) + 1) // 2; bs = bs[:h] if part == '0' else bs[h:]
    tok, model = P.load(); ad = P.Ouro(model); b_ = ad.b
    L = b_.config.num_hidden_layers; mid = min(MID, L) - 1
    if P.TINY:
        shots = [('What is %d plus %d?' % (i, i + 1), str(2 * i + 1)) for i in range(4)]
    else:
        from datasets import load_dataset
        tr = load_dataset('openai/gsm8k', 'main', split='train')
        shots = [(tr[i]['question'], V181.gold(tr[i]['answer'])) for i in range(4)]
    head = ''.join('Question: %s\nAnswer: %s\n\n' % s for s in shots)
    log('v184 extract | %s | %d base questions (part %s) | dtype %s' % (P.REPO, len(bs), part, P.DTYPE))

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

    q0 = variants(bs[0], 'end', 'irr')[0][0]
    ids = torch.tensor(tok(head + 'Question: %s\nAnswer:' % q0).input_ids, device=P.DEV).unsqueeze(0)
    S, lg = forward(ids)
    with torch.no_grad():
        o = b_(input_ids=ids, use_cache=False)
    dl = float((lg[-1][0].float() - model.lm_head(o[1][-1])[0].float()).abs().max())
    log('check vs Ouro forward: max |logit diff| %.2e' % dl)
    if not dl < 1e-3:
        raise SystemExit('v184 CHECK FAILED')

    d = b_.config.hidden_size; N = 40 * len(bs); c0 = len(head) + len('Question: ')
    X = np.zeros((N, len(SLOTS), len(STATES), d), np.float16); meta = []; k = 0; fam = 0
    for bi, f in enumerate(bs):
        for where in ('start', 'end'):
            for kind in ('irr', 'rel'):
                ref = None
                for dg, (q, c) in enumerate(variants(f, where, kind)):
                    p = head + 'Question: %s\nAnswer:' % q
                    e = tok(p, return_offsets_mapping=True); ids_l = e.input_ids; off = e.offset_mapping
                    q_end = len(tok(head + 'Question: %s' % q).input_ids) - 1
                    j = next(i for i, (s_, e_) in enumerate(off) if s_ <= c + c0 < e_)
                    pos = [j, q_end, len(ids_l) - 1]
                    if ref is None:
                        ref = (ids_l, pos)
                    diff = [i for i, (x, y) in enumerate(zip(ids_l, ref[0])) if x != y]
                    ok = (len(ids_l) == len(ref[0]) and pos == ref[1] and set(diff) <= {j}
                          and tok.decode([ids_l[j]]).strip() == str(dg))
                    S, lg = forward(torch.tensor(ids_l, device=P.DEV).unsqueeze(0))
                    X[k] = torch.stack([torch.stack([s[0, jj] for s in S]) for jj in pos]).float().cpu().numpy()
                    nxt = [lg[t][0, -1].float().softmax(-1) for t in range(4)]
                    meta.append(dict(k=k, fam=fam, tid=f['tid'], where=where, kind=kind, digit=dg, pos=pos, pair_ok=bool(ok),
                                     top=[int(p_.argmax()) for p_ in nxt], top_p=[round(float(p_.max()), 4) for p_ in nxt]))
                    k += 1
                fam += 1
        if bi % 5 == 0 or bi == len(bs) - 1:
            log('%3d/%d base questions (%.0fs)' % (bi + 1, len(bs), time.time() - t0))
    np.save(os.path.join(OUT, 'v184_states.npy'), X)
    json.dump(dict(model=P.REPO, n=N, d=d, check_logit=dl, slots=SLOTS, states=STATES, irr_sentence=IRR, items=meta,
                   wall_s=round(time.time() - t0)), open(os.path.join(OUT, 'v184_meta.json'), 'w'))
    log('done %ds; minimal pairs ok %d/%d' % (time.time() - t0, sum(m['pair_ok'] for m in meta), N))


if __name__ == '__main__':
    main()
