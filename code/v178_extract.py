"""v178 state extraction (frozen Ouro-1.4B, fp32, no training; docs/prereg_v178_ouro_latent_intermediates.md).

For each item of v178_labels.json: the v176/v177 4-shot answer-only prompt; ONE forward pass over prompt + ' <final>'.
States at two positions:
  'last' = the last prompt token (after 'Answer:'), 'q' = the question's final token;
in order [h0, mid1, end1, mid2, end2, mid3, end3, mid4, end4]:
  h0 = token embedding (pre-loop); midT = residual after layer 12 of loop T; endT = normed state after loop T (what
  Ouro passes to the next loop and reads out).
Also answer-only correctness after each loop (v176b scorer: number tokens are the argmax, next argmax not a digit/./,).
Check (logged, abort on failure): this forward's final logits and gates == Ouro's own forward on item 0.
Outputs (OUT_DIR): v178_states_last.npy, v178_states_q.npy (float16, N x 9 x d), v178_meta.json.
Env: TINY=1 (tiny random Ouro + first 12 items, local test), V178_LABELS (path), OUT_DIR.
"""
import json, os, time
os.environ.setdefault('LOOP_MODEL', 'ouro')
import numpy as np
import torch

import loop_grad_probe as P

MID = 12                                  # 1-based layer index inside each loop (24 layers in Ouro-1.4B)
OUT = P.OUT


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v178_log.txt'), 'a').write(s + '\n')


def gold(ans):
    return ans.split('####')[-1].strip().replace(',', '')


def main():
    t0 = time.time()
    lab = json.load(open(os.environ.get('V178_LABELS', 'v178_labels.json')))
    items = lab['items'][:12] if P.TINY else lab['items']
    tok, model = P.load(); ad = P.Ouro(model); b = ad.b
    L = b.config.num_hidden_layers; mid = min(MID, L) - 1
    if P.TINY:
        shots = [('What is %d plus %d?' % (i, i + 1), str(2 * i + 1)) for i in range(4)]
    else:
        from datasets import load_dataset
        tr = load_dataset('openai/gsm8k', 'main', split='train')
        shots = [(tr[i]['question'], gold(tr[i]['answer'])) for i in range(4)]
    head = ''.join('Question: %s\nAnswer: %s\n\n' % s for s in shots)
    log('v178 extract | %s | %d items | layers/loop %d, mid layer %d | dtype %s' % (P.REPO, len(items), L, mid + 1, P.DTYPE))

    @torch.no_grad()
    def forward(ids):
        h, ctx = ad.prepare(ids); S = [h]; logits = []; gates = []
        for t in range(4):
            for li, layer in enumerate(b.layers[:L]):
                h = layer(h, attention_mask=ctx['mask'], position_ids=ctx['pos'], past_key_value=None, use_cache=False,
                          cache_position=ctx['cp'], position_embeddings=ctx['pe'], current_ut=t)
                h = h[0] if isinstance(h, tuple) else h
                if li == mid:
                    S.append(h)
            h = b.norm(h); S.append(h); gates.append(b.early_exit_gate(h)); logits.append(ad.readout(h, ctx))
        return S, logits, gates

    # check vs Ouro's own forward
    it = items[0]; p = head + 'Question: %s\nAnswer:' % it['question']
    ids = torch.tensor(tok(p + ' %d' % it['final']).input_ids, device=P.DEV).unsqueeze(0)
    S, lg, gt = forward(ids)
    with torch.no_grad():
        o = b(input_ids=ids, use_cache=False)
    dl = float((lg[-1][0].float() - model.lm_head(o[1][-1])[0].float()).abs().max())
    dg = max(float((a - r).abs().max()) for a, r in zip(gt, o[2]))
    log('check vs Ouro forward: max |logit diff| %.2e, max |gate diff| %.2e' % (dl, dg))
    if not (dl < 1e-3 and dg < 1e-3):
        raise SystemExit('v178 CHECK FAILED')

    d = b.config.hidden_size; N = len(items)
    A = np.zeros((N, 9, d), np.float16); Q = np.zeros((N, 9, d), np.float16); meta = []
    for k, it in enumerate(items):
        p = head + 'Question: %s\nAnswer:' % it['question']
        p_ids = tok(p).input_ids; f_ids = tok(p + ' %d' % it['final']).input_ids
        q_len = len(tok(head + 'Question: %s' % it['question']).input_ids)
        ok_tok = f_ids[:len(p_ids)] == p_ids and tok(head + 'Question: %s' % it['question']).input_ids == p_ids[:q_len]
        ids = torch.tensor(f_ids, device=P.DEV).unsqueeze(0); n0 = len(p_ids)
        S, lg, _ = forward(ids)
        A[k] = torch.stack([s[0, n0 - 1] for s in S]).float().cpu().numpy()
        Q[k] = torch.stack([s[0, q_len - 1] for s in S]).float().cpu().numpy()
        tgt = ids[0, n0:]; corr = []
        for t in range(4):
            am = lg[t][0, n0 - 1:].float().argmax(-1)
            nxt = tok.decode([int(am[-1])]) if len(am) > len(tgt) else ''
            corr.append(bool((am[:len(tgt)] == tgt).all()) and not (nxt[:1].isdigit() or nxt[:1] in '.,'))
        meta.append(dict(k=k, tid=it['tid'], inst=it['inst'], tok_ok=ok_tok, correct=corr))
        if k % 200 == 0 or k == N - 1:
            log('%4d/%d  acc so far %s  (%.0fs)' % (k + 1, N, np.round(np.mean([m['correct'] for m in meta], 0), 3).tolist(),
                                                   time.time() - t0))
    np.save(os.path.join(OUT, 'v178_states_last.npy'), A); np.save(os.path.join(OUT, 'v178_states_q.npy'), Q)
    json.dump(dict(model=P.REPO, n=N, d=d, mid_layer=mid + 1, check=dict(logit=dl, gate=dg),
                   order=['h0', 'mid1', 'end1', 'mid2', 'end2', 'mid3', 'end3', 'mid4', 'end4'], items=meta,
                   wall_s=round(time.time() - t0)), open(os.path.join(OUT, 'v178_meta.json'), 'w'))
    log('done %ds; tokenisation ok %d/%d' % (time.time() - t0, sum(m['tok_ok'] for m in meta), N))


if __name__ == '__main__':
    main()
