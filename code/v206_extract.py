"""v206: the v205 B2/B3 design on Ouro-1.4B (docs/prereg_v206_ouro_template_curve.md), Kaggle CPU.

Same items as v205 (v205_extract.gsm_items: every usable GSM8K training question, one random units digit each, a fixed
random 1,000 test questions; B3 = minimal pairs on the first test questions). Ouro forward as in v181 (loop_grad_probe:
24 layers x 4 loops, the loop-end state normed and fed back), with the 4-shot prefix run once and its key-value cache
(UniversalTransformerCache, indexed loop x 24 + layer) reused through a fresh copy per prompt. Checked against a full
uncached forward on the first prompt of each part (logits and every stored state; abort if max |diff| >= 1e-3).
States: h0, mid1 (after layer 12 of loop 1), end1-end4 (normed loop ends), at slots +0, qmark, last.
Outputs (OUT_DIR): v206_<part>_states.npy (float16, N x 3 x 6 x d), v206_<part>_meta.json.
Env: V206_PART (b2 | b3), V206_SHARD / V206_NSHARD (b2 shards; items k with k % NSHARD == SHARD), V206_N_TRAIN (cap on
the training pool, taken in permutation order; default all), V206_N_B3 (default 100), V206_PILOT=1 (8 prompts), OUT_DIR.
"""
import copy, json, os, time
os.environ.setdefault('LOOP_MODEL', 'ouro')
import numpy as np

MID = 12
STATES = ['h0', 'mid1', 'end1', 'end2', 'end3', 'end4']
OUT = os.environ.get('OUT_DIR', '.')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v206_log.txt'), 'a').write(s + '\n')


def main():
    import torch
    import loop_grad_probe as P
    import v202_extract as E2
    import v205_extract as E5
    from datasets import load_dataset
    t0 = time.time(); part = os.environ['V206_PART']; pilot = os.environ.get('V206_PILOT') == '1'
    torch.set_grad_enabled(False)
    tok, model = P.load(); ad = P.Ouro(model); b = ad.b; L = b.config.num_hidden_layers; mid = min(MID, L) - 1
    mod = ad.mod; dev = P.DEV
    tr = load_dataset('openai/gsm8k', 'main', split='train')
    shots = [(tr[i]['question'], E2.gold(tr[i]['answer'])) for i in range(4)]
    head = ''.join('Question: %s\nAnswer: %s\n\n' % s for s in shots)
    c0 = len(head) + len('Question: ')
    # Ouro's tokenizer merges across the head/question boundary, so cache only the longest token prefix shared with full
    # prompts, minus 2 tokens of margin; every prompt is checked against it (and run in full if it differs).
    h_full = tok(head).input_ids; ex = tok(head + 'Question: x\nAnswer:').input_ids
    lcp = next((i for i, (u, v) in enumerate(zip(h_full, ex)) if u != v), min(len(h_full), len(ex)))
    head_ids = h_full[:max(1, lcp - 2)]; n_head = len(head_ids)
    d = b.config.hidden_size

    def forward(ids, cache=None, start=0):
        """Ouro's 4 loops over ids (absolute positions start..), with an optional cache holding positions < start."""
        x = torch.tensor([ids], device=dev); h = b.embed_tokens(x)
        cp = torch.arange(start, start + len(ids), device=dev); pos = cp.unsqueeze(0)
        n_tot = start + len(ids)                         # the full-length causal mask, sliced to the new rows
        full = mod.create_causal_mask(config=b.config, input_embeds=h.new_zeros(1, n_tot, h.shape[-1]), attention_mask=None,
                                      cache_position=torch.arange(n_tot, device=dev), past_key_values=None,
                                      position_ids=torch.arange(n_tot, device=dev).unsqueeze(0))
        if full is None:
            full = torch.ones(n_tot, n_tot, dtype=torch.bool, device=dev).tril()[None, None]
        mask = full[:, :, start:, :]
        pe = b.rotary_emb(h, pos); S = [h]
        for t in range(4):
            for li, layer in enumerate(b.layers[:L]):
                h = layer(h, attention_mask=mask, position_ids=pos, past_key_value=cache, use_cache=cache is not None,
                          cache_position=cp, position_embeddings=pe, current_ut=t)
                h = h[0] if isinstance(h, tuple) else h
                if li == mid and t == 0:
                    S.append(h)
            h = b.norm(h); S.append(h)
        return S, model.lm_head(h)[0, -1]

    class KVCache:
        """Ouro's attention only calls update(k, v, loop * 24 + layer, kwargs); its own UniversalTransformerCache does
        not construct under transformers 4.54.1, so this minimal per-index cache stands in for it."""
        def __init__(self):
            self.k, self.v = {}, {}

        def update(self, k, v, idx, kw=None):
            if idx in self.k:
                self.k[idx] = torch.cat([self.k[idx], k], 2); self.v[idx] = torch.cat([self.v[idx], v], 2)
            else:
                self.k[idx], self.v[idx] = k, v
            return self.k[idx], self.v[idx]

    cache0 = KVCache()
    forward(head_ids, cache0, 0)

    def run(ids):
        if ids[:n_head] == head_ids:
            S, lg = forward(ids[n_head:], copy.deepcopy(cache0), n_head); return S, lg, n_head, True
        S, lg = forward(ids); return S, lg, 0, False

    allit = E5.gsm_items(tr)
    if part == 'b2':
        cap = os.environ.get('V206_N_TRAIN')
        if cap is not None:                                  # keep the test set; cap the training pool in permutation order
            allit = [it for it in allit if it['split'] == 'test' or it['rank'] < E5.N_TEST + int(cap)]
        sh, nsh = int(os.environ.get('V206_SHARD', '0')), int(os.environ.get('V206_NSHARD', '1'))
        its = [it for k, it in enumerate(allit) if k % nsh == sh]
    else:
        nb3 = int(os.environ.get('V206_N_B3', '100'))
        te = sorted([it for it in allit if it['split'] == 'test'], key=lambda it: it['rank'])[:nb3]
        its = [dict(idx=it['idx'], question=it['orig'][:it['cu']] + str(dg) + it['orig'][it['cu'] + 1:], cu=it['cu'],
                    digit=dg, split='test', rank=it['rank']) for it in te for dg in range(10)]
    its = its[:8] if pilot else its
    log('v206 extract | %s | part %s | pilot %s | %d prompts | cached prefix %d of %d head tokens | device %s | dtype %s' % (
        P.REPO, part, pilot, len(its), n_head, len(h_full), dev, P.DTYPE))
    X = np.zeros((len(its), 3, len(STATES), d), np.float16); meta = []; chk = None; t1 = time.time()
    for k, it in enumerate(its):
        p = head + 'Question: %s\nAnswer:' % it['question']
        e = tok(p, return_offsets_mapping=True); ids = e.input_ids
        j = next(i for i, (s_, e_) in enumerate(e.offset_mapping) if s_ <= it['cu'] + c0 < e_)
        q_end = len(tok(head + 'Question: %s' % it['question']).input_ids) - 1; pos = [j, q_end, len(ids) - 1]
        ok = tok.decode([ids[j]]).strip() == str(it['digit'])
        S, lg, off, cached = run(ids)
        if chk is None:
            S2, lg2 = forward(ids)
            dl = float((lg - lg2).abs().max())
            dh = max(float((S[s][0, pp - off] - S2[s][0, pp]).abs().max()) for s in range(len(STATES)) for pp in pos)
            chk = max(dl, dh); log('check cached vs full forward: max |logit diff| %.2e, max |state diff| %.2e' % (dl, dh))
            if not chk < 1e-3:
                raise SystemExit('v206 CHECK FAILED')
        X[k] = torch.stack([torch.stack([S[s][0, pp - off] for s in range(len(STATES))]) for pp in pos]).float().cpu().numpy()
        meta.append(dict(k=k, idx=it['idx'], fam=it['idx'], tid=it['idx'], digit=it['digit'], pos=pos, ok=bool(ok),
                         cached=cached, split=it['split'], rank=it['rank']))
        if k % 50 == 0 or k == len(its) - 1:
            log('%s %4d/%d prompts (%.0fs, %.2f s/prompt)' % (part, k + 1, len(its), time.time() - t0, (time.time() - t1) / (k + 1)))
    name = 'b2_s%d' % sh if part == 'b2' else 'b3'
    np.save(os.path.join(OUT, 'v206_%s_states.npy' % name), X)
    json.dump(dict(model=P.REPO, part=part, states=STATES, slots=['+0', 'qmark', 'last'], check=chk, n_head=n_head,
                   items=meta, wall_s=round(time.time() - t0)), open(os.path.join(OUT, 'v206_%s_meta.json' % name), 'w'))
    log('%s done: %d prompts, ok %d, cached %d (%.0fs)' % (name, len(meta), sum(m['ok'] for m in meta),
                                                         sum(m['cached'] for m in meta), time.time() - t0))


if __name__ == '__main__':
    main()
