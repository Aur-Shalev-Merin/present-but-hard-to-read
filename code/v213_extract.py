"""v213 extraction (docs/prereg_v213_scale_up.md): the note's Section 3 measurements in Qwen2.5-7B and Ouro-2.6B.

Models (V213_MODEL):
  qwen7b  Qwen/Qwen2.5-7B, fp16, split over the visible GPUs (device_map='auto'); 28 layers; stored states after layers
          [0, 4, 7, 14, 21, 28] = h0, l4, l7, l14, l21, l28 (as v202). Prefix cache as v202 (past_key_values, a fresh copy
          per prompt). Cached-vs-full check: max relative difference (stored states and logits) < 1e-2 (fp16).
  ouro26  ByteDance/Ouro-2.6B, fp32, one GPU; 48 layers x 4 loops, the loop-end state normed and fed back; stored states
          h0, mid1 (after layer 24 of loop 1), end1-end4 (as v181/v206 with the mid point at half depth). v206's minimal
          per-index KV cache and longest-shared-prefix rule. Cached-vs-full check: max absolute difference < 1e-3.
Same 4-shot answer-only prompt as v176-v212. Parts (V213_PART):
  v181  137 minimal-pair families from v178_labels.json (v202_extract.families), units digit 0..9; slots +0, +1, +3,
        sent, qmark, last; the last slot also at every depth (qwen7b: 29) or every stored state (ouro26). As v202's v181.
  b1    v205's random-perturbation null on the same families: 10 fixed random word tokens (v205_extract.swap_tokens,
        seed 205) swapped in at the digit's token, and 10 fixed Gaussian vectors (seed 205) of the median input-embedding
        norm added to its input embedding. Slots +0, qmark, last.
  b2    the many-question pool (v213_items: GSM8K 5,249 train + 1,000 test, Orca 40,000 train + 1,000 test), sharded by
        V213_SHARD / V213_NSHARD (items k with k % NSHARD == SHARD). Slots +0, qmark, last.
  b3    minimal pairs on the first 100 GSM8K and the first 100 Orca test questions (by rank), digit 0..9. Slots as b2.
Outputs (OUT_DIR): v213_<part>[_s<k>]_states.npy (float16), v213_v181_last_all.npy, v213_<part>[_s<k>]_meta.json.
Env: V213_MODEL, V213_PART, V213_SHARD, V213_NSHARD, V213_PILOT=1 (a few prompts per part), OUT_DIR.
"""
import copy, json, os, re, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.')
MODEL = os.environ.get('V213_MODEL', 'qwen7b')
NB3 = 100


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v213_log.txt'), 'a').write(s + '\n')


class Backend:
    """run(ids, j=None, vec=None, cached=True) -> (S, depth, logits, off): S[s] is the (T', d) state for stored state s
    over the rows the call computed (positions off..), depth (T', D, d) or None, logits at the last position."""

    def __init__(self, head):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch; torch.set_grad_enabled(False)
        if MODEL == 'qwen7b':
            self.repo = 'Qwen/Qwen2.5-7B'; self.layers = [0, 4, 7, 14, 21, 28]
            self.states = ['h0', 'l4', 'l7', 'l14', 'l21', 'l28']
            self.tok = AutoTokenizer.from_pretrained(self.repo)
            self.m = AutoModelForCausalLM.from_pretrained(self.repo, torch_dtype=torch.float16, device_map='auto').eval()
            self.dev = self.m.get_input_embeddings().weight.device; self.emb = self.m.get_input_embeddings()
            self.head_ids = self.tok(head).input_ids; self.n_head = len(self.head_ids)
            self.prefix = self.m(input_ids=torch.tensor([self.head_ids], device=self.dev), use_cache=True).past_key_values
            self.d = self.m.config.hidden_size; self.rel = True
        else:
            self.repo = 'ByteDance/Ouro-2.6B'; self.states = ['h0', 'mid1', 'end1', 'end2', 'end3', 'end4']
            self.tok = AutoTokenizer.from_pretrained(self.repo, trust_remote_code=True)
            self.m = AutoModelForCausalLM.from_pretrained(self.repo, trust_remote_code=True,
                                                          torch_dtype=torch.float32).to('cuda').eval()
            self.b = self.m.model; self.L = self.b.config.num_hidden_layers; self.mid = self.L // 2 - 1
            self.mod = __import__(type(self.b).__module__, fromlist=['x']); self.dev = 'cuda'; self.emb = self.b.embed_tokens
            h_full = self.tok(head).input_ids; ex = self.tok(head + 'Question: x\nAnswer:').input_ids
            lcp = next((i for i, (u, v) in enumerate(zip(h_full, ex)) if u != v), min(len(h_full), len(ex)))
            self.head_ids = h_full[:max(1, lcp - 2)]; self.n_head = len(self.head_ids)
            self.d = self.b.config.hidden_size; self.rel = False
            self.cache0 = self.KV(); self._ouro(self.head_ids, None, None, self.cache0, 0)

    class KV:
        def __init__(self):
            self.k, self.v = {}, {}

        def update(self, k, v, idx, kw=None):
            import torch
            if idx in self.k:
                self.k[idx] = torch.cat([self.k[idx], k], 2); self.v[idx] = torch.cat([self.v[idx], v], 2)
            else:
                self.k[idx], self.v[idx] = k, v
            return self.k[idx], self.v[idx]

    def _ouro(self, ids, j, vec, cache, start):
        torch, b, dev = self.torch, self.b, self.dev
        h = b.embed_tokens(torch.tensor([ids], device=dev))
        if vec is not None:
            h[0, j - start] = h[0, j - start] + vec
        cp = torch.arange(start, start + len(ids), device=dev); pos = cp.unsqueeze(0); n_tot = start + len(ids)
        full = self.mod.create_causal_mask(config=b.config, input_embeds=h.new_zeros(1, n_tot, h.shape[-1]), attention_mask=None,
                                           cache_position=torch.arange(n_tot, device=dev), past_key_values=None,
                                           position_ids=torch.arange(n_tot, device=dev).unsqueeze(0))
        if full is None:
            full = torch.ones(n_tot, n_tot, dtype=torch.bool, device=dev).tril()[None, None]
        mask = full[:, :, start:, :]; pe = b.rotary_emb(h, pos); S = [h[0]]
        for t in range(4):
            for li, layer in enumerate(b.layers[:self.L]):
                h = layer(h, attention_mask=mask, position_ids=pos, past_key_value=cache, use_cache=cache is not None,
                          cache_position=cp, position_embeddings=pe, current_ut=t)
                h = h[0] if isinstance(h, tuple) else h
                if li == self.mid and t == 0:
                    S.append(h[0])
            h = b.norm(h); S.append(h[0])
        return S, self.m.lm_head(h)[0, -1]

    def run(self, ids, j=None, vec=None, cached=True):
        torch = self.torch
        use = cached and ids[:self.n_head] == self.head_ids
        if MODEL == 'qwen7b':
            x = self.emb(torch.tensor([ids], device=self.dev))
            if vec is not None:
                x[0, j] = x[0, j] + vec.to(x.dtype)
            if use:
                o = self.m(inputs_embeds=x[:, self.n_head:], past_key_values=copy.deepcopy(self.prefix), use_cache=True,
                           output_hidden_states=True); off = self.n_head
            else:
                o = self.m(inputs_embeds=x, use_cache=False, output_hidden_states=True); off = 0
            H = [h[0] for h in o.hidden_states]
            return [H[l] for l in self.layers], H, o.logits[0, -1], off, use
        if use:
            S, lg = self._ouro(ids[self.n_head:], j, vec, copy.deepcopy(self.cache0), self.n_head); return S, S, lg, self.n_head, True
        S, lg = self._ouro(ids, j, vec, None, 0); return S, S, lg, 0, False

    def check(self, ids, pos, j=None, vec=None):
        S, D, lg, off, _ = self.run(ids, j, vec); S2, D2, lg2, _, _ = self.run(ids, j, vec, cached=False)
        f = lambda a: a.float().cpu()
        if self.rel:
            dl = float((f(lg) - f(lg2)).norm() / f(lg2).norm())
            dh = max(float((f(S[s][p - off]) - f(S2[s][p])).norm() / f(S2[s][p]).norm()) for s in range(len(S)) for p in pos)
            lim = 1e-2
        else:
            dl = float((f(lg) - f(lg2)).abs().max())
            dh = max(float((f(S[s][p - off]) - f(S2[s][p])).abs().max()) for s in range(len(S)) for p in pos)
            lim = 1e-3
        log('check cached vs full (%s): logits %.2e, states %.2e (limit %.0e)' % ('relative' if self.rel else 'absolute', dl, dh, lim))
        if not max(dl, dh) < lim:
            raise SystemExit('v213 CHECK FAILED')
        return max(dl, dh)


def main():
    import torch
    import v202_extract as E2
    import v205_extract as E5
    import v213_items as I
    from datasets import load_dataset
    t0 = time.time(); part = os.environ['V213_PART']; pilot = os.environ.get('V213_PILOT') == '1'
    tr = load_dataset('openai/gsm8k', 'main', split='train')
    head = ''.join('Question: %s\nAnswer: %s\n\n' % (tr[i]['question'], E2.gold(tr[i]['answer'])) for i in range(4))
    c0 = len(head) + len('Question: ')
    B = Backend(head); tok = B.tok; d = B.d
    log('v213 extract | %s | part %s | pilot %s | prefix %d tokens | %.0fs to load' % (B.repo, part, pilot, B.n_head, time.time() - t0))
    meta, chk, extra = [], None, {}

    def tok_at(off, ch):
        return next(i for i, (s_, e_) in enumerate(off) if s_ <= ch + c0 < e_)

    def stack(S, off, pos):
        return torch.stack([torch.stack([S[s][p - off].float().cpu() for s in range(len(S))]) for p in pos]).numpy()

    if part in ('v181', 'b1'):
        fams = E2.families(json.load(open('v178_labels.json'))['items']); fams = fams[:3] if pilot else fams
        if part == 'v181':
            slots = ['+0', '+1', '+3', 'sent', 'qmark', 'last']; N = 10 * len(fams)
            X = np.zeros((N, len(slots), len(B.states), d), np.float16); XL = None; k = 0
            for fi, f in enumerate(fams):
                ref = None
                for dg in range(10):
                    q = f['question'][:f['cu']] + str(dg) + f['question'][f['cu'] + 1:]
                    e = tok(head + 'Question: %s\nAnswer:' % q, return_offsets_mapping=True); ids = e.input_ids
                    q_end = len(tok(head + 'Question: %s' % q).input_ids) - 1; j = tok_at(e.offset_mapping, f['cu'])
                    pos = [j, min(j + 1, q_end), min(j + 3, q_end), tok_at(e.offset_mapping, f['se']), q_end, len(ids) - 1]
                    ref = ref or (ids, pos); diff = [i for i, (a, b) in enumerate(zip(ids, ref[0])) if a != b]
                    ok = len(ids) == len(ref[0]) and pos == ref[1] and set(diff) <= {j} and tok.decode([ids[j]]).strip() == str(dg)
                    if chk is None:
                        chk = B.check(ids, pos)
                    S, D, lg, off, cached = B.run(ids)
                    X[k] = stack(S, off, pos)
                    dep = torch.stack([D[li][pos[-1] - off].float().cpu() for li in range(len(D))]).numpy()
                    if XL is None:
                        XL = np.zeros((N, dep.shape[0], d), np.float16)
                    XL[k] = dep
                    meta.append(dict(k=k, fam=fi, digit=dg, pos=pos, pair_ok=bool(ok), cached=cached, top=int(lg.argmax()),
                                     tid=f['tid'], inst=f['inst'], anchor=f['anchor']))
                    k += 1
                if fi % 20 == 0 or fi == len(fams) - 1:
                    log('v181 %3d/%d families (%.0fs)' % (fi + 1, len(fams), time.time() - t0))
            np.save(os.path.join(OUT, 'v213_v181_last_all.npy'), XL)
        else:
            sw = E5.swap_tokens(tok); wn = B.emb.weight.norm(dim=1).float().median().item()
            g = torch.Generator().manual_seed(205); noise = torch.randn(10, d, generator=g)
            noise = (noise / noise.norm(dim=1, keepdim=True) * wn).to(B.dev)
            log('swap tokens %s | noise norm %.3f (median embedding norm)' % ([tok.decode([i]) for i in sw], wn))
            slots = ['+0', 'qmark', 'last']; X = np.zeros((len(fams) * 20, 3, len(B.states), d), np.float16); k = 0
            for fi, f in enumerate(fams):
                q = f['question']; e = tok(head + 'Question: %s\nAnswer:' % q, return_offsets_mapping=True); ids0 = e.input_ids
                j = tok_at(e.offset_mapping, f['cu']); q_end = len(tok(head + 'Question: %s' % q).input_ids) - 1
                pos = [j, q_end, len(ids0) - 1]
                for kind in ('swap', 'noise'):
                    for c in range(10):
                        ids = list(ids0); vec = None
                        if kind == 'swap':
                            ids[j] = sw[c]
                        else:
                            vec = noise[c]
                        if fi == 0 and c == 0:
                            chk = max(chk or 0, B.check(ids, pos, j, vec))
                        S, D, lg, off, cached = B.run(ids, j, vec)
                        X[k] = stack(S, off, pos)
                        meta.append(dict(k=k, fam='%d_%s' % (fi, kind), kind=kind, tid=f['tid'], anchor=f['anchor'], digit=c,
                                         pos=pos, top=int(lg.argmax()), cached=cached))
                        k += 1
                if fi % 20 == 0 or fi == len(fams) - 1:
                    log('b1 %3d/%d families (%.0fs)' % (fi + 1, len(fams), time.time() - t0))
            extra = dict(swap_tokens=[tok.decode([i]) for i in sw], swap_ids=sw, noise_norm=wn)
        name = part
    else:
        items, h, info = I.load_checked(); extra = dict(items_sha=h, **info)
        if part == 'b2':
            sh, nsh = int(os.environ.get('V213_SHARD', '0')), int(os.environ.get('V213_NSHARD', '1'))
            its = [dict(it, gi=k) for k, it in enumerate(items) if k % nsh == sh]; name = 'b2_s%d' % sh
        else:
            its = []
            for src in ('gsm', 'orca'):
                te = sorted([it for it in items if it['source'] == src and it['split'] == 'test'], key=lambda it: it['rank'])[:NB3]
                its += [dict(it, question=it['orig'][:it['cu']] + str(dg) + it['orig'][it['cu'] + 1:], digit=dg)
                        for it in te for dg in range(10)]
            name = 'b3'
        its = its[:12] if pilot else its
        slots = ['+0', 'qmark', 'last']; X = np.zeros((len(its), 3, len(B.states), d), np.float16); t1 = time.time()
        for k, it in enumerate(its):
            e = tok(head + 'Question: %s\nAnswer:' % it['question'], return_offsets_mapping=True); ids = e.input_ids
            j = tok_at(e.offset_mapping, it['cu']); q_end = len(tok(head + 'Question: %s' % it['question']).input_ids) - 1
            pos = [j, q_end, len(ids) - 1]; ok = tok.decode([ids[j]]).strip() == str(it['digit'])
            if chk is None:
                chk = B.check(ids, pos)
            S, D, lg, off, cached = B.run(ids)
            X[k] = stack(S, off, pos)
            meta.append(dict(k=k, idx=it['idx'], fam=it['idx'], tid=it['idx'], digit=it['digit'], pos=pos, ok=bool(ok),
                             cached=cached, split=it['split'], rank=it['rank'], source=it['source'], gi=it.get('gi', -1)))
            if k % 500 == 0 or k == len(its) - 1:
                log('%s %5d/%d prompts (%.0fs, %.3f s/prompt)' % (name, k + 1, len(its), time.time() - t0, (time.time() - t1) / (k + 1)))
    np.save(os.path.join(OUT, 'v213_%s_states.npy' % name), X)
    json.dump(dict(model=B.repo, part=part, states=B.states, slots=slots, check=chk, check_kind='relative' if B.rel else 'absolute',
                   n_head=B.n_head, items=meta, wall_s=round(time.time() - t0), **extra),
              open(os.path.join(OUT, 'v213_%s_meta.json' % name), 'w'))
    log('%s done: %d prompts, cached %d, ok %d (%.0fs)' % (name, len(meta), sum(m.get('cached', False) for m in meta),
                                                          sum(m.get('ok', m.get('pair_ok', True)) for m in meta), time.time() - t0))


if __name__ == '__main__':
    main()
