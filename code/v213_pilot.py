"""v213 pilot (pre-freeze; timing and correctness only, no probe, no treatment statistic).

For the two scale-up models:
  qwen7b  Qwen/Qwen2.5-7B in fp16, split over the visible GPUs (device_map='auto'); the v202 prompt and prefix cache.
          Checks: cached vs full forward (logits and the stored states); no inf/nan; fp16 vs bf16 agreement on 2 prompts
          (relative difference of the stored states) as the fp16-overflow check.
  ouro26  ByteDance/Ouro-2.6B in fp32 on one GPU, 48 layers x 4 loops, with v206's minimal per-index KV cache and the
          longest-shared-prefix rule. Checks: cached vs full forward; our loop forward vs the model's own forward (logits).
On 8 GSM8K prompts (the first 8 usable training questions after the shots), reporting seconds per prompt.

  OUT_DIR=... V213_MODEL=qwen7b|ouro26 python v213_pilot.py
"""
import copy, json, os, re, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.')
MODEL = os.environ.get('V213_MODEL', 'qwen7b')
NUM = r'\d+(?:,\d{3})*(?:\.\d+)?'


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v213_pilot_%s.txt' % MODEL), 'a').write(s + '\n')


def gold(ans):
    return ans.split('####')[-1].strip().replace(',', '')


def prompts(tr, n=8):
    out = []
    for idx in range(4, len(tr)):
        q = tr[idx]['question']
        if [m for m in re.finditer(NUM, q) if '.' not in m.group() and len(m.group().replace(',', '')) >= 2]:
            out.append(q)
        if len(out) == n:
            return out


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    torch.set_grad_enabled(False); t0 = time.time()
    tr = load_dataset('openai/gsm8k', 'main', split='train')
    head = ''.join('Question: %s\nAnswer: %s\n\n' % (tr[i]['question'], gold(tr[i]['answer'])) for i in range(4))
    qs = prompts(tr)
    log('v213 pilot | %s | GPUs %d %s' % (MODEL, torch.cuda.device_count(),
                                         [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    if MODEL == 'qwen7b':
        repo = 'Qwen/Qwen2.5-7B'; LAYERS = [0, 4, 7, 14, 21, 28]
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
        dev = model.get_input_embeddings().weight.device
        log('loaded %s fp16 in %.0fs; device map %s' % (repo, time.time() - t0, sorted(set(str(v) for v in model.hf_device_map.values()))))
        head_ids = tok(head).input_ids; n_head = len(head_ids)
        prefix = model(input_ids=torch.tensor([head_ids], device=dev), use_cache=True).past_key_values

        def run(ids, cached=True):
            if cached:
                o = model(input_ids=torch.tensor([ids[n_head:]], device=dev), past_key_values=copy.deepcopy(prefix),
                          use_cache=True, output_hidden_states=True)
                return [h[0, -1].float().cpu() for h in o.hidden_states], o.logits[0, -1].float().cpu()
            o = model(input_ids=torch.tensor([ids], device=dev), use_cache=False, output_hidden_states=True)
            return [h[0, -1].float().cpu() for h in o.hidden_states], o.logits[0, -1].float().cpu()

        times = []; bad = 0; chk = 0.0
        for k, q in enumerate(qs):
            ids = tok(head + 'Question: %s\nAnswer:' % q).input_ids
            assert ids[:n_head] == head_ids
            t1 = time.time(); H, lg = run(ids); torch.cuda.synchronize(); times.append(time.time() - t1)
            bad += int(not all(torch.isfinite(H[l]).all() for l in LAYERS)) + int(not torch.isfinite(lg).all())
            if k < 2:
                H2, lg2 = run(ids, cached=False)
                rel = max(float((H[l] - H2[l]).norm() / H2[l].norm()) for l in LAYERS)
                chk = max(chk, rel, float((lg - lg2).abs().max() / lg2.abs().max()))
        log('cached vs full (fp16): max relative diff %.2e | non-finite prompts %d | %.3f s/prompt (median of %d, after the first)' % (
            chk, bad, float(np.median(times[1:])), len(times) - 1))
        # fp16 vs bf16 on 2 prompts (the overflow check)
        model = model.to(torch.bfloat16); prefix = model(input_ids=torch.tensor([head_ids], device=dev), use_cache=True).past_key_values
        rels = []
        for q in qs[:2]:
            ids = tok(head + 'Question: %s\nAnswer:' % q).input_ids
            Hb, _ = run(ids, cached=False)
            model.half(); Hh, _ = run(ids, cached=False); model.to(torch.bfloat16)
            rels.append(max(float((Hh[l] - Hb[l]).norm() / Hb[l].norm()) for l in LAYERS))
        log('fp16 vs bf16 (full forward, stored layers %s): max relative state diff %s' % (LAYERS, ['%.3f' % r for r in rels]))
    else:
        import torch.nn as nn
        repo = 'ByteDance/Ouro-2.6B'
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.float32).to('cuda').eval()
        b = model.model; L = b.config.num_hidden_layers; mod = __import__(type(b).__module__, fromlist=['x']); dev = 'cuda'
        log('loaded %s fp32 in %.0fs; layers %d, loops %s' % (repo, time.time() - t0, L, getattr(b.config, 'total_ut_steps', '?')))
        h_full = tok(head).input_ids; ex = tok(head + 'Question: x\nAnswer:').input_ids
        lcp = next((i for i, (u, v) in enumerate(zip(h_full, ex)) if u != v), min(len(h_full), len(ex)))
        head_ids = h_full[:max(1, lcp - 2)]; n_head = len(head_ids)

        class KVCache:
            def __init__(self):
                self.k, self.v = {}, {}

            def update(self, k, v, idx, kw=None):
                if idx in self.k:
                    self.k[idx] = torch.cat([self.k[idx], k], 2); self.v[idx] = torch.cat([self.v[idx], v], 2)
                else:
                    self.k[idx], self.v[idx] = k, v
                return self.k[idx], self.v[idx]

        def forward(ids, cache=None, start=0):
            x = torch.tensor([ids], device=dev); h = b.embed_tokens(x)
            cp = torch.arange(start, start + len(ids), device=dev); pos = cp.unsqueeze(0); n_tot = start + len(ids)
            full = mod.create_causal_mask(config=b.config, input_embeds=h.new_zeros(1, n_tot, h.shape[-1]), attention_mask=None,
                                          cache_position=torch.arange(n_tot, device=dev), past_key_values=None,
                                          position_ids=torch.arange(n_tot, device=dev).unsqueeze(0))
            if full is None:
                full = torch.ones(n_tot, n_tot, dtype=torch.bool, device=dev).tril()[None, None]
            mask = full[:, :, start:, :]; pe = b.rotary_emb(h, pos); S = []
            for t in range(4):
                for layer in b.layers[:L]:
                    h = layer(h, attention_mask=mask, position_ids=pos, past_key_value=cache, use_cache=cache is not None,
                              cache_position=cp, position_embeddings=pe, current_ut=t)
                    h = h[0] if isinstance(h, tuple) else h
                h = b.norm(h); S.append(h[0, -1].float().cpu())
            return S, model.lm_head(h)[0, -1].float().cpu()

        cache0 = KVCache(); forward(head_ids, cache0, 0)
        times = []; chk = 0.0; own = None
        for k, q in enumerate(qs):
            ids = tok(head + 'Question: %s\nAnswer:' % q).input_ids
            t1 = time.time()
            if ids[:n_head] == head_ids:
                S, lg = forward(ids[n_head:], copy.deepcopy(cache0), n_head)
            else:
                S, lg = forward(ids)
            torch.cuda.synchronize(); times.append(time.time() - t1)
            if k < 2:
                S2, lg2 = forward(ids)
                chk = max(chk, float((lg - lg2).abs().max()), max(float((a - c).abs().max()) for a, c in zip(S, S2)))
            if k == 0:
                o = model(input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0, -1].float().cpu()
                own = float((o - lg2).abs().max())
        log('cached prefix %d of %d head tokens | cached vs full: max abs diff %.2e | our forward vs the model\'s own: max |logit diff| %.2e | %.3f s/prompt (median of %d, after the first)' % (
            n_head, len(h_full), chk, own, float(np.median(times[1:])), len(times) - 1))
    log('total %.0fs; peak GPU memory %s GB' % (time.time() - t0, ['%.1f' % (torch.cuda.max_memory_allocated(i) / 1e9) for i in range(torch.cuda.device_count())]))


if __name__ == '__main__':
    main()
