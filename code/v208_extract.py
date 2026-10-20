"""v208: linear response of the summary position to a change at the varied digit's input embedding
(docs/prereg_v208_input_sensitivity.md). Qwen2.5-1.5B and Ouro-1.4B, fp32, frozen, forward passes only (Kaggle CPU).

Prompts: the v181 questions with their original digit, one per template (the 'first' anchor of the first V208_N templates
in v202_extract.families order), in the v176-v206 4-shot answer-only prompt. j = the anchor's units-digit token.
Directions at the input embedding of token j (unit vectors; fixed across prompts except DIG and SWAPD):
  DIG     (e_c - e_o)/|.| for the 9 other digits c (o = the original digit)
  PAIR    (e_w1 - e_w2)/|.| for 30 random word pairs (words as v205.swap_tokens: ' ' + 3-8 lowercase letters; seed 208)
  ISO     64 isotropic Gaussian directions (seed 208); their responses also estimate the amplified output
          directions used by Part B (v208_method.py)
  SWAPD   (e_w - e_o)/|.| for the 10 v205 B1 swap words
  NOISED  the 10 v205 B1 noise directions
Linear response r(u) = (h(e_j + s u) - h(e_j - s u)) / (2 s), s = V208_STEP x the median input-embedding norm.
Full-size changes (linearity, descriptive): the 9 digit swaps, the 10 v205 swap words, the 10 v205 noise vectors (norm =
median embedding norm) added to the digit, each as h(changed) - h(base).
Slots: j (the varied token), qmark (end of the question), last (last prompt token). States:
  qwen  hidden states after layers 4, 7, 14, 21, 28;   ouro  mid1 (after layer 12 of loop 1), end1-end4 (normed loop ends).
Speed: positions < j do not depend on token j (causal attention), so their key-value cache is computed once per prompt
and every changed forward runs tokens j..end only. Checked on the first prompt of each run (base and one noise change)
against a full uncached forward; abort if max |diff| >= 1e-3.
Step check (every run, first prompt): relative change of r between s and s/2 for the first 2 directions of each class.
Pilot (V208_PILOT=1): 2 prompts; cache check, step check at several steps, timing. Saves NO responses.
Outputs (OUT_DIR): v208_<model>_s<k>_lin.npy (float32, n x 123 x 3 x 5 x d), v208_<model>_s<k>_full.npy (n x 29 x 3 x 5 x d),
v208_<model>_s<k>_meta.json.
Env: V208_MODEL (qwen | ouro), V208_SHARD / V208_NSHARD, V208_N (default 24), V208_STEP (default 0.02), V208_PILOT,
TINY=1 (random tiny model of the same architecture; local code test), OUT_DIR.
"""
import copy, json, os, re, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.')
MODEL = os.environ.get('V208_MODEL', 'qwen')
N_PROMPTS = int(os.environ.get('V208_N', '24'))
STEP = float(os.environ.get('V208_STEP', '0.02'))
PILOT = os.environ.get('V208_PILOT') == '1'
TINY = os.environ.get('TINY') == '1'
SLOTS = ['j', 'qmark', 'last']
STATES = {'qwen': ['l4', 'l7', 'l14', 'l21', 'l28'], 'ouro': ['mid1', 'end1', 'end2', 'end3', 'end4']}[MODEL]
CLASSES = [('DIG', 9), ('PAIR', 30), ('ISO', 64), ('SWAPD', 10), ('NOISED', 10)]
FULL = [('DIG', 9), ('SWAP', 10), ('NOISE', 10)]
if MODEL == 'ouro':
    os.environ.setdefault('LOOP_MODEL', 'ouro')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v208_log.txt'), 'a').write(s + '\n')


def pick_prompts(fams):
    seen, out = set(), []
    for f in fams:
        if f['anchor'] == 'first' and f['tid'] not in seen:
            seen.add(f['tid']); out.append(f)
    return out[:N_PROMPTS]


def word_tokens(tok, n, seed):
    rng = np.random.default_rng(seed); out = []
    for i in rng.permutation(len(tok)):
        if re.fullmatch(r' [a-z]{3,8}', tok.decode([int(i)])):
            out.append(int(i))
        if len(out) == n:
            return out


class QwenA:
    def __init__(self, torch):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import v202_extract as E
        self.t = torch; self.tok = AutoTokenizer.from_pretrained(E.REPO)
        if TINY:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(E.REPO); cfg.num_hidden_layers = 28; cfg.hidden_size = 64
            cfg.intermediate_size = 128; cfg.num_attention_heads = 4; cfg.num_key_value_heads = 2; cfg.head_dim = 16
            torch.manual_seed(0); m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
        else:
            m = AutoModelForCausalLM.from_pretrained(E.REPO, torch_dtype=torch.float32)
        self.m = m.eval(); self.W = m.get_input_embeddings().weight; self.d = m.config.hidden_size
        self.LI = [4, 7, 14, 21, 28]

    def cache(self, x):
        return self.m(inputs_embeds=x, use_cache=True).past_key_values

    def run(self, x, cache=None):
        o = self.m(inputs_embeds=x, past_key_values=copy.deepcopy(cache) if cache is not None else None,
                   use_cache=cache is not None, output_hidden_states=True)
        return [o.hidden_states[li][0] for li in self.LI]


class OuroA:
    def __init__(self, torch):
        import loop_grad_probe as P
        self.t = torch; self.tok, model = P.load(); ad = P.Ouro(model); self.b = b = ad.b; self.mod = ad.mod
        self.L = b.config.num_hidden_layers; self.mid = min(12, self.L) - 1; self.d = b.config.hidden_size
        self.W = b.embed_tokens.weight

        class KVCache:
            """Ouro's attention only calls update(k, v, loop * 24 + layer, kwargs); a minimal per-index cache (as v206)."""
            def __init__(s):
                s.k, s.v = {}, {}

            def update(s, k, v, idx, kw=None):
                if idx in s.k:
                    s.k[idx] = torch.cat([s.k[idx], k], 2); s.v[idx] = torch.cat([s.v[idx], v], 2)
                else:
                    s.k[idx], s.v[idx] = k, v
                return s.k[idx], s.v[idx]
        self.KV = KVCache

    def _fwd(self, x, cache, start):
        t, b = self.t, self.b; n = x.shape[1]; n_tot = start + n
        dv = x.device; cp = t.arange(start, n_tot, device=dv); pos = cp.unsqueeze(0)
        full = self.mod.create_causal_mask(config=b.config, input_embeds=x.new_zeros(1, n_tot, x.shape[-1]),
                                           attention_mask=None, cache_position=t.arange(n_tot, device=dv), past_key_values=None,
                                           position_ids=t.arange(n_tot, device=dv).unsqueeze(0))
        if full is None:
            full = t.ones(n_tot, n_tot, dtype=t.bool, device=dv).tril()[None, None]
        mask = full[:, :, start:, :]; pe = b.rotary_emb(x, pos); h = x; S = []
        for ut in range(4):
            for li, layer in enumerate(b.layers[:self.L]):
                h = layer(h, attention_mask=mask, position_ids=pos, past_key_value=cache, use_cache=cache is not None,
                          cache_position=cp, position_embeddings=pe, current_ut=ut)
                h = h[0] if isinstance(h, tuple) else h
                if li == self.mid and ut == 0:
                    S.append(h[0])
            h = b.norm(h); S.append(h[0])
        return S

    def cache(self, x):
        c = self.KV(); self._fwd(x, c, 0); c.start = x.shape[1]; return c

    def run(self, x, cache=None):
        if cache is None:
            return self._fwd(x, None, 0)
        return self._fwd(x, copy.deepcopy(cache), cache.start)


def main():
    import torch
    import v202_extract as E2
    import v205_extract as E5
    torch.set_grad_enabled(False); t0 = time.time()
    A = QwenA(torch) if MODEL == 'qwen' else OuroA(torch); tok, W, d = A.tok, A.W, A.d
    if TINY:
        shots = [('What is %d plus %d?' % (i, i + 1), str(2 * i + 1)) for i in range(4)]
    else:
        from datasets import load_dataset
        tr = load_dataset('openai/gsm8k', 'main', split='train')
        shots = [(tr[i]['question'], E2.gold(tr[i]['answer'])) for i in range(4)]
    head = ''.join('Question: %s\nAnswer: %s\n\n' % s for s in shots); c0 = len(head) + len('Question: ')
    prompts = pick_prompts(E2.families(json.load(open('v178_labels.json'))['items']))
    sh, nsh = int(os.environ.get('V208_SHARD', '0')), int(os.environ.get('V208_NSHARD', '1'))
    mine = [f for k, f in enumerate(prompts) if k % nsh == sh]
    mine = mine[:2] if PILOT else mine

    wn = float(W.norm(dim=1).median()); s = STEP * wn
    dig = []
    for c in range(10):
        ii = tok(str(c), add_special_tokens=False).input_ids; assert len(ii) == 1, (c, ii); dig.append(ii[0])
    wt = word_tokens(tok, 60, 208)
    unit = lambda v: v / v.norm(dim=-1, keepdim=True)
    PAIR = unit(W[wt[0::2]] - W[wt[1::2]])
    g = torch.Generator().manual_seed(208); ISO = unit(torch.randn(64, d, generator=g)).to(W.device)
    sw = E5.swap_tokens(tok)                                     # the 10 v205 B1 swap words (seed 205)
    g5 = torch.Generator().manual_seed(205); noise = torch.randn(10, d, generator=g5)
    noise = (noise / noise.norm(dim=1, keepdim=True) * wn).to(W.device)   # exactly v205 B1's noise vectors
    NOISED = unit(noise)
    log('v208 extract | %s | %s | shard %d/%d | %d prompts | pilot %s | step %.4f (s = %.5f, median emb norm %.4f) | d %d'
        % (MODEL, 'TINY' if TINY else 'real', sh, nsh, len(mine), PILOT, STEP, s, wn, d))
    log('swap words %s | pair words %s ...' % ([tok.decode([i]) for i in sw], [tok.decode([i]) for i in wt[:6]]))

    def setup(f):
        q = f['question']; p = head + 'Question: %s\nAnswer:' % q
        e = tok(p, return_offsets_mapping=True); ids = e.input_ids
        j = next(i for i, (a, b_) in enumerate(e.offset_mapping) if a <= f['cu'] + c0 < b_)
        assert tok.decode([ids[j]]).strip() == q[f['cu']] and ids[j] == dig[f['orig_digit']], (q, j)
        q_end = len(tok(head + 'Question: %s' % q).input_ids) - 1; last = len(ids) - 1
        assert j <= q_end <= last
        x = W[torch.tensor(ids, device=W.device)].unsqueeze(0).clone()
        return ids, j, [0, q_end - j, last - j], x, [j, q_end, last]

    def take(S, sl):
        return torch.stack([torch.stack([S[k][p] for k in range(len(STATES))]) for p in sl])   # 3 x 5 x d

    def lin(x, cache, sl, u, step):
        xp = x.clone(); xp[0, 0] += step * u; xm = x.clone(); xm[0, 0] -= step * u
        return (take(A.run(xp, cache), sl) - take(A.run(xm, cache), sl)) / (2 * step)

    N = len(mine); n_lin = sum(n for _, n in CLASSES); n_full = sum(n for _, n in FULL)
    LIN = None if PILOT else np.zeros((N, n_lin, 3, len(STATES), d), np.float32)
    FUL = None if PILOT else np.zeros((N, n_full, 3, len(STATES), d), np.float32)
    meta = []; checks = {}
    for k, f in enumerate(mine):
        t1 = time.time(); ids, j, sl, xall, absl = setup(f); o = f['orig_digit']
        cache = A.cache(xall[:, :j]); x = xall[:, j:]
        S0 = take(A.run(x, cache), sl)
        if k == 0:                                               # cache check: base and one noise change, vs full forward
            Sf = take(A.run(xall), absl); xn = xall.clone(); xn[0, j] += noise[0]
            xs = x.clone(); xs[0, 0] += noise[0]
            dmax = max(float((S0 - Sf).abs().max()), float((take(A.run(xs, cache), sl) - take(A.run(xn), absl)).abs().max()))
            checks['cache_vs_full_maxabs'] = dmax; log('check cached vs full forward: max |state diff| %.2e' % dmax)
            if not dmax < 1e-3:
                raise SystemExit('v208 CACHE CHECK FAILED')
        DIG = unit(W[[dig[c] for c in range(10) if c != o]] - W[dig[o]]); SWAPD = unit(W[sw] - W[dig[o]])
        dirs = torch.cat([DIG, PAIR, ISO, SWAPD, NOISED])
        if k == 0:                                               # step check (numerical accuracy only)
            sel = [0, 1, 9, 10, 39, 40, 103, 104, 113, 114]
            steps = [STEP * wn * m for m in ([0.25, 0.5, 1, 2.5] if PILOT else [1])]
            for st in steps:
                rel = []
                for i in sel:
                    a_, b2 = lin(x, cache, sl, dirs[i], st), lin(x, cache, sl, dirs[i], st / 2)
                    rel.append(float((a_ - b2).norm() / b2.norm()))
                checks.setdefault('step_rel', {})['%.3f' % (st / wn)] = [max(rel), float(np.median(rel))]
                log('step check: step %.4f x median norm: rel change s vs s/2 max %.2e, median %.2e' % (st / wn, max(rel),
                                                                                                      float(np.median(rel))))
        if PILOT:
            t2 = time.time()
            for i in range(4):
                lin(x, cache, sl, dirs[i], s)
            per = (time.time() - t2) / 8
            log('pilot prompt %d: %d tokens, j %d, segment %d tokens | %.2f s per forward -> about %.0f s per prompt' % (
                k, len(ids), j, len(ids) - j, per, per * (2 * n_lin + n_full + 2)))
            continue
        for i in range(n_lin):
            LIN[k, i] = lin(x, cache, sl, dirs[i], s).float().cpu().numpy()
        fn = []; r = 0
        for c in range(10):
            if c != o:
                xc = x.clone(); xc[0, 0] = W[dig[c]]; FUL[k, r] = (take(A.run(xc, cache), sl) - S0).cpu().numpy()
                fn.append(float((W[dig[c]] - W[dig[o]]).norm())); r += 1
        for w in sw:
            xc = x.clone(); xc[0, 0] = W[w]; FUL[k, r] = (take(A.run(xc, cache), sl) - S0).cpu().numpy()
            fn.append(float((W[w] - W[dig[o]]).norm())); r += 1
        for v in noise:
            xc = x.clone(); xc[0, 0] += v; FUL[k, r] = (take(A.run(xc, cache), sl) - S0).cpu().numpy()
            fn.append(float(v.norm())); r += 1
        meta.append(dict(k=k, tid=f['tid'], orig_digit=o, n_tokens=len(ids), j=j, slots_abs=absl, full_input_norms=fn,
                         s=(time.time() - t1)))
        log('prompt %d/%d (tid %s): %d tokens, segment %d | %.0f s (total %.0f s)' % (k + 1, N, f['tid'], len(ids),
                                                                                    len(ids) - j, time.time() - t1,
                                                                                    time.time() - t0))
    if PILOT:
        log('pilot done (%.0f s); no responses saved' % (time.time() - t0)); return
    tag = 'v208_%s_s%d' % (MODEL, sh)
    np.save(os.path.join(OUT, tag + '_lin.npy'), LIN); np.save(os.path.join(OUT, tag + '_full.npy'), FUL)
    json.dump(dict(model=MODEL, tiny=TINY, states=STATES, slots=SLOTS, classes=CLASSES, full=FULL, step=STEP, s=s,
                   median_emb_norm=wn, swap_words=[tok.decode([i]) for i in sw], pair_words=[tok.decode([i]) for i in wt],
                   checks=checks, items=meta, wall_s=round(time.time() - t0)),
              open(os.path.join(OUT, tag + '_meta.json'), 'w'))
    log('%s done: %d prompts (%.0f s)' % (tag, N, time.time() - t0))


if __name__ == '__main__':
    main()
