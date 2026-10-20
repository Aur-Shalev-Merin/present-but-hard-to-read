"""v205 Part B extraction on Qwen2.5-1.5B (docs/prereg_v205_swamping_robustness.md). The v202 forward pass, prompt and
prefix cache, unchanged (v202_extract.py helpers are imported).

  b1  random-perturbation null. The 137 v181 families with their original digit; at the varied digit's token, 10 fixed
      "classes" of a change that carries no digit:
        swap   the digit's token id replaced by one of 10 fixed random word tokens (same position, same length);
        noise  one of 10 fixed Gaussian vectors, of norm equal to the median input-embedding norm, added to the digit
               token's input embedding.
      137 x 2 kinds x 10 = 2,740 prompts. Slots +0, qmark, last; states h0, l4, l7, l14, l21, l28; last at all 29 depths.
  b2  template-count curve. EVERY usable GSM8K training question (index >= 4, so the four shots are excluded; usable =
      contains an integer of >= 2 digits); the last such integer's units digit is set to a digit drawn at random (seed
      205); one prompt per question. A fixed random 1,000 (permutation seed 2051) are the test set, the rest the training
      pool. Run in 3 shards (V205_SHARD = 0, 1, 2: items k with k % 3 == shard). Slots +0, qmark, last.
  b3  positive control for b2: minimal pairs on the first 150 test questions (in permutation order), each with its
      units digit set to 0..9 (1,500 prompts). Slots +0, qmark, last.
Check: cached vs full forward on the first prompt of each part (and of each b1 kind); abort if max |diff| >= 1e-3.
Outputs (OUT_DIR): v205_<part>_states.npy, v205_<part>_last_all.npy (b1 only), v205_<part>_meta.json.
Env: V205_PART (b1 | b2 | b3), V205_SHARD (b2), V205_PILOT=1 (b1: 3 families; b2: 40 questions; b3: 4 questions), TINY=1,
OUT_DIR.
"""
import copy, json, os, re, time
import numpy as np

import v202_extract as E

OUT = os.environ.get('OUT_DIR', '.')
N_TEST, N_B3 = 1000, 150


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v205_log.txt'), 'a').write(s + '\n')


def swap_tokens(tok, n=10, seed=205):
    """10 fixed word tokens: decode to ' ' + 3-8 lowercase letters, drawn at random from the vocabulary."""
    rng = np.random.default_rng(seed); ids = rng.permutation(len(tok)); out = []
    for i in ids:
        s = tok.decode([int(i)])
        if re.fullmatch(r' [a-z]{3,8}', s):
            out.append(int(i))
        if len(out) == n:
            return out


def gsm_items(tr, seed=205):
    """Every usable question, with its random digit and its split (test = a fixed random 1,000)."""
    rng = np.random.default_rng(seed); out = []
    for idx in range(4, len(tr)):
        q = tr[idx]['question']
        ms = [m for m in re.finditer(E.NUM, q) if '.' not in m.group() and len(m.group().replace(',', '')) >= 2]
        if not ms:
            continue
        cu = ms[-1].end() - 1; d = int(rng.integers(10))
        out.append(dict(idx=idx, orig=q, question=q[:cu] + str(d) + q[cu + 1:], cu=cu, digit=d))
    perm = np.random.default_rng(2051).permutation(len(out))
    for r, k in enumerate(perm):
        out[k]['split'] = 'test' if r < N_TEST else 'train'; out[k]['rank'] = r
    return out


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0 = time.time(); part = os.environ['V205_PART']; pilot = os.environ.get('V205_PILOT') == '1'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(E.REPO)
    if E.TINY:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(E.REPO); cfg.num_hidden_layers = 28; cfg.hidden_size = 64; cfg.intermediate_size = 128
        cfg.num_attention_heads = 4; cfg.num_key_value_heads = 2; cfg.head_dim = 16
        torch.manual_seed(0); model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
        shots = [('What is %d plus %d?' % (i, i + 1), str(2 * i + 1)) for i in range(4)]
    else:
        model = AutoModelForCausalLM.from_pretrained(E.REPO, torch_dtype=torch.float32)
    from datasets import load_dataset
    tr = load_dataset('openai/gsm8k', 'main', split='train')
    if not E.TINY:
        shots = [(tr[i]['question'], E.gold(tr[i]['answer'])) for i in range(4)]
    model = model.to(dev).eval(); L = model.config.num_hidden_layers; emb = model.get_input_embeddings()
    head = ''.join('Question: %s\nAnswer: %s\n\n' % s for s in shots)
    head_ids = tok(head).input_ids; n_head = len(head_ids); c0 = len(head) + len('Question: ')
    prefix = model(input_ids=torch.tensor([head_ids], device=dev), use_cache=True).past_key_values
    d = model.config.hidden_size
    log('v205 extract | part %s | pilot %s | %s | prefix %d tokens | device %s' % (part, pilot, 'TINY' if E.TINY else E.REPO,
                                                                               n_head, dev))

    def run(ids, j=None, vec=None, cached=True):
        x = emb(torch.tensor([ids], device=dev))
        if vec is not None:
            x[0, j] = x[0, j] + vec
        if cached:
            assert ids[:n_head] == head_ids
            o = model(inputs_embeds=x[:, n_head:], past_key_values=copy.deepcopy(prefix), use_cache=True,
                      output_hidden_states=True)
            return [h[0] for h in o.hidden_states], o.logits[0, -1], n_head
        o = model(inputs_embeds=x, use_cache=False, output_hidden_states=True)
        return [h[0] for h in o.hidden_states], o.logits[0, -1], 0

    def check(name, ids, pos, j=None, vec=None):
        H, lg, off = run(ids, j, vec); H2, lg2, _ = run(ids, j, vec, cached=False)
        dl = float((lg - lg2).abs().max())
        dh = max(float((H[li][p - off] - H2[li][p]).abs().max()) for li in E.LAYERS for p in pos)
        log('%s check cached vs full: max |logit diff| %.2e, max |state diff| %.2e' % (name, dl, dh))
        if not max(dl, dh) < 1e-3:
            raise SystemExit('v205 CHECK FAILED')
        return max(dl, dh)

    meta = []; chk = []
    if part == 'b1':
        fams = E.families(json.load(open('v178_labels.json'))['items'])
        fams = fams[:3] if pilot else fams
        sw = swap_tokens(tok); wn = emb.weight.norm(dim=1).median().item()
        g = torch.Generator().manual_seed(205); noise = torch.randn(10, d, generator=g); noise = noise / noise.norm(dim=1, keepdim=True) * wn
        noise = noise.to(dev)
        log('swap tokens %s | noise norm %.3f (median embedding norm)' % ([tok.decode([i]) for i in sw], wn))
        slots = ['+0', 'qmark', 'last']; N = len(fams) * 20
        X = np.zeros((N, 3, len(E.LAYERS), d), np.float16); XL = np.zeros((N, L + 1, d), np.float16); k = 0
        for fi, f in enumerate(fams):
            q = f['question']; p = head + 'Question: %s\nAnswer:' % q
            e = tok(p, return_offsets_mapping=True); ids0 = e.input_ids
            j = next(i for i, (s_, e_) in enumerate(e.offset_mapping) if s_ <= f['cu'] + c0 < e_)
            q_end = len(tok(head + 'Question: %s' % q).input_ids) - 1; pos = [j, q_end, len(ids0) - 1]
            assert tok.decode([ids0[j]]).strip() == q[f['cu']], (q, j)
            for kind in ('swap', 'noise'):
                for c in range(10):
                    ids = list(ids0); vec = None
                    if kind == 'swap':
                        ids[j] = sw[c]
                    else:
                        vec = noise[c]
                    if fi == 0 and c == 0:
                        chk.append(check('b1 %s' % kind, ids, pos, j, vec))
                    H, lg, off = run(ids, j, vec)
                    X[k] = torch.stack([torch.stack([H[li][pp - off] for li in E.LAYERS]) for pp in pos]).float().cpu().numpy()
                    XL[k] = torch.stack([H[li][pos[-1] - off] for li in range(L + 1)]).float().cpu().numpy()
                    meta.append(dict(k=k, fam='%d_%s' % (fi, kind), kind=kind, tid=f['tid'], anchor=f['anchor'], digit=c,
                                     pos=pos, top=int(lg.argmax())))
                    k += 1
            if fi % 10 == 0 or fi == len(fams) - 1:
                log('b1 %3d/%d families (%.0fs)' % (fi + 1, len(fams), time.time() - t0))
        np.save(os.path.join(OUT, 'v205_b1_last_all.npy'), XL)
        extra = dict(swap_tokens=[tok.decode([i]) for i in sw], swap_ids=sw, noise_norm=wn)
    else:
        allit = gsm_items(tr)
        if part == 'b2':
            sh = int(os.environ.get('V205_SHARD', '0'))
            its = [it for k, it in enumerate(allit) if k % 3 == sh]
            its = its[:40] if pilot else its
        else:                                           # b3: minimal pairs on the first 150 test questions
            te = sorted([it for it in allit if it['split'] == 'test'], key=lambda it: it['rank'])[:4 if pilot else N_B3]
            its = [dict(idx=it['idx'], question=it['orig'][:it['cu']] + str(dg) + it['orig'][it['cu'] + 1:], cu=it['cu'],
                        digit=dg, split='test', rank=it['rank']) for it in te for dg in range(10)]
        log('%s: %d usable questions in all, %d prompts here' % (part, len(allit), len(its)))
        slots = ['+0', 'qmark', 'last']; X = np.zeros((len(its), 3, len(E.LAYERS), d), np.float16)
        for k, it in enumerate(its):
            p = head + 'Question: %s\nAnswer:' % it['question']
            e = tok(p, return_offsets_mapping=True); ids = e.input_ids
            j = next(i for i, (s_, e_) in enumerate(e.offset_mapping) if s_ <= it['cu'] + c0 < e_)
            q_end = len(tok(head + 'Question: %s' % it['question']).input_ids) - 1; pos = [j, q_end, len(ids) - 1]
            ok = tok.decode([ids[j]]).strip() == str(it['digit']) and ids[:n_head] == head_ids
            if k == 0:
                chk.append(check(part, ids, pos))
            H, lg, off = run(ids) if ok else run(ids, cached=False)
            X[k] = torch.stack([torch.stack([H[li][pp - off] for li in E.LAYERS]) for pp in pos]).float().cpu().numpy()
            meta.append(dict(k=k, idx=it['idx'], fam=it['idx'], tid=it['idx'], digit=it['digit'], pos=pos, ok=bool(ok),
                             split=it['split'], rank=it['rank']))
            if k % 200 == 0 or k == len(its) - 1:
                log('%s %4d/%d prompts (%.0fs)' % (part, k + 1, len(its), time.time() - t0))
        extra = dict(n_usable=len(allit), n_test=N_TEST)
        if part == 'b2':
            part = 'b2_s%d' % sh
    np.save(os.path.join(OUT, 'v205_%s_states.npy' % part), X)
    json.dump(dict(model=E.REPO, part=part, layers=E.LAYERS, states=E.STATES, slots=slots, check=max(chk), n_head=n_head,
                   items=meta, wall_s=round(time.time() - t0), **extra), open(os.path.join(OUT, 'v205_%s_meta.json' % part), 'w'))
    log('%s done: %d prompts (%.0fs)' % (part, len(meta), time.time() - t0))


if __name__ == '__main__':
    main()
