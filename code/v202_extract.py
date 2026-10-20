"""v202 state extraction: the Ouro note's Sections 3.2-3.3 in a non-looped model (docs/prereg_v202_qwen_swamping.md).

Model: Qwen/Qwen2.5-1.5B (base; 28 layers, d = 1536; one token per digit), fp32, frozen. Same 4-shot answer-only prompt
as v176-v199 (the first four GSM8K training problems, each followed by its final answer), then the test question and
"Answer:".
Designs (questions rebuilt exactly as for Ouro, from v178_labels.json):
  v181: 137 families (one GSM-Symbolic question per template, anchors = first and last integer with >= 2 digits), the
        anchor's units digit set to 0..9. Slots +0, +1, +3, sent, qmark, last (as v181).
  v185: 77 base questions x {start, end} x {col: vary a colour word (D = 3), dig: vary D (colour red)} in the added
        sentence "Unrelated fact: a nearby park has 4D <colour> benches." Slots +0, qmark, last (as v185).
States: hidden states after layers [0 (embeddings), 4, 7, 14, 21, 28 (final, normed)] = h0, l4, l7, l14, l21, l28,
the analogues of Ouro's h0, mid1, end1-end4. The last slot is also stored at all 29 depths (descriptive profile).
Speed: the shared 4-shot prefix is run once and its key-value cache reused (a fresh copy per prompt); checked against a
full forward on the first prompt of each design (logits and stored states; abort if max |diff| >= 1e-3). A prompt whose
tokens do not start with the prefix's tokens is run in full.
Outputs (OUT_DIR): v202_<design>_states.npy (float16, N x slots x 6 x d), v202_<design>_last_all.npy (N x 29 x d),
v202_<design>_meta.json.
Env: V202_DESIGNS (default v181,v185), V202_PILOT=1 (first 6 families / 2 bases; plus a tokenisation-only check of all
items), TINY=1 (random tiny Qwen2 config, for a local code test), OUT_DIR.
"""
import copy, json, os, re, time
import numpy as np

NUM = r'\d+(?:,\d{3})*(?:\.\d+)?'
REPO = 'Qwen/Qwen2.5-1.5B'
LAYERS = [0, 4, 7, 14, 21, 28]
STATES = ['h0', 'l4', 'l7', 'l14', 'l21', 'l28']
SLOTS = {'v181': ['+0', '+1', '+3', 'sent', 'qmark', 'last'], 'v185': ['+0', 'qmark', 'last']}
IRR = 'Unrelated fact: a nearby park has 4%d %s benches. '
COLOURS = ['red', 'blue', 'green', 'black', 'white', 'brown', 'yellow', 'pink', 'gray', 'orange']
OUT = os.environ.get('OUT_DIR', '.')
TINY = os.environ.get('TINY') == '1'


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v202_log.txt'), 'a').write(s + '\n')


def gold(ans):
    return ans.split('####')[-1].strip().replace(',', '')


def families(items, seed=181):
    """v181_extract.families, unchanged."""
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


def bases(items):
    """v185_extract.bases, unchanged."""
    out = {}
    for f in families(items):
        if f['tid'] not in out or f['anchor'] == 'last':
            out[f['tid']] = f
    return [out[t] for t in sorted(out)]


def variants185(f, where, kind):
    """v185_extract.variants, unchanged: 10 (question, char index of the varied token) pairs."""
    q = f['question']
    ins = 0 if where == 'start' else q.rfind('. ', 0, len(q) - 1) + 2
    if where == 'end' and ins == 1:
        ins = 0
    out = []
    for k in range(10):
        dg, col = (3, COLOURS[k]) if kind == 'col' else (k, 'red')
        s = IRR % (dg, col)
        qq = q[:ins] + s + q[ins:]
        c = ins + s.index(' benches') - len(col) if kind == 'col' else ins + s.index(' %s benches' % col) - 1
        assert qq[c:].startswith(col if kind == 'col' else str(dg)), (qq, c, k)
        out.append((qq, c))
    return out


def build(design, items, pilot):
    """List of families: each a dict(meta=..., variants=[(question, varied char index, se char index or None, label)])."""
    fams = []
    if design == 'v181':
        fs = families(items)
        for f in (fs[:6] if pilot else fs):
            vs = [(f['question'][:f['cu']] + str(dg) + f['question'][f['cu'] + 1:], f['cu'], f['se'], dg) for dg in range(10)]
            fams.append(dict(meta=dict(tid=f['tid'], inst=f['inst'], anchor=f['anchor']), variants=vs, check=lambda d: str(d)))
    else:
        bs = bases(items)
        for f in (bs[:2] if pilot else bs):
            for where in ('start', 'end'):
                for kind in ('col', 'dig'):
                    vs = [(q, c, None, k) for k, (q, c) in enumerate(variants185(f, where, kind))]
                    fams.append(dict(meta=dict(tid=f['tid'], where=where, kind=kind), variants=vs,
                                     check=(lambda d: COLOURS[d]) if kind == 'col' else (lambda d: str(d))))
    return fams


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0 = time.time(); pilot = os.environ.get('V202_PILOT') == '1'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(REPO)
    if TINY:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(REPO); cfg.num_hidden_layers = 28; cfg.hidden_size = 64; cfg.intermediate_size = 128
        cfg.num_attention_heads = 4; cfg.num_key_value_heads = 2; cfg.head_dim = 16
        torch.manual_seed(0); model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32)
    model = model.to(dev).eval(); L = model.config.num_hidden_layers; assert L == 28, L
    if TINY:
        shots = [('What is %d plus %d?' % (i, i + 1), str(2 * i + 1)) for i in range(4)]
    else:
        from datasets import load_dataset
        tr = load_dataset('openai/gsm8k', 'main', split='train')
        shots = [(tr[i]['question'], gold(tr[i]['answer'])) for i in range(4)]
    head = ''.join('Question: %s\nAnswer: %s\n\n' % s for s in shots)
    head_ids = tok(head).input_ids; n_head = len(head_ids)
    pre = model(input_ids=torch.tensor([head_ids], device=dev), use_cache=True)
    prefix = pre.past_key_values
    items = json.load(open('v178_labels.json'))['items']
    designs = os.environ.get('V202_DESIGNS', 'v181,v185').split(',')
    log('v202 extract | %s | %s | designs %s | pilot %s | prefix %d tokens | device %s' % (
        REPO if not TINY else 'TINY', torch.__version__, designs, pilot, n_head, dev))

    def run(ids, cached=True):
        """Hidden states (29 x T x d, prompt positions only for the cached part) and last logits."""
        if cached and ids[:n_head] == head_ids:
            o = model(input_ids=torch.tensor([ids[n_head:]], device=dev), past_key_values=copy.deepcopy(prefix),
                      use_cache=True, output_hidden_states=True)
            return [h[0] for h in o.hidden_states], o.logits[0, -1], n_head, True
        o = model(input_ids=torch.tensor([ids], device=dev), use_cache=False, output_hidden_states=True)
        return [h[0] for h in o.hidden_states], o.logits[0, -1], 0, False

    for design in designs:
        fams = build(design, items, pilot); c0 = len(head) + len('Question: ')
        if pilot:                                        # tokenisation-only check over every item of the design
            allf = build(design, items, False); bad = 0
            for f in allf:
                ref = None
                for q, c, se, lab in f['variants']:
                    e = tok(head + 'Question: %s\nAnswer:' % q, return_offsets_mapping=True)
                    j = next(i for i, (s_, e_) in enumerate(e.offset_mapping) if s_ <= c + c0 < e_)
                    ok = tok.decode([e.input_ids[j]]).strip() == f['check'](lab) and e.input_ids[:n_head] == head_ids
                    if ref is not None:
                        diff = [i for i, (x, y) in enumerate(zip(e.input_ids, ref)) if x != y]
                        ok = ok and len(e.input_ids) == len(ref) and set(diff) <= {j}
                    ref = ref or e.input_ids; bad += not ok
            log('%s tokenisation check over all %d families: %d variants fail' % (design, len(allf), bad))
        slots = SLOTS[design]; N = 10 * len(fams); d = model.config.hidden_size
        X = np.zeros((N, len(slots), len(LAYERS), d), np.float16); XL = np.zeros((N, L + 1, d), np.float16)
        meta = []; k = 0; checked = None; t1 = time.time()
        for fi, f in enumerate(fams):
            ref = None
            for q, c, se, lab in f['variants']:
                p = head + 'Question: %s\nAnswer:' % q
                e = tok(p, return_offsets_mapping=True); ids = e.input_ids; off = e.offset_mapping
                q_end = len(tok(head + 'Question: %s' % q).input_ids) - 1

                def tok_at(ch):
                    ch += c0
                    return next(i for i, (s_, e_) in enumerate(off) if s_ <= ch < e_)
                j = tok_at(c)
                pos = ([j, min(j + 1, q_end), min(j + 3, q_end), tok_at(se), q_end, len(ids) - 1] if design == 'v181'
                       else [j, q_end, len(ids) - 1])
                if ref is None:
                    ref = (ids, pos)
                diff = [i for i, (x, y) in enumerate(zip(ids, ref[0])) if x != y]
                ok = (len(ids) == len(ref[0]) and pos == ref[1] and set(diff) <= {j}
                      and tok.decode([ids[j]]).strip() == f['check'](lab))
                H, lg, off0, cached = run(ids)
                if checked is None:                      # cached forward vs full forward, first prompt of the design
                    H2, lg2, _, _ = run(ids, cached=False)
                    dl = float((lg - lg2).abs().max())
                    dh = max(float((H[li][pp - off0] - H2[li][pp]).abs().max()) for li in LAYERS for pp in pos if pp >= off0)
                    checked = max(dl, dh)
                    log('%s check cached vs full forward: max |logit diff| %.2e, max |state diff| %.2e' % (design, dl, dh))
                    if not checked < 1e-3:
                        raise SystemExit('v202 CHECK FAILED')
                assert all(pp >= off0 for pp in pos), (pos, off0)
                X[k] = torch.stack([torch.stack([H[li][pp - off0] for li in LAYERS]) for pp in pos]).float().cpu().numpy()
                XL[k] = torch.stack([H[li][pos[-1] - off0] for li in range(L + 1)]).float().cpu().numpy()
                meta.append(dict(k=k, fam=fi, digit=lab, pos=pos, pair_ok=bool(ok), cached=cached,
                                 top=int(lg.argmax()), **f['meta']))
                k += 1
            if fi % 10 == 0 or fi == len(fams) - 1:
                log('%s %3d/%d families (%.0fs, %.2f s/prompt)' % (design, fi + 1, len(fams), time.time() - t0,
                                                                   (time.time() - t1) / k))
        np.save(os.path.join(OUT, 'v202_%s_states.npy' % design), X)
        np.save(os.path.join(OUT, 'v202_%s_last_all.npy' % design), XL)
        json.dump(dict(model=REPO, design=design, n=N, d=d, layers=LAYERS, states=STATES, slots=slots, check=checked,
                       n_head=n_head, items=meta, wall_s=round(time.time() - t1)),
                  open(os.path.join(OUT, 'v202_%s_meta.json' % design), 'w'))
        log('%s done: %d prompts, minimal pairs ok %d/%d, cached %d' % (
            design, N, sum(m['pair_ok'] for m in meta), N, sum(m['cached'] for m in meta)))
    log('all done %ds' % (time.time() - t0))


if __name__ == '__main__':
    main()
