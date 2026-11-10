"""v235 pilot 2 (GPU; docs/v235_design.md): ground truth for the two-hop positive control, before any learned clamp.

Targets: 8 cities per country (seed 2350) among the 1,046 cities whose country (full continuation) and two-hop capital
(full continuation and first token) Qwen2.5-7B gets right in pilot 1. Target prompt: pilot 1's two-hop template
"The capital of the country where the city of {X} is located is" (three demonstrations).
Whole-state patches at one position, one layer at a time (hidden_states l = output of decoder layer l-1, l in LAYERS),
plus 'all' (every layer output 1-27 at that position; hidden_states 28 is after the final norm). Positions: e1 (the city's last token) and last (the prompt's last
token). Sources (the source state is taken at the matching position of the source prompt):
  city_other   a known city of another country, same template          (hit = the source's capital)
  city_same    another known city of the same country                   (the answer should stay)
  person       a known person born in another country, pilot 1's person two-hop template
  country      another country's name in "The capital of {name} is"
Mechanics: a random 24-dim subspace clamp, h <- h - U U^T (h - h'), at e1, hidden_states 8, 12, 16, 20, source city_other
(should do nothing); no learned subspace is used here.
Outcome: argmax over the 25 capitals' first tokens at the last position. hit = the source country's capital, kept = the
target's.
States: for all 1,046 cities, hidden_states 0, 4, 8, 12, 16, 20, 24, 27 at e1 and last (fp16), and the mean of
hidden_states 0 over the city-name tokens (the copy's surface code) -> v235_states.npz, for readout checks on CPU.
"""
import json, os, sys, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.'); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v235_pilot import prompt

LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24]; RAND_L = [8, 12, 16, 20]; SAVE_L = [0, 4, 8, 12, 16, 20, 24, 27]
SOURCES = ['city_other', 'city_same', 'person', 'country']; POS = ['e1', 'last']


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v235_pilot2_log.txt'), 'a').write(s + '\n')


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False); t0 = time.time()
    D = json.load(open(os.path.join(HERE, 'v235_cities.json'), encoding='utf-8'))
    S = json.load(open(os.path.join(HERE, 'v235_pilot2_items.json'), encoding='utf-8'))
    keys = S['keys']; caps = [D['countries'][k]['capital'] for k in keys]; names = [D['countries'][k]['name'] for k in keys]
    repo = 'Qwen/Qwen2.5-7B'; tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    dev = m.get_input_embeddings().weight.device; layers = m.model.layers; NL = len(layers)
    first = [tok.encode(' ' + c, add_special_tokens=False)[0] for c in caps]
    assert len(set(first)) == len(first)

    def enc(kind, x):
        p = prompt(kind, x); e = tok(p, return_offsets_mapping=True); off = e['offset_mapping']
        end = p.rfind(x) + len(x) - 1
        t = next(j for j, (u, v) in enumerate(off) if u <= end < v)
        name = [j for j, (u, v) in enumerate(off) if v > p.rfind(x) and u <= end]
        return e['input_ids'], t, len(e['input_ids']) - 1, name

    st = {'mode': None}

    def hook(li):
        hi = li + 1

        def f(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if st['mode'] == 'patch':
                for j, ls in enumerate(st['rows']):
                    if hi in ls:
                        h[j, st['t']] = st['src'][hi].to(h.device, h.dtype)
            elif st['mode'] == 'rand' and hi in RAND_L:
                U = st['U'].to(h.device, h.dtype); d = h[0, st['t']] - st['src'][hi].to(h.device, h.dtype)
                h[0, st['t']] = h[0, st['t']] - U @ (U.T @ d)
            return out
        return f
    for li in range(NL):
        layers[li].register_forward_hook(hook(li))

    def run(ids, n=1):
        x = torch.tensor([ids] * n, device=dev)
        return m(input_ids=x).logits[:, -1, first].float().argmax(-1).tolist()

    def states(ids):
        hs = m(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states
        return hs
    log('loaded in %.0fs | %d targets | %d cities for states' % (time.time() - t0, len(S['targets']), len(S['all_cities'])))

    # 1) states for all known cities
    sv = {'e1': [], 'last': [], 'name0': []}; t1 = time.time()
    for n, c in enumerate(S['all_cities']):
        ids, te, tl, name = enc('city_cap', c['city']); hs = states(ids)
        sv['e1'].append(np.stack([hs[l][0, te].float().cpu().numpy() for l in SAVE_L]).astype(np.float16))
        sv['last'].append(np.stack([hs[l][0, tl].float().cpu().numpy() for l in SAVE_L]).astype(np.float16))
        sv['name0'].append(hs[0][0, name].float().mean(0).cpu().numpy().astype(np.float16))
        if n == 0:
            log('example: e1 token %r, last token %r, name tokens %r' % (tok.decode([ids[te]]), tok.decode([ids[tl]]),
                                                                           [tok.decode([ids[j]]) for j in name]))
    np.savez_compressed(os.path.join(OUT, 'v235_states.npz'), e1=np.stack(sv['e1']), last=np.stack(sv['last']),
                        name0=np.stack(sv['name0']), y=np.array([keys.index(c['ckey']) for c in S['all_cities']]),
                        save_layers=np.array(SAVE_L))
    log('states saved (%.0fs)' % (time.time() - t1))

    # 2) patches
    rows = [[l] for l in LAYERS] + [list(range(1, NL))]; rlab = ['L%d' % l for l in LAYERS] + ['all']
    rng = np.random.default_rng(2351); U = torch.tensor(np.linalg.qr(rng.standard_normal((m.config.hidden_size, 24)))[0])
    res = []; t1 = time.time()
    for n, tg in enumerate(S['targets']):
        ids, te, tl, _ = enc('city_cap', tg['city']); y = keys.index(tg['ckey'])
        r = dict(city=tg['city'], y=y, none=run(ids)[0])
        for s in SOURCES:
            src = tg[s]; kind = {'city_other': 'city_cap', 'city_same': 'city_cap', 'person': 'person_cap',
                                 'country': 'country_cap'}[s]
            sx = src['x'] if s != 'country' else names[keys.index(src['ckey'])]
            sids, se, sl, _ = enc(kind, sx); hs = states(sids); ys = keys.index(src['ckey'])
            r[s + '_y'] = ys
            for pos, t, ts in (('e1', te, se), ('last', tl, sl)):
                st.update(mode='patch', rows=rows, t=t, src={l: hs[l][0, ts] for l in range(1, NL + 1)})
                out = run(ids, len(rows)); st['mode'] = None
                r['%s_%s' % (s, pos)] = dict(zip(rlab, out))
            if s == 'city_other':
                st.update(mode='rand', t=te, U=U, src={l: hs[l][0, se] for l in RAND_L})
                r['rand_e1'] = run(ids)[0]; st['mode'] = None
        res.append(r)
        if n + 1 in (1, 10, 50) or (n + 1) % 50 == 0 or n + 1 == len(S['targets']):
            log('%d/%d targets (%.2f s/target)' % (n + 1, len(S['targets']), (time.time() - t1) / (n + 1)))
    json.dump(dict(keys=keys, rows=rlab, results=res), open(os.path.join(OUT, 'v235_pilot2.json'), 'w'))

    # 3) summary
    log('\nnone: answer = target capital %.3f | random 24-dim clamp at e1 (8-20): kept %.3f, hit %.3f' % (
        np.mean([r['none'] == r['y'] for r in res]), np.mean([r['rand_e1'] == r['y'] for r in res]),
        np.mean([r['rand_e1'] == r['city_other_y'] for r in res])))
    for s in SOURCES:
        for pos in POS:
            hit = ' '.join('%s %.2f' % (k, np.mean([r['%s_%s' % (s, pos)][k] == r[s + '_y'] for r in res])) for k in rlab)
            kept = ' '.join('%s %.2f' % (k, np.mean([r['%s_%s' % (s, pos)][k] == r['y'] for r in res])) for k in rlab)
            log('%-10s %-4s hit  %s' % (s, pos, hit)); log('%-10s %-4s kept %s' % ('', '', kept))
    log('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
