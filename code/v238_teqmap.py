"""v238 post hoc timing of ' =' (descriptive; no decision rule): after which single layers does the target line's ' ='
alone carry what the position map found (0.058 / 0.054 with every layer swapped)? Same pairs as v238 (v238_pairs.json, exported from
results_v238/relay/v238_relay.json), same model, hook and positions (v236_pilot.positions).

Rows (x''s hidden_states replace x's at the listed positions and layers; ALL = layers 1-27):
  none; relay (the five positions, ALL; replicates v238);
  one_<p>  one position alone, ALL             for p in semi (' ;'), qa (question letter), last ('?'), ta (target
                                                 letter), teq (' =')
  drop_<p> the other four positions, ALL
  relay_L<l>  the five positions after layer l only, l in 8, 12, 16, 20, 22, 23
hit = x''s answer A'; kept = x's answer A. Per row: mean hit with a 97.5% cluster-bootstrap interval over x's family
(seed 2393), descriptive only.

  GPU:  OUT_DIR=... python v238_posmap.py run
  V238_MECH=1 V238_REPO=<tiny model> V238_N=3: local mechanics check (full patch reproduces x' exactly)
"""
import json, os, sys, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v236_relay import boot
MECH = os.environ.get('V238_MECH') == '1'; N = int(os.environ.get('V238_N', '1000'))
RP = ['semi', 'qa', 'last', 'ta', 'teq']; LS = [8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24]
ROWS = ['none', 'one_teq'] + ['teq_L%d' % l for l in LS] + ['relay_L%d' % l for l in (17, 18, 19)]


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v238_teqmap_log.txt'), 'a').write(s + '\n')


def row_spec(P, NL):
    allL = range(1, NL); pos = {p: P[p] for p in RP}; relay = tuple(pos[p] for p in RP)
    assert relay == tuple(range(P['eq'] + 1, P['pre'])), (relay, P)
    spec = {'none': {}, 'one_teq': {l: (pos['teq'],) for l in allL}}
    for l in LS:
        spec['teq_L%d' % l] = {l: (pos['teq'],)}
    for l in (17, 18, 19):
        spec['relay_L%d' % l] = {l: relay}
    return [spec[r] for r in ROWS]


def run():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import v215_items as V5, v234_items as I
    from v236_pilot import positions
    torch.set_grad_enabled(False); t0 = time.time()
    its = I.build(); s = V5.sha16(its); assert s == I.ITEMS_SHA, s
    pairs = json.load(open(os.path.join(HERE, 'v238_pairs.json')))['pairs']
    repo = os.environ.get('V238_REPO', 'Qwen/Qwen2.5-7B'); tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    dev = m.get_input_embeddings().weight.device; layers = m.model.layers; NL = len(layers)
    digit = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(10)]
    st = {'on': False}

    def hook(li):
        hi = li + 1

        def f(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if st['on']:
                for j, sp in enumerate(st['spec']):
                    if hi in sp:
                        ix = list(sp[hi]); h[j, ix] = st['src'][hi][0, ix].to(h.device, h.dtype)
            return out
        return f
    for li in range(NL):
        layers[li].register_forward_hook(hook(li))
    log('loaded in %.0fs | items sha %s | %d pairs | rows %s' % (time.time() - t0, s, len(pairs), ROWS))
    res = []; t1 = time.time(); cnt = {}
    for t, j, mm in pairs:
        if cnt.get(t, 0) >= N:
            continue
        x, x2 = its[j], its[mm]
        assert x['k'] == x2['k'] and x['A'] != x2['A']
        e = tok(x['prompt'] + x['target'], return_offsets_mapping=True); ids = e['input_ids']; P = positions(x, e['offset_mapping'])
        ids2 = tok(x2['prompt'] + x2['target'])['input_ids']; assert len(ids) == len(ids2)
        H = m(input_ids=torch.tensor([ids2], device=dev), output_hidden_states=True)
        if MECH:
            L1 = m(input_ids=torch.tensor([ids], device=dev)).logits[0, P['pre']].float()
            full = {l: tuple(range(len(ids))) for l in range(1, NL)}; st.update(on=True, spec=[full], src=H.hidden_states)
            lg = m(input_ids=torch.tensor([ids], device=dev)).logits[0, P['pre']].float(); st['on'] = False
            L2 = H.logits[0, P['pre']].float(); dp, dx = (lg - L2).abs().max().item(), (L1 - L2).abs().max().item()
            log("mech: max|patched - x'| %.2e, max|x - x'| %.2e" % (dp, dx)); assert dp < 0.05 * dx, (dp, dx)
        st.update(on=True, spec=row_spec(P, NL), src=H.hidden_states)
        out = m(input_ids=torch.tensor([ids] * len(ROWS), device=dev)).logits[:, P['pre'], digit].float().argmax(-1).tolist()
        st['on'] = False
        res.append(dict(type=t, x=j, x2=mm, A=x['A'], P=x2['A'], fam=x['fam'], out=dict(zip(ROWS, out))))
        cnt[t] = cnt.get(t, 0) + 1
        if len(res) in (1, 20, 100) or len(res) % 250 == 0:
            log('%d pairs (%.2f s/pair) | one_teq so far %.3f' % (len(res), (time.time() - t1) / len(res), np.mean([r['out']['one_teq'] == r['P'] for r in res])))
    json.dump(dict(items_sha=s, rows=ROWS, results=res), open(os.path.join(OUT, 'v238_teqmap.json'), 'w'))
    for t in ('E1', 'E2'):
        rs = [r for r in res if r['type'] == t]
        if not rs:
            continue
        cl = np.array([r['fam'] for r in rs])
        log('\n%s (%d pairs): hit A\' [97.5%% CI] (kept A)' % (t, len(rs)))
        for k in ROWS:
            h = np.array([r['out'][k] == r['P'] for r in rs], float)
            d, lo, hi = boot(h, cl, seed=2393)
            log('  %-10s %.3f [%.3f, %.3f] (%.3f)' % (k, d, lo, hi, np.mean([r['out'][k] == r['A'] for r in rs])))
    log('done (%.0fs)' % (time.time() - t0))


if __name__ == '__main__':
    run()
