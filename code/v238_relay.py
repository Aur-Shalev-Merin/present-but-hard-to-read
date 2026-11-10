"""v238 relay test with different answers (docs/prereg_v238_kudo_relay_diff.md): without a chain, is anything that
depends on b (b itself, an operand, or the finished answer) carried by the positions between b's equation and the
answer position, in a form the answer uses? Closes v236's blind spot (equal-answer pairs cannot see a carried answer).
Whole-state patches only; no learned subspace.

Items: v234's frozen items (ITEMS_SHA ac4121c22f88fe31). Pairs share family, version and k, both answered correctly
(argmax over the ten digits at pre), and differ in one operand, so b and the answer differ:
  E1  same k and d2, different d1     operand token = d1, other operand token = eq (d2, the same token in both)
  E2  same k and d1, different d2     operand token = eq (d2), other operand token = d1
hit = x''s answer A' (what a carried b', operand or finished answer predicts, since k is shared); kept = x's A.
Pair order: candidates shuffled with seed 2380; up to N_PER = 1,000 pairs per type.

Positions, rows, hook and statistic as v236 (rows, row_spec, label, verdict and boot imported from the frozen
v236_relay.py): relay = the five tokens strictly between eq and pre, after every layer 1-27 (primary); op = the operand
token after every layer (gate); oth; op_L and pre_L after single layers 16, 18, 20, 21, 22, 23, 24 (descriptive).
Statistic per type: hit(relay) - hit(none), cluster bootstrap over x's family, 10,000 resamples, seed 2383, 97.5%.
UNSCORABLE if hit(op) < 0.30; NOT-HELD if upper < 0.02; HELD if >= 0.10 with lower > 0; UNCLEAR otherwise.
Verdict as v236.

  GPU:  OUT_DIR=... python v238_relay.py run
  CPU:  python v238_relay.py analyse <dir with v238_relay.json>
        python v238_relay.py --selftest
  V238_MECH=1 (local, tiny random model via V238_REPO): no correctness filter; checks that patching every position
        at every layer reproduces x''s logits.
"""
import glob, json, os, sys, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v236_relay import ROWS, SINGLE, row_spec, label, verdict, boot   # frozen in v236

N_PER = int(os.environ.get('V238_N', '1000')); TYPES = ['E1', 'E2']; OPTOK = {'E1': 'D1', 'E2': 'D2'}
MECH = os.environ.get('V238_MECH') == '1'


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v238_relay_log.txt'), 'a').write(s + '\n')


def candidates(its, rng):
    """E1: same (fam, ver, k, d2), different d1; E2: same (fam, ver, k, d1), different d2. Answers differ."""
    out = {}
    for t, same, diff in (('E1', 'd2', 'd1'), ('E2', 'd1', 'd2')):
        key = {}
        for j, i in enumerate(its):
            key.setdefault((i['fam'], i['ver'], i['k'], i[same]), []).append(j)
        c = [(j, m) for g in key.values() for j in g for m in g if its[j][diff] != its[m][diff]]
        out[t] = [c[q] for q in rng.permutation(len(c))]
    return out


def run():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import v215_items as V5, v234_items as I
    from v236_pilot import positions
    torch.set_grad_enabled(False); t0 = time.time()
    its = I.build(); s = V5.sha16(its); assert s == I.ITEMS_SHA, s
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

    def enc(x):
        e = tok(x['prompt'] + x['target'], return_offsets_mapping=True); return e['input_ids'], positions(x, e['offset_mapping'])

    log('loaded in %.0fs | items sha %s | layers %d | rows %s' % (time.time() - t0, s, NL, ROWS))
    cand = candidates(its, np.random.default_rng(2380)); res = []; t1 = time.time()
    for t in TYPES:
        n_ok = n_tried = 0
        log('%s: %d candidates' % (t, len(cand[t])))
        for j, mm in cand[t]:
            if n_ok >= N_PER:
                break
            x, x2 = its[j], its[mm]; n_tried += 1
            assert x['k'] == x2['k'] and x['A'] != x2['A']
            ids, P = enc(x); ids2, P2 = enc(x2); assert len(ids) == len(ids2) and P == P2
            H = m(input_ids=torch.tensor([ids2], device=dev), output_hidden_states=True)
            ok2 = int(H.logits[0, P['pre'], digit].float().argmax()) == x2['A']
            L1 = m(input_ids=torch.tensor([ids], device=dev)).logits[0, P['pre']].float(); ok1 = int(L1[digit].argmax()) == x['A']
            if MECH:
                full = {l: tuple(range(len(ids))) for l in range(1, NL)}; st.update(on=True, spec=[full], src=H.hidden_states)
                lg = m(input_ids=torch.tensor([ids], device=dev)).logits[0, P['pre']].float(); st['on'] = False
                L2 = H.logits[0, P['pre']].float(); dp, dx = (lg - L2).abs().max().item(), (L1 - L2).abs().max().item()
                log("mech: max|patched - x'| %.2e, max|x - x'| %.2e" % (dp, dx)); assert dp < 0.05 * dx, (dp, dx)
            elif not (ok1 and ok2):
                continue
            spec, relay = row_spec(OPTOK[t], P, NL); assert len(relay) == 5, relay
            st.update(on=True, spec=spec, src=H.hidden_states)
            out = m(input_ids=torch.tensor([ids] * len(ROWS), device=dev)).logits[:, P['pre'], digit].float().argmax(-1).tolist()
            st['on'] = False
            res.append(dict(type=t, x=j, x2=mm, A=x['A'], P=x2['A'], fam=x['fam'], out=dict(zip(ROWS, out))))
            n_ok += 1
            if n_ok in (1, 20, 100) or n_ok % 200 == 0:
                rs = [r for r in res if r['type'] == t]
                log('%s: %d pairs (%d tried; %.2f s/pair) | hit so far: %s' % (t, n_ok, n_tried, (time.time() - t1) / len(res),
                    ' '.join('%s %.3f' % (k, np.mean([r['out'][k] == r['P'] for r in rs])) for k in ('none', 'relay', 'op', 'oth'))))
        log('%s: %d pairs from %d candidates tried' % (t, n_ok, n_tried))
    json.dump(dict(items_sha=s, rows=ROWS, results=res), open(os.path.join(OUT, 'v238_relay.json'), 'w'))
    log('done (%.0fs)' % (time.time() - t0))
    analyse(OUT, say=log)


def analyse(root, say=print):
    R = json.load(open(glob.glob(os.path.join(root, '**', 'v238_relay.json'), recursive=True)[0]))
    out = {}; L = {}
    for t in TYPES:
        rs = [r for r in R['results'] if r['type'] == t]; cl = np.array([r['fam'] for r in rs])
        hit = {k: np.array([r['out'][k] == r['P'] for r in rs], float) for k in ROWS}
        kept = {k: float(np.mean([r['out'][k] == r['A'] for r in rs])) for k in ROWS}
        d, lo, hi = boot(hit['relay'] - hit['none'], cl, seed=2383); L[t] = label(hit['op'].mean(), d, lo, hi)
        out[t] = dict(n=len(rs), families=int(len(np.unique(cl))), hit={k: float(hit[k].mean()) for k in ROWS}, kept=kept,
                      stat=[d, lo, hi], label=L[t])
        say('\n%s (%d pairs, %d families): relay - none %+.3f [%+.3f, %+.3f] -> %s (gate: hit(op) %.3f)'
            % (t, len(rs), out[t]['families'], d, lo, hi, L[t], hit['op'].mean()))
        say('  ' + '  '.join('%s %.3f(%.3f)' % (k, hit[k].mean(), kept[k]) for k in ROWS[:4]) + "   [hit A' (kept A)]")
        say('  op  single: ' + '  '.join('L%d %.3f(%.3f)' % (l, hit['op_L%d' % l].mean(), kept['op_L%d' % l]) for l in SINGLE))
        say('  pre single: ' + '  '.join('L%d %.3f(%.3f)' % (l, hit['pre_L%d' % l].mean(), kept['pre_L%d' % l]) for l in SINGLE))
    out['labels'] = L; out['verdict'] = verdict(L)
    say('labels %s -> verdict %s' % (L, out['verdict']))
    json.dump(out, open(os.path.join(OUT, 'v238_result.json'), 'w'), indent=1)
    return out


def selftest():
    import v234_items as I, v215_items as V5
    its = I.build(); assert V5.sha16(its) == I.ITEMS_SHA
    c = candidates(its, np.random.default_rng(2380))
    for t, same, diff in (('E1', 'd2', 'd1'), ('E2', 'd1', 'd2')):
        assert len(c[t]) > 5000, (t, len(c[t]))
        for j, m in c[t][:3000]:
            x, y = its[j], its[m]
            assert x['fam'] == y['fam'] and x['ver'] == y['ver'] and x['k'] == y['k'] and x[same] == y[same]
            assert x[diff] != y[diff] and x['A'] != y['A'] and x['B'] != y['B']
    P = dict(d1=10, o1=11, sp=12, eq=13, semi=14, qa=15, last=16, ta=17, teq=18, pre=19)
    S = dict(zip(ROWS, row_spec(OPTOK['E1'], P, 28)[0])); assert S['op'][1] == (10,) and S['oth'][1] == (13,)
    S = dict(zip(ROWS, row_spec(OPTOK['E2'], P, 28)[0])); assert S['op'][1] == (13,) and S['oth'][1] == (10,)
    assert label(0.9, 0.0, 0.0, 0.015) == 'NOT-HELD' and label(0.2, 0.0, 0.0, 0.0) == 'UNSCORABLE'
    print('selftest ok | candidates E1 %d, E2 %d' % (len(c['E1']), len(c['E2'])))


if __name__ == '__main__':
    a = sys.argv[1]
    selftest() if a == '--selftest' else run() if a == 'run' else analyse(sys.argv[2])
