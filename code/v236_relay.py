"""v236 relay test (docs/prereg_v236_kudo_relay.md): without a chain, is b (or its operands) carried by any position
between b's equation and the answer, in a form the answer uses? Whole-state patches only; no learned subspace.

Items: v234's frozen items (v234_items.build(), ITEMS_SHA ac4121c22f88fe31), Kudo et al.'s Implicit format without a
chain ("a=k+b, b=d1+d2 ; a?" -> " a = A"). Qwen2.5-7B base, fp16.
Pairs share family and version and have equal answers A (k varies), both answered correctly (argmax over the ten
digits at pre):
  D1  same d2, different d1      operand token = d1, other operand token = eq (d2)
  D2  same d1, different d2      operand token = eq (d2), other operand token = d1
P = k o2 b' (x's k with x''s b): the answer a used b' predicts; P != A always. hit = P, kept = A.
Pair order: v236_pilot's candidate order (seed 2360); the candidates the pilot tried (D1 445, D2 476) are skipped, and so
is any later candidate that is the same unordered pair, so no pilot pair is reused. Up to N_PER = 1,000 pairs per type.

Positions: relay = every token strictly between eq (b's last digit, d2) and pre (the token before the answer digit):
' ;', the question letter, '?', the target letter, ' =' (5 tokens in every item). pre is excluded (the answer position).
Rows (one batched pass per pair; x''s hidden_states l = output of decoder layer l - 1 replace x's at the listed
positions and layers; ALL = layers 1-27):
  none              no patch
  relay             relay positions, ALL                       (primary)
  op                the operand token, ALL                     (positive control / gate)
  oth               the other operand token, ALL               (descriptive)
  op_L, pre_L       the operand token / pre, single layer L in 16, 18, 20, 21, 22, 23, 24   (descriptive)

Analysis (per type): stat = hit(relay) - hit(none), cluster bootstrap over x's family, 10,000 resamples, seed 2363,
97.5% interval. UNSCORABLE if hit(op) < 0.30; else NOT-HELD if upper < 0.02; HELD if stat >= 0.10 with lower > 0;
UNCLEAR otherwise. Verdict: UNSCORABLE if both types are; HELD if any scorable type is HELD; NOT-HELD if every scorable
type is NOT-HELD; UNCLEAR otherwise.

  GPU:  OUT_DIR=... python v236_relay.py run
  V236_MECH=1 (local, tiny random model via V236_REPO): no correctness filter; checks that patching every position
        at every layer reproduces x''s logits.
  CPU:  python v236_relay.py analyse <dir with v236_relay.json>
        python v236_relay.py --selftest
"""
import glob, json, os, sys, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
N_PER = int(os.environ.get('V236_N', '1000')); TYPES = ['D1', 'D2']; PILOT_TRIED = {'D1': 445, 'D2': 476}
SINGLE = [16, 18, 20, 21, 22, 23, 24]
ROWS = ['none', 'relay', 'op', 'oth'] + ['op_L%d' % l for l in SINGLE] + ['pre_L%d' % l for l in SINGLE]
OPTOK = {'D1': ('d1', 'eq'), 'D2': ('eq', 'd1')}
MECH = os.environ.get('V236_MECH') == '1'  # local mechanics check on a tiny random model only


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v236_relay_log.txt'), 'a').write(s + '\n')


def pair_order(cand, t):
    """Candidates after the ones the pilot tried, without any unordered pair the pilot tried."""
    seen = {frozenset(c) for c in cand[t][:PILOT_TRIED[t]]}
    return [c for c in cand[t][PILOT_TRIED[t]:] if frozenset(c) not in seen]


def row_spec(t, P, NL):
    allL = list(range(1, NL)); op, oth = (P[OPTOK[t][0]],), (P[OPTOK[t][1]],); relay = tuple(range(P['eq'] + 1, P['pre']))
    spec = {'none': {}, 'relay': {l: relay for l in allL}, 'op': {l: op for l in allL}, 'oth': {l: oth for l in allL}}
    for l in SINGLE:
        spec['op_L%d' % l] = {l: op}; spec['pre_L%d' % l] = {l: (P['pre'],)}
    return [spec[r] for r in ROWS], relay


def run():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import v215_items as V5, v234_items as I
    from v215_pilot import op
    from v236_pilot import candidates, positions
    torch.set_grad_enabled(False); t0 = time.time()
    its = I.build(); s = V5.sha16(its); assert s == I.ITEMS_SHA, s
    repo = os.environ.get('V236_REPO', 'Qwen/Qwen2.5-7B'); tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
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
    cand = candidates(its, np.random.default_rng(2360)); res = []; t1 = time.time()
    for t in TYPES:
        order = pair_order(cand, t); n_ok = n_tried = 0
        log('%s: %d candidates after skipping the pilot\'s' % (t, len(order)))
        for j, mm in order:
            if n_ok >= N_PER:
                break
            x, x2 = its[j], its[mm]; n_tried += 1
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
            spec, relay = row_spec(t, P, NL); assert len(relay) == 5, relay
            st.update(on=True, spec=spec, src=H.hidden_states)
            out = m(input_ids=torch.tensor([ids] * len(ROWS), device=dev)).logits[:, P['pre'], digit].float().argmax(-1).tolist()
            st['on'] = False
            res.append(dict(type=t, x=j, x2=mm, A=x['A'], P=op(x['k'], x['o2'], x2['B']), fam=x['fam'], out=dict(zip(ROWS, out))))
            n_ok += 1
            if n_ok in (1, 20, 100) or n_ok % 200 == 0:
                rs = [r for r in res if r['type'] == t]
                log('%s: %d pairs (%d tried; %.2f s/pair) | hit so far: %s' % (t, n_ok, n_tried, (time.time() - t1) / len(res),
                    ' '.join('%s %.3f' % (k, np.mean([r['out'][k] == r['P'] for r in rs])) for k in ('none', 'relay', 'op', 'oth'))))
        log('%s: %d pairs from %d candidates tried' % (t, n_ok, n_tried))
    json.dump(dict(items_sha=s, rows=ROWS, results=res), open(os.path.join(OUT, 'v236_relay.json'), 'w'))
    log('done (%.0fs)' % (time.time() - t0))
    analyse(OUT, say=log)


def boot(d, cl, seed=2363, n=10000):
    u = np.unique(cl); ix = np.searchsorted(u, cl); S = np.zeros(len(u)); N = np.zeros(len(u))
    np.add.at(S, ix, d); np.add.at(N, ix, 1); rng = np.random.default_rng(seed); bs = np.empty(n)
    for i in range(n):
        q = rng.integers(0, len(u), len(u)); bs[i] = S[q].sum() / N[q].sum()
    lo, hi = np.percentile(bs, (1.25, 98.75)); return float(d.mean()), float(lo), float(hi)


def label(gate, d, lo, hi):
    if gate < 0.30:
        return 'UNSCORABLE'
    return 'HELD' if d >= 0.10 and lo > 0 else 'NOT-HELD' if hi < 0.02 else 'UNCLEAR'


def verdict(L):
    sc = [v for v in L.values() if v != 'UNSCORABLE']
    if not sc:
        return 'UNSCORABLE'
    if 'HELD' in sc:
        return 'HELD'
    return 'NOT-HELD' if all(v == 'NOT-HELD' for v in sc) else 'UNCLEAR'


def analyse(root, say=print):
    R = json.load(open(glob.glob(os.path.join(root, '**', 'v236_relay.json'), recursive=True)[0]))
    out = {}; L = {}
    for t in TYPES:
        rs = [r for r in R['results'] if r['type'] == t]; cl = np.array([r['fam'] for r in rs])
        hit = {k: np.array([r['out'][k] == r['P'] for r in rs], float) for k in ROWS}
        kept = {k: float(np.mean([r['out'][k] == r['A'] for r in rs])) for k in ROWS}
        d, lo, hi = boot(hit['relay'] - hit['none'], cl); L[t] = label(hit['op'].mean(), d, lo, hi)
        out[t] = dict(n=len(rs), families=int(len(np.unique(cl))), hit={k: float(hit[k].mean()) for k in ROWS}, kept=kept,
                      stat=[d, lo, hi], label=L[t])
        say('\n%s (%d pairs, %d families): relay - none %+.3f [%+.3f, %+.3f] -> %s (gate: hit(op) %.3f)'
            % (t, len(rs), out[t]['families'], d, lo, hi, L[t], hit['op'].mean()))
        say('  ' + '  '.join('%s %.3f(%.3f)' % (k, hit[k].mean(), kept[k]) for k in ROWS[:4]) + '   [hit P (kept A)]')
        say('  op  single: ' + '  '.join('L%d %.3f(%.3f)' % (l, hit['op_L%d' % l].mean(), kept['op_L%d' % l]) for l in SINGLE))
        say('  pre single: ' + '  '.join('L%d %.3f(%.3f)' % (l, hit['pre_L%d' % l].mean(), kept['pre_L%d' % l]) for l in SINGLE))
    out['labels'] = L; out['verdict'] = verdict(L)
    say('labels %s -> verdict %s' % (L, out['verdict']))
    json.dump(out, open(os.path.join(OUT, 'v236_result.json'), 'w'), indent=1)
    return out


def selftest():
    assert label(0.9, 0.12, 0.05, 0.2) == 'HELD' and label(0.9, 0.0, 0.0, 0.015) == 'NOT-HELD'
    assert label(0.9, 0.05, 0.02, 0.08) == 'UNCLEAR' and label(0.9, 0.12, -0.01, 0.2) == 'UNCLEAR'
    assert label(0.29, 0.0, 0.0, 0.0) == 'UNSCORABLE'
    assert verdict({'D1': 'NOT-HELD', 'D2': 'NOT-HELD'}) == 'NOT-HELD'
    assert verdict({'D1': 'NOT-HELD', 'D2': 'UNSCORABLE'}) == 'NOT-HELD'
    assert verdict({'D1': 'NOT-HELD', 'D2': 'HELD'}) == 'HELD' and verdict({'D1': 'UNCLEAR', 'D2': 'HELD'}) == 'HELD'
    assert verdict({'D1': 'NOT-HELD', 'D2': 'UNCLEAR'}) == 'UNCLEAR'
    assert verdict({'D1': 'UNSCORABLE', 'D2': 'UNSCORABLE'}) == 'UNSCORABLE'
    d = np.r_[np.zeros(95), np.ones(5)]; cl = np.repeat(np.arange(20), 5); m, lo, hi = boot(d, cl)
    assert abs(m - 0.05) < 1e-12 and lo <= m <= hi
    cand = {'D1': [(0, 1), (2, 3)] + [(1, 0), (4, 5), (3, 2), (6, 7)], 'D2': []}
    old = PILOT_TRIED['D1']; PILOT_TRIED['D1'] = 2
    assert pair_order(cand, 'D1') == [(4, 5), (6, 7)]; PILOT_TRIED['D1'] = old
    P = dict(d1=10, o1=11, sp=12, eq=13, semi=14, qa=15, last=16, ta=17, teq=18, pre=19)
    sp, relay = row_spec('D1', P, 28); S = dict(zip(ROWS, sp))
    assert relay == (14, 15, 16, 17, 18) and S['op'][5] == (10,) and S['oth'][27] == (13,) and 28 not in S['relay']
    assert sorted(S['relay']) == list(range(1, 28)) and S['pre_L20'] == {20: (19,)} and S['none'] == {}
    assert dict(zip(ROWS, row_spec('D2', P, 28)[0]))['op'][1] == (13,)
    print('selftest ok')


if __name__ == '__main__':
    a = sys.argv[1]
    selftest() if a == '--selftest' else run() if a == 'run' else analyse(sys.argv[2])
