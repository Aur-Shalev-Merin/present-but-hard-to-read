"""v215 pilot (exploratory; no minimal-pair statistic on the computed value). Kudo et al. (Findings of EACL 2026) task
in their format, Qwen2.5-7B fp16 over two T4s.

Task (their 2-step "Level 3" config, re-implemented; their repo has no licence): two equations, the queried one first,
  a=k+b , b=d1+d2 ; a?     (ops + or -, every value incl. intermediate and answer in 0..9, lowercase letters)
rendered at character level as in their preprocessor ("a = 1 + b , b = 2 + 3 ; a?"), target
  " a = 1 + b , b = 2 + 3 , b = 5 , a = 1 + b , a = 1 + 5 , a = 6\\n" (their simple_backtracking CoT), three
same-level demonstrations before the test source (source + target concatenated, no other text).

Checks:
  1. greedy decoding on N_GEN problems: exact CoT match and final-answer accuracy (their Table 7: ~100% for 7B)
  2. states at the test source's d2 token (end of b's equation), ';' and the last token ('?'), layers 0,4,..,28,
     on N_PROBE distinct problems; seconds per prompt
  3. their standard probe, small: logistic on single prompts (train 80% / test 20%, problems distinct) for b's value,
     a's value and the copies d1, d2 at each position and layer (their pre-CoT b is ~50%, a ~18% with 10,000)
  4. minimal-pair family count: how many (d1, d2, op1) combinations a family with fixed k, op2, letters and demos has
"""
import json, os, sys, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.')
N_GEN = int(os.environ.get('V215_NGEN', '200')); N_PROBE = int(os.environ.get('V215_NPROBE', '2000'))
LAYERS = list(range(0, 29, 4))
LET = 'abcdefghijklmnopqrstuvwxyz'


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v215_pilot_log.txt'), 'a').write(s + '\n')


def op(x, o, y):
    return x + y if o == '+' else x - y


def spaced(s):
    return ' '.join(s.replace(' ', ''))


def render(p):
    """p: dict(a, b, k, o2, d1, o1, d2). Returns (source, target)."""
    a, b, k, o2, d1, o1, d2 = (p[x] for x in ('a', 'b', 'k', 'o2', 'd1', 'o1', 'd2'))
    B = op(d1, o1, d2); A = op(k, o2, B)
    passage = '%s=%d%s%s, %s=%d%s%d' % (a, k, o2, b, b, d1, o1, d2)
    steps = ['%s=%d%s%s' % (a, k, o2, b), '%s=%d%s%d' % (b, d1, o1, d2), '%s=%d' % (b, B), '%s=%d%s%s' % (a, k, o2, b),
             '%s=%d%s%d' % (a, k, o2, B), '%s=%d' % (a, A)]
    return '%s ; %s?' % (spaced(passage), spaced(a)), ' ' + spaced(' , '.join(steps)) + '\n'


def valid(p):
    B = op(p['d1'], p['o1'], p['d2'])
    return 0 <= B <= 9 and 0 <= op(p['k'], p['o2'], B) <= 9


def sample(rng):
    while True:
        a, b = rng.choice(list(LET), 2, replace=False)
        p = dict(a=str(a), b=str(b), k=int(rng.integers(10)), o2=str(rng.choice(['+', '-'])), d1=int(rng.integers(10)),
                 o1=str(rng.choice(['+', '-'])), d2=int(rng.integers(10)))
        if valid(p):
            return p


def key(p):
    return (p['k'], p['o2'], p['d1'], p['o1'], p['d2'])


def build(rng, n, used):
    """n distinct test problems (by their full text, letters included; only 1,210 arithmetic combinations exist, so
    the probe split is made by arithmetic key later), each with three demonstrations."""
    out = []
    while len(out) < n:
        p = sample(rng)
        if render(p)[0] in used:
            continue
        used.add(render(p)[0]); demos = [sample(rng) for _ in range(3)]
        src, tgt = render(p)
        out.append(dict(p=p, prompt=''.join(''.join(render(q)) for q in demos) + src, target=tgt, src=src))
    return out


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False)
    t0 = time.time(); repo = 'Qwen/Qwen2.5-7B'
    tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    dev = m.get_input_embeddings().weight.device
    log('v215 pilot | %s | %.0fs to load | layers %s' % (repo, time.time() - t0, LAYERS))
    rng = np.random.default_rng(215); used = set()
    ex = build(rng, 1, set())[0]; log('example prompt:\n' + ex['prompt'] + '\n--- expected target:' + repr(ex['target']))
    ids = tok(ex['prompt']).input_ids; log('tokens: %d | last 16: %s' % (len(ids), [tok.decode([i]) for i in ids[-16:]]))

    # 1. greedy decoding (skipped with V215_NGEN=0; first run: exact CoT 0.995, answer 1.000, 2.79 s/problem)
    gen = build(rng, N_GEN, used); exact = ans = 0; t1 = time.time(); bad = []
    for g in gen:
        x = tok(g['prompt'], return_tensors='pt').to(dev)
        o = m.generate(**x, max_new_tokens=48, do_sample=False, eos_token_id=198, pad_token_id=tok.eos_token_id)
        txt = tok.decode(o[0, x.input_ids.shape[1]:])
        exact += txt.split('\n')[0] == g['target'].rstrip('\n')
        want = g['target'].rstrip('\n').split('=')[-1].strip(); got = txt.split('\n')[0].split('=')[-1].strip()
        ans += got == want
        if txt.split('\n')[0] != g['target'].rstrip('\n') and len(bad) < 5:
            bad.append((g['target'], txt))
    if N_GEN: log('1. greedy on %d problems: exact CoT %.3f, final answer %.3f (%.2f s/problem)' % (
        N_GEN, exact / N_GEN, ans / N_GEN, (time.time() - t1) / N_GEN))
    for b in bad:
        log('   mismatch: want %r got %r' % b)

    # 2. states at three positions of the test source
    pr = build(rng, N_PROBE, used); P = ['d2', 'semi', 'last']
    X = np.zeros((len(pr), len(P), len(LAYERS), m.config.hidden_size), np.float16); t2 = time.time()
    for i, g in enumerate(pr):
        enc = tok(g['prompt'], return_offsets_mapping=True); off = enc['offset_mapping']
        base = len(g['prompt']) - len(g['src']); s = g['src']
        cpos = {'d2': base + s.index(';') - 2, 'semi': base + s.index(';'), 'last': len(g['prompt']) - 1}
        tix = {k: next(t for t, (u, v) in enumerate(off) if u <= c < v) for k, c in cpos.items()}
        if i == 0:
            log('positions: %s' % {k: (t, tok.decode([enc['input_ids'][t]])) for k, t in tix.items()})
        hs = m(input_ids=torch.tensor([enc['input_ids']], device=dev), output_hidden_states=True).hidden_states
        for a_, k in enumerate(P):
            for b_, l in enumerate(LAYERS):
                X[i, a_, b_] = hs[l][0, tix[k]].float().cpu().numpy()
        if i in (0, 99, len(pr) - 1):
            log('   %d/%d prompts (%.3f s/prompt)' % (i + 1, len(pr), (time.time() - t2) / (i + 1)))
    np.save(os.path.join(OUT, 'v215_pilot_states.npy'), X)
    json.dump([g['p'] for g in pr], open(os.path.join(OUT, 'v215_pilot_items.json'), 'w'))
    if os.environ.get('V215_PROBE_ON_GPU') != '1':        # probes run in a CPU kernel (v215_pilot_probe.py)
        log('states saved (%.0fs); probes run separately' % (time.time() - t0)); return

    # 3. their standard probe, small
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    lab = {'b': [op(g['p']['d1'], g['p']['o1'], g['p']['d2']) for g in pr],
           'a': [op(g['p']['k'], g['p']['o2'], op(g['p']['d1'], g['p']['o1'], g['p']['d2'])) for g in pr],
           'd1': [g['p']['d1'] for g in pr], 'd2': [g['p']['d2'] for g in pr]}
    ntr = int(0.8 * len(pr))
    for name, y in lab.items():
        y = np.array(y); row = []
        for a_, k in enumerate(P):
            best = 0
            for b_, l in enumerate(LAYERS):
                Z = X[:, a_, b_].astype(np.float32); sc = StandardScaler().fit(Z[:ntr])
                acc = (LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(Z[:ntr]), y[:ntr])
                       .predict(sc.transform(Z[ntr:])) == y[ntr:]).mean()
                best = max(best, acc)
            row.append('%s %.3f' % (k, best))
        log('3. standard probe (%d train / %d test, best layer): %-2s %s | majority %.3f' % (
            ntr, len(pr) - ntr, name, ' | '.join(row), np.bincount(y[ntr:]).max() / (len(pr) - ntr)))

    # 4. minimal-pair family size
    fam = [(d1, o1, d2) for d1 in range(10) for o1 in '+-' for d2 in range(10)]
    for k in (0, 3, 5, 9):
        for o2 in '+-':
            n = sum(valid(dict(k=k, o2=o2, d1=d1, o1=o1, d2=d2)) for d1, o1, d2 in fam)
            log('4. family k=%d o2=%s: %d valid (d1, op1, d2) combinations' % (k, o2, n))
    log('done (%.0fs)' % (time.time() - t0))


if __name__ == '__main__':
    main()
