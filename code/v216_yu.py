"""v216 Yu pilot (exploratory; no minimal-pair statistic). Yu (arXiv 2411.15862) "Do LLMs Really Think Step-by-step In
Implicit Reasoning?" re-implemented from their repository (generate_arithmetic.chain_num, get_hidden_states.py):
  chain of length L: values r_0..r_{L-1} uniform in 0..19, addends r_i - r_{i-1}; equations, one per line,
  "C = 8 ;" / "B = C - 5 ;" / "A = B + 2 ;" (variables from the end of A..Z), then "\\n\\nA=?" (their probing format;
  chat_format=False), wrapped in the Instruct chat template as the user turn, then "A=" appended. Answer = r_{L-1}.
Model: Qwen2.5-7B-Instruct fp16 over two T4s (they used Qwen2.5-72B-Instruct, 4-bit).

  gpu    V216_L=3,5 (chain lengths), V216_N per length: greedy answer accuracy (up to 4 new tokens, first integer),
         and the last prompt token's state at hidden_states 0, 2, ..., 28 -> v216yu_L<L>_states.npy / _meta.json
  probe  CPU: their probe (linear map on the state, here logistic regression on standardised states, C = 0.01), one
         per (step, layer), random 80/20 split (their train_test_split, test_size 0.2); labels r_i (0..19)
"""
import glob, json, os, re, sys, time
import numpy as np

OUT = os.environ.get('OUT_DIR', '.'); LAYERS = list(range(0, 29, 2)); VARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v216yu_log.txt'), 'a').write(s + '\n')


def chain(rng, L, lo=0, hi=19):
    r = [int(x) for x in rng.integers(lo, hi + 1, L)]
    add = [r[0]] + [r[i] - r[i - 1] for i in range(1, L)]
    s = lambda a: ' + %d' % a if a > 0 else ' - %d' % -a          # their addend_str (0 -> " - 0")
    eq = ['%s = %d ;' % (VARS[L - 1], add[0])] + ['%s = %s%s ;' % (VARS[L - i - 1], VARS[L - i], s(add[i])) for i in range(1, L)]
    return '\n'.join(eq) + '\n\nA=?', r, add


def gpu():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False); t0 = time.time(); repo = 'Qwen/Qwen2.5-7B-Instruct'
    tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map='auto').eval()
    dev = m.get_input_embeddings().weight.device
    Ls = [int(x) for x in os.environ.get('V216_L', '3,5').split(',')]; N = int(os.environ.get('V216_N', '2000'))
    log('v216 yu pilot | %s | %.0fs to load | lengths %s, %d each' % (repo, time.time() - t0, Ls, N))
    rng = np.random.default_rng(2160)
    for L in Ls:
        X = np.zeros((N, len(LAYERS), m.config.hidden_size), np.float16); meta = []; t1 = time.time()
        for n in range(N):
            q, r, add = chain(rng, L)
            prompt = tok.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True) + 'A='
            ids = tok(prompt, return_tensors='pt').input_ids.to(dev)
            out = m.generate(ids, max_new_tokens=4, do_sample=False, output_hidden_states=True, return_dict_in_generate=True,
                             pad_token_id=tok.eos_token_id)
            hs = out.hidden_states[0]                     # the prompt pass
            for j, l in enumerate(LAYERS):
                X[n, j] = hs[l][0, -1].float().cpu().numpy()
            txt = tok.decode(out.sequences[0, ids.shape[1]:]); mm = re.match(r'\s*(-?\d+)', txt)
            pred = int(mm.group(1)) if mm else None
            meta.append(dict(r=r, add=add, pred=pred, ok=pred == r[-1]))
            if n == 0:
                log('L=%d example prompt:\n%s\n--- model: %r (answer %d)' % (L, prompt, txt, r[-1]))
            if n + 1 in (100, 500) or (n + 1) % 1000 == 0:
                log('L=%d %d/%d (%.3f s/problem) | accuracy so far %.3f' % (L, n + 1, N, (time.time() - t1) / (n + 1),
                                                                           np.mean([x['ok'] for x in meta])))
        np.save(os.path.join(OUT, 'v216yu_L%d_states.npy' % L), X)
        json.dump(dict(L=L, layers=LAYERS, items=meta), open(os.path.join(OUT, 'v216yu_L%d_meta.json' % L), 'w'))
        log('L=%d: answer accuracy %.3f (their Qwen2.5-72B: %s)' % (L, np.mean([x['ok'] for x in meta]),
                                                                     {3: '99.8%', 4: '83.2%', 5: '65.2%'}.get(L, '?')))
    log('done (%.0fs)' % (time.time() - t0))


G = {}


def init(d):
    G.update(d)


def pjob(args):
    L, step, j = args
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y, tr = G['X%d' % L], G['y%d' % L][:, step], G['tr%d' % L]
    Z = np.asarray(X[:, j], np.float32); sc = StandardScaler().fit(Z[tr])
    m = LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(Z[tr]), y[tr])
    return args, float((m.predict(sc.transform(Z[~tr])) == y[~tr]).mean())


def probe(root):
    from multiprocessing import Pool
    d = {}; Ls = []
    for f in sorted(glob.glob(os.path.join(root, '**', 'v216yu_L*_states.npy'), recursive=True)):
        mt = json.load(open(f.replace('_states.npy', '_meta.json'))); L = mt['L']; Ls.append(L)
        d['X%d' % L] = np.load(f); d['y%d' % L] = np.array([x['r'] for x in mt['items']])
        d['tr%d' % L] = np.random.default_rng(1).permutation(len(mt['items'])) >= int(0.2 * len(mt['items']))
        print('L=%d: %d problems, answer accuracy %.3f' % (L, len(mt['items']), np.mean([x['ok'] for x in mt['items']])), flush=True)
    jobs = [(L, s, j) for L in Ls for s in range(L) for j in range(len(LAYERS))]
    with Pool(4, initializer=init, initargs=(d,)) as pool:
        res = dict(pool.imap_unordered(pjob, jobs, chunksize=1))
    for L in Ls:
        print('--- L=%d: probe accuracy by layer (hidden_states %s), chance about 0.05 ---' % (L, LAYERS), flush=True)
        for s in range(L):
            row = [res[(L, s, j)] for j in range(len(LAYERS))]
            print('  value %d (%s)%s: %s | best %.3f' % (s, VARS[L - 1 - s], ' = answer' if s == L - 1 else
                  ' = first, copied' if s == 0 else '', ' '.join('%.2f' % v for v in row), max(row)), flush=True)


if __name__ == '__main__':
    {'gpu': gpu, 'probe': lambda: probe(sys.argv[2])}[sys.argv[1]]()
