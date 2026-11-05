"""v216 post hoc, descriptive (no label; docs/prereg_v216_answer_only.md section 10): Kudo et al.'s standard probe in
their Implicit (answer-only) format, as v215 Part C. Items: v215's replication set (same problems, demonstrations and
held-out b equations; seed 2156, test triples seed 2154) rendered with v216's implicit renderer. States from v216's
extractor (eq / pre / last x hidden_states 0, 4, ..., 28; correct = the answer digit is the argmax at pre).
Probe: v215_analyze.cjob's settings (standardised states, logistic C = 0.01), train 10,000 / test 2,000, b's and a's
values, each position and layer; test accuracy also split by whether the model answers the test problem correctly.
Their Table 6 (Qwen2.5-7B, Implicit): b 69.1%, a 40.8%.

  GPU:  OUT_DIR=... python v216_theirprobe.py gpu          -> v216_rep_states.npy, v216_rep_meta.json
  CPU:  python v216_theirprobe.py probe <dir>             -> v216_theirprobe.json
        python v216_theirprobe.py --check                 -> same problems as v215's replication set
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

import v215_items as V5
import v216_items as I
from v215_pilot import render, sample

OUT = os.environ.get('OUT_DIR', '.'); LAYERS = list(range(0, 29, 4)); POS = ['eq', 'pre', 'last']
THEIRS = {'b': '69.1%', 'a': '40.8%'}
G = {}


def build_rep():
    """v215_items.replication with v216's implicit item (same random draws, so the same problems)."""
    rng = np.random.default_rng(2156)
    triples = sorted({(d1, o1, d2) for d1 in range(10) for o1 in '+-' for d2 in range(10)
                      if 0 <= V5.op(d1, o1, d2) <= 9})
    test = set(triples[i] for i in np.random.default_rng(2154).permutation(len(triples))[:len(triples) // 5])
    out, seen = [], set()
    for split, n in (('train', V5.N_REP_TRAIN), ('test', V5.N_REP_TEST)):
        c = 0
        while c < n:
            p = sample(rng)
            if ((p['d1'], p['o1'], p['d2']) in test) != (split == 'test') or render(p)[0] in seen:
                continue
            seen.add(render(p)[0]); out.append(I.item(p, [sample(rng) for _ in range(3)], split=split)); c += 1
    return out


def gpu():
    import v216_extract as E
    I.build = build_rep; I.ITEMS_SHA = ''                 # the extractor's loop on these items; hash logged instead
    E.main()
    for f in ('states.npy', 'meta.json'):
        os.replace(os.path.join(OUT, 'v216_fam_' + f), os.path.join(OUT, 'v216_rep_' + f))


def init(d):
    G.update(d)


def job(args):
    lab, pi, li = args
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y, tr = G['X'], G[lab], G['tr']
    Z = np.asarray(X[:, pi, li], np.float32); sc = StandardScaler().fit(Z[tr])
    m = LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(Z[tr]), y[tr])
    return args, (m.predict(sc.transform(Z[~tr])) == y[~tr]).astype(int).tolist()


def probe(root, procs=4, say=print):
    its = build_rep(); s = V5.sha16(its)
    f = glob.glob(os.path.join(root, '**', 'v216_rep_states.npy'), recursive=True)[0]
    X = np.load(f, mmap_mode='r'); m = json.load(open(f.replace('_states.npy', '_meta.json')))
    assert m['items_sha'] == s and len(m['items']) == len(its), 'items differ from the extraction'
    tr = np.array([i['split'] == 'train' for i in its]); ok = np.array([x['correct'] for x in m['items']])[~tr]
    say('items sha %s | train %d / test %d (no shared b equation) | model answers correctly: train %.3f, test %.3f' % (
        s, tr.sum(), (~tr).sum(), np.mean([x['correct'] for x in m['items']][:tr.sum()]), ok.mean()))
    d = dict(X=X, tr=tr, b=np.array([i['B'] for i in its]), a=np.array([i['A'] for i in its]))
    jobs = [(lab, pi, li) for lab in ('b', 'a') for pi in range(len(POS)) for li in range(len(LAYERS))]
    with Pool(procs, initializer=init, initargs=(d,)) as pool:
        res = {k: np.array(v) for k, v in pool.imap_unordered(job, jobs, chunksize=1)}
    out = {}
    for lab in ('b', 'a'):
        y = d[lab][~tr]
        say('%s (majority %.3f), test accuracy by position and layer %s:' % (lab, np.bincount(y).max() / len(y), LAYERS))
        for pi, p in enumerate(POS):
            say('  %-4s all   %s' % (p, ' '.join('%.2f' % res[(lab, pi, li)].mean() for li in range(len(LAYERS)))))
            say('  %-4s right %s' % (p, ' '.join('%.2f' % res[(lab, pi, li)][ok].mean() for li in range(len(LAYERS)))))
            say('  %-4s wrong %s' % (p, ' '.join('%.2f' % res[(lab, pi, li)][~ok].mean() for li in range(len(LAYERS)))))
        best = max(((pi, li) for pi in range(len(POS)) for li in range(len(LAYERS))), key=lambda k: res[(lab,) + k].mean())
        r = res[(lab,) + best]
        say('  %s max over positions and layers %.3f at %s layer %d (theirs %s); there, correctly answered %.3f, wrongly %.3f' % (
            lab, r.mean(), POS[best[0]], LAYERS[best[1]], THEIRS[lab], r[ok].mean(), r[~ok].mean()))
        out[lab] = dict(max=float(r.mean()), at=[POS[best[0]], LAYERS[best[1]]], right=float(r[ok].mean()),
                        wrong=float(r[~ok].mean()), n_right=int(ok.sum()), n_wrong=int((~ok).sum()),
                        grid={'%s_%d' % (POS[pi], LAYERS[li]): float(res[(lab, pi, li)].mean())
                              for pi in range(len(POS)) for li in range(len(LAYERS))})
    json.dump(out, open(os.path.join(OUT, 'v216_theirprobe.json'), 'w'), indent=1)
    return out


def by_op(root, cells=(('pre', 28), ('eq', 20)), say=print):
    """Section 10b: b's per-item test hits at the named cells (refit; the fit is deterministic), wrong - right within
    each operation pair, and the stratified difference (weights = wrong counts; pairs with < 20 in either group left out)."""
    its = build_rep(); s = V5.sha16(its)
    f = glob.glob(os.path.join(root, '**', 'v216_rep_states.npy'), recursive=True)[0]
    X = np.load(f, mmap_mode='r'); m = json.load(open(f.replace('_states.npy', '_meta.json')))
    assert m['items_sha'] == s and len(m['items']) == len(its), 'items differ from the extraction'
    tr = np.array([i['split'] == 'train' for i in its]); ok = np.array([x['correct'] for x in m['items']])[~tr]
    B = np.array([i['B'] for i in its]); ops = np.array([i['o1'] + i['o2'] for i in its])[~tr]; Bt = B[~tr]
    init(dict(X=X, tr=tr, b=B)); out = {}
    for p, l in cells:
        hit = np.array(job(('b', POS.index(p), LAYERS.index(l)))[1], float)
        say('b at %s layer %d: all %.3f | right %.3f, wrong %.3f' % (p, l, hit.mean(), hit[ok].mean(), hit[~ok].mean()))
        keep = [str(o) for o in sorted(set(ops)) if min((ops[ok] == o).sum(), (ops[~ok] == o).sum()) >= 20]
        for o in sorted(set(ops)):
            r, w = ok & (ops == o), ~ok & (ops == o)
            maj = lambda g: np.bincount(Bt[g], minlength=10).max() / max(g.sum(), 1)
            say('  %s: right %4d %.3f (majority %.3f) | wrong %4d %.3f (majority %.3f) | wrong - right %+.3f%s' % (
                o, r.sum(), hit[r].mean() if r.any() else np.nan, maj(r), w.sum(), hit[w].mean() if w.any() else np.nan,
                maj(w), (hit[w].mean() - hit[r].mean()) if r.any() and w.any() else np.nan, '' if o in keep else ' (left out)'))

        def strat(ix):
            h, k, q = hit[ix], ok[ix], ops[ix]
            wt = np.array([(~k & (q == o)).sum() for o in keep], float)
            dd = np.array([h[~k & (q == o)].mean() - h[k & (q == o)].mean() for o in keep])
            return float((wt * dd).sum() / wt.sum())
        est = strat(np.arange(len(hit))); rng = np.random.default_rng(2166); bs = []
        idx = {o: (np.where(ok & (ops == o))[0], np.where(~ok & (ops == o))[0]) for o in keep}
        for _ in range(10000):                                       # resample problems within each (pair, group) cell
            bs.append(strat(np.concatenate([rng.choice(g, len(g)) for o in keep for g in idx[o]])))
        lo, hi = np.percentile(bs, (2.5, 97.5))
        lab = ('b reads better on failures' if lo > 0 else 'less represented on failures' if hi < 0 else
               'not less represented on failures' if lo > -0.05 else 'unclear')
        say('  stratified wrong - right (pairs %s): %+.4f [%+.4f, %+.4f] -> %s' % (keep, est, lo, hi, lab))
        out['%s_%d' % (p, l)] = dict(est=est, ci=[float(lo), float(hi)], reading=lab, all=float(hit.mean()),
                                    right=float(hit[ok].mean()), wrong=float(hit[~ok].mean()))
    json.dump(out, open(os.path.join(OUT, 'v216_theirprobe_byop.json'), 'w'), indent=1)
    return out


def check():
    rep = build_rep(); v5 = V5.build()[1]
    same = len(rep) == len(v5) and all(a['src'] == b['src'] and a['split'] == b['split'] for a, b in zip(rep, v5))
    print(json.dumps(dict(n=len(rep), test=sum(i['split'] == 'test' for i in rep), same_problems_as_v215_rep=same,
                          sha=V5.sha16(rep))))
    print(rep[0]['prompt'] + rep[0]['target'])
    return same


if __name__ == '__main__':
    if sys.argv[1] == '--check':
        sys.exit(0 if check() else 1)
    {'gpu': lambda: gpu(), 'probe': lambda: probe(sys.argv[2]), 'byop': lambda: by_op(sys.argv[2])}[sys.argv[1]]()
