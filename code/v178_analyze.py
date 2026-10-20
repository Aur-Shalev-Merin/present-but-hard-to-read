"""v178 analysis (written and selftested BEFORE any v178 states exist; docs/prereg_v178_ouro_latent_intermediates.md).

  python v178_analyze.py <dir with v178_states_last.npy, v178_states_q.npy, v178_meta.json> [labels.json]
  python v178_analyze.py --selftest

Probe: StandardScaler -> PCA(256, fit on the training fold) -> multinomial logistic regression, C in {0.01, 0.1, 1}, or the
constant majority-class predictor (selftest: an overfit input baseline fell below majority and inflated S by ~0.025),
chosen by inner 3-fold grouped CV. Outer: 5-fold CV grouped by template (fold = template id rank mod 5 after a seeded
shuffle, seed 178). Input baseline: the same classifier on question features (first 8 question numbers: one-hot units
and tens digits, log10 magnitude, presence flag).

PRIMARY label: units digit of the first non-copy intermediate (y_target), at the last prompt token. Two splits:
  cross  = 5-fold grouped by template (decoder must generalise to unseen problem types); baseline = question features;
  within = 5-fold by instance index (every template in every fold; new numbers, same problem types); baseline = question
           features + template one-hot (so a decoder that only learns each template's usual digit gains nothing).
  S_t = acc(state end_t) - acc(baseline), t = 1..4, per split; CI = paired cluster bootstrap over templates (10,000,
  seed 178), Bonferroni over 8 tests (99.375%).
  Gates (cross split): G1 copy control (y_copy) accuracy at end4 >= 0.5; G2 shuffled-within-template y_target accuracy
  at end4 <= majority share + 0.02. Either failing -> UNSCORABLE.
  LATENT-INTERMEDIATE       some S_t (either split) >= 0.05 with CI lower > 0  (report the first loop and the split)
  NO-SELECTIVE-INTERMEDIATE every S_t in BOTH splits has CI upper < 0.05
  UNCLEAR                   otherwise
  H2 (only if LATENT): first passing loop <= 2 and S(mid_t) < S(end_t) in the split that passed.
Secondary (reported): all 9 states for y_target / y_copy / y_final; the 'q' position; S split by correct-at-loop-4.
"""
import json, os, sys
import numpy as np

ORDER = ['h0', 'mid1', 'end1', 'mid2', 'end2', 'mid3', 'end3', 'mid4', 'end4']
ENDS = [2, 4, 6, 8]
BAR = 0.05


def input_features(qnums):
    F = []
    for q in qnums:
        f = []
        for j in range(8):
            if j < len(q):
                v = q[j]; u = int(v) % 10; te = int(v) // 10 % 10
                f += [1.0] + list(np.eye(10)[u]) + list(np.eye(10)[te]) + [np.log10(v + 1)]
            else:
                f += [0.0] * 22
        F.append(f)
    return np.array(F)


def folds(tids, k=5, seed=178):
    u = np.unique(tids); perm = np.random.default_rng(seed).permutation(len(u))
    f_of = {t: perm[i] % k for i, t in enumerate(u)}
    return np.array([f_of[t] for t in tids])


def probe_predict(X, y, groups, fold, pca_dim=256):
    """Held-out predictions for every item (outer grouped CV, inner grouped CV over C)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pred = np.full(len(y), -1)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        dim = min(pca_dim, X.shape[1], tr.sum() - 1)
        red = make_pipeline(StandardScaler(), PCA(dim, random_state=0)).fit(X[tr])   # once per outer fold (unsupervised)
        Ztr, Zte, ytr = red.transform(X[tr]), red.transform(X[te]), y[tr]
        inner = folds(groups[tr], k=3, seed=1781)
        best, best_acc = None, -1
        for C in ('const', 0.01, 0.1, 1.0):          # 'const' = training-fold majority class (probe may not beat it)
            acc = []
            for g in range(3):
                a, b = inner != g, inner == g
                if C == 'const':
                    acc.append((ytr[b] == np.bincount(ytr[a]).argmax()).mean())
                else:
                    m = LogisticRegression(C=C, max_iter=300).fit(Ztr[a], ytr[a]); acc.append((m.predict(Ztr[b]) == ytr[b]).mean())
            if np.mean(acc) > best_acc:
                best, best_acc = C, np.mean(acc)
        if best == 'const':
            pred[te] = np.bincount(ytr).argmax()
        else:
            pred[te] = LogisticRegression(C=best, max_iter=300).fit(Ztr, ytr).predict(Zte)
    return pred


def cluster_ci(tids, d, q=(2.5, 97.5), n=10000, seed=178):
    u, inv = np.unique(tids, return_inverse=True)
    sums = np.bincount(inv, d); cnts = np.bincount(inv)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(u), (n, len(u)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(d.mean()), float(np.percentile(bs, q[0])), float(np.percentile(bs, q[1]))


def analyse(A, Qs, labels, meta, verbose=True, secondary=True):
    out = []; say = out.append
    tids = np.array([i['tid'] for i in labels]); fold = folds(tids)
    y = np.array([i['y_target'] for i in labels]); maj = np.bincount(y).max() / len(y)
    Xin = input_features([i['qnums'] for i in labels])
    p_in = probe_predict(Xin, y, tids, fold); c_in = (p_in == y).astype(float)
    say('items %d, templates %d; majority share %.3f; input-baseline accuracy %.3f' % (len(y), len(np.unique(tids)), maj, c_in.mean()))
    # gates
    yc = np.array([-1 if i['y_copy'] is None else i['y_copy'] for i in labels]); okc = yc >= 0
    pc = probe_predict(A[okc, 8], yc[okc], tids[okc], fold[okc]); g1 = (pc == yc[okc]).mean()
    rng = np.random.default_rng(178); ysh = y.copy()
    for t in np.unique(tids):
        m = tids == t; ysh[m] = rng.permutation(ysh[m])
    g2 = (probe_predict(A[:, 8], ysh, tids, fold) == ysh).mean()
    say('G1 copy-control accuracy at end4 %.3f (need >= 0.5); G2 shuffled-label accuracy at end4 %.3f (need <= %.3f)' % (g1, g2, maj + 0.02))
    corr = {}
    for st in (range(9) if secondary else sorted(set(ENDS) | {e - 1 for e in ENDS})):
        corr[st] = (probe_predict(A[:, st], y, tids, fold) == y).astype(float)
    # within-template split: folds by instance index; inner CV groups = items; baseline also knows the template
    inst = np.array([i['inst'] for i in labels]); fold_w = folds(inst); items_ = np.arange(len(y))
    u_t = {t: k for k, t in enumerate(np.unique(tids))}
    Xin_w = np.hstack([Xin, np.eye(len(u_t))[[u_t[t] for t in tids]]])
    c_in_w = (probe_predict(Xin_w, y, items_, fold_w) == y).astype(float)
    corr_w = {st: (probe_predict(A[:, st], y, items_, fold_w) == y).astype(float)
              for st in sorted(set(ENDS) | {e - 1 for e in ENDS})}
    say('within-template baseline accuracy %.3f' % c_in_w.mean())
    Q = (0.3125, 99.6875)
    if g1 < 0.5 or g2 > maj + 0.02:
        label, first = 'UNSCORABLE', None
    else:
        res = {(sp, e): cluster_ci(tids, (cc[e] - cb), q=Q) for sp, cc, cb in (('cross', corr, c_in), ('within', corr_w, c_in_w))
               for e in ENDS}
        passed = sorted([(ENDS.index(e), sp) for (sp, e), r in res.items() if r[0] >= BAR and r[1] > 0])
        if passed:
            label = 'LATENT-INTERMEDIATE'; first = (ENDS[passed[0][0]], passed[0][1])
        elif all(r[2] < BAR for r in res.values()):
            label, first = 'NO-SELECTIVE-INTERMEDIATE', None
        else:
            label, first = 'UNCLEAR', None
        for (sp, e), r in sorted(res.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            say('S %-6s at %s: %+.3f  99.375%% CI [%+.3f, %+.3f]' % ((sp, ORDER[e]) + r))
    say('PRIMARY -> %s%s' % (label, (' (first at %s, %s split)' % (ORDER[first[0]], first[1])) if first else ''))
    if label == 'LATENT-INTERMEDIATE':
        e, sp = first; cc, cb = (corr, c_in) if sp == 'cross' else (corr_w, c_in_w)
        t = ENDS.index(e) + 1; smid = cc[e - 1].mean() - cb.mean(); send = cc[e].mean() - cb.mean()
        say('H2: first loop %d (<= 2: %s); S mid %.3f vs end %.3f (mid < end: %s)' % (t, t <= 2, smid, send, smid < send))
    if not secondary:
        if verbose:
            print('\n'.join(out))
        return label
    say('--- secondary ---')
    say('y_target accuracy per state (cross): %s' % ' '.join('%s %.3f' % (ORDER[st], corr[st].mean()) for st in range(9)))
    if meta is not None:
        c4 = np.array([m['correct'][3] for m in meta])
        for nm, mk in (('correct@4', c4), ('wrong@4', ~c4)):
            if mk.sum() >= 20:
                say('S (end4) on %s items (n=%d): %+.3f' % (nm, mk.sum(), corr[8][mk].mean() - c_in[mk].mean()))
        say('answer-only accuracy per loop: %s' % np.round(np.mean([m['correct'] for m in meta], 0), 3).tolist())
    if Qs is not None:
        say('y_target accuracy at the question-end token: %s' % ' '.join(
            '%s %.3f' % (ORDER[s], (probe_predict(Qs[:, s], y, tids, fold) == y).mean()) for s in ENDS))
    yf = np.array([i['y_final'] for i in labels])
    say('y_final accuracy (last token) end1..end4: %s; input baseline %.3f' % (
        ' '.join('%.3f' % (probe_predict(A[:, s], yf, tids, fold) == yf).mean() for s in ENDS),
        (probe_predict(Xin, yf, tids, fold) == yf).mean()))
    if verbose:
        print('\n'.join(out))
    return label


def selftest():
    """Synthetic worlds with the real label structure (84 x 50 items), small states (d = 48) for speed."""
    lab = json.load(open('v178_labels.json'))['items']
    rng = np.random.default_rng(0); N = len(lab); d = 48
    y = np.array([i['y_target'] for i in lab]); yc = np.array([i['y_copy'] or 0 for i in lab])
    q = np.array([[int(i['qnums'][j]) % 10 if j < len(i['qnums']) else 0 for j in range(2)] for i in lab])
    W = rng.normal(size=(10, d)); tid = np.array([i['tid'] for i in lab])
    trank = {t: k for k, t in enumerate(np.unique(tid))}; tr_ = np.array([trank[t] for t in tid])
    Wt = rng.normal(size=(len(trank), 10, d)); Tt = rng.normal(size=(len(trank), d))

    def world(kind):
        A = rng.normal(size=(N, 9, d))
        A[:, 1:, :] += 1.5 * W[yc][:, None, :]                       # copy control decodable after the embedding
        if kind == 'compute':
            A[:, 3:, :] += 1.2 * W[y][:, None, :] @ np.eye(d)[::-1]  # target digit present from mid2 on
        elif kind == 'operand_only':                                   # operands' digits present, product's not
            A[:, 1:, :] += 0.8 * (W[q[:, 0]] + W[q[:, 1]] @ np.eye(d)[::-1])[:, None, :]
        elif kind == 'template_specific':                              # digit present, but coded differently per template
            A[:, 3:, :] += 1.2 * Wt[tr_, y][:, None, :]
        elif kind == 'template_prior':                                 # state knows the template only (its usual digit)
            A[:, 1:, :] += 2.0 * Tt[tr_][:, None, :]
        elif kind == 'broken':
            A[:, 1:, :] -= 1.5 * W[yc][:, None, :]                     # no copy signal -> G1 fails
        return A
    # 'template_specific' (a fully separate code per template) is NOT in the selftest: with 50 items per template no
    # pooled probe can learn 84 separate codes (it scored NO-SELECTIVE). A null therefore means 'no shared code'.
    want = dict(compute='LATENT-INTERMEDIATE', template_prior='NO-SELECTIVE-INTERMEDIATE',
                operand_only='NO-SELECTIVE-INTERMEDIATE', none='NO-SELECTIVE-INTERMEDIATE', broken='UNSCORABLE')
    ok = True
    for k, w in want.items():
        lab_ = analyse(world(k), None, lab, None, verbose=False, secondary=False)
        print('selftest %-13s -> %-26s (want %s)' % (k, lab_, w), flush=True); ok &= lab_ == w
    print('SELFTEST', 'OK' if ok else 'FAILED'); return ok


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        sys.exit(0 if selftest() else 1)
    root = sys.argv[1]; labels = json.load(open(sys.argv[2] if len(sys.argv) > 2 else 'v178_labels.json'))['items']
    meta = json.load(open(os.path.join(root, 'v178_meta.json')))
    A = np.load(os.path.join(root, 'v178_states_last.npy')).astype(np.float32)
    Qs = np.load(os.path.join(root, 'v178_states_q.npy')).astype(np.float32)
    assert len(labels) == len(meta['items']) == len(A), 'item count mismatch'
    assert all(m['tid'] == l['tid'] and m['inst'] == l['inst'] for m, l in zip(meta['items'], labels)), 'item order mismatch'
    analyse(A, Qs, labels, meta['items'])
