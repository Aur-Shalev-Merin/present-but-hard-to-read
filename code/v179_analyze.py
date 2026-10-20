"""v179 pilot analysis (control labels only; exploratory, no prereg).

  python v179_analyze.py <dir with v179_states.npy, v179_meta.json>

Probe = v178's probe_predict (standardise -> PCA 256 -> logistic, C in {const, 0.01, 0.1, 1} by inner grouped CV),
5-fold CV grouped by template (v178 folds). Labels: units digit of the first / last / largest question number
(integers only). Each label is probed at its own anchor's 4 slots (+0, +1, +3, sentence end) and at the two global
slots ('?', last prompt token), at every stored state.
Apparatus check: '<label>+0' at h0 is the digit token's own embedding -> must read ~1.0.
Reading rule (fixed before the data): a slot READS the copied digit at a state if accuracy >= max(0.5, majority + 0.15)
(v178's G1 bar was 0.5 with a 0.408 majority).
"""
import json, os, sys
import numpy as np

import v178_analyze as V


def main(root):
    meta = json.load(open(os.path.join(root, 'v179_meta.json')))
    X = np.load(os.path.join(root, 'v179_states.npy')).astype(np.float32)
    its = meta['items']; slots, states = meta['slots'], meta['states']
    assert len(its) == len(X)
    tids = np.array([i['tid'] for i in its])
    print('items %d, templates %d, digit tokens ok %d; check |logit diff| %s' % (
        len(its), len(set(tids)), sum(i['digit_tok_ok'] for i in its), meta.get('check_logit')))
    reads, accs = {}, {}
    for lab in ('first', 'last', 'big'):
        y = np.array([-1 if i['labels'][lab] is None else i['labels'][lab] for i in its]); ok = y >= 0
        maj = np.bincount(y[ok]).max() / ok.sum(); bar = max(0.5, maj + 0.15)
        fold = V.folds(tids[ok])
        print('\n== label: units digit of the %s number | n %d | majority %.3f | reading bar %.3f' % (lab, ok.sum(), maj, bar))
        print('%-12s %s' % ('slot', ' '.join('%6s' % s for s in states)))
        for sl in [lab + '+0', lab + '+1', lab + '+3', lab + '_sent', 'qmark', 'last']:
            si = slots.index(sl); row = []
            for st in range(len(states)):
                acc = (V.probe_predict(X[ok, si, st], y[ok], tids[ok], fold) == y[ok]).mean()
                row.append(acc); reads[(lab, sl, states[st])] = acc >= bar; accs[(lab, sl, states[st])] = acc
            print('%-12s %s' % (sl, ' '.join('%6.3f' % a for a in row)), flush=True)
    print('\napparatus check (<label>+0 at h0 >= 0.9):', all(
        accs[(lab, lab + '+0', 'h0')] >= 0.9 for lab in ('first', 'last', 'big')))
    for sl in ('sent', 'qmark', 'last'):
        hits = sorted({(lab, st) for (lab, s, st), r in reads.items() if r and (s.endswith(sl) if sl == 'sent' else s == sl)})
        print('reads at %-5s: %s' % (sl, hits if hits else 'none'))


if __name__ == '__main__':
    main(sys.argv[1])
