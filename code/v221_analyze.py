"""v221 analysis (CPU; docs/prereg_v221_positive_control.md). The protocol, unchanged from v215's short-chain arm, at
eq / last and three tokens of the chain's step " b = B": letb, eqb, inb (B three, two and one token ahead):
  Part A   v215_analyze.part_a (probe, labels S, M, copy ceiling, band, ex_N, I, bootstrap, G1 = b step correct >= 0.90,
           G2 copies >= 0.50) -> v221_partA.json
  plan     v215_analyze.patch_plan for eq, eqb and inb when COMBINED (B, d2-copy, B orthogonalised against the copy, random)
           -> v215_plan.npz, v215_pairs.json (the names v215_patch reads)
  null     v215_null.null (copies + one generic mixture of d1, d2, o1; no B) for each COMBINED position:
           BEYOND-MIXING if the observed I exceeds every calibrated world's I

  python v221_analyze.py <root>          -> Part A and the plan
  python v221_analyze.py null <root>     -> the mixing null (needs v221_partA.json)
  python v221_analyze.py --selftest
"""
import glob, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

import v215_analyze as A

A.POS = ['eq', 'last', 'letb', 'eqb', 'inb']; POS = A.POS; OUT = os.environ.get('OUT_DIR', '.')
PLAN_POS = ['eq', 'eqb', 'inb']


def analyse(X, its, meta, say=print, outdir=OUT):
    for v in ('plus', 'minus'):
        say('b step correct, %s: %.3f' % (v, np.mean([meta[k]['correct'] for k, i in enumerate(its) if i['ver'] == v])))
    out = A.part_a(X, its, meta, say=say, positions=POS, plan=False, outdir=outdir, tag='')
    os.replace(os.path.join(outdir, 'v215_partA.json'), os.path.join(outdir, 'v221_partA.json'))
    todo = [p for p in PLAN_POS if out[p]['label'] == 'COMBINED']
    say('Part B plan for positions: %s' % (todo or 'none'))
    if todo:
        A.patch_plan(X, its, meta, todo, outdir, say)
    return out


def main(root):
    import v215_items as V5, v221_items as I
    its = I.build(); s = V5.sha16(its)
    print('items sha %s (frozen %s)' % (s, I.ITEMS_SHA), flush=True)
    if I.ITEMS_SHA and s != I.ITEMS_SHA:
        sys.exit('items hash differs')
    f = glob.glob(os.path.join(root, '**', 'v221_fam_states.npy'), recursive=True)[0]
    X = np.load(f, mmap_mode='r'); m = json.load(open(f.replace('_states.npy', '_meta.json')))
    assert m['items_sha'] == s and len(m['items']) == len(its)
    analyse(X, its, m['items'])


def null_main(root):
    import v215_null as N
    obs = json.load(open(glob.glob(os.path.join(root, '**', 'v221_partA.json'), recursive=True)[0])); its = N.items()
    for pos, r in obs.items():
        if r['label'] != 'COMBINED':
            print('%s: %s, no null needed' % (pos, r['label'])); continue
        print('--- %s: observed ex_N %+.4f, I %+.4f ---' % (pos, r['exN'][0], r['I'][0]), flush=True)
        w = N.null(r['exN'][0], its, say=lambda s: print(s, flush=True))
        Is = [i for _, v in w.values() for _, i in v]
        lab = 'UNCALIBRATED' if not Is else ('BEYOND-MIXING' if r['I'][0] > max(Is) else 'NOT-SEPARABLE')
        print('%s: null I max %s vs observed %+.4f -> %s' % (pos, '%+.4f' % max(Is) if Is else '-', r['I'][0], lab), flush=True)


def selftest():
    """60 synthetic families on the real design: b computed (a shared code of B at every position) -> COMBINED + FOLLOWS;
    copies only -> NOT-COMBINED, not FOLLOWS; then the plan runs for eq, eqb and inb."""
    import tempfile, v221_items as I
    fam = I.build(); keep = set(sorted(set(i['fam'] for i in fam))[::4]); its = [i for i in fam if i['fam'] in keep]
    meta = [dict(correct=True) for _ in its]; rng = np.random.default_rng(0); dd = 64; ok = True
    U1, U2, CB = (rng.standard_normal((10, dd)) for _ in range(3)); off = {f: rng.standard_normal(dd) * 3 for f in keep}
    for world, want, wswap in (('computed', 'COMBINED', 'FOLLOWS'), ('copies', 'NOT-COMBINED', 'not FOLLOWS')):
        X = np.zeros((len(its), len(POS), len(A.LAYERS), dd), np.float32)
        for k, i in enumerate(its):
            v = off[i['fam']] + U1[i['d1']] + U2[i['d2']] + rng.standard_normal((len(POS), len(A.LAYERS), dd))
            X[k] = v + (0.8 * CB[i['B']] if world == 'computed' else 0)
        out = A.part_a(X, its, meta, say=lambda s: None, positions=POS, plan=False, outdir=tempfile.mkdtemp())
        for p, r in out.items():
            good = r['label'] == want and (r['swap'] != 'FOLLOWS' if wswap == 'not FOLLOWS' else r['swap'] == wswap)
            print('selftest %-8s %-4s ex_N %+.3f [%+.3f, %+.3f] -> %-12s I %+.3f -> %-15s %s' % (
                world, p, *r['exN'], r['label'], r['I'][0], r['swap'], 'OK' if good else 'FAIL'), flush=True)
            ok &= good
        if world == 'computed':
            d = tempfile.mkdtemp(); A.patch_plan(X, its, meta, PLAN_POS, d, lambda s: None)
            pj = json.load(open(os.path.join(d, 'v215_pairs.json')))
            print('selftest plan: positions %s, %d pairs, %d bases' % (pj['positions'], len(pj['pairs']),
                                                                         len(np.load(os.path.join(d, 'v215_plan.npz')).files)), flush=True)
    return ok


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        sys.exit(0 if selftest() else 1)
    if sys.argv[1] == 'null':
        null_main(sys.argv[2])
    else:
        main(sys.argv[1])
