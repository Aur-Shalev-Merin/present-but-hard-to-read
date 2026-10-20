"""v194 Part B: causal patching (docs/prereg_v194_ouro_operation_swap.md, sections 3 and 9).

Pairs come from v194_patch_plan.npz / v194_patch_pairs.json, written by v194_analyze.py (patch_plan). For each pair, run
both items to the end of loop 3 (teacher-forced prompt + ' <answer>', as in the extraction); at P1 (answer token before
the units digit) replace x's component in a 9-dim subspace U by the partner's:  h <- h + U U^T (h_partner - h_x);
run loop 4 and read the predicted next token at P1 (the units digit). Subspaces were fitted on the other template half.

B1 (as frozen): pairs (x, x') with the same c_u, the same parity of a_u + b_u, a different (a_u + b_u) mod 10.
    Conditions none, ab, amb, rand, ans (positive control). Hit: prediction == x''s answer units digit
    (= (a' + b' - c) mod 10).
B2 (amendment 2026-10-14, before any analysis): answer-fixed pairs (x, x'') with the SAME answer, a different
    (a_u + b_u) mod 10 of the same parity (c compensates). Conditions none, ab, amb, rand. Hit: prediction ==
    (a'' + b'' - c_x) mod 10, the digit x would write if it used x'''s a + b with its own c. Under "the state holds the
    answer, with c added back" x'' carries the same answer, so an ab patch should not produce hits.

Patch points (amendment 2026-10-14, before any analysis): end of loop 3 (then loop 4 runs; subspaces fitted on end3
states), and end of loop 2 (then loops 3 and 4 run; subspaces fitted on end2 states; the random subspace is shared).

Statistic per type and patch point: hit(ab) - max(hit(amb), hit(rand)); cluster bootstrap over x's family, 10,000
resamples, seed 1942, 99.375% CI (Bonferroni: Part A 2.5%; B1 and B2 at two patch points, 0.625% each).
  UNSCORABLE  B1's hit(ans) at that patch point < 0.30 (the method cannot move the prediction; applies to both types)
  USED        difference >= 0.10 with CI lower > 0
  NOT-USED    CI upper < 0.02
  UNCLEAR     otherwise
Descriptive: principal angles between the ab subspace and the ans / amb subspaces, per half.

  GPU:  OUT_DIR=... V194_PART=0/1 python v194_patch.py extract <dir with the plan | SMOKE>
  CPU:  python v194_patch.py analyse <dir with v194_patch_part*.json> [<dir with the plan>]
"""
import glob, json, os, sys, time
import numpy as np

CONDS = {'B1': ('none', 'ab', 'amb', 'rand', 'ans'), 'B2': ('none', 'ab', 'amb', 'rand')}
POINTS = (('e3', 1, [3]), ('e2', 0, [2, 3]))     # (patch point, which saved state, loops run after the patch)
Q = (0.3125, 99.6875)


def gold(ans):
    return ans.split('####')[-1].strip().replace(',', '')


def target(kind, x, p):
    return p['answer'] % 10 if kind == 'B1' else (p['a'] + p['b'] - x['c']) % 10


def extract(plan_dir):
    os.environ.setdefault('LOOP_MODEL', 'ouro')
    import torch
    import loop_grad_probe as P
    t0 = time.time(); OUT = P.OUT
    smoke = plan_dir == 'SMOKE'          # code check on pilot pairs: random subspaces, control conditions only
    conds = {'B1': ('none', 'rand', 'ans'), 'B2': ('none', 'rand')} if smoke else CONDS
    if smoke:
        items = [i for i in json.load(open('v194_items.json'))['items'] if i['set'] == 'pilot']
        rng = np.random.default_rng(0)
        plan = {'U_%s_%d%s' % (c, h, e): np.linalg.qr(rng.normal(size=(2048, 9)))[0].astype(np.float32)
                for c in ('rand', 'ans') for h in (0, 1) for e in ('', '_e2')}
        plus = [k for k, i in enumerate(items) if i['op'] == 'plus']
        same = lambda k, j: items[j]['fam'] == items[k]['fam'] and j != k
        b1 = [dict(half=0, k=k, kp=j) for k in plus for j in plus if same(k, j) and items[j]['labels']['c'] == items[k]['labels']['c']]
        b2 = [dict(half=0, k=k, kp=j) for k in plus for j in plus if same(k, j) and items[j]['answer'] == items[k]['answer']]
        todo = [('B1', p) for p in b1[:3]] + [('B2', p) for p in b2[:3]]
        plan_files = list(plan)
    else:
        items = [i for i in json.load(open('v194_items.json'))['items'] if i['set'] in ('main', 'bare')]
        plan = np.load(sorted(glob.glob(os.path.join(plan_dir, '**', 'v194_patch_plan.npz'), recursive=True))[0])
        plan_files = plan.files
        pj = json.load(open(sorted(glob.glob(os.path.join(plan_dir, '**', 'v194_patch_pairs.json'), recursive=True))[0]))
        todo = [('B1', p) for p in pj['pairs']] + [('B2', p) for p in pj['pairs_fixed']]
    part = os.environ.get('V194_PART')
    if part is not None and not smoke:
        h = (len(todo) + 1) // 2; todo = todo[:h] if part == '0' else todo[h:]
    tok, model = P.load(); ad = P.Ouro(model); b_ = ad.b; L = b_.config.num_hidden_layers
    from datasets import load_dataset
    tr = load_dataset('openai/gsm8k', 'main', split='train')
    head = ''.join('Question: %s\nAnswer: %s\n\n' % (tr[i]['question'], gold(tr[i]['answer'])) for i in range(4))
    dev = P.DEV
    U = {k: torch.tensor(plan[k], device=dev, dtype=torch.float32) for k in plan_files}

    def run(h, ctx, loops):
        for t in loops:
            for layer in b_.layers[:L]:
                h = layer(h, attention_mask=ctx['mask'], position_ids=ctx['pos'], past_key_value=None, use_cache=False,
                          cache_position=ctx['cp'], position_embeddings=ctx['pe'], current_ut=t)
                h = h[0] if isinstance(h, tuple) else h
            h = b_.norm(h)
        return h

    def ids_of(it):
        p = head + 'Question: %s\nAnswer:' % it['question']
        return tok(p + ' %d' % it['answer']).input_ids

    res = []
    with torch.no_grad():
        for n, (kind, pr) in enumerate(todo):
            x, xp = items[pr['k']], items[pr['kp']]
            fx, fxp = ids_of(x), ids_of(xp)
            assert len(fx) == len(fxp) and x['fam'] == xp['fam'], (pr, len(fx), len(fxp))
            p1 = len(fx) - 2
            hs = {}
            for nm, f in (('x', fx), ('xp', fxp)):
                ids = torch.tensor(f, device=dev).unsqueeze(0); h0, ctx = ad.prepare(ids)
                h2 = run(h0, ctx, range(2)); hs[nm] = (h2, run(h2, ctx, [2]), ctx)
            ctx = hs['x'][2]
            row = dict(type=kind, half=pr['half'], k=pr['k'], kp=pr['kp'], fam=x['fam'], own=x['answer'] % 10,
                       target=target(kind, x, xp))
            for pt, slot, rest in POINTS:
                hx, hxp = hs['x'][slot], hs['xp'][slot]
                diff = (hxp[0, p1] - hx[0, p1]).float()
                for c in conds[kind]:
                    if c == 'none' and pt != 'e3':
                        continue
                    h = hx.clone()
                    if c != 'none':
                        Uc = U['U_%s_%d%s' % (c, pr['half'], '' if pt == 'e3' or c == 'rand' else '_e2')]
                        h[0, p1] = h[0, p1] + (Uc @ (Uc.T @ diff)).to(h.dtype)
                    h4 = run(h, ctx, rest)
                    lg = ad.readout(h4, ctx)[0, p1].float()
                    row[c if c == 'none' else '%s@%s' % (c, pt)] = tok.decode([int(lg.argmax())]).strip()
            res.append(row)
            if n % 100 == 0 or n == len(todo) - 1:
                print('%4d/%d %.0fs | %s %s | own %d target %d' % (
                    n + 1, len(todo), time.time() - t0, kind, ' '.join('%s %s' % (k, v) for k, v in row.items()
                                                                       if k == 'none' or '@' in k),
                    row['own'], row['target']), flush=True)
    json.dump(dict(rows=res, wall_s=round(time.time() - t0)), open(os.path.join(OUT, 'v194_patch_part%s.json' % part), 'w'))
    print('done %ds, %d pairs' % (time.time() - t0, len(res)), flush=True)


def analyse(rdir, plan_dir=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(rdir, '**', 'v194_patch_part*.json'), recursive=True)):
        rows += json.load(open(f))['rows']
    b1 = [r for r in rows if r['type'] == 'B1']
    pos = {pt: np.mean([r['ans@%s' % pt] == str(r['target']) for r in b1]) for pt, _, _ in POINTS}
    for kind in ('B1', 'B2'):
        rs = [r for r in rows if r['type'] == kind]
        fam = np.array([r['fam'] for r in rs])
        print('%s: pairs %d, families %d | unpatched: own digit %.3f, target %.3f' % (
            kind, len(rs), len(set(fam)), np.mean([r['none'] == str(r['own']) for r in rs]),
            np.mean([r['none'] == str(r['target']) for r in rs])))
        for pt, _, _ in POINTS:
            hit = {c: np.array([r['%s@%s' % (c, pt)] == str(r['target']) for r in rs], float) for c in CONDS[kind] if c != 'none'}
            own = {c: np.array([r['%s@%s' % (c, pt)] == str(r['own']) for r in rs], float) for c in hit}
            print('  patch at %s: %s' % (pt, ' | '.join('%s hit %.3f own %.3f' % (c, hit[c].mean(), own[c].mean()) for c in hit)))
            u = np.unique(fam); idx = {f: np.where(fam == f)[0] for f in u}
            rng = np.random.default_rng(1942); bs = []
            for _ in range(10000):
                pick = np.concatenate([idx[f] for f in rng.choice(u, len(u))])
                bs.append(hit['ab'][pick].mean() - max(hit['amb'][pick].mean(), hit['rand'][pick].mean()))
            d = hit['ab'].mean() - max(hit['amb'].mean(), hit['rand'].mean()); lo, hi = np.percentile(bs, Q)
            if pos[pt] < 0.30:
                lab = 'UNSCORABLE'
            elif d >= 0.10 and lo > 0:
                lab = 'USED'
            elif hi < 0.02:
                lab = 'NOT-USED'
            else:
                lab = 'UNCLEAR'
            print('  hit(ab) - max(hit(amb), hit(rand)) %+.3f [%+.3f, %+.3f] (99.375%%) -> %s at %s: %s '
                  '(positive control hit %.3f)' % (d, lo, hi, kind, pt, lab, pos[pt]))
    if plan_dir:
        plan = np.load(sorted(glob.glob(os.path.join(plan_dir, '**', 'v194_patch_plan.npz'), recursive=True))[0])
        for e in ('', '_e2'):
            for h in (0, 1):
                for other in ('ans', 'amb', 'rand'):
                    Uo = plan['U_%s_%d%s' % (other, h, '' if other == 'rand' else e)]
                    sv = np.linalg.svd(plan['U_ab_%d%s' % (h, e)].T @ Uo, compute_uv=False)
                    print('subspace overlap %s half %d: ab vs %-4s cos of principal angles %s | mean cos^2 %.3f' % (
                        'end3' if e == '' else 'end2', h, other, ' '.join('%.2f' % v for v in sv), float((sv ** 2).mean())))


if __name__ == '__main__':
    if sys.argv[1] == 'extract':
        extract(sys.argv[2])
    else:
        analyse(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
