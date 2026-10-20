"""v209: earlier and answer-free patching of a+b at P1 in Ouro (docs/prereg_v209_ouro_early_patch.md).

Follow-up to v194 Part B (NOT-USED at the end of loop 3). An outside review objected that patching a weak a+b component
at the end of loop 3, after the answer has formed, cannot detect use: the intact answer would win. v209 keeps v194's
items, pairs (B1 same-c, B2 answer-fixed), subspace recipe, forward pass and hit definitions, and adds three tests:
  e1   patch at the end of loop 1 (subspaces fitted on end1 states; then loops 2-4 run), where a+b and the answer read
       about equally (Studies 1-2).
  cl   clamp: patch at the end of loops 1, 2 and 3 (each with that point's subspace), setting x's component to x''s
       unpatched component each time, so later loops cannot rebuild it; then loop 4.
  ab3  at the end of loop 3, first ablate the answer subspace to x's family mean, h <- h - Ua Ua^T (h - mu_fam), then
       patch the condition subspace orthogonalised against Ua (so the patch cannot carry x''s answer back in); loop 4.
       Condition 'abl' ablates only.
Conditions: e1 and cl: ab, amb, rand, and ans (positive control, B1 only); ab3: abl, ab, amb, rand; plus 'none'.
Hit (as v194): B1 prediction == x''s answer units digit; B2 prediction == (a'' + b'' - c_x) mod 10.

Statistic per test and pair type: hit(ab) - max(hit(amb), hit(rand)); cluster bootstrap over x's family, 10,000
resamples, seed 2091, 99.1667% CI (5% family-wise over the 6 tests).
  UNSCORABLE  (e1, cl) B1's hit(ans) for that test < 0.30; applies to both pair types
  USED        difference >= 0.10 with CI lower > 0
  NOT-USED    CI upper < 0.02
  UNCLEAR     otherwise
Descriptive: own-digit and target rates under every condition (incl. 'abl'); principal angles of the end1 subspaces.

  GPU:  OUT_DIR=... V209_PART=k V209_NPART=n python v209_patch.py extract <dir with v209_patch_plan.npz, pairs | SMOKE>
  CPU:  python v209_patch.py analyse <dir with v209_patch_part*.json>
        python v209_patch.py selftest
"""
import glob, json, os, sys, time
import numpy as np

CONDS = {'e1': ('ab', 'amb', 'rand', 'ans'), 'cl': ('ab', 'amb', 'rand', 'ans'), 'ab3': ('abl', 'ab', 'amb', 'rand')}
TESTS = ('e1', 'cl', 'ab3')
Q = (100 * 0.05 / 12, 100 - 100 * 0.05 / 12)          # two-sided 5% over 6 tests


def gold(ans):
    return ans.split('####')[-1].strip().replace(',', '')


def target(kind, x, p):
    return p['answer'] % 10 if kind == 'B1' else (p['a'] + p['b'] - x['c']) % 10


def conds_for(test, kind, smoke=False):
    c = CONDS[test] if kind == 'B1' else tuple(x for x in CONDS[test] if x != 'ans')
    return tuple(x for x in c if x in ('rand', 'ans', 'abl')) if smoke else c


def subspace(U, c, h, pt):
    """pt in e1/e2/e3; the random subspace is shared across points (as v194)."""
    if c == 'rand':
        return U['U_rand_%d' % h]
    return U['U_%s_%d%s' % (c, h, {'e1': '_e1', 'e2': '_e2', 'e3': ''}[pt])]


def extract(plan_dir):
    os.environ.setdefault('LOOP_MODEL', 'ouro')
    import torch
    import loop_grad_probe as P
    t0 = time.time(); OUT = P.OUT
    smoke = plan_dir == 'SMOKE'          # code check on pilot pairs: random subspaces, control conditions only
    if smoke:
        items = [i for i in json.load(open('v194_items.json'))['items'] if i['set'] == 'pilot']
        rng = np.random.default_rng(0)
        plan = {'U_%s_%d%s' % (c, h, e): np.linalg.qr(rng.normal(size=(2048, 9)))[0].astype(np.float32)
                for c in ('rand', 'ans', 'ab', 'amb') for h in (0, 1) for e in ('', '_e1', '_e2')}
        plus = [k for k, i in enumerate(items) if i['op'] == 'plus']
        for f in {items[k]['fam'] for k in plus}:
            plan['mu_e3_%s' % f] = None            # filled from the pilot states below
        same = lambda k, j: items[j]['fam'] == items[k]['fam'] and j != k
        b1 = [dict(half=0, k=k, kp=j) for k in plus for j in plus if same(k, j) and items[j]['labels']['c'] == items[k]['labels']['c']]
        b2 = [dict(half=0, k=k, kp=j) for k in plus for j in plus if same(k, j) and items[j]['answer'] == items[k]['answer']]
        todo = [('B1', p) for p in b1[:3]] + [('B2', p) for p in b2[:3]]
    else:
        items = [i for i in json.load(open('v194_items.json'))['items'] if i['set'] in ('main', 'bare')]
        z = np.load(sorted(glob.glob(os.path.join(plan_dir, '**', 'v209_patch_plan.npz'), recursive=True))[0])
        plan = {k: z[k] for k in z.files}
        pj = json.load(open(sorted(glob.glob(os.path.join(plan_dir, '**', 'v194_patch_pairs.json'), recursive=True))[0]))
        todo = [('B1', p) for p in pj['pairs']] + [('B2', p) for p in pj['pairs_fixed']]
    part, npart = int(os.environ.get('V209_PART', '0')), int(os.environ.get('V209_NPART', '1'))
    if not smoke:
        todo = todo[part::npart]
    tok, model = P.load(); ad = P.Ouro(model); b_ = ad.b; L = b_.config.num_hidden_layers
    from datasets import load_dataset
    tr = load_dataset('openai/gsm8k', 'main', split='train')
    head = ''.join('Question: %s\nAnswer: %s\n\n' % (tr[i]['question'], gold(tr[i]['answer'])) for i in range(4))
    dev = P.DEV

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

    if smoke:   # family means for the pilot plan: mean P1 end3 state of the pilot plus items of each family
        with torch.no_grad():
            acc = {}
            for k in plus:
                f = ids_of(items[k]); ids = torch.tensor(f, device=dev).unsqueeze(0); h0, ctx = ad.prepare(ids)
                acc.setdefault(items[k]['fam'], []).append(run(h0, ctx, range(3))[0, len(f) - 2].float().cpu().numpy())
            for fm, v in acc.items():
                plan['mu_e3_%s' % fm] = np.mean(v, 0).astype(np.float32)
    U = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in plan.items() if k.startswith('U_')}
    MU = {k[len('mu_e3_'):]: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in plan.items() if k.startswith('mu_e3_')}

    def setc(h, p1, Uc, ref):
        """Set h's component in span(Uc) at P1 to ref's (h, ref: (1, T, d))."""
        h = h.clone(); d = (ref[0, p1] - h[0, p1]).float()
        h[0, p1] = h[0, p1] + (Uc @ (Uc.T @ d)).to(h.dtype)
        return h

    res = []
    with torch.no_grad():
        for n, (kind, pr) in enumerate(todo):
            x, xp = items[pr['k']], items[pr['kp']]
            fx, fxp = ids_of(x), ids_of(xp)
            assert len(fx) == len(fxp) and x['fam'] == xp['fam'], (pr, len(fx), len(fxp))
            p1 = len(fx) - 2; hh = pr['half']
            st = {}
            for nm, f in (('x', fx), ('xp', fxp)):
                ids = torch.tensor(f, device=dev).unsqueeze(0); h0, ctx = ad.prepare(ids)
                e1 = run(h0, ctx, [0]); e2 = run(e1, ctx, [1]); e3 = run(e2, ctx, [2])
                st[nm] = dict(e1=e1, e2=e2, e3=e3, ctx=ctx)
            ctx = st['x']['ctx']; X, XP = st['x'], st['xp']

            def read(h):
                return tok.decode([int(ad.readout(h, ctx)[0, p1].float().argmax())]).strip()

            row = dict(type=kind, half=hh, k=pr['k'], kp=pr['kp'], fam=x['fam'], own=x['answer'] % 10,
                       target=target(kind, x, xp), none=read(run(X['e3'], ctx, [3])))
            for c in conds_for('e1', kind, smoke):
                row['%s@e1' % c] = read(run(setc(X['e1'], p1, subspace(U, c, hh, 'e1'), XP['e1']), ctx, [1, 2, 3]))
            for c in conds_for('cl', kind, smoke):
                h = X['e1']
                for pt, nxt in (('e1', 1), ('e2', 2), ('e3', 3)):
                    h = run(setc(h, p1, subspace(U, c, hh, pt), XP[pt]), ctx, [nxt])
                row['%s@cl' % c] = read(h)
            Ua = subspace(U, 'ans', hh, 'e3'); mu = MU[str(x['fam'])]
            base = X['e3'].clone(); dv = (base[0, p1].float() - mu)
            base[0, p1] = base[0, p1] - (Ua @ (Ua.T @ dv)).to(base.dtype)
            for c in conds_for('ab3', kind, smoke):
                h = base
                if c != 'abl':
                    Uc = subspace(U, c, hh, 'e3'); Uc = Uc - Ua @ (Ua.T @ Uc); Uc = torch.linalg.qr(Uc)[0]
                    h = setc(base, p1, Uc, XP['e3'])
                row['%s@ab3' % c] = read(run(h, ctx, [3]))
            res.append(row)
            if n % 100 == 0 or n == len(todo) - 1:
                print('%4d/%d %.0fs | %s | own %d target %d | %s' % (
                    n + 1, len(todo), time.time() - t0, kind, row['own'], row['target'],
                    ' '.join('%s %s' % (k, v) for k, v in row.items() if k == 'none' or '@' in k)), flush=True)
    json.dump(dict(rows=res, wall_s=round(time.time() - t0)), open(os.path.join(OUT, 'v209_patch_part%d.json' % part), 'w'))
    print('done %ds, %d pairs' % (time.time() - t0, len(res)), flush=True)


def analyse_rows(rows, say=print):
    b1 = [r for r in rows if r['type'] == 'B1']
    pos = {t: np.mean([r['ans@%s' % t] == str(r['target']) for r in b1]) for t in ('e1', 'cl')}
    labels = {}
    for kind in ('B1', 'B2'):
        rs = [r for r in rows if r['type'] == kind]
        fam = np.array([r['fam'] for r in rs])
        say('%s: pairs %d, families %d | unpatched: own digit %.3f, target %.3f' % (
            kind, len(rs), len(set(fam.tolist())), np.mean([r['none'] == str(r['own']) for r in rs]),
            np.mean([r['none'] == str(r['target']) for r in rs])))
        for t in TESTS:
            cs = conds_for(t, kind)
            hit = {c: np.array([r['%s@%s' % (c, t)] == str(r['target']) for r in rs], float) for c in cs}
            own = {c: np.array([r['%s@%s' % (c, t)] == str(r['own']) for r in rs], float) for c in cs}
            say('  %-3s: %s' % (t, ' | '.join('%s hit %.3f own %.3f' % (c, hit[c].mean(), own[c].mean()) for c in cs)))
            u = np.unique(fam); idx = {f: np.where(fam == f)[0] for f in u}
            rng = np.random.default_rng(2091); bs = []
            for _ in range(10000):
                pick = np.concatenate([idx[f] for f in rng.choice(u, len(u))])
                bs.append(hit['ab'][pick].mean() - max(hit['amb'][pick].mean(), hit['rand'][pick].mean()))
            d = hit['ab'].mean() - max(hit['amb'].mean(), hit['rand'].mean()); lo, hi = np.percentile(bs, Q)
            if t in pos and pos[t] < 0.30:
                lab = 'UNSCORABLE'
            elif d >= 0.10 and lo > 0:
                lab = 'USED'
            elif hi < 0.02:
                lab = 'NOT-USED'
            else:
                lab = 'UNCLEAR'
            labels[(kind, t)] = lab
            say('  hit(ab) - max(hit(amb), hit(rand)) %+.3f [%+.3f, %+.3f] (99.17%%) -> %s %s: %s%s' % (
                d, lo, hi, kind, t, lab, ' (positive control hit %.3f)' % pos[t] if t in pos else ''))
    return labels


def analyse(rdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(rdir, '**', 'v209_patch_part*.json'), recursive=True)):
        rows += json.load(open(f))['rows']
    analyse_rows(rows)


def selftest():
    """Synthetic rows: a world where Ouro follows the patched a+b, one where it ignores it, one where nothing moves."""
    rng = np.random.default_rng(0); ok = True
    for world, want in (('used', 'USED'), ('ignored', 'NOT-USED'), ('stuck', 'UNSCORABLE')):
        rows = []
        for kind in ('B1', 'B2'):
            for n in range(2000):
                own, tgt = int(rng.integers(10)), int(rng.integers(10))
                while tgt == own:
                    tgt = int(rng.integers(10))
                r = dict(type=kind, fam=int(n % 36), own=own, target=tgt, none=str(own))
                for t in TESTS:
                    for c in conds_for(t, kind):
                        if c == 'ans':
                            p = 0.9 if world != 'stuck' else 0.05
                        elif c == 'ab':
                            p = 0.6 if world == 'used' else 0.002
                        else:
                            p = 0.003
                        r['%s@%s' % (c, t)] = str(tgt if rng.random() < p else own)
                rows.append(r)
        lab = analyse_rows(rows, say=lambda *a: None)
        got = {lab[(k, t)] for k in ('B1', 'B2') for t in TESTS if not (world == 'stuck' and t == 'ab3')}
        good = got == {want}
        print('selftest %-8s -> %s %s' % (world, sorted(got), 'OK' if good else 'FAIL (want %s)' % want), flush=True)
        ok &= good
    return ok


if __name__ == '__main__':
    if sys.argv[1] == 'extract':
        extract(sys.argv[2])
    elif sys.argv[1] == 'selftest':
        sys.exit(0 if selftest() else 1)
    else:
        analyse(sys.argv[2])
