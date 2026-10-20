"""v199 Part B: clamped patching over the tail of the sentence that gives b (docs/prereg_v199_ouro_running_total.md, section 3).

Pairs and subspaces come from v199_patch_plan.npz / v199_patch_pairs.json, written by v199_analyze.py (patch_plan).
For each pair (x, x'), same family, same c_u, different needed first step N (mod 10, same parity), x correct:
  1. run x' unpatched (teacher-forced prompt + ' <answer>', as in the extraction) and record its states at the tail
     (Sb-k for k < tail = Sb - bU: every token after b's units digit up to and including the period) and at bU (b's units
     digit) at the seven patch points mid1, end1, mid2, end2, mid3, end3, mid4;
  2. run x, and at each of the seven points overwrite part of the state with x''s state at that point:
       N       every tail token Sb-k, the 9-dim subspace of the needed first step fitted at that point and offset k:
               h <- h + U U^T (h_x' - h)
       F       the same for the sign-flipped twin
       rand    every tail token, a random 9-dim subspace
       full    every tail token, the whole state
       Nsb     Sb only, the needed first step's subspace (secondary)
       fullsb  Sb only, the whole state (secondary)
       btok    bU, the whole state (method control: replaces the second number itself)
       none    nothing
  3. read the predicted token at P1 after loop 4 (the answer's units digit).
Hit (all but btok): prediction == x''s answer units digit (= (N_x' - c) mod 10, since c is shared).
Hit (btok): prediction == (a_x +/- b_x' - c_x) mod 10, scored on pairs whose b_u differ.
Statistic: hit(N) - max(hit(F), hit(rand)); cluster bootstrap over x's family, 10,000 resamples, seed 1993, 97.5% CI.
  UNSCORABLE  hit(btok) < 0.30 (clamping a question position cannot steer the answer)
  USED        difference >= 0.10 with CI lower > 0
  NOT-USED    CI upper < 0.02
  UNCLEAR     otherwise
Descriptive: hit(full), hit(Nsb), hit(fullsb), each condition's own-digit rate, per version.

  GPU:  OUT_DIR=... V199_PART=0/1 python v199_patch.py extract <dir with the plan | SMOKE>
  CPU:  python v199_patch.py analyse <dir with v199_patch_part*.json>
"""
import glob, json, os, sys, time
import numpy as np

POINTS = ['mid1', 'end1', 'mid2', 'end2', 'mid3', 'end3', 'mid4']
CONDS = ('none', 'N', 'F', 'rand', 'full', 'Nsb', 'fullsb', 'btok')
SMOKE_CONDS = ('none', 'rand', 'btok')       # pre-freeze pilot: control conditions only
Q = (1.25, 98.75)


def need10(i):
    L = i['labels']
    return (L['a'] - L['b']) % 10 if i['op'] == 'minus' else (L['a'] + L['b']) % 10


def btok_target(x, xp):
    s = 1 if x['op'] == 'plus' else -1
    return (x['labels']['a'] + s * xp['labels']['b'] - x['labels']['c']) % 10


def extract(plan_dir):
    os.environ.setdefault('LOOP_MODEL', 'ouro')
    import torch
    import loop_grad_probe as P
    import v199_extract as E
    t0 = time.time(); OUT = P.OUT
    tok, model = P.load(); ad = P.Ouro(model); b_ = ad.b; L = b_.config.num_hidden_layers; mid = min(12, L) - 1
    smoke = plan_dir == 'SMOKE'
    conds = SMOKE_CONDS if smoke else CONDS
    if smoke:
        items = [i for i in json.load(open('v194_items.json'))['items'] if i['set'] == 'pilot']
        rng = np.random.default_rng(0)
        plan = {'U_rand_%d' % h: np.linalg.qr(rng.normal(size=(b_.config.hidden_size, 9)))[0].astype(np.float32) for h in (0, 1)}
        byfam = {}
        for k, i in enumerate(items):
            byfam.setdefault(i['fam'], []).append(k)
        cand = [(k, j) for k, i in enumerate(items) for j in byfam[i['fam']]
                if items[j]['labels']['c'] == i['labels']['c'] and need10(items[j]) != need10(i) and need10(items[j]) % 2 == need10(i) % 2]
        todo = [dict(op=items[cand[t][0]]['op'], half=int(items[cand[t][0]]['tid'] >= 6), k=cand[t][0], kp=cand[t][1], pos=None)
                for t in sorted(rng.choice(len(cand), min(int(os.environ.get('V199_SMOKE_N', 120)), len(cand)), replace=False))]
    else:
        items = [i for i in json.load(open('v194_items.json'))['items'] if i['set'] == 'main']
        plan = dict(np.load(sorted(glob.glob(os.path.join(plan_dir, '**', 'v199_patch_plan.npz'), recursive=True))[0]))
        todo = json.load(open(sorted(glob.glob(os.path.join(plan_dir, '**', 'v199_patch_pairs.json'), recursive=True))[0]))['pairs']
    part = os.environ.get('V199_PART')
    if part is not None:
        h = (len(todo) + 1) // 2; todo = todo[:h] if part == '0' else todo[h:]
    head = E.head_text()
    dev = P.DEV
    U = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in plan.items()}

    def run(h, ctx, hook, start=0):
        """Loops 1-4 from h (start=0: embeddings; start=1: the mid1 state, i.e. continue loop 1 after layer 12)."""
        for t in range(4):
            for li, layer in enumerate(b_.layers[:L]):
                if t == 0 and start == 1 and li <= mid:
                    continue
                h = layer(h, attention_mask=ctx['mask'], position_ids=ctx['pos'], past_key_value=None, use_cache=False,
                          cache_position=ctx['cp'], position_embeddings=ctx['pe'], current_ut=t)
                h = h[0] if isinstance(h, tuple) else h
                if li == mid:
                    h = hook(h, 'mid%d' % (t + 1))
            h = b_.norm(h)
            if t < 3:
                h = hook(h, 'end%d' % (t + 1))
        return h

    def to_mid1(h, ctx):
        for li, layer in enumerate(b_.layers[:mid + 1]):
            h = layer(h, attention_mask=ctx['mask'], position_ids=ctx['pos'], past_key_value=None, use_cache=False,
                      cache_position=ctx['cp'], position_embeddings=ctx['pe'], current_ut=0)
            h = h[0] if isinstance(h, tuple) else h
        return h

    res = []
    with torch.no_grad():
        for n, pr in enumerate(todo):
            x, xp = items[pr['k']], items[pr['kp']]
            px, fx, _ = E.positions(tok, head, x); pxp, fxp, _ = E.positions(tok, head, xp)
            assert px == pxp and len(fx) == len(fxp) and x['fam'] == xp['fam'], pr
            assert pr['pos'] is None or pr['pos'] == px, (pr, px)
            sb, bu, p1, tl = px['Sb'], px['bU'], px['P1'], px['tail']
            at = [sb - o for o in range(tl)]              # the tail, offset 0 (the period) first
            rec = {}

            def record(h, p):
                rec[p] = h[0, at + [bu]].float().clone(); return h
            ids = torch.tensor(fxp, device=dev).unsqueeze(0); h0, ctx = ad.prepare(ids); run(h0, ctx, record)
            ids = torch.tensor(fx, device=dev).unsqueeze(0); h0, ctx = ad.prepare(ids); hm = to_mid1(h0, ctx)
            row = dict(op=x['op'], half=pr['half'], k=pr['k'], kp=pr['kp'], fam=x['fam'], own=x['answer'] % 10,
                       target=xp['answer'] % 10, target_b=btok_target(x, xp), b_differs=bool(x['labels']['b'] != xp['labels']['b']))
            for c in conds:
                def hook(h, p, c=c):
                    if c == 'none':
                        return h
                    h = h.clone()
                    if c == 'btok':
                        h[0, bu] = rec[p][tl].to(h.dtype); return h
                    offs = [0] if c in ('Nsb', 'fullsb') else range(tl)
                    for o in offs:
                        if c in ('full', 'fullsb'):
                            h[0, at[o]] = rec[p][o].to(h.dtype); continue
                        nm = 'N' if c == 'Nsb' else c
                        Uc = U['U_rand_%d' % pr['half']] if c == 'rand' else U['U_%s_%s_%d_%s_o%d' % (nm, x['op'], pr['half'], p, o)]
                        diff = rec[p][o] - h[0, at[o]].float()
                        h[0, at[o]] = h[0, at[o]] + (Uc @ (Uc.T @ diff)).to(h.dtype)
                    return h
                h4 = run(hook(hm, 'mid1'), ctx, hook, start=1)
                lg = ad.readout(h4, ctx)[0, p1].float()
                row[c] = tok.decode([int(lg.argmax())]).strip()
            res.append(row)
            if n % 50 == 0 or n == len(todo) - 1:
                print('%4d/%d %.0fs | %s %s | own %d target %d target_b %d' % (
                    n + 1, len(todo), time.time() - t0, x['op'], ' '.join('%s %s' % (c, row[c]) for c in conds),
                    row['own'], row['target'], row['target_b']), flush=True)
    json.dump(dict(rows=res, conds=conds, smoke=smoke, wall_s=round(time.time() - t0)),
              open(os.path.join(OUT, 'v199_patch_part%s.json' % part), 'w'))
    print('done %ds, %d pairs' % (time.time() - t0, len(res)), flush=True)


def analyse(rdir):
    rows, conds = [], None
    for f in sorted(glob.glob(os.path.join(rdir, '**', 'v199_patch_part*.json'), recursive=True)):
        d = json.load(open(f)); rows += d['rows']; conds = d['conds']
    smoke = 'N' not in conds
    if smoke:            # pilot pairs: x's correctness is unknown, so score rows where Ouro writes its own digit unpatched
        rows = [r for r in rows if r['none'] == str(r['own'])]
    bd = [r for r in rows if r['b_differs']]
    pos = np.mean([r['btok'] == str(r['target_b']) for r in bd])
    print('pairs %d (b differs in %d), families %d | unpatched own digit %.3f' % (
        len(rows), len(bd), len({r['fam'] for r in rows}), np.mean([r['none'] == str(r['own']) for r in rows])))
    print('method control btok: hit %.3f, own %.3f (UNSCORABLE below 0.30)' % (
        pos, np.mean([r['btok'] == str(r['own']) for r in bd])))
    for op in ('plus', 'minus'):
        rs = [r for r in rows if r['op'] == op]
        if rs:
            print('  %s: %s' % (op, ' | '.join('%s hit %.3f own %.3f' % (
                c, np.mean([r[c] == str(r['target_b' if c == 'btok' else 'target']) for r in rs if c != 'btok' or r['b_differs']]),
                np.mean([r[c] == str(r['own']) for r in rs])) for c in conds if c != 'none')))
    if smoke:
        return
    hit = {c: np.array([r[c] == str(r['target']) for r in rows], float) for c in ('N', 'F', 'rand', 'full', 'Nsb', 'fullsb')}
    fam = np.array([r['fam'] for r in rows]); u = np.unique(fam); idx = {f: np.where(fam == f)[0] for f in u}
    rng = np.random.default_rng(1993); bs = []
    for _ in range(10000):
        pick = np.concatenate([idx[f] for f in rng.choice(u, len(u))])
        bs.append(hit['N'][pick].mean() - max(hit['F'][pick].mean(), hit['rand'][pick].mean()))
    dlt = hit['N'].mean() - max(hit['F'].mean(), hit['rand'].mean()); lo, hi = np.percentile(bs, Q)
    lab = ('UNSCORABLE' if pos < 0.30 else 'USED' if dlt >= 0.10 and lo > 0 else 'NOT-USED' if hi < 0.02 else 'UNCLEAR')
    print('tail clamp: hit N %.3f F %.3f rand %.3f full %.3f | period only (secondary): hit N %.3f full %.3f' % tuple(
        hit[c].mean() for c in ('N', 'F', 'rand', 'full', 'Nsb', 'fullsb')))
    print('PART B: hit(N) - max(hit(F), hit(rand)) %+.3f [%+.3f, %+.3f] (97.5%%) -> %s' % (dlt, lo, hi, lab))


if __name__ == '__main__':
    if sys.argv[1] == 'extract':
        extract(sys.argv[2])
    else:
        analyse(sys.argv[2])
