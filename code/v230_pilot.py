"""v230 pilot (docs/v230_design.md, section 6): train small transformers on a task that forces an unwritten
intermediate, and find where it lives with DAS (ground truth). The protocol (share swap, clamp) is NOT run here.

Task 'pos' (mod 23): tokens  [pad]*(6-k) x op z ; [fill]*k w =  ->  y = (T[(x op z) mod 23] + w) mod 23,
  T a fixed random permutation (seed 230), op in {+, -}, k in 0..6 filler tokens. Every sequence has 13 tokens:
  ';' sits at 9-k, w at 10, '=' at 11 (predicts y). s = x op z is never written.
Task 'neg' (negative control): same tokens, y = (T1[x] + T2[z] op w) mod 23: op matters but x op z is never needed.
20% of the (op, x, z, w) combinations are held out of training (all k) and form the evaluation set.
Models: v220's Net (pre-LN GPT, d 192, 6 heads): std (6 layers) and loop (2-layer block x 3). AdamW as v220.

DAS (pos task only): for variable S at (position p, layer l), learn an m-dim orthonormal subspace B; the intervention
replaces the base state's coordinates in B with the source's: h' = h + B B^T (h_src - h). Pairs share k; the source
has a different T[s] and a different w, so the target T[s_src] + w_base differs from both the base answer and the
source answer (an answer-code patch cannot score). Guarded: B lies in the orthogonal complement of the linear copy
span of x, z, w and op at (p, l) (least-squares class codes on 4,096 problems), so a patch of the copies alone cannot
score. Reported: held-out IIA (guarded and unguarded), and a random m-dim guarded subspace (no training).
Pilot 5 adds X / Z DAS at ';' (one operand takes the source's value; target T[x_src op z_base] + w_base) and
known-answer cells at layer 0 (each input at its own token).
Env: CELLS "arch:task:seed[:scan[:steps]],..." (default std:pos:0,loop:pos:0,std:neg:0), STEPS (30000), LR (1e-3),
DAS_M (16), DAS_STEPS (2000), SCAN (full | key | xz: S cells scanned besides X / Z), TINY=1 (seconds-long CPU smoke
test), OUT_DIR.
"""
import json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from v220_train import Net

OUT = os.environ.get('OUT_DIR', '.'); TINY = os.environ.get('TINY') == '1'
DEV = 'cuda' if torch.cuda.is_available() and not TINY else 'cpu'
STEPS = 40 if TINY else int(os.environ.get('STEPS', '30000')); EVAL_EVERY = 20 if TINY else 500
LR = float(os.environ.get('LR', '1e-3')); DAS_M = int(os.environ.get('DAS_M', '16'))
DAS_STEPS = 10 if TINY else int(os.environ.get('DAS_STEPS', '2000'))
GUARDS = tuple(c == '1' for c in os.environ.get('GUARDS', '1,0').split(','))    # '1' guarded, '0' unguarded
SCAN = os.environ.get('SCAN', 'full')                                            # 'full' or 'key'
P, KMAX = 23, 6
PLUS, MINUS, SEMI, FILL, EQ, PAD = 23, 24, 25, 26, 27, 28
V, L = 29, 13
rng0 = np.random.default_rng(230)
T = torch.tensor(rng0.permutation(P)); T1 = torch.tensor(rng0.permutation(P)); T2 = torch.tensor(rng0.permutation(P))


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v230_log.txt'), 'a').write(s + '\n')


class Task:
    def __init__(self, kind):
        self.kind = kind
        allc = np.array([(op, x, z, w) for op in range(2) for x in range(P) for z in range(P) for w in range(P)])
        perm = np.random.default_rng(2301).permutation(len(allc)); n = int(0.2 * len(allc))
        self.test, self.train = torch.tensor(allc[perm[:n]]), torch.tensor(allc[perm[n:]])

    def answer(self, op, x, z, w):
        if self.kind == 'pos':
            return (T[torch.where(op == 0, x + z, x - z) % P] + w) % P
        return (T1[x] + T2[z] + torch.where(op == 0, w, -w)) % P

    def sample(self, n, g, split='train'):
        src = self.train if split == 'train' else self.test
        c = src[torch.randint(0, len(src), (n,), generator=g)]
        k = torch.randint(0, KMAX + 1, (n,), generator=g)
        return torch.cat([c, k[:, None]], 1)                                    # op, x, z, w, k

    def encode(self, c):
        op, x, z, w, k = c.unbind(1); n = len(op)
        seq = torch.full((n, L), FILL, dtype=torch.long)
        idx = torch.arange(L)[None].expand(n, L)
        start = KMAX - k                                                        # x sits at 6-k
        seq[idx < start[:, None]] = PAD
        r = torch.arange(n)
        seq[r, start] = x; seq[r, start + 1] = PLUS + op; seq[r, start + 2] = z; seq[r, start + 3] = SEMI
        seq[:, 10] = w; seq[:, 11] = EQ; seq[:, 12] = self.answer(op, x, z, w)
        return seq


def forward_to(net, idx, layer, patch=None):
    """Run net; after block `layer` (1-based, counting loops; 0 = the embedding), apply patch(h) to the residual stream.
    Returns logits (if patch is given or layer is None) or the residual after `layer`."""
    x = net.emb(idx) + net.pos[:, :idx.shape[1]]; i = 0
    if layer == 0:
        if patch is None:
            return x
        x = patch(x)
    for _ in range(net.loops):
        for b in net.blocks:
            x = b(x); i += 1
            if i == layer:
                if patch is None:
                    return x
                x = patch(x)
    return net.head(net.lnf(x))


def n_layers(net):
    return net.loops * len(net.blocks)


def train(arch, kind, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(1000 + seed); task = Task(kind)
    kw = dict(d=32, h=2, n_std=2, n_blk=1, loops=2) if TINY else {}
    net = Net(V, L, arch, **kw).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=0.1)
    sched = lambda s: min(1.0, (s + 1) / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
    scaler = torch.amp.GradScaler(enabled=DEV == 'cuda')
    ev = task.encode(task.sample(8192, torch.Generator().manual_seed(9230), 'test')).to(DEV)
    curve, hit, t0, bs = [], None, time.time(), 32 if TINY else 256
    for s in range(STEPS):
        for gr in opt.param_groups:
            gr['lr'] = LR * sched(s)
        x = task.encode(task.sample(bs, g)).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == 'cuda'):
            loss = F.cross_entropy(net(x[:, :-1])[:, -1].float(), x[:, -1])
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update()
        if (s + 1) % EVAL_EVERY == 0:
            net.eval()
            with torch.no_grad():
                acc = float((net(ev[:, :-1])[:, -1].argmax(-1) == ev[:, -1]).float().mean())
            net.train(); curve.append([s + 1, round(float(loss), 4), acc])
            if hit is None and acc >= 0.99:
                hit = s + 1
            if not math.isfinite(float(loss)) or (hit is not None and s + 1 >= hit + 2000):
                break
    net.eval()
    log('train %s %s seed %d | to 0.99: %s | final acc %.4f | %d steps, %d s'
        % (arch, kind, seed, hit, curve[-1][2] if curve else float('nan'), s + 1, time.time() - t0))
    torch.save(net.state_dict(), os.path.join(OUT, 'v230_%s_%s_s%d.pt' % (arch, kind, seed)))
    return net, task, dict(step_to_0p99=hit, final_acc=curve[-1][2] if curve else None, curve=curve, steps=s + 1)


@torch.no_grad()
def copy_basis(net, task, layer, pos, k, g):
    """Orthonormal basis of the linear copy span (one-hot class codes of the inputs visible at pos: x, z, op, and w
    from its own token on) at (pos, layer) for this k. Rank: singular values above the largest from 5 shuffled-label
    fits (pilot 2 kept all 67 at 1e-3 * S0; the directions fitted to noise, w's among them before w is seen, held
    84% of s's class-mean energy, so the guard removed s itself)."""
    c = task.sample(16384 if not TINY else 256, g); c[:, 4] = k
    h = forward_to(net, task.encode(c)[:, :-1].to(DEV), layer)[:, pos].float().cpu()
    op, x, z, w, _ = c.unbind(1)
    cols = [(x, P), (z, P), (op, 2)] + ([(w, P)] if pos >= 10 else [])
    hc = h - h.mean(0)

    def fit(shuffle):
        X = torch.cat([F.one_hot(v[torch.randperm(len(v), generator=g)] if shuffle else v, m) for v, m in cols], 1).float()
        # centred one-hot groups are rank-deficient: CUDA's default 'gels' assumes full rank and returns ~1e6 junk
        # directions (pilot 1's guard removed 4 of them), so solve on CPU with 'gelsd'
        coef = torch.linalg.lstsq(X - X.mean(0), hc, driver='gelsd').solution
        return torch.linalg.svd(coef, full_matrices=False)
    _, S, Vh = fit(False)
    null = max(float(fit(True)[1][0]) for _ in range(5))
    r = int((S > null).sum())
    return Vh[:r].T.to(DEV)                                                     # (d, r)


def pairs(task, n, k, g, split, var='S'):
    b = task.sample(n, g, split); s_ = task.sample(n, g, split); b[:, 4] = k; s_[:, 4] = k
    sb = torch.where(b[:, 0] == 0, b[:, 1] + b[:, 2], b[:, 1] - b[:, 2]) % P
    ss = torch.where(s_[:, 0] == 0, s_[:, 1] + s_[:, 2], s_[:, 1] - s_[:, 2]) % P
    # X / Z: one operand takes the source's value, s' = x_src op_base z_base (or x_base op_base z_src); pairs where s'
    # equals the base's s or the source's s are dropped, so neither a copy-free s patch nor no effect can score
    sx = torch.where(b[:, 0] == 0, s_[:, 1] + b[:, 2], s_[:, 1] - b[:, 2]) % P
    sz = torch.where(b[:, 0] == 0, b[:, 1] + s_[:, 2], b[:, 1] - s_[:, 2]) % P
    sv = {'X': sx, 'Z': sz}.get(var, ss)
    ok = (T[sb] != T[ss]) & (b[:, 3] != s_[:, 3]) & (sv != sb) & ((sv != ss) | (var not in 'XZ'))
    b, s_, sb, ss, sv = b[ok], s_[ok], sb[ok], ss[ok], sv[ok]
    # S: the intermediate takes the source's value; W (DAS sanity check): w takes the source's value
    target = (T[sb] + s_[:, 3]) % P if var == 'W' else (T[sv] + b[:, 3]) % P
    return task.encode(b)[:, :-1].to(DEV), task.encode(s_)[:, :-1].to(DEV), target.to(DEV), task.encode(b)[:, -1].to(DEV)


def das(net, task, layer, pos, k, guard, g, var='S'):
    for p_ in net.parameters():
        p_.requires_grad_(False)
    d = net.emb.weight.shape[1]
    Q = torch.eye(d, device=DEV)
    if guard:
        C = copy_basis(net, task, layer, pos, k, g)
        Q = torch.linalg.svd(torch.eye(d, device=DEV) - C @ C.T)[0][:, :d - C.shape[1]]
    m = min(DAS_M, Q.shape[1])
    W = torch.nn.Parameter(torch.randn(Q.shape[1], m, device=DEV) * 0.1)
    opt = torch.optim.Adam([W], lr=1e-2)

    def run(bi, si, B):
        with torch.no_grad():
            hs = forward_to(net, si, layer)[:, pos]

        def patch(h):
            h = h.clone(); hb = h[:, pos]
            h[:, pos] = hb + (hs - hb) @ B @ B.T
            return h
        return forward_to(net, bi, layer, patch)[:, -1]

    for t in range(DAS_STEPS):
        bi, si, tgt, _ = pairs(task, 512 if not TINY else 64, k, g, 'train', var)
        B = Q @ torch.linalg.qr(W)[0]
        loss = F.cross_entropy(run(bi, si, B).float(), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        bi, si, tgt, base = pairs(task, 4096 if not TINY else 128, k, g, 'test', var)
        B = Q @ torch.linalg.qr(W)[0]; pred = run(bi, si, B).argmax(-1)
        iia = float((pred == tgt).float().mean()); kept = float((pred == base).float().mean())
        R = Q @ torch.linalg.qr(torch.randn(Q.shape[1], m, device=DEV))[0]; pr = run(bi, si, R).argmax(-1)
        rnd = float((pr == tgt).float().mean()); rnd_kept = float((pr == base).float().mean())
    return dict(var=var, layer=layer, pos=pos, k=k, guard=guard, m=m, complement=Q.shape[1], iia=round(iia, 4),
                base_kept=round(kept, 4), rand_iia=round(rnd, 4), rand_base_kept=round(rnd_kept, 4))


@torch.no_grad()
def guard_check(net, task, g):
    """Known-answer check of the copy guard before any DAS (uses x, z, w only, never s). For each visible input whose
    class-mean code is clearly present (energy >= 5x its shuffled-label floor), the energy left outside the span must
    be <= 4x that floor (noise-sized; pilot 3's raw-fraction rule failed seed 1 on a copy at 4x noise). The rank must
    be >= 5 and below its maximum (pilot 2's guard was always at it). Returns (ok, rows)."""
    rows, ok = [], True
    for layer, pos, k, labs in [(4, 3, 6, 'xz'), (1, 3, 6, 'xz'), (1, 10, 6, 'xzw')]:
        C = copy_basis(net, task, layer, pos, k, g).cpu()
        c = task.sample(16384 if not TINY else 256, g); c[:, 4] = k
        h = forward_to(net, task.encode(c)[:, :-1].to(DEV), layer)[:, pos].float().cpu()
        rmax = 22 * len(labs) + 1
        row = dict(layer=layer, pos=pos, k=k, rank=C.shape[1], rank_max=rmax)

        def cm(v):
            M = torch.stack([h[v == i].mean(0) for i in range(P)]); return M - M.mean(0)
        for name in labs:
            v = c[:, {'x': 1, 'z': 2, 'w': 3}[name]]
            M, Mn = cm(v), cm(v[torch.randperm(len(v), generator=g)])
            e, noise, ins = float((M ** 2).sum()), float((Mn ** 2).sum()), float(((M @ C) ** 2).sum())
            row[name] = dict(signal_x_noise=round(e / noise, 1), outside_x_noise=round((e - ins) / noise, 2))
            if e >= 5 * noise:
                ok &= (e - ins) <= 4 * noise
        ok &= 5 <= C.shape[1] <= rmax - 3
        rows.append(row); log('  guard check', row)
    return (ok or TINY), rows


def positions(k):
    semi = 9 - k
    return sorted({semi, (semi + 9) // 2, 9, 10, 11})                           # ';', a middle filler, last pre-w, w, '='


if __name__ == '__main__':
    cells = os.environ.get('CELLS', 'std:pos:0,loop:pos:0,std:neg:0').split(',')
    log('v230 pilot | device %s | TINY %s | steps %d | lr %g | DAS m %d, %d steps | cells %s'
        % (DEV, TINY, STEPS, LR, DAS_M, DAS_STEPS, cells))
    for c in cells:
        # arch:kind:seed[:scan[:steps]] (per-cell overrides; the LR schedule depends on STEPS)
        f = c.split(':'); arch, kind, seed = f[0], f[1], int(f[2])
        scan = f[3] if len(f) > 3 else SCAN
        STEPS = 40 if TINY else int(f[4]) if len(f) > 4 else int(os.environ.get('STEPS', '30000'))
        net, task, tr = train(arch, kind, seed)
        res = dict(arch=arch, kind=kind, seed=seed, train=tr, das=[])
        if kind == 'pos' and (tr['final_acc'] or 0) >= (0 if TINY else 0.95):
            g = torch.Generator().manual_seed(7000 + seed); t0 = time.time()
            gok, res['guard_check'] = guard_check(net, task, g); guards = GUARDS
            if not gok:
                log('  GUARD CHECK FAILED: no guarded DAS for this cell'); guards = tuple(gd for gd in GUARDS if not gd)
            # known answers: each input at its own token in the embedding (layer 0) must give IIA ~1; W at layer 1 is
            # pilots 3-4's sanity (0.90-0.93 on seed 0, 0.52 on seed 2, where '=' may read w within block 1)
            res['das_sanity'] = []
            for var, layer, pos in [('W', 0, 10), ('X', 0, 0), ('Z', 0, 2), ('W', 1, 10)]:
                r = das(net, task, layer, pos, 6, False, g, var); res['das_sanity'].append(r)
                log('  das sanity (%s at its token, layer %d) | IIA %.3f (base kept %.3f) | random %.3f'
                    % (var, layer, r['iia'], r['base_kept'], r['rand_iia']))
            if min(r['iia'] for r in res['das_sanity'][:3]) < (0 if TINY else 0.9):
                log('  DAS KNOWN-ANSWER CHECK FAILED (layer-0 IIA < 0.90)')
            # X / Z at ';' (unguarded): is the answer moved by a single operand there? If ';' holds s and not the
            # operands for later reads, X and Z cannot score (the patch would have to compute s' itself)
            for k in (6, 0):
                for var in 'XZ':
                    for layer in range(1, n_layers(net) + 1):
                        r = das(net, task, layer, 9 - k, k, False, g, var); res['das'].append(r)
                        log('  das %s k %d pos %2d layer %d %-7s | IIA %.3f (base kept %.3f) | random %.3f (kept %.3f)'
                            % (var, k, 9 - k, layer, 'noguard', r['iia'], r['base_kept'], r['rand_iia'], r['rand_base_kept']))
            for k in (6, 0) if scan != 'xz' else ():
                for pos in positions(k):
                    for layer in range(1, n_layers(net) + 1):
                        # SCAN=key: only ';' at layers 3-4 and '=' at layer 5 (the cells with any DAS signal in pilots 1-3)
                        if scan == 'key' and (pos, layer) not in {(9 - k, 3), (9 - k, 4), (11, 5)}:
                            continue
                        for guard in guards:
                            r = das(net, task, layer, pos, k, guard, g); res['das'].append(r)
                            log('  das S k %d pos %2d layer %d %-7s | IIA %.3f (base kept %.3f) | random %.3f (kept %.3f) | complement %d'
                                % (k, pos, layer,'guard' if guard else 'noguard', r['iia'], r['base_kept'],
                                   r['rand_iia'], r['rand_base_kept'], r['complement']))
            log('  das done in %d s' % (time.time() - t0))
        json.dump(res, open(os.path.join(OUT, 'v230_pilot_%s_%s_s%d.json' % (arch, kind, seed)), 'w'))
