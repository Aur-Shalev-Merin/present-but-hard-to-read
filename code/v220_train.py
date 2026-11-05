"""v220 pilot: do small transformers learn two-step arithmetic answer-only, and at which learning rate?
(design: docs/v220_design.md). Learnability only: accuracy curves and steps to criterion. No probe or ladder statistic.

Tasks (answer-only; loss on the answer tokens only):
  dig  "AB+CD-EF=" or "AB-CD-EF=" -> two answer digits; a, b, c in 10..99 and the answer in 10..99 (rejection sampling).
       Fresh problems every step (about 1.5M possible problems); a fixed held-out set of 4,096 for evaluation.
  mod  "a+b-c=" or "a-b-c=" mod 23, one token per number, one answer token. 20% of the (op, a, b, c) combinations are
       held out of training (seed 220) and form the evaluation set, so accuracy measures generalisation, not recall.
Models (pre-LN GPT-2 style, learned positions, d = 192, 6 heads, MLP 4d):
  std   6 distinct layers;   loop  one block of 2 layers applied 3 times (weight-tied, same FLOPs, no input injection);
  std2  2 distinct layers applied once (the loop block without reuse: the same parameters as loop, a third of the FLOPs).
Training: AdamW (0.9, 0.98), weight decay 0.1, batch 256, warmup 500, cosine to 10%, fp16 autocast on GPU.
Evaluation every 500 steps (overall, plus and minus problems separately). A run stops 2,000 steps after reaching 0.99,
or at V220_STEPS (default 30,000).
Env: V220_CELLS "arch:fmt:lr:seed,..." (default: the 12-cell pilot grid), V220_STEPS, TINY=1 (tiny model, 60 steps, CPU),
OUT_DIR. Output: v220_pilot_<arch>_<fmt>_<lr>_s<seed>.json per cell, and v220_log.txt.
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get('OUT_DIR', '.'); TINY = os.environ.get('TINY') == '1'
STEPS = 60 if TINY else int(os.environ.get('V220_STEPS', '30000'))
EVAL_EVERY = 20 if TINY else 500
DEV = 'cuda' if torch.cuda.is_available() and not TINY else 'cpu'
P = 23
GRID = ','.join('%s:%s:%s:0' % (a, f, lr) for a in ('std', 'loop') for f in ('dig', 'mod') for lr in ('3e-4', '1e-3', '3e-3'))


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v220_log.txt'), 'a').write(s + '\n')


# ---------------------------------------------------------------- tasks
class Dig:
    """Tokens 0-9 digits, 10 '+', 11 '-', 12 '='. Input: a1 a2 op b1 b2 - c1 c2 = ; answer d1 d2."""
    V, L_IN, L_ANS = 13, 9, 2

    def sample(self, n, g):
        out = []
        while sum(len(o) for o in out) < n:
            m = 2 * n
            a, b, c = (torch.randint(10, 100, (m,), generator=g) for _ in range(3))
            op = torch.randint(0, 2, (m,), generator=g)                        # 0: a+b-c, 1: a-b-c
            r = torch.where(op == 0, a + b - c, a - b - c); ok = (r >= 10) & (r <= 99)
            out.append(torch.stack([a, b, c, op, r], 1)[ok])
        return torch.cat(out)[:n]

    def encode(self, t):
        a, b, c, op, r = t.unbind(1); d = lambda x: [x // 10, x % 10]
        return torch.stack(d(a) + [10 + op] + d(b) + [torch.full_like(a, 11)] + d(c) + [torch.full_like(a, 12)] + d(r), 1)


class Mod:
    """Tokens 0-22 numbers, 23 '+', 24 '-', 25 '='. Input: a op b - c = ; answer (a op b - c) mod 23."""
    V, L_IN, L_ANS = 26, 6, 1

    def __init__(self):
        allc = np.array([(op, a, b, c) for op in range(2) for a in range(P) for b in range(P) for c in range(P)])
        perm = np.random.default_rng(220).permutation(len(allc)); k = int(0.2 * len(allc))
        self.test, self.train = torch.tensor(allc[perm[:k]]), torch.tensor(allc[perm[k:]])

    def sample(self, n, g, split='train'):
        src = self.train if split == 'train' else self.test
        x = src[torch.randint(0, len(src), (n,), generator=g)] if split == 'train' else src
        op, a, b, c = x.unbind(1); r = torch.where(op == 0, a + b - c, a - b - c) % P
        return torch.stack([a, b, c, op, r], 1)

    def encode(self, t):
        a, b, c, op, r = t.unbind(1)
        return torch.stack([a, 23 + op, b, torch.full_like(a, 24), c, torch.full_like(a, 25), r], 1)


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); self.h = h

    def forward(self, x):
        B, T, D = x.shape; q, k, v = self.qkv(self.ln1(x)).split(D, 2)
        sh = lambda z: z.view(B, T, self.h, D // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.mlp(self.ln2(x))


class Net(nn.Module):
    def __init__(self, V, T, arch, d=192, h=6, n_std=6, n_blk=2, loops=3):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Parameter(torch.zeros(1, T, d))
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_std if arch == 'std' else n_blk)])
        self.loops = loops if arch == 'loop' else 1; self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, idx):
        x = self.emb(idx) + self.pos[:, :idx.shape[1]]
        for _ in range(self.loops):
            for b in self.blocks:
                x = b(x)
        return self.head(self.lnf(x))


# ---------------------------------------------------------------- training
def run_cell(arch, fmt, lr, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(1000 + seed)
    task = Dig() if fmt == 'dig' else Mod(); T = task.L_IN + task.L_ANS
    kw = dict(d=32, h=2, n_std=2, n_blk=1, loops=2) if TINY else {}
    net = Net(task.V, T, arch, **kw).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.1)
    sched = lambda s: min(1.0, (s + 1) / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
    scaler = torch.amp.GradScaler(enabled=DEV == 'cuda')
    ev = task.encode(task.sample(4096, torch.Generator().manual_seed(9220)) if fmt == 'dig' else task.sample(0, g, 'test')).to(DEV)
    curve, hit, t0, bs = [], None, time.time(), 32 if TINY else 256

    def evaluate():
        net.eval()
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=DEV == 'cuda'):
            lg = net(ev[:, :-1])[:, task.L_IN - 1:]
        pred = lg.argmax(-1); ok = (pred == ev[:, task.L_IN:]).all(1).float()
        opcol = ev[:, 2] - 10 if fmt == 'dig' else ev[:, 1] - 23
        net.train()
        return float(ok.mean()), float(ok[opcol == 0].mean()), float(ok[opcol == 1].mean())

    for s in range(STEPS):
        for gr in opt.param_groups:
            gr['lr'] = lr * sched(s)
        x = task.encode(task.sample(bs, g)).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == 'cuda'):
            lg = net(x[:, :-1])[:, task.L_IN - 1:]
            loss = F.cross_entropy(lg.float().reshape(-1, task.V), x[:, task.L_IN:].reshape(-1))
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update()
        if (s + 1) % EVAL_EVERY == 0:
            acc, acc_p, acc_m = evaluate(); curve.append([s + 1, round(float(loss), 4), acc, acc_p, acc_m])
            if hit is None and acc >= 0.99:
                hit = s + 1
            if not math.isfinite(float(loss)) or (hit is not None and s + 1 >= hit + 2000):
                break
    res = dict(arch=arch, fmt=fmt, lr=lr, seed=seed, params=n_par, steps_run=s + 1, step_to_0p99=hit,
               step_to_0p95=next((c[0] for c in curve if c[2] >= 0.95), None), final=curve[-1] if curve else None,
               curve=curve, seconds=round(time.time() - t0), device=DEV)
    json.dump(res, open(os.path.join(OUT, 'v220_pilot_%s_%s_%s_s%d.json' % (arch, fmt, lr, seed)), 'w'))
    log('%-4s %-3s lr %-6g seed %d | params %d | to 0.95: %s, to 0.99: %s | final acc %.3f (plus %.3f, minus %.3f) | %d steps, %d s'
        % (arch, fmt, lr, seed, n_par, res['step_to_0p95'], hit, *(res['final'][2:5] if curve else (float('nan'),) * 3),
           s + 1, res['seconds']))


if __name__ == '__main__':
    cells = os.environ.get('V220_CELLS', GRID).split(',')
    log('v220 pilot | device %s | TINY %s | steps %d | %d cells' % (DEV, TINY, STEPS, len(cells)))
    for c in cells:
        a, f, lr, sd = c.split(':'); run_cell(a, f, float(lr), int(sd))
