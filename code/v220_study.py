"""v220 study: train and measure at every checkpoint (prereg docs/prereg_v220_training_order.md).
Training as the pilot (v220_train.py: tasks, Net, AdamW (0.9, 0.98), wd 0.1, batch 256, warmup 500, cosine over 30,000
steps to 10%, fp16 autocast, clip 1.0), with no early stop: dig runs 30,000 steps, mod 10,000 (same 30,000-step
schedule as the pilot). Learning rates fixed from the pilots: dig std 3e-3, loop 3e-4, std2 3e-4; mod 1e-3 for all.
Measurement (v220_measure.measure) at step 0 and every 100 steps (dig) or 50 steps (mod), in fp32, eval mode.
Held-out accuracy (as the pilot) every 500 steps. fp16 weights saved at 40 log-spaced steps and the last step.
Env: V220_CELLS "arch:fmt:seed,..." (default: the 18 study cells), TINY=1 (tiny model, 60 steps, measure every 20, CPU),
OUT_DIR. Output: v220_study_<arch>_<fmt>_s<seed>.json and ckpt_<arch>_<fmt>_s<seed>/step<N>.pt.
"""
import json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
import v220_measure as M
from v220_train import Dig, Mod, Net

OUT = os.environ.get('OUT_DIR', '.'); TINY = os.environ.get('TINY') == '1'
DEV = 'cuda' if torch.cuda.is_available() and not TINY else 'cpu'
LR = {('std', 'dig'): 3e-3, ('loop', 'dig'): 3e-4, ('std2', 'dig'): 3e-4,
      ('std', 'mod'): 1e-3, ('loop', 'mod'): 1e-3, ('std2', 'mod'): 1e-3}
RUN = {'dig': 30000, 'mod': 10000}; EVERY = {'dig': 100, 'mod': 50}; SCHED = 30000
GRID = ','.join('%s:%s:%d' % (a, f, s) for f in ('dig', 'mod') for a in ('std', 'loop', 'std2') for s in (1, 2, 3))


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v220_study_log.txt'), 'a').write(s + '\n')


def run_cell(arch, fmt, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(1000 + seed)
    task = Dig() if fmt == 'dig' else Mod(); T = task.L_IN + task.L_ANS
    kw = dict(d=32, h=2, n_std=2, n_blk=1, loops=2) if TINY else {}
    net = Net(task.V, T, arch, **kw).to(DEV); lr = LR[(arch, fmt)]
    steps, every = (60, 20) if TINY else (RUN[fmt], EVERY[fmt]); sched_len = 60 if TINY else SCHED
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.1)
    sched = lambda s: min(1.0, (s + 1) / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, sched_len) / sched_len)))
    scaler = torch.amp.GradScaler(enabled=DEV == 'cuda')
    ev = task.encode(task.sample(4096, torch.Generator().manual_seed(9220)) if fmt == 'dig' else task.sample(0, g, 'test')).to(DEV)
    ps = M.probe_set(fmt); ceil = M.copy_ceilings(ps)
    save_at = set(int(x) for x in np.unique(np.round(np.logspace(1, math.log10(steps), 40)))) | {steps}
    tag = '%s_%s_s%d' % (arch, fmt, seed); cdir = os.path.join(OUT, 'ckpt_' + tag); os.makedirs(cdir, exist_ok=True)
    meas, curve, t0, bs = [], [], time.time(), 32 if TINY else 256

    def evaluate():
        net.eval()
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=DEV == 'cuda'):
            lg = net(ev[:, :-1])[:, task.L_IN - 1:]
        ok = (lg.argmax(-1) == ev[:, task.L_IN:]).all(1).float()
        opcol = ev[:, 2] - 10 if fmt == 'dig' else ev[:, 1] - 23
        net.train()
        return float(ok.mean()), float(ok[opcol == 0].mean()), float(ok[opcol == 1].mean())

    def record(step):
        ex, macc = M.measure(net, ps, ceil, fmt, DEV)
        meas.append(dict(step=step, ex=np.round(ex, 4).tolist(), model_acc=np.round(macc, 4).tolist()))

    record(0)
    for s in range(steps):
        for gr in opt.param_groups:
            gr['lr'] = lr * sched(s)
        x = task.encode(task.sample(bs, g)).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == 'cuda'):
            lg = net(x[:, :-1])[:, task.L_IN - 1:]
            loss = F.cross_entropy(lg.float().reshape(-1, task.V), x[:, task.L_IN:].reshape(-1))
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update()
        if (s + 1) % every == 0:
            record(s + 1)
        if (s + 1) % (20 if TINY else 500) == 0:
            curve.append([s + 1, round(float(loss), 4), *evaluate()])
        if s + 1 in save_at:
            torch.save({k: v.half() for k, v in net.state_dict().items()}, os.path.join(cdir, 'step%d.pt' % (s + 1)))
    res = dict(arch=arch, fmt=fmt, seed=seed, lr=lr, steps=steps, every=every, depth=len(meas[0]['ex'][0]),
               params=sum(p.numel() for p in net.parameters()), ceil={k: np.round(v, 4).tolist() for k, v in ceil.items()},
               labels=M.LABELS, positions=['b', 'c', 'out'], meas=meas, curve=curve, seconds=round(time.time() - t0),
               device=DEV)
    json.dump(res, open(os.path.join(OUT, 'v220_study_%s.json' % tag), 'w'))
    log('%-4s %s s%d lr %g | final acc %.3f (plus %.3f, minus %.3f) | %d checkpoints measured | %d s'
        % (arch, fmt, seed, lr, *curve[-1][2:5], len(meas), res['seconds']))


if __name__ == '__main__':
    cells = os.environ.get('V220_CELLS', GRID).split(',')
    log('v220 study | device %s | TINY %s | %d cells' % (DEV, TINY, len(cells)))
    for c in cells:
        a, f, sd = c.split(':'); run_cell(a, f, int(sd))
