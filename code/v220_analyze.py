"""v220 locked analysis (prereg docs/prereg_v220_training_order.md, section 4). CPU, seconds.
Usage: python v220_analyze.py <dir with v220_study_*.json> [...]   |   python v220_analyze.py --selftest

Per run and checkpoint t, at a position (layers = every block application):
  I(t)    max over layers of mean_op(e_p1 - e_twin); D+, D- = e_p1 - e_twin on op 0 / op 1 at that layer
  dA(t)   max over layers of mean_op e_ans, minus the same at step 0
  m(t)    model-output accuracy (mean over ops) - max(copy ceiling of ans, 1/K)
  dX(t)   for X in p1, ac, bc: max over layers of mean_op e_X, minus the same at step 0
Every series is smoothed by a centred running median of 5 checkpoints. A "run" is >= 3 consecutive checkpoints.
Run labels (positions out [primary] and c):
  AHEAD          a run with I >= 0.05, D+ > 0, D- > 0, dA <= 0.02 and m <= 0.02
  WITH-OR-AFTER  no such run, but a run with I >= 0.05, D+ > 0, D- > 0
  NEVER          otherwise
Cell label (arch, fmt, position): the label of >= 2 of the 3 seeds, else SPLIT.
Ranking (descriptive label, positions c and out): t_X = first step of a run with dX >= 0.05; FIRST = the earliest X if
the next X starts >= 3 checkpoints later (or never), TIED otherwise, NONE if no X reaches it; cell = majority, else SPLIT.
Times: t_I (first run of I >= 0.05 with D+, D- > 0), t_A (first run of dA >= 0.05), t_m (first run of m >= 0.05),
lead = (t_A - t_I) / t_A. Architecture comparison (descriptive): LOOP-DIFFERS if loop's seed range of lead is disjoint
from both std's and std2's ranges on the same side, else NO-DIFFERENCE (NA if any lead is undefined).
"""
import glob, json, os, sys
import numpy as np

TH_I, TH_ABS, TH_PRES, RUNLEN, GAP = 0.05, 0.02, 0.05, 3, 3
POSI = {'b': 0, 'c': 1, 'out': 2}


def smooth(x, w=5):
    x = np.asarray(x, float); h = w // 2
    return np.array([np.median(x[max(0, i - h):i + h + 1]) for i in range(len(x))])


def first_run(mask, n=RUNLEN):
    c = 0
    for i, v in enumerate(mask):
        c = c + 1 if v else 0
        if c >= n:
            return i - n + 1
    return None


def series(r, pos):
    ex = np.array([m['ex'] for m in r['meas']])[:, POSI[pos]]          # [T, depth, op, lab]
    steps = np.array([m['step'] for m in r['meas']])
    Il = (ex[..., 0] - ex[..., 1])                                     # [T, depth, op]
    Im = Il.mean(-1); arg = Im.argmax(1); t = np.arange(len(ex))
    I, Dp, Dm = Im[t, arg], Il[t, arg, 0], Il[t, arg, 1]
    mx = ex.mean(2).max(1)                                             # [T, lab] max over layers of mean_op
    K = 5 if r['fmt'] == 'dig' else 23
    ceil_ans = np.mean([r['ceil']['out'][o][4] for o in (0, 1)])
    m = np.array([np.mean(mm['model_acc']) for mm in r['meas']]) - max(ceil_ans, 1.0 / K)
    s = dict(steps=steps, I=smooth(I), Dp=smooth(Dp), Dm=smooth(Dm), dA=smooth(mx[:, 4] - mx[0, 4]), m=smooth(m))
    for j, X in ((0, 'p1'), (2, 'ac'), (3, 'bc')):
        s['d' + X] = smooth(mx[:, j] - mx[0, j])
    return s


def run_label(s):
    win = (s['I'] >= TH_I) & (s['Dp'] > 0) & (s['Dm'] > 0)
    absent = (s['dA'] <= TH_ABS) & (s['m'] <= TH_ABS)
    st = lambda i: None if i is None else int(s['steps'][i])
    iI, iA, im = first_run(win), first_run(s['dA'] >= TH_PRES), first_run(s['m'] >= TH_PRES)
    lab = 'AHEAD' if first_run(win & absent) is not None else ('WITH-OR-AFTER' if iI is not None else 'NEVER')
    tI, tA = st(iI), st(iA)
    lead = (tA - tI) / tA if (tI is not None and tA) else None
    return dict(label=lab, t_I=tI, t_A=tA, t_m=st(im), lead=lead, ahead_at=st(first_run(win & absent)))


def ranking(s):
    t = {X: first_run(s['d' + X] >= TH_PRES) for X in ('p1', 'ac', 'bc')}
    got = sorted((v, X) for X, v in t.items() if v is not None)
    if not got:
        return 'NONE', t
    if len(got) == 1 or got[1][0] - got[0][0] >= GAP:
        return got[0][1].upper() + '-FIRST', t
    return 'TIED', t


def majority(labels):
    for l in set(labels):
        if labels.count(l) >= 2:
            return l
    return 'SPLIT'


def analyze(runs, out=print):
    cells = {}
    for r in runs:
        cells.setdefault((r['fmt'], r['arch']), []).append(r)
    summary = {}
    for (fmt, arch), rs in sorted(cells.items()):
        rs = sorted(rs, key=lambda r: r['seed'])
        for pos in ('out', 'c'):
            per = [run_label(series(r, pos)) for r in rs]
            rk = [ranking(series(r, pos)) for r in rs]
            cl = majority([p['label'] for p in per]); rl = majority([k[0] for k in rk])
            summary[(fmt, arch, pos)] = dict(label=cl, runs=per, ranking=rl)
            out('%s %-4s %-3s | %-13s | seeds %s | t_I %s t_A %s t_m %s | lead %s | ranking %s %s'
                % (fmt, arch, pos, cl, [p['label'] for p in per], [p['t_I'] for p in per], [p['t_A'] for p in per],
                   [p['t_m'] for p in per], ['%.2f' % p['lead'] if p['lead'] is not None else None for p in per],
                   rl, [k[0] for k in rk]))
        sb = [series(r, 'b') for r in rs]
        out('%s %-4s b   | t_I(b) %s (descriptive; c not yet seen)'
            % (fmt, arch, [None if first_run((s['I'] >= TH_I) & (s['Dp'] > 0) & (s['Dm'] > 0)) is None else
                           int(s['steps'][first_run((s['I'] >= TH_I) & (s['Dp'] > 0) & (s['Dm'] > 0))]) for s in sb]))
    for fmt in ('dig', 'mod'):
        L = {a: [p['lead'] for p in summary.get((fmt, a, 'out'), {}).get('runs', [])] for a in ('std', 'loop', 'std2')}
        if any(len(v) == 0 or None in v for v in L.values()):
            out('%s architecture comparison: NA (leads %s)' % (fmt, L)); continue
        rng = {a: (min(v), max(v)) for a, v in L.items()}
        above = all(rng['loop'][0] > rng[a][1] for a in ('std', 'std2'))
        below = all(rng['loop'][1] < rng[a][0] for a in ('std', 'std2'))
        out('%s architecture comparison: %s (lead ranges %s)'
            % (fmt, 'LOOP-DIFFERS' if above or below else 'NO-DIFFERENCE', {a: tuple(round(x, 2) for x in v) for a, v in rng.items()}))
    return summary


# ---------------------------------------------------------------- selftest (synthetic curves)
def _fake(world, seed, fmt='dig', T=301, every=100):
    rng = np.random.default_rng(seed); steps = np.arange(T) * every
    ramp = lambda at: np.clip((steps - at) / 1500, 0, 1)
    I = np.zeros(T); A = np.zeros(T); M = np.zeros(T); Dm_flip = False; base_A = 0.0
    if world == 'ahead':
        I, A, M = 0.3 * ramp(3000), 0.6 * ramp(9000), 0.7 * ramp(9000)
    elif world == 'with':
        I, A, M = 0.3 * ramp(5000), 0.6 * ramp(5000), 0.7 * ramp(5000)
    elif world == 'after':
        I, A, M = 0.3 * ramp(9000), 0.6 * ramp(5000), 0.7 * ramp(5000)
    elif world == 'never':
        A, M = 0.6 * ramp(5000), 0.7 * ramp(5000)
    elif world == 'spike':
        I, A, M = 0.3 * ramp(5000), 0.6 * ramp(5000), 0.7 * ramp(5000)
        I[np.arange(T) % 10 == 3] += 0.2                                # isolated spikes before the answer
    elif world == 'onesided':
        I, A, M = 0.3 * ramp(3000), 0.6 * ramp(9000), 0.7 * ramp(9000); Dm_flip = True
    elif world == 'baseline':
        I, A, M = 0.3 * ramp(3000), 0.6 * ramp(9000), 0.7 * ramp(9000); base_A = 0.06
    meas = []
    for i in range(T):
        ex = np.zeros((3, 2, 2, 5)) + rng.normal(0, 0.01, (3, 2, 2, 5))
        for p in range(3):
            ex[p, 0, 0, 0] += I[i] * (1.6 if Dm_flip else 1.0)          # D+ = e_p1 - e_twin on op 0
            ex[p, 0, 1, 0] += -0.1 * min(1, I[i] / 0.1) if Dm_flip else I[i]
            ex[p, 1, :, 4] += A[i] + base_A
            ex[p, 1, :, 2] += 0.3 * ramp(12000)[i]; ex[p, 1, :, 3] += 0.3 * ramp(12000)[i]   # ac, bc later than p1
        meas.append(dict(step=int(steps[i]), ex=ex.tolist(), model_acc=[0.2 + M[i]] * 2))
    ceil = {k: [[0.2] * 5, [0.2] * 5] for k in ('b', 'c', 'out')}
    return dict(arch={'ahead': 'std', 'with': 'loop', 'after': 'std2'}.get(world, world), fmt=fmt, seed=seed,
                ceil=ceil, meas=meas)


def selftest():
    want = {'ahead': 'AHEAD', 'with': 'WITH-OR-AFTER', 'after': 'WITH-OR-AFTER', 'never': 'NEVER',
            'spike': 'WITH-OR-AFTER', 'onesided': 'NEVER', 'baseline': 'AHEAD'}
    ok = True
    for w, lab in want.items():
        got = [run_label(series(_fake(w, s), 'out'))['label'] for s in (1, 2, 3)]
        rk = ranking(series(_fake(w, 1), 'out'))[0]
        print('%-9s expect %-13s got %s | ranking %s' % (w, lab, got, rk)); ok &= all(g == lab for g in got)
    print('--- full analysis on the three-architecture fake set (ahead=std, with=loop, after=std2):')
    analyze([_fake(w, s) for w in ('ahead', 'with', 'after') for s in (1, 2, 3)])
    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        sys.exit(0 if selftest() else 1)
    runs = [json.load(open(f)) for d in sys.argv[1:] for f in sorted(glob.glob(os.path.join(d, '**', 'v220_study_*.json'), recursive=True))]
    print('%d runs' % len(runs)); analyze(runs)
