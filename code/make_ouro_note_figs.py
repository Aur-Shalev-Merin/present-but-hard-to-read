"""Figures for the short Ouro note (paper_ouro_note/). Run: python make_ouro_note_figs.py

Fig 1 (answer-only accuracy vs loops) uses the counts reported in the result docs (v176b/v177 ran on Colab; their
outputs were pasted into docs/v177_plan.md, so the numbers are copied here with their sources).
Fig 2 (where the copied digit can be read) is parsed from results_v180/ and results_v181/ analysis files.
Fig 3 (the computation test) is parsed from results_v188/cpu/v188_analysis.txt.
"""
import os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = 'paper_ouro_note'
os.makedirs(OUT, exist_ok=True)


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


# --- Fig 1 -----------------------------------------------------------------------------------------------------------
# docs/v177_plan.md "v176b RESULT": GSM8K first 200 test problems, fixed scorer
ouro = ([1, 2, 3, 4], [0.065, 0.200, 0.225, 0.230], 200)
huginn = ([1, 2, 4, 8, 16, 32], [0.005, 0.015, 0.005, 0.030, 0.045, 0.055], 200)
# GSM-Symbolic (docs/v177_plan.md, prereg_v178) is reported in the note's Appendix A text only; its panel was cut (v7.2).

fig, a = plt.subplots(1, 1, figsize=(4.3, 3.1))
for (x, y, n), lab, c, m in ((ouro, 'Ouro-1.4B', 'C0', 'o'), (huginn, 'Huginn-3.5B', 'C3', 's')):
    lo, hi = wilson(np.array(y), n)
    a.errorbar(x, 100 * np.array(y), yerr=[100 * (np.array(y) - lo), 100 * (hi - np.array(y))], marker=m, color=c,
               capsize=2, label=lab)
a.set_xscale('log', base=2); a.set_xticks([1, 2, 3, 4, 8, 16, 32]); a.set_xticklabels(['1', '2', '3', '4', '8', '16', '32'])
a.set_xlabel('loops'); a.set_ylabel('answer-only accuracy (%)'); a.set_title('GSM8K (n = 200)', fontsize=9)
a.legend(fontsize=7.5, frameon=False)
a.spines[['top', 'right']].set_visible(False); a.set_ylim(bottom=0)
fig.tight_layout(); [fig.savefig(os.path.join(OUT, 'fig1_accuracy.' + e), dpi=200) for e in ('png', 'pdf')]


# --- Fig 2 -----------------------------------------------------------------------------------------------------------
def rows(path):
    return [l.rstrip('\n') for l in open(path, encoding='utf-8')]


STATES = ['h0', 'mid1', 'end1', 'end2', 'end3', 'end4']
v180 = rows('results_v180/v180_analysis.txt')
bare = {}
op = None
for l in v180:
    m = re.match(r'== (add|mul) ', l)
    if m:
        op = m.group(1)
    m = re.match(r'([ab]) @ last\s+((?:[\d.]+\s+){5}[\d.]+)', l)
    if m and op:
        bare[(op, m.group(1))] = [float(v) for v in m.group(2).split()]
bare_mean = np.mean([bare[k] for k in bare], 0)
v181 = rows('results_v181/v181_analysis.txt')
std, con = {}, {}
for l in v181:
    m = re.match(r'(\+0|\+1|\+3|sent|qmark|last)\s+(.*)$', l)
    if m and '/' in l:
        cells = re.findall(r'([\d.]+) / ([\d.]+)', m.group(2))
        pad = [np.nan] * (6 - len(cells))
        std[m.group(1)] = pad + [float(a) for a, b in cells]; con[m.group(1)] = pad + [float(b) for a, b in cells]

v184 = {}
for l in rows('results_v184/cpu/v184_analyze.txt'):
    m = re.match(r'(irr|rel)_(start|end)\s+(contrastive|standard) @ last\s+((?:[\d.]+\s+){4}[\d.]+)', l)
    if m:
        v184[(m.group(1), m.group(2), m.group(3))] = [float(v) for v in m.group(4).split()]
for l in rows('results_v185/cpu/v185_analyze.txt'):          # colour-word families (v185)
    m = re.match(r'(col)_(start|end)\s+(contrastive|standard) @ last\s+((?:[\d.]+\s+){4}[\d.]+)', l)
    if m:
        v184[(m.group(1), m.group(2), m.group(3))] = [float(v) for v in m.group(4).split()]

fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.1))
x = np.arange(6)
ax[0].plot(x, bare_mean, 'o-', color='C2', label='bare arithmetic, standard probe')
ax[0].plot(x[1:], con['last'][1:], 's-', color='C0', label='word problem, minimal-pair probe')
ax[0].plot(x[1:], std['last'][1:], '^-', color='C3', label='word problem, standard probe')
ax[0].axhline(0.10, color='grey', lw=0.8, ls=':'); ax[0].text(5, 0.12, 'chance', fontsize=7, color='grey', ha='right')
ax[0].set_xticks(x); ax[0].set_xticklabels(STATES); ax[0].set_ylim(0, 1.05)
ax[0].set_ylabel('accuracy, copied units digit'); ax[0].set_title('last prompt token, by loop', fontsize=9)
ax[0].legend(fontsize=6.5, frameon=False, loc='center right')
slots = ['+0', '+1', '+3', 'sent', 'qmark', 'last']; names = ['digit', '+1', '+3', 'sent. end', "'?'", 'last']
w = 0.38
ax[1].bar(np.arange(6) - w / 2, [std[s][5] for s in slots], w, color='C3', label='standard probe')
ax[1].bar(np.arange(6) + w / 2, [con[s][5] for s in slots], w, color='C0', label='minimal-pair probe')
ax[1].axhline(0.10, color='grey', lw=0.8, ls=':')
ax[1].set_xticks(np.arange(6)); ax[1].set_xticklabels(names, fontsize=8); ax[1].set_ylim(0, 1.05)
ax[1].set_title('word problems, after loop 4, by position', fontsize=9); ax[1].legend(fontsize=7, frameon=False)
xs = np.arange(1, 6)
for kind, c, m, lab in (('rel', 'C0', 's', 'number the problem needs'), ('irr', 'C1', 'D', 'unrelated number'),
                        ('col', 'C4', 'o', 'colour word')):
    ax[2].plot(xs, np.mean([v184[(kind, w, 'contrastive')] for w in ('start', 'end')], 0), m + '-', color=c,
               label=lab + ', minimal-pair')
    ax[2].plot(xs, np.mean([v184[(kind, w, 'standard')] for w in ('start', 'end')], 0), m + ':', color=c, alpha=0.7,
               label=lab + ', standard')
ax[2].axhline(0.10, color='grey', lw=0.8, ls=':')
ax[2].set_xticks(xs); ax[2].set_xticklabels(STATES[1:]); ax[2].set_ylim(0, 1.05)
ax[2].set_title('last prompt token: which token is varied', fontsize=9); ax[2].legend(fontsize=6.3, frameon=False, loc='center right')
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); [fig.savefig(os.path.join(OUT, 'fig2_readout.' + e), dpi=200) for e in ('png', 'pdf')]

# --- Fig 4 -----------------------------------------------------------------------------------------------------------
# referee check: all 31 mod-5 forms x*a + y*b + z*c at P1 after loop 4, excess over own copy ceiling, both studies
def pretty(s):
    if s in ('a+b', 'a-b', 'a+c', 'ans', 'ans-c', 'ans+2c'):
        return s
    co = [int(v) for v in re.match(r'(\d)a\+(\d)b\+(\d)c', s).groups()]
    out = ''
    for c, v in zip(co, 'abc'):
        if c == 0:
            continue
        c = c - 5 if c > 2 else c
        term = ('' if abs(c) == 1 else str(abs(c))) + v
        out += ('-' if c < 0 else ('+' if out else '')) + term
    return out


import v188_line_check as L                                     # line structure + the fast cluster bootstrap


def fname(f):
    f = tuple(int(x) for x in f)
    if f in L.PARTIAL.values():
        x = [k for k, g in L.PARTIAL.items() if g == f][0]
        return '%s = ans%s%s' % (L.PNAME[x], '+' if x == 'c' else '-', x)
    return L.lname(f) if f in L.ON else pretty('%da+%db+%dc' % f)


LCOL = {'c': ('C0', '#9ecae1'), 'b': ('C1', '#fdd0a2'), 'a': ('C2', '#c7e9c0')}       # (partial, rest of line)
fig, ax = plt.subplots(1, 2, figsize=(8.2, 5.6), sharex=True)
for a, (v, title) in zip(ax, ((188, 'Study 2 (new templates)'), (186, 'Study 1'))):
    z = np.load('results_v188/line/v188_line_items_%d.npz' % v)
    ex = z['state'][0].astype(float) - z['ceil'][0]                     # end4, frozen ceiling
    forms = [tuple(int(x) for x in f) for f in z['forms']]
    keep = [i for i, f in enumerate(forms) if f != L.ANS and f not in L.COPIES]
    keep.sort(key=lambda i: -ex[i].mean())
    cis = [L.cluster_ci(z['fam'], ex[i], n=2000) for i in keep]
    col = []
    for i in keep:
        f = forms[i]; x = [k for k, l in L.LINES.items() if f in {g for _, g in l}]
        col.append('C7' if not x else LCOL[x[0]][0] if f == L.PARTIAL[x[0]] else LCOL[x[0]][1])
    ys = np.arange(len(keep))[::-1]
    a.barh(ys, [c[0] for c in cis], color=col, xerr=[[c[0] - c[1] for c in cis], [c[2] - c[0] for c in cis]],
           error_kw=dict(lw=0.6, capsize=1.2))
    a.set_yticks(ys); a.set_yticklabels([fname(forms[i]) for i in keep], fontsize=6.3); a.axvline(0, color='k', lw=0.6)
    ans = ex[forms.index(L.ANS)].mean()
    a.set_title('%s; answer digit (not shown) +%.2f' % (title, ans), fontsize=8.5)
    a.spines[['top', 'right']].set_visible(False); a.tick_params(axis='x', labelsize=7.5)
    a.set_xlabel('accuracy above copy ceiling (mod 5)', fontsize=8)
fig.tight_layout(); [fig.savefig(os.path.join(OUT, 'fig4_allforms.' + e), dpi=200) for e in ('png', 'pdf')]

# --- Fig 3 -----------------------------------------------------------------------------------------------------------
# v188 (pre-registered confirmation, 12 new templates): excess over each label's own copy ceiling, mod 5, end of loop 4
ex = {}
for l in rows('results_v188/cpu/v188_analysis.txt'):
    m = re.match(r'(H1|H2) excess: (.*)$', l)
    if m:
        ex[m.group(1)] = dict((k, float(v)) for k, v in re.findall(r'(\w+) ([+-][\d.]+)', m.group(2)))
NAMES = {'m5': 'a+b\n(target)', 'amb5': 'a-b', 'ap2b5': 'a+2b', 'tapb5': '2a+b', 'apc5': 'a+c', 'abmc5': 'ans-c',
         'abc5': 'ans+2c', 'rnd5': 'random\nf(a,b)'}
fig, ax = plt.subplots(1, 2, figsize=(8.2, 2.9), gridspec_kw={'width_ratios': [5, 6]})
for a, h, title in ((ax[0], 'H1', 'bare addition, last prompt token'),
                    (ax[1], 'H2', 'word problems, just before the units digit')):
    ks = ['m5'] + [k for k in ex[h] if k not in ('m5', 'rnd5')] + ['rnd5']
    a.bar(range(len(ks)), [ex[h][k] for k in ks], color=['C0'] + ['C7'] * (len(ks) - 2) + ['C3'])
    a.axhline(0, color='k', lw=0.6)
    a.set_xticks(range(len(ks))); a.set_xticklabels([NAMES[k] for k in ks], fontsize=7.5)
    a.set_title(title, fontsize=9); a.spines[['top', 'right']].set_visible(False)
ax[0].set_ylabel('accuracy above copy ceiling\n(mod 5, after loop 4)', fontsize=8)
fig.tight_layout(); [fig.savefig(os.path.join(OUT, 'fig3_computation.' + e), dpi=200) for e in ('png', 'pdf')]
print('v188 excess:', ex)
print('bare-arith mean at last:', np.round(bare_mean, 3).tolist())
print('v181 last std/con:', std['last'], con['last'])
print('wrote', OUT)
