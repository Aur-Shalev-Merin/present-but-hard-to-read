"""Many-question curve figure for the Ouro note (paper_ouro_note/fig5_curve.pdf). Run: python make_ouro_note_fig_curve.py

Parsed from the pre-registered B2/B3 analysis outputs: Qwen2.5-1.5B results_v205/bcpu/v205_B_analysis.txt (v205),
Ouro-1.4B results_v206/bcpu/v206_analysis.txt (v206). For each training-set size, the best accuracy over the
pre-registered states (Qwen layers 7-28; Ouro loop ends 1-4), separately for the PCA-256 probe and the full-state probe.
The horizontal lines are the B3 minimal-pair means over the same states.
"""
import os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = 'paper_ouro_note'


def curve(path):
    """N grid, best-over-states accuracy per probe, and the minimal-pair mean."""
    lines = open(path, encoding='utf-8').read().splitlines()
    n = [int(x) for x in next(l for l in lines if l.startswith('N ')).split()[1:]]
    acc = {'pca': [], 'full': []}
    for l in lines:
        m = re.match(r'^(\w+)\s+(pca|full)\s+(.*)$', l)
        if m and m.group(1) not in ('B2', 'B3'):
            acc[m.group(2)].append([float(v) for v in re.findall(r'(\d\.\d{3}) \(C', m.group(3))])
    mp = float(re.search(r'minimal-pair mean over Q ([\d.]+)', '\n'.join(lines)).group(1))
    return np.array(n), {k: np.max(np.array(v), axis=0) for k, v in acc.items()}, mp


fig, a = plt.subplots(1, 1, figsize=(4.6, 3.2))
for path, lab, c in (('results_v206/bcpu/v206_analysis.txt', 'Ouro-1.4B', 'C0'),
                     ('results_v205/bcpu/v205_B_analysis.txt', 'Qwen2.5-1.5B', 'C1')):
    n, acc, mp = curve(path)
    a.plot(n, acc['full'], marker='o', ms=3.5, color=c, label=lab + ', full-state probe')
    a.plot(n, acc['pca'], marker='o', ms=3, ls='--', color=c, alpha=0.7, label=lab + ', PCA-256 probe')
    a.axhline(mp, color=c, ls=':', lw=1.2)
    a.text(n[0], mp + (0.015 if lab.startswith('Ouro') else -0.055), '%s, minimal pairs (%.2f)' % (lab, mp), color=c, fontsize=7)
    print(lab, dict(zip(n.tolist(), np.round(acc['full'], 3).tolist())), 'pca', np.round(acc['pca'], 3).tolist(), 'mp', mp)
a.axhline(0.10, color='0.6', lw=0.8); a.text(n[-1], 0.075, 'chance', color='0.5', fontsize=7, ha='right')
a.set_xscale('log'); a.set_xticks([60, 250, 1000, 5249]); a.set_xticklabels(['60', '250', '1,000', '5,249'])
a.set_xlabel('distinct training questions'); a.set_ylabel('test accuracy, single prompts')
a.set_ylim(0, 1.05); a.legend(fontsize=6.5, frameon=False, loc='center left')
a.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); [fig.savefig(os.path.join(OUT, 'fig5_curve.' + e), dpi=200) for e in ('png', 'pdf')]
print('wrote', OUT)
