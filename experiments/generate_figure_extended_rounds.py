#!/usr/bin/env python3
"""
Generate Extended Rounds Figure: 7-round diversity trajectories.

Shows per-round snapshot diversity over 7 rounds for 3 conditions:
DDS α=0.5, MAP-Elites, Independent.

Highlights the saturation pattern (growth plateaus around R2-3).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from nature_style import apply_nature_style, panel_label, SINGLE_COL, COND_COLORS, COND_LABELS, COND_MARKERS

apply_nature_style()

# Locate data
results_dir = Path("results/extended_rounds")
files = sorted(results_dir.glob("extended_rounds_*.json"))
# Exclude partial files
files = [f for f in files if "partial" not in f.name]
if not files:
    print("No extended rounds results found.")
    exit(1)
input_file = files[-1]

FIGURE_DIR = Path("../paper/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("GENERATING EXTENDED ROUNDS FIGURE")
print("=" * 70)
print(f"Data: {input_file}")

with open(input_file) as f:
    data = json.load(f)

conditions = data['conditions']
n_rounds = data['config']['n_rounds']  # 7
rounds = list(range(n_rounds))

# Compute per-round diversity for each condition
fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL, SINGLE_COL * 0.75))

plot_order = ['dds_alpha_0.5', 'map_elites', 'independent']

for cond_name in plot_order:
    cond_data = conditions[cond_name]

    # Collect per-round diversities across all tasks × trials
    round_divs_by_round = {r: [] for r in rounds}
    for task_data in cond_data:
        for trial in task_data['trials']:
            for r_idx, rd in enumerate(trial['round_diversities']):
                round_divs_by_round[r_idx].append(rd)

    means = [np.mean(round_divs_by_round[r]) for r in rounds]
    sems = [stats.sem(round_divs_by_round[r]) for r in rounds]

    if cond_name == 'independent':
        # Show as horizontal band
        overall_mean = np.mean(means)
        overall_sem = np.mean(sems)
        ax.axhline(y=overall_mean, color=COND_COLORS[cond_name], linestyle=':',
                   linewidth=0.8, alpha=0.7, label=COND_LABELS[cond_name], zorder=5)
        ax.axhspan(overall_mean - overall_sem, overall_mean + overall_sem,
                   alpha=0.08, color=COND_COLORS[cond_name], zorder=0)
    else:
        lw = 1.5 if cond_name == 'dds_alpha_0.5' else 1.0
        ms = 5 if cond_name == 'dds_alpha_0.5' else 4
        zorder = 10 if cond_name == 'dds_alpha_0.5' else 9
        linestyle = '--' if cond_name == 'map_elites' else '-'

        ax.errorbar(rounds, means, yerr=sems,
                    color=COND_COLORS[cond_name],
                    marker=COND_MARKERS[cond_name],
                    markersize=ms,
                    linewidth=lw,
                    linestyle=linestyle,
                    capsize=2,
                    label=COND_LABELS[cond_name],
                    zorder=zorder)

    # Print stats
    n_obs = len(round_divs_by_round[0])
    growth = (means[-1] - means[0]) / means[0] * 100 if means[0] > 0 else 0
    print(f"{cond_name}: R0={means[0]:.4f}, R6={means[-1]:.4f}, "
          f"growth={growth:+.1f}%, n={n_obs}")

ax.set_xlabel('Round')
ax.set_ylabel('Pairwise cosine distance')
ax.set_xticks(rounds)
ax.legend(loc='lower right', framealpha=0.9, fontsize=6)

plt.tight_layout()

# Save
for fmt in ['pdf', 'png']:
    outpath = FIGURE_DIR / f'fig_extended_rounds.{fmt}'
    fig.savefig(outpath)
    print(f"Saved: {outpath}")

plt.close()

# Statistical summary
print("\n--- Statistical tests ---")
for cond_name in ['dds_alpha_0.5', 'map_elites']:
    cond_data = conditions[cond_name]
    r0_divs = []
    r6_divs = []
    for task_data in cond_data:
        for trial in task_data['trials']:
            r0_divs.append(trial['round_diversities'][0])
            r6_divs.append(trial['round_diversities'][6])

    t_stat, p_val = stats.ttest_rel(r6_divs, r0_divs)
    diff = np.array(r6_divs) - np.array(r0_divs)
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
    print(f"{cond_name}: R0→R6 t({len(r0_divs)-1})={t_stat:.3f}, "
          f"p={p_val:.4f}, d={d:.3f}")

    # Round-to-round consecutive tests
    print(f"  Consecutive round tests:")
    for r in range(n_rounds - 1):
        ra, rb = [], []
        for task_data in cond_data:
            for trial in task_data['trials']:
                ra.append(trial['round_diversities'][r])
                rb.append(trial['round_diversities'][r + 1])
        t_stat, p_val = stats.ttest_rel(rb, ra)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"    R{r}→R{r+1}: t={t_stat:.3f}, p={p_val:.4f} ({sig})")

print("\nDone.")
