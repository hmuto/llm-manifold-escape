#!/usr/bin/env python3
"""
Generate Figure 4: Alpha Sweep from extended real API experiment.
Uses alpha_sweep_extended data (GPT-4o-mini, n=80 per alpha, 7 alpha values).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy import stats

from nature_style import apply_nature_style, panel_label, DOUBLE_COL, COLORS
apply_nature_style()

# Paths
DATA_FILE = Path("results/alpha_sweep_extended/alpha_sweep_extended_20260206_040505.json")
FIGURE_DIR = Path("../paper/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("GENERATING FIGURE 4: ALPHA SWEEP (REAL API, GPT-4o-mini)")
print("=" * 70)
print()

# Load data
with open(DATA_FILE) as f:
    data = json.load(f)

config = data['config']
alpha_values = sorted(config['dds_alpha_values'])
n_trials_per_task = data['n_trials_per_task']
tasks = [t['task_id'] for t in data['experiments']['alpha_sweep']]

print(f"Data source: {DATA_FILE.name}")
print(f"  Backend: {config['backend']}")
print(f"  Model: {config['model']}")
print(f"  n_agents: {config['n_agents']}")
print(f"  n_rounds: {config['n_rounds']}")
print(f"  Alpha values: {alpha_values}")
print(f"  Tasks: {tasks}")
print(f"  Trials per task: {n_trials_per_task}")
print()

# Extract diversity data per alpha, organized by observation index
# Each observation = (task, trial) pair, consistent across alphas
alpha_divs = {a: [] for a in alpha_values}
task_alpha_divs = {task: {a: [] for a in alpha_values} for task in tasks}

for task_data in data['experiments']['alpha_sweep']:
    task_id = task_data['task_id']
    for trial in task_data['trials']:
        for a in alpha_values:
            key = f"alpha_{a}"
            div = trial[key]['diversity']
            alpha_divs[a].append(div)
            task_alpha_divs[task_id][a].append(div)

# Convert to arrays
for a in alpha_values:
    alpha_divs[a] = np.array(alpha_divs[a])

n_obs = len(alpha_divs[alpha_values[0]])
print(f"Total observations per alpha: {n_obs}")
print()

# ============================================================
# Descriptive Statistics
# ============================================================
print("Descriptive Statistics:")
print(f"{'Alpha':>6s}  {'Mean':>7s}  {'Std':>7s}  {'SEM':>7s}  {'n':>4s}")
print("-" * 40)
for a in alpha_values:
    d = alpha_divs[a]
    print(f"{a:6.1f}  {np.mean(d):7.4f}  {np.std(d, ddof=1):7.4f}  {stats.sem(d):7.4f}  {len(d):4d}")
print()

# ============================================================
# Statistical Tests
# ============================================================

# 1. One-way repeated measures: Friedman test (non-parametric)
# Organize data as matrix (n_obs x n_alphas)
data_matrix = np.column_stack([alpha_divs[a] for a in alpha_values])
friedman_stat, friedman_p = stats.friedmanchisquare(*[alpha_divs[a] for a in alpha_values])
print(f"Friedman test: chi2={friedman_stat:.3f}, p={friedman_p:.6f}")
print()

# 2. Pairwise comparisons: peak (alpha=0.5) vs each higher alpha
peak_alpha = 0.5
peak_divs = alpha_divs[peak_alpha]
print(f"Pairwise comparisons (peak alpha={peak_alpha} vs others):")
print(f"{'Comparison':>20s}  {'t':>7s}  {'p':>9s}  {'d':>7s}  {'Sig':>5s}")
print("-" * 55)

pairwise_results = {}
for a in alpha_values:
    if a == peak_alpha:
        continue
    t_stat, p_val = stats.ttest_rel(peak_divs, alpha_divs[a])
    diff = peak_divs - alpha_divs[a]
    d_cohen = np.mean(diff) / np.std(diff, ddof=1)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    pairwise_results[a] = {'t': t_stat, 'p': p_val, 'd': d_cohen, 'sig': sig}
    print(f"  0.5 vs {a:<4.1f}       {t_stat:7.3f}  {p_val:9.6f}  {d_cohen:7.3f}  {sig:>5s}")

print()

# 3. Key comparison: alpha=0.5 (peak) vs alpha=3.0 (minimum)
min_alpha = 3.0
t_peak_min, p_peak_min = stats.ttest_rel(alpha_divs[peak_alpha], alpha_divs[min_alpha])
diff_peak_min = alpha_divs[peak_alpha] - alpha_divs[min_alpha]
d_peak_min = np.mean(diff_peak_min) / np.std(diff_peak_min, ddof=1)
ci_95 = stats.t.interval(0.95, df=len(diff_peak_min)-1,
                         loc=np.mean(diff_peak_min),
                         scale=stats.sem(diff_peak_min))
reduction_pct = (np.mean(alpha_divs[peak_alpha]) - np.mean(alpha_divs[min_alpha])) / np.mean(alpha_divs[peak_alpha]) * 100

print(f"Key comparison: alpha={peak_alpha} vs alpha={min_alpha}")
print(f"  Paired t-test: t({n_obs-1})={t_peak_min:.3f}, p={p_peak_min:.6f}")
print(f"  Cohen's d: {d_peak_min:.3f}")
print(f"  Mean difference: {np.mean(diff_peak_min):.4f} [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
print(f"  Reduction: {reduction_pct:.1f}%")
print()

# 4. Also report alpha=1.0 vs alpha=2.0 (for comparison with earlier experiment)
t_1v2, p_1v2 = stats.ttest_rel(alpha_divs[1.0], alpha_divs[2.0])
diff_1v2 = alpha_divs[1.0] - alpha_divs[2.0]
d_1v2 = np.mean(diff_1v2) / np.std(diff_1v2, ddof=1)
ci_1v2 = stats.t.interval(0.95, df=len(diff_1v2)-1,
                           loc=np.mean(diff_1v2),
                           scale=stats.sem(diff_1v2))
red_1v2 = (np.mean(alpha_divs[1.0]) - np.mean(alpha_divs[2.0])) / np.mean(alpha_divs[1.0]) * 100

print(f"Comparison: alpha=1.0 vs alpha=2.0")
print(f"  Paired t-test: t({n_obs-1})={t_1v2:.3f}, p={p_1v2:.6f}")
print(f"  Cohen's d: {d_1v2:.3f}")
print(f"  Mean difference: {np.mean(diff_1v2):.4f} [{ci_1v2[0]:.4f}, {ci_1v2[1]:.4f}]")
print(f"  Reduction: {red_1v2:.1f}%")
print()

# ============================================================
# FIGURE GENERATION
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4),
                                gridspec_kw={'width_ratios': [1.2, 1]})

# Colors from Nature-style palette
COLOR_MAIN = COLORS['blue']
COLOR_PEAK = COLORS['green']
COLOR_LOW = COLORS['red']
COLOR_TASK = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red']]
TASK_LABELS = {
    'creative_1': 'Creative 1',
    'creative_2': 'Creative 2',
    'problem_1': 'Problem Solving',
    'debate_1': 'Debate'
}

# ---- Left panel: Diversity vs Alpha ----
means = [np.mean(alpha_divs[a]) for a in alpha_values]
sems = [stats.sem(alpha_divs[a]) for a in alpha_values]

# Plot task-level means (thin lines)
for i, task in enumerate(tasks):
    task_means = [np.mean(task_alpha_divs[task][a]) for a in alpha_values]
    ax1.plot(alpha_values, task_means, 'o--', color=COLOR_TASK[i],
             alpha=0.35, linewidth=0.7, markersize=3,
             label=TASK_LABELS.get(task, task))

# Plot overall mean with error bars (SEM)
ax1.errorbar(alpha_values, means, yerr=sems, fmt='s-',
             color=COLORS['black'], linewidth=1.5, markersize=5, capsize=2, capthick=1.0,
             markerfacecolor='white', markeredgewidth=1.0, markeredgecolor=COLORS['black'],
             zorder=10, label=f'Overall mean (n={n_obs})')

# Mark peak
peak_idx = np.argmax(means)
ax1.annotate(f'Peak\n({alpha_values[peak_idx]:.1f}, {means[peak_idx]:.3f})',
             xy=(alpha_values[peak_idx], means[peak_idx]),
             xytext=(alpha_values[peak_idx] + 0.4, means[peak_idx] + 0.025),
             fontsize=5, fontweight='bold', color=COLOR_PEAK,
             arrowprops=dict(arrowstyle='->', color=COLOR_PEAK, lw=1.0))

# Labels and formatting
ax1.set_xlabel(r'Selection pressure $\alpha$')
ax1.set_ylabel('Mean pairwise distance')
ax1.set_xticks(alpha_values)
ax1.set_xlim(-0.2, 3.3)
ax1.set_ylim(0, max(means) + max(sems) + 0.045)
ax1.legend(loc='upper right', fontsize=5, framealpha=0.9)
panel_label(ax1, 'a')

# ---- Right panel: Violin plot peak vs minimum ----
data_peak = alpha_divs[peak_alpha]
data_min = alpha_divs[min_alpha]
positions = [0, 1]

parts = ax2.violinplot([data_peak, data_min], positions=positions,
                        widths=0.6, showmeans=False, showmedians=False)

# Customize violins
violin_colors = [COLOR_PEAK, COLOR_LOW]
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(violin_colors[i])
    pc.set_alpha(0.5)
    pc.set_edgecolor('0.3')
    pc.set_linewidth(0.5)

for key in ('cbars', 'cmins', 'cmaxes'):
    if key in parts:
        parts[key].set_edgecolor('0.3')
        parts[key].set_linewidth(0.5)

# Add box plots inside
bp = ax2.boxplot([data_peak, data_min], positions=positions, widths=0.15,
                  patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor='white', edgecolor='0.3', linewidth=0.5),
                  medianprops=dict(color=COLORS['black'], linewidth=1.0),
                  whiskerprops=dict(color='0.3', linewidth=0.5),
                  capprops=dict(color='0.3', linewidth=0.5))

# Add mean markers
for i, (pos, d) in enumerate(zip(positions, [data_peak, data_min])):
    ax2.scatter(pos, np.mean(d), color=violin_colors[i], s=25, marker='D',
                zorder=5, edgecolors='0.3', linewidth=0.5, label='Mean' if i == 0 else None)

# Significance bracket
y_max = max(np.max(data_peak), np.max(data_min))
y_bracket = y_max + 0.02
ax2.plot([0, 0, 1, 1], [y_bracket, y_bracket + 0.005, y_bracket + 0.005, y_bracket],
         'k-', linewidth=1.0)

sig_text = f'p < 0.001' if p_peak_min < 0.001 else f'p = {p_peak_min:.4f}'
ax2.text(0.5, y_bracket + 0.008, f'{sig_text}\nd = {d_peak_min:.2f}',
         ha='center', va='bottom', fontsize=6, fontweight='bold')

# Statistics box
stats_text = (
    f"$\\alpha$=0.5: {np.mean(data_peak):.3f} $\\pm$ {np.std(data_peak, ddof=1):.3f}\n"
    f"$\\alpha$=3.0: {np.mean(data_min):.3f} $\\pm$ {np.std(data_min, ddof=1):.3f}\n"
    f"Reduction: {reduction_pct:.1f}%"
)
ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes,
         fontsize=5, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, edgecolor='0.3', linewidth=0.3))

# Labels
ax2.set_xticks(positions)
ax2.set_xticklabels([f'$\\alpha$ = {peak_alpha}', f'$\\alpha$ = {min_alpha}'])
ax2.set_ylabel('Mean pairwise distance')
ax2.set_xlim(-0.5, 1.5)
panel_label(ax2, 'b')

plt.tight_layout()

# Save
pdf_path = FIGURE_DIR / "fig4_alpha_sweep.pdf"
png_path = FIGURE_DIR / "fig4_alpha_sweep.png"
plt.savefig(pdf_path, bbox_inches='tight', dpi=600)
plt.savefig(png_path, bbox_inches='tight', dpi=600)
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
plt.close()

print()
print("=" * 70)
print("FIGURE 4 GENERATION COMPLETE")
print("=" * 70)
