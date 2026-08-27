#!/usr/bin/env python3
"""
Generate Figure 1: Dynamics with MAP-Elites Comparison.

6 conditions: DDS alpha=0.0/0.5/1.0, Debate, Independent, MAP-Elites
Panel (a): Diversity trajectories over rounds
Panel (b): Final diversity comparison with significance brackets

Usage:
    python generate_figure1_with_mapelites.py [results_file.json]
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from nature_style import (apply_nature_style, panel_label, DOUBLE_COL,
                           COND_COLORS, COND_LABELS, COND_MARKERS)
apply_nature_style()

# Paths
import sys
if len(sys.argv) > 1:
    DATA_FILE = Path(sys.argv[1])
else:
    # Auto-detect latest results file
    results_dir = Path("results/dynamics_mapelites")
    files = sorted(results_dir.glob("dynamics_mapelites_*.json"))
    if files:
        DATA_FILE = files[-1]
    else:
        print("No results files found. Run run_dynamics_with_mapelites.py first.")
        sys.exit(1)

FIGURE_DIR = Path("../paper/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("GENERATING FIGURE 1: DYNAMICS WITH MAP-ELITES")
print("=" * 70)
print()

# Load data
with open(DATA_FILE) as f:
    data = json.load(f)

config = data['config']
n_rounds = config['n_rounds']
rounds = list(range(n_rounds))

print(f"Data source: {DATA_FILE}")
print(f"  Model: {config['model']}, n_agents: {config['n_agents']}, n_rounds: {n_rounds}")
print()

# ============================================================
# Extract per-round diversity for each condition
# ============================================================
conditions = data['conditions']


def extract_round_data(cond_data, n_rounds):
    """Extract per-round diversity arrays from condition data."""
    round_divs = {r: [] for r in range(n_rounds)}
    for task_data in cond_data:
        for trial in task_data['trials']:
            for r_idx, div in enumerate(trial['round_diversities']):
                if r_idx in round_divs:
                    round_divs[r_idx].append(div)
    return round_divs


# Extract data
cond_round_data = {}
cond_final_divs = {}
for cond_name, cond_data in conditions.items():
    nr = n_rounds if cond_name != 'independent' else 1
    cond_round_data[cond_name] = extract_round_data(cond_data, nr)
    cond_final_divs[cond_name] = [
        trial['final_diversity']
        for task_data in cond_data
        for trial in task_data['trials']
    ]

# Print statistics
print("Per-round statistics:")
print(f"{'Condition':<18s}  {'Round 0':>10s}  {'Round 1':>10s}  {'Round 2':>10s}  {'Change':>10s}")
print("-" * 65)
plot_order = ['dds_alpha_0.5', 'dds_alpha_0.0', 'dds_alpha_1.0', 'map_elites', 'independent']
for cond_name in plot_order:
    if cond_name not in cond_round_data:
        continue
    rd = cond_round_data[cond_name]
    available_rounds = sorted(rd.keys())
    means = [np.mean(rd[r]) for r in available_rounds if rd[r]]
    if len(means) >= 2:
        change = (means[-1] - means[0]) / means[0] * 100
        r1 = f"{means[1]:10.4f}" if len(means) > 1 else f"{'--':>10s}"
        r2 = f"{means[2]:10.4f}" if len(means) > 2 else f"{'--':>10s}"
        print(f"{cond_name:<18s}  {means[0]:10.4f}  {r1}  {r2}  {change:+9.1f}%")
    elif means:
        print(f"{cond_name:<18s}  {means[0]:10.4f}  {'--':>10s}  {'--':>10s}  {'N/A':>10s}")
print()

# ============================================================
# FIGURE GENERATION
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.45),
                                gridspec_kw={'width_ratios': [1.2, 1]})

# ---- Left panel: Diversity over rounds ----
for cond_name in plot_order:
    if cond_name not in cond_round_data:
        continue
    rd = cond_round_data[cond_name]
    available_rounds = sorted(r for r in rd.keys() if rd[r])

    means = [np.mean(rd[r]) for r in available_rounds]
    sems = [stats.sem(rd[r]) for r in available_rounds]

    # Independent: show as horizontal dashed line
    if cond_name == 'independent':
        ax1.axhline(y=means[0], color=COND_COLORS[cond_name], linestyle=':', linewidth=0.8,
                     alpha=0.7, label=COND_LABELS[cond_name], zorder=5)
        ax1.axhspan(means[0] - sems[0], means[0] + sems[0],
                     alpha=0.08, color=COND_COLORS[cond_name], zorder=0)
        continue

    lw = 1.5 if cond_name == 'dds_alpha_0.5' else 1.0
    ms = 5 if cond_name == 'dds_alpha_0.5' else 4
    zorder = 10 if cond_name == 'dds_alpha_0.5' else 5
    ls = '--' if cond_name == 'map_elites' else '-'

    ax1.errorbar(available_rounds, means, yerr=sems,
                 fmt=f'{COND_MARKERS[cond_name]}{ls}', color=COND_COLORS[cond_name],
                 linewidth=lw, markersize=ms, capsize=2, capthick=1.0,
                 markerfacecolor='white', markeredgewidth=1.0,
                 markeredgecolor=COND_COLORS[cond_name],
                 label=COND_LABELS[cond_name], zorder=zorder)

# Annotations for key conditions
for cond_name, color in [('dds_alpha_0.5', COND_COLORS['dds_alpha_0.5']),
                          ('map_elites', COND_COLORS['map_elites'])]:
    if cond_name not in cond_round_data:
        continue
    rd = cond_round_data[cond_name]
    available_rounds = sorted(r for r in rd.keys() if rd[r])
    if len(available_rounds) < 2:
        continue
    means = [np.mean(rd[r]) for r in available_rounds]
    change = (means[-1] - means[0]) / means[0] * 100

    y_offset = 0.015 if change > 0 else -0.015
    sign = '+' if change > 0 else ''
    ax1.annotate(f'{sign}{change:.0f}%',
                 xy=(available_rounds[-1], means[-1]),
                 xytext=(available_rounds[-1] + 0.15, means[-1] + y_offset),
                 fontsize=6, fontweight='bold', color=color)

ax1.set_xlabel('Selection round')
ax1.set_ylabel('Mean pairwise cosine distance')
ax1.set_title('Diversity dynamics over rounds')
panel_label(ax1, 'a')
ax1.legend(loc='upper left', framealpha=0.9, ncol=1)
ax1.set_xticks(rounds)
ax1.set_xlim(-0.2, 2.6)
ax1.set_ylim(0, 0.40)
ax1.set_axisbelow(True)

# ---- Cumulative diversity per trial (pool responses across rounds, re-embed) ----
# Panel (b) reports cumulative diversity (all responses pooled across rounds),
# not the final-round snapshot. This matches Table 2 (cumulative row) and the
# caption. Independent has a single round, so its cumulative == round 0.
def compute_cumulative(cond_data, model):
    from sklearn.metrics.pairwise import cosine_distances
    vals = []
    for task_data in cond_data:
        for trial in task_data['trials']:
            texts = [r['text'] for rt in trial.get('response_texts', []) for r in rt]
            if len(texts) < 2:
                continue
            emb = model.encode(texts, show_progress_bar=False)
            dm = cosine_distances(emb)
            iu = np.triu_indices(len(texts), k=1)
            vals.append(float(dm[iu].mean()))
    return vals


from sentence_transformers import SentenceTransformer
_embed_model = SentenceTransformer('all-MiniLM-L6-v2')
cond_cumulative_divs = {c: compute_cumulative(cd, _embed_model)
                        for c, cd in conditions.items()}

# ---- Right panel: Cumulative diversity comparison ----
bar_order = ['dds_alpha_0.5', 'dds_alpha_0.0', 'dds_alpha_1.0', 'map_elites', 'independent']
bar_order = [c for c in bar_order if c in cond_cumulative_divs]
bar_means = [np.mean(cond_cumulative_divs[c]) for c in bar_order]
bar_sems = [stats.sem(cond_cumulative_divs[c]) for c in bar_order]
bar_ns = [len(cond_cumulative_divs[c]) for c in bar_order]
bar_colors = [COND_COLORS[c] for c in bar_order]
bar_labels = [COND_LABELS[c] for c in bar_order]
x_pos = np.arange(len(bar_order))

bars = ax2.bar(x_pos, bar_means, yerr=bar_sems, capsize=2,
               color=bar_colors, alpha=0.7, edgecolor='0.3', linewidth=0.5,
               error_kw={'linewidth': 1.0, 'capthick': 1.0})

# Add n labels
for i, (x, m, n) in enumerate(zip(x_pos, bar_means, bar_ns)):
    ax2.text(x, m + bar_sems[i] + 0.008, f'n={n}',
             ha='center', va='bottom', fontsize=5, color='gray')

# Significance bracket: DDS alpha=0.5 vs Independent (the headline contrast)
if 'dds_alpha_0.5' in cond_cumulative_divs and 'independent' in cond_cumulative_divs:
    dds_idx = bar_order.index('dds_alpha_0.5')
    ind_idx = bar_order.index('independent')
    y_top = max(bar_means) + max(bar_sems) + 0.025
    ax2.plot([dds_idx, dds_idx, ind_idx, ind_idx],
             [y_top, y_top + 0.005, y_top + 0.005, y_top],
             'k-', linewidth=0.8)

    dds05_c = cond_cumulative_divs['dds_alpha_0.5']
    ind_c = cond_cumulative_divs['independent']
    n_paired = min(len(dds05_c), len(ind_c))
    _, p_dds_ind = stats.ttest_rel(dds05_c[:n_paired], ind_c[:n_paired])
    sig_str = '***' if p_dds_ind < 0.001 else '**' if p_dds_ind < 0.01 else '*' if p_dds_ind < 0.05 else 'n.s.'
    ax2.text((dds_idx + ind_idx) / 2, y_top + 0.008, f'{sig_str}\np={p_dds_ind:.4f}',
             ha='center', va='bottom', fontsize=6, fontweight='bold')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(bar_labels, fontsize=6, rotation=25, ha='right')
ax2.set_ylabel('Cumulative mean pairwise distance')
ax2.set_title('Cumulative diversity comparison')
panel_label(ax2, 'b')
ax2.set_ylim(0, 0.42)
ax2.set_axisbelow(True)

plt.tight_layout()

# Save
pdf_path = FIGURE_DIR / "fig1_dynamics.pdf"
png_path = FIGURE_DIR / "fig1_dynamics.png"
plt.savefig(pdf_path, bbox_inches='tight', dpi=600)
plt.savefig(png_path, bbox_inches='tight', dpi=300)
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
plt.close()

# ============================================================
# Print key findings for paper
# ============================================================
print()
print("=" * 70)
print("KEY FINDINGS FOR PAPER")
print("=" * 70)
print()

# DDS alpha=0.5 statistics
if 'dds_alpha_0.5' in cond_round_data:
    rd = cond_round_data['dds_alpha_0.5']
    available_rounds = sorted(r for r in rd.keys() if rd[r])
    means = [np.mean(rd[r]) for r in available_rounds]
    change = (means[-1] - means[0]) / means[0] * 100
    print(f"DDS alpha=0.5 diversity growth: +{change:.1f}%")
    print(f"  Round 0: {means[0]:.4f}, Round {available_rounds[-1]}: {means[-1]:.4f}")

# MAP-Elites statistics
if 'map_elites' in cond_round_data:
    rd = cond_round_data['map_elites']
    available_rounds = sorted(r for r in rd.keys() if rd[r])
    means = [np.mean(rd[r]) for r in available_rounds]
    change = (means[-1] - means[0]) / means[0] * 100
    print(f"MAP-Elites diversity change: {change:+.1f}%")

    # Archive stats
    all_cov = []
    all_adiv = []
    for task_data in conditions.get('map_elites', []):
        for trial in task_data['trials']:
            if 'archive_coverage' in trial:
                all_cov.append(trial['archive_coverage'][-1])
            if 'archive_diversity' in trial:
                all_adiv.append(trial['archive_diversity'][-1])
    if all_cov:
        print(f"  Archive coverage: {np.mean(all_cov):.3f} +/- {np.std(all_cov, ddof=1):.3f}")
    if all_adiv:
        print(f"  Archive diversity: {np.mean(all_adiv):.3f} +/- {np.std(all_adiv, ddof=1):.3f}")

# Pairwise comparisons
print()
print("Statistical comparisons:")
print("-" * 50)
comparisons = [
    ('dds_alpha_0.5', 'map_elites', 'DDS vs MAP-Elites'),
    ('dds_alpha_0.5', 'independent', 'DDS vs Independent'),
    ('map_elites', 'independent', 'MAP-Elites vs Independent'),
]
for c1, c2, label in comparisons:
    if c1 not in cond_final_divs or c2 not in cond_final_divs:
        continue
    v1 = cond_final_divs[c1]
    v2 = cond_final_divs[c2]
    n_paired = min(len(v1), len(v2))
    if n_paired < 3:
        continue
    t_stat, p_val = stats.ttest_rel(v1[:n_paired], v2[:n_paired])
    diff = np.array(v1[:n_paired]) - np.array(v2[:n_paired])
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    print(f"  {label}: t({n_paired-1})={t_stat:.3f}, p={p_val:.6f}, d={d:.3f} ({sig})")

print()
print("FIGURE 1 GENERATION COMPLETE")
print("=" * 70)
