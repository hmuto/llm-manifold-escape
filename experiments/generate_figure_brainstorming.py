#!/usr/bin/env python3
"""
Generate Brainstorming Case Study Figure.

2-panel figure showing:
(a) Unique ideas per condition (bar chart with per-task breakdown)
(b) Semantic diversity per condition (bar chart)

Both show DDS and MAP-Elites significantly outperforming Independent,
with comparable quality maintained.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from nature_style import apply_nature_style, panel_label, DOUBLE_COL, COND_COLORS, COND_LABELS

apply_nature_style()

# Locate data
results_dir = Path("results/brainstorming_case_study")
files = sorted(results_dir.glob("brainstorming_*.json"))
if not files:
    print("No brainstorming results found.")
    exit(1)
input_file = files[-1]

FIGURE_DIR = Path("../paper/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("GENERATING BRAINSTORMING CASE STUDY FIGURE")
print("=" * 70)
print(f"Data: {input_file}")

with open(input_file) as f:
    data = json.load(f)

# Conditions and display
conditions = ['dds_alpha_0.5', 'map_elites', 'independent']

task_ids = ['brainstorm_urban', 'brainstorm_remote', 'brainstorm_food']
TASK_LABELS = {
    'brainstorm_urban': 'Urban\nSustainability',
    'brainstorm_remote': 'Remote Work\nLoneliness',
    'brainstorm_food': 'Food Waste\nReduction',
}

# Extract per-trial data
idea_counts = {}  # cond -> task -> [trial values]
diversities = {}  # cond -> task -> [trial values]

for cond_name in conditions:
    idea_counts[cond_name] = {}
    diversities[cond_name] = {}

    # Diversity from conditions data
    cond_data = data['conditions'][cond_name]
    for task_entry in cond_data:
        task_id = task_entry['task_id']
        trial_divs = [trial['final_diversity'] for trial in task_entry['trials']]
        diversities[cond_name][task_id] = trial_divs
        idea_counts[cond_name][task_id] = []

    # Idea counts from separate idea_counts section
    for entry in data['idea_counts'][cond_name]:
        task_id = entry['task_id']
        idea_counts[cond_name].setdefault(task_id, []).append(entry['n_unique_ideas'])

# ============================================================
# Figure: 2-panel layout
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))

# --- Panel (a): Unique ideas by task ---
x = np.arange(len(task_ids))
n_conds = len(conditions)
width = 0.22
offsets = np.linspace(-(n_conds - 1) / 2 * width, (n_conds - 1) / 2 * width, n_conds)

for i, cond_name in enumerate(conditions):
    means = [np.mean(idea_counts[cond_name][tid]) for tid in task_ids]
    sems = [stats.sem(idea_counts[cond_name][tid]) for tid in task_ids]

    ax1.bar(x + offsets[i], means, width, yerr=sems,
            color=COND_COLORS[cond_name], alpha=0.8,
            label=COND_LABELS[cond_name], capsize=1.5,
            edgecolor='0.3', linewidth=0.3)

ax1.set_xlabel('Brainstorming Task')
ax1.set_ylabel('Unique ideas generated')
ax1.set_xticks(x)
ax1.set_xticklabels([TASK_LABELS[tid] for tid in task_ids])
ax1.legend(loc='upper right', framealpha=0.9, fontsize=6)
panel_label(ax1, 'a')

# Add significance brackets for each task
for task_idx, tid in enumerate(task_ids):
    dds_vals = idea_counts['dds_alpha_0.5'][tid]
    indep_vals = idea_counts['independent'][tid]
    n = min(len(dds_vals), len(indep_vals))
    t_stat, p_val = stats.ttest_rel(dds_vals[:n], indep_vals[:n])

    max_val = max(
        np.mean(idea_counts[c][tid]) + stats.sem(idea_counts[c][tid])
        for c in conditions
    )

    if p_val < 0.001:
        sig_text = '***'
    elif p_val < 0.01:
        sig_text = '**'
    elif p_val < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'

    y_bracket = max_val + 3
    ax1.annotate(sig_text, xy=(task_idx, y_bracket),
                fontsize=6, ha='center', color='#333333')

# --- Panel (b): Diversity by task ---
for i, cond_name in enumerate(conditions):
    means = [np.mean(diversities[cond_name][tid]) for tid in task_ids]
    sems = [stats.sem(diversities[cond_name][tid]) for tid in task_ids]

    ax2.bar(x + offsets[i], means, width, yerr=sems,
            color=COND_COLORS[cond_name], alpha=0.8,
            label=COND_LABELS[cond_name], capsize=1.5,
            edgecolor='0.3', linewidth=0.3)

ax2.set_xlabel('Brainstorming Task')
ax2.set_ylabel('Pairwise cosine distance')
ax2.set_xticks(x)
ax2.set_xticklabels([TASK_LABELS[tid] for tid in task_ids])
ax2.legend(loc='upper right', framealpha=0.9, fontsize=6)
panel_label(ax2, 'b')

# Add significance annotations
for task_idx, tid in enumerate(task_ids):
    dds_vals = diversities['dds_alpha_0.5'][tid]
    indep_vals = diversities['independent'][tid]
    n = min(len(dds_vals), len(indep_vals))
    t_stat, p_val = stats.ttest_rel(dds_vals[:n], indep_vals[:n])

    max_val = max(
        np.mean(diversities[c][tid]) + stats.sem(diversities[c][tid])
        for c in conditions
    )

    if p_val < 0.001:
        sig_text = '***'
    elif p_val < 0.01:
        sig_text = '**'
    elif p_val < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'

    y_bracket = max_val + 0.01
    ax2.annotate(sig_text, xy=(task_idx, y_bracket),
                fontsize=6, ha='center', color='#333333')

plt.tight_layout()

# Save
for fmt in ['pdf', 'png']:
    outpath = FIGURE_DIR / f'fig_brainstorming.{fmt}'
    fig.savefig(outpath)
    print(f"Saved: {outpath}")

plt.close()

# ============================================================
# Print summary for paper text
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY FOR PAPER")
print("=" * 70)

# Overall statistics
for cond_name in conditions:
    all_ideas = []
    all_divs = []
    for tid in task_ids:
        all_ideas.extend(idea_counts[cond_name][tid])
        all_divs.extend(diversities[cond_name][tid])
    print(f"\n{COND_LABELS[cond_name]}:")
    print(f"  Ideas: {np.mean(all_ideas):.1f} +/- {np.std(all_ideas, ddof=1):.1f}")
    print(f"  Diversity: {np.mean(all_divs):.4f} +/- {np.std(all_divs, ddof=1):.4f}")

# Pairwise tests (overall, paired by trial)
print("\n--- Pairwise tests (overall) ---")
pairs = [
    ('dds_alpha_0.5', 'independent'),
    ('map_elites', 'independent'),
    ('dds_alpha_0.5', 'map_elites'),
]
for a, b in pairs:
    a_ideas = []
    b_ideas = []
    a_divs = []
    b_divs = []
    for tid in task_ids:
        a_ideas.extend(idea_counts[a][tid])
        b_ideas.extend(idea_counts[b][tid])
        a_divs.extend(diversities[a][tid])
        b_divs.extend(diversities[b][tid])

    n = min(len(a_ideas), len(b_ideas))
    t_i, p_i = stats.ttest_rel(a_ideas[:n], b_ideas[:n])
    d_i = np.mean(np.array(a_ideas[:n]) - np.array(b_ideas[:n])) / np.std(np.array(a_ideas[:n]) - np.array(b_ideas[:n]), ddof=1)
    t_d, p_d = stats.ttest_rel(a_divs[:n], b_divs[:n])
    d_d = np.mean(np.array(a_divs[:n]) - np.array(b_divs[:n])) / np.std(np.array(a_divs[:n]) - np.array(b_divs[:n]), ddof=1)

    sig_i = "***" if p_i < 0.001 else "**" if p_i < 0.01 else "*" if p_i < 0.05 else "n.s."
    sig_d = "***" if p_d < 0.001 else "**" if p_d < 0.01 else "*" if p_d < 0.05 else "n.s."

    print(f"\n{COND_LABELS[a]} vs {COND_LABELS[b]}:")
    print(f"  Ideas: t({n-1})={t_i:.3f}, p={p_i:.4f}, d={d_i:.3f} ({sig_i})")
    print(f"  Diversity: t({n-1})={t_d:.3f}, p={p_d:.4f}, d={d_d:.3f} ({sig_d})")

# Multiplier
dds_ideas = []
indep_ideas = []
for tid in task_ids:
    dds_ideas.extend(idea_counts['dds_alpha_0.5'][tid])
    indep_ideas.extend(idea_counts['independent'][tid])
print(f"\nDDS/Independent ratio (ideas): {np.mean(dds_ideas) / np.mean(indep_ideas):.1f}x")

me_ideas = []
for tid in task_ids:
    me_ideas.extend(idea_counts['map_elites'][tid])
print(f"MAP-Elites/Independent ratio (ideas): {np.mean(me_ideas) / np.mean(indep_ideas):.1f}x")

print("\nDone.")
