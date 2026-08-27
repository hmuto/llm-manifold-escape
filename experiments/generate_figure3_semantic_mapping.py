#!/usr/bin/env python3
"""
Generate Figure 3: Semantic Mapping.

PCA scatter plot of diverse responses showing emergent semantic axes.
Uses semantic_mapping results (n_agents=8).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nature_style import apply_nature_style, panel_label, SINGLE_COL, COLORS

apply_nature_style()

# Locate data
SEMANTIC_FILE = Path("results/semantic_mapping/semantic_mapping_20260206_021358.json")
FIGURE_DIR = Path("../paper/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("GENERATING FIGURE 3: SEMANTIC MAPPING")
print("=" * 70)
print(f"Data: {SEMANTIC_FILE}")

with open(SEMANTIC_FILE) as f:
    semantic_data = json.load(f)

sem_interp = semantic_data['experiment']['semantic_interpretation']

# Extract coordinates and axis information
coordinates = np.array(sem_interp['coordinates'])
axes_info = sem_interp['axes']
n_responses = len(coordinates)

# Create figure
fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))

# Plot points with gradient coloring
scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1],
                     c=range(n_responses), cmap='viridis',
                     s=80, alpha=0.85, edgecolors='0.3', linewidth=0.5,
                     zorder=10)

# Add response labels
for i, (x, y) in enumerate(coordinates):
    ax.annotate(f'R{i+1}', (x, y), fontsize=5, ha='center', va='center',
                fontweight='bold', color='white', zorder=11)

# Extract axis information
pc1_label = axes_info[0]['label']
pc1_var = axes_info[0]['explained_variance']
pc1_pos = axes_info[0]['positive_pole']
pc1_neg = axes_info[0]['negative_pole']

pc2_label = axes_info[1]['label']
pc2_var = axes_info[1]['explained_variance']
pc2_pos = axes_info[1]['positive_pole']
pc2_neg = axes_info[1]['negative_pole']

total_var = pc1_var + pc2_var

# Set axis labels with semantic interpretation
ax.set_xlabel(f"PC1: {pc1_label} ({pc1_var*100:.1f}%)\n{pc1_neg} \u2190 \u2192 {pc1_pos}")
ax.set_ylabel(f"PC2: {pc2_label} ({pc2_var*100:.1f}%)\n{pc2_neg} \u2190 \u2192 {pc2_pos}")

# Add explanation box
textstr = (f'Variance explained:\n{total_var*100:.1f}%\n'
           f'N = {n_responses} responses\n'
           f'Axes discovered\npost-hoc via PCA')
props = dict(boxstyle='round', facecolor=COLORS['cyan'], alpha=0.2,
             edgecolor='0.5', linewidth=0.3)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=5,
        verticalalignment='top', bbox=props)

plt.tight_layout()

# Save
for fmt in ['pdf', 'png']:
    outpath = FIGURE_DIR / f'fig3_semantic_mapping.{fmt}'
    fig.savefig(outpath)
    print(f"Saved: {outpath}")

plt.close()

print(f"\n  Responses: {n_responses}")
print(f"  PC1: {pc1_label} ({pc1_var*100:.1f}%)")
print(f"  PC2: {pc2_label} ({pc2_var*100:.1f}%)")
print(f"  Total variance: {total_var*100:.1f}%")
print("\nDone.")
