#!/usr/bin/env python3
"""
Small figure for the structured-task confinement (Section: support).

Panel (a): cumulative coverage of the independent-128 reference by the DDS pool,
by round. The structured task (problem_1) saturates by round 1 and round 2 adds
nothing, whereas open-ended tasks keep gaining.
Panel (b): mean distance to the reference centroid for reference points the loop
covers vs.\ misses. Missed points are more peripheral on every task.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nature_style import apply_nature_style, panel_label, DOUBLE_COL, COLORS

apply_nature_style()
FIGURE_DIR = Path("../paper/figures"); FIGURE_DIR.mkdir(parents=True, exist_ok=True)
d = json.load(open("results/support_vs_loop/structured_shortfall.json"))

tasks = list(d.keys())
style = {"creative_1": (COLORS["blue"], "o"), "creative_2": (COLORS["green"], "s"),
         "problem_1": (COLORS["orange"], "^"), "debate_1": (COLORS["purple"], "D")}

fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))

# (a) coverage by round
for tid in tasks:
    c, m = style.get(tid, (COLORS["gray"], "o"))
    cov = [v * 100 for v in d[tid]["cov_by_round"]]
    lw = 2.0 if tid == "problem_1" else 1.0
    a.plot([0, 1, 2], cov, marker=m, color=c, label=tid, markersize=3.5, linewidth=lw)
a.set_xticks([0, 1, 2]); a.set_xlabel("Round (cumulative)")
a.set_ylabel("Coverage of reference (%)")
a.legend(fontsize=5, loc="lower right", framealpha=0.9)

# (b) covered vs missed radius
x = np.arange(len(tasks)); w = 0.38
cov_r = [d[t]["radius_covered"] for t in tasks]
mis_r = [d[t]["radius_missed"] for t in tasks]
a2 = b
a2.bar(x - w / 2, cov_r, w, label="covered", color=COLORS["blue"], alpha=0.8,
       edgecolor="0.3", linewidth=0.5)
a2.bar(x + w / 2, mis_r, w, label="missed", color=COLORS["red"], alpha=0.8,
       edgecolor="0.3", linewidth=0.5)
a2.set_xticks(x); a2.set_xticklabels(tasks, fontsize=5.5, rotation=20, ha="right")
a2.set_ylabel("Radius (dist. to centroid)")
a2.legend(fontsize=5, loc="upper right", framealpha=0.9)

panel_label(a, "a"); panel_label(b, "b")
plt.tight_layout()
for fmt in ["pdf", "png"]:
    fig.savefig(FIGURE_DIR / f"fig_structured_shortfall.{fmt}")
plt.close()
print("Saved fig_structured_shortfall.{pdf,png}")
