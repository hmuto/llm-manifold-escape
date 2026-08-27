#!/usr/bin/env python3
"""
Supplementary figure: saturation of the output-support estimate.

Reads the vs-N curve saved by analyze_support_vs_loop.py and plots two extent
measures of the independent sample against N per task:
  (a) radius (mean distance to the reference centroid) -- the spatial extent /
      outer boundary of the accessible region.
  (b) participation-ratio effective dimension.

The point: radius saturates early (by N ~ 16-32), so N = 128 estimates the
extent of the accessible region well; effective dimension keeps rising with N,
which is a known finite-sample property of the participation-ratio estimator,
not evidence that the region keeps growing. The decisive matched-k comparison
controls for sample size and is unaffected by this N-dependence.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from nature_style import apply_nature_style, panel_label, DOUBLE_COL, COLORS

apply_nature_style()

FIGURE_DIR = Path("../paper/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DATA = Path("results/support_vs_loop/support_vs_loop.json")

TASK_STYLE = {
    "creative_1": (COLORS["blue"],   "o", "creative_1"),
    "creative_2": (COLORS["green"],  "s", "creative_2"),
    "problem_1":  (COLORS["orange"], "^", "problem_1"),
    "debate_1":   (COLORS["purple"], "D", "debate_1"),
}

print("=" * 70)
print("GENERATING SUPPLEMENTARY FIGURE: SUPPORT-ESTIMATE SATURATION")
print("=" * 70)

with open(DATA) as f:
    d = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))
ax_r, ax_e = axes

for tid, s in d.items():
    color, marker, label = TASK_STYLE.get(tid, (COLORS["gray"], "o", tid))
    Ns = [c["N"] for c in s["curve"]]
    radii = [float(c["radius"]) for c in s["curve"]]
    effs = [float(c["eff_dim"]) for c in s["curve"]]
    ax_r.plot(Ns, radii, marker=marker, color=color, label=label,
              markersize=3.5, linewidth=1.0)
    ax_e.plot(Ns, effs, marker=marker, color=color, label=label,
              markersize=3.5, linewidth=1.0)

for ax in (ax_r, ax_e):
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128])
    ax.set_xticklabels(["8", "16", "32", "64", "128"])
    ax.set_xlabel("Independent sample size $N$")

ax_r.set_ylabel("Radius (extent of region)")
ax_r.set_ylim(bottom=0)
ax_e.set_ylabel("Effective dimension $d_{\\mathrm{eff}}$")
ax_e.set_ylim(bottom=0)
ax_e.legend(loc="upper left", fontsize=5, framealpha=0.9)

panel_label(ax_r, "a")
panel_label(ax_e, "b")

plt.tight_layout()
for fmt in ["pdf", "png"]:
    outpath = FIGURE_DIR / f"figS_support_saturation.{fmt}"
    fig.savefig(outpath)
    print(f"Saved: {outpath}")
plt.close()
print("\nDone.")
