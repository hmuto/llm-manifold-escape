#!/usr/bin/env python3
"""Concept figure v3: schematic only (the verdict matrix now lives in a native
LaTeX table, tab:levers, for sharp text).

Three ellipse panels: (a) selection, (b) temperature, (c) prompt.
Monochrome base + ONE accent (vermilion) marking what each lever changes.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica Neue", "Helvetica"],
    "font.size": 10.5,
    "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})

# ---------------------------------------------------------------- palette
INK    = "#1a1a1a"   # solid outcome ellipse, main text, kept directions
GRAY   = "#9aa0a6"   # reference (dashed), reference arrows
LIGHT  = "#c9cdd2"   # bulk sample dots
SOFT   = "#5f6368"   # secondary text
ACCENT = "#D55E00"   # Okabe-Ito vermilion: the ONE accent (what changed)

# ---------------------------------------------------------------- layout
fig = plt.figure(figsize=(13.2, 3.9))
gs = fig.add_gridspec(nrows=1, ncols=3, left=0.012, right=0.988,
                      top=0.90, bottom=0.02, wspace=0.05)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

for ax in axes:
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.30, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")

REF_W, REF_H = 2.9, 1.05


def dashed_ref(ax, cx=0.0, cy=0.0, lw=1.6):
    ax.add_patch(Ellipse((cx, cy), REF_W, REF_H, fill=False,
                 edgecolor=GRAY, lw=lw, ls=(0, (5, 4)), zorder=2))


def dir_arrow(ax, x0, y0, dx, dy, color, lw=2.3, zorder=6, mutation=13):
    ax.add_patch(FancyArrowPatch((x0, y0), (x0 + dx, y0 + dy),
                 arrowstyle="-|>", mutation_scale=mutation, lw=lw,
                 color=color, zorder=zorder, shrinkA=0, shrinkB=0))


def panel_header(ax, letter, interv, form):
    """Two-tone centred header: '(letter) Intervention ->' in gray, form in bold.

    The intervention names the lever; the bold form names what it achieves, so a
    reader can map lever -> geometric form from the figure alone.
    """
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    y = 1.46
    t_pre = ax.text(0, y, f"({letter})  {interv} → ", ha="left", va="top",
                    fontsize=13.5, color=SOFT)
    t_form = ax.text(0, y, form, ha="left", va="top",
                     fontsize=14, fontweight="bold", color=INK)

    def dwidth(t):
        bb = t.get_window_extent(renderer=renderer)
        (x0, _), (x1, _) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y0))
        return x1 - x0

    w_pre, w_form = dwidth(t_pre), dwidth(t_form)
    start = -(w_pre + w_form) / 2
    t_pre.set_x(start)
    t_form.set_x(start + w_pre)


def panel_footer(ax, text):
    ax.text(0, -1.14, text, ha="center", va="center",
            fontsize=10.4, style="italic", color=SOFT)


# ============================================================ (a) SELECTION
ax = axes[0]
panel_header(ax, "a", "Closed loop", "Tail reach")
ax.add_patch(Ellipse((0, 0), REF_W, REF_H, fill=False, edgecolor=INK, lw=2.6, zorder=1))
dashed_ref(ax)
dir_arrow(ax, 0, 0, REF_W/2*0.88, 0, GRAY)
dir_arrow(ax, 0, 0, 0, REF_H/2*0.80, GRAY)
rng = np.random.default_rng(7)
angles = np.array([160, 197, 235, 305, 342, 20, 62, 118]) * np.pi / 180
for th in angles:
    ex, ey = REF_W/2*0.94*np.cos(th), REF_H/2*0.94*np.sin(th)
    sx, sy = ex*0.30 + rng.normal(0, .05), ey*0.30 + rng.normal(0, .04)
    ax.scatter([ex], [ey], s=300, color=ACCENT, alpha=0.10, linewidths=0, zorder=2)
    ax.add_patch(FancyArrowPatch((sx, sy), (ex*0.965, ey*0.965), arrowstyle="-|>",
                 mutation_scale=9, lw=1.3, color=ACCENT, zorder=5, shrinkA=2, shrinkB=2,
                 connectionstyle=f"arc3,rad={rng.uniform(-.14,.14):.2f}"))
    ax.scatter([sx], [sy], s=26, facecolor="white", edgecolor=ACCENT, linewidths=1.1, zorder=6)
    ax.scatter([ex], [ey], s=30, color=ACCENT, zorder=6)
ax.text(REF_W/2*0.98, REF_H/2*0.72, "tails", fontsize=9.6, color=ACCENT, ha="left", style="italic")
panel_footer(ax, "points reach the tails;\nregion and directions unchanged")

# ============================================================ (b) TEMPERATURE
ax = axes[1]
panel_header(ax, "b", "Temperature", "Dimensional expansion")
dashed_ref(ax)
ax.add_patch(Ellipse((0, 0), REF_W*1.10, REF_H*1.85, fill=False, edgecolor=INK, lw=2.6, zorder=4))
dir_arrow(ax, 0, 0, REF_W/2*0.88, 0, GRAY)
dir_arrow(ax, 0, 0, 0, REF_H/2*0.80, GRAY)
dir_arrow(ax, 0, 0, 0, REF_H*1.85/2*0.90, ACCENT, lw=2.7, mutation=15)
ax.text(0.10, REF_H*1.85/2*0.62, "minor direction\nbecomes active", fontsize=9.0, color=ACCENT, ha="left", va="center")
panel_footer(ax, "more directions carry variance —\nsame span, same quality")

# ============================================================ (c) PROMPT
ax = axes[2]
panel_header(ax, "c", "Prompt", "Directional novelty")
cref = (-1.02, -0.46)
dashed_ref(ax, *cref)
dir_arrow(ax, *cref, REF_W/2*0.80, 0, GRAY, lw=2.0)
dir_arrow(ax, *cref, 0, REF_H/2*0.72, GRAY, lw=2.0)
cnew = (0.74, 0.26); theta = 24; t = np.deg2rad(theta)
ax.add_patch(Ellipse(cnew, REF_W, REF_H, angle=theta, fill=False, edgecolor=INK, lw=2.6, zorder=4))
dir_arrow(ax, *cnew, REF_W/2*0.86*np.cos(t), REF_W/2*0.86*np.sin(t), ACCENT, lw=2.5, mutation=14)
dir_arrow(ax, *cnew, -REF_H/2*0.78*np.sin(t), REF_H/2*0.78*np.cos(t), ACCENT, lw=2.5, mutation=14)
ax.add_patch(FancyArrowPatch(cref, (cnew[0]-0.42, cnew[1]-0.20), arrowstyle="-|>",
             mutation_scale=12, lw=1.5, color=ACCENT, alpha=0.55, ls=(0, (2, 2)),
             zorder=5, shrinkA=6, shrinkB=4))
ax.text(-0.52, -0.32, "relocation", fontsize=9.0, color=ACCENT, alpha=0.75, rotation=24, ha="center", style="italic")
ax.text(cnew[0]+0.82, cnew[1]-0.86, "new directions", fontsize=9.3, color=ACCENT, ha="center", va="center")
panel_footer(ax, "same-size region, new location and\norientation — new directions, same count on average")

_OUT = "fig_concept.pdf"
_PREV = "fig_concept_preview.png"
fig.savefig(_PREV, dpi=160)
fig.savefig(_OUT)
print("done")
