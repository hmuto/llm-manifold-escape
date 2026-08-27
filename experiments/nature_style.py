"""
Nature Reviews artwork style configuration.

Based on:
- https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/
- https://www.nature.com/documents/natrev-artworkguide.pdf

Key requirements:
- Font: Sans-serif (Arial/Helvetica), 5-7 pt for text, 8 pt bold for panel labels
- Line width: minimum 0.25 pt
- Dimensions: single column 89 mm, double column 183 mm, max height 247 mm
- Resolution: 300+ DPI, vector preferred (PDF/EPS)
- Color: RGB, colorblind-accessible
- Axes: include axis lines and tick marks, label with units in parentheses
- No background gridlines, drop shadows, patterns, or colored text
"""

import matplotlib.pyplot as plt

# Dimensions in inches (converted from mm)
SINGLE_COL = 89 / 25.4    # 3.504 in
ONE_HALF_COL = 120 / 25.4  # 4.724 in
DOUBLE_COL = 183 / 25.4    # 7.205 in
MAX_HEIGHT = 247 / 25.4    # 9.724 in

# Colorblind-accessible palette (Wong, 2011; Nature recommended)
# https://www.nature.com/articles/nmeth.1618
COLORS = {
    'blue':    '#0072B2',
    'orange':  '#E69F00',
    'green':   '#009E73',
    'red':     '#D55E00',
    'purple':  '#CC79A7',
    'cyan':    '#56B4E9',
    'yellow':  '#F0E442',
    'black':   '#000000',
    'gray':    '#999999',
}

# Condition-specific colors (colorblind-safe)
COND_COLORS = {
    'dds_alpha_0.5': COLORS['green'],
    'dds_alpha_0.0': COLORS['blue'],
    'dds_alpha_1.0': COLORS['orange'],
    'map_elites':    COLORS['purple'],
    'independent':   COLORS['gray'],
    'debate':        COLORS['red'],
}

COND_LABELS = {
    'dds_alpha_0.5': r'DDS $\alpha$=0.5',
    'dds_alpha_0.0': r'DDS $\alpha$=0.0',
    'dds_alpha_1.0': r'DDS $\alpha$=1.0',
    'map_elites':    'MAP-Elites',
    'independent':   'Independent',
    'debate':        'Debate',
}

COND_MARKERS = {
    'dds_alpha_0.5': 'o',
    'dds_alpha_0.0': 's',
    'dds_alpha_1.0': 'D',
    'map_elites':    'P',
    'independent':   '^',
    'debate':        'v',
}


def apply_nature_style():
    """Apply Nature Reviews figure style globally."""
    plt.rcParams.update({
        # Font: sans-serif, Arial/Helvetica
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],

        # Font sizes (Nature: 5-7 pt for text, 8 pt for panel labels)
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,

        # Line widths (Nature: minimum 0.25 pt)
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,

        # Tick marks
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Remove top and right spines (cleaner look)
        'axes.spines.top': False,
        'axes.spines.right': False,

        # No grid
        'axes.grid': False,

        # Resolution
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.borderpad': 0.3,
        'legend.handlelength': 1.5,

        # Padding
        'axes.titlepad': 4,
        'axes.labelpad': 3,
    })


def panel_label(ax, label, x=-0.12, y=1.08):
    """Add Nature-style panel label (8 pt bold lowercase)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left')
