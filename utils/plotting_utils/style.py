import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

thesis_red = (0.64, 0.11, 0.19)
thesis_blue = (0, 0.38, 0.68)
ifisc_green = (0.73, 0.83, 0.01)

mycmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_cmap", [thesis_blue, ifisc_green, thesis_red]
)

RC_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["newpx"],
    "text.latex.preamble": r"\usepackage{newpxtext,newpxmath}",
    "font.size": 21,
    "axes.titlesize": 24,
    "axes.labelsize": 21,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 27,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
}

_style_applied = False


def apply_style():
    global _style_applied
    if not _style_applied:
        plt.rcParams.update(RC_PARAMS)
        _style_applied = True
