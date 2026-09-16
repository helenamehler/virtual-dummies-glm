"""
Fig. 1 of the paper: TPP (top) and FDP (bottom) against the SNR for the four
response models, drawn from the summaries the four sweeps wrote.

    python experiments/sec4_1_equivalence/plot_snr_panel.py

Reads  results/sec4_1_equivalence/<family>/summary.csv
Writes results/sec4_1_equivalence/fig1.pdf and fig1.png
"""
# NEW in this repository (Mehler, Koka, Muma, 2026), built on the virtual dummy construction of Koka et al. (2026). GPLv3, see NOTICE.
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# Dash patterns are given in points and must not scale with the line width.
matplotlib.rcParams["lines.scale_dashes"] = False
# 9 pt Times body text of the ICASSP template; STIX is the metric-compatible
# fallback that ships with matplotlib.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = [
    "Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "Liberation Serif",
    "STIXGeneral", "DejaVu Serif",
]
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FONT = {"title": 8.0, "xlabel": 7.5, "ylabel": 8.0, "tick": 6.5, "legend": 7.5}
FIGSIZE = (7.16, 1.3)        # inches; 7.16 is the ICASSP full text width
FDP_ROW = 0.62               # height of the FDP row relative to the TPP row
WEIGHT = 0.8                 # scales all line widths and marker sizes
ALPHA = 0.1                  # target FDR, drawn as a dotted line
L_FACTOR = 5

# Paul Tol's colour-blind safe "vibrant" palette. Every curve also has its own
# dash pattern and marker, so the figure reads in greyscale.
ROLES = {
    "ad":    dict(label="AD-GLM",      color="#CC3311", marker="o", ms=4.2, lw=2.3, z=2,
                  ls=(0, ())),
    "vd":    dict(label="VD-GLM",      color="#0077BB", marker="s", ms=3.7, lw=1.45, z=5,
                  ls=(0, (5.0, 2.4))),
    "gauss": dict(label="VD-Gauss",    color="#EE7733", marker="^", ms=3.5, lw=1.35, z=3,
                  ls=(0, (6.0, 1.8, 1.2, 1.8))),
    "ko":    dict(label="MX-Knockoff", color="#AA3377", marker="X", ms=3.8, lw=1.35, z=4,
                  ls=(0, (8.0, 2.0, 2.6, 2.0))),
}
OPEN_MARKER = ("gauss",)
MARKER_EDGE_WIDTH = 1.15
LEGEND_LW, LEGEND_MS, LEGEND_MEW = 1.5, 4.0, 1.15

# Column order and the method names each sweep writes, per role.
FAMILIES = [
    ("logistic", "Logistic", {"ad": "AD-AFS-Log-T-Rex", "vd": "VD-AFS-Log-T-Rex",
                              "gauss": "VD-AFS-Gauss", "ko": "MX-Knockoff-Log"}),
    ("poisson", "Poisson", {"ad": "AD-AFS-Pois-T-Rex", "vd": "VD-AFS-Pois-T-Rex",
                            "gauss": "VD-AFS-Gauss", "ko": "MX-Knockoff-Pois"}),
    ("cox", "Cox", {"ad": "AD-AFS-Cox-T-Rex", "vd": "VD-AFS-Cox-T-Rex",
                    "gauss": "VD-AFS-Gauss", "ko": "MX-Knockoff-Cox"}),
    ("multinomial", "Multinomial", {"ad": "AD-AFS-Multinom-T-Rex",
                                    "vd": "VD-AFS-Multinom-T-Rex",
                                    "gauss": "VD-AFS-Gauss (ordinal)",
                                    "ko": "MX-Knockoff-Multinom"}),
]


def thin_ticks(xs, min_gap_frac=0.18):
    """Tick on the simulated grid values, dropping labels that would collide."""
    xs = np.sort(np.unique(np.asarray(xs, dtype=float)))
    if xs.size <= 2 or xs[-1] <= xs[0]:
        return xs
    span = xs[-1] - xs[0]
    keep = [xs[0]]
    for v in xs[1:-1]:
        if (v - keep[-1]) / span >= min_gap_frac:
            keep.append(v)
    if (xs[-1] - keep[-1]) / span < min_gap_frac:
        keep.pop()
    keep.append(xs[-1])
    return np.array(keep)


def build_figure(panels):
    mew = MARKER_EDGE_WIDTH * WEIGHT
    rows = ("tpp", "fdp")
    fig, axes = plt.subplots(
        2, len(panels), figsize=FIGSIZE, sharey="row",
        gridspec_kw=dict(hspace=0.13, wspace=0.10, height_ratios=[1.0, FDP_ROW]),
    )
    handles = {}
    for c, (title, df, methods) in enumerate(panels):
        for r, metric in enumerate(rows):
            ax = axes[r, c]
            for role, st in ROLES.items():
                g = df[df["method"] == methods[role]].sort_values("snr")
                if g.empty:
                    continue
                (line,) = ax.plot(
                    g["snr"], g[metric], color=st["color"], linestyle=st["ls"],
                    linewidth=st["lw"] * WEIGHT, marker=st["marker"],
                    markersize=st["ms"] * WEIGHT,
                    markerfacecolor="none" if role in OPEN_MARKER else st["color"],
                    markeredgewidth=mew, zorder=st["z"], clip_on=True,
                )
                handles.setdefault(role, line)
            if metric == "fdp":
                ax.axhline(ALPHA, color="k", ls=(0, (1, 1.6)), lw=1.2, zorder=1)
            else:
                ax.set_ylim(-0.03, 1.03)
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            if r == 0:
                ax.set_title(title, fontsize=FONT["title"], pad=3)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("SNR", fontsize=FONT["xlabel"], labelpad=1.5)
            ax.grid(True, ls="--", alpha=0.30, linewidth=0.6)
            ax.tick_params(labelsize=FONT["tick"], length=2.5, pad=1.5)
            ax.set_xticks(thin_ticks(df["snr"].unique()))
            if c == 0:
                ax.set_ylabel(metric.upper(), fontsize=FONT["ylabel"], labelpad=2)

    # The FDP row stops just above the target line and the drawn curves.
    fdp_max = max(df[df["method"].isin(methods.values())]["fdp"].max()
                  for _, df, methods in panels)
    top = max(ALPHA * 1.45, float(fdp_max) * 1.30)
    for ax in axes[1]:
        ax.set_ylim(-top * 0.07, top)
        ax.locator_params(axis="y", nbins=4)

    leg = fig.legend([handles[r] for r in ROLES if r in handles],
                     [ROLES[r]["label"] for r in ROLES if r in handles],
                     loc="upper center", ncol=len(handles), frameon=False,
                     fontsize=FONT["legend"], bbox_to_anchor=(0.5, 0.0),
                     handlelength=3.6, columnspacing=1.3, handletextpad=0.45,
                     numpoints=1, borderpad=0.0, borderaxespad=0.0)
    for h in leg.legend_handles:           # one weight for every legend entry
        h.set_linewidth(LEGEND_LW * WEIGHT)
        h.set_markersize(LEGEND_MS * WEIGHT)
        h.set_markeredgewidth(LEGEND_MEW * WEIGHT)
    # Place the legend a fixed distance in inches below the x-axis labels.
    leg.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom_px = min(ax.get_tightbbox(renderer).y0 for ax in axes[-1])
    leg.set_visible(True)
    leg.set_bbox_to_anchor(((0.5, (bottom_px / fig.dpi - 0.06) / fig.get_figheight())),
                           transform=fig.transFigure)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/sec4_1_equivalence")
    ap.add_argument("--stem", default="fig1", help="output file name without extension")
    args = ap.parse_args(argv)

    root = Path(args.results)
    panels = []
    for key, title, methods in FAMILIES:
        df = pd.read_csv(root / key / "summary.csv")
        panels.append((title, df[df["L"] == L_FACTOR], methods))

    fig = build_figure(panels)
    for ext in ("pdf", "png"):
        fig.savefig(root / f"{args.stem}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {root / (args.stem + '.pdf')} (+ .png)")


if __name__ == "__main__":
    main()
