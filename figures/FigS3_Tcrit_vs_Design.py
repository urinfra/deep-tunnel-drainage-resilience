"""
Extended Data Fig. S2 | Critical Return Period (T_crit) versus Design Standard
===============================================================================

独立绘图脚本 -- 从同目录下的 FigS3_Tcrit_vs_Design.xlsx 读取数据

用法:
    python FigS3_Tcrit_vs_Design.py

输出:
    FigS3_Tcrit_vs_Design.pdf
    FigS3_Tcrit_vs_Design.tiff
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


DATA_FILE = os.path.join(os.path.dirname(__file__), "FigS3_Tcrit_vs_Design.xlsx")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Nature-style type colours
TYPE_CONFIG = {
    "pollution_control": {
        "label": "Pollution Control",
        "color": "#3498DB",
        "marker": "o",
    },
    "flood_control": {
        "label": "Flood Control",
        "color": "#27AE60",
        "marker": "s",
    },
    "multi_function": {
        "label": "Multi-function",
        "color": "#E74C3C",
        "marker": "D",
    },
}


def plot_ed_s2():
    """T_crit vs design standard as a dumbbell plot: one row per case, hollow
    marker = design standard, filled marker = T_crit. The row layout makes
    every case label readable (no point/label overlap possible)."""
    df = pd.read_excel(DATA_FILE, sheet_name="15Case_RFC_Data")

    hist = df[df["Scenario"] == "Historical"].copy()
    cases = hist.groupby(["Case", "Type"]).agg(
        Design_Std_yr=("Design_Std_yr", "first"),
        T_crit_yr=("T_crit_yr", "first"),
        Storage_m3=("Storage_m3", "first"),
    ).reset_index()
    cases = cases.sort_values(["Design_Std_yr", "T_crit_yr"]).reset_index(drop=True)
    n = len(cases)
    ypos = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.subplots_adjust(left=0.26, right=0.97, bottom=0.10, top=0.93)

    for i, (_, r) in enumerate(cases.iterrows()):
        cfg = TYPE_CONFIG[r["Type"]]
        y = ypos[i]
        ax.plot([r["Design_Std_yr"], r["T_crit_yr"]], [y, y],
                color="grey", lw=1.2, alpha=0.55, zorder=2)
        ax.scatter([r["Design_Std_yr"]], [y], marker="o", s=140,
                   facecolor="white", edgecolor=cfg["color"], linewidth=1.8,
                   zorder=3)
        ax.scatter([r["T_crit_yr"]], [y], marker=cfg["marker"], s=150,
                   c=cfg["color"], edgecolors="white", linewidth=1.0, zorder=4)

    ax.set_yticks(ypos)
    ax.set_yticklabels(cases["Case"].values, fontsize=12)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 215)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("Return period (years)")
    ax.grid(axis="x", alpha=0.15)

    groups = [
        cases[cases["Type"] == t]["T_crit_yr"].values
        for t in ["pollution_control", "flood_control", "multi_function"]
    ]
    if all(len(g) > 0 for g in groups):
        h_stat, p_val = stats.kruskal(*groups)
        ax.text(
            0.98, 0.82,
            f"Kruskal-Wallis: H = {h_stat:.1f}, p = {p_val:.4f}",
            transform=ax.transAxes, fontsize=11, ha="right", va="top",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="lightgray", alpha=0.9),
        )

    ax.set_title(
        "Critical return period versus design standard, "
        "all 14 cases under Historical climate",
        fontweight="bold",
    )

    from matplotlib.lines import Line2D
    type_handles = [
        Line2D([], [], marker=cfg["marker"], color=cfg["color"], linestyle="none",
               markersize=8, label=cfg["label"])
        for cfg in TYPE_CONFIG.values()
    ]
    leg1 = ax.legend(handles=type_handles, loc="upper center", fontsize=12,
                     ncol=3, frameon=True, framealpha=0.9,
                     edgecolor="lightgray", bbox_to_anchor=(0.50, 0.97))
    ax.add_artist(leg1)
    ax.text(
        0.98, 0.70,
        "○ Design standard\n● T_crit (critical)",
        transform=ax.transAxes, fontsize=12, color="grey",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="lightgray", alpha=0.9),
    )

    # Save outputs
    output_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(output_dir, "FigS3_Tcrit_vs_Design.pdf")
    tiff_path = os.path.join(output_dir, "FigS3_Tcrit_vs_Design.tiff")

    try:
        fig.savefig(pdf_path, dpi=300, facecolor="white")
        print(f"Saved PDF: {pdf_path}")
    except PermissionError:
        alt = os.path.join(output_dir, "FigS3_Tcrit_vs_Design_new.pdf")
        fig.savefig(alt, dpi=300, facecolor="white")
        print(f"Original PDF occupied. Saved as: {alt}")

    try:
        fig.savefig(tiff_path, dpi=600, facecolor="white")
        print(f"Saved TIFF: {tiff_path}")
    except PermissionError:
        alt = os.path.join(output_dir, "FigS3_Tcrit_vs_Design_new.tiff")
        fig.savefig(alt, dpi=600, facecolor="white")
        print(f"Original TIFF occupied. Saved as: {alt}")

    plt.close(fig)


if __name__ == "__main__":
    plot_ed_s2()
