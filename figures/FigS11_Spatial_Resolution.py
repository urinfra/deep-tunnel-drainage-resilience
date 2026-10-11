"""
Extended Data Fig. S10 | Spatial Resolution MAUP Sensitivity (SA6)
====================================================================

独立绘图脚本 -- 从同目录下的 FigS11_Spatial_Resolution.xlsx 读取数据

用法:
    python FigS11_Spatial_Resolution.py

输出:
    FigS11_Spatial_Resolution.pdf
    FigS11_Spatial_Resolution.tiff
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_FILE = os.path.join(os.path.dirname(__file__), "FigS11_Spatial_Resolution.xlsx")

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

CITY_COLORS = {
    "Guangzhou": "#E74C3C",
    "Chicago": "#3498DB",
    "Tokyo": "#2ECC71",
    "Kuala Lumpur": "#F39C12",
}


def plot_ed_s10():
    """MAUP sensitivity: Gini coefficient vs grid resolution."""
    df = pd.read_excel(DATA_FILE, sheet_name="MAUP_Sensitivity")

    gini_cols = [c for c in df.columns if c.startswith("Gini_")]
    city_names = [c.replace("Gini_", "") for c in gini_cols]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.14, top=0.90, wspace=0.25
    )

    # ---- Panel a: Gini vs Grid Resolution ----
    ax = axes[0]
    for i, gini_col in enumerate(gini_cols):
        city = city_names[i]
        ax.plot(
            df["Grid_Size"],
            df[gini_col],
            "o-",
            color=CITY_COLORS.get(city, "#999"),
            linewidth=2.5,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=city,
        )

    ax.set_xlabel("Grid Resolution (cells per side)")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("a  Gini vs. Grid Resolution", fontweight="bold", loc="left")
    ax.legend(
        fontsize=13,
        frameon=True,
        framealpha=0.90,
        edgecolor="lightgray",
        loc="best",
    )
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.grid(True, alpha=0.15)

    # ---- Panel b: CV bar chart ----
    ax = axes[1]
    cv_values = []
    for i, gini_col in enumerate(gini_cols):
        vals = df[gini_col].values
        cv = np.std(vals) / np.mean(vals) * 100
        cv_values.append(cv)

    bar_colors = [CITY_COLORS.get(c, "#999") for c in city_names]
    # Dot plot (editorial E.6: individual data points instead of summary bars)
    ax.scatter(range(len(city_names)), cv_values, c=bar_colors, s=110,
               edgecolors="white", linewidth=1.2, zorder=3)

    for i, cv in enumerate(cv_values):
        ax.text(
            i,
            cv + max(cv_values) * 0.03,
            f"{cv:.1f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=bar_colors[i],
        )

    ax.set_xticks(range(len(city_names)))
    ax.set_xticklabels(city_names, fontsize=14)
    ax.set_ylabel("Coefficient of Variation (%)")
    ax.set_title("b  CV Across Resolutions", fontweight="bold", loc="left")
    ax.set_ylim(0, 1.1)
    ax.text(
        0.98, 0.97,
        "MAUP threshold: 2%\n(all cities far below)",
        transform=ax.transAxes, fontsize=12, color="red",
        ha="right", va="top", style="italic",
    )
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.grid(True, alpha=0.15, axis="y")

    fig.suptitle(
        "Fig. S11 | Spatial resolution MAUP sensitivity (SA6)",
        fontweight="bold",
        fontsize=17,
    )

    # Save outputs
    output_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(output_dir, "FigS11_Spatial_Resolution.pdf")
    tiff_path = os.path.join(output_dir, "FigS11_Spatial_Resolution.tiff")

    try:
        fig.savefig(pdf_path, dpi=300, facecolor="white")
        print(f"Saved PDF: {pdf_path}")
    except PermissionError:
        alt = os.path.join(output_dir, "FigS11_Spatial_Resolution_new.pdf")
        fig.savefig(alt, dpi=300, facecolor="white")
        print(f"Original PDF occupied. Saved as: {alt}")

    try:
        fig.savefig(tiff_path, dpi=600, facecolor="white")
        print(f"Saved TIFF: {tiff_path}")
    except PermissionError:
        alt = os.path.join(output_dir, "FigS11_Spatial_Resolution_new.tiff")
        fig.savefig(alt, dpi=600, facecolor="white")
        print(f"Original TIFF occupied. Saved as: {alt}")

    plt.close(fig)


if __name__ == "__main__":
    plot_ed_s10()
