"""
ED Fig. S5 | 4R Weight Sensitivity Analysis
===========================================

独立绘图脚本 -- 从同目录下的 FigS6_Weight_Sensitivity.xlsx 读取数据

用法:
    python FigS6_Weight_Sensitivity.py

输出:
    FigS6_Weight_Sensitivity.pdf
    FigS6_Weight_Sensitivity.tiff
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = os.path.join(os.path.dirname(__file__), "FigS6_Weight_Sensitivity.xlsx")

CITY_LABELS = ["Guangzhou", "Chicago", "Tokyo", "Kuala Lumpur"]

CITY_COLORS = {
    "Guangzhou": "#E74C3C",
    "Chicago": "#3498DB",
    "Tokyo": "#2ECC71",
    "Kuala Lumpur": "#F39C12",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def plot_sa1():
    rank_df = pd.read_excel(DATA_FILE, sheet_name="Rank_Matrix", index_col=0)

    available_labels = [c for c in CITY_LABELS if c in rank_df.index]
    if len(available_labels) == len(CITY_LABELS):
        rank_df = rank_df.loc[CITY_LABELS]
    else:
        rank_df = rank_df.iloc[:4, :]
        rank_df.index = CITY_LABELS[:len(rank_df)]

    rank_matrix = rank_df.values.astype(float)
    n_samples = rank_matrix.shape[1]
    unique_ranks = np.sort(np.unique(rank_matrix[~np.isnan(rank_matrix)]))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6.8),
        gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.14,
        top=0.92,
        wspace=0.28
    )

    ax = axes[0]
    im = ax.imshow(
        rank_matrix,
        aspect="auto",
        cmap="RdYlBu_r",
        vmin=np.nanmin(rank_matrix),
        vmax=np.nanmax(rank_matrix),
        interpolation="nearest"
    )

    ax.set_xlabel("Weight perturbation sample (#)")
    ax.set_ylabel("")
    ax.set_yticks(np.arange(len(rank_df.index)))
    ax.set_yticklabels(rank_df.index)

    xticks = np.arange(0, n_samples, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i) for i in xticks])

    ax.set_title(
        "a  Rank stability across 27 Dirichlet samples",
        fontweight="bold",
        loc="left",
        fontsize=18
    )

    ax.tick_params(axis="both", which="major", labelsize=15)

    cbar = fig.colorbar(
        im,
        ax=ax,
        shrink=0.78,
        pad=0.045
    )
    cbar.set_label("Rank (1 = highest DTFRI)", fontsize=15)
    cbar.set_ticks(unique_ranks)
    cbar.ax.tick_params(labelsize=14)

    ax = axes[1]
    n_cities = len(rank_df.index)
    n_ranks = len(unique_ranks)

    if n_ranks == 1:
        bar_width = 0.12
    else:
        bar_width = min(0.65 / n_cities, 0.18)

    offsets = np.linspace(
        -bar_width * (n_cities - 1) / 2,
        bar_width * (n_cities - 1) / 2,
        n_cities
    )

    for i, city in enumerate(rank_df.index):
        counts = np.array([
            np.sum(rank_matrix[i, :] == r)
            for r in unique_ranks
        ])
        frequency_pct = counts / n_samples * 100.0

        ax.bar(
            unique_ranks + offsets[i],
            frequency_pct,
            width=bar_width,
            color=CITY_COLORS.get(city, "#999999"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
            label=city
        )

    if len(unique_ranks) == 1:
        x0 = unique_ranks[0]
        ax.set_xlim(x0 - 1.5, x0 + 1.5)
    else:
        ax.set_xlim(unique_ranks.min() - 0.75, unique_ranks.max() + 0.75)

    ax.set_ylim(0, 105)
    ax.set_xticks(unique_ranks)
    ax.set_xticklabels([f"{r:g}" for r in unique_ranks])
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency (%)")

    ax.set_title(
        "b  Rank frequency distribution",
        fontweight="bold",
        loc="left",
        fontsize=18
    )

    ax.legend(
        loc="upper right",
        fontsize=13,
        frameon=True,
        framealpha=0.90,
        edgecolor="lightgray"
    )

    ax.grid(True, axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=15)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(output_dir, "FigS6_Weight_Sensitivity.pdf")
    tiff_path = os.path.join(output_dir, "FigS6_Weight_Sensitivity.tiff")

    try:
        fig.savefig(pdf_path, dpi=300, facecolor="white")
        print(f"Saved PDF: {pdf_path}")
    except PermissionError:
        pdf_path_new = os.path.join(output_dir, "FigS6_Weight_Sensitivity_new.pdf")
        fig.savefig(pdf_path_new, dpi=300, facecolor="white")
        print(f"Original PDF is occupied. Saved PDF as: {pdf_path_new}")

    try:
        fig.savefig(tiff_path, dpi=600, facecolor="white")
        print(f"Saved TIFF: {tiff_path}")
    except PermissionError:
        tiff_path_new = os.path.join(output_dir, "FigS6_Weight_Sensitivity_new.tiff")
        fig.savefig(tiff_path_new, dpi=600, facecolor="white")
        print(f"Original TIFF is occupied. Saved TIFF as: {tiff_path_new}")

    plt.close(fig)

if __name__ == "__main__":
    plot_sa1()