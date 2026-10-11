"""
ED Fig. SS9 |
SA5 | Monte Carlo uncertainty
{'='*50}
独立绘图脚本 -- 从 FigS10_Monte_Carlo.xlsx 读取数据
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = os.path.join(os.path.dirname(__file__), "FigS10_Monte_Carlo.xlsx")

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

def plot_sa5():
    df = pd.read_excel(DATA_FILE, sheet_name="Monte_Carlo_Samples")

    cities = ["Guangzhou", "Chicago", "Tokyo", "Kuala Lumpur"]
    cities = [c for c in cities if c in df["City"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.15,
        top=0.88,
        wspace=0.28
    )

    ax = axes[0]

    ufri_data = [
        df[df["City"] == c]["DTFRI_T50"].dropna().values
        for c in cities
    ]

    positions = np.arange(1, len(cities) + 1)

    bp = ax.boxplot(
        ufri_data,
        positions=positions,
        patch_artist=True,
        widths=0.60,
        showfliers=True
    )

    for patch, city in zip(bp["boxes"], cities):
        patch.set_facecolor(CITY_COLORS.get(city, "#999999"))
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_linewidth(1.2)

    ax.set_xticks(positions)
    ax.set_xticklabels(cities, rotation=15, ha="right")
    ax.set_ylabel("DTFRI at T=50 years")
    ax.set_title("a", fontweight="bold", loc="left", fontsize=18, pad=4)
    ax.grid(True, alpha=0.20, axis="y")
    ax.tick_params(axis="both", which="major", labelsize=15)

    ax = axes[1]

    for city in cities:
        cd = df[df["City"] == city]
        ax.scatter(
            cd["Climate_Factor"],
            cd["DTFRI_T50"],
            c=CITY_COLORS.get(city, "#999999"),
            alpha=0.35,
            s=28,
            label=city,
            edgecolors="none"
        )

    ax.set_xlabel("Climate Factor")
    ax.set_ylabel("DTFRI at T=50 years")
    ax.set_title("b", fontweight="bold", loc="left", fontsize=18, pad=4)

    ax.legend(
        fontsize=13,
        markerscale=2.0,
        loc="upper left",
        bbox_to_anchor=(0.08, 1.085),
        ncol=len(cities),
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
        borderaxespad=0.0
    )

    ax.grid(True, alpha=0.20)
    ax.tick_params(axis="both", which="major", labelsize=15)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(output_dir, "FigS10_Monte_Carlo.pdf")
    tiff_path = os.path.join(output_dir, "FigS10_Monte_Carlo.tiff")

    try:
        fig.savefig(pdf_path, dpi=300, facecolor="white")
        print(f"Saved PDF: {pdf_path}")
    except PermissionError:
        pdf_path_new = os.path.join(output_dir, "FigS10_Monte_Carlo_new.pdf")
        fig.savefig(pdf_path_new, dpi=300, facecolor="white")
        print(f"Original PDF is occupied. Saved PDF as: {pdf_path_new}")

    try:
        fig.savefig(tiff_path, dpi=600, facecolor="white")
        print(f"Saved TIFF: {tiff_path}")
    except PermissionError:
        tiff_path_new = os.path.join(output_dir, "FigS10_Monte_Carlo_new.tiff")
        fig.savefig(tiff_path_new, dpi=600, facecolor="white")
        print(f"Original TIFF is occupied. Saved TIFF as: {tiff_path_new}")

    plt.close(fig)

if __name__ == "__main__":
    plot_sa5()