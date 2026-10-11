"""
ED Fig. SS6 |
SA2 | Climate factor sensitivity
{'='*50}
独立绘图脚本 -- 从 FigS7_Climate_Factor_Sensitivity.xlsx 读取数据
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = os.path.join(os.path.dirname(__file__), "FigS7_Climate_Factor_Sensitivity.xlsx")

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
    "legend.fontsize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def plot_sa2():
    df = pd.read_excel(DATA_FILE, sheet_name="Tcrit_vs_CF")

    fig, ax = plt.subplots(figsize=(10, 7))

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.14,
        top=0.96
    )

    cf_vals = df["Climate_Factor"].values

    for col in df.columns:
        if col.startswith("T_crit_"):
            city = col.replace("T_crit_", "")
            ax.plot(
                cf_vals,
                df[col].values,
                "o-",
                color=CITY_COLORS.get(city, "#999999"),
                linewidth=2.6,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=city
            )

    ax.axvline(
        x=1.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.55
    )

    ax.set_xlabel("Climate Amplification Factor")
    ax.set_ylabel(r"Critical Return Period, $T_{crit}$ (years)")

    ax.tick_params(axis="both", which="major", labelsize=15)

    ax.legend(
        loc="best",
        fontsize=14,
        frameon=True,
        framealpha=0.90,
        edgecolor="lightgray"
    )

    ax.grid(True, alpha=0.20)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(output_dir, "FigS7_Climate_Factor_Sensitivity.pdf")
    tiff_path = os.path.join(output_dir, "FigS7_Climate_Factor_Sensitivity.tiff")

    try:
        fig.savefig(pdf_path, dpi=300, facecolor="white")
        print(f"Saved PDF: {pdf_path}")
    except PermissionError:
        pdf_path_new = os.path.join(output_dir, "FigS7_Climate_Factor_Sensitivity_new.pdf")
        fig.savefig(pdf_path_new, dpi=300, facecolor="white")
        print(f"Original PDF is occupied. Saved PDF as: {pdf_path_new}")

    try:
        fig.savefig(tiff_path, dpi=600, facecolor="white")
        print(f"Saved TIFF: {tiff_path}")
    except PermissionError:
        tiff_path_new = os.path.join(output_dir, "FigS7_Climate_Factor_Sensitivity_new.tiff")
        fig.savefig(tiff_path_new, dpi=600, facecolor="white")
        print(f"Original TIFF is occupied. Saved TIFF as: {tiff_path_new}")

    plt.close(fig)

if __name__ == "__main__":
    plot_sa2()