"""
Extended Data Fig. S3 | Within-Type RFC Variation
===================================================

独立绘图脚本 -- 从同目录下的 FigS4_WithinType_Variation.xlsx 读取数据

用法:
    python FigS4_WithinType_Variation.py

输出:
    FigS4_WithinType_Variation.pdf
    FigS4_WithinType_Variation.tiff
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_FILE = os.path.join(os.path.dirname(__file__), "FigS4_WithinType_Variation.xlsx")


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


def plot_ed_s3():
    df = pd.read_excel(DATA_FILE, sheet_name="15Case_RFC_Data")

    required_cols = ["Type", "Case", "Scenario", "Return_Period_yr", "DTFRI"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    types = {
        "pollution_control": ("Pollution Control", "#3498DB"),
        "flood_control": ("Flood Control", "#27AE60"),
        "multi_function": ("Multi-function", "#E74C3C"),
    }

    rp_dense = np.logspace(np.log10(1.5), np.log10(250), 80)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes = axes.flatten()

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.14,
        top=0.90,
        wspace=0.24
    )

    for idx, (typ_key, (typ_label, color)) in enumerate(types.items()):
        ax = axes[idx]

        td = df[df["Type"] == typ_key].copy()
        cases = td["Case"].dropna().unique()

        all_curves = []

        for case in cases:
            cd = td[
                (td["Case"] == case)
                & (td["Scenario"] == "Historical")
            ].sort_values("Return_Period_yr")

            if len(cd) >= 4:
                ufris = np.interp(
                    rp_dense,
                    cd["Return_Period_yr"],
                    cd["DTFRI"]
                )

                all_curves.append(ufris)

                ax.semilogx(
                    rp_dense,
                    ufris,
                    color=color,
                    alpha=0.32,
                    linewidth=1.1
                )

        if all_curves:
            ac = np.array(all_curves)


            ax.fill_between(
                rp_dense,
                ac.min(axis=0),
                ac.max(axis=0),
                color="gray",
                alpha=0.16,
                label="Min-max range"
            )


            ax.semilogx(
                rp_dense,
                np.median(ac, axis=0),
                color="black",
                linestyle="-",
                linewidth=3.0,
                label=f"Median (n={len(cases)})"
            )

        ax.axhline(
            y=0.5,
            color="red",
            linestyle=":",
            linewidth=1.3,
            alpha=0.45
        )

        ax.set_xlim(1.5, 250)
        ax.set_ylim(0, 1.05)

        ax.set_xlabel("Return Period (years)")
        ax.set_ylabel("DTFRI")

        ax.set_title(
            f"{typ_label}\n({len(cases)} cases)",
            fontweight="bold",
            fontsize=18
        )

        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.grid(True, alpha=0.15)

        ax.legend(
            loc="lower left",
            fontsize=14,
            frameon=True,
            framealpha=0.90,
            edgecolor="lightgray",
            handlelength=1.8,
            borderpad=0.5,
            labelspacing=0.45
        )


    output_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(output_dir, "FigS4_WithinType_Variation.pdf")
    tiff_path = os.path.join(output_dir, "FigS4_WithinType_Variation.tiff")

    try:
        fig.savefig(pdf_path, dpi=300, facecolor="white")
        print(f"Saved PDF: {pdf_path}")
    except PermissionError:
        pdf_path_new = os.path.join(output_dir, "FigS4_WithinType_Variation_new.pdf")
        fig.savefig(pdf_path_new, dpi=300, facecolor="white")
        print(f"Original PDF is occupied. Saved PDF as: {pdf_path_new}")

    try:
        fig.savefig(tiff_path, dpi=600, facecolor="white")
        print(f"Saved TIFF: {tiff_path}")
    except PermissionError:
        tiff_path_new = os.path.join(output_dir, "FigS4_WithinType_Variation_new.tiff")
        fig.savefig(tiff_path_new, dpi=600, facecolor="white")
        print(f"Original TIFF is occupied. Saved TIFF as: {tiff_path_new}")

    plt.close(fig)


if __name__ == "__main__":
    plot_ed_s3()