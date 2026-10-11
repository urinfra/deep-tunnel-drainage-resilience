"""
ED Fig. SS7 |
SA3 | Design standard sensitivity
{'='*50}
独立绘图脚本 -- 从 FigS8_Design_Standard_Sensitivity.xlsx 读取数据
用法: python FigS8_Design_Standard_Sensitivity.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = os.path.join(os.path.dirname(__file__), "FigS8_Design_Standard_Sensitivity.xlsx")

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

def plot_sa3():
    xl = pd.ExcelFile(DATA_FILE)

    fig, ax = plt.subplots(figsize=(10, 7))

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.14,
        top=0.96
    )

    data_sheets = [s for s in xl.sheet_names if s != "Description"]
    numeric_cols = []

    if data_sheets:
        df = pd.read_excel(DATA_FILE, sheet_name=data_sheets[0])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            x = df[numeric_cols[0]].values

            for col in numeric_cols[1:min(6, len(numeric_cols))]:
                ax.plot(
                    x,
                    df[col].values,
                    "o-",
                    linewidth=2.5,
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=str(col)
                )

            ax.legend(
                fontsize=14,
                loc="best",
                frameon=True,
                framealpha=0.90,
                edgecolor="lightgray"
            )

        else:
            ax.text(
                0.5,
                0.5,
                f"Data preview:\n{df.head().to_string()}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                family="monospace"
            )

    ax.set_xlabel(numeric_cols[0] if len(numeric_cols) > 0 else "X")
    ax.set_ylabel("Value")

    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.grid(True, alpha=0.20)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(output_dir, "FigS8_Design_Standard_Sensitivity.pdf")
    tiff_path = os.path.join(output_dir, "FigS8_Design_Standard_Sensitivity.tiff")

    try:
        fig.savefig(pdf_path, dpi=300, facecolor="white")
        print(f"Saved PDF: {pdf_path}")
    except PermissionError:
        pdf_path_new = os.path.join(output_dir, "FigS8_Design_Standard_Sensitivity_new.pdf")
        fig.savefig(pdf_path_new, dpi=300, facecolor="white")
        print(f"Original PDF is occupied. Saved PDF as: {pdf_path_new}")

    try:
        fig.savefig(tiff_path, dpi=600, facecolor="white")
        print(f"Saved TIFF: {tiff_path}")
    except PermissionError:
        tiff_path_new = os.path.join(output_dir, "FigS8_Design_Standard_Sensitivity_new.tiff")
        fig.savefig(tiff_path_new, dpi=600, facecolor="white")
        print(f"Original TIFF is occupied. Saved TIFF as: {tiff_path_new}")

    plt.close(fig)

if __name__ == "__main__":
    plot_sa3()