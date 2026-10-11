"""
ED Fig. SS13 |
SA9 | Sediment long-term impact
{'='*50}
独立绘图脚本 -- 从 FigS14_Sediment_Impact.xlsx 读取数据
用法: python FigS14_Sediment_Impact.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = os.path.join(os.path.dirname(__file__), "FigS14_Sediment_Impact.xlsx")

CITY_ORDER = ["Guangzhou", "Chicago", "Tokyo", "Kuala Lumpur"]

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

def get_axis_limits(x_vals, y_vals):
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    x_min = np.nanmin(x_vals)
    x_max = np.nanmax(x_vals)
    y_min = np.nanmin(y_vals)
    y_max = np.nanmax(y_vals)

    if x_min == x_max:
        x_pad = 1.0
    else:
        x_pad = max(1.0, 0.05 * (x_max - x_min))

    if y_min == y_max:
        y_pad = 0.03
    else:
        y_pad = max(0.03, 0.08 * (y_max - y_min))

    x_lower = x_min - x_pad
    x_upper = x_max + x_pad

    y_lower = max(0.0, y_min - y_pad)
    y_upper = min(1.05, y_max + y_pad)

    return x_lower, x_upper, y_lower, y_upper

def plot_sa9():
    xl = pd.ExcelFile(DATA_FILE)
    data_sheets = [s for s in xl.sheet_names if str(s).lower() not in ["description", "readme"]]

    if not data_sheets:
        raise ValueError("No valid data sheet found in the Excel file.")

    df = pd.read_excel(DATA_FILE, sheet_name=data_sheets[0])

    required_cols = ["City", "Year", "DTFRI_at_T50", "End_n_Target"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    city_list = [c for c in CITY_ORDER if c in df["City"].dropna().unique()]
    if len(city_list) == 0:
        city_list = list(df["City"].dropna().unique())[:4]

    target_list = sorted(df["End_n_Target"].dropna().unique())
    cmap = plt.get_cmap("viridis", len(target_list))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.96,
        wspace=0.22,
        hspace=0.28
    )

    panel_labels = ["a", "b", "c", "d"]

    for i, ax in enumerate(axes):
        if i >= len(city_list):
            ax.axis("off")
            continue

        city = city_list[i]
        city_df = df[df["City"] == city].copy()

        plotted_x = []
        plotted_y = []

        for j, target in enumerate(target_list):
            sub = city_df[city_df["End_n_Target"] == target].copy()
            sub = sub.dropna(subset=["Year", "DTFRI_at_T50"]).sort_values("Year")

            if len(sub) == 0:
                continue

            x_vals = sub["Year"].values
            y_vals = sub["DTFRI_at_T50"].values

            plotted_x.extend(x_vals.tolist())
            plotted_y.extend(y_vals.tolist())

            ax.plot(
                x_vals,
                y_vals,
                "o-",
                linewidth=2.4,
                markersize=6.8,
                markeredgecolor="white",
                markeredgewidth=0.8,
                color=cmap(j),
                label=f"End n = {target:.3f}"
            )

        if len(plotted_x) > 0 and len(plotted_y) > 0:
            x_lower, x_upper, y_lower, y_upper = get_axis_limits(plotted_x, plotted_y)
            ax.set_xlim(x_lower, x_upper)
            ax.set_ylim(y_lower, y_upper)

            unique_years = sorted(city_df["Year"].dropna().unique())
            ax.set_xticks(unique_years)

        ax.set_title(
            f"{panel_labels[i]}  {city}",
            fontweight="bold",
            loc="left",
            fontsize=18,
            pad=4
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("DTFRI at T=50 years")
        ax.grid(True, alpha=0.20)
        ax.tick_params(axis="both", which="major", labelsize=15)

        ax.legend(
            fontsize=13.5,
            loc="best",
            frameon=True,
            framealpha=0.92,
            edgecolor="lightgray",
            handlelength=2.0,
            borderpad=0.6,
            labelspacing=0.4
        )

    output_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(output_dir, "FigS14_Sediment_Impact.pdf")
    tiff_path = os.path.join(output_dir, "FigS14_Sediment_Impact.tiff")

    try:
        fig.savefig(pdf_path, dpi=300, facecolor="white")
        print(f"Saved PDF: {pdf_path}")
    except PermissionError:
        pdf_path_new = os.path.join(output_dir, "FigS14_Sediment_Impact_new.pdf")
        fig.savefig(pdf_path_new, dpi=300, facecolor="white")
        print(f"Original PDF is occupied. Saved PDF as: {pdf_path_new}")

    try:
        fig.savefig(tiff_path, dpi=600, facecolor="white")
        print(f"Saved TIFF: {tiff_path}")
    except PermissionError:
        tiff_path_new = os.path.join(output_dir, "FigS14_Sediment_Impact_new.tiff")
        fig.savefig(tiff_path_new, dpi=600, facecolor="white")
        print(f"Original TIFF is occupied. Saved TIFF as: {tiff_path_new}")

    plt.close(fig)

if __name__ == "__main__":
    plot_sa9()