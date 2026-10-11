"""
Supplementary Fig. S1 | RFC Curves for All 15 Deep Tunnel Cases
================================================================

独立绘图脚本 -- 从同目录下的 FigS2_All14_RFC.xlsx 读取数据

用法:
    python FigS2_All14_RFC.py

输出:
    FigS2_All14_RFC.pdf
    FigS2_All14_RFC.tiff
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATA_FILE = os.path.join(os.path.dirname(__file__), "FigS2_All14_RFC.xlsx")


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],

    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 12,

    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})



TYPE_INFO = {
    "pollution_control": {
        "label": "Pollution Control",
        "color_map": "tab20",
    },
    "flood_control": {
        "label": "Flood Control",
        "color_map": "tab10",
    },
    "multi_function": {
        "label": "Multi-function",
        "color_map": "Set1",
    },
}


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]


def _get_case_design_year(case_data):
   
    possible_cols = [
        "Design_Std_yr",
        "Design_Standard_yr",
        "Design_Return_Period_yr",
        "Design_yr",
        "Design"
    ]

    for col in possible_cols:
        if col in case_data.columns:
            values = case_data[col].dropna().unique()
            if len(values) > 0:
                try:
                    return float(values[0])
                except Exception:
                    return None

    return None


def _plot_design_marker(ax, sc_data, design_year, color, marker):
   
    if sc_data.empty:
        return

    x = sc_data["Return_Period_yr"].values
    y = sc_data["DTFRI"].values

    if design_year is None:
        target_x = 50.0
    else:
        target_x = design_year


    idx_near = np.argmin(np.abs(x - target_x))
    mx = x[idx_near]
    my = y[idx_near]

    ax.plot(
        mx,
        my,
        marker=marker,
        markersize=7,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.7,
        linestyle="None",
        zorder=10
    )


def plot_ed_s1():
    df = pd.read_excel(DATA_FILE, sheet_name="15Case_RFC_Data")

    required_cols = ["Type", "Case", "Scenario", "Return_Period_yr", "DTFRI"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8))


    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.14,
        top=0.88,
        wspace=0.22
    )

    for ax_idx, (typ_key, info) in enumerate(TYPE_INFO.items()):
        ax = axes[ax_idx]

        type_data = df[df["Type"] == typ_key].copy()
        cases = list(type_data["Case"].dropna().unique())


        cmap = plt.get_cmap(info["color_map"])
        colors = [cmap(i / max(len(cases) - 1, 1)) for i in range(len(cases))]

        case_legend_handles = []

        for i, case in enumerate(cases):
            case_data = type_data[type_data["Case"] == case].copy()
            case_data = case_data.sort_values("Return_Period_yr")

            color = colors[i]
            marker = MARKERS[i % len(MARKERS)]


            hist = case_data[case_data["Scenario"] == "Historical"].sort_values("Return_Period_yr")
            if not hist.empty:
                ax.semilogx(
                    hist["Return_Period_yr"],
                    hist["DTFRI"],
                    color=color,
                    linestyle="-",
                    linewidth=1.8,
                    alpha=0.95
                )


            fut = case_data[
                case_data["Scenario"].isin(["SSP5-8.5_2080s", "SSP5-8.5 2080s"])
            ].sort_values("Return_Period_yr")

            if not fut.empty:
                ax.semilogx(
                    fut["Return_Period_yr"],
                    fut["DTFRI"],
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.85
                )

            design_year = _get_case_design_year(case_data)
            if not hist.empty:
                _plot_design_marker(ax, hist, design_year, color, marker)


            if design_year is not None:
                legend_label = f"{case} ({int(design_year)} years)"
            else:
                legend_label = f"{case}"

            case_legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker=marker,
                    linestyle="-",
                    linewidth=1.8,
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    label=legend_label
                )
            )

        ax.axhline(
            y=0.5,
            color="red",
            linestyle=":",
            linewidth=1.2,
            alpha=0.35
        )

        ax.set_xlim(1.5, 250)
        ax.set_ylim(0, 1.05)

        ax.set_xlabel("Return Period (years)")
        ax.set_ylabel("DTFRI")

        ax.set_title(
            f"{info['label']}\n({len(cases)} cases)",
            fontweight="bold",
            fontsize=18
        )

        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.grid(True, alpha=0.15)


        ax.legend(
            handles=case_legend_handles,
            loc="lower left",
            fontsize=12,
            frameon=True,
            framealpha=0.90,
            edgecolor="lightgray",
            handlelength=1.8,
            borderpad=0.4,
            labelspacing=0.35
        )



    output_dir = os.path.dirname(__file__)

    pdf_path = os.path.join(output_dir, "FigS2_All14_RFC.pdf")
    tiff_path = os.path.join(output_dir, "FigS2_All14_RFC.tiff")

    fig.savefig(pdf_path, dpi=300, facecolor="white")
    fig.savefig(tiff_path, dpi=600, facecolor="white")

    plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved TIFF: {tiff_path}")


if __name__ == "__main__":
    plot_ed_s1()