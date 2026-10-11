"""
ED Fig. SS12 |
SA8 | T_crit clustering
{'='*50}
独立绘图脚本 -- 从 FigS13_Tcrit_Clustering.xlsx 读取数据
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

DATA_FILE = os.path.join(os.path.dirname(__file__), "FigS13_Tcrit_Clustering.xlsx")

TYPE_ORDER = ["Pollution control", "Flood control", "Multi-function"]

TYPE_COLORS = {
    "Pollution control": "#3498DB",
    "Flood control": "#27AE60",
    "Multi-function": "#E74C3C",
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

def adjust_label_positions(y_values, min_gap=8.0):
    y_values = np.asarray(y_values, dtype=float)
    order = np.argsort(y_values)
    sorted_y = y_values[order].copy()

    for i in range(1, len(sorted_y)):
        if sorted_y[i] - sorted_y[i - 1] < min_gap:
            sorted_y[i] = sorted_y[i - 1] + min_gap

    adjusted = np.empty_like(sorted_y)
    adjusted[order] = sorted_y
    return adjusted
def add_case_labels(ax, df):
    label_df = df.sort_values(["T_crit_Historical", "T_crit_SSP585_2080s"]).reset_index(drop=True)

    manual_offsets = {
        "HK HATS": {
            "xytext": (0, 18),
            "ha": "center",
            "va": "bottom"
        },
        "Chicago": {
            "xytext": (14, -8),
            "ha": "left",
            "va": "top"
        }
    }

    manual_cases = set(manual_offsets.keys())
    rest_df = label_df[~label_df["Case"].astype(str).isin(manual_cases)].copy()

    if len(rest_df) > 0:
        adjusted_y = adjust_label_positions(rest_df["T_crit_SSP585_2080s"].values, min_gap=10.0)
    else:
        adjusted_y = []

    for _, r in label_df.iterrows():
        case = str(r["Case"])
        x = r["T_crit_Historical"]
        y = r["T_crit_SSP585_2080s"]

        if case in manual_offsets:
            p = manual_offsets[case]
            ax.annotate(
                case,
                xy=(x, y),
                xytext=p["xytext"],
                textcoords="offset points",
                fontsize=12,
                ha=p["ha"],
                va=p["va"],
                color="black",
                clip_on=False,
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78
                }
            )

    for (_, r), label_y in zip(rest_df.iterrows(), adjusted_y):
        ax.text(
            r["T_crit_Historical"] + 8.0,
            label_y,
            str(r["Case"]),
            fontsize=12,
            ha="left",
            va="center",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78
            }
        )

def plot_sa8():
    df = pd.read_excel(DATA_FILE, sheet_name="Tcrit_Clustering")

    types = [t for t in TYPE_ORDER if t in df["Type"].unique()]
    if len(types) == 0:
        types = list(df["Type"].dropna().unique())

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7)
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.17,
        top=0.92,
        wspace=0.30
    )

    ax = axes[0]

    data_by_type = [
        df[df["Type"] == t]["T_crit_Historical"].dropna().values
        for t in types
    ]

    bp = ax.boxplot(
        data_by_type,
        labels=types,
        patch_artist=True,
        widths=0.52,
        showfliers=True
    )

    for patch, t in zip(bp["boxes"], types):
        patch.set_facecolor(TYPE_COLORS.get(t, "#999999"))
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_linewidth(1.2)

    ax.set_ylabel(r"Critical Return Period, $T_{crit}$ (years)")
    ax.set_title("a", fontweight="bold", loc="left", fontsize=18, pad=4)
    ax.grid(True, alpha=0.20, axis="y")
    ax.tick_params(axis="both", which="major", labelsize=15)

    for tick in ax.get_xticklabels():
        tick.set_rotation(12)
        tick.set_ha("right")

    ax = axes[1]

    for t in types:
        td = df[df["Type"] == t]
        ax.scatter(
            td["T_crit_Historical"],
            td["T_crit_SSP585_2080s"],
            c=TYPE_COLORS.get(t, "#999999"),
            s=110,
            label=t,
            edgecolors="white",
            linewidths=0.9,
            alpha=0.90
        )

    add_case_labels(ax, df)

    max_val = max(
        df["T_crit_Historical"].max(),
        df["T_crit_SSP585_2080s"].max()
    )

    axis_max = max(250, np.ceil(max_val / 50) * 50)

    ax.plot(
        [0, axis_max],
        [0, axis_max],
        "k--",
        alpha=0.30,
        linewidth=1.2
    )

    ax.set_xlim(0, axis_max * 1.08)
    ax.set_ylim(0, axis_max * 1.05)

    ax.set_xlabel(r"$T_{crit}$ Historical (years)")
    ax.set_ylabel(r"$T_{crit}$ SSP5-8.5 2080s (years)")
    ax.set_title("b", fontweight="bold", loc="left", fontsize=18, pad=4)

    ax.legend(
        fontsize=15,
        loc="upper left",
        bbox_to_anchor=(0.08, 1.09),
        ncol=3,
        frameon=False,
        handlelength=1.3,
        columnspacing=1.2,
        borderaxespad=0.0,
        markerscale=1.25
    )

    ax.grid(True, alpha=0.20)
    ax.tick_params(axis="both", which="major", labelsize=15)

    h_stat, p_val = stats.kruskal(*data_by_type)

    fig.text(
        0.50,
        0.035,
        f"Kruskal-Wallis: H={h_stat:.2f}, p={p_val:.4f}",
        ha="center",
        fontsize=14,
        fontstyle="italic"
    )

    output_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(output_dir, "FigS13_Tcrit_Clustering.pdf")
    tiff_path = os.path.join(output_dir, "FigS13_Tcrit_Clustering.tiff")

    try:
        fig.savefig(pdf_path, dpi=300, facecolor="white")
        print(f"Saved PDF: {pdf_path}")
    except PermissionError:
        pdf_path_new = os.path.join(output_dir, "FigS13_Tcrit_Clustering_new.pdf")
        fig.savefig(pdf_path_new, dpi=300, facecolor="white")
        print(f"Original PDF is occupied. Saved PDF as: {pdf_path_new}")

    try:
        fig.savefig(tiff_path, dpi=600, facecolor="white")
        print(f"Saved TIFF: {tiff_path}")
    except PermissionError:
        tiff_path_new = os.path.join(output_dir, "FigS13_Tcrit_Clustering_new.tiff")
        fig.savefig(tiff_path_new, dpi=600, facecolor="white")
        print(f"Original TIFF is occupied. Saved TIFF as: {tiff_path_new}")

    plt.close(fig)

if __name__ == "__main__":
    plot_sa8()