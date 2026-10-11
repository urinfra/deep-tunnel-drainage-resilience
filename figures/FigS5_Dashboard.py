"""
Extended Data Fig. S4 | 14-Case Deep Tunnel Resilience Dashboard
=================================================================

独立绘图脚本 -- 从同目录下的 FigS5_Dashboard.xlsx 读取数据

用法:
    python FigS5_Dashboard.py

输出:
    FigS5_Dashboard.pdf
    FigS5_Dashboard.tiff
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATA_FILE = os.path.join(os.path.dirname(__file__), "FigS5_Dashboard.xlsx")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Nature-style type colours
TYPE_COLORS = {
    "pollution_control": "#3498DB",
    "flood_control": "#27AE60",
    "multi_function": "#E74C3C",
}
TYPE_LABELS = {
    "pollution_control": "Pollution Control",
    "flood_control": "Flood Control",
    "multi_function": "Multi-function",
}
TYPE_MARKERS = {
    "pollution_control": "o",
    "flood_control": "s",
    "multi_function": "D",
}


def plot_ed_s4():
    """4-panel resilience dashboard for all 14 cases."""
    df = pd.read_excel(DATA_FILE, sheet_name="15Case_RFC_Data")

    # Extract per-case summary (Historical scenario)
    hist = df[df["Scenario"] == "Historical"].copy()
    ssp = df[df["Scenario"].isin(["SSP5-8.5_2080s", "SSP5-8.5 2080s"])].copy()

    summary = hist.groupby(["Case", "Type"]).agg(
        Design_Std_yr=("Design_Std_yr", "first"),
        T_crit_yr=("T_crit_yr", "first"),
        Storage_m3=("Storage_m3", "first"),
        DTFRI_T50_hist=("DTFRI", lambda x: x.iloc[5] if len(x) > 5 else np.nan),
    ).reset_index()

    # Get SSP5-8.5 DTFRI at T=50 for delta
    ssp_t50 = ssp.groupby("Case")["DTFRI"].apply(
        lambda x: x.iloc[5] if len(x) > 5 else np.nan
    ).reset_index()
    ssp_t50.columns = ["Case", "DTFRI_T50_ssp"]
    summary = summary.merge(ssp_t50, on="Case", how="left")
    summary["Delta_DTFRI_pct"] = (
        (summary["DTFRI_T50_ssp"] - summary["DTFRI_T50_hist"])
        / summary["DTFRI_T50_hist"].clip(lower=0.01)
        * 100
    )

    # Sort by Type then T_crit
    type_order = {"pollution_control": 0, "flood_control": 1, "multi_function": 2}
    summary["_type_sort"] = summary["Type"].map(type_order)
    summary = summary.sort_values(["_type_sort", "T_crit_yr"]).reset_index(drop=True)
    case_names = summary["Case"].tolist()
    y_positions = range(len(case_names))

    colors = [TYPE_COLORS.get(t, "#999999") for t in summary["Type"]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.subplots_adjust(
        left=0.10, right=0.97, bottom=0.07, top=0.93,
        hspace=0.35, wspace=0.30,
    )

    # ---- Panel a: Design Standard ----
    ax = axes[0, 0]
    ds_vals = summary["Design_Std_yr"].values
    # Dot plot (editorial E.6: individual data points instead of summary bars)
    ax.scatter(ds_vals, y_positions, c=colors, s=90, edgecolors="white",
               linewidth=0.8, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(case_names, fontsize=10)
    ax.set_xlabel("Design Return Period (years)")
    ax.set_title("a  Design Standard", fontweight="bold", loc="left", fontsize=15)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=12)

    # ---- Panel b: T_crit ----
    ax = axes[0, 1]
    tcrit_vals = summary["T_crit_yr"].values
    # Dot plot (editorial E.6)
    ax.scatter(tcrit_vals, y_positions, c=colors, s=90, edgecolors="white",
               linewidth=0.8, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(case_names, fontsize=10)
    ax.set_xlabel("Critical Return Period T_crit (years)")
    ax.set_title("b  Critical Threshold T_crit", fontweight="bold", loc="left", fontsize=15)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=12)

    # ---- Panel c: Storage vs Design Standard ----
    ax = axes[1, 0]
    for i, (_, r) in enumerate(summary.iterrows()):
        typ = r["Type"]
        ax.scatter(
            r["Design_Std_yr"],
            r["Storage_m3"] / 1e4,
            c=TYPE_COLORS.get(typ, "#999"),
            marker=TYPE_MARKERS.get(typ, "o"),
            s=120,
            edgecolors="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            r["Case"],
            (r["Design_Std_yr"], r["Storage_m3"] / 1e4),
            fontsize=7.5,
            ha="center", va="bottom",
            color=TYPE_COLORS.get(typ, "#333"),
            alpha=0.85,
            zorder=10,
            xytext=(0, 4),
            textcoords="offset points",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Design Return Period (years)")
    ax.set_ylabel("Storage Volume (10$^4$ m$^3$)  [log scale]")
    ax.set_title("c  Storage Volume vs. Design Standard", fontweight="bold", loc="left", fontsize=15)
    ax.set_xlim(-5, 120)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.15)

    # Legend for types
    legend_handles = []
    for typ_key, cfg_label in TYPE_LABELS.items():
        legend_handles.append(
            Line2D(
                [0], [0],
                marker=TYPE_MARKERS[typ_key],
                color="w",
                markerfacecolor=TYPE_COLORS[typ_key],
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=cfg_label,
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=10,
        frameon=True,
        framealpha=0.85,
        edgecolor="lightgray",
    )

    # ---- Panel d: Climate Vulnerability (Delta DTFRI %) ----
    ax = axes[1, 1]
    delta_vals = summary["Delta_DTFRI_pct"].values
    bar_colors_d = ["#E74C3C" if d < -5 else "#F39C12" if d < 0 else "#3498DB"
                     for d in delta_vals]
    # Dot plot (editorial E.6)
    ax.scatter(delta_vals, y_positions, c=bar_colors_d, s=90, edgecolors="white",
               linewidth=0.8, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(case_names, fontsize=10)
    ax.set_xlabel("DTFRI Change at T=50 under SSP5-8.5 2080s (%)")
    ax.set_title("d  Climate Vulnerability", fontweight="bold", loc="left", fontsize=15)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.4, linewidth=1)
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=12)

    fig.suptitle(
        "Extended Data Fig. S4 | 14-Case Deep Tunnel Resilience Dashboard",
        fontweight="bold",
        fontsize=17,
    )

    # Save outputs
    output_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(output_dir, "FigS5_Dashboard.pdf")
    tiff_path = os.path.join(output_dir, "FigS5_Dashboard.tiff")

    try:
        fig.savefig(pdf_path, dpi=300, facecolor="white")
        print(f"Saved PDF: {pdf_path}")
    except PermissionError:
        alt = os.path.join(output_dir, "FigS5_Dashboard_new.pdf")
        fig.savefig(alt, dpi=300, facecolor="white")
        print(f"Original PDF occupied. Saved as: {alt}")

    try:
        fig.savefig(tiff_path, dpi=600, facecolor="white")
        print(f"Saved TIFF: {tiff_path}")
    except PermissionError:
        alt = os.path.join(output_dir, "FigS5_Dashboard_new.tiff")
        fig.savefig(alt, dpi=600, facecolor="white")
        print(f"Original TIFF occupied. Saved as: {alt}")

    plt.close(fig)


if __name__ == "__main__":
    plot_ed_s4()
