"""
Fig. S1 | Model calibration and validation for all four cities
===============================================================
Nature Cities submission — Supplementary Information figure.

Contract:
  Core conclusion: The parameterized resilience modelling framework is calibrated
                  against 4 years of Guangzhou operational data and validated
                  against published performance metrics for Chicago, Tokyo, and
                  Kuala Lumpur.
  Archetype: quantitative grid (2 × 2)
  Backend: Python / matplotlib
  Output: FigS1_Calibration.pdf, FigS1_Calibration.tiff
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(__file__)
DATA_SWMM = os.path.join(
    os.path.dirname(os.path.dirname(SCRIPT_DIR)),
    "data", "swmm"
)
GZ_CALIB = os.path.join(DATA_SWMM, "calibrated_params_gz.json")
CITY_PARAMS = os.path.join(DATA_SWMM, "verified_city_params.json")

# ── Nature-figure: font + style ────────────────────────────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.linewidth": 1.0,
    "legend.frameon": False,
    "figure.dpi": 150,
})

# ── Nature colour palette ──────────────────────────────────────────────────
PAL = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "orange":         "#E6761E",
    "red_strong":     "#B64342",
    "green":          "#3A7A5C",
    "grey":           "#7F7F7F",
}


def load_data():
    """Load calibration and validation data."""
    with open(GZ_CALIB, "r", encoding="utf-8") as f:
        gz = json.load(f)
    with open(CITY_PARAMS, "r", encoding="utf-8") as f:
        cities = json.load(f)
    return gz, cities


INPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(SCRIPT_DIR)),
    "Inputs"
)


def load_gz_events():
    """Load positive peak-storage events (2022–2025) from the raw monitoring
    workbooks: rows with a valid Excel-serial date and a positive storage value."""
    events = []
    for yr in ["2022", "2023", "2024", "2025"]:
        path = os.path.join(INPUT_DIR, "深隧数据%s.xlsx" % yr)
        xl = pd.ExcelFile(path)
        sheet = [s for s in xl.sheet_names if "峰值蓄水量" in s][0]
        df = pd.read_excel(path, sheet_name=sheet)
        for _, row in df.iterrows():
            try:
                v = float(row.iloc[1])
                if pd.isna(v) or v <= 0:
                    continue
                s = float(row.iloc[0])
                if 40000 <= s <= 50000:
                    events.append(v)
            except Exception:
                continue
    return np.array(events)


def panel_a_gz_calibration(ax, gz):
    """Guangzhou: observed peak-storage distribution against design capacity."""
    events = load_gz_events()
    n_ev = len(events)
    n_design = gz["tunnel_parameters"]["n_design"]
    n_calib = gz["tunnel_parameters"]["n_calibrated"]
    design = gz["tunnel_parameters"]["design_storage_m3"]

    # Raw event points (editorial E.6: show the individual data points)
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.8, n_ev)
    ax.scatter(jitter, events, s=16, alpha=0.35, color=PAL["blue_main"],
               edgecolors="none", label=f"Observed events (n = {n_ev})", zorder=3)
    ax.set_xlim(-8, 10)

    # Observed percentiles
    for p, lab in [(50, "P50"), (95, "P95"), (99, "P99")]:
        v = np.percentile(events, p)
        ax.axhline(v, color=PAL["grey"], linestyle="--", linewidth=1.0,
                   alpha=0.7, zorder=2)
        ax.text(9.2, v, f"{lab} = {v/1000:.1f} × 10³", va="center",
                fontsize=12, color=PAL["grey"])

    # Design-storage reference
    ax.axhline(design, color=PAL["red_strong"], linestyle=":", linewidth=1.8,
               zorder=4)
    ax.text(9.2, design, "Design storage\n100 × 10³ m³", va="center",
            fontsize=12, color=PAL["red_strong"], fontweight="bold")
    ax.annotate(f"Max event {events.max()/1000:.0f} × 10³ m³\n"
                f"({events.max()/design*100:.0f}% of design)",
                xy=(0, events.max()), xytext=(-7.5, events.max() * 1.02),
                fontsize=12, color=PAL["red_strong"], ha="left", va="bottom")

    ax.set_xticks([])
    ax.set_ylabel("Peak storage (m³)")
    ax.set_title("a  Guangzhou: Observed Peak-Storage Distribution (2022–2025)",
                 loc="left", fontweight="bold", fontsize=17)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(axis="y", alpha=0.15)
    ax.set_ylim(0, design * 1.22)
    textstr = (
        f"Manning’s n\n"
        f"  Design: {n_design:.3f}\n"
        f"  Calibrated: {n_calib:.3f}\n"
        f"  Increase: +{(n_calib / n_design - 1) * 100:.0f}%"
    )
    ax.text(0.03, 0.05, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment="bottom", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="grey", alpha=0.85))


def panel_b_chicago(ax, cities):
    """Chicago: CSO capture validation."""
    chi = cities["Chicago"]["validation"]
    labels = ["Pre-TARP\n(~1980s)", "Post-TARP\n(current)"]
    values = [chi["cso_events_historical"], chi["cso_events_current"]]
    colors = [PAL["orange"], PAL["green"]]
    # Dot plot (editorial E.6: individual data points instead of summary bars)
    for xv, val, c in zip([0, 1], values, colors):
        ax.plot(xv, val, "o", markersize=16, color=c,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    for xv, val in zip([0, 1], values):
        ax.text(xv, val + 2.5, f"{int(val)}", ha="center", va="bottom",
                fontsize=15, fontweight="bold", color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) * 1.35)
    ax.set_ylabel("Annual CSO Events")
    ax.set_title("b  Chicago TARP: CSO Capture Validation",
                 loc="left", fontweight="bold", fontsize=17)
    ax.grid(axis="y", alpha=0.15)
    ax.annotate(f"↓ {chi['cso_reduction_pct']:.0f}%",
                xy=(0.5, 0.5), xycoords="axes fraction",
                ha="center", va="center", fontsize=18,
                fontweight="bold", color=PAL["red_strong"],
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=PAL["red_strong"], alpha=0.9))
    ax.text(0.97, 0.95,
            f"Cumulative capture\n(1985–2023):\n"
            f"{chi['cumulative_capture_m3'] / 1e9:.1f} billion m³",
            transform=ax.transAxes, fontsize=12,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="grey", alpha=0.85))


def panel_c_tokyo(ax, cities):
    """Tokyo: flood damage reduction validation."""
    tokyo = cities["Tokyo"]["validation"]
    hagibis = tokyo["record_event_hagibis_2019"]
    labels = ["Without\nMAOUDC", "With\nMAOUDC"]
    values = [hagibis["homes_flooded_without"], hagibis["homes_flooded"]]
    colors = [PAL["red_strong"], PAL["blue_main"]]
    # Dot plot (editorial E.6: individual data points instead of summary bars)
    for xv, val, c in zip([0, 1], values, colors):
        ax.plot(xv, val, "o", markersize=16, color=c,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    for xv, val in zip([0, 1], values):
        ax.text(xv, val + max(values) * 0.015, f"{val:,}", ha="center",
                va="bottom", fontsize=15, fontweight="bold", color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) * 1.3)
    ax.set_ylabel("Homes Flooded (Typhoon Hagibis 2019)")
    ax.set_title("c  Tokyo MAOUDC: Flood Damage Validation",
                 loc="left", fontweight="bold", fontsize=17)
    ax.grid(axis="y", alpha=0.15)
    ax.annotate(f"↓ {hagibis['reduction_pct']:.1f}%",
                xy=(0.5, 0.55), xycoords="axes fraction",
                ha="center", va="center", fontsize=18,
                fontweight="bold", color=PAL["green"],
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=PAL["green"], alpha=0.9))
    ax.text(0.97, 0.95,
            f"Annual activations:\n"
            f"{tokyo['annual_activation_avg']} (range {tokyo['annual_activation_range']})\n"
            f"Cumulative damage\nprevented: ¥"
            f"{tokyo['cumulative_damage_prevented_jpy'] / 1e9:.1f}B",
            transform=ax.transAxes, fontsize=12,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="grey", alpha=0.85))


def panel_d_kl(ax, cities):
    """Kuala Lumpur: mode operation validation."""
    kl = cities["KualaLumpur"]["validation"]
    modes = ["Mode I: Flood only (<70)",
             "Mode II: Combined (70–150)",
             "Mode III: Full diversion (>150)"]
    thresholds = [70, 80, 150]
    mode_colors = [PAL["blue_main"], PAL["blue_secondary"], PAL["red_strong"]]
    y_positions = [2, 1, 0]
    for y, mode, thresh, color in zip(y_positions, modes, thresholds, mode_colors):
        ax.barh(y, thresh, height=0.55, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.8)
        ax.text(thresh + 3, y, mode, va="center", fontsize=13,
                fontweight="bold", color=color)
    ax.set_xlim(0, 260)
    ax.set_xlabel("Discharge Threshold (m³/s)")
    ax.set_yticks([])
    ax.set_title("d  Kuala Lumpur SMART: Mode Operation Validation",
                 loc="left", fontweight="bold", fontsize=17)
    ax.grid(axis="x", alpha=0.15)
    ax.text(0.97, 0.92,
            f"Mode III frequency:\n1–2 events/year\n"
            f"Reopening: ≤ {kl['mode3_reopen_hours']} h\n"
            f"Traffic: {cities['KualaLumpur']['daily_traffic']:,} veh/day",
            transform=ax.transAxes, fontsize=12,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="grey", alpha=0.85))


def main():
    gz, cities = load_data()
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.subplots_adjust(
        left=0.07, right=0.97, bottom=0.08, top=0.94,
        wspace=0.28, hspace=0.32
    )
    panel_a_gz_calibration(axes[0, 0], gz)
    panel_b_chicago(axes[0, 1], cities)
    panel_c_tokyo(axes[1, 0], cities)
    panel_d_kl(axes[1, 1], cities)

    pdf_path = os.path.join(SCRIPT_DIR, "FigS1_Calibration.pdf")
    tiff_path = os.path.join(SCRIPT_DIR, "FigS1_Calibration.tiff")
    fig.savefig(pdf_path, dpi=300, facecolor="white")
    fig.savefig(tiff_path, dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {tiff_path}")


if __name__ == "__main__":
    main()
