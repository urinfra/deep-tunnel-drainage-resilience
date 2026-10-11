"""
Fig. 2 | The Deep Tunnel Safety Paradox (Nature Cities)
=========================================================
Nature-figure conventions applied to safety paradox land-use trajectories.
Data: Fig3_Safety_Paradox.xlsx
Output: Fig3_Safety_Paradox.pdf
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "Fig2_Safety_Paradox.xlsx")
OUT_DIR = os.path.dirname(__file__)

# ── Nature-figure: mandatory font + SVG rules ──────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'

# ── Publication style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.spines.right': True,
    'axes.spines.top': True,
    'axes.linewidth': 1.0,
    'legend.frameon': False,
    'figure.dpi': 150,
})

# ── Colour palette ─────────────────────────────────────────────────────────
CITY_COLORS = {
    "Guangzhou":    "#B64342",
    "Chicago":      "#0F4D92",
    "Tokyo":        "#42949E",
    "Kuala Lumpur": "#1B7A3D",
}
CONTROL_COLOR = "#767676"
TUNNEL_LINE_COLOR = "#B64342"

# ── City metadata ──────────────────────────────────────────────────────────
TUNNELS = {
    "Guangzhou": "Donghaochong",
    "Chicago": "TARP Phase I",
    "Tokyo": "MAOUDC",
    "Kuala Lumpur": "SMART",
}
TYPES = {
    "Guangzhou": "Multi-function, 50-yr design",
    "Chicago": "Pollution control, 10-yr design",
    "Tokyo": "Flood control, 50-yr design",
    "Kuala Lumpur": "Multi-function, 100-yr design",
}

# ── Helper ─────────────────────────────────────────────────────────────────
def add_panel_label(ax, label, x=-0.07, y=1.03, fontsize=14,
                    color="#272727", fontweight='bold'):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight=fontweight, color=color, ha='left', va='bottom')

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_FILE, sheet_name="LandUse_Trajectories")
summary = pd.read_excel(DATA_FILE, sheet_name="Summary")
loss_ratios = {r["City"]: r["Loss_Ratio_T200"] for _, r in summary.iterrows()}
tunnel_years = {r["City"]: r["Tunnel_Year"] for _, r in summary.iterrows()}

CITIES = ["Guangzhou", "Chicago", "Tokyo", "Kuala Lumpur"]

# ── Common y-axis range across all four cities (Reviewer 2, R2.6) ──────────
GLOBAL_YMIN = min(df["BuiltUp_Density"].min(), 0)
GLOBAL_YMAX = df["BuiltUp_Density"].max()
GLOBAL_YPAD = 0.12 * (GLOBAL_YMAX - GLOBAL_YMIN)

# ── Build figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.subplots_adjust(hspace=0.32, wspace=0.22, left=0.08, right=0.97,
                    top=0.94, bottom=0.08)
axes = axes.flatten()
panel_labels = ['a', 'b', 'c', 'd']

for idx, city in enumerate(CITIES):
    ax = axes[idx]
    cd = df[df["City"] == city].copy()
    tunnel_yr = tunnel_years[city]
    loss_ratio = loss_ratios[city]
    color = CITY_COLORS[city]

    protected = cd[cd["Zone"] == "Protected"].sort_values("Year")
    control = cd[cd["Zone"] == "Control"].sort_values("Year")

    # ── Development rate differential ──────────────────────────────────────
    pre_prot = protected[protected["Year"] < tunnel_yr]
    post_prot = protected[protected["Year"] >= tunnel_yr]
    pre_ctrl = control[control["Year"] < tunnel_yr]
    post_ctrl = control[control["Year"] >= tunnel_yr]

    # Pre-tunnel rates (require ≥ 2 data points before tunnel commissioning)
    if len(pre_prot) >= 2:
        rate_prot_pre = (pre_prot["BuiltUp_Density"].iloc[-1] -
                         pre_prot["BuiltUp_Density"].iloc[0]) / \
                        (pre_prot["Year"].iloc[-1] - pre_prot["Year"].iloc[0])
    else:
        rate_prot_pre = None

    if len(pre_ctrl) >= 2:
        rate_ctrl_pre = (pre_ctrl["BuiltUp_Density"].iloc[-1] -
                         pre_ctrl["BuiltUp_Density"].iloc[0]) / \
                        (pre_ctrl["Year"].iloc[-1] - pre_ctrl["Year"].iloc[0])
    else:
        rate_ctrl_pre = None

    # Post-tunnel rates (require ≥ 2 data points after tunnel commissioning)
    if len(post_prot) >= 2:
        rate_prot_post = (post_prot["BuiltUp_Density"].iloc[-1] -
                          post_prot["BuiltUp_Density"].iloc[0]) / \
                         (post_prot["Year"].iloc[-1] - post_prot["Year"].iloc[0])
    else:
        rate_prot_post = None

    if len(post_ctrl) >= 2:
        rate_ctrl_post = (post_ctrl["BuiltUp_Density"].iloc[-1] -
                          post_ctrl["BuiltUp_Density"].iloc[0]) / \
                         (post_ctrl["Year"].iloc[-1] - post_ctrl["Year"].iloc[0])
    else:
        rate_ctrl_post = None

    if rate_prot_post is not None and rate_ctrl_post is not None:
        diff_post = (rate_prot_post - rate_ctrl_post) * 100
    else:
        diff_post = None

    # ── Plot trajectories ──────────────────────────────────────────────────
    ax.plot(protected["Year"], protected["BuiltUp_Density"],
            "o-", color=color, linewidth=2.4, markersize=7,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Protected zone", zorder=3)

    ax.plot(control["Year"], control["BuiltUp_Density"],
            "s--", color=CONTROL_COLOR, linewidth=2.0, markersize=6,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Control zone", zorder=2)

    # ── Tunnel commissioning line ──────────────────────────────────────────
    ax.axvline(x=tunnel_yr, color=TUNNEL_LINE_COLOR, linestyle=":",
               linewidth=1.6, alpha=0.55, zorder=1)

    y_max = cd["BuiltUp_Density"].max()  # city max, used only for annotation placement
    ax.set_ylim(GLOBAL_YMIN - 0.02, GLOBAL_YMAX + GLOBAL_YPAD)

    # Per-city annotation offsets (1 cm ≈ 28.3 points)
    anno_offsets = {
        "Guangzhou":    (-111, 9),    # left 2cm
        "Chicago":      (-27, -5),    # right 1cm, down 0.5cm
        "Tokyo":        (-55, -14),   # down 0.8cm
        "Kuala Lumpur": (-55, 9),     # unchanged
    }
    dx, dy = anno_offsets.get(city, (-55, 9))

    ax.annotate(
        f"Tunnel commissioned\n({int(tunnel_yr)})",
        xy=(tunnel_yr, y_max * 0.88), xycoords="data",
        xytext=(dx, dy), textcoords="offset points",
        fontsize=10, color=TUNNEL_LINE_COLOR, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor=TUNNEL_LINE_COLOR, alpha=0.85),
        zorder=10
    )

    # ── Inset: key metrics (top-left corner) ───────────────────────────────
    if diff_post is not None:
        diff_str = f"{diff_post:+.2f}"
    else:
        diff_str = "N/A"
    metrics_text = (
        f"Dev. rate diff. (post): {diff_str} pp/yr\n"
        f"Loss ratio (T=200 yr): {loss_ratio:.2f}"
    )
    ax.text(0.03, 0.94, metrics_text, transform=ax.transAxes,
            fontsize=9, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#CFCECE", alpha=0.92, linewidth=0.6),
            zorder=12)

    # ── Axis labels ────────────────────────────────────────────────────────
    ax.set_xlabel("Year")
    ax.set_ylabel("Built-up area density")
    ax.set_title(f"{city} — {TUNNELS[city]}\n{TYPES[city]}",
                 fontweight="bold", fontsize=11, pad=6)
    ax.grid(True, alpha=0.10, linewidth=0.5)
    ax.legend(loc="lower right", fontsize=10, handlelength=1.5,
              handletextpad=0.5, borderpad=0.3, labelspacing=0.3)

    add_panel_label(ax, panel_labels[idx], fontsize=13)

# ── Finalise & save ────────────────────────────────────────────────────────
pdf_path = os.path.join(OUT_DIR, "Fig2_Safety_Paradox.pdf")
fig.savefig(pdf_path, dpi=300, facecolor="white", bbox_inches="tight",
            pad_inches=0.1)
plt.close(fig)
print(f"Saved: {pdf_path}")
print(f"Size: {os.path.getsize(pdf_path):,} bytes")
