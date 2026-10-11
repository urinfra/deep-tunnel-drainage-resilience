"""
Fig. 1 V2 | Resilience Failure Curves of Deep Tunnel Systems
=============================================================
Nature Cities submission figure. Uses nature-figure skill conventions.

Contract:
  Core conclusion: Deep tunnel resilience boundaries are shaped by functional type;
                   pollution-control systems operate at inherently lower resilience
                   and uniquely show T_crit shift under climate change.
  Archetype: quantitative grid (2x2)
  Backend: Python / matplotlib
  Output: fig1V2.pdf
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "Fig1_RFC_Curves.xlsx")
OUT_DIR = os.path.dirname(__file__)

# ── Nature-figure: mandatory font + SVG rules ──────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'

# ── Nature-figure: publication style ───────────────────────────────────────
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 13,
    'axes.spines.right': True,
    'axes.spines.top': True,
    'axes.linewidth': 1.2,
    'legend.frameon': False,
    'figure.dpi': 150,
})

# ── Nature-figure colour palette ───────────────────────────────────────────
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid":   "#767676",
    "neutral_dark":  "#4D4D4D",
    "neutral_black": "#272727",
    "teal":   "#42949E",
}

SCENARIO_COLORS = {
    "Historical":        "#272727",
    "SSP2-4.5 2050s":    "#0F4D92",
    "SSP5-8.5 2050s":    "#42949E",
    "SSP5-8.5 2080s":    "#B64342",
}
SCENARIO_LINESTYLES = {
    "Historical":        "-",
    "SSP2-4.5 2050s":    "--",
    "SSP5-8.5 2050s":    "-.",
    "SSP5-8.5 2080s":    ":",
}
SCENARIO_MARKERS = {
    "Historical":        "o",
    "SSP2-4.5 2050s":    "s",
    "SSP5-8.5 2050s":    "^",
    "SSP5-8.5 2080s":    "D",
}

# ── Load data ──────────────────────────────────────────────────────────────
rfc_df   = pd.read_excel(DATA_FILE, sheet_name="RFC_Curves_All")
tcrit_df = pd.read_excel(DATA_FILE, sheet_name="T_crit_Summary")
gz_obs   = pd.read_excel(DATA_FILE, sheet_name="Guangzhou_Observed")
params_df = pd.read_excel(DATA_FILE, sheet_name="City_Parameters")

CITIES    = ["Guangzhou", "Chicago", "Tokyo", "Kuala Lumpur"]
TUNNELS   = {r["City"]: r["Tunnel"] for _, r in params_df.iterrows()}
TYPES     = {r["City"]: r["Type"] for _, r in params_df.iterrows()}
DESIGN_STD = {r["City"]: r["Design_Std_yr"] for _, r in params_df.iterrows()}

# ── Helper: add_panel_label ────────────────────────────────────────────────
def add_panel_label(ax, label, x=-0.08, y=1.02, fontsize=14,
                    color="#272727", fontweight='bold'):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight=fontweight, color=color, ha='left', va='bottom')

# ── Build figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.subplots_adjust(hspace=0.35, wspace=0.25, left=0.08, right=0.96,
                    top=0.94, bottom=0.08)
axes = axes.flatten()

panel_labels = ['a', 'b', 'c', 'd']

for idx, city in enumerate(CITIES):
    ax = axes[idx]
    city_data = rfc_df[rfc_df["City"] == city]

    # ── Plot RFC curves ────────────────────────────────────────────────────
    for sc_name in ["Historical", "SSP2-4.5 2050s", "SSP5-8.5 2050s", "SSP5-8.5 2080s"]:
        sc_data = city_data[city_data["Scenario"] == sc_name].sort_values("Return_Period_yr")
        if len(sc_data) == 0:
            continue
        ax.semilogx(
            sc_data["Return_Period_yr"], sc_data["DTFRI"],
            linestyle=SCENARIO_LINESTYLES.get(sc_name, "-"),
            marker=SCENARIO_MARKERS.get(sc_name, "o"),
            color=SCENARIO_COLORS.get(sc_name, "#767676"),
            linewidth=2.5, markersize=9,
            markeredgecolor="white", markeredgewidth=0.8,
            label=sc_name, zorder=3
        )

    # ── Design standard (dashed grey vertical) ─────────────────────────────
    ds = DESIGN_STD[city]
    ax.axvline(x=ds, color="#767676", linestyle="--",
               alpha=0.5, linewidth=1.0, zorder=1)
    ax.text(ds * 0.92, 0.06, f"Design\n{ds}-yr",
            fontsize=11, color="#767676", va="bottom", ha="right")

    # ── DTFRI = 0.5 threshold line ─────────────────────────────────────────
    ax.axhline(y=0.5, color="#B64342", linestyle=":",
               alpha=0.35, linewidth=0.8, zorder=1)

    # ── T_crit annotations ─────────────────────────────────────────────────
    tc = tcrit_df[tcrit_df["City"] == city].copy()
    tc = tc[tc["Scenario"].isin(["Historical", "SSP5-8.5 2080s"])]

    label_pos = {
        "Guangzhou": {
            "Historical":       (0.42, 0.40),
            "SSP5-8.5 2080s":   (0.52, 0.30),
        },
        "Chicago": {
            "Historical":       (0.38, 0.60),
            "SSP5-8.5 2080s":   (0.50, 0.38),
        },
        "Tokyo": {
            "Historical":       (0.32, 0.46),
            "SSP5-8.5 2080s":   (0.36, 0.34),
        },
        "Kuala Lumpur": {
            "Historical":       (0.54, 0.46),
            "SSP5-8.5 2080s":   (0.44, 0.34),
        },
    }

    for _, tr in tc.iterrows():
        sc = tr["Scenario"]
        sc_color = SCENARIO_COLORS.get(sc, "#767676")
        tcrit_val = tr["T_crit_yr"]

        if sc == "Historical":
            label = rf"$T_{{\mathrm{{crit}}}}$ = {tcrit_val:.1f} yr (Historical)"
        else:
            label = rf"$T_{{\mathrm{{crit}}}}$ = {tcrit_val:.1f} yr (SSP5-8.5 2080s)"

        x_text, y_text = label_pos[city][sc]

        ax.annotate(
            label,
            xy=(tcrit_val, 0.5), xycoords="data",
            xytext=(x_text, y_text), textcoords="axes fraction",
            fontsize=10, fontweight="bold", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=sc_color, alpha=0.95, linewidth=0.8),
            arrowprops=dict(arrowstyle="-", color=sc_color, lw=0.7, alpha=0.7),
            clip_on=False, zorder=20
        )
        ax.plot(tcrit_val, 0.5, "o", color=sc_color, markersize=9,
                zorder=19, markeredgecolor="white", markeredgewidth=0.8)

    # ── Guangzhou observed data (red crosses) ──────────────────────────────
    if city == "Guangzhou" and len(gz_obs) > 0:
        # Subset visible on main panel (T >= 1 yr)
        gz_main = gz_obs[gz_obs["Estimated_Return_Period_yr"] >= 1.0]
        ax.scatter(
            gz_main["Estimated_Return_Period_yr"],
            gz_main["Estimated_DTFRI"],
            c="#B64342", marker="x", s=50, alpha=0.50,
            linewidths=1.2, zorder=10,
            label="Observed (2022–2025)"
        )

        max_idx = gz_obs["Storage_m3"].idxmax()
        max_rp = gz_obs["Estimated_Return_Period_yr"].iloc[max_idx]
        max_ufri = gz_obs["Estimated_DTFRI"].iloc[max_idx]
        max_storage = gz_obs["Storage_m3"].iloc[max_idx]
        ax.annotate(
            f"Max observed\n({max_storage:.0f} m³)",
            xy=(max_rp, max_ufri), fontsize=10,
            color="#B64342",
            xytext=(max_rp * 3.0, max_ufri + 0.10),
            arrowprops=dict(arrowstyle="->", color="#B64342",
                            alpha=0.5, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#B64342", alpha=0.85, linewidth=0.8),
            zorder=21
        )

        # ── Inset: zoomed linear-scale view of low-T observed data ──────────
        inset_ax = ax.inset_axes([0.12, 0.12, 0.38, 0.38])
        # Tiny dots reveal density; 307 points at T~0.01 overlap severely
        inset_ax.scatter(
            gz_obs["Estimated_Return_Period_yr"],
            gz_obs["Estimated_DTFRI"],
            c="#B64342", marker="x", s=14, alpha=0.35,
            linewidths=0.5, zorder=10
        )
        inset_ax.set_xlim(0.0, 3.2)
        inset_ax.set_ylim(0.48, 1.02)
        inset_ax.set_xlabel("Return period (yr)", fontsize=9, labelpad=1)
        inset_ax.set_ylabel("DTFRI", fontsize=9, labelpad=1)
        inset_ax.tick_params(labelsize=8)
        inset_ax.grid(True, alpha=0.12, linewidth=0.4)
        for spine in inset_ax.spines.values():
            spine.set_color("#767676")
            spine.set_linewidth(0.6)

    # ── Axis limits & labels ───────────────────────────────────────────────
    ax.set_xlim(1.0, 260)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("DTFRI")
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_title(
        f"{city} — {TUNNELS[city]}\n({TYPES[city]}, design: {DESIGN_STD[city]}-year)",
        fontweight="bold", fontsize=13, pad=8
    )
    ax.grid(True, alpha=0.08, linewidth=0.5)

    legend_loc = "upper right" if city == "Guangzhou" else "lower left"
    ax.legend(loc=legend_loc, fontsize=10, handlelength=1.8,
              handletextpad=0.6, borderpad=0.3, labelspacing=0.25)

    add_panel_label(ax, panel_labels[idx])

# ── Finalise & save ────────────────────────────────────────────────────────
pdf_path = os.path.join(OUT_DIR, "Fig1_RFC_Curves.pdf")
fig.savefig(pdf_path, dpi=300, facecolor="white", bbox_inches="tight",
            pad_inches=0.1)
plt.close(fig)

print(f"Saved: {pdf_path}")
print(f"Size: {os.path.getsize(pdf_path):,} bytes")
