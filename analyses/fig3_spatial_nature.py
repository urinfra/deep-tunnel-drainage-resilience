"""
Fig. 3 | Spatial Distribution of Deep Tunnel Flood Protection (Nature Cities)
==============================================================================
Nature-figure style spatial protection intensity maps across four cities.
Data: Fig2_Risk_Transfer_Map.xlsx
Output: Fig3_Spatial_Distribution_Protection.pdf
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE, "Submission", "Figure Data & Codes",
                         "Fig2_Risk_Transfer_Map.xlsx")
OUT_DIR = os.path.join(BASE, "Submission", "Figure Data & Codes")

# ── Nature-figure mandatory rules ──────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'axes.spines.right': True, 'axes.spines.top': True,
    'axes.linewidth': 0.8, 'legend.frameon': False, 'figure.dpi': 150,
})

# ── City metadata ──────────────────────────────────────────────────────────
CITY_META = {
    "Guangzhou": {"title": "Guangzhou (Donghaochong)",
                  "design": "50-year", "n_val": "0.025",
                  "vmax": 0.96, "shaft_count": 5},
    "Chicago": {"title": "Chicago (TARP)",
                "design": "10-year", "n_val": "0.025",
                "vmax": 0.24, "shaft_count": 10},
    "Tokyo": {"title": "Tokyo (Outer Discharge Channel)",
              "design": "50-year", "n_val": "0.018",
              "vmax": 0.96, "shaft_count": 8},
    "Kuala Lumpur": {"title": "Kuala Lumpur (SMART)",
                     "design": "100-year", "n_val": "0.025",
                     "vmax": 0.96, "shaft_count": 10},
}

# ── Nature sequential colormap (blue-to-red via muted tones) ───────────────
nature_protection = LinearSegmentedColormap.from_list("nature_protect", [
    "#E8EEF4", "#B4C6DC", "#6B93BE", "#3B6DA6",
    "#C4827E", "#B64342",
], N=256)

# ── Helpers ────────────────────────────────────────────────────────────────
def add_panel_label(ax, label, x=-0.05, y=1.03, fontsize=13, color="#272727",
                    fontweight='bold'):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight=fontweight, color=color, ha='left', va='bottom')

def interpolate_surface(cd, x_min, x_max, y_min, y_max):
    xi = np.linspace(x_min, x_max, 200)
    yi = np.linspace(y_min, y_max, 200)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    points = (cd["X_km"], cd["Y_km"])
    values = cd["Protection_Intensity"]
    try:
        zi = griddata(points, values, (xi_grid, yi_grid),
                      method="cubic", fill_value=np.nan)
    except Exception:
        zi = griddata(points, values, (xi_grid, yi_grid),
                      method="linear", fill_value=np.nan)
    zi_nearest = griddata(points, values, (xi_grid, yi_grid), method="nearest")
    zi = np.where(np.isnan(zi), zi_nearest, zi)
    return xi_grid, yi_grid, np.clip(zi, 0, 1)

def estimate_tunnel_centerline_y(cd):
    high = cd[cd["Protection_Intensity"] >= cd["Protection_Intensity"].quantile(0.85)]
    if len(high) > 0 and high["Protection_Intensity"].sum() > 0:
        return np.average(high["Y_km"], weights=high["Protection_Intensity"])
    return cd["Y_km"].median()

def make_shaft_positions(x_min, x_max, center_y, shaft_count):
    shaft_x = np.linspace(x_min + 0.12*(x_max-x_min),
                          x_max - 0.10*(x_max-x_min), shaft_count)
    return shaft_x, np.full_like(shaft_x, center_y)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_FILE, sheet_name="Protection_Intensity")
CITIES = ["Guangzhou", "Chicago", "Tokyo", "Kuala Lumpur"]

# ── Build figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
fig.subplots_adjust(hspace=0.30, wspace=0.28, top=0.90, bottom=0.08,
                    left=0.07, right=0.94)
axes = axes.flatten()
panel_labels = ['a', 'b', 'c', 'd']

for idx, city in enumerate(CITIES):
    ax = axes[idx]
    cd = df[df["City"] == city].dropna(subset=["X_km", "Y_km", "Protection_Intensity"])
    cd["Protection_Intensity"] = cd["Protection_Intensity"].clip(0, 1)
    meta = CITY_META[city]

    x_min, x_max = cd["X_km"].min(), cd["X_km"].max()
    y_min, y_max = cd["Y_km"].min(), cd["Y_km"].max()
    x_pad = 0.02*(x_max-x_min)
    y_pad = 0.02*(y_max-y_min)

    xi_grid, yi_grid, zi = interpolate_surface(cd, x_min, x_max, y_min, y_max)
    vmax = meta["vmax"]
    levels = np.linspace(0, vmax, 14)

    im = ax.contourf(xi_grid, yi_grid, zi, levels=levels,
                     cmap=nature_protection, vmin=0, vmax=vmax,
                     extend="max", alpha=0.95)

    # Tunnel centreline (dashed)
    center_y = estimate_tunnel_centerline_y(cd)
    ax.axhline(y=center_y, color="#272727", linestyle="--",
               linewidth=1.0, alpha=0.4, zorder=5)

    # Intake shaft markers (triangles)
    shaft_x, shaft_y = make_shaft_positions(x_min, x_max, center_y,
                                            meta["shaft_count"])
    ax.scatter(shaft_x, shaft_y, marker="^", c="#B64342", s=70,
               edgecolors="white", linewidth=0.6, zorder=10)

    ax.set_xlim(x_min-x_pad, x_max+x_pad)
    ax.set_ylim(y_min-y_pad, y_max+y_pad)
    ax.set_xlabel("Distance along tunnel (km)")
    ax.set_ylabel("Lateral distance (km)")
    ax.set_title(f"{meta['title']}\nDesign: {meta['design']}, n = {meta['n_val']}",
                 fontsize=11, fontweight="bold", pad=4)

    cbar = fig.colorbar(im, ax=ax, shrink=0.80, pad=0.04)
    cbar.set_label("Flood protection intensity", fontsize=10)
    cbar.set_ticks(np.linspace(0, vmax, 7))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(labelsize=9)

    add_panel_label(ax, panel_labels[idx])

# ── Save ───────────────────────────────────────────────────────────────────
pdf_path = os.path.join(OUT_DIR, "Fig3_Spatial_Distribution_Protection.pdf")
fig.savefig(pdf_path, dpi=600, facecolor="white", bbox_inches="tight",
            pad_inches=0.1)
plt.close(fig)
print(f"Saved: {pdf_path}")
print(f"Size: {os.path.getsize(pdf_path):,} bytes")
