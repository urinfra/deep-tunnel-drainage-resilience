"""
Fig. S15 | SA10 — Subdaily rainfall scaling sensitivity
========================================================
Response to Reviewer 1 (R1.4f): bound the effect of subdaily rainfall peaks
that are not resolved by the daily-resolution rainfall inputs.

The framework's flood-volume proxy (the effective return period, see SA2
climate-factor machinery) is amplified by factors delta in [1.0, 2.0]. This
range spans published subdaily-to-daily intensification ratios: hourly extreme
intensity increases ~134% faster than the daily one in the Pearl River Delta
(Deng et al. 2024, J. Hydrol. 633, 131025); urbanization raises hourly extreme
intensity by ~26% versus ~13% for daily (~2x, same source); and hourly rainfall
intensities are projected to increase a further 10-17% by the 2080s in tropical
cities (Blagojevic et al. 2026, Earth's Future).

Engine: calc_ufri / tunnel_flood_reduction / find_Tcrit — identical to
generate_all.py and update_figures.py.
Output: FigS15_Subdaily_Scaling.pdf/.tiff + FigS15_Subdaily_Scaling.xlsx
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CITY_PARAMS = {
    "Guangzhou":    {"type": "Multi-function",   "design_std_yr": 50,  "roughness_n": 0.025},
    "Chicago":      {"type": "Pollution control", "design_std_yr": 10,  "roughness_n": 0.025},
    "Tokyo":        {"type": "Flood control",    "design_std_yr": 50,  "roughness_n": 0.018},
    "Kuala Lumpur": {"type": "Multi-function",   "design_std_yr": 100, "roughness_n": 0.025},
}
CITY_COLORS = {
    "Guangzhou": "#B64342",
    "Chicago": "#0F4D92",
    "Tokyo": "#42949E",
    "Kuala Lumpur": "#1B7A3D",
}
RETURN_PERIODS = [2, 5, 10, 20, 30, 50, 75, 100, 150, 200]
RETURN_PERIODS_FINE = np.arange(1, 251, 0.5)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.spines.right": True, "axes.spines.top": True,
    "axes.linewidth": 1.0, "legend.frameon": False, "figure.dpi": 150,
})


def tunnel_flood_reduction(rp, cp, climate_factor=1.0):
    ds = cp["design_std_yr"]
    n_ratio = cp.get("roughness_n", 0.015) / 0.015
    effective_ds = ds / (n_ratio ** 0.6)
    rp_eff = rp * climate_factor
    if cp["type"] == "Pollution control":
        eta = 1.0 / (1.0 + np.exp((rp_eff - effective_ds * 0.8) / (effective_ds * 0.3)))
    elif cp["type"] == "Flood control":
        eta = 1.0 / (1.0 + np.exp((rp_eff - effective_ds) / (effective_ds * 0.2)))
    else:
        eta = 1.0 / (1.0 + np.exp((rp_eff - effective_ds * 1.1) / (effective_ds * 0.4)))
    return max(0.0, min(1.0, eta))


def calc_ufri(rp, cp, climate_factor=1.0, infrastructure="deep_tunnel"):
    if infrastructure == "no_tunnel":
        return 0.0
    eta = tunnel_flood_reduction(rp, cp, climate_factor)
    ds = cp["design_std_yr"]
    if infrastructure == "deep_tunnel":
        robustness = eta
        rapidity_base = 0.85 if rp <= ds else 0.85 * (ds / rp) ** 0.4
        redundancy = max(0.1, 1.0 - 0.8 * (rp / (ds * 3)))
        resourcefulness = 0.7 if cp["type"] == "Multi-function" else 0.5
    else:
        return 0.0
    rapidity = min(1.0, rapidity_base)
    if cp["type"] == "Pollution control":
        w = [0.35, 0.25, 0.20, 0.20]
    elif cp["type"] == "Flood control":
        w = [0.25, 0.35, 0.15, 0.25]
    else:
        w = [0.20, 0.20, 0.30, 0.30]
    ufri = w[0] * robustness + w[1] * rapidity + w[2] * redundancy + w[3] * resourcefulness
    return max(0.0, min(1.0, ufri))


def find_Tcrit(cp, climate_factor=1.0):
    for rp in RETURN_PERIODS_FINE:
        if calc_ufri(rp, cp, climate_factor) < 0.5:
            return float(rp)
    return 250.0


def add_panel_label(ax, label, x=-0.06, y=1.03, fontsize=15, color="#272727",
                    fontweight="bold"):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight=fontweight, color=color, ha="left", va="bottom")


def main():
    deltas = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0]
    cities = list(CITY_PARAMS)

    tcrit_rows = []
    dtfri_rows = []
    for delta in deltas:
        row_t = {"Subdaily_Factor": delta}
        row_d = {"Subdaily_Factor": delta}
        for city in cities:
            cp = CITY_PARAMS[city]
            row_t["T_crit_" + city] = find_Tcrit(cp, delta)
            row_d["DTFRI_T50_" + city] = round(calc_ufri(50, cp, delta), 4)
        tcrit_rows.append(row_t)
        dtfri_rows.append(row_d)

    df_t = pd.DataFrame(tcrit_rows)
    df_d = pd.DataFrame(dtfri_rows)
    with pd.ExcelWriter(os.path.join(SCRIPT_DIR, "FigS15_Subdaily_Scaling.xlsx"),
                        engine="openpyxl") as w:
        df_t.to_excel(w, sheet_name="Tcrit_vs_delta", index=False)
        df_d.to_excel(w, sheet_name="DTFRI_T50_vs_delta", index=False)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.13, top=0.92,
                        wspace=0.28)

    # Panel a: T_crit vs subdaily amplification
    for city in cities:
        ax_a.plot(deltas, df_t["T_crit_" + city], "o-",
                  color=CITY_COLORS[city], linewidth=2.4, markersize=8,
                  markeredgecolor="white", markeredgewidth=0.7, label=city)
    ax_a.axvline(x=1.0, color="#767676", linestyle="--", linewidth=1.3, alpha=0.7)
    ax_a.set_xlabel("Subdaily amplification factor δ")
    ax_a.set_ylabel("Critical return period T_crit (years)")
    ax_a.set_title("a  T_crit under subdaily rainfall amplification",
                   fontweight="bold", loc="left", fontsize=14)
    ax_a.legend(loc="upper right", frameon=False)
    ax_a.grid(True, alpha=0.15)
    add_panel_label(ax_a, "a")

    # Panel b: DTFRI at T = 50 vs subdaily amplification
    for city in cities:
        ax_b.plot(deltas, df_d["DTFRI_T50_" + city], "o-",
                  color=CITY_COLORS[city], linewidth=2.4, markersize=8,
                  markeredgecolor="white", markeredgewidth=0.7, label=city)
    ax_b.axvline(x=1.0, color="#767676", linestyle="--", linewidth=1.3, alpha=0.7)
    ax_b.axhline(y=0.5, color="#B64342", linestyle=":", linewidth=1.2, alpha=0.6)
    ax_b.set_xlabel("Subdaily amplification factor δ")
    ax_b.set_ylabel("DTFRI at T = 50 years")
    ax_b.set_title("b  DTFRI(T = 50) under subdaily rainfall amplification",
                   fontweight="bold", loc="left", fontsize=14)
    ax_b.legend(loc="upper right", frameon=False)
    ax_b.grid(True, alpha=0.15)
    add_panel_label(ax_b, "b")

    pdf_path = os.path.join(SCRIPT_DIR, "FigS15_Subdaily_Scaling.pdf")
    tiff_path = os.path.join(SCRIPT_DIR, "FigS15_Subdaily_Scaling.tiff")
    fig.savefig(pdf_path, dpi=300, facecolor="white")
    fig.savefig(tiff_path, dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("Saved: %s" % pdf_path)
    print("Saved: %s" % tiff_path)
    print(df_t.to_string(index=False))


if __name__ == "__main__":
    main()
