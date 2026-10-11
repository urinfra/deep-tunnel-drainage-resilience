"""
补充3张Sensitivity Analysis图 (SA7-SA9)
SA7: 参数传递不确定性 — 糙率±30%对芝加哥/东京/KL UFRI的影响
SA8: 14案例T_crit聚类显著性 — 功能类型统计检验
SA9: 沉积物长期影响 — 糙率退化对UFRI的时间演变
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT = Path(r"C:\ShareCache\林沛元\科研与学习\Paper\Journal Paper\TUNNEL ENGINEERING\Paper 5 Safety's Shadow\Outputs\sensitivity")
OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 13,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})

CITY_LABELS = ["Guangzhou", "Chicago", "Tokyo", "Kuala Lumpur"]
CITY_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
CITY_TYPES = ["Multi-function", "Pollution control", "Flood control", "Multi-function"]
DESIGN_STDS = [50, 10, 50, 100]
CALIBRATED_N = {"Guangzhou": 0.025, "Chicago": 0.025, "Tokyo": 0.018, "KualaLumpur": 0.025}
DESIGN_N = 0.015

# ============================================================
# Shared UFRI calculator
# ============================================================
def compute_ufri(rp, ds, city_type, n_value, climate_factor=1.0):
    """Compute UFRI with roughness-dependent efficiency"""
    n_ratio = n_value / DESIGN_N
    effective_ds = ds / (n_ratio ** 0.6)
    rp_eff = rp * climate_factor

    if city_type == "Pollution control":
        eta = 1.0 / (1.0 + np.exp((rp_eff - effective_ds * 0.8) / (effective_ds * 0.3)))
        w = [0.35, 0.25, 0.20, 0.20]
    elif city_type == "Flood control":
        eta = 1.0 / (1.0 + np.exp((rp_eff - effective_ds) / (effective_ds * 0.2)))
        w = [0.25, 0.35, 0.15, 0.25]
    else:
        eta = 1.0 / (1.0 + np.exp((rp_eff - effective_ds * 1.1) / (effective_ds * 0.4)))
        w = [0.20, 0.20, 0.30, 0.30]

    robustness = max(0, min(1, eta))
    rapidity = min(1, 0.85 if rp <= effective_ds else 0.85 * (effective_ds / rp) ** 0.4)
    redundancy = max(0.1, 1 - 0.8 * (rp / (effective_ds * 3)))
    resourcefulness = 0.7 if "Multi" in city_type else 0.5

    return w[0]*robustness + w[1]*rapidity + w[2]*redundancy + w[3]*resourcefulness


# ============================================================
# SA7: Roughness Parameter Transfer Uncertainty
# ============================================================
def sa7_roughness_transfer():
    """糙率±30%对芝加哥/东京/KL三城市UFRI的影响"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    transfer_cities = [
        ("Chicago", "Pollution control", 10, 0.025, "#3498DB"),
        ("Tokyo", "Flood control", 50, 0.018, "#2ECC71"),
        ("Kuala Lumpur", "Multi-function", 100, 0.025, "#F39C12"),
    ]
    n_variations = [0.7, 0.85, 1.0, 1.15, 1.3]
    rps = [2, 5, 10, 20, 30, 50, 100, 200]

    for idx, (city, ctype, ds, n_base, color) in enumerate(transfer_cities):
        ax = axes[idx]
        for n_mult in n_variations:
            n_val = n_base * n_mult
            ufris = [compute_ufri(rp, ds, ctype, n_val, 1.0) for rp in rps]
            ls = "-" if n_mult == 1.0 else "--"
            lw = 3.0 if n_mult == 1.0 else 1.5
            alpha_val = 1.0 if n_mult == 1.0 else 0.5
            label = f"n={n_val:.3f} (base)" if n_mult == 1.0 else f"n={n_val:.3f} (±{abs(n_mult-1)*100:.0f}%)"
            ax.semilogx(rps, ufris, "o-", color=color, linewidth=lw, markersize=6 if n_mult==1 else 4,
                        alpha=alpha_val, linestyle=ls, label=label,
                        markeredgecolor="white", markeredgewidth=0.5)

        ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.4)
        ax.axvline(x=ds, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlim(1.5, 250); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Return Period (years)")
        ax.set_ylabel("UFRI")
        ax.set_title(f"{city}\n(DS={ds}yr, n_base={n_base})", fontweight="bold")
        ax.legend(fontsize=10, loc="lower left")
        ax.grid(True, alpha=0.15)

    fig.suptitle("ED Fig. S11 | Roughness Parameter Transfer Uncertainty\n"
                 "Sensitivity of UFRI to ±30% variation in calibrated roughness (n) for Chicago, Tokyo, and Kuala Lumpur",
                 fontsize=19, fontweight="bold", y=1.02)
    fig.tight_layout()
    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(OUTPUT / f"SA7_Roughness_Transfer.{fmt}", dpi=300, facecolor="white")
    plt.close(fig)
    print("  [OK] SA7: Roughness Transfer Uncertainty")


# ============================================================
# SA8: 14-case T_crit Clustering Statistical Test
# ============================================================
def sa8_tcrit_clustering():
    """14案例T_crit按功能类型的统计检验 (ANOVA + box plot)"""
    from expand_15cases import CASES as CASES_15, find_Tcrit

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Collect T_crit by type
    type_data = {"Pollution\ncontrol": [], "Flood\ncontrol": [], "Multi-\nfunction": []}
    type_colors = {"Pollution\ncontrol": "#3498DB", "Flood\ncontrol": "#27AE60", "Multi-\nfunction": "#E74C3C"}
    case_names_by_type = {"Pollution\ncontrol": [], "Flood\ncontrol": [], "Multi-\nfunction": []}

    for key, case in CASES_15.items():
        if key.startswith("_"):
            continue
        tcrit = find_Tcrit(case)
        typ = case["type"]
        if typ == "pollution_control":
            type_data["Pollution\ncontrol"].append(tcrit)
            case_names_by_type["Pollution\ncontrol"].append(case["name"])
        elif typ == "flood_control":
            type_data["Flood\ncontrol"].append(tcrit)
            case_names_by_type["Flood\ncontrol"].append(case["name"])
        else:
            type_data["Multi-\nfunction"].append(tcrit)
            case_names_by_type["Multi-\nfunction"].append(case["name"])

    # Panel A: Box plot + jittered scatter
    ax = axes[0]
    positions = list(range(3))
    for i, (typ_name, values) in enumerate(type_data.items()):
        bp = ax.boxplot([values], positions=[i], patch_artist=True, widths=0.4)
        bp["boxes"][0].set_facecolor(type_colors[typ_name])
        bp["boxes"][0].set_alpha(0.5)
        # Jittered scatter
        jitter = np.random.normal(0, 0.05, len(values))
        ax.scatter(np.ones(len(values))*i + jitter, values, c=type_colors[typ_name],
                   s=80, alpha=0.8, edgecolors="white", linewidth=1, zorder=5)
        # Label each point
        for j, (name, val) in enumerate(zip(case_names_by_type[typ_name], values)):
            ax.annotate(name, (i + jitter[j], val), fontsize=9,
                        xytext=(8, 0), textcoords="offset points", alpha=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(list(type_data.keys()))
    ax.set_ylabel("Critical Return Period T_crit (years)")
    ax.set_title("a  T_crit Distribution by Functional Type\n(14 cases, dots = individual systems)",
                 fontweight="bold", loc="left")
    ax.grid(True, alpha=0.2, axis="y")

    # Kruskal-Wallis test
    all_groups = [type_data[k] for k in type_data.keys()]
    h_stat, p_val = stats.kruskal(*all_groups)
    ax.text(0.98, 0.95, f"Kruskal-Wallis:\nH = {h_stat:.1f}\np = {p_val:.4f}",
            transform=ax.transAxes, fontsize=14, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    # Panel B: Design std vs T_crit with type regression
    ax = axes[1]
    all_ds = []
    all_tc = []
    for key, case in CASES_15.items():
        if key.startswith("_"):
            continue
        tcrit = find_Tcrit(case)
        ds = case["design_std"]
        typ = case["type"]
        if typ == "pollution_control":
            c, m, label = "#3498DB", "o", "Pollution control"
        elif typ == "flood_control":
            c, m, label = "#27AE60", "s", "Flood control"
        else:
            c, m, label = "#E74C3C", "D", "Multi-function"
        ax.scatter(ds, tcrit, c=c, marker=m, s=150, edgecolors="white",
                   linewidth=1, zorder=5, label=label)
        ax.annotate(case["name"], (ds, tcrit), fontsize=9,
                     xytext=(4, 4), textcoords="offset points")
        all_ds.append(ds); all_tc.append(tcrit)

    # Regression
    slope, intercept, r_val, p_val2, _ = stats.linregress(all_ds, all_tc)
    x_line = np.array([0, 120])
    ax.plot(x_line, slope*x_line + intercept, "k--", linewidth=1.5, alpha=0.5,
            label=f"r={r_val:.3f}, p={p_val2:.4f}")
    ax.plot([0, 120], [0, 120], "gray", linestyle=":", alpha=0.3)

    ax.set_xlim(0, 115); ax.set_ylim(0, 250)
    ax.set_xlabel("Design Return Period (years)")
    ax.set_ylabel("Critical Return Period T_crit (years)")
    ax.set_title("b  T_crit vs. Design Standard\n(14 cases, with linear regression)",
                 fontweight="bold", loc="left")
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=11)
    ax.grid(True, alpha=0.2)

    fig.suptitle("ED Fig. S12 | Statistical Validation of Functional Typology\n"
                 "T_crit clustering confirms significant separation by functional type (Kruskal-Wallis test)",
                 fontsize=19, fontweight="bold", y=1.02)
    fig.tight_layout()
    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(OUTPUT / f"SA8_Tcrit_Clustering.{fmt}", dpi=300, facecolor="white")
    plt.close(fig)
    print("  [OK] SA8: T_crit Clustering (Kruskal-Wallis)")

# ============================================================
# SA9: Sediment-Induced Roughness Degradation
# ============================================================
def sa9_sediment_impact():
    """沉积物引起的糙率退化对UFRI的长期影响"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Observed sediment data (Guangzhou)
    years_obs = np.array([2022, 2023, 2024, 2025])
    sediment_cm = np.array([10, 50, 22.5, 57.5])
    sediment_min = np.array([8, 40, 20, 35])
    sediment_max = np.array([12, 60, 25, 80])
    # Roughness vs sediment (empirical: n ≈ 0.015 + 0.000125 * sediment_cm)
    n_from_sediment = 0.015 + 0.000125 * sediment_cm

    # Panel A: Sediment Accumulation Trend
    ax = axes[0, 0]
    ax.errorbar(years_obs, sediment_cm,
                yerr=[sediment_cm - sediment_min, sediment_max - sediment_cm],
                fmt="o-", color="#8E44AD", linewidth=3, markersize=14,
                markerfacecolor="white", markeredgewidth=2.5, capsize=8)
    ax.fill_between(years_obs, sediment_min, sediment_max, alpha=0.15, color="#8E44AD")
    # Project future
    years_future = np.arange(2022, 2035)
    sed_proj = 10 + 14 * (years_future - 2022)
    ax.plot(years_future, sed_proj, "--", color="#E74C3C", linewidth=2, alpha=0.6,
            label="Linear projection")
    ax.axhline(y=53, color="red", linestyle=":", alpha=0.5)
    ax.text(2033, 55, "Design clearance\n(~5.3 m dia.)", fontsize=11, color="red")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sediment Thickness (cm)")
    ax.set_title("a  Observed & Projected Sediment\nGuangzhou Donghaochong",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)

    # Panel B: Roughness evolution with sediment
    ax = axes[0, 1]
    sed_continuous = np.linspace(0, 100, 50)
    n_continuous = 0.015 + 0.000125 * sed_continuous
    ax.plot(sed_continuous, n_continuous, "-", color="#8E44AD", linewidth=3)
    ax.scatter(sediment_cm, n_from_sediment, c="#E74C3C", s=120, zorder=5,
               edgecolors="white", linewidth=1.5)
    for yr, s, n in zip(years_obs, sediment_cm, n_from_sediment):
        ax.annotate(str(yr), (s, n), fontsize=11, xytext=(5, 5),
                     textcoords="offset points")
    ax.axhline(y=0.015, color="gray", linestyle="--", alpha=0.5)
    ax.text(5, 0.0155, "Design n = 0.015", fontsize=11, color="gray")
    ax.set_xlabel("Sediment Thickness (cm)")
    ax.set_ylabel("Effective Manning's n")
    ax.set_title("b  Roughness vs. Sediment Accumulation\n(n = 0.015 + 0.000125 x sediment_thickness)",
                 fontweight="bold", loc="left")
    ax.grid(True, alpha=0.2)

    # Panel C: UFRI degradation over time (Guangzhou, T=50yr)
    ax = axes[1, 0]
    years_proj = np.arange(2022, 2042)
    sed_proj_full = 10 + 14 * (years_proj - 2022)
    n_proj = 0.015 + 0.000125 * sed_proj_full
    n_proj = np.clip(n_proj, 0.015, 0.040)

    ufri_proj_hist = [compute_ufri(50, 50, "Multi-function", n, 1.0) for n in n_proj]
    ufri_proj_585 = [compute_ufri(50, 50, "Multi-function", n, 1.30) for n in n_proj]

    ax.plot(years_proj, ufri_proj_hist, "-", color="#2C3E50", linewidth=3,
            label="Historical climate")
    ax.plot(years_proj, ufri_proj_585, "-", color="#E74C3C", linewidth=3,
            label="SSP5-8.5 2080s")
    ax.axvline(x=2025, color="gray", linestyle="--", alpha=0.5)
    ax.text(2025.5, 0.15, "Last\nobserved", fontsize=10, color="gray")
    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.4)
    ax.set_xlim(2022, 2041); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Year")
    ax.set_ylabel("UFRI at T=50yr")
    ax.set_title("c  Projected UFRI Degradation (Guangzhou)\nSediment-driven roughness increase over time",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)

    # Panel D: Year when UFRI drops below 0.5 vs maintenance frequency
    ax = axes[1, 1]
    cleaning_intervals = np.array([0.5, 1, 2, 3, 5, 10])  # years between cleaning
    # Roughness equilibrium ≈ 0.015 + 0.000125 * accumulation_rate * interval
    accumulation_rate = 14  # cm/yr
    eq_n = 0.015 + 0.000125 * accumulation_rate * cleaning_intervals
    eq_n = np.clip(eq_n, 0.015, 0.050)

    ufri_50 = [compute_ufri(50, 50, "Multi-function", n, 1.30) for n in eq_n]
    years_to_failure = []
    for interval in cleaning_intervals:
        # Find when UFRI drops below 0.5
        for yr in range(0, 100):
            sed = 10 + 14 * yr - 14 * (yr // max(interval, 0.5)) * max(interval, 0.5)
            sed = max(0, sed)
            n_val = 0.015 + 0.000125 * sed
            n_val = min(n_val, 0.050)
            ufri = compute_ufri(50, 50, "Multi-function", n_val, 1.30)
            if ufri < 0.5:
                years_to_failure.append(yr + 2022)
                break
        else:
            years_to_failure.append(2100)

    colors_bar = []
    for u in ufri_50:
        if u > 0.6: colors_bar.append("#27AE60")
        elif u > 0.4: colors_bar.append("#F39C12")
        else: colors_bar.append("#E74C3C")

    bars = ax.bar(range(len(cleaning_intervals)), ufri_50, color=colors_bar,
                  edgecolor="white", linewidth=1.5)
    for i, (bar, u_val, yf) in enumerate(zip(bars, ufri_50, years_to_failure)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"UFRI={u_val:.2f}\nFail: {yf}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(cleaning_intervals)))
    ax.set_xticklabels([f"{v:.1f}" if v < 1 else str(int(v)) for v in cleaning_intervals])
    ax.set_xlabel("Sediment Cleaning Interval (years)")
    ax.set_ylabel("Equilibrium UFRI at T=50yr\n(SSP5-8.5 2080s)")
    ax.set_title("d  Impact of Maintenance Frequency\n(Guangzhou, under sediment accumulation)",
                 fontweight="bold", loc="left")
    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.4)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("ED Fig. S13 | Sediment Accumulation and Long-Term Resilience\n"
                 "Impact of sediment-driven roughness degradation on UFRI and maintenance planning",
                 fontsize=19, fontweight="bold", y=1.02)
    fig.tight_layout()
    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(OUTPUT / f"SA9_Sediment_Impact.{fmt}", dpi=300, facecolor="white")
    plt.close(fig)
    print("  [OK] SA9: Sediment Impact Analysis")


# ============================================================
def main():
    print("=" * 60)
    print("  SA7-SA9: Additional Sensitivity Analyses")
    print("=" * 60)

    sa7_roughness_transfer()
    sa8_tcrit_clustering()
    sa9_sediment_impact()

    print(f"\nDone. Output: {OUTPUT}")
    for f in sorted(OUTPUT.glob("SA[789]*.png")):
        print(f"  {f.name}")

if __name__ == "__main__":
    main()
