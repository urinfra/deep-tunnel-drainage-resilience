"""
15案例扩展分析 — 批量RFC计算 + Supplementary图表生成
======================================================
基于Lin et al.(2026)全球15深隧案例分类框架
生成类型学验证和within-type变异分析
"""
import numpy as np
import json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(r"C:\ShareCache\林沛元\科研与学习\Paper\Journal Paper\TUNNEL ENGINEERING\Paper 5 Safety's Shadow")
OUTPUT = BASE / "Outputs" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ============================================================
# 15案例完整数据库
# ============================================================
CASES = {
    # === 污染控制型 (Pollution Control) - 5 cases ===
    "Sydney_NST": {
        "name": "Sydney NST", "tunnel": "Northern Sewage Tunnel", "city": "Sydney, Australia",
        "type": "pollution_control", "design_std": 10,
        "length_km": 16, "diameter_m": 5.2, "depth_m": 70,
        "storage_m3": 480_000, "pump_m3s": 15, "catchment_km2": 120,
        "cost_M": 300, "year": 2000, "rainfall_mm": 1210,
        "color": "#2980B9", "marker": "o",
        "source": "Sydney Water; 深隧论文改写稿 Table 1",
    },
    "London_Tideway": {
        "name": "London", "tunnel": "Thames Tideway", "city": "London, UK",
        "type": "pollution_control", "design_std": 30,
        "length_km": 25, "diameter_m": 7.22, "depth_m": 65,
        "storage_m3": 1_600_000, "pump_m3s": 27, "catchment_km2": 200,
        "cost_M": 4500, "year": 2025, "rainfall_mm": 620,
        "color": "#3498DB", "marker": "s",
        "source": "Tideway London; 深隧论文改写稿 Table 1",
    },
    "Louisville_Tunnel": {
        "name": "Louisville", "tunnel": "Louisville Tunnel", "city": "Louisville, USA",
        "type": "pollution_control", "design_std": 10,
        "length_km": 33.8, "diameter_m": 6.3, "depth_m": 45,
        "storage_m3": 1_240_000, "pump_m3s": 11, "catchment_km2": 180,
        "cost_M": 2500, "year": 2022, "rainfall_mm": 1150,
        "color": "#1ABC9C", "marker": "D",
        "source": "MSD Louisville; 深隧论文改写稿 Table 1",
    },
    "HK_HATS": {
        "name": "Hong Kong HATS", "tunnel": "Harbour Area Treatment", "city": "Hong Kong, China",
        "type": "pollution_control", "design_std": 15,
        "length_km": 44, "diameter_m": 2.0, "depth_m": 100,
        "storage_m3": 600_000, "pump_m3s": 50, "catchment_km2": 80,
        "cost_M": 3300, "year": 2015, "rainfall_mm": 2400,
        "color": "#9B59B6", "marker": "^",
        "source": "DSD Hong Kong; 深隧论文改写稿 Table 1",
    },
    "Wuhan_DadongLake": {
        "name": "Wuhan", "tunnel": "Dadong Lake", "city": "Wuhan, China",
        "type": "pollution_control", "design_std": 30,
        "length_km": 17.5, "diameter_m": 3.2, "depth_m": 51.5,
        "storage_m3": 350_000, "pump_m3s": 7, "catchment_km2": 30,
        "cost_M": 440, "year": 2020, "rainfall_mm": 1300,
        "color": "#8E44AD", "marker": "v",
        "source": "China Water & Wastewater; 深隧论文改写稿 Table 1",
    },
    "Chicago_TARP": {
        "name": "Chicago", "tunnel": "TARP Phase I", "city": "Chicago, USA",
        "type": "pollution_control", "design_std": 10,
        "length_km": 176, "diameter_m": 8.0, "depth_m": 80,
        "storage_m3": 66_200_000, "pump_m3s": 150, "catchment_km2": 950,
        "cost_M": 3830, "year": 2006, "rainfall_mm": 970,
        "color": "#2980B9", "marker": "P",
        "source": "MWRD; EPA. Primary: pollution control (CSO abatement). [8] classifies as multi-function incl. Phase II.",
    },
    "Singapore_DTSS": {
        "name": "Singapore", "tunnel": "DTSS", "city": "Singapore",
        "type": "pollution_control", "design_std": 30,
        "length_km": 48, "diameter_m": 4.5, "depth_m": 55,
        "storage_m3": 800_000, "pump_m3s": 37, "catchment_km2": 150,
        "cost_M": 6500, "year": 2008, "rainfall_mm": 2340,
        "color": "#34495E", "marker": "X",
        "source": "PUB Singapore. Primary: sewerage/water quality. [8] classifies as multi-function incl. NEWater.",
    },

    # === 洪涝控制型 (Flood Control) - 5 cases ===
    "Tokyo_MAOUDC": {
        "name": "Tokyo", "tunnel": "MAOUDC", "city": "Tokyo, Japan",
        "type": "flood_control", "design_std": 50,
        "length_km": 6.3, "diameter_m": 10.6, "depth_m": 50,
        "storage_m3": 670_000, "pump_m3s": 200, "catchment_km2": 65,
        "cost_M": 1630, "year": 2006, "rainfall_mm": 1530,
        "color": "#27AE60", "marker": "o",
        "source": "MLIT Japan; verified_city_params.json",
    },
    "Osaka_Neyagawa": {
        "name": "Osaka", "tunnel": "Neyagawa", "city": "Osaka, Japan",
        "type": "flood_control", "design_std": 50,
        "length_km": 2.2, "diameter_m": 8.5, "depth_m": 50,
        "storage_m3": 540_000, "pump_m3s": 100, "catchment_km2": 15,
        "cost_M": 340, "year": 2006, "rainfall_mm": 1350,
        "color": "#2ECC71", "marker": "s",
        "source": "MLIT Japan; 深隧论文改写稿 Table 1",
    },
    "HK_LaiChiKok": {
        "name": "HK Lai Chi Kok", "tunnel": "Lai Chi Kok Drainage", "city": "Hong Kong, China",
        "type": "flood_control", "design_std": 50,
        "length_km": 10.5, "diameter_m": 6.75, "depth_m": 100,
        "storage_m3": 400_000, "pump_m3s": 150, "catchment_km2": 12,
        "cost_M": 435, "year": 2012, "rainfall_mm": 2400,
        "color": "#16A085", "marker": "D",
        "source": "DSD Hong Kong; 深隧论文改写稿 Table 1",
    },
    "HK_TsuenWan": {
        "name": "HK Tsuen Wan", "tunnel": "Tsuen Wan Drainage", "city": "Hong Kong, China",
        "type": "flood_control", "design_std": 50,
        "length_km": 3.7, "diameter_m": 4.9, "depth_m": 40,
        "storage_m3": 180_000, "pump_m3s": 102, "catchment_km2": 7,
        "cost_M": 215, "year": 2012, "rainfall_mm": 2400,
        "color": "#1ABC9C", "marker": "*",
        "source": "DSD Hong Kong; 深隧论文改写稿 Table 1",
    },
    "Macau_Drainage": {
        "name": "Macau", "tunnel": "Macau Drainage", "city": "Macau, China",
        "type": "flood_control", "design_std": 50,
        "length_km": 5.1, "diameter_m": 6.5, "depth_m": 50,
        "storage_m3": 300_000, "pump_m3s": 220, "catchment_km2": 10,
        "cost_M": 190, "year": 2013, "rainfall_mm": 2100,
        "color": "#F1C40F", "marker": "^",
        "source": "DSAMA Macau; 深隧论文改写稿 Table 1",
    },

    # === 多功能型 (Multi-function) - 2 cases ===
    "KualaLumpur_SMART": {
        "name": "KL", "tunnel": "SMART", "city": "Kuala Lumpur, Malaysia",
        "type": "multi_function", "design_std": 100,
        "length_km": 9.7, "diameter_m": 11.83, "depth_m": 40,
        "storage_m3": 3_000_000, "pump_m3s": 280, "catchment_km2": 55,
        "cost_M": 515, "year": 2007, "rainfall_mm": 2500,
        "color": "#E74C3C", "marker": "s",
        "source": "Xiao(2015); Isah & Ali(2015); 深隧论文改写稿 Table 1",
    },
    "Guangzhou_DHC": {
        "name": "Guangzhou", "tunnel": "Donghaochong", "city": "Guangzhou, China",
        "type": "multi_function", "design_std": 50,
        "length_km": 1.77, "diameter_m": 5.3, "depth_m": 40,
        "storage_m3": 100_000, "pump_m3s": 48, "catchment_km2": 12.5,
        "cost_M": 100, "year": 2018, "rainfall_mm": 1800,
        "color": "#C0392B", "marker": "o",
        "source": "Guangzhou Water Bureau; calibrated n=0.025; 深隧论文改写稿 Table 1",
    },

    # === NOTE: Chicago TARP Phase I (pollution_control above) and        ===
    # === Singapore DTSS (pollution_control above) are classified in the   ===
    # === 深隧论文改写稿[8] as "multi-function" under a broader framework ===
    # === that includes secondary co-benefits. For the Nature Cities RFC   ===
    # === analysis, the classification follows hydraulic bottleneck type.  ===
}

RETURN_PERIODS = [2, 5, 10, 20, 30, 50, 100, 200]

# ============================================================
# UFRI/RFC 计算 (与正文模型一致)
# ============================================================

def calc_tunnel_efficiency(rp, case, climate_factor=1.0):
    """隧道洪水削减效率"""
    ds = case["design_std"]
    rp_eff = rp * climate_factor
    typ = case["type"]

    if typ == "pollution_control":
        eta = 1.0 / (1.0 + np.exp((rp_eff - ds * 0.8) / (ds * 0.3)))
    elif typ == "flood_control":
        eta = 1.0 / (1.0 + np.exp((rp_eff - ds) / (ds * 0.2)))
    else:
        eta = 1.0 / (1.0 + np.exp((rp_eff - ds * 1.1) / (ds * 0.4)))
    return max(0.0, min(1.0, eta))

def calc_ufri(rp, case, climate_factor=1.0):
    """计算UFRI"""
    eta = calc_tunnel_efficiency(rp, case, climate_factor)
    ds = case["design_std"]
    typ = case["type"]

    robustness = eta
    rapidity = min(1.0, 0.85 if rp <= ds else 0.85 * (ds / rp) ** 0.4)
    redundancy = max(0.1, 1.0 - 0.8 * (rp / (ds * 3)))
    resourcefulness = 0.7 if typ == "multi_function" else 0.5

    if typ == "pollution_control":
        w = [0.35, 0.25, 0.20, 0.20]
    elif typ == "flood_control":
        w = [0.25, 0.35, 0.15, 0.25]
    else:
        w = [0.20, 0.20, 0.30, 0.30]

    return max(0.0, min(1.0,
        w[0]*robustness + w[1]*rapidity + w[2]*redundancy + w[3]*resourcefulness))

def find_Tcrit(case, climate_factor=1.0):
    """寻找UFRI降至0.5的临界重现期"""
    for rp in RETURN_PERIODS:
        if calc_ufri(rp, case, climate_factor) < 0.5:
            return rp
    return 250  # 超出范围

# ============================================================
# Fig S1: 15案例RFC汇总 (按类型分组)
# ============================================================

def fig_s1_all_rfc():
    """Supplementary Fig S1: 全部15案例RFC曲线，按功能类型分3个子图"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    type_groups = {
        "Pollution Control": ("pollution_control", axes[0]),
        "Flood Control": ("flood_control", axes[1]),
        "Multi-function": ("multi_function", axes[2]),
    }

    climate_scenarios = [
        ("Historical", 1.0, "#2C3E50", "-"),
        ("SSP5-8.5 2080s", 1.30, "#E74C3C", "--"),
    ]

    for title, (typ, ax) in type_groups.items():
        cases_in_type = {k: v for k, v in CASES.items() if v["type"] == typ}

        for case_key, case in cases_in_type.items():
            for sc_name, cf, color, ls in climate_scenarios:
                ufris = [calc_ufri(rp, case, cf) for rp in RETURN_PERIODS]
                ax.semilogx(RETURN_PERIODS, ufris,
                             color=color, linestyle=ls, linewidth=1.5,
                             alpha=0.7 if sc_name == "SSP5-8.5 2080s" else 1.0)

            # 标注设计标准点
            ds = case["design_std"]
            ufri_ds = calc_ufri(ds, case, 1.0)
            ax.plot(ds, ufri_ds, marker=case["marker"], color=case["color"],
                    markersize=10, markeredgecolor="white", markeredgewidth=0.8,
                    label=f"{case['name']} ({ds}yr)", zorder=5)

        ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.4, linewidth=1)
        ax.set_xlim(1.5, 250); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Return Period (years)")
        ax.set_ylabel("UFRI")
        ax.set_title(f"{title}\n({len(cases_in_type)} cases)", fontweight="bold")
        ax.legend(fontsize=8, loc="lower left", framealpha=0.8)
        ax.grid(True, alpha=0.15)

    fig.suptitle("Supplementary Fig. S1 | RFC Curves for All 15 Deep Tunnel Cases\n"
                 "Grouped by functional type. Solid: Historical; Dashed: SSP5-8.5 2080s",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()

    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(OUTPUT / f"ED_FigS1_All15_RFC.{fmt}", dpi=300,
                     facecolor="white")
    plt.close(fig)
    print("  [OK] ED Fig S1: 15-case RFC curves")


# ============================================================
# Fig S2: T_crit vs 设计标准 (15案例散点+回归)
# ============================================================

def fig_s2_tcrit_vs_design():
    """Supplementary Fig S2: T_crit vs Design Standard"""
    fig, ax = plt.subplots(figsize=(10, 7))

    type_config = {
        "pollution_control": ("Pollution control", "#3498DB", "o"),
        "flood_control": ("Flood control", "#27AE60", "s"),
        "multi_function": ("Multi-function", "#E74C3C", "D"),
    }

    for typ, (label, color, marker) in type_config.items():
        cases_in_type = {k: v for k, v in CASES.items() if v["type"] == typ}
        xs = [v["design_std"] for v in cases_in_type.values()]
        ys = [find_Tcrit(v) for v in cases_in_type.values()]
        names = [v["name"] for v in cases_in_type.values()]

        ax.scatter(xs, ys, c=color, marker=marker, s=150,
                    edgecolors="white", linewidth=1, zorder=5, label=label)

        for x, y, n in zip(xs, ys, names):
            offset = 3 if n != "Singapore" else -8
            ax.annotate(n, (x, y), fontsize=8, ha="center",
                         xytext=(0, offset), textcoords="offset points")

    # 1:1参考线
    ax.plot([0, 120], [0, 120], "k--", alpha=0.3, linewidth=1, label="1:1 line")
    ax.axhline(y=10, color="gray", linestyle=":", alpha=0.3)

    ax.set_xlim(0, 115); ax.set_ylim(0, 250)
    ax.set_xlabel("Design Return Period (years)")
    ax.set_ylabel("Critical Return Period T_crit (UFRI < 0.5)")
    ax.set_title("Supplementary Fig. S2 | T_crit vs. Design Standard\n"
                 "All 15 cases under Historical climate", fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(OUTPUT / f"ED_FigS2_Tcrit_vs_Design.{fmt}", dpi=300,
                     facecolor="white")
    plt.close(fig)
    print("  [OK] ED Fig S2: T_crit vs Design Standard")


# ============================================================
# Fig S3: Within-type变异分析 (同类型内的RFC band)
# ============================================================

def fig_s3_within_type_variation():
    """Supplementary Fig S3: 同类型内的RFC变异范围 (band plot)"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    type_groups = {
        "Pollution Control (4 cases)": ("pollution_control", axes[0]),
        "Flood Control (5 cases)": ("flood_control", axes[1]),
        "Multi-function (6 cases)": ("multi_function", axes[2]),
    }

    rp_dense = np.logspace(np.log10(1.5), np.log10(250), 50)

    for title, (typ, ax) in type_groups.items():
        cases_in_type = {k: v for k, v in CASES.items() if v["type"] == typ}

        all_ufri_curves = []
        for case_key, case in cases_in_type.items():
            ufris = [calc_ufri(rp, case, 1.0) for rp in rp_dense]
            all_ufri_curves.append(ufris)
            ax.semilogx(rp_dense, ufris, alpha=0.3, linewidth=0.8,
                         color=case["color"])

        all_ufri_curves = np.array(all_ufri_curves)
        median_curve = np.median(all_ufri_curves, axis=0)
        lower_curve = np.min(all_ufri_curves, axis=0)
        upper_curve = np.max(all_ufri_curves, axis=0)

        ax.fill_between(rp_dense, lower_curve, upper_curve,
                         alpha=0.15, color="gray")
        ax.semilogx(rp_dense, median_curve, "k-", linewidth=3,
                     label=f"Median (n={len(cases_in_type)})")
        ax.semilogx(rp_dense, lower_curve, "k--", linewidth=1, alpha=0.6,
                     label="Min–Max range")
        ax.semilogx(rp_dense, upper_curve, "k--", linewidth=1, alpha=0.6)

        # 标注变异系数最大的点
        cv = np.std(all_ufri_curves, axis=0) / (np.abs(median_curve) + 0.01)
        max_cv_idx = np.argmax(cv)
        ax.annotate(f"Max CV = {cv[max_cv_idx]:.2f}\n@ T ≈ {rp_dense[max_cv_idx]:.0f}yr",
                     xy=(rp_dense[max_cv_idx], median_curve[max_cv_idx]),
                     fontsize=9, ha="left",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.4)
        ax.set_xlim(1.5, 250); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Return Period (years)")
        ax.set_ylabel("UFRI")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(True, alpha=0.15)

    fig.suptitle("Supplementary Fig. S3 | Within-Type RFC Variation\n"
                 "Shaded bands show min–max range within each functional type",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()

    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(OUTPUT / f"ED_FigS3_WithinType_Variation.{fmt}", dpi=300,
                     facecolor="white")
    plt.close(fig)
    print("  [OK] ED Fig S3: Within-type variation")


# ============================================================
# Fig S4: 15案例全景仪表板
# ============================================================

def fig_s4_dashboard():
    """Supplementary Fig S4: 15案例关键指标仪表板"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    case_names = [c["name"] for c in CASES.values()]
    case_colors = [c["color"] for c in CASES.values()]

    # Panel A: Design standard bar chart
    ax = axes[0, 0]
    ds_vals = [c["design_std"] for c in CASES.values()]
    bars = ax.barh(range(len(case_names)), ds_vals, color=case_colors,
                    edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(case_names)))
    ax.set_yticklabels(case_names, fontsize=9)
    ax.set_xlabel("Design Return Period (years)")
    ax.set_title("a  Design Standard", fontweight="bold", loc="left")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.2, axis="x")

    # Panel B: T_crit bar chart
    ax = axes[0, 1]
    tcrit_vals = [find_Tcrit(c) for c in CASES.values()]
    bars = ax.barh(range(len(case_names)), tcrit_vals, color=case_colors,
                    edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(case_names)))
    ax.set_yticklabels(case_names, fontsize=9)
    ax.set_xlabel("Critical Return Period T_crit (years)")
    ax.set_title("b  Critical Threshold T_crit (UFRI < 0.5)", fontweight="bold", loc="left")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.2, axis="x")

    # Panel C: Storage vs Design Standard scatter
    ax = axes[1, 0]
    type_markers = {"pollution_control": "o", "flood_control": "s", "multi_function": "D"}
    for case_key, case in CASES.items():
        ax.scatter(case["design_std"], case["storage_m3"], c=case["color"],
                    marker=type_markers[case["type"]], s=120,
                    edgecolors="white", linewidth=0.8)
        ax.annotate(case["name"], (case["design_std"], case["storage_m3"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("Design Return Period (years)")
    ax.set_ylabel("Storage Volume (m³, log scale)")
    ax.set_title("c  Storage Volume vs. Design Standard", fontweight="bold", loc="left")
    ax.grid(True, alpha=0.2)

    # Legend for types
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498DB",
               markersize=10, label="Pollution control"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#27AE60",
               markersize=10, label="Flood control"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#E74C3C",
               markersize=10, label="Multi-function"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    # Panel D: Climate vulnerability (ΔUFRI at T=50 under SSP5-8.5 2080s)
    ax = axes[1, 1]
    deltas = []
    for case in CASES.values():
        ufri_hist = calc_ufri(50, case, 1.0)
        ufri_future = calc_ufri(50, case, 1.30)
        deltas.append((ufri_hist - ufri_future) / max(ufri_hist, 0.01) * 100)

    bars = ax.barh(range(len(case_names)), deltas, color=case_colors,
                    edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(case_names)))
    ax.set_yticklabels(case_names, fontsize=9)
    ax.set_xlabel("UFRI Decline at T=50yr (%)")
    ax.set_title("d  Climate Vulnerability\n(SSP5-8.5 2080s vs Historical, T=50yr)",
                 fontweight="bold", loc="left")
    ax.axvline(x=20, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle("Supplementary Fig. S4 | 15-Case Deep Tunnel System Dashboard\n"
                 "Key resilience indicators across all cases",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()

    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(OUTPUT / f"ED_FigS4_Dashboard.{fmt}", dpi=300,
                     facecolor="white")
    plt.close(fig)
    print("  [OK] ED Fig S4: 15-case dashboard")


# ============================================================
# 生成15案例参数表
# ============================================================

def generate_case_table():
    """生成15案例参数汇总CSV"""
    import pandas as pd
    rows = []
    for key, c in CASES.items():
        rows.append({
            "Case": c["name"],
            "Tunnel": c["tunnel"],
            "City": c["city"],
            "Type": c["type"].replace("_", " ").title(),
            "Design_Std_yr": c["design_std"],
            "Length_km": c["length_km"],
            "Diameter_m": c["diameter_m"],
            "Depth_m": c["depth_m"],
            "Storage_m3": c["storage_m3"],
            "Pump_m3s": c["pump_m3s"],
            "Catchment_km2": c["catchment_km2"],
            "Cost_MUSD": c["cost_M"],
            "Year": c["year"],
            "Rainfall_mm": c["rainfall_mm"],
            "T_crit_yr": find_Tcrit(c),
            "UFRI_at_T50": round(calc_ufri(50, c, 1.0), 3),
            "UFRI_at_T100": round(calc_ufri(100, c, 1.0), 3),
            "Delta_UFRI_585_pct": round(
                (calc_ufri(50, c, 1.0) - calc_ufri(50, c, 1.30))
                / max(calc_ufri(50, c, 1.0), 0.01) * 100, 1),
            "Source": c["source"],
        })

    df = pd.DataFrame(rows)
    csv_path = BASE / "Outputs" / "supplementary" / "ED_Table_S1_15Case_Parameters.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] 15-case parameter table: {csv_path.name}")

    # Also print summary stats
    for typ in ["pollution_control", "flood_control", "multi_function"]:
        subset = df[df["Type"] == typ.replace("_", " ").title()]
        print(f"\n  {typ}: n={len(subset)}, "
              f"Design std: {subset['Design_Std_yr'].min()}-{subset['Design_Std_yr'].max()} yr, "
              f"T_crit: {subset['T_crit_yr'].min()}-{subset['T_crit_yr'].max()} yr, "
              f"ΔUFRI: {subset['Delta_UFRI_585_pct'].min()}-{subset['Delta_UFRI_585_pct'].max()}%")

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  15案例扩展分析 — Supplementary Materials")
    print("=" * 60)

    print(f"\n案例总数: {len(CASES)}")
    for typ in ["pollution_control", "flood_control", "multi_function"]:
        n = sum(1 for c in CASES.values() if c["type"] == typ)
        names = [c["name"] for c in CASES.values() if c["type"] == typ]
        print(f"  {typ}: {n} cases — {', '.join(names)}")

    print("\n[1/5] ED Fig S1: 15-case RFC Curves...")
    fig_s1_all_rfc()

    print("\n[2/5] ED Fig S2: T_crit vs Design Standard...")
    fig_s2_tcrit_vs_design()

    print("\n[3/5] ED Fig S3: Within-type Variation...")
    fig_s3_within_type_variation()

    print("\n[4/5] ED Fig S4: 15-case Dashboard...")
    fig_s4_dashboard()

    print("\n[5/5] Parameter Table...")
    df = generate_case_table()

    print(f"\n{'='*60}")
    print(f"  Supplementary materials generated in: {OUTPUT}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
