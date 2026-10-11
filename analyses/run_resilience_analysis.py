#!/usr/bin/env python3
"""
深隧韧性分析集成管线
=====================
连接 SWMM 建模 → 韧性指标计算 → RFC曲线生成 → 公平性分析 的端到端流程。

用法:
    python run_resilience_analysis.py --city Guangzhou --config config.json

配置项 (config.json):
    - cities: 目标城市列表
    - scenarios: 气候情景 [historical, ssp245_2050, ssp585_2050, ssp585_2080]
    - return_periods: 重现期 [2, 5, 10, 20, 50, 100, 200]
    - infrastructures: 基础设施方案 [no_tunnel, deep_tunnel, green_infra, hybrid]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 导入自定义模块
sys.path.insert(0, str(Path(__file__).parent))

from resilience_framework import (
    ResilienceMetrics, ResilienceFailureCurve,
    EquityAnalyzer, SafetyParadoxDetector,
    CounterfactualManager, FloodSimResult
)

DATA_DIR = Path(r"c:\Users\Peiyuan Lin_SYSU\Desktop\学会\data")
OUTPUT_DIR = Path(r"c:\Users\Peiyuan Lin_SYSU\Desktop\学会\output")
for d in [DATA_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 默认配置
# ============================================================

DEFAULT_CONFIG = {
    "cities": ["Guangzhou", "Chicago", "Tokyo", "KualaLumpur"],
    "tunnel_types": {
        "Guangzhou": "multi_function",
        "Chicago": "pollution_control",
        "Tokyo": "flood_control",
        "KualaLumpur": "multi_function",
    },
    "scenarios": ["historical", "ssp245_2050", "ssp585_2050", "ssp585_2080"],
    "return_periods": [2, 5, 10, 20, 50, 100, 200],
    "infrastructures": ["no_tunnel", "deep_tunnel", "green_infra", "hybrid"],
    "design_standards": {
        "Guangzhou": 50,      # 东濠涌设计标准: 50年
        "Chicago": 10,        # TARP一阶段设计: 10年
        "Tokyo": 50,          # 外郭放水路: 50年
        "KualaLumpur": 100,   # SMART: 100年
    },
}


# ============================================================
# 第1部分: 模拟结果生成器 (联结SWMM输出)
# ============================================================

def generate_synthetic_results(config):
    """
    从SWMM输出文件或合成数据生成 FloodSimResult 列表

    实际使用时, 此函数替换为从SWMM .rpt文件中解析真实模拟结果。
    """
    np.random.seed(42)
    results = []

    for city in config["cities"]:
        tunnel_type = config["tunnel_types"].get(city, "flood_control")
        design_std = config["design_standards"].get(city, 50)

        for scenario in config["scenarios"]:
            # 气候因子
            cc_factors = {
                "historical": 1.0,
                "ssp245_2050": 1.12,
                "ssp585_2050": 1.18,
                "ssp585_2080": 1.28,
            }
            cc = cc_factors.get(scenario, 1.0)

            for infra in config["infrastructures"]:
                for T in config["return_periods"]:
                    # 基准洪量 (无深隧, 历史气候)
                    base_volume = 10**(5 + np.random.normal(0, 0.3)) * T**1.3 * cc

                    if infra == "no_tunnel":
                        # 无深隧: 洪量 = 基准
                        flood_vol = base_volume
                        recovery = 24.0 + T * 0.8
                        utilization = 0.0

                    elif infra == "deep_tunnel":
                        # 深隧: 削减效果随T增大而递减
                        reduction = 0.85 / (1.0 + T / design_std)
                        # 超过设计标准后急剧下降
                        if T > design_std * 1.5:
                            reduction *= 0.3
                        flood_vol = base_volume * (1.0 - reduction)
                        recovery = (24.0 + T * 0.8) * (1.0 - reduction * 0.6)
                        utilization = min(20 + T * 0.5 * cc, 98)

                    elif infra == "green_infra":
                        # GI: 中低重现期效果较好, 极端事件下效果有限
                        reduction = 0.50 / (1.0 + T / 15)
                        flood_vol = base_volume * (1.0 - reduction)
                        recovery = (24.0 + T * 0.8) * (1.0 - reduction * 0.3)
                        utilization = 0.0

                    elif infra == "hybrid":
                        # 混合方案: 深隧 + GI 组合
                        reduction = 0.88 / (1.0 + T / (design_std * 1.2))
                        flood_vol = base_volume * (1.0 - reduction)
                        recovery = (24.0 + T * 0.8) * (1.0 - reduction * 0.7)
                        utilization = min(15 + T * 0.35 * cc, 90)

                    else:
                        continue

                    result = FloodSimResult(
                        city=city, scenario=scenario, return_period=T,
                        infrastructure=infra,
                        flood_volume_m3=flood_vol,
                        flooded_area_km2=flood_vol / 5000,
                        recovery_time_h=recovery,
                        tunnel_peak_utilization=utilization,
                        pump_energy_kwh=flood_vol * 0.02 if infra != "no_tunnel" else 0,
                    )
                    results.append(result)

    return results


# ============================================================
# 第2部分: 韧性分析主流程
# ============================================================

def run_full_resilience_analysis(config, results):
    """
    完整的韧性分析流程:
    1. UFRI 计算
    2. RFC 曲线生成
    3. 公平性分析
    4. 反事实比较
    """

    output = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "ufri_summary": None,
        "rfc_summary": None,
        "equity_results": {},
        "counterfactual_comparison": None,
        "safety_paradox": {},
    }

    # ---- 步骤1: UFRI 计算 ----
    print("\n[Step 1] Computing UFRI for all scenarios...")

    all_profiles = []
    for city in config["cities"]:
        tunnel_type = config["tunnel_types"].get(city, "flood_control")
        bench = [r for r in results
                  if r.city == city and r.scenario == "historical"
                  and r.infrastructure == "no_tunnel"]

        for bench_r in bench:
            T = bench_r.return_period
            rm = ResilienceMetrics(bench_r)

            for scenario in config["scenarios"]:
                for infra in config["infrastructures"]:
                    if infra == "no_tunnel":
                        continue
                    matches = [r for r in results
                                if r.city == city and r.scenario == scenario
                                and r.return_period == T
                                and r.infrastructure == infra]
                    if matches:
                        profile = rm.compute_ufri(matches[0])
                        profile_dict = vars(profile)
                        profile_dict["city"] = city
                        profile_dict["tunnel_type"] = tunnel_type
                        all_profiles.append(profile_dict)

    df_ufri = pd.DataFrame(all_profiles)
    output["ufri_summary"] = df_ufri

    # ---- 步骤2: RFC曲线 ----
    print("[Step 2] Generating Resilience Failure Curves...")

    rfc = ResilienceFailureCurve()
    for city in config["cities"]:
        bench = [r for r in results
                  if r.city == city and r.scenario == "historical"
                  and r.infrastructure == "no_tunnel"]

        for scenario in config["scenarios"]:
            periods = []
            ufri_vals = []
            for T in config["return_periods"]:
                bench_r = [b for b in bench if b.return_period == T]
                if not bench_r:
                    continue
                rm = ResilienceMetrics(bench_r[0])

                matches = [r for r in results
                            if r.city == city and r.scenario == scenario
                            and r.return_period == T
                            and r.infrastructure == "deep_tunnel"]
                if matches:
                    profile = rm.compute_ufri(matches[0])
                    periods.append(T)
                    ufri_vals.append(profile.ufri)

            if periods:
                rfc.add_curve(city, scenario, "deep_tunnel", periods, ufri_vals)

    output["rfc_summary"] = rfc.summary_table()

    # ---- 步骤3: 空间公平性分析 ----
    print("[Step 3] Spatial equity analysis...")

    # 使用合成空间数据进行演示
    np.random.seed(123)
    H, W = 100, 100
    for city in config["cities"]:
        flood_no = np.random.gamma(2, 0.3, (H, W))
        flood_with = flood_no * (0.3 + 0.7 * np.random.beta(2, 5, (H, W)))
        pop = np.random.lognormal(5, 1.5, (H, W))
        inc = np.random.lognormal(10, 0.8, (H, W))

        ea = EquityAnalyzer()
        change = ea.compute_flood_exposure_change(flood_no, flood_with)
        equity = ea.analyze(change, pop, inc)
        output["equity_results"][city] = equity

    # ---- 步骤4: 反事实比较 ----
    print("[Step 4] Counterfactual scenario comparison...")

    cfm = CounterfactualManager()
    cfm.define_scenario("no_tunnel", "none", 0, "Baseline: no additional investment")
    cfm.define_scenario("deep_tunnel", "grey", 500, "Deep tunnel drainage system")
    cfm.define_scenario("green_infra", "green", 350, "Green infrastructure")
    cfm.define_scenario("hybrid", "hybrid", 550, "Deep tunnel + GI hybrid")

    # 组织结果
    scenario_results = {}
    for infra in config["infrastructures"]:
        scenario_results[infra] = [r for r in results if r.infrastructure == infra]

    comparison = cfm.compare_scenarios(scenario_results)
    output["counterfactual_comparison"] = comparison

    # ---- 步骤5: 安全悖论检验 ----
    print("[Step 5] Safety paradox detection...")

    spd = SafetyParadoxDetector()
    for city in config["cities"]:
        # 合成土地利用趋势数据
        years = np.arange(1985, 2025)
        protected = 20 + 0.8 * (years - 1985) + np.random.normal(0, 1, len(years))
        control = 20 + 0.4 * (years - 1985) + np.random.normal(0, 1, len(years))

        result = spd.test_development_incentive(protected, control)
        output["safety_paradox"][city] = result

    print("\n[Analysis Complete]")
    return output


# ============================================================
# 第3部分: 可视化
# ============================================================

def generate_figures(output, output_dir=OUTPUT_DIR):
    """生成论文级别图表"""

    # 设置中文字体
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

    # ---- Figure 1: RFC 韧性失效曲线 ----
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"Guangzhou": "#2196F3", "Chicago": "#F44336",
               "Tokyo": "#4CAF50", "KualaLumpur": "#FF9800"}
    scenarios = ["historical", "ssp585_2080"]
    styles = {"historical": "-", "ssp585_2080": "--"}

    # 重建RFC获取曲线数据
    config = output["config"]
    results_list = output.get("_results", [])
    if results_list:
        rfc = ResilienceFailureCurve()
        for city in config["cities"]:
            for scenario in scenarios:
                periods = config["return_periods"]
                # 使用UFRI表重建
                ufri_df = output["ufri_summary"]
                if ufri_df is not None:
                    subset = ufri_df[(ufri_df["city"] == city) &
                                      (ufri_df["scenario"] == scenario) &
                                      (ufri_df["infrastructure"] == "deep_tunnel")]
                    if len(subset) > 0:
                        ax.semilogx(
                            subset["return_period"], subset["ufri"],
                            color=colors.get(city, "gray"),
                            linestyle=styles.get(scenario, "-"),
                            linewidth=2.5,
                            label=f"{city} ({scenario})"
                        )

    # 标注关键阈值
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(150, 0.52, "UFRI = 0.5 (critical threshold)",
            fontsize=9, color="gray", ha="right")

    ax.axvline(x=50, color="gray", linestyle=":", alpha=0.3, linewidth=1)
    ax.text(52, 0.05, "Typical\ndesign\nstandard\n(50yr)", fontsize=8, color="gray")

    ax.set_xlabel("Return Period (years)", fontsize=13, fontweight="bold")
    ax.set_ylabel("UFRI (Urban Flood Resilience Index)", fontsize=13, fontweight="bold")
    ax.set_title("Resilience Failure Curve (RFC)\nDeep Tunnel Performance Under Climate Change",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(1, 300)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_RFC_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 1 saved: {output_dir / 'fig1_RFC_curves.png'}")

    # ---- Figure 4: 减灾效益的洛伦兹曲线 ----
    fig, ax = plt.subplots(figsize=(8, 8))

    for city in config["cities"]:
        equity = output["equity_results"].get(city, {})
        lx = equity.get("lorenz_x", [])
        ly = equity.get("lorenz_y", [])
        gini = equity.get("gini_coefficient", 0)

        if lx and ly:
            ax.plot(lx, ly, color=colors.get(city, "gray"), linewidth=2,
                    label=f"{city} (Gini={gini:.3f})")

    # 45度均等线
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect equality")

    ax.set_xlabel("Cumulative Population Share", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Flood Protection Benefit Share", fontsize=13, fontweight="bold")
    ax.set_title("Lorenz Curve of Flood Protection Benefits\nSpatial Equity Across Cities",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_lorenz_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 4 saved: {output_dir / 'fig4_lorenz_curves.png'}")

    # ---- Figure 5: 反事实情景雷达图 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(projection="polar"))
    axes = axes.flatten()

    comparison = output.get("counterfactual_comparison")
    if comparison is not None:
        for idx, city in enumerate(config["cities"]):
            if idx >= 4:
                break
            ax = axes[idx]

            city_data = comparison[comparison["scenario"].isin(
                ["deep_tunnel", "green_infra", "hybrid"]
            )]

            metrics = ["ufri", "robustness", "rapidity"]
            n_metrics = len(metrics)
            angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
            angles += angles[:1]

            for infra in ["deep_tunnel", "green_infra", "hybrid"]:
                infra_data = city_data[city_data["scenario"] == infra]
                if len(infra_data) > 0:
                    row = infra_data.iloc[0]
                    values = [row.get(m, 0) for m in metrics]
                    values += values[:1]
                    ax.plot(angles, values, "o-", linewidth=2, label=infra)
                    ax.fill(angles, values, alpha=0.1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_title(city, fontsize=12, fontweight="bold")
            ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.0))

    fig.suptitle("Multi-Dimensional Resilience Comparison\nDeep Tunnel vs Green Infrastructure vs Hybrid",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_radar_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 5 saved: {output_dir / 'fig5_radar_comparison.png'}")

    return output_dir


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Deep Tunnel Resilience Analysis Pipeline"
    )
    parser.add_argument("--city", type=str, default=None,
                        help="Focus on single city")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    # 加载配置
    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config, "r") as f:
            user_config = json.load(f)
            config.update(user_config)

    if args.city:
        config["cities"] = [args.city]

    print("=" * 60)
    print("Deep Tunnel Resilience Analysis Pipeline")
    print("=" * 60)
    print(f"  Cities: {config['cities']}")
    print(f"  Scenarios: {config['scenarios']}")
    print(f"  Return Periods: {config['return_periods']}")

    # 步骤1: 生成/加载模拟结果
    print("\n[Loading] Generating simulation results...")
    results = generate_synthetic_results(config)
    print(f"  Generated {len(results)} simulation results")

    # 步骤2: 完整韧性分析
    output = run_full_resilience_analysis(config, results)
    output["_results"] = results  # 保存以便可视化使用

    # 步骤3: 打印摘要
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if output["ufri_summary"] is not None:
        df = output["ufri_summary"]
        print(f"\n[UFRI Summary]")
        for city in config["cities"]:
            city_df = df[df["city"] == city]
            city_df_hist = city_df[city_df["scenario"] == "historical"]
            city_df_585 = city_df[city_df["scenario"] == "ssp585_2080"]
            mean_ufri_hist = city_df_hist["ufri"].mean() if len(city_df_hist) else 0
            mean_ufri_585 = city_df_585["ufri"].mean() if len(city_df_585) else 0
            print(f"  {city}: UFRI(hist)={mean_ufri_hist:.3f}, "
                  f"UFRI(SSP5-8.5)={mean_ufri_585:.3f}")

    if output["rfc_summary"] is not None:
        print(f"\n[RFC Critical Thresholds]")
        rfc_df = output["rfc_summary"]
        for _, row in rfc_df.iterrows():
            print(f"  {row['city']}/{row['scenario']}: "
                  f"T_crit={row['T_crit_UFRI0.5']:.1f}yr")

    print(f"\n[Equity Analysis - Gini Coefficients]")
    for city, equity in output["equity_results"].items():
        print(f"  {city}: Gini = {equity['gini_coefficient']:.3f}")

    print(f"\n[Safety Paradox Detection]")
    for city, sp in output["safety_paradox"].items():
        print(f"  {city}: {sp['evidence']} "
              f"(slope_diff = {sp['slope_difference']:.4f})")

    # 步骤4: 生成图表
    if not args.no_figures:
        print("\n[Figures] Generating...")
        generate_figures(output)

    # 保存结果
    print(f"\n[Saving] Results to JSON...")
    save_path = OUTPUT_DIR / f"resilience_analysis_{datetime.now():%Y%m%d_%H%M%S}.json"

    # 简化输出以利于JSON序列化
    simple_output = {
        "config": output["config"],
        "timestamp": output["timestamp"],
        "ufri_summary": output["ufri_summary"].to_dict("records") if output["ufri_summary"] is not None else None,
        "rfc_summary": output["rfc_summary"].to_dict("records") if output["rfc_summary"] is not None else None,
        "equity_gini": {c: e["gini_coefficient"] for c, e in output["equity_results"].items()},
        "safety_paradox": output["safety_paradox"],
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(simple_output, f, indent=2, ensure_ascii=False, default=str)

    print(f"  Results saved: {save_path}")
    print("\n[Pipeline Complete]")


if __name__ == "__main__":
    main()
