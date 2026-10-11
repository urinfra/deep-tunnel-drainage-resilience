#!/usr/bin/env python3
"""
CMIP6 极端降水数据下载与降尺度处理脚本
=========================================
从 NASA NEX-GDDP-CMIP6 下载4个目标城市的日降水数据，
进行统计降尺度和极端降水频率分析，生成设计暴雨输入文件。

目标城市: 广州、芝加哥、东京、吉隆坡
气候情景: historical (1985-2014), SSP2-4.5 (2050s), SSP5-8.5 (2050s, 2080s)
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

# ============================================================
# 配置
# ============================================================

DATA_DIR = Path(r"c:\Users\Peiyuan Lin_SYSU\Desktop\学会\data\cmip6")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 4个目标城市坐标 [lat, lon, name]
CITIES = [
    {"name": "Guangzhou", "lat": 23.13, "lon": 113.26, "elev": 8.0},
    {"name": "Chicago", "lat": 41.88, "lon": -87.63, "elev": 182.0},
    {"name": "Tokyo", "lat": 35.68, "lon": 139.76, "elev": 40.0},
    {"name": "KualaLumpur", "lat": 3.14, "lon": 101.69, "elev": 22.0},
]

# NASA NEX-GDDP-CMIP6 可用模型
CMIP6_MODELS = [
    "ACCESS-CM2", "BCC-CSM2-MR", "CMCC-ESM2", "EC-Earth3",
    "GFDL-ESM4", "GISS-E2-1-G", "INM-CM5-0", "MIROC6",
    "MPI-ESM1-2-HR", "MRI-ESM2-0", "NorESM2-LM", "UKESM1-0-LL",
]

# 情景配置
SCENARIOS = {
    "historical": {"years": (1985, 2014), "member": "r1i1p1f1"},
    "ssp245": {"years": (2041, 2070), "member": "r1i1p1f1"},
    "ssp585": {"years": (2041, 2070), "member": "r1i1p1f1"},
    "ssp585_2080s": {"years": (2071, 2100), "member": "r1i1p1f1"},
}

# 重现期 (年)
RETURN_PERIODS = [2, 5, 10, 20, 30, 50, 100, 200]


# ============================================================
# 第1部分: NASA NEX-GDDP-CMIP6 数据获取
# ============================================================

def generate_download_urls():
    """
    生成 NASA NEX-GDDP-CMIP6 数据下载URL。

    NASA NEX-GDDP-CMIP6 数据可通过以下方式获取:
    1. Web: https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6
    2. AWS S3 (推荐批量下载): s3://nex-gddp-cmip6/NEX-GDDP-CMIP6/
    3. Google Earth Engine: 数据集ID = 'NASA/NEX-GDDP-CMIP6'

    由于数据量巨大 (~300GB 全量)，本脚本生成城市网格单元级别的下载清单。
    """
    base_url = "https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com/NEX-GDDP-CMIP6"

    downloads = []
    for city in CITIES:
        # 找到最近的网格点 (NEX-GDDP-CMIP6 分辨率为 0.25° ≈ 25km)
        grid_lat = round(city["lat"] * 4) / 4  # 四舍五入到0.25°
        grid_lon = round(city["lon"] * 4) / 4

        for model in CMIP6_MODELS:
            for scenario_key, scenario_info in SCENARIOS.items():
                variant = scenario_info["member"]
                # 构建文件路径
                if scenario_key == "historical":
                    file_pattern = f"{base_url}/{model}/historical/{variant}/pr/pr_day_{model}_historical_{variant}_{grid_lat}_{grid_lon}.nc"
                else:
                    # SSP情景: SSP2-4.5 -> ssp245, SSP5-8.5 -> ssp585
                    ssp_tag = "ssp245" if "245" in scenario_key else "ssp585"
                    year_range = f"{scenario_info['years'][0]}_{scenario_info['years'][1]}"
                    file_pattern = f"{base_url}/{model}/{ssp_tag}/{variant}/pr/pr_day_{model}_{ssp_tag}_{variant}_{grid_lat}_{grid_lon}.nc"

                downloads.append({
                    "city": city["name"],
                    "lat": grid_lat,
                    "lon": grid_lon,
                    "model": model,
                    "scenario": scenario_key,
                    "url": file_pattern,
                })

    print(f"生成了 {len(downloads)} 个下载任务")
    print(f"  {len(CITIES)} 城市 × {len(CMIP6_MODELS)} 模型 × {len(SCENARIOS)} 情景")

    return downloads


def write_download_script(downloads, output_path):
    """生成 AWS CLI 批量下载脚本"""
    lines = ["#!/bin/bash", "# NASA NEX-GDDP-CMIP6 批量下载脚本"]
    lines.append(f"# 生成时间: {datetime.now().isoformat()}")
    lines.append(f"# 总文件数: {len(downloads)}\n")
    lines.append("DATA_DIR=\"$1\"")
    lines.append("mkdir -p \"$DATA_DIR\"\n")

    for dl in downloads:
        city = dl["city"].replace(" ", "_")
        model = dl["model"]
        scenario = dl["scenario"]
        out_name = f"{city}_{model}_{scenario}_pr_day.nc"
        # AWS S3 URL
        s3_path = dl["url"].replace(
            "https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com", "s3://nex-gddp-cmip6"
        )
        lines.append(f"echo 'Downloading {city}/{model}/{scenario}...'")
        lines.append(f"aws s3 cp {s3_path} \"$DATA_DIR/{out_name}\" --no-sign-request")

    lines.append("\necho 'Download complete!'")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"下载脚本已保存: {output_path}")
    print(f"运行: bash {output_path} <目标目录>")


# ============================================================
# 第2部分: 统计降尺度与极端降水频率分析
# ============================================================

class ExtremePrecipitationAnalyzer:
    """
    极端降水频率分析器

    方法:
    1. 年最大值法 (AM - Annual Maxima)
    2. 超阈值峰量法 (POT - Peaks Over Threshold) [可选]
    3. GEV (广义极值) 分布拟合
    4. 设计暴雨量推算 (各重现期)
    """

    def __init__(self, data_dict=None):
        """
        Parameters
        ----------
        data_dict : dict, optional
            {city_name: {scenario: np.array of daily precipitation}}
        """
        self.data = data_dict or {}
        self.results = {}

    def load_netcdf_to_dataframe(self, filepath, city_name, scenario):
        """从NetCDF文件加载日降水数据"""
        try:
            import xarray as xr
            ds = xr.open_dataset(filepath)
            pr = ds["pr"].values  # kg/m2/s -> 需转换为 mm/day
            # NEX-GDDP pr 单位是 kg m-2 s-1
            pr_mm_day = pr * 86400  # 转换为 mm/day

            time = pd.date_range(
                start=str(ds.time.values[0])[:10],
                periods=len(pr_mm_day),
                freq="D"
            )

            df = pd.DataFrame({"date": time, "pr": pr_mm_day})
            df.set_index("date", inplace=True)

            if city_name not in self.data:
                self.data[city_name] = {}
            self.data[city_name][scenario] = df

            return df
        except ImportError:
            print("需要安装 xarray 和 netCDF4: pip install xarray netCDF4 h5netcdf")
            return None

    def extract_annual_maxima(self, df):
        """提取年最大值序列"""
        return df["pr"].resample("YE").max().dropna()

    def fit_gev(self, am_series):
        """
        拟合广义极值分布 (GEV)

        GEV累积分布函数:
        F(x; μ, σ, ξ) = exp{ -[1 + ξ(x-μ)/σ]^(-1/ξ) }

        返回: (mu, sigma, xi) 参数
        """
        # 使用 scipy genextreme 拟合
        # genextreme 参数化: c = -ξ (shape)
        shape, loc, scale = stats.genextreme.fit(am_series.values)
        # shape in scipy = -ξ in standard notation
        xi = -shape
        mu = loc
        sigma = scale
        return mu, sigma, xi

    def estimate_return_levels(self, mu, sigma, xi, return_periods):
        """
        基于GEV参数推算各重现期设计暴雨量

        return_level = μ + (σ/ξ) * [(-ln(1-1/T))^(-ξ) - 1]  for ξ ≠ 0
        return_level = μ - σ * ln(-ln(1-1/T))              for ξ = 0 (Gumbel)
        """
        T = np.array(return_periods, dtype=float)
        p = 1.0 - 1.0 / T

        if abs(xi) < 1e-6:
            # Gumbel case
            return_levels = mu - sigma * np.log(-np.log(p))
        else:
            return_levels = mu + (sigma / xi) * (np.power(-np.log(p), -xi) - 1.0)

        return return_levels

    def calculate_confidence_intervals(self, mu, sigma, xi, return_periods,
                                        n_bootstrap=1000, alpha=0.05):
        """Bootstrap方法计算设计暴雨量的置信区间"""
        # 从拟合的GEV中生成bootstrap样本
        shape = -xi  # scipy parameterization
        n_samples = 100  # 模拟年最大值样本量

        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            sample = stats.genextreme.rvs(shape, loc=mu, scale=sigma, size=n_samples)
            try:
                b_shape, b_loc, b_scale = stats.genextreme.fit(sample)
                b_mu, b_sigma, b_xi = b_loc, b_scale, -b_shape
                rl = self.estimate_return_levels(b_mu, b_sigma, b_xi, return_periods)
                bootstrap_estimates.append(rl)
            except Exception:
                continue

        if bootstrap_estimates:
            estimates = np.array(bootstrap_estimates)
            lower = np.percentile(estimates, 100 * alpha / 2, axis=0)
            upper = np.percentile(estimates, 100 * (1 - alpha / 2), axis=0)
            return lower, upper
        return None, None

    def analyze_all(self, return_periods=RETURN_PERIODS):
        """对全部城市×情景进行极端降水频率分析"""
        for city, scenarios in self.data.items():
            if city not in self.results:
                self.results[city] = {}

            for scenario, df in scenarios.items():
                am = self.extract_annual_maxima(df)
                if len(am) < 20:
                    print(f"  [WARN] {city}/{scenario}: sample too small ({len(am)} years)")
                    continue

                mu, sigma, xi = self.fit_gev(am)
                rl = self.estimate_return_levels(mu, sigma, xi, return_periods)
                ci_low, ci_up = self.calculate_confidence_intervals(mu, sigma, xi, return_periods)

                self.results[city][scenario] = {
                    "gev_params": {"mu": mu, "sigma": sigma, "xi": xi},
                    "return_levels": dict(zip(
                        [f"T{int(t)}" for t in return_periods], rl
                    )),
                    "ci_lower": dict(zip(
                        [f"T{int(t)}" for t in return_periods], ci_low
                    )) if ci_low is not None else None,
                    "ci_upper": dict(zip(
                        [f"T{int(t)}" for t in return_periods], ci_up
                    )) if ci_up is not None else None,
                    "n_years": len(am),
                    "mean_annual_max": float(am.mean()),
                    "trend_sen_slope": float(self._sen_slope(am)),
                }

                print(f"  [OK] {city}/{scenario}: T100={rl[6]:.1f} mm/d")

        return self.results

    def _sen_slope(self, series):
        """Sen's slope 趋势估计 (非参数)"""
        n = len(series)
        slopes = []
        for i in range(n):
            for j in range(i+1, n):
                slopes.append((series.iloc[j] - series.iloc[i]) / (j - i))
        return np.median(slopes) if slopes else np.nan

    def generate_hyetograph(self, design_rainfall, duration_hours=24, method="Chicago"):
        """
        生成设计暴雨时程分配 (设计雨型)

        Parameters
        ----------
        design_rainfall : float
            设计暴雨总量 (mm)
        duration_hours : int
            暴雨历时 (h)
        method : str
            "Chicago" - 芝加哥雨型 (基于IDF曲线的峰前峰后分配)
            "PilgrimCordery" - PC方法
            "Uniform" - 均匀分配

        Returns
        -------
        pd.Series: 时段降雨量 (mm/Δt)
        """
        delta_t = 5 / 60  # 5分钟时间步长 (h)
        n_steps = int(duration_hours / delta_t)
        t = np.arange(0, duration_hours, delta_t)

        if method == "Chicago":
            # 芝加哥雨型: r = 0.4 (典型峰前系数)
            r = 0.4  # 雨峰位置系数
            t_peak = duration_hours * r

            # 使用 I = a / (t + b)^c 形式的雨强公式
            # 这里简化为: 先构造无量纲雨型再乘以总雨量
            relative_intensity = np.ones(n_steps)
            for i in range(n_steps):
                if t[i] <= t_peak:
                    # 峰前
                    relative_intensity[i] = (1 - r) * (t[i] / t_peak) ** 0.5
                else:
                    # 峰后
                    relative_intensity[i] = r * ((duration_hours - t[i]) / (duration_hours - t_peak)) ** 1.5

            # 归一化
            relative_intensity = relative_intensity / relative_intensity.sum()
            hyetograph = relative_intensity * design_rainfall

        elif method == "Uniform":
            hyetograph = np.ones(n_steps) * design_rainfall / n_steps

        else:
            raise ValueError(f"Unknown method: {method}")

        return pd.Series(hyetograph, index=t, name="rainfall_mm")


# ============================================================
# 第3部分: 气候变化响应因子 (Climate Change Factor)
# ============================================================

def calculate_climate_change_factors(analyzer_results, baseline_scenario="historical"):
    """
    计算气候变化响应因子 (CCF):
    CCF(T, scenario) = RL(T, scenario) / RL(T, historical)

    用于将历史设计暴雨"放大"为未来气候情景下的设计暴雨
    """
    ccf_results = {}

    for city, scenarios in analyzer_results.items():
        ccf_results[city] = {}
        baseline = scenarios.get(baseline_scenario, {})
        if not baseline or "return_levels" not in baseline:
            continue

        base_rl = baseline["return_levels"]

        for scenario, result in scenarios.items():
            if scenario == baseline_scenario:
                continue
            if "return_levels" not in result:
                continue

            ccf = {}
            for period, rl_value in result["return_levels"].items():
                base_value = base_rl.get(period, None)
                if base_value and base_value > 0:
                    ccf[period] = rl_value / base_value

            ccf_results[city][scenario] = {
                "ccf": ccf,
                "mean_ccf": np.mean(list(ccf.values())),
                "max_ccf": max(list(ccf.values())) if ccf else None,
            }

    return ccf_results


# ============================================================
# 第4部分: 主程序入口
# ============================================================

def create_synthetic_demo_data():
    """
    创建用于演示的合成降水数据
    在实际使用中，此函数将被真实CMIP6数据替换

    使用各城市的真实气候特征参数生成合成日降水序列
    """
    np.random.seed(42)

    # 各城市气候参数 (年降水均值mm, 变异系数CV, 极端事件的广义极值形状参数)
    climate_params = {
        "Guangzhou": {"mean_annual": 1800, "cv": 0.25, "wet_days": 150, "shape": 0.15},
        "Chicago": {"mean_annual": 965, "cv": 0.35, "wet_days": 125, "shape": 0.05},
        "Tokyo": {"mean_annual": 1528, "cv": 0.30, "wet_days": 130, "shape": 0.10},
        "KualaLumpur": {"mean_annual": 2500, "cv": 0.20, "wet_days": 200, "shape": 0.20},
    }

    synthetic_data = {}

    for city, params in climate_params.items():
        synthetic_data[city] = {}

        for scenario_key, scenario_info in SCENARIOS.items():
            start_year, end_year = scenario_info["years"]
            n_years = end_year - start_year + 1

            # 生成降雨日序列 (Gamma分布)
            wet_prob = params["wet_days"] / 365.25
            daily_shape = 0.8
            mean_wet_day = params["mean_annual"] / params["wet_days"]
            daily_scale = mean_wet_day / daily_shape

            # 气候情景调整因子
            if scenario_key == "historical":
                cc_factor = 1.0
            elif "ssp245" in scenario_key:
                cc_factor = 1.12  # SSP2-4.5: 平均+12%
            elif "ssp585" in scenario_key and "2080" in scenario_key:
                cc_factor = 1.28  # SSP5-8.5 2080s: +28%
            else:  # ssp585 2050s
                cc_factor = 1.18  # SSP5-8.5 2050s: +18%

            # 极端降水额外放大 (极端事件比均值增加更快)
            extreme_amplification = {
                "historical": 1.0,
                "ssp245": 1.08,   # 极端降水额外+8%
                "ssp585": 1.14,   # +14%
                "ssp585_2080s": 1.22,  # +22%
            }
            ext_factor = extreme_amplification.get(scenario_key, 1.0)

            dates = pd.date_range(start=f"{start_year}-01-01",
                                  end=f"{end_year}-12-31", freq="D")

            # 基础降水生成
            rain = np.zeros(len(dates))
            is_wet = np.random.random(len(dates)) < wet_prob

            # 湿日降雨量
            wet_days_mask = is_wet
            n_wet = wet_days_mask.sum()
            # Gamma分布 + 少量极端值 (GEV)
            base_rain = np.random.gamma(daily_shape, daily_scale * cc_factor, n_wet)

            # 对部分降雨日额外放大 (模拟极端事件强度增加)
            extreme_idx = np.random.random(n_wet) < 0.05  # 5%的湿日
            base_rain[extreme_idx] *= ext_factor

            rain[wet_days_mask] = base_rain

            # 添加季节循环
            day_of_year = dates.dayofyear
            season_factor = 1.0 + 0.4 * np.sin(2 * np.pi * (day_of_year - 180) / 365)  # 夏季峰值
            rain = rain * season_factor
            rain = np.maximum(rain, 0)

            df = pd.DataFrame({"date": dates, "pr": rain})
            df.set_index("date", inplace=True)
            synthetic_data[city][scenario_key] = df

    return synthetic_data


def main():
    print("=" * 60)
    print("CMIP6 极端降水数据处理与分析")
    print("深隧韧性研究 - 阶段1: 气候数据管道")
    print("=" * 60)

    # 步骤1: 生成下载清单
    print("\n[1/4] 生成NASA NEX-GDDP-CMIP6下载清单...")
    downloads = generate_download_urls()

    # 保存下载清单
    manifest_path = DATA_DIR / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(downloads, f, indent=2, default=str)
    print(f"  下载清单已保存: {manifest_path}")

    # 生成 AWS CLI 下载脚本
    script_path = DATA_DIR / "download_cmip6.sh"
    write_download_script(downloads, script_path)

    # 步骤2: 使用合成数据进行方法演示
    print("\n[2/4] 使用合成数据进行极端降水频率分析演示...")
    print("  (注: 实际研究中使用真实CMIP6数据替换合成数据)")

    synthetic_data = create_synthetic_demo_data()

    analyzer = ExtremePrecipitationAnalyzer(synthetic_data)
    results = analyzer.analyze_all()

    # 步骤3: 计算气候变化响应因子
    print("\n[3/4] 计算气候变化响应因子 (CCF)...")
    ccf = calculate_climate_change_factors(analyzer.results)

    for city in ccf:
        print(f"\n  {city}:")
        for scenario, info in ccf[city].items():
            print(f"    {scenario}: 平均CCF={info['mean_ccf']:.3f}, "
                  f"最大CCF={info.get('max_ccf', 'N/A')}")

    # 步骤4: 生成设计暴雨输入文件
    print("\n[4/4] 生成设计暴雨时程分配...")

    # 以广州T100为例
    guangzhou_hist = results.get("Guangzhou", {}).get("historical", {})
    if guangzhou_hist and "return_levels" in guangzhou_hist:
        t100_rain = guangzhou_hist["return_levels"]["T100"]
        hyeto = analyzer.generate_hyetograph(t100_rain, duration_hours=24)
        print(f"  广州 T100 设计暴雨: {t100_rain:.1f} mm/d")
        print(f"  5min峰值雨强: {hyeto.max():.1f} mm/5min")

    # 保存完整结果
    print("\n" + "=" * 60)
    print("保存结果...")

    output = {
        "analysis_timestamp": datetime.now().isoformat(),
        "cities": [c["name"] for c in CITIES],
        "scenarios": list(SCENARIOS.keys()),
        "return_periods": RETURN_PERIODS,
        "results": {},
        "ccf": {},
    }

    # 转换numpy类型为Python原生类型
    for city, scenarios in results.items():
        output["results"][city] = {}
        for scenario, result in scenarios.items():
            output["results"][city][scenario] = {
                "gev_params": {k: float(v) for k, v in result["gev_params"].items()},
                "return_levels": {k: float(v) for k, v in result["return_levels"].items()},
                "n_years": result["n_years"],
                "mean_annual_max": float(result["mean_annual_max"]),
            }

    for city, scenarios in ccf.items():
        output["ccf"][city] = {}
        for scenario, info in scenarios.items():
            output["ccf"][city][scenario] = {
                "ccf": {k: float(v) for k, v in info["ccf"].items()},
                "mean_ccf": float(info["mean_ccf"]),
            }

    results_path = DATA_DIR / "extreme_precipitation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {results_path}")

    print("\n[OK] CMIP6 data pipeline configured successfully")
    print(f"  真实数据下载脚本: {script_path}")
    print(f"  分析结果: {results_path}")

    return output


if __name__ == "__main__":
    main()
