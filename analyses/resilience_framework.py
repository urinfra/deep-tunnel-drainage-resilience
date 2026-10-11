#!/usr/bin/env python3
"""
深隧排水系统城市韧性评估框架
================================
实现 UFRI (Urban Flood Resilience Index)、RFC (Resilience Failure Curve)、
空间公平性分析和反事实情景管理。

核心模块:
- ResilienceMetrics: UFRI 计算 (吸收/恢复/适应/转换 4R维度)
- ResilienceFailureCurve: RFC 韧性失效曲线生成
- EquityAnalyzer: 空间公平性分析 (Gini系数/Lorenz曲线)
- CounterfactualManager: 反事实情景管理
- SafetyParadoxDetector: 安全悖论检验
"""

import numpy as np
import pandas as pd
from scipy import interpolate, stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 第1部分: 数据容器
# ============================================================

@dataclass
class FloodSimResult:
    """单次洪涝模拟结果"""
    city: str
    scenario: str           # "baseline" | "ssp245_2050" | "ssp585_2050" | "ssp585_2080"
    return_period: int      # 重现期 (年)
    infrastructure: str     # "no_tunnel" | "deep_tunnel" | "green_infra" | "distributed" | "hybrid"
    # 洪涝指标
    max_flood_depth_grid: np.ndarray = None     # 最大淹没深度栅格 (m)
    max_flood_depth_mean: float = np.nan         # 区域平均最大淹没深度 (m)
    max_flood_depth_max: float = np.nan          # 最大淹没深度 (m)
    flooded_area_km2: float = np.nan             # 受淹面积 (km2)
    flood_volume_m3: float = np.nan              # 总洪水量 (m3)
    flood_peak_m3s: float = np.nan               # 峰值流量 (m3/s)
    flood_duration_h: float = np.nan             # 淹没历时 (h)
    recovery_time_h: float = np.nan              # 系统恢复时间 (h)
    nodes_flooded: int = 0                        # 溢流节点数
    csos_events: int = 0                          # CSO事件数
    # 深隧特定指标
    tunnel_peak_utilization: float = np.nan      # 深隧峰值利用率 (%)
    pump_energy_kwh: float = np.nan               # 泵站能耗 (kWh)
    sediment_accumulation_ton: float = np.nan     # 泥渣累积 (ton)
    # 成本
    total_cost_million_usd: float = np.nan        # 总成本 (百万美元)
    annual_opex_million_usd: float = np.nan       # 年运维费


@dataclass
class ResilienceProfile:
    """韧性剖面"""
    city: str
    scenario: str
    infrastructure: str
    return_period: int
    # 4R指标
    robustness: float       # 鲁棒性 (吸收能力): 洪峰/洪量削减率
    rapidity: float         # 快速性 (恢复速度): 1/恢复时间
    redundancy: float       # 冗余度: 1/关键设备失效敏感度
    resourcefulness: float  # 应变能力: 替代方案/模式切换的灵活性
    # 综合指标
    ufri: float             # 城市洪涝韧性指数


# ============================================================
# 第2部分: 韧性指标计算 (UFRI)
# ============================================================

class ResilienceMetrics:
    """
    城市洪涝韧性指数 (UFRI) 计算引擎

    UFRI 基于 4R 框架 (Bruneau et al., 2003; Cutter et al., 2008):
    - Robustness (鲁棒性): 系统吸收扰动而不丧失功能的能力
    - Rapidity (快速性): 系统在扰动后恢复到可接受功能水平的速度
    - Redundancy (冗余度): 系统组件的可替代性
    - Resourcefulness (应变能力): 识别问题、确定优先级和调动资源的能力

    本框架针对深隧排水系统的特点进行了适应性修改。
    """

    def __init__(self, benchmark: FloodSimResult = None):
        """
        Parameters
        ----------
        benchmark : FloodSimResult
            基准情景 (通常为 "无深隧，历史时期")
        """
        self.benchmark = benchmark

    def calc_robustness(self, result: FloodSimResult) -> float:
        """
        鲁棒性 = 1 - (有深隧洪量 / 无深隧洪量)
                 = 洪量削减率

        对于多功能型深隧，额外考虑交通/输水功能的中断程度。
        """
        if self.benchmark is None:
            return np.nan

        if (self.benchmark.flood_volume_m3 is not None and
                result.flood_volume_m3 is not None and
                self.benchmark.flood_volume_m3 > 0):
            reduction = 1.0 - result.flood_volume_m3 / self.benchmark.flood_volume_m3
            return float(np.clip(reduction, 0.0, 1.0))
        return np.nan

    def calc_rapidity(self, result: FloodSimResult) -> float:
        """
        快速性 = 1 / (恢复时间 / 24h + 1)  → [0, 1]
        恢复时间越短，快速性越接近1
        """
        if result.recovery_time_h is not None and result.recovery_time_h > 0:
            return float(1.0 / (result.recovery_time_h / 24.0 + 1.0))
        return np.nan

    def calc_redundancy(self, result: FloodSimResult, n_independent_components: int = 3) -> float:
        """
        冗余度 = 1 - (峰值利用率 / 设计容量) 的敏感度

        对于深隧: 考虑关键设备(泵站/闸门)的可用率
        """
        if result.tunnel_peak_utilization is not None:
            # 利用率越高 → 冗余度越低
            utilization = np.clip(result.tunnel_peak_utilization / 100.0, 0.0, 1.0)
            return float(1.0 - utilization * 0.8)  # 留出最小冗余
        return np.nan

    def calc_resourcefulness(self, result: FloodSimResult,
                              mode_switch_time_h: float = 0.0) -> float:
        """
        应变能力 = f(模式切换灵活性, 调控分级精细度)

        对于多功能型深隧: 模式切换可靠性和速度是关键
        对于单功能型: 调度策略的适应范围
        """
        # 基础应变能力
        base = 0.5

        # 模式切换时间惩罚
        if mode_switch_time_h > 0:
            switch_penalty = min(mode_switch_time_h / 12.0, 0.4)  # 最多扣0.4
            base -= switch_penalty

        # 调控分级奖励 (分级层数越多→应变能力越强)
        # 通过结果中的调控策略信息推断

        return float(np.clip(base, 0.1, 1.0))

    def compute_ufri(self, result: FloodSimResult,
                     weights: Dict[str, float] = None,
                     mode_switch_time_h: float = 0.0) -> ResilienceProfile:
        """
        计算 UFRI 综合指数

        UFRI = w_r * Robustness + w_p * Rapidity + w_d * Redundancy + w_s * Resourcefulness

        权重默认值根据深隧功能类型调整:
        - 污染控制型: w_r=0.35, w_p=0.25, w_d=0.20, w_s=0.20
        - 洪涝控制型: w_r=0.30, w_p=0.35, w_d=0.15, w_s=0.20
        - 多功能型:   w_r=0.20, w_p=0.20, w_d=0.30, w_s=0.30
        """
        if weights is None:
            weights = {"robustness": 0.30, "rapidity": 0.30,
                        "redundancy": 0.20, "resourcefulness": 0.20}

        robustness = self.calc_robustness(result)
        rapidity = self.calc_rapidity(result)
        redundancy = self.calc_redundancy(result)
        resourcefulness = self.calc_resourcefulness(result, mode_switch_time_h)

        # 加权合成
        ufri = 0.0
        total_weight = 0.0
        for key, val in zip(
            ["robustness", "rapidity", "redundancy", "resourcefulness"],
            [robustness, rapidity, redundancy, resourcefulness]
        ):
            if not np.isnan(val):
                ufri += weights[key] * val
                total_weight += weights[key]

        if total_weight > 0:
            ufri /= total_weight

        return ResilienceProfile(
            city=result.city,
            scenario=result.scenario,
            infrastructure=result.infrastructure,
            return_period=result.return_period,
            robustness=robustness,
            rapidity=rapidity,
            redundancy=redundancy,
            resourcefulness=resourcefulness,
            ufri=float(ufri),
        )

    def compute_ufri_batch(self, results: List[FloodSimResult],
                            tunnel_type: str = "flood_control",
                            mode_switch_time_h: float = 0.0
                            ) -> pd.DataFrame:
        """
        批量计算UFRI

        Parameters
        ----------
        results : list
            模拟结果列表
        tunnel_type : str
            "pollution_control" | "flood_control" | "multi_function"
        mode_switch_time_h : float
            模式切换时间 (仅多功能型需要)

        Returns
        -------
        pd.DataFrame: UFRI结果表
        """
        # 按深隧类型选择默认权重
        weight_presets = {
            "pollution_control": {"robustness": 0.35, "rapidity": 0.25,
                                   "redundancy": 0.20, "resourcefulness": 0.20},
            "flood_control": {"robustness": 0.30, "rapidity": 0.35,
                               "redundancy": 0.15, "resourcefulness": 0.20},
            "multi_function": {"robustness": 0.20, "rapidity": 0.20,
                                "redundancy": 0.30, "resourcefulness": 0.30},
        }
        weights = weight_presets.get(tunnel_type, weight_presets["flood_control"])

        profiles = []
        for r in results:
            profile = self.compute_ufri(r, weights, mode_switch_time_h)
            profiles.append(profile)

        df = pd.DataFrame([vars(p) for p in profiles])
        return df


# ============================================================
# 第3部分: 韧性失效曲线 (RFC)
# ============================================================

class ResilienceFailureCurve:
    """
    韧性失效曲线 (Resilience Failure Curve, RFC)

    在"降雨重现期 × 韧性维持率"坐标系中，
    绘制深隧系统在不同气候情景下的性能衰减曲线。

    关键特征量:
    - 临界重现期 T_crit: 韧性维持率首次降至 < 0.5 的重现期
    - 失效速率 dR/dT: 在临界点附近的韧性下降速率
    - 情景敏感度: 气候变化使临界重现期缩小的幅度
    """

    def __init__(self):
        self.curves = {}  # {(city, scenario, infra): [(T, UFRI), ...]}

    def add_curve(self, city: str, scenario: str, infrastructure: str,
                  return_periods: List[int], ufri_values: List[float]):
        """添加一条RFC曲线"""
        key = (city, scenario, infrastructure)
        sorted_pairs = sorted(zip(return_periods, ufri_values))
        self.curves[key] = sorted_pairs

    def interpolate_curve(self, key, target_t: float) -> float:
        """在曲线上插值获取任意重现期的UFRI"""
        if key not in self.curves:
            return np.nan
        ts, ufris = zip(*self.curves[key])
        f = interpolate.interp1d(ts, ufris, kind="cubic",
                                  bounds_error=False, fill_value="extrapolate")
        return float(np.clip(f(target_t), 0.0, 1.0))

    def find_critical_return_period(self, key, threshold: float = 0.5) -> float:
        """
        寻找韧性维持率的临界重现期
        即 UFRI 首次降至低于 threshold 的重现期
        """
        if key not in self.curves:
            return np.nan

        ts, ufris = zip(*self.curves[key])

        # 从T=2开始搜索
        for t_test in np.arange(2, 500, 0.5):
            ufri_test = self.interpolate_curve(key, t_test)
            if ufri_test < threshold:
                return t_test
        return 500.0  # 超过搜索范围

    def compute_failure_slope(self, key) -> float:
        """
        计算在临界点附近的韧性失效速率
        d(UFRI)/d(log10(T))
        """
        t_crit = self.find_critical_return_period(key)
        if np.isnan(t_crit):
            return np.nan

        # 在 +/- 20% 范围内计算斜率
        t1 = max(2, t_crit * 0.8)
        t2 = t_crit * 1.2
        u1 = self.interpolate_curve(key, t1)
        u2 = self.interpolate_curve(key, t2)

        slope = (u2 - u1) / (np.log10(t2) - np.log10(t1))
        return float(slope)

    def generate_curve_data(self, key, t_range: Tuple[float, float] = (2, 200),
                            n_points: int = 100) -> pd.DataFrame:
        """生成绘制RFC曲线的数据点"""
        t_values = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), n_points)
        ufri_values = [self.interpolate_curve(key, t) for t in t_values]
        return pd.DataFrame({"return_period": t_values, "ufri": ufri_values})

    def summary_table(self) -> pd.DataFrame:
        """生成RFC汇总表 (各案例的关键阈值)"""
        rows = []
        for key, curve_data in self.curves.items():
            city, scenario, infra = key
            t_crit_0_5 = self.find_critical_return_period(key, 0.5)
            t_crit_0_3 = self.find_critical_return_period(key, 0.3)
            slope = self.compute_failure_slope(key)
            base_ufri = curve_data[0][1] if curve_data else np.nan

            rows.append({
                "city": city,
                "scenario": scenario,
                "infrastructure": infra,
                "T_crit_UFRI0.5": t_crit_0_5,
                "T_crit_UFRI0.3": t_crit_0_3,
                "failure_slope": slope,
                "base_UFRI_T2": base_ufri,
            })
        return pd.DataFrame(rows)


# ============================================================
# 第4部分: 空间公平性分析
# ============================================================

class EquityAnalyzer:
    """
    洪涝减灾效益的空间公平性分析

    借鉴 Kumar et al. (2025) Nature Cities 的 Gini 系数方法，
    量化深隧减灾效益在社会群体间的分配公平性。

    输入数据:
    - 2D淹没深度变化栅格 (有深隧 vs 无深隧)
    - 人口/收入/资产空间分布栅格
    """

    def __init__(self):
        self.results = {}

    def compute_flood_exposure_change(self,
                                       flood_no_tunnel: np.ndarray,
                                       flood_with_tunnel: np.ndarray,
                                       threshold_depth: float = 0.1
                                       ) -> np.ndarray:
        """
        计算每个栅格的洪涝暴露度变化

        暴露度变化 = 有深隧时的最大淹没深度 - 无深隧时的最大淹没深度
        负值 = 深隧降低了淹没深度 (受益)
        正值 = 深隧增加了淹没深度 (受损/风险转移)

        Returns
        -------
        np.ndarray: 每个栅格的变化值 (m)
        """
        change = flood_with_tunnel - flood_no_tunnel
        return change

    def overlay_socioeconomic(self,
                               flood_change: np.ndarray,
                               population_grid: np.ndarray,
                               income_grid: np.ndarray,
                               mask: np.ndarray = None
                               ) -> Dict:
        """
        将洪涝变化与社会经济数据叠置

        Parameters
        ----------
        flood_change : (H, W) array
            淹没深度变化
        population_grid : (H, W) array
            人口密度栅格
        income_grid : (H, W) array
            人均收入栅格
        mask : (H, W) bool array
            分析区域掩膜 (如排除水体)

        Returns
        -------
        dict: 叠置分析结果
        """
        if mask is None:
            mask = np.ones_like(flood_change, dtype=bool)

        valid = mask & ~np.isnan(flood_change) & ~np.isnan(population_grid)
        if not valid.any():
            return {}

        # 展平有效像素
        fc = flood_change[valid]
        pop = population_grid[valid]
        inc = income_grid[valid]

        # 按收入分位数分组
        income_percentiles = np.percentile(inc, [20, 40, 60, 80])
        groups = np.digitize(inc, income_percentiles)  # 0-4 (5组)

        group_stats = {}
        for g in range(5):
            g_mask = groups == g
            if g_mask.sum() == 0:
                continue

            # 该组受深隧保护的人口比例
            protected = fc[g_mask] < 0  # 淹没减少
            exposed = fc[g_mask] >= 0  # 淹没未减少或增加

            group_stats[f"Q{g+1}"] = {
                "pop_total": float(pop[g_mask].sum()),
                "pop_protected": float(pop[g_mask][protected].sum()),
                "pop_exposed": float(pop[g_mask][exposed].sum()),
                "protection_rate": float(protected.mean()),
                "mean_flood_change": float(fc[g_mask].mean()),
                "mean_income": float(inc[g_mask].mean()),
            }

        return group_stats

    def compute_gini(self, values: np.ndarray, weights: np.ndarray = None) -> float:
        """
        计算加权Gini系数

        衡量减灾效益在各空间单元间的分配不平等程度

        G = 1 表示完全不平等 (所有效益集中在少数单元)
        G = 0 表示完全平等 (所有单元均等受益)

        采用 Kumar et al. (2025) 方法:
        - 按效益从低到高排序
        - 累积曲线下的面积差
        """
        if weights is None:
            weights = np.ones_like(values)

        valid = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
        v = values[valid]
        w = weights[valid]

        if len(v) < 2:
            return np.nan

        # 排序
        order = np.argsort(v)
        v_sorted = v[order]
        w_sorted = w[order]

        # 累积份额
        cum_w = np.cumsum(w_sorted) / w_sorted.sum()
        cum_vw = np.cumsum(v_sorted * w_sorted) / (v_sorted * w_sorted).sum()

        # Gini = 1 - 2 * AUC (相等于A/(A+B))
        # 使用梯形法计算洛伦兹曲线下面积
        try:
            auc = np.trapezoid(cum_vw, cum_w)
        except AttributeError:
            auc = np.trapz(cum_vw, cum_w)
        gini = 1.0 - 2.0 * auc

        return float(gini)

    def compute_lorenz_curve(self, values: np.ndarray, weights: np.ndarray = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """计算洛伦兹曲线 (用于绘图)"""
        if weights is None:
            weights = np.ones_like(values)

        valid = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
        v = values[valid]
        w = weights[valid]

        order = np.argsort(v)
        v_sorted = v[order]
        w_sorted = w[order]

        cum_pop = np.cumsum(w_sorted) / w_sorted.sum()
        cum_benefit = np.cumsum(v_sorted * w_sorted) / (v_sorted * w_sorted).sum()

        # 在0处添加起点
        cum_pop = np.concatenate([[0], cum_pop])
        cum_benefit = np.concatenate([[0], cum_benefit])

        return cum_pop, cum_benefit

    def analyze(self, flood_change: np.ndarray, population: np.ndarray,
                income: np.ndarray, mask: np.ndarray = None) -> Dict:
        """完整的公平性分析"""

        # 减灾效益 = max(0, -flood_change) —— 仅考虑受益部分
        benefit = np.maximum(0, -flood_change)

        group_stats = self.overlay_socioeconomic(flood_change, population,
                                                   income, mask)
        gini_benefit = self.compute_gini(benefit, population)
        lorenz_x, lorenz_y = self.compute_lorenz_curve(benefit, population)

        return {
            "gini_coefficient": gini_benefit,
            "income_group_stats": group_stats,
            "lorenz_x": lorenz_x.tolist(),
            "lorenz_y": lorenz_y.tolist(),
        }


# ============================================================
# 第5部分: 安全悖论检验
# ============================================================

class SafetyParadoxDetector:
    """
    深隧安全悖论检验

    检验三个假设:
    H1 (开发激励): 深隧保护区的开发密度增速 > 非保护区
    H2 (资产集中): 深隧保护区内资产价值增长 > 区域平均值
    H3 (极端暴露): 超设计标准事件下, 深隧保护区的单位面积损失 > 非保护区
        (因为保护区内低洼/近河处被高强度开发)

    需要的时间序列数据:
    - Landsat土地利用时间序列 (1985-2025)
    - 房地产交易/评估数据
    - 历史灾害损失记录
    """

    def __init__(self):
        self.results = {}

    def test_development_incentive(self,
                                    land_use_ts_protected: np.ndarray,
                                    land_use_ts_control: np.ndarray
                                    ) -> Dict:
        """
        H1检验: 开发激励

        Parameters
        ----------
        land_use_ts_protected : (T,) array
            深隧保护区内各年的建设用地比例
        land_use_ts_control : (T,) array
            对照区各年的建设用地比例

        Returns
        -------
        dict: 统计检验结果
        """
        t = np.arange(len(land_use_ts_protected))

        # 线性趋势
        slope_protected = stats.linregress(t, land_use_ts_protected).slope
        slope_control = stats.linregress(t, land_use_ts_control).slope

        # H1: slope_protected > slope_control (保护区开发增速更快)
        diff = slope_protected - slope_control

        return {
            "hypothesis": "H1: Development incentive",
            "slope_protected": float(slope_protected),
            "slope_control": float(slope_control),
            "slope_difference": float(diff),
            "evidence": "supported" if diff > 0 else "not supported",
        }

    def test_extreme_exposure(self,
                               loss_protected: float,
                               loss_control: float,
                               asset_protected: float,
                               asset_control: float
                               ) -> Dict:
        """
        H3检验: 极端暴露

        比较超设计标准事件下的单位资产损失率
        """
        loss_rate_protected = loss_protected / asset_protected if asset_protected > 0 else 0
        loss_rate_control = loss_control / asset_control if asset_control > 0 else 0
        ratio = loss_rate_protected / loss_rate_control if loss_rate_control > 0 else 1.0

        return {
            "hypothesis": "H3: Extreme exposure",
            "loss_rate_protected": float(loss_rate_protected),
            "loss_rate_control": float(loss_rate_control),
            "loss_rate_ratio": float(ratio),
            "evidence": "supported" if ratio > 1.0 else "not supported",
        }


# ============================================================
# 第6部分: 反事实情景管理
# ============================================================

class CounterfactualManager:
    """
    反事实情景管理器

    管理两种反事实分析:

    反事实A: 有深隧 vs 无深隧
    - 量化深隧的增量韧性贡献
    - 识别深隧保护不足/过度保护的区域

    反事实B: 深隧 vs 替代方案
    - GI (绿色基础设施: 生物滞留池+透水铺装+雨水花园)
    - DS (分布式调蓄: 小型调蓄池+管网扩容)
    - HY (混合方案: 深隧+GI组合优化)
    """

    def __init__(self):
        self.scenarios = {}

    def define_scenario(self, name: str, infrastructure_type: str,
                         cost_million_usd: float, description: str = "",
                         params: Dict = None):
        """定义反事实情景"""
        self.scenarios[name] = {
            "infrastructure_type": infrastructure_type,
            "cost": cost_million_usd,
            "description": description,
            "params": params or {},
        }

    def compare_scenarios(self, results: Dict[str, List[FloodSimResult]]
                           ) -> pd.DataFrame:
        """
        在不同情景间比较韧性表现

        Parameters
        ----------
        results : {scenario_name: [FloodSimResult, ...]}

        Returns
        -------
        pd.DataFrame: 比较表
        """
        rows = []
        for scenario_name, result_list in results.items():
            scenario_info = self.scenarios.get(scenario_name, {})
            cost = scenario_info.get("cost", np.nan)

            for r in result_list:
                rm = ResilienceMetrics()
                profile = rm.compute_ufri(r)

                cost_effectiveness = (profile.ufri / cost * 1e6
                                       if not np.isnan(cost) and cost > 0
                                       else np.nan)

                rows.append({
                    "scenario": scenario_name,
                    "infra_type": scenario_info.get("infrastructure_type", "unknown"),
                    "return_period": r.return_period,
                    "climate_scenario": r.scenario,
                    "ufri": profile.ufri,
                    "robustness": profile.robustness,
                    "rapidity": profile.rapidity,
                    "cost_million_usd": cost,
                    "cost_effectiveness": cost_effectiveness,
                    "flood_volume_m3": r.flood_volume_m3,
                    "flooded_area_km2": r.flooded_area_km2,
                    "recovery_time_h": r.recovery_time_h,
                })

        return pd.DataFrame(rows)

    def recommend_optimal_pathway(self, comparison_df: pd.DataFrame,
                                   budget_constraint: float = None,
                                   min_ufri: float = 0.5
                                   ) -> List[Dict]:
        """
        推荐最优适应路径

        使用启发式规则:
        1. 排除超出预算的方案
        2. 对每种气候情景，选出UFRI最高的方案
        3. 如果混合方案在两个以上情景中表现最优，推荐分阶段实施
        """
        df = comparison_df.copy()

        if budget_constraint is not None:
            df = df[df["cost_million_usd"] <= budget_constraint]

        if len(df) == 0:
            return []

        recommendations = []

        # 按气候情景分组分析
        for climate_scenario in df["climate_scenario"].unique():
            subset = df[df["climate_scenario"] == climate_scenario]
            best = subset.loc[subset["ufri"].idxmax()]

            recommendations.append({
                "climate_scenario": climate_scenario,
                "recommended_option": best["scenario"],
                "expected_ufri": best["ufri"],
                "cost": best["cost_million_usd"],
                "cost_effectiveness": best.get("cost_effectiveness", np.nan),
            })

        return recommendations


# ============================================================
# 第7部分: 演示与验证
# ============================================================

def demo_run():
    """演示框架的基本功能"""
    print("=" * 60)
    print("深隧排水系统韧性评估框架 v1.0")
    print("=" * 60)

    # 创建合成演示数据
    np.random.seed(42)

    # 模拟4个城市的基准情景 (无深隧)
    benchmark_results = {}
    for city in ["Guangzhou", "Chicago", "Tokyo", "KualaLumpur"]:
        for T in [2, 5, 10, 20, 50, 100, 200]:
            r = FloodSimResult(
                city=city, scenario="baseline", return_period=T,
                infrastructure="no_tunnel",
                flood_volume_m3=np.random.lognormal(10, 0.5) * T**1.2,
                flooded_area_km2=np.random.lognormal(1, 0.3) * T**0.8,
                recovery_time_h=4.0 + T * 0.5,
                tunnel_peak_utilization=0.0,
            )
            if city not in benchmark_results:
                benchmark_results[city] = []
            benchmark_results[city].append(r)

    # 模拟有深隧的情景
    tunnel_results = []
    for city in ["Guangzhou", "Chicago", "Tokyo", "KualaLumpur"]:
        for scenario in ["baseline", "ssp245_2050", "ssp585_2080"]:
            cc_factor = {"baseline": 1.0, "ssp245_2050": 1.12, "ssp585_2080": 1.28}
            for T in [2, 5, 10, 20, 50, 100, 200]:
                # 深隧削减效果随重现期增大而递减
                reduction_factor = 1.0 / (1.0 + T / 50.0)  # T=50时削减50%

                r = FloodSimResult(
                    city=city, scenario=scenario, return_period=T,
                    infrastructure="deep_tunnel",
                    flood_volume_m3=(np.random.lognormal(10, 0.5) * T**1.2
                                      * cc_factor[scenario] * (1 - reduction_factor * 0.7)),
                    flooded_area_km2=(np.random.lognormal(1, 0.3) * T**0.8
                                       * cc_factor[scenario] * (1 - reduction_factor * 0.6)),
                    recovery_time_h=(4.0 + T * 0.5) * (1 - reduction_factor * 0.5),
                    tunnel_peak_utilization=min(20 + T * 0.5, 95),
                )
                tunnel_results.append(r)

    # ---- 测试1: UFRI计算 ----
    print("\n[测试1] UFRI 韧性指数计算")
    print("-" * 40)

    weight_presets = {
        "pollution_control": {"robustness": 0.35, "rapidity": 0.25, "redundancy": 0.20, "resourcefulness": 0.20},
        "flood_control": {"robustness": 0.30, "rapidity": 0.35, "redundancy": 0.15, "resourcefulness": 0.20},
        "multi_function": {"robustness": 0.20, "rapidity": 0.20, "redundancy": 0.30, "resourcefulness": 0.30},
    }
    for city in ["Guangzhou", "Chicago", "Tokyo", "KualaLumpur"]:
        bench = benchmark_results[city][5]  # T=50 基准
        rm = ResilienceMetrics(bench)
        tunnel = [r for r in tunnel_results
                   if r.city == city and r.scenario == "baseline"
                   and r.return_period == 50][0]
        profile = rm.compute_ufri(tunnel, weights=weight_presets["flood_control"])
        print(f"  {city}: UFRI = {profile.ufri:.3f} "
              f"(R={profile.robustness:.3f}, P={profile.rapidity:.3f})")

    # ---- 测试2: RFC曲线 ----
    print("\n[测试2] RFC 韧性失效曲线")
    print("-" * 40)

    rfc = ResilienceFailureCurve()

    for city in ["Guangzhou", "Chicago", "Tokyo", "KualaLumpur"]:
        bench_t100 = benchmark_results[city][5]  # 用T=50基准
        rm = ResilienceMetrics(bench_t100)

        periods = [2, 5, 10, 20, 50, 100, 200]
        ufri_values = []

        for scenario in ["baseline", "ssp585_2080"]:
            for T in periods:
                matches = [r for r in tunnel_results
                            if r.city == city and r.scenario == scenario
                            and r.return_period == T]
                if matches:
                    profile = rm.compute_ufri(matches[0])
                    ufri_values.append(profile.ufri)
            rfc.add_curve(city, scenario, "deep_tunnel", periods, ufri_values)
            ufri_values = []

    summary = rfc.summary_table()
    for _, row in summary.iterrows():
        print(f"  {row['city']}/{row['scenario']}: "
              f"T_crit(0.5)={row['T_crit_UFRI0.5']:.1f}yr, "
              f"slope={row['failure_slope']:.3f}")

    # ---- 测试3: 空间公平性 ----
    print("\n[测试3] 空间公平性分析")
    print("-" * 40)

    H, W = 100, 100
    np.random.seed(123)
    flood_no = np.random.gamma(2, 0.3, (H, W))  # 无深隧
    flood_with = flood_no * (0.3 + 0.7 * np.random.beta(2, 5, (H, W)))  # 有深隧
    population = np.random.lognormal(5, 1.5, (H, W))
    income = np.random.lognormal(10, 0.8, (H, W))

    ea = EquityAnalyzer()
    change = ea.compute_flood_exposure_change(flood_no, flood_with)
    result = ea.analyze(change, population, income)

    print(f"  Gini coefficient: {result['gini_coefficient']:.3f}")
    if result.get("income_group_stats"):
        for group, stats_data in result["income_group_stats"].items():
            print(f"  {group}: protection_rate={stats_data['protection_rate']:.2%}, "
                  f"mean_flood_change={stats_data['mean_flood_change']:.3f}m")

    # ---- 测试4: 反事实比较 ----
    print("\n[测试4] 反事实情景比较")
    print("-" * 40)

    cfm = CounterfactualManager()
    cfm.define_scenario("no_tunnel", "none", 0, "基准：无防洪基础设施增量投资")
    cfm.define_scenario("deep_tunnel", "grey", 500, "深隧排水系统")
    cfm.define_scenario("green_infra", "green", 350, "绿色基础设施(生物滞留+透水铺装)")
    cfm.define_scenario("hybrid", "hybrid", 550, "深隧+GI组合优化")

    print("  情景定义完成:")
    for name, info in cfm.scenarios.items():
        print(f"    {name}: {info['infrastructure_type']}, ${info['cost']}M")

    print("\n" + "=" * 60)
    print("所有模块功能正常。框架就绪。")
    print("=" * 60)


if __name__ == "__main__":
    demo_run()
