"""
交互式图表编辑器 — Fig 1 RFC Curves
=====================================
Flask web app: 实时预览 + 参数调整 + 自动生成Python代码
启动: python figure_editor.py
访问: http://localhost:5100
"""
import sys, os, json, io, base64, copy
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from flask import Flask, render_template_string, request, jsonify

# ============================================================
# Paths & Data (mirrors update_figures.py)
# ============================================================
BASE = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE / "Inputs"
OUTPUT_DIR = BASE / "Outputs"
FIG_DIR = OUTPUT_DIR / "figures"

# ============================================================
# Default Parameters (extracted from fig1_rfc_updated)
# ============================================================
DEFAULT_PARAMS = {
    # Figure layout
    "figsize_w": 16,
    "figsize_h": 14,
    "suptitle_fontsize": 16,
    "suptitle_y": 1.01,
    "dpi": 150,

    # Font sizes
    "font_family": "Arial",
    "base_fontsize": 18,
    "title_fontsize": 20,
    "label_fontsize": 18,
    "tick_fontsize": 16,
    "legend_fontsize": 11.5,
    "legend_loc": "lower left",
    "legend_framealpha": 0.8,

    # RFC Curves
    "rfc_linewidth": 2.2,
    "rfc_markersize": 7,
    "rfc_markeredgewidth": 0.8,

    # Design standard line
    "design_line_color": "gray",
    "design_line_linestyle": "--",
    "design_line_alpha": 0.6,
    "design_line_width": 1.2,
    "design_text_fontsize": 10,
    "design_text_color": "gray",
    "design_text_y": 0.08,

    # UFRI threshold line
    "threshold_color": "red",
    "threshold_linestyle": ":",
    "threshold_alpha": 0.4,
    "threshold_linewidth": 1,

    # Guangzhou observed data
    "obs_marker_color": "#E74C3C",
    "obs_marker": "x",
    "obs_marker_size": 60,
    "obs_marker_alpha": 0.7,
    "obs_linewidths": 1.5,
    "obs_n_points": 30,
    "obs_label": "Observed (2022-2025)",

    # Max observed annotation
    "maxobs_fontsize": 10,
    "maxobs_xytext_x_mult": 2.5,
    "maxobs_xytext_y_offset": 0.08,

    # Calibration box
    "calib_fontsize": 11,
    "calib_x": 0.98,
    "calib_y": 0.20,
    "calib_box_color": "#FFF3CD",

    # T_crit annotations
    "tcrit_fontsize": 11,
    "tcrit_y_base": 0.42,
    "tcrit_y_step": 0.08,
    "tcrit_x_mult": 0.55,

    # Axis
    "xlim_min": 1.5,
    "xlim_max": 250,
    "ylim_min": 0,
    "ylim_max": 1.05,
    "grid_alpha": 0.15,
    "grid_which": "both",

    # Scenario colors
    "color_Historical": "#2C3E50",
    "color_SSP245_2050s": "#3498DB",
    "color_SSP585_2050s": "#E67E22",
    "color_SSP585_2080s": "#E74C3C",

    # City colors (for title bars)
    "color_Guangzhou": "#E74C3C",
    "color_Chicago": "#3498DB",
    "color_Tokyo": "#2ECC71",
    "color_KualaLumpur": "#F39C12",
}

# ============================================================
# Model (mirrors update_figures.py)
# ============================================================
CITY_PARAMS = {
    "Guangzhou": {"name_cn": "Guangzhou", "tunnel": "Donghaochong",
        "type": "Multi-function", "design_std_yr": 50,
        "tunnel_length_km": 1.77, "tunnel_diameter_m": 5.3,
        "storage_m3": 100000, "roughness_n": 0.025, "color": "#E74C3C"},
    "Chicago": {"name_cn": "Chicago", "tunnel": "TARP",
        "type": "Pollution control", "design_std_yr": 10,
        "tunnel_length_km": 176, "tunnel_diameter_m": 8.0,
        "storage_m3": 8.7e6, "roughness_n": 0.025, "color": "#3498DB"},
    "Tokyo": {"name_cn": "Tokyo", "tunnel": "Outer Discharge Channel",
        "type": "Flood control", "design_std_yr": 50,
        "tunnel_length_km": 6.3, "tunnel_diameter_m": 10.6,
        "storage_m3": 540000, "roughness_n": 0.018, "color": "#2ECC71"},
    "KualaLumpur": {"name_cn": "Kuala Lumpur", "tunnel": "SMART",
        "type": "Multi-function", "design_std_yr": 100,
        "tunnel_length_km": 9.7, "tunnel_diameter_m": 11.83,
        "storage_m3": 600000, "roughness_n": 0.025, "color": "#F39C12"},
}

RETURN_PERIODS = [2, 5, 10, 20, 30, 50, 100, 200]
SCENARIO_KEYS = [
    ("Historical", 1.0),
    ("SSP2-4.5 2050s", 1.12),
    ("SSP5-8.5 2050s", 1.20),
    ("SSP5-8.5 2080s", 1.30),
]
CITY_KEYS = ["Guangzhou", "Chicago", "Tokyo", "KualaLumpur"]


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
    if infrastructure == "deep_tunnel":
        robustness = eta
        rapidity_base = 0.85 if rp <= cp["design_std_yr"] else 0.85 * (cp["design_std_yr"] / rp) ** 0.4
        redundancy = max(0.1, 1.0 - 0.8 * (rp / (cp["design_std_yr"] * 3)))
        resourcefulness = 0.7 if cp["type"] == "Multi-function" else 0.5
    elif infrastructure == "green_infra":
        robustness = eta * 0.9 * np.exp(-rp / 30)
        rapidity_base = 0.7; redundancy = 0.4; resourcefulness = 0.8
    elif infrastructure == "distributed":
        robustness = eta * 0.75; rapidity_base = 0.6; redundancy = 0.5; resourcefulness = 0.6
    elif infrastructure == "hybrid":
        robustness = eta * 1.05
        rapidity_base = 0.9 if rp <= cp["design_std_yr"] else 0.9 * (cp["design_std_yr"] / rp) ** 0.3
        redundancy = max(0.2, 1.0 - 0.6 * (rp / (cp["design_std_yr"] * 4)))
        resourcefulness = 0.85
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


def load_observed_gz_storage():
    all_storage = []
    for f in ['深隧数据2022.xlsx', '深隧数据2023.xlsx', '深隧数据2024.xlsx', '深隧数据2025.xlsx']:
        path = INPUT_DIR / f
        if not path.exists():
            continue
        xl = pd.ExcelFile(path)
        sheet = [s for s in xl.sheet_names if '峰值蓄水量' in s][0]
        df = pd.read_excel(path, sheet_name=sheet)
        for _, row in df.iterrows():
            try:
                val = float(row.iloc[1])
                if not pd.isna(val) and val > 0:
                    all_storage.append(val)
            except:
                continue
    return np.array(sorted(all_storage))


def observed_return_period_estimate(storage_values, n_years=4):
    sorted_vals = np.sort(storage_values)[::-1]
    n = len(sorted_vals)
    ranks = np.arange(1, n + 1)
    rp = (n_years + 0.12) / (ranks - 0.44)
    return sorted_vals, rp


def gz_observed_ufri_estimate(storage_values, design_storage=100000):
    ufri_est = 1.0 - 0.6 * (storage_values / design_storage)
    return np.clip(ufri_est, 0.0, 1.0)


# ============================================================
# Figure Generation with Parameter Overrides
# ============================================================
def generate_fig1(params):
    """Generate Fig 1 with given parameters, return base64 PNG"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [params.get("font_family", "Arial"), "DejaVu Sans"],
        "font.size": params.get("base_fontsize", 18),
        "axes.titlesize": params.get("title_fontsize", 20),
        "axes.labelsize": params.get("label_fontsize", 18),
        "xtick.labelsize": params.get("tick_fontsize", 16),
        "ytick.labelsize": params.get("tick_fontsize", 16),
        "legend.fontsize": params.get("legend_fontsize", 14),
        "figure.dpi": params.get("dpi", 150),
    })

    fig, axes = plt.subplots(2, 2, figsize=(params.get("figsize_w", 16), params.get("figsize_h", 14)))
    axes = axes.flatten()

    scenario_colors = {
        "Historical": params.get("color_Historical", "#2C3E50"),
        "SSP2-4.5 2050s": params.get("color_SSP245_2050s", "#3498DB"),
        "SSP5-8.5 2050s": params.get("color_SSP585_2050s", "#E67E22"),
        "SSP5-8.5 2080s": params.get("color_SSP585_2080s", "#E74C3C"),
    }

    for idx, city_key in enumerate(CITY_KEYS):
        ax = axes[idx]
        cp = CITY_PARAMS[city_key]

        # RFC curves
        for sc_name, cf in SCENARIO_KEYS:
            ufris = []
            for rp in RETURN_PERIODS:
                ufris.append(calc_ufri(rp, cp, cf, "deep_tunnel"))
            ax.semilogx(RETURN_PERIODS, ufris, "o-",
                        color=scenario_colors[sc_name],
                        linewidth=params.get("rfc_linewidth", 2.2),
                        markersize=params.get("rfc_markersize", 7),
                        label=sc_name,
                        markeredgecolor="white",
                        markeredgewidth=params.get("rfc_markeredgewidth", 0.8))

        # Design standard line
        design_std = cp["design_std_yr"]
        ax.axvline(x=design_std,
                   color=params.get("design_line_color", "gray"),
                   linestyle=params.get("design_line_linestyle", "--"),
                   alpha=params.get("design_line_alpha", 0.6),
                   linewidth=params.get("design_line_width", 1.2))
        ax.text(design_std * 1.05, params.get("design_text_y", 0.08),
                f"Design\n({design_std}-yr)",
                fontsize=params.get("design_text_fontsize", 10),
                color=params.get("design_text_color", "gray"), va="bottom")

        # UFRI=0.5 line
        ax.axhline(y=0.5,
                   color=params.get("threshold_color", "red"),
                   linestyle=params.get("threshold_linestyle", ":"),
                   alpha=params.get("threshold_alpha", 0.4),
                   linewidth=params.get("threshold_linewidth", 1))

        # Guangzhou observed data
        if city_key == "Guangzhou":
            obs_storage = load_observed_gz_storage()
            sorted_vals, rp_est = observed_return_period_estimate(obs_storage)
            obs_ufri = gz_observed_ufri_estimate(sorted_vals)
            n_obs = min(params.get("obs_n_points", 30), len(sorted_vals))
            idx_sample = np.linspace(0, len(sorted_vals) - 1, n_obs, dtype=int)

            ax.scatter(rp_est[idx_sample], obs_ufri[idx_sample],
                       c=params.get("obs_marker_color", "#E74C3C"),
                       marker=params.get("obs_marker", "x"),
                       s=params.get("obs_marker_size", 60),
                       alpha=params.get("obs_marker_alpha", 0.7),
                       linewidths=params.get("obs_linewidths", 1.5),
                       zorder=10,
                       label=params.get("obs_label", "Observed (2022-2025)"))

            # Max observed annotation
            max_rp = rp_est[0]
            max_ufri = obs_ufri[0]
            ax.annotate(f"Max observed\n({sorted_vals[0]:.0f} m³)",
                        xy=(max_rp, max_ufri),
                        xytext=(max_rp * params.get("maxobs_xytext_x_mult", 2.5),
                                max_ufri + params.get("maxobs_xytext_y_offset", 0.08)),
                        fontsize=params.get("maxobs_fontsize", 10),
                        color=params.get("obs_marker_color", "#E74C3C"),
                        arrowprops=dict(arrowstyle="->",
                                        color=params.get("obs_marker_color", "#E74C3C"),
                                        alpha=0.6),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor=params.get("obs_marker_color", "#E74C3C"),
                                  alpha=0.8))

            # Calibration annotation
            ax.text(params.get("calib_x", 0.98), params.get("calib_y", 0.20),
                    f"Calibrated n = {cp['roughness_n']}\n(design n = 0.015)",
                    transform=ax.transAxes,
                    fontsize=params.get("calib_fontsize", 11),
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor=params.get("calib_box_color", "#FFF3CD"),
                              edgecolor=params.get("obs_marker_color", "#E74C3C"),
                              alpha=0.9))

        # T_crit annotations
        tcrit_list = []
        for sc_name, cf in SCENARIO_KEYS:
            if sc_name not in ["Historical", "SSP5-8.5 2080s"]:
                continue
            for rp in RETURN_PERIODS:
                ufri = calc_ufri(rp, cp, cf, "deep_tunnel")
                if ufri < 0.5:
                    tcrit_list.append((rp, sc_name, scenario_colors[sc_name]))
                    break
        for i, (rp_val, sc_name, color) in enumerate(tcrit_list):
            y_text = params.get("tcrit_y_base", 0.42) - i * params.get("tcrit_y_step", 0.08)
            x_text = min(rp_val * params.get("tcrit_x_mult", 0.55), 200)
            short_name = "Hist" if "Hist" in sc_name else "SSP585"
            label = f"T_crit={rp_val}yr ({short_name})"
            ax.annotate(label, xy=(rp_val, 0.5),
                        fontsize=params.get("tcrit_fontsize", 11),
                        color="black", fontweight="bold",
                        xytext=(x_text, y_text), alpha=1.0, clip_on=True,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                  edgecolor=color, linewidth=1.5, alpha=0.9))

        ax.set_xlim(params.get("xlim_min", 1.5), params.get("xlim_max", 250))
        ax.set_ylim(params.get("ylim_min", 0), params.get("ylim_max", 1.05))
        ax.set_xlabel("Return Period (years)")
        ax.set_ylabel("Urban Flood Resilience Index (UFRI)")
        ax.set_title(f"{cp['name_cn']} — {cp['tunnel']}\n({cp['type']}, Design: {cp['design_std_yr']}-yr)",
                     fontweight="bold")
        ax.legend(fontsize=params.get("legend_fontsize", 11.5),
                  loc=params.get("legend_loc", "lower left"),
                  framealpha=params.get("legend_framealpha", 0.8))
        ax.grid(True, alpha=params.get("grid_alpha", 0.15),
                which=params.get("grid_which", "both"))

    fig.suptitle("Fig. 1 | Resilience Failure Curves (RFC) of Deep Tunnel Systems\n"
                 "With operational data validation for Guangzhou (2022–2025)",
                 fontsize=params.get("suptitle_fontsize", 16),
                 fontweight="bold",
                 y=params.get("suptitle_y", 1.01))
    fig.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=params.get("save_dpi", 150), facecolor="white",
                bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


# ============================================================
# Code Generator
# ============================================================
def generate_python_code(params):
    """Generate the Python function code reflecting current parameters"""
    code = '''def fig1_rfc_updated():
    """生成Fig 1: RFC韧性失效曲线 + 广州观测验证"""
    fig, axes = plt.subplots(2, 2, figsize=({figsize_w}, {figsize_h}))
    axes = axes.flatten()

    city_keys = ["Guangzhou", "Chicago", "Tokyo", "KualaLumpur"]
    scenario_colors = {{
        "Historical": "{color_Historical}",
        "SSP2-4.5 2050s": "{color_SSP245_2050s}",
        "SSP5-8.5 2050s": "{color_SSP585_2050s}",
        "SSP5-8.5 2080s": "{color_SSP585_2080s}",
    }}

    for idx, city_key in enumerate(city_keys):
        ax = axes[idx]
        cp = CITY_PARAMS[city_key]

        # RFC曲线
        for sc_name, cf in climate_scenarios:
            ufris = []
            for rp in RETURN_PERIODS:
                ufri = calc_ufri(rp, cp, cf, "deep_tunnel")
                ufris.append(ufri)
            ax.semilogx(RETURN_PERIODS, ufris, "o-",
                        color=scenario_colors[sc_name],
                        linewidth={rfc_linewidth}, markersize={rfc_markersize},
                        label=sc_name,
                        markeredgecolor="white", markeredgewidth={rfc_markeredgewidth})

        # 设计标准竖线
        design_std = cp["design_std_yr"]
        ax.axvline(x=design_std, color="{design_line_color}", linestyle="{design_line_linestyle}",
                   alpha={design_line_alpha}, linewidth={design_line_width})
        ax.text(design_std * 1.05, {design_text_y}, f"Design\\n({{design_std}}-yr)",
                fontsize={design_text_fontsize}, color="{design_text_color}", va="bottom")

        # UFRI=0.5临界线
        ax.axhline(y=0.5, color="{threshold_color}", linestyle="{threshold_linestyle}",
                   alpha={threshold_alpha}, linewidth={threshold_linewidth})
'''

    # Add Guangzhou-specific block
    code += '''
        # === 广州: 叠加观测数据 ===
        if city_key == "Guangzhou":
            obs_storage = load_observed_gz_storage()
            sorted_vals, rp_est = observed_return_period_estimate(obs_storage)
            obs_ufri = gz_observed_ufri_estimate(sorted_vals)
            n_obs = min({obs_n_points}, len(sorted_vals))
            idx_sample = np.linspace(0, len(sorted_vals)-1, n_obs, dtype=int)

            ax.scatter(rp_est[idx_sample], obs_ufri[idx_sample],
                       c="{obs_marker_color}", marker="{obs_marker}",
                       s={obs_marker_size}, alpha={obs_marker_alpha},
                       linewidths={obs_linewidths}, zorder=10,
                       label="{obs_label}")

            # 标注最大观测事件
            max_rp = rp_est[0]
            max_ufri = obs_ufri[0]
            ax.annotate(f"Max observed\\n({{sorted_vals[0]:.0f}} m\\u00b3)",
                        xy=(max_rp, max_ufri),
                        xytext=(max_rp * {maxobs_xytext_x_mult},
                                max_ufri + {maxobs_xytext_y_offset}),
                        fontsize={maxobs_fontsize}, color="{obs_marker_color}",
                        arrowprops=dict(arrowstyle="->", color="{obs_marker_color}", alpha=0.6),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="{obs_marker_color}", alpha=0.8))

            # 率定标注
            ax.text({calib_x}, {calib_y},
                    f"Calibrated n = {{cp['roughness_n']}}\\n"
                    f"(design n = 0.015)",
                    transform=ax.transAxes, fontsize={calib_fontsize},
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="{calib_box_color}",
                             edgecolor="{obs_marker_color}", alpha=0.9))
'''

    code += '''
        # 标注T_crit
        tcrit_list = []
        for sc_name, cf in climate_scenarios:
            if sc_name not in ["Historical", "SSP5-8.5 2080s"]:
                continue
            for rp in RETURN_PERIODS:
                ufri = calc_ufri(rp, cp, cf, "deep_tunnel")
                if ufri < 0.5:
                    tcrit_list.append((rp, sc_name, scenario_colors[sc_name]))
                    break
        for i, (rp_val, sc_name, color) in enumerate(tcrit_list):
            y_text = {tcrit_y_base} - i * {tcrit_y_step}
            x_text = min(rp_val * {tcrit_x_mult}, 200)
            short_name = "Hist" if "Hist" in sc_name else "SSP585"
            label = f"T_crit={{rp_val}}yr ({{short_name}})"
            ax.annotate(label, xy=(rp_val, 0.5),
                        fontsize={tcrit_fontsize}, color="black", fontweight="bold",
                        xytext=(x_text, y_text), alpha=1.0, clip_on=True,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                  edgecolor=color, linewidth=1.5, alpha=0.9))

        ax.set_xlim({xlim_min}, {xlim_max})
        ax.set_ylim({ylim_min}, {ylim_max})
        ax.set_xlabel("Return Period (years)")
        ax.set_ylabel("Urban Flood Resilience Index (UFRI)")
        ax.set_title(f"{{cp['name_cn']}} \\u2014 {{cp['tunnel']}}\\n"
                     f"({{cp['type']}}, Design: {{cp['design_std_yr']}}-yr)",
                     fontweight="bold")
        ax.legend(fontsize={legend_fontsize}, loc="{legend_loc}",
                  framealpha={legend_framealpha})
        ax.grid(True, alpha={grid_alpha}, which="{grid_which}")

    fig.suptitle("Fig. 1 | Resilience Failure Curves (RFC) of Deep Tunnel Systems\\n"
                 "With operational data validation for Guangzhou (2022\\u20132025)",
                 fontsize={suptitle_fontsize}, fontweight="bold", y={suptitle_y})
    fig.tight_layout()

    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"Fig1_RFC_Curves.{{fmt}}", dpi=300, facecolor="white")
    plt.close(fig)
    print("  [OK] Fig 1: RFC Curves")
'''
    return code.format(**params)


# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fig 1 – Interactive Editor</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Arial,sans-serif;background:#1a1a2e;color:#e0e0e0;display:flex;height:100vh;overflow:hidden}
#preview{flex:1;display:flex;align-items:center;justify-content:center;background:#0f0f1a;padding:10px}
#preview img{max-width:100%;max-height:98vh;object-fit:contain;border-radius:4px}
#sidebar{width:380px;background:#16213e;overflow-y:auto;padding:15px;border-left:2px solid #0f3460}
#sidebar h2{color:#e94560;font-size:18px;margin-bottom:10px;padding-bottom:5px;border-bottom:1px solid #0f3460}
.panel{margin-bottom:12px;border:1px solid #0f3460;border-radius:6px;overflow:hidden}
.panel-header{background:#0f3460;padding:8px 12px;cursor:pointer;font-weight:bold;font-size:13px;display:flex;justify-content:space-between;align-items:center}
.panel-header:hover{background:#1a4a7a}
.panel-body{display:none;padding:10px 12px;background:#1a1a3e}
.panel.open .panel-body{display:block}
.row{display:flex;align-items:center;margin-bottom:6px;gap:8px}
.row label{flex:0 0 100px;font-size:12px;color:#aab}
.row input,.row select{flex:1;background:#0f0f2a;border:1px solid #0f3460;color:#e0e0e0;padding:4px 8px;border-radius:4px;font-size:12px}
.row input[type="color"]{flex:0 0 40px;padding:2px;height:28px;cursor:pointer}
.row input[type="range"]{flex:1}
.row .val{flex:0 0 35px;font-size:11px;color:#e94560;text-align:right}
h3{color:#e94560;font-size:13px;margin:6px 0 4px 0}
.btn-row{display:flex;gap:10px;margin-top:15px;position:sticky;bottom:0;background:#16213e;padding:10px 0}
.btn{padding:10px 18px;border:none;border-radius:6px;font-size:14px;font-weight:bold;cursor:pointer;transition:all .2s}
.btn-apply{background:#0f3460;color:#e0e0e0;flex:1}
.btn-apply:hover{background:#1a4a7a}
.btn-save{background:#e94560;color:white;flex:1}
.btn-save:hover{background:#c23152}
.toast{position:fixed;top:20px;right:20px;background:#27ae60;color:white;padding:12px 20px;border-radius:6px;font-weight:bold;z-index:999;display:none}
</style>
</head>
<body>

<div id="preview">
    <img id="fig-img" src="data:image/png;base64,{{ initial_image }}" alt="Fig 1 Preview">
</div>

<div id="sidebar">
    <h2>🎨 Fig 1 — Interactive Editor</h2>

    <!-- Figure Layout -->
    <div class="panel open">
        <div class="panel-header" onclick="this.parentElement.classList.toggle('open')">
            📐 Figure Layout <span>▼</span>
        </div>
        <div class="panel-body">
            <div class="row"><label>Width</label><input type="range" id="figsize_w" min="10" max="24" step="0.5" value="16"><span class="val" id="val_figsize_w">16</span></div>
            <div class="row"><label>Height</label><input type="range" id="figsize_h" min="8" max="20" step="0.5" value="14"><span class="val" id="val_figsize_h">14</span></div>
            <div class="row"><label>Title Size</label><input type="range" id="suptitle_fontsize" min="10" max="24" step="0.5" value="16"><span class="val" id="val_suptitle_fontsize">16</span></div>
            <div class="row"><label>Title Y Pos</label><input type="range" id="suptitle_y" min="0.95" max="1.08" step="0.005" value="1.01"><span class="val" id="val_suptitle_y">1.01</span></div>
            <div class="row"><label>Preview DPI</label><input type="range" id="save_dpi" min="80" max="200" step="10" value="150"><span class="val" id="val_save_dpi">150</span></div>
        </div>
    </div>

    <!-- Font Sizes -->
    <div class="panel">
        <div class="panel-header" onclick="this.parentElement.classList.toggle('open')">
            🔤 Font Sizes <span>▼</span>
        </div>
        <div class="panel-body">
            <div class="row"><label>Base</label><input type="range" id="base_fontsize" min="10" max="24" step="0.5" value="18"><span class="val" id="val_base_fontsize">18</span></div>
            <div class="row"><label>Subplot Title</label><input type="range" id="title_fontsize" min="10" max="26" step="0.5" value="20"><span class="val" id="val_title_fontsize">20</span></div>
            <div class="row"><label>Axis Label</label><input type="range" id="label_fontsize" min="10" max="24" step="0.5" value="18"><span class="val" id="val_label_fontsize">18</span></div>
            <div class="row"><label>Tick Label</label><input type="range" id="tick_fontsize" min="8" max="22" step="0.5" value="16"><span class="val" id="val_tick_fontsize">16</span></div>
            <div class="row"><label>Legend</label><input type="range" id="legend_fontsize" min="6" max="18" step="0.5" value="11.5"><span class="val" id="val_legend_fontsize">11.5</span></div>
            <div class="row"><label>Legend Loc</label>
                <select id="legend_loc">
                    <option value="lower left" selected>lower left</option>
                    <option value="lower right">lower right</option>
                    <option value="upper left">upper left</option>
                    <option value="upper right">upper right</option>
                    <option value="best">best</option>
                </select>
            </div>
        </div>
    </div>

    <!-- RFC Curves -->
    <div class="panel">
        <div class="panel-header" onclick="this.parentElement.classList.toggle('open')">
            📈 RFC Curves <span>▼</span>
        </div>
        <div class="panel-body">
            <div class="row"><label>Line Width</label><input type="range" id="rfc_linewidth" min="0.5" max="5" step="0.1" value="2.2"><span class="val" id="val_rfc_linewidth">2.2</span></div>
            <div class="row"><label>Marker Size</label><input type="range" id="rfc_markersize" min="2" max="15" step="0.5" value="7"><span class="val" id="val_rfc_markersize">7</span></div>
            <div class="row"><label>Marker Edge W</label><input type="range" id="rfc_markeredgewidth" min="0" max="2.5" step="0.1" value="0.8"><span class="val" id="val_rfc_markeredgewidth">0.8</span></div>
        </div>
    </div>

    <!-- Design Standard Line -->
    <div class="panel">
        <div class="panel-header" onclick="this.parentElement.classList.toggle('open')">
            📏 Design Standard <span>▼</span>
        </div>
        <div class="panel-body">
            <div class="row"><label>Color</label><input type="color" id="design_line_color" value="#808080"></div>
            <div class="row"><label>Style</label>
                <select id="design_line_linestyle">
                    <option value="--" selected>-- dashed</option><option value="-">- solid</option>
                    <option value=":">: dotted</option><option value="-.">-. dashdot</option>
                </select>
            </div>
            <div class="row"><label>Alpha</label><input type="range" id="design_line_alpha" min="0.1" max="1" step="0.05" value="0.6"><span class="val" id="val_design_line_alpha">0.6</span></div>
            <div class="row"><label>Width</label><input type="range" id="design_line_width" min="0.5" max="4" step="0.1" value="1.2"><span class="val" id="val_design_line_width">1.2</span></div>
            <div class="row"><label>Text Size</label><input type="range" id="design_text_fontsize" min="6" max="16" step="0.5" value="10"><span class="val" id="val_design_text_fontsize">10</span></div>
            <div class="row"><label>Text Y</label><input type="range" id="design_text_y" min="0.02" max="0.20" step="0.01" value="0.08"><span class="val" id="val_design_text_y">0.08</span></div>
        </div>
    </div>

    <!-- UFRI Threshold -->
    <div class="panel">
        <div class="panel-header" onclick="this.parentElement.classList.toggle('open')">
            ⚠ UFRI Threshold (0.5) <span>▼</span>
        </div>
        <div class="panel-body">
            <div class="row"><label>Color</label><input type="color" id="threshold_color" value="#ff0000"></div>
            <div class="row"><label>Style</label>
                <select id="threshold_linestyle">
                    <option value=":" selected>: dotted</option>
                    <option value="--">-- dashed</option><option value="-">- solid</option>
                    <option value="-.">-. dashdot</option>
                </select>
            </div>
            <div class="row"><label>Alpha</label><input type="range" id="threshold_alpha" min="0.1" max="1" step="0.05" value="0.4"><span class="val" id="val_threshold_alpha">0.4</span></div>
            <div class="row"><label>Width</label><input type="range" id="threshold_linewidth" min="0.5" max="3" step="0.1" value="1"><span class="val" id="val_threshold_linewidth">1</span></div>
        </div>
    </div>

    <!-- Observed Data (Guangzhou) -->
    <div class="panel">
        <div class="panel-header" onclick="this.parentElement.classList.toggle('open')">
            🔴 Guangzhou Observed <span>▼</span>
        </div>
        <div class="panel-body">
            <div class="row"><label>Color</label><input type="color" id="obs_marker_color" value="#E74C3C"></div>
            <div class="row"><label>Marker</label>
                <select id="obs_marker">
                    <option value="x" selected>x</option><option value="+">+</option>
                    <option value="o">o</option><option value="s">s</option>
                    <option value="D">D</option><option value="^">^</option>
                    <option value="v">v</option><option value="*">*</option>
                </select>
            </div>
            <div class="row"><label>Size</label><input type="range" id="obs_marker_size" min="20" max="150" step="5" value="60"><span class="val" id="val_obs_marker_size">60</span></div>
            <div class="row"><label>Alpha</label><input type="range" id="obs_marker_alpha" min="0.2" max="1" step="0.05" value="0.7"><span class="val" id="val_obs_marker_alpha">0.7</span></div>
            <div class="row"><label>N Points</label><input type="range" id="obs_n_points" min="5" max="50" step="5" value="30"><span class="val" id="val_obs_n_points">30</span></div>
        </div>
    </div>

    <!-- Annotations -->
    <div class="panel">
        <div class="panel-header" onclick="this.parentElement.classList.toggle('open')">
            💬 Annotations <span>▼</span>
        </div>
        <div class="panel-body">
            <h3>T_crit Labels</h3>
            <div class="row"><label>Font Size</label><input type="range" id="tcrit_fontsize" min="6" max="16" step="0.5" value="11"><span class="val" id="val_tcrit_fontsize">11</span></div>
            <div class="row"><label>Y Base</label><input type="range" id="tcrit_y_base" min="0.25" max="0.55" step="0.01" value="0.42"><span class="val" id="val_tcrit_y_base">0.42</span></div>
            <div class="row"><label>Y Step</label><input type="range" id="tcrit_y_step" min="0.04" max="0.15" step="0.01" value="0.08"><span class="val" id="val_tcrit_y_step">0.08</span></div>
            <div class="row"><label>X Mult</label><input type="range" id="tcrit_x_mult" min="0.3" max="1.5" step="0.05" value="0.55"><span class="val" id="val_tcrit_x_mult">0.55</span></div>
            <h3>Max Observed Arrow</h3>
            <div class="row"><label>Font Size</label><input type="range" id="maxobs_fontsize" min="6" max="16" step="0.5" value="10"><span class="val" id="val_maxobs_fontsize">10</span></div>
            <div class="row"><label>X Text Mult</label><input type="range" id="maxobs_xytext_x_mult" min="1" max="5" step="0.25" value="2.5"><span class="val" id="val_maxobs_xytext_x_mult">2.5</span></div>
            <div class="row"><label>Y Text Offset</label><input type="range" id="maxobs_xytext_y_offset" min="0.02" max="0.20" step="0.01" value="0.08"><span class="val" id="val_maxobs_xytext_y_offset">0.08</span></div>
            <h3>Calibration Box</h3>
            <div class="row"><label>Font Size</label><input type="range" id="calib_fontsize" min="6" max="16" step="0.5" value="11"><span class="val" id="val_calib_fontsize">11</span></div>
            <div class="row"><label>X Pos</label><input type="range" id="calib_x" min="0.7" max="1.0" step="0.01" value="0.98"><span class="val" id="val_calib_x">0.98</span></div>
            <div class="row"><label>Y Pos</label><input type="range" id="calib_y" min="0.05" max="0.40" step="0.01" value="0.20"><span class="val" id="val_calib_y">0.20</span></div>
        </div>
    </div>

    <!-- Scenario Colors -->
    <div class="panel">
        <div class="panel-header" onclick="this.parentElement.classList.toggle('open')">
            🎨 Scenario Colors <span>▼</span>
        </div>
        <div class="panel-body">
            <div class="row"><label>Historical</label><input type="color" id="color_Historical" value="#2C3E50"></div>
            <div class="row"><label>SSP2-4.5</label><input type="color" id="color_SSP245_2050s" value="#3498DB"></div>
            <div class="row"><label>SSP5-8.5 2050s</label><input type="color" id="color_SSP585_2050s" value="#E67E22"></div>
            <div class="row"><label>SSP5-8.5 2080s</label><input type="color" id="color_SSP585_2080s" value="#E74C3C"></div>
        </div>
    </div>

    <!-- Axis & Grid -->
    <div class="panel">
        <div class="panel-header" onclick="this.parentElement.classList.toggle('open')">
            📊 Axis & Grid <span>▼</span>
        </div>
        <div class="panel-body">
            <div class="row"><label>X Min</label><input type="range" id="xlim_min" min="0.5" max="5" step="0.1" value="1.5"><span class="val" id="val_xlim_min">1.5</span></div>
            <div class="row"><label>X Max</label><input type="range" id="xlim_max" min="100" max="500" step="10" value="250"><span class="val" id="val_xlim_max">250</span></div>
            <div class="row"><label>Y Max</label><input type="range" id="ylim_max" min="0.9" max="1.3" step="0.01" value="1.05"><span class="val" id="val_ylim_max">1.05</span></div>
            <div class="row"><label>Grid Alpha</label><input type="range" id="grid_alpha" min="0.05" max="0.5" step="0.01" value="0.15"><span class="val" id="val_grid_alpha">0.15</span></div>
        </div>
    </div>

    <!-- Buttons -->
    <div class="btn-row">
        <button class="btn btn-apply" onclick="applyChanges()">🔄 Apply Changes</button>
        <button class="btn btn-save" onclick="saveAndGenerate()">💾 Generate Code</button>
    </div>
</div>

<div id="toast" class="toast"></div>

<script>
// Update range value displays
document.querySelectorAll('input[type="range"]').forEach(slider => {
    slider.addEventListener('input', function() {
        const valEl = document.getElementById('val_' + this.id);
        if (valEl) valEl.textContent = this.value;
    });
});

// Collect all parameters
function collectParams() {
    const params = {};
    document.querySelectorAll('#sidebar input, #sidebar select').forEach(el => {
        if (el.type === 'range' || el.type === 'color' || el.tagName === 'SELECT') {
            params[el.id] = el.type === 'range' ? parseFloat(el.value) : el.value;
        }
    });
    return params;
}

// Apply changes
async function applyChanges() {
    const btn = document.querySelector('.btn-apply');
    btn.textContent = '⏳ Rendering...';
    btn.disabled = true;

    try {
        const resp = await fetch('/update', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(collectParams())
        });
        const data = await resp.json();
        document.getElementById('fig-img').src = 'data:image/png;base64,' + data.image;
        showToast('✅ Figure updated');
    } catch(e) {
        showToast('❌ Error: ' + e.message);
    }

    btn.textContent = '🔄 Apply Changes';
    btn.disabled = false;
}

// Save & Generate Code
async function saveAndGenerate() {
    const btn = document.querySelector('.btn-save');
    btn.textContent = '⏳ Generating...';
    btn.disabled = true;

    try {
        const resp = await fetch('/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(collectParams())
        });
        const data = await resp.json();
        document.getElementById('fig-img').src = 'data:image/png;base64,' + data.image;
        // Download the Python code
        const blob = new Blob([data.code], {type: 'text/python'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'fig1_rfc_updated.py';
        a.click();
        URL.revokeObjectURL(url);
        showToast('✅ Code generated & downloaded!');
    } catch(e) {
        showToast('❌ Error: ' + e.message);
    }

    btn.textContent = '💾 Generate Code';
    btn.disabled = false;
}

function showToast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.style.display = 'block';
    setTimeout(() => t.style.display = 'none', 2500);
}

// Keyboard shortcut: Enter to apply
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        applyChanges();
    } else if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        saveAndGenerate();
    }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    img_b64 = generate_fig1(DEFAULT_PARAMS)
    return render_template_string(HTML_TEMPLATE, initial_image=img_b64)


@app.route("/update", methods=["POST"])
def update():
    params = request.get_json()
    merged = {**DEFAULT_PARAMS, **params}
    # Ensure numeric types
    for k, v in merged.items():
        if k in DEFAULT_PARAMS and isinstance(DEFAULT_PARAMS[k], (int, float)):
            try:
                merged[k] = float(v) if isinstance(DEFAULT_PARAMS[k], float) else int(float(v))
            except:
                merged[k] = DEFAULT_PARAMS[k]
    img_b64 = generate_fig1(merged)
    return jsonify({"image": img_b64})


@app.route("/generate", methods=["POST"])
def generate():
    params = request.get_json()
    merged = {**DEFAULT_PARAMS, **params}
    for k, v in merged.items():
        if k in DEFAULT_PARAMS and isinstance(DEFAULT_PARAMS[k], (int, float)):
            try:
                merged[k] = float(v) if isinstance(DEFAULT_PARAMS[k], float) else int(float(v))
            except:
                merged[k] = DEFAULT_PARAMS[k]
    img_b64 = generate_fig1(merged)
    code = generate_python_code(merged)
    return jsonify({"image": img_b64, "code": code})


if __name__ == "__main__":
    print("=" * 60)
    print("  Fig 1 Interactive Editor")
    print("  Open: http://localhost:5100")
    print("  Ctrl+Enter = Apply  |  Ctrl+S = Generate Code")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5100, debug=False)
