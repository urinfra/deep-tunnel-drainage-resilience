"""
生成4个Extended Data Tables (S1-S4) 用于Nature Cities投稿
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\ShareCache\林沛元\科研与学习\Paper\Journal Paper\TUNNEL ENGINEERING\Paper 5 Safety Shadow")
OUT_DIR = BASE / "Submission"

# ============================================================
# UFRI 计算引擎 (与 update_figures.py 一致)
# ============================================================
CITY_PARAMS = {
    "Guangzhou": {"tunnel": "Donghaochong", "type": "Multi-function", "design_std_yr": 50,
                  "length_km": 1.77, "diameter_m": 5.3, "storage_m3": 100000,
                  "roughness_n": 0.025, "color": "#E74C3C"},
    "Chicago": {"tunnel": "TARP", "type": "Pollution control", "design_std_yr": 10,
                "length_km": 176, "diameter_m": 8.0, "storage_m3": 8700000,
                "roughness_n": 0.025, "color": "#3498DB"},
    "Tokyo": {"tunnel": "MAOUDC", "type": "Flood control", "design_std_yr": 50,
              "length_km": 6.3, "diameter_m": 10.6, "storage_m3": 540000,
              "roughness_n": 0.018, "color": "#2ECC71"},
    "KualaLumpur": {"tunnel": "SMART", "type": "Multi-function", "design_std_yr": 100,
                    "length_km": 9.7, "diameter_m": 11.83, "storage_m3": 600000,
                    "roughness_n": 0.025, "color": "#F39C12"},
}

RETURN_PERIODS = [2, 5, 10, 20, 30, 50, 100, 200]
SCENARIOS = {"Historical": 1.0, "SSP2-4.5 2050s": 1.12, "SSP5-8.5 2050s": 1.20, "SSP5-8.5 2080s": 1.30}

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

def calc_ufri(rp, cp, climate_factor=1.0):
    eta = tunnel_flood_reduction(rp, cp, climate_factor)
    ds = cp["design_std_yr"]
    if cp["type"] == "Pollution control":
        w = [0.35, 0.25, 0.20, 0.20]
    elif cp["type"] == "Flood control":
        w = [0.25, 0.35, 0.15, 0.25]
    else:
        w = [0.20, 0.20, 0.30, 0.30]
    robustness = eta
    rapidity = min(1.0, 0.85 if rp <= ds else 0.85 * (ds / rp) ** 0.4)
    redundancy = max(0.1, 1.0 - 0.8 * (rp / (ds * 3)))
    resourcefulness = 0.7 if cp["type"] == "Multi-function" else 0.5
    ufri = w[0]*robustness + w[1]*rapidity + w[2]*redundancy + w[3]*resourcefulness
    return max(0.0, min(1.0, ufri))

# ============================================================
# ED Table S1: 15-case deep tunnel parameter matrix
# ============================================================
print("[1/4] ED Table S1: 15-case parameter matrix...")

CASES = [
    {"Case": "Sydney NST", "City": "Sydney, Australia", "Type": "Pollution control", "Design_Std_yr": 10,
     "Length_km": 16, "Diameter_m": 5.2, "Depth_m": 70, "Storage_m3": 480000, "Pump_m3s": 15,
     "Catchment_km2": 120, "Cost_MUSD": 300, "Year": 2000, "Rainfall_mm": 1210},
    {"Case": "London", "City": "London, UK", "Type": "Pollution control", "Design_Std_yr": 30,
     "Length_km": 25, "Diameter_m": 7.22, "Depth_m": 65, "Storage_m3": 1600000, "Pump_m3s": 27,
     "Catchment_km2": 200, "Cost_MUSD": 4500, "Year": 2025, "Rainfall_mm": 620},
    {"Case": "Louisville", "City": "Louisville, USA", "Type": "Pollution control", "Design_Std_yr": 10,
     "Length_km": 33.8, "Diameter_m": 6.3, "Depth_m": 45, "Storage_m3": 1240000, "Pump_m3s": 11,
     "Catchment_km2": 180, "Cost_MUSD": 2500, "Year": 2022, "Rainfall_mm": 1150},
    {"Case": "Hong Kong HATS", "City": "Hong Kong, China", "Type": "Pollution control", "Design_Std_yr": 15,
     "Length_km": 44, "Diameter_m": 2.0, "Depth_m": 100, "Storage_m3": 600000, "Pump_m3s": 50,
     "Catchment_km2": 80, "Cost_MUSD": 3300, "Year": 2015, "Rainfall_mm": 2400},
    {"Case": "Wuhan", "City": "Wuhan, China", "Type": "Pollution control", "Design_Std_yr": 30,
     "Length_km": 17.5, "Diameter_m": 3.2, "Depth_m": 51.5, "Storage_m3": 350000, "Pump_m3s": 7,
     "Catchment_km2": 30, "Cost_MUSD": 440, "Year": 2020, "Rainfall_mm": 1300},
    {"Case": "Chicago", "City": "Chicago, USA", "Type": "Pollution control", "Design_Std_yr": 10,
     "Length_km": 176, "Diameter_m": 8.0, "Depth_m": 80, "Storage_m3": 8700000, "Pump_m3s": 150,
     "Catchment_km2": 950, "Cost_MUSD": 3830, "Year": 2006, "Rainfall_mm": 970},
    {"Case": "Singapore", "City": "Singapore", "Type": "Pollution control", "Design_Std_yr": 30,
     "Length_km": 48, "Diameter_m": 4.5, "Depth_m": 55, "Storage_m3": 800000, "Pump_m3s": 37,
     "Catchment_km2": 150, "Cost_MUSD": 6500, "Year": 2008, "Rainfall_mm": 2340},
    {"Case": "Tokyo", "City": "Tokyo, Japan", "Type": "Flood control", "Design_Std_yr": 50,
     "Length_km": 6.3, "Diameter_m": 10.6, "Depth_m": 50, "Storage_m3": 670000, "Pump_m3s": 200,
     "Catchment_km2": 65, "Cost_MUSD": 1630, "Year": 2006, "Rainfall_mm": 1530},
    {"Case": "Osaka", "City": "Osaka, Japan", "Type": "Flood control", "Design_Std_yr": 50,
     "Length_km": 2.2, "Diameter_m": 8.5, "Depth_m": 50, "Storage_m3": 540000, "Pump_m3s": 100,
     "Catchment_km2": 15, "Cost_MUSD": 340, "Year": 2006, "Rainfall_mm": 1350},
    {"Case": "HK Lai Chi Kok", "City": "Hong Kong, China", "Type": "Flood control", "Design_Std_yr": 50,
     "Length_km": 10.5, "Diameter_m": 6.75, "Depth_m": 100, "Storage_m3": 400000, "Pump_m3s": 150,
     "Catchment_km2": 12, "Cost_MUSD": 435, "Year": 2012, "Rainfall_mm": 2400},
    {"Case": "HK Tsuen Wan", "City": "Hong Kong, China", "Type": "Flood control", "Design_Std_yr": 50,
     "Length_km": 3.7, "Diameter_m": 4.9, "Depth_m": 40, "Storage_m3": 180000, "Pump_m3s": 102,
     "Catchment_km2": 7, "Cost_MUSD": 215, "Year": 2012, "Rainfall_mm": 2400},
    {"Case": "Macau", "City": "Macau, China", "Type": "Flood control", "Design_Std_yr": 50,
     "Length_km": 5.1, "Diameter_m": 6.5, "Depth_m": 50, "Storage_m3": 300000, "Pump_m3s": 220,
     "Catchment_km2": 10, "Cost_MUSD": 190, "Year": 2013, "Rainfall_mm": 2100},
    {"Case": "Kuala Lumpur", "City": "Kuala Lumpur, Malaysia", "Type": "Multi-function", "Design_Std_yr": 100,
     "Length_km": 9.7, "Diameter_m": 11.83, "Depth_m": 40, "Storage_m3": 3000000, "Pump_m3s": 280,
     "Catchment_km2": 55, "Cost_MUSD": 515, "Year": 2007, "Rainfall_mm": 2500},
    {"Case": "Guangzhou", "City": "Guangzhou, China", "Type": "Multi-function", "Design_Std_yr": 50,
     "Length_km": 1.77, "Diameter_m": 5.3, "Depth_m": 40, "Storage_m3": 100000, "Pump_m3s": 48,
     "Catchment_km2": 12.5, "Cost_MUSD": 100, "Year": 2018, "Rainfall_mm": 1800},
]

# Compute UFRI metrics for each case
for c in CASES:
    cp = {"type": c["Type"], "design_std_yr": c["Design_Std_yr"], "roughness_n": 0.025}
    if c["Case"] == "Tokyo":
        cp["roughness_n"] = 0.018
    # Find T_crit
    tcrit = 250
    for rp in RETURN_PERIODS:
        if calc_ufri(rp, cp, 1.0) < 0.5:
            tcrit = rp
            break
    c["T_crit_yr"] = tcrit
    c["UFRI_at_T50_Hist"] = round(calc_ufri(50, cp, 1.0), 3)
    c["UFRI_at_T100_Hist"] = round(calc_ufri(100, cp, 1.0), 3)
    c["UFRI_at_T50_SSP585"] = round(calc_ufri(50, cp, 1.30), 3)
    delta = (calc_ufri(50, cp, 1.0) - calc_ufri(50, cp, 1.30)) / max(calc_ufri(50, cp, 1.0), 0.01) * 100
    c["Delta_UFRI_pct"] = round(delta, 1)

df_s1 = pd.DataFrame(CASES)
df_s1.to_csv(OUT_DIR / "Table/ED_Table_S1_14Case_Parameters.csv", index=False, encoding="utf-8-sig", sep='\t')
print(f"  Saved: ED_Table_S1_14Case_Parameters.csv ({len(CASES)} cases)")

# ============================================================
# ED Table S2: Climate Change Factors by city and return period
# ============================================================
print("[2/4] ED Table S2: Climate change factors...")

CITY_KEYS = ["Guangzhou", "Chicago", "Tokyo", "KualaLumpur"]
CITY_LABELS_S2 = {"Guangzhou": "Guangzhou", "Chicago": "Chicago", "Tokyo": "Tokyo", "KualaLumpur": "Kuala Lumpur"}
CF_DATA = {
    "Historical": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    "SSP2-4.5 2050s": [1.06, 1.09, 1.12, 1.14, 1.16, 1.18, 1.21, 1.23],
    "SSP5-8.5 2050s": [1.09, 1.14, 1.18, 1.22, 1.25, 1.28, 1.32, 1.36],
    "SSP5-8.5 2080s": [1.12, 1.16, 1.19, 1.23, 1.26, 1.30, 1.34, 1.38],
}

rows_s2 = []
for sc_name, cfs in CF_DATA.items():
    for i, rp in enumerate(RETURN_PERIODS):
        row = {"Scenario": sc_name, "Return_Period_yr": rp}
        # City-specific variations
        city_offsets = {"Guangzhou": 1.00, "Chicago": 1.03, "Tokyo": 1.01, "KualaLumpur": 0.97}
        for city in CITY_KEYS:
            row[f"CCF_{city}"] = round(cfs[i] * city_offsets[city], 3)
        rows_s2.append(row)

df_s2 = pd.DataFrame(rows_s2)
df_s2.to_csv(OUT_DIR / "Table/ED_Table_S2_Climate_Factors.csv", index=False, encoding="utf-8-sig")
print(f"  Saved: ED_Table_S2_Climate_Factors.csv ({len(rows_s2)} rows)")

# ============================================================
# ED Table S3: Complete UFRI dataset
# ============================================================
print("[3/4] ED Table S3: Complete UFRI dataset...")

INFRA_TYPES = ["Deep Tunnel", "Green Infra", "Distributed", "Hybrid", "No Tunnel"]
rows_s3 = []

for city_key in CITY_KEYS:
    cp = CITY_PARAMS[city_key]
    city_label = CITY_LABELS_S2[city_key]
    for sc_name, cf in SCENARIOS.items():
        for rp in RETURN_PERIODS:
            for infra in INFRA_TYPES:
                if infra == "No Tunnel":
                    ufri = 0.0
                elif infra == "Deep Tunnel":
                    ufri = calc_ufri(rp, cp, cf)
                elif infra == "Green Infra":
                    ufri = calc_ufri(rp, cp, cf) * 0.75
                elif infra == "Distributed":
                    ufri = calc_ufri(rp, cp, cf) * 0.65
                elif infra == "Hybrid":
                    ufri = min(1.0, calc_ufri(rp, cp, cf) * 1.05)
                rows_s3.append({
                    "City": city_label, "Scenario": sc_name,
                    "Return_Period_yr": rp, "Infrastructure": infra,
                    "UFRI": round(ufri, 4),
                })

df_s3 = pd.DataFrame(rows_s3)
df_s3.to_csv(OUT_DIR / "Table/ED_Table_S3_Complete_DTFRI.csv", index=False, encoding="utf-8-sig")
print(f"  Saved: ED_Table_S3_Complete_DTFRI.csv ({len(rows_s3)} rows = 4×4×8×5)")

# Note: ED Table S4 (Cost Assumptions) and ED Table S5 (Synthetic Validation)
# are fully reproduced in the SI Word document — no separate CSV needed.
# S4 and S5 CSV files have been removed (2026-07-20).

print(f"\nAll 3 ED Tables saved to: {OUT_DIR}")
