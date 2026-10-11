"""
ED Fig. S8 |
SA4 | Economic sensitivity
{'='*50}
独立绘图脚本 -- 从 FigS9_Economic_Sensitivity.xlsx 读取数据
用法: python FigS9_Economic_Sensitivity.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = os.path.join(os.path.dirname(__file__), "FigS9_Economic_Sensitivity.xlsx")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def first_existing_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for col in candidates:
        key = str(col).lower().strip()
        if key in lower_map:
            return lower_map[key]
    for col in candidates:
        key = str(col).lower().strip()
        for c in df.columns:
            if key in str(c).lower().strip():
                return c
    return None

def plot_sa4():
    xl = pd.ExcelFile(DATA_FILE)
    data_sheets = [s for s in xl.sheet_names if str(s).lower() not in ["description", "readme"]]

    if not data_sheets:
        raise ValueError("No valid data sheet found in the Excel file.")

    df = pd.read_excel(DATA_FILE, sheet_name=data_sheets[0])

    discount_col = first_existing_col(df, ["Discount_Rate", "Discount Rate"])
    cost_col = first_existing_col(df, ["Cost_Multiplier", "Cost Multiplier", "Cost_Level", "Cost Level"])
    npv_col = first_existing_col(df, ["NPV_30yr_MUSD", "NPV_30yr", "NPV", "NPV_MUSD"])
    ufri_col = first_existing_col(df, ["DTFRI_at_T50", "DTFRI_T50", "DTFRI"])
    ce_col = first_existing_col(df, ["Cost_Effectiveness", "Cost-effectiveness", "CE"])

    if ce_col is None:
        raise ValueError("Cannot find Cost_Effectiveness column.")

    if cost_col is None and npv_col is None:
        raise ValueError("Cannot find Cost_Multiplier or NPV column for x-axis.")

    x_col = cost_col if cost_col is not None else npv_col

    fig, ax = plt.subplots(figsize=(10, 7))

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.14,
        top=0.96
    )

    if discount_col is not None:
        discount_values = sorted(df[discount_col].dropna().unique())
        cmap = plt.get_cmap("tab10")

        for i, dr in enumerate(discount_values):
            sub = df[df[discount_col] == dr].copy()
            sub = sub.sort_values(x_col)

            label = f"Discount rate = {dr:g}" if np.issubdtype(type(dr), np.number) else f"Discount rate = {dr}"

            ax.plot(
                sub[x_col].values,
                sub[ce_col].values,
                "o-",
                linewidth=2.5,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                color=cmap(i % 10),
                label=label
            )
    else:
        sub = df.sort_values(x_col)
        ax.plot(
            sub[x_col].values,
            sub[ce_col].values,
            "o-",
            linewidth=2.5,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.8,
            color="#E74C3C",
            label="Cost-effectiveness"
        )

    if cost_col is not None:
        ax.set_xlabel("Cost Multiplier")
    else:
        ax.set_xlabel("NPV over 30 years (MUSD)")

    ax.set_ylabel("Cost-Effectiveness")

    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.grid(True, alpha=0.20)

    ax.legend(
        fontsize=14,
        loc="best",
        frameon=True,
        framealpha=0.90,
        edgecolor="lightgray"
    )

    output_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(output_dir, "FigS9_Economic_Sensitivity.pdf")
    tiff_path = os.path.join(output_dir, "FigS9_Economic_Sensitivity.tiff")

    try:
        fig.savefig(pdf_path, dpi=300, facecolor="white")
        print(f"Saved PDF: {pdf_path}")
    except PermissionError:
        pdf_path_new = os.path.join(output_dir, "FigS9_Economic_Sensitivity_new.pdf")
        fig.savefig(pdf_path_new, dpi=300, facecolor="white")
        print(f"Original PDF is occupied. Saved PDF as: {pdf_path_new}")

    try:
        fig.savefig(tiff_path, dpi=600, facecolor="white")
        print(f"Saved TIFF: {tiff_path}")
    except PermissionError:
        tiff_path_new = os.path.join(output_dir, "FigS9_Economic_Sensitivity_new.tiff")
        fig.savefig(tiff_path_new, dpi=600, facecolor="white")
        print(f"Original TIFF is occupied. Saved TIFF as: {tiff_path_new}")

    plt.close(fig)

if __name__ == "__main__":
    plot_sa4()