"""
ED Fig. SS11 |
SA7 | Roughness transfer uncertainty
{'='*50}
独立绘图脚本 -- 从 FigS12_Roughness_Transfer.xlsx 读取数据
用法: python FigS12_Roughness_Transfer.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), "FigS12_Roughness_Transfer.xlsx")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 15, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

def plot_sa7():
    xl = pd.ExcelFile(DATA_FILE)
    print(f"Sheets: {xl.sheet_names}")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Read first data sheet (skip Description)
    data_sheets = [s for s in xl.sheet_names if s != "Description"]
    if data_sheets:
        df = pd.read_excel(DATA_FILE, sheet_name=data_sheets[0])
        # Simple visualization: plot all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            x = df[numeric_cols[0]].values
            for col in numeric_cols[1:min(6, len(numeric_cols))]:
                ax.plot(x, df[col].values, "o-", linewidth=2, markersize=4, label=col)
        else:
            ax.text(0.5, 0.5, f"Data preview:\n{df.head().to_string()}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=10, family="monospace")

    ax.set_xlabel(numeric_cols[0] if len(numeric_cols) > 0 else "X")
    ax.set_ylabel("Value")
    ax.set_title("SA7 | Roughness transfer uncertainty", fontweight="bold")
    ax.legend(fontsize=9, loc="best"); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "FigS12_Roughness_Transfer.pdf")
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_sa7()
