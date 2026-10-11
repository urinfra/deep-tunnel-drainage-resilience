
"""
Fig. 5 | Cost-Effectiveness and Performance of Infrastructure Strategies (Nature Cities)
=========================================================================================
a–d, DTFRI vs return period for infrastructure strategies across four cities.
e, Cost-efficiency (DTFRI per bn USD NPV) at T=50 benchmark.

Reads: Fig5_Cost_Resilience.xlsx
Output: Fig5_Cost_Resilience.pdf
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = os.path.join(os.path.dirname(__file__), "Fig5_Cost_Resilience.xlsx")
OUT_DIR = os.path.dirname(__file__)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 17,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 13,
    'axes.spines.right': True, 'axes.spines.top': True,
    'axes.linewidth': 1.2, 'legend.frameon': False, 'figure.dpi': 150,
})

INFRA_COLORS = {"Deep Tunnel":"#0F4D92","Green Infra":"#1B7A3D",
                "Distributed":"#9A4D8E","Hybrid":"#B64342"}
INFRA_MARKERS = {"Deep Tunnel":"o","Green Infra":"s","Distributed":"D","Hybrid":"^"}
INFRA_ORDER = ["Deep Tunnel","Green Infra","Distributed","Hybrid"]

def add_panel_label(ax,label,x=-0.06,y=1.03,fs=16,c="#272727",fw='bold'):
    ax.text(x,y,label,transform=ax.transAxes,fontsize=fs,fontweight=fw,color=c,ha='left',va='bottom')

df = pd.read_excel(DATA_FILE, sheet_name="Cost_Resilience")
CITIES = ["Guangzhou","Chicago","Tokyo","Kuala Lumpur"]

fig = plt.figure(figsize=(15, 16))
# 3 rows × 2 cols: rows 1-2 for DTFRI curves (4 cities), row 3 for cost-efficiency
gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.25,
                      height_ratios=[1, 1, 0.7],
                      left=0.07, right=0.97, top=0.96, bottom=0.05)

panel_labels_ufri = ['a','b','c','d']

# Panels a-d: DTFRI vs return period for each city
for idx, city in enumerate(CITIES):
    row, col = idx // 2, idx % 2
    ax = fig.add_subplot(gs[row, col])
    cd = df[df["City"]==city]
    
    for infra in INFRA_ORDER:
        idata = cd[cd["Infrastructure"]==infra].sort_values("Return_Period_yr")
        if len(idata)==0: continue
        ax.semilogx(idata["Return_Period_yr"],idata["DTFRI"],
                    marker=INFRA_MARKERS[infra],color=INFRA_COLORS[infra],
                    linewidth=2.5,markersize=10,markeredgecolor="white",
                    markeredgewidth=0.6,label=infra,zorder=3)
    
    ax.axhline(y=0.5,color="#B64342",linestyle=":",alpha=0.35,linewidth=0.8,zorder=1)
    ax.set_xlim(1.5,250); ax.set_ylim(0,1.02)
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("DTFRI")
    ax.set_title(f"{city}",fontweight="bold",fontsize=16)
    ax.grid(True,alpha=0.10,linewidth=0.5)
    if idx == 0:
        ax.legend(loc="lower left",fontsize=13,handlelength=1.5,handletextpad=0.5,borderpad=0.3,labelspacing=0.3)
    add_panel_label(ax,panel_labels_ufri[idx])

# Panel e: Relative cost-efficiency at T=50 as dot plot (individual data points; editorial E.6)
ax_e = fig.add_subplot(gs[2, :])
bar_width = 0.18
x = np.arange(len(CITIES))
other_infra = ["Green Infra","Distributed","Hybrid"]
n_other = len(other_infra)
offsets = np.linspace(-bar_width*(n_other-1)/2, bar_width*(n_other-1)/2, n_other)

for i, infra in enumerate(other_infra):
    vals = []
    for city in CITIES:
        row_dt = df[(df["City"]==city)&(df["Infrastructure"]=="Deep Tunnel")&(df["Return_Period_yr"]==50)]
        row_infra = df[(df["City"]==city)&(df["Infrastructure"]==infra)&(df["Return_Period_yr"]==50)]
        if len(row_dt)==0 or len(row_infra)==0: vals.append(0); continue
        ce_dt = row_dt["DTFRI"].values[0] / row_dt["NPV_30yr_MUSD"].values[0]
        ce_infra = row_infra["DTFRI"].values[0] / row_infra["NPV_30yr_MUSD"].values[0]
        vals.append(ce_infra / ce_dt)
    ax_e.plot(x+offsets[i],vals,
              linestyle="none",marker=INFRA_MARKERS[infra],markersize=11,
              markeredgecolor="white",markeredgewidth=0.8,
              color=INFRA_COLORS[infra],label=infra,zorder=3)

ax_e.axhline(y=1.0,color="#0F4D92",linestyle="--",alpha=0.5,linewidth=1.0)
ax_e.set_xticks(x); ax_e.set_xticklabels(CITIES)
ax_e.set_ylabel("Relative cost-efficiency\n(Deep Tunnel = 1.0 at T=50)")
ax_e.set_title("e  Relative cost-efficiency (T = 50 years)",fontweight="bold",fontsize=16,loc="left")
ax_e.legend(loc="upper right",fontsize=13,handlelength=1.5,handletextpad=0.5,borderpad=0.3,labelspacing=0.3)
ax_e.grid(True,axis="y",alpha=0.10,linewidth=0.5); ax_e.set_axisbelow(True)
ax_e.set_ylim(0, None)

fig.savefig(os.path.join(OUT_DIR,"Fig5_Cost_Resilience.pdf"),dpi=600,facecolor="white",bbox_inches="tight",pad_inches=0.1)
plt.close(fig)
print(f"Saved: Fig5_Cost_Resilience.pdf")
