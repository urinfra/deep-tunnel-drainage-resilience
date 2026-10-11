
"""
Fig. 4 | Spatial Equity of Deep Tunnel Flood Protection Benefits (Nature Cities)
=================================================================================
a, Lorenz curves of cumulative population vs cumulative protection benefit.
b, Protection coverage by income quintile.

Reads: Fig4_Lorenz_Equity.xlsx
Output: Fig4_Lorenz_Equity.pdf
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = os.path.join(os.path.dirname(__file__), "Fig4_Lorenz_Equity.xlsx")
OUT_DIR = os.path.dirname(__file__)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({
    'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'axes.spines.right': True, 'axes.spines.top': True,
    'axes.linewidth': 1.0, 'legend.frameon': False, 'figure.dpi': 150,
})

CITY_COLORS = {"Guangzhou":"#B64342","Chicago":"#0F4D92",
               "Tokyo":"#42949E","Kuala Lumpur":"#1B7A3D"}

def add_panel_label(ax,label,x=-0.06,y=1.02,fs=15,c="#272727",fw='bold'):
    ax.text(x,y,label,transform=ax.transAxes,fontsize=fs,fontweight=fw,color=c,ha='left',va='bottom')

lorenz_df = pd.read_excel(DATA_FILE, sheet_name="Lorenz_Curves")
quintile_df = pd.read_excel(DATA_FILE, sheet_name="Quintile_Summary")
cities = ["Guangzhou","Chicago","Tokyo","Kuala Lumpur"]

fig,(ax_a,ax_b)=plt.subplots(1,2,figsize=(15,7),gridspec_kw={"width_ratios":[1.05,1.0]})
fig.subplots_adjust(wspace=0.28,top=0.93,bottom=0.13,left=0.07,right=0.98)

# Panel a: Lorenz curves
for city in cities:
    cd=lorenz_df[(lorenz_df["City"]==city)&(lorenz_df["Cumulative_Population_pct"]>=0)].sort_values("Cumulative_Population_pct")
    if cd.empty: continue
    gini=cd["Gini"].iloc[0]; color=CITY_COLORS.get(city,"#333")
    ax_a.plot(cd["Cumulative_Population_pct"],cd["Cumulative_Benefit_pct"],
              color=color,linewidth=2.4,label=f"{city} (Gini={gini:.3f})")
ax_a.plot([0,100],[0,100],linestyle="--",color="#767676",linewidth=1.2,alpha=0.85,label="Equality")
ax_a.set_xlim(0,100); ax_a.set_ylim(0,100)
ax_a.set_xlabel("Cumulative population (%)")
ax_a.set_ylabel("Cumulative flood protection benefit (%)")
ax_a.legend(loc="upper left",fontsize=11,handlelength=1.5,handletextpad=0.5,borderpad=0.3,labelspacing=0.3)
ax_a.grid(True,alpha=0.10,linewidth=0.5)
ax_a.set_aspect("equal",adjustable="box")
add_panel_label(ax_a,'a')

# Panel b: Quintile coverage as dot plot (individual data points; editorial E.6)
qdf=quintile_df.copy()
qdf["QL"]=qdf["Quintile"].replace({"Q1_Lowest":"Q1\n(Lowest)","Q2":"Q2","Q3":"Q3","Q4":"Q4","Q5_Highest":"Q5\n(Highest)"})
qorder=["Q1\n(Lowest)","Q2","Q3","Q4","Q5\n(Highest)"]
qdf["QL"]=pd.Categorical(qdf["QL"],categories=qorder,ordered=True)
x=np.arange(len(qorder)); n_cities=len(cities)
dot_offset=0.18
offsets=np.linspace(-dot_offset*(n_cities-1)/2,dot_offset*(n_cities-1)/2,n_cities)
CITY_MARKERS = {"Guangzhou":"o","Chicago":"s","Tokyo":"D","Kuala Lumpur":"^"}
for i,city in enumerate(cities):
    cd2=qdf[qdf["City"]==city].sort_values("QL")
    if cd2.empty: continue
    ax_b.plot(x+offsets[i],cd2["Coverage_pct"].values,
              linestyle="none",marker=CITY_MARKERS[city],markersize=9,
              markeredgecolor="white",markeredgewidth=0.7,
              color=CITY_COLORS.get(city,"#333"),label=city,zorder=3)
ax_b.set_xticks(x); ax_b.set_xticklabels(qorder)
ax_b.set_ylabel("Protection coverage (%)")
ax_b.legend(loc="upper left",fontsize=11,handlelength=1.5,handletextpad=0.5,borderpad=0.3,labelspacing=0.3)
ax_b.grid(True,axis="y",alpha=0.10,linewidth=0.5); ax_b.set_axisbelow(True)
ax_b.set_ylim(0,52)
add_panel_label(ax_b,'b')

fig.savefig(os.path.join(OUT_DIR,"Fig4_Lorenz_Equity.pdf"),dpi=600,facecolor="white",bbox_inches="tight",pad_inches=0.1)
plt.close(fig)
print(f"Saved: Fig4_Lorenz_Equity.pdf")
