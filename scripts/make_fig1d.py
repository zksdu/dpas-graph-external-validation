# -*- coding: utf-8 -*-
"""Fig1D：外部验证 SP 均值 —— 训练预算 × 模型对比条形图。

数据源（全部为已归档真实结果）：
- DPAS-Graph 20ep：results/gse_external_per_protein.csv（proc，两 ckpt × 4 样本，31 标志物逐蛋白均值）
- DPAS-Graph 100ep：gse_external_summary_100ep.csv（proc，8 行样本级 SP 均值，从服务器日志重建）
- 竞品：results/competitor_benchmark_summary.csv
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Arial", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DPAS 20ep（逐蛋白 → 标志物均值）
e20 = pd.read_csv("results/gse_external_per_protein.csv")
dpas20 = float(e20[(e20.input == "proc") & (e20.isotype == False)].spearman.mean())
iso20 = float(e20[(e20.input == "proc") & (e20.isotype == True)].spearman.mean())

# DPAS 100ep（样本级 SP 均值，proc）
s100 = pd.read_csv("gse_external_summary_100ep.csv")
dpas100 = float(s100[s100["mode"] == "proc"].SP.mean())
iso100 = float(s100[s100["mode"] == "proc"].isoSP.mean())

# 竞品
cb = pd.read_csv("results/competitor_benchmark_summary.csv").set_index("model").sp_mean

bars = [
    ("DPAS-Graph\n20 epochs\n(published budget)", dpas20, "#c0392b", iso20),
    ("DPAS-Graph\n100 epochs\n(full budget)", dpas100, "#e67e22", iso100),
    ("kNN (k=15)\non RNA-PCA", float(cb["knn15"]), "#2980b9", None),
    ("Ridge\n(RNA-PCA-50)", float(cb["ridge_pc"]), "#16a085", None),
    ("Ridge\n(single-tissue\nmodels, 2-fold avg)", float(cb["ridge_pc_single_avg"]), "#16a085", None),
]

fig, ax = plt.subplots(figsize=(7.2, 3.9))
x = range(len(bars))
for i, (lab, v, c, iso) in enumerate(bars):
    ax.bar(i, v, width=0.62, color=c, zorder=3)
    ax.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold")
    if iso is not None:
        ax.hlines(iso, i - 0.31, i + 0.31, color="k", ls=":", lw=1.2, zorder=4)
        ax.text(i + 0.33, iso, f"isotype floor {iso:.3f}", fontsize=6.5,
                va="center", color="0.35", zorder=10)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(list(x))
ax.set_xticklabels([b[0] for b in bars], fontsize=7)
ax.set_ylabel("external GSE263617 mean Spearman ρ\n(proc mode, marker proteins)")
ax.set_title("D  External validation: training budget vs model architecture\n(central negative result)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/fig1d_budget_vs_baselines.png", dpi=600)
print(f"saved fig1d | dpas20={dpas20:.4f} dpas100={dpas100:.4f} "
      f"ridge={cb['ridge_pc']:.4f} single={cb['ridge_pc_single_avg']:.4f} knn15={cb['knn15']:.4f}")
