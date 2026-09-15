# -*- coding: utf-8 -*-
"""Fig4 v2（架构对比 + matched-spot 补正）——生成 results/fig4_architecture.png。

在原 make_fig3_fig4.py 的 Fig4 基础上扩为 6 面板：
  A: 3-seed 满预算稳定性（不变）
  B: DGAT 700 基因 spot 过滤保留率（不变）
  C: reference pipeline 逐蛋白散点 DGAT vs ridge（不变）
  D: reference pipeline 配对差 DGAT − ridge（不变）
  E: [新] 三种过滤条件下的 DGAT 总均值 vs DPAS-Graph/ridge 基准线
  F: [新] matched-spot 重跑配对差 DGAT_matched − ridge（4/31）

全部数字从存档结果表实时计算，无硬编码统计值。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

C_COMP, C_LYMPH, C_OTHER = "#c0392b", "#2980b9", "#95a5a6"

# ---------- 公共：31 标志物分组 ----------
imp = pd.read_csv("results/designC_marker_importance.csv")
grp = dict(zip(imp.dropped, imp.dropped_group))
def gcol(p): return {"compartment": C_COMP, "lymph_subtype": C_LYMPH}.get(grp.get(p), C_OTHER)
markers = list(grp.keys())

tot = pd.read_csv("results/internal_vs_external_summary.csv")
ms = pd.read_csv("results/lodo100ep_multiseed_summary.csv")
ridge = pd.read_csv("results/competitor_benchmark_per_protein.csv")
ridge_m = (ridge[ridge.model == "ridge_pc"].groupby("protein").spearman.mean())
dgat = pd.read_csv("server_round2/dgat_lodo_gse_per_protein.csv")
dgat_s = dgat.groupby(["sample", "protein"]).spearman.mean().reset_index()
dgat_m = dgat_s.groupby("protein").spearman.mean()
dgat_spots = dgat.groupby("sample").n_spots.first()
tot_ext = tot[tot["set"] == "外部 GSE"].set_index("sample").n_spots
ridge_mean = float(ridge[ridge.model == "ridge_pc"].groupby("sample").spearman.mean().mean())

# ---------- matched-spot 重跑数据 ----------
p500 = pd.read_csv("results_server_final/dgat_uniformqc/dgat_panel500_gse_per_protein.csv")
p500_m = p500.groupby(["sample", "protein"]).spearman.mean().reset_index().groupby("protein").spearman.mean()
nof = pd.read_csv("results_server_final/dgat_uniformqc/dgat_nofilter_gse_per_protein.csv")
nof_m = nof.groupby(["sample", "protein"]).spearman.mean().reset_index().groupby("protein").spearman.mean()
dpas_seed0 = float(ms[ms.seed == 0].spear_marker_mean.iloc[0])
dpas_3seed = (float(ms.spear_marker_mean.mean()), float(ms.spear_marker_mean.std(ddof=1)))

fig = plt.figure(figsize=(7.2, 9.2))
fig.suptitle("Architecture comparison under the identical external protocol", fontsize=9.5, y=0.995)
gs = fig.add_gridspec(3, 2, hspace=0.62, wspace=0.42)

# A: 3-seed stability at 100 epochs
ax = fig.add_subplot(gs[0, 0])
seeds = ms.seed.astype(int).tolist()
vals = ms.spear_marker_mean.tolist()
floors = ms.spear_isotype_mean.tolist()
ax.bar(range(len(seeds)), vals, color="#e67e22", width=0.55, zorder=3)
for i, (v, f) in enumerate(zip(vals, floors)):
    ax.hlines(f, i - 0.27, i + 0.27, color="k", ls=":", lw=1.2, zorder=4)
ax.axhline(ridge_mean, color="#16a085", lw=1.2, ls="--", zorder=2)
ax.text(len(seeds) - 0.45, ridge_mean + 0.008, f"ridge {ridge_mean:.3f}",
        ha="right", fontsize=7, color="#16a085")
m, sd = np.mean(vals), np.std(vals, ddof=1)
ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=7.5)
ax.set_ylabel("external mean Spearman")
ax.set_ylim(0, 0.45)
ax.set_title(f"A  Full-budget stability: {m:.3f} ± {sd:.3f} across seeds",
             fontsize=8.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)

# B: DGAT spot retention
ax = fig.add_subplot(gs[0, 1])
samples = list(tot_ext.index)
ret = [100.0 * float(dgat_spots.get(s, np.nan)) / float(tot_ext[s]) for s in samples]
ax.bar(range(len(samples)), ret, color="#8e44ad", width=0.6, zorder=3)
for i, (s, r) in enumerate(zip(samples, ret)):
    ax.text(i, r + 1.5, f"{dgat_spots.get(s, 0):.0f}/{tot_ext[s]:.0f}",
            ha="center", fontsize=6.5)
ax.axhline(100, color="k", lw=0.6, ls=":")
ax.set_xticks(range(len(samples))); ax.set_xticklabels(samples, fontsize=7)
ax.set_ylabel("spots retained (%)")
ax.set_ylim(0, 112)
ax.set_title("B  DGAT spot filter: retention per section", fontsize=8.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)

# C: ridge vs DGAT per-protein scatter (reference pipeline)
ax = fig.add_subplot(gs[1, 0])
for p in markers:
    ax.scatter(ridge_m[p], dgat_m[p], s=30, c=gcol(p), zorder=3,
               edgecolor="w", linewidth=0.4)
lim = [-0.15, 0.85]
ax.plot(lim, lim, "k--", lw=0.8, zorder=2)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("ridge on PCA-50 (mean external ρ)")
ax.set_ylabel("DGAT (mean external ρ)")
for p in ["PAX5", "CXCR5", "PDCD1", "MS4A1", "CD4", "CD8A"]:
    ax.annotate(p, (ridge_m[p], dgat_m[p]), fontsize=6, xytext=(3, 3),
                textcoords="offset points")
nwin_ref = int((dgat_m[markers] - ridge_m[markers] > 0).sum())
ax.set_title(f"C  Reference pipeline: DGAT vs ridge ({nwin_ref}/31 to DGAT)",
             fontsize=8.5, loc="left")
ax.text(0.97, 0.05, f"means: DGAT {dgat_m[markers].mean():.3f} vs ridge {ridge_m[markers].mean():.3f}",
        transform=ax.transAxes, fontsize=6.5, va="bottom", ha="right",
        bbox=dict(fc="w", ec="0.7", alpha=0.85))
ax.spines[["top", "right"]].set_visible(False)

# D: paired difference DGAT(ref) - ridge
ax = fig.add_subplot(gs[1, 1])
diff = (dgat_m[markers] - ridge_m[markers]).sort_values()
ax.barh(range(len(diff)), diff.values, color=[gcol(p) for p in diff.index],
        height=0.72, zorder=3)
ax.set_yticks(range(len(diff)))
ax.set_yticklabels(diff.index, fontsize=5.5)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("DGAT − ridge  (Spearman)")
ax.set_title(f"D  DGAT wins {nwin_ref} of 31 markers vs ridge", fontsize=8.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)

# E: matched-spot condition summary
ax = fig.add_subplot(gs[2, 0])
conds = [
    ("reference\n(700-of-panel)", dgat_m[markers].mean(), "#8e44ad"),
    ("panel-internal\n500", p500_m[markers].mean(), "#9b59b6"),
    ("matched\nspot sets", nof_m[markers].mean(), "#2c3e50"),
]
ax.bar(range(len(conds)), [c[1] for c in conds], color=[c[2] for c in conds],
       width=0.55, zorder=3)
for i, (lab, v, _) in enumerate(conds):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5, zorder=5)
ax.axhline(dpas_seed0, color="#e67e22", lw=1.3, ls="--", zorder=2)
ax.axhline(ridge_mean, color="#16a085", lw=1.3, ls="--", zorder=2)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], color="#e67e22", lw=1.3, ls="--",
           label=f"DPAS-Graph seed 0 ({dpas_seed0:.3f})"),
    Line2D([], [], color="#16a085", lw=1.3, ls="--",
           label=f"ridge ({ridge_mean:.3f})"),
], loc="upper right", fontsize=6.3, framealpha=0.92)
ax.set_xticks(range(len(conds)))
ax.set_xticklabels(["DGAT " + c[0] for c in conds], fontsize=6.2)
ax.set_ylabel("external mean Spearman (31 markers)")
ax.set_ylim(0, 0.45)
ax.set_title("E  Spot-set confound removed: DGAT ≈ DPAS-Graph",
             fontsize=8.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)

# F: paired difference DGAT(matched) - ridge
ax = fig.add_subplot(gs[2, 1])
diff_m = (nof_m[markers] - ridge_m[markers]).sort_values()
ax.barh(range(len(diff_m)), diff_m.values, color=[gcol(p) for p in diff_m.index],
        height=0.72, zorder=3)
ax.set_yticks(range(len(diff_m)))
ax.set_yticklabels(diff_m.index, fontsize=5.5)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("DGAT (matched) − ridge  (Spearman)")
nwin_m = int((diff_m > 0).sum())
ax.set_title(f"F  Matched spot sets: DGAT wins {nwin_m} of 31 vs ridge",
             fontsize=8.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)

plt.savefig("results/fig4_architecture.png", dpi=600)
plt.close(fig)
print(f"fig4 v2 saved | ref={dgat_m[markers].mean():.4f} panel500={p500_m[markers].mean():.4f} "
      f"matched={nof_m[markers].mean():.4f} | DPAS seed0={dpas_seed0:.4f} 3seed={dpas_3seed[0]:.4f}±{dpas_3seed[1]:.4f} "
      f"| ridge={ridge_mean:.4f} | ref wins {nwin_ref}/31, matched wins {nwin_m}/31")
