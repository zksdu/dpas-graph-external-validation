# -*- coding: utf-8 -*-
"""论文主图：
Fig1 基准与外部验证（内部 LODO vs GSE263617 外部）；
Fig2 PE 空间应用拼版（niche 热图 + 疾病轴热图）。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

matplotlib.rcParams["font.sans-serif"] = ["Arial", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------- Figure 1 ----------------
loads = {
    "LODO holdout\ntonsil (int.)": "runs/lodo_v3/ckpt/holdout_tonsil/per_protein_metrics.csv",
    "LODO holdout\nbreast (int.)": "runs/lodo_v3/ckpt/holdout_breast_cancer/per_protein_metrics.csv",
}
internal = []
for name, f in loads.items():
    d = pd.read_csv(f)[["protein", "spearman"]]
    d["set"] = name
    internal.append(d)
internal = pd.concat(internal)

ext = pd.read_csv("results/gse_external_per_protein.csv")
ext_proc = ext[ext.input == "proc"]
ext_groups = {
    "GSE ext.\n(tonsil ckpt)": ext_proc[ext_proc.ckpt == "holdout_tonsil"],
    "GSE ext.\n(breast ckpt)": ext_proc[ext_proc.ckpt == "holdout_breast_cancer"],
}
ext_agg = pd.concat(
    [g.groupby(["protein", "isotype"])[["spearman"]].mean().reset_index().assign(set=k)
     for k, g in ext_groups.items()])
ext_agg["set"] = ext_agg["set"]

allp = pd.concat([internal, ext_agg[["protein", "spearman", "isotype", "set"]]])
allp["isotype"] = allp["isotype"].fillna(False).astype(bool)
order = list(loads.keys()) + list(ext_groups.keys())
iso = allp[allp.isotype]
mk = allp[~allp.isotype]

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9),
                         gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
rng = np.random.default_rng(0)
for i, s in enumerate(order):
    d1 = mk[mk.set == s]
    d0 = iso[iso.set == s]
    x1 = i + rng.uniform(-0.15, 0.15, len(d1))
    x0 = i + rng.uniform(-0.15, 0.15, len(d0))
    ax.scatter(x1, d1.spearman, s=18, c="#c0392b", zorder=3,
               label="marker proteins (31)" if i == 0 else None)
    ax.scatter(x0, d0.spearman, s=26, c="#7f8c8d", marker="s", zorder=3,
               label="isotype controls (4)" if i == 0 else None)
    ax.hlines(d1.spearman.median(), i - 0.25, i + 0.25, color="#c0392b", lw=2)
ax.axhline(0, color="k", lw=0.6, ls="--")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, fontsize=7)
ax.set_ylabel("per-protein Spearman ρ (pred vs true)")
ax.set_title("A  Internal LODO vs fully external validation (per-protein)", fontsize=8.5)
ax.legend(loc="lower left", fontsize=6.5)

ax = axes[1]
m_int = internal.groupby("protein").spearman.mean()
m_ext = (ext_agg[~ext_agg.isotype]
         .groupby("protein").spearman.mean())   # 两 checkpoint 取均值
common = m_int.index.intersection(m_ext.index)
x, y = m_int[common], m_ext[common]
ax.scatter(x, y, s=26, c="#2c3e50")
lim = [-0.45, 0.65]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("internal LODO (mean of 2 folds) ρ")
ax.set_ylabel("external GSE263617 ρ")
ax.set_title("B  Per-protein generalization gap (diagonal = no gap)", fontsize=8.5)
for g in ["ACTA2", "PECAM1", "PTPRC_2", "CXCR5", "CD14", "CD4", "CR2", "PAX5", "CD68"]:
    if g in common:
        ax.annotate(g, (m_int[g], m_ext[g]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
plt.tight_layout()
plt.savefig("results/fig1_benchmark_external.png", dpi=600)
plt.close()
print("saved fig1")

# ---------------- Figure 2 ----------------
panels_row1 = [("results/niche2_sectionA_heatmap_adj.png", "A"),
               ("results/niche2_sectionB_heatmap_adj.png", "B")]
panel_row2 = ("results/disease_protein_assoc_heatmap.png", "C")
fig = plt.figure(figsize=(7.5, 9.7))
gs = fig.add_gridspec(2, 2, hspace=0.16, wspace=0.05)
for j, (f, t) in enumerate(panels_row1):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(mpimg.imread(f))
    ax.axis("off")
    ax.set_title(t, fontsize=11, fontweight="bold", loc="left")
ax = fig.add_subplot(gs[1, :])
ax.imshow(mpimg.imread(panel_row2[0]))
ax.axis("off")
ax.set_title(panel_row2[1], fontsize=11, fontweight="bold", loc="left")
plt.savefig("results/fig2_spatial_application.png", dpi=600, bbox_inches="tight")
plt.close()
print("saved fig2")
