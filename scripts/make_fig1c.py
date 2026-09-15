# -*- coding: utf-8 -*-
"""Fig1C：panel 缩减敏感性（设计 B2，修正管线统一 log1p+CLR）。
组均值 Spearman（区室 vs 淋巴亚型）随 panel 大小 35→25→20 的变化，
双 fold 分面；误差棒 = 跨 seed 的组均值 SD（诚实反映 seed 变异性）；
35-panel 基线 = lodo_v4_100ep 主运行（n=1，无误差棒）。
数据：results_server_final/runs/designB2/*/ckpt/<fold>/per_protein_metrics.csv
      results_server_final/runs/lodo_v4_100ep/ckpt/<fold>/per_protein_metrics.csv
输出：results/fig1c_panel_reduction.png（300 dpi）
（v1 版本基于 runs/designB 约定不一致管线，已作废，见 实验结果_设计B_panel缩减.md 顶部声明）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SR = os.path.join(ROOT, "results_server_final", "runs")
FOLDS = [("Tonsil (LODO holdout)", "holdout_tonsil"),
         ("Breast cancer (LODO holdout)", "holdout_breast_cancer")]
GROUPS = {
    "compartment": ["ACTA2", "PECAM1", "PTPRC_1", "PTPRC_2", "EPCAM", "KRT5", "VIM",
                    "CD14", "CD68", "CD163", "ITGAM", "ITGAX", "FCGR3A", "HLA_DRA"],
    "lymph_subtype": ["CD3E", "CD4", "CD8A", "CD19", "MS4A1", "PAX5", "CR2", "CXCR5",
                      "CCR7", "PDCD1", "CD27", "SDC1"],
}

SIZES = [("35", [f"base_seed{i}" for i in (1, 2, 3, 4)]),
         ("25", [f"size25_seed{i}" for i in (1, 2, 3, 4, 5)]),
         ("20", [f"size20_seed{i}" for i in (1, 2, 3, 4, 5)])]

def load(size, run):
    base_root = os.path.join(ROOT, "runs", "designB")     # base_seed1-4（20ep 基线）
    var_root = os.path.join(SR, "designB2")               # 服务器变体（20ep）
    root = base_root if run.startswith("base") else var_root
    fp_ = os.path.join(root, run, "ckpt")
    fold_map = {}
    for _, fold in FOLDS:
        p = os.path.join(fp_, fold, "per_protein_metrics.csv")
        if os.path.exists(p):
            fold_map[fold] = pd.read_csv(p)[["protein", "spearman"]]
    return fold_map

# 每 seed 的组均值（seed 优先聚合，与误差棒口径一致）
rows = []
for size, runs in SIZES:
    for run in runs:
        seed = int(run.split("seed")[-1]) if "seed" in run else 0
        for fold, d in load(size, run).items():
            for gname, prots in GROUPS.items():
                sub = d[d.protein.isin(prots)]
                if len(sub):
                    rows.append(dict(size=size, seed=seed, fold=fold, group=gname,
                                     group_mean=sub.spearman.mean()))
long = pd.DataFrame(rows)
stat = (long.groupby(["fold", "size", "group"])["group_mean"]
            .agg(["std", "count"]).reset_index())
# 点估计与正文同源：蛋白优先聚合（designB2_group_summary.csv）
gs = pd.read_csv("results/designB2_group_summary.csv")
stat["size"] = stat["size"].astype(str)
gs["size"] = gs["size"].astype(str)
stat = stat.merge(gs[["fold", "size", "group", "sp_mean"]],
                  on=["fold", "size", "group"])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
xmap = {"35": 0, "25": 1, "20": 2}
colors = {"compartment": "#c0392b", "lymph_subtype": "#2471a3"}
labels = {"compartment": "Compartment markers (14)",
          "lymph_subtype": "Lymphocyte-subtype markers (12)"}

for ax, (title, fold) in zip(axes, FOLDS):
    for gname in GROUPS:
        s = stat[(stat.fold == fold) & (stat.group == gname)].copy()
        s["x"] = s["size"].map(xmap)
        s = s.sort_values("x")
        yerr = s["std"].to_numpy(dtype=float).copy()
        yerr[np.isnan(yerr)] = 0.0
        # 标签避线：红（区室，线在上）放点上方，蓝（淋巴亚型）放点下方；白色描边垫底防穿透
        off = (0, 9) if gname == "compartment" else (0, -14)
        halo = [pe.withStroke(linewidth=1.8, foreground="white")]
        ax.errorbar(s.x, s.sp_mean, yerr=yerr, marker="o", ms=7, lw=2.2,
                    capsize=4, color=colors[gname], label=labels[gname])
        for _, r in s.iterrows():
            ax.annotate(f"{r.sp_mean:.3f}", (r.x, r.sp_mean),
                        textcoords="offset points", xytext=off,
                        ha="center", fontsize=6, color=colors[gname],
                        path_effects=halo, zorder=6)
    ax.axhline(0, color="#999999", lw=0.8, ls=":")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["35 markers\n(4 seeds, 20 ep)",
                        "25 markers\n(5 seeds)", "20 markers\n(5 seeds)"])
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Antibody panel size")
    ax.set_xlim(-0.35, 2.35)
    # 上下留白，保证标签和图例不顶到边框
    ymax = (stat[stat.fold == fold]["sp_mean"] +
            stat[stat.fold == fold]["std"].fillna(0)).max()
    ymin = (stat[stat.fold == fold]["sp_mean"] -
            stat[stat.fold == fold]["std"].fillna(0)).min()
    ax.set_ylim(ymin - 0.10, ymax + 0.12)

axes[0].set_ylabel("Group-mean per-protein Spearman\n(seed mean ± SD)")
axes[0].legend(loc="upper left", fontsize=6.5, frameon=False)
fig.suptitle("D  Panel-reduction sensitivity (design B2, corrected pipeline): compartment vs lymphocyte-subtype markers",
             fontsize=8.5, y=1.02)
fig.tight_layout()
fig.savefig("results/fig1c_panel_reduction.png", dpi=600, bbox_inches="tight")
print("[saved] results/fig1c_panel_reduction.png")
print(stat.to_string(index=False))
