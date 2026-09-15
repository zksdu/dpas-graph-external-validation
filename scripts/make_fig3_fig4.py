# -*- coding: utf-8 -*-
"""Fig3（机制与对照）与 Fig4（架构对比）——全部数字从存档结果表实时计算，无硬编码统计值。

面板按正文首次引用顺序排列（Fig3A 在 §5 切片身份、Fig4A 在 §6 预算、其余随节）：

- Figure 3（机制归因与对照诊断）= results/fig3_mechanism_controls.png
  Fig3A: results/visium_eope_score.csv 逐 spot 签名分 → Mann-Whitney AUC(A>B)（切片身份）
  Fig3B: results/internal_vs_external_summary.csv（6 样本模型 SP + 同型地板）
  Fig3C: results/designC_marker_importance.csv（31 标志物消融）
  Fig3D: results/rna_coverage_diagnosis.csv（两数据集 RNA 覆盖）
- Figure 4（架构对比）= results/fig4_architecture.png
  Fig4A: results/lodo100ep_multiseed_summary.csv（3 seeds 满预算稳定性）
  Fig4B: results/competitor_benchmark_per_protein.csv (ridge_pc) +
         server_round2/dgat_lodo_gse_per_protein.csv（逐蛋白散点）
  Fig4C: 同上（配对差瀑布）
  Fig4D: DGAT 700 基因 spot 过滤保留率（dgat 表 vs internal_vs_external_summary.csv）

分组（compartment/lymph_subtype/other）取自 designC_marker_importance.csv 的 dropped_group。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

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
dgat_s = dgat.groupby(["sample", "protein"]).spearman.mean().reset_index()  # 样本先行（2 折平均）
dgat_m = dgat_s.groupby("protein").spearman.mean()
dgat_spots = dgat.groupby("sample").n_spots.first()
tot_ext = tot[tot["set"] == "外部 GSE"].set_index("sample").n_spots
ridge_mean = float(ridge[ridge.model == "ridge_pc"].groupby("sample").spearman.mean().mean())

# ================= Figure 3：机制归因与对照诊断 =================
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.1))
fig.suptitle("Mechanism attribution and control diagnostics", fontsize=9.5, y=0.995)

# A: section identity AUC (recomputed from per-spot signature scores)
ax = axs[0, 0]
sc = pd.read_csv("results/visium_eope_score.csv")
sigs = ["vSTB1", "vSTB2", "vSTBjuv", "vCTB", "vHBC", "vVEC"]
if set(sc.section.unique()) >= {"A", "B"}:
    A, B = sc[sc.section == "A"], sc[sc.section == "B"]
else:  # 分别从 _A/_B 文件合并
    a = pd.read_csv("results/visium_eope_score_A.csv")
    b = pd.read_csv("results/visium_eope_score_B.csv")
    a["section"], b["section"] = "A", "B"
    AB = pd.concat([a, b], ignore_index=True)
    A, B = AB[AB.section == "A"], AB[AB.section == "B"]
aucs = {}
for s in sigs:
    xa = A[f"score_{s}"].dropna().values
    xb = B[f"score_{s}"].dropna().values
    u, p = mannwhitneyu(xa, xb, alternative="two-sided")
    aucs[s] = u / (len(xa) * len(xb))
# 合并签名
xa = A["score"].dropna().values; xb = B["score"].dropna().values
u, p = mannwhitneyu(xa, xb, alternative="two-sided")
aucs["combined"] = u / (len(xa) * len(xb))
names = sigs + ["combined"]
vals_c = [aucs[n] for n in names]
ax.barh(range(len(names)), vals_c, color="#2c3e50", height=0.6, zorder=3)
ax.axvline(0.5, color="#c0392b", lw=1.1, ls="--")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
ax.set_xlim(0.40, 0.60)
ax.invert_yaxis()
for i, v in enumerate(vals_c):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=6.5)
ax.set_xlabel("AUC (A vs B, Mann–Whitney U)")
ax.set_title("A  Section identity unresolved: all AUCs ≈ 0.5", fontsize=8.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)

# B: internal vs external per sample
ax = axs[0, 1]
iv = tot.copy()
labels = iv["sample"].tolist()
colors = ["#7f8c8d" if s == "内部 LODO" else "#c0392b" for s in iv["set"]]
iv["set_en"] = iv["set"].map({"内部 LODO": "Internal LODO", "外部 GSE": "External GSE"})
iv["set_short"] = iv["set"].map({"内部 LODO": "int.", "外部 GSE": "ext."})
ax.bar(range(len(iv)), iv.model_spear, color=colors, width=0.6, zorder=3)
ax.scatter(range(len(iv)), iv.iso_spear, marker="_", s=220, color="k", zorder=4,
           label="isotype floor")
ax.set_xticks(range(len(iv)))
NAME_SHORT = {"tonsil": "tonsil", "breast_cancer": "breast", "A1LN": "A1LN",
              "A1TNSL": "A1TNSL", "D1LN": "D1LN", "D1TNSL": "D1TNSL"}
short_lab = [NAME_SHORT.get(t, t) for t in labels]
ax.set_xticklabels([f"{t}\n({g})" for t, g in zip(short_lab, iv["set_short"])], fontsize=6)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("mean Spearman (20-epoch ckpts)")
ax.legend(fontsize=6.5, loc="upper right")
ax.set_title("B  Internal LODO vs external sections",
             fontsize=8.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)

# C: designC ablation
ax = axs[1, 0]
d = imp.set_index("dropped").mean_delta.sort_values()
ax.barh(range(len(d)), d.values, color=[gcol(p) for p in d.index], height=0.72, zorder=3)
ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=5.5)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("mean ΔSpearman when marker dropped")
ax.set_title("C  Per-marker ablation (31 retrainings)",
             fontsize=8.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)

# D: RNA coverage
ax = axs[1, 1]
rc = pd.read_csv("results/rna_coverage_diagnosis.csv")
mk = {"tonsil": "o", "breast": "^"}
for ds, sub in rc.groupby("dataset"):
    for gname, c in [("compartment", C_COMP), ("lymph_subtype", C_LYMPH)]:
        key = "lymph" if gname == "lymph_subtype" else gname  # rna_coverage_diagnosis.csv 用 "lymph"
        s2 = sub[(sub.group == key) & (sub.in_panel.astype(str) == "True")]
        assert len(s2) > 0, f"empty subset: {ds}/{gname} (key={key})"
        ax.scatter(s2.mean_expr, s2.detect_rate, s=30, c=c, marker=mk.get(ds, "o"),
                   alpha=0.75, edgecolor="w", linewidth=0.3, zorder=3,
                   label=f"{gname}, {ds}")
ax.set_xscale("symlog", linthresh=0.1)
ax.set_xlabel("namesake expression (log10)")
ax.set_ylabel("detection rate (fraction of spots)")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
ax.set_title("D  RNA coverage of namesake genes",
             fontsize=8.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("results/fig3_mechanism_controls.png", dpi=600)
plt.close(fig)
print(f"fig3 saved | AUCs {({k: round(v,3) for k,v in aucs.items()})} "
      f"| ablation range [{d.min():.3f},{d.max():.3f}] | n rc rows {len(rc)}")

# ================= Figure 4：架构对比 =================
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.1))
fig.suptitle("Architecture comparison under the identical external protocol", fontsize=9.5, y=0.995)

# A: 3-seed stability at 100 epochs
ax = axs[0, 0]
seeds = ms.seed.astype(int).tolist()
vals = ms.spear_marker_mean.tolist()
floors = ms.spear_isotype_mean.tolist()
ax.bar(range(len(seeds)), vals, color="#e67e22", width=0.55, zorder=3)
for i, (v, f) in enumerate(zip(vals, floors)):
    ax.hlines(f, i - 0.27, i + 0.27, color="k", ls=":", lw=1.2, zorder=4)
ax.axhline(ridge_mean, color="#16a085", lw=1.2, ls="--", zorder=2)
ax.text(len(seeds) - 0.45, ridge_mean + 0.006, f"ridge {ridge_mean:.3f}",
        ha="right", fontsize=7, color="#16a085")
m, sd = np.mean(vals), np.std(vals, ddof=1)
ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=7.5)
ax.set_ylabel("external mean Spearman")
ax.set_ylim(0, 0.45)
ax.set_title(f"A  Full-budget stability: {m:.3f} ± {sd:.3f} across seeds",
             fontsize=8.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)

# C: ridge vs DGAT per-protein scatter
ax = axs[1, 0]
for p in markers:
    ax.scatter(ridge_m[p], dgat_m[p], s=34, c=gcol(p), zorder=3,
               edgecolor="w", linewidth=0.4)
lim = [-0.15, 0.85]
ax.plot(lim, lim, "k--", lw=0.8, zorder=2)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("ridge on PCA-50 (mean external ρ)")
ax.set_ylabel("DGAT (mean external ρ)")
for p in ["PAX5", "CXCR5", "PDCD1", "MS4A1", "CD4", "CD8A"]:
    ax.annotate(p, (ridge_m[p], dgat_m[p]), fontsize=6.5, xytext=(3, 3),
                textcoords="offset points")
ax.set_title("C  Per-protein: DGAT vs ridge (31 markers)", fontsize=8.5, loc="left")
ax.text(0.03, 0.96, f"means: DGAT {dgat_m[markers].mean():.3f} vs ridge {ridge_m[markers].mean():.3f}",
        transform=ax.transAxes, fontsize=7, va="top",
        bbox=dict(fc="w", ec="0.7", alpha=0.85))
ax.spines[["top", "right"]].set_visible(False)

# D: paired difference DGAT - ridge
ax = axs[1, 1]
diff = (dgat_m[markers] - ridge_m[markers]).sort_values()
ax.barh(range(len(diff)), diff.values, color=[gcol(p) for p in diff.index],
        height=0.72, zorder=3)
ax.set_yticks(range(len(diff)))
ax.set_yticklabels(diff.index, fontsize=5.5)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("DGAT − ridge  (Spearman)")
nwin = int((diff > 0).sum())
ax.set_title(f"D  DGAT wins {nwin} of 31 markers vs ridge", fontsize=8.5, loc="left")
ax.spines[["top", "right"]].set_visible(False)

# B: DGAT spot retention
ax = axs[0, 1]
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

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("results/fig4_architecture.png", dpi=600)
plt.close(fig)
print(f"fig4 saved | ridge={ridge_mean:.4f} DGAT={dgat_m[markers].mean():.4f} "
      f"DGAT wins {nwin}/31 | seeds {[round(v,3) for v in vals]} mean {m:.3f}±{sd:.3f} "
      f"| retention {[round(r,1) for r in ret]}")
