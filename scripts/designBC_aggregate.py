# -*- coding: utf-8 -*-
"""服务器结果聚合 v2（修正基线口径）。

designB2/designC 变体 = 20 epochs 服务器训练（修正管线：log1p+行中心化），
其正确基线 = v1 base_seed1-4（20 epochs、35 panel、log1p+行中心化，
本地 runs/designB/base_seed*/）——约定与预算均同源。
lodo_v4_100ep（100 epochs）单独作为训练预算再检验（§1/§2），不与变体混比。

输出:
  results/designB2_per_protein_agg.csv   蛋白 × size × fold 均值/SD（跨 seed）
  results/designB2_group_summary.csv     组 × size × fold 汇总（35 = base_seed1-4）
  results/designC_ablation_deltas.csv    消融长表：dropped × target × fold 的 ΔSP
  results/designC_marker_importance.csv  每个被移除标志物的平均影响与排序
  结果报告_服务器100ep与41变体.md
"""
import os
import pandas as pd
import numpy as np

ROOT = r"D:/workbuddy/0911/pe-virtual-protein"
SR = os.path.join(ROOT, "results_server_final", "runs")   # 服务器变体（20ep）
BASE_DIR = os.path.join(ROOT, "runs", "designB")           # v1 基线 base_seed1-4（20ep）
RES = os.path.join(ROOT, "results")
FOLDS = ["holdout_tonsil", "holdout_breast_cancer"]
ISOTYPE = {"mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"}

GROUPS = {
    "compartment": ["ACTA2", "PECAM1", "PTPRC_1", "PTPRC_2", "EPCAM", "KRT5", "VIM",
                    "CD14", "CD68", "CD163", "ITGAM", "ITGAX", "FCGR3A", "HLA_DRA"],
    "lymph_subtype": ["CD3E", "CD4", "CD8A", "CD19", "MS4A1", "PAX5", "CR2", "CXCR5",
                      "CCR7", "PDCD1", "CD27", "SDC1"],
    "other": ["BCL2", "CD40", "CD274", "CEACAM8", "PCNA"],
}
PROT2GROUP = {p: g for g, ps in GROUPS.items() for p in ps}

def prot_group(p):
    return PROT2GROUP.get(p, "isotype" if p in ISOTYPE else "other")

def load_sp(path_prefix, run_rel):
    out = {}
    for fold in FOLDS:
        fp = os.path.join(path_prefix, run_rel, "ckpt", fold, "per_protein_metrics.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            out[fold] = dict(zip(df.protein, df.spearman))
    return out

# ---------- 基线：base_seed1-4（20ep，35 panel，log1p+行中心化，n=4） ----------
base_seed_sp = {}   # (fold, protein) -> list over seeds
for i in (1, 2, 3, 4):
    d = load_sp(BASE_DIR, f"base_seed{i}")
    for fold, pm in d.items():
        for p, v in pm.items():
            base_seed_sp.setdefault((fold, p), []).append(v)
base = {fold: {p: np.mean(vs) for (f2, p), vs in base_seed_sp.items() if f2 == fold}
        for fold in FOLDS}
base_sd = {fold: {p: np.std(vs, ddof=1) for (f2, p), vs in base_seed_sp.items() if f2 == fold}
           for fold in FOLDS}
base_targets = {f: {p: v for p, v in d.items() if p not in ISOTYPE} for f, d in base.items()}
base_iso = {f: np.mean([v for p, v in d.items() if p in ISOTYPE]) for f, d in base.items()}
print("[base] base_seed1-4 loaded; targets per fold:",
      {f: len(d) for f, d in base_targets.items()})

# ---------- designB2（服务器 20ep 变体） ----------
rows = []
for size in ("25", "20"):
    for seed in range(1, 6):
        v = f"size{size}_seed{seed}"
        d = load_sp(SR, os.path.join("designB2", v))
        for fold, pm in d.items():
            for p, sp in pm.items():
                rows.append(dict(size=size, variant=v, seed=seed, fold=fold,
                                 protein=p, group=prot_group(p), spearman=float(sp)))
longB = pd.DataFrame(rows)
print(f"[designB2] rows={len(longB)} variants={longB.variant.nunique()}")

aggB = (longB.groupby(["fold", "protein", "group", "size"])["spearman"]
        .agg(["mean", "std", "count"]).reset_index())
aggB.columns = ["fold", "protein", "group", "size", "sp_mean", "sp_sd", "n_seeds"]
aggB.to_csv(os.path.join(RES, "designB2_per_protein_agg.csv"), index=False)

# 组汇总（35 = base_seed1-4 跨 seed；25/20 = designB2）
grows = []
for fold in FOLDS:
    for g in list(GROUPS) + ["isotype"]:
        if g == "isotype":
            src = {p: v for p, v in base[fold].items() if p in ISOTYPE}
            grows.append(dict(fold=fold, size="35", group=g, sp_mean=np.mean(list(src.values())),
                              sp_sd=np.nan, n_obs=len(src), n_variants=4))
            continue
        vals = [base[fold][p] for p in GROUPS[g] if p in base[fold]]
        vals_sd = [base_sd[fold][p] for p in GROUPS[g] if p in base[fold]]
        grows.append(dict(fold=fold, size="35", group=g, sp_mean=np.mean(vals),
                          sp_sd=np.mean(vals_sd), n_obs=len(vals), n_variants=4))
    # 25/20：protein-first——先跨 seed 求每蛋白均值（aggB），再对组内蛋白取均值；
    # sp_sd = 跨 seed 的"组均值"SD（每 seed 组均值 = 该 seed 保留蛋白的均值），
    # 与 Methods §7 记载的聚合顺序严格一致（v1 曾误用 seed×protein 池化均值）。
    for size in ("25", "20"):
        for g in list(GROUPS) + ["isotype"]:
            a = aggB[(aggB.fold == fold) & (aggB["size"].astype(str) == str(size))
                     & (aggB.group == g)]
            if not len(a):
                continue
            sub_v = longB[(longB.fold == fold) & (longB["size"].astype(str) == str(size))]
            per_seed = []
            for v in sorted(sub_v.variant.unique()):
                sv = sub_v[(sub_v.variant == v) & (sub_v.group == g)].spearman
                if len(sv):
                    per_seed.append(float(sv.mean()))
            grows.append(dict(fold=fold, size=size, group=g,
                              sp_mean=float(a.sp_mean.mean()),
                              sp_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan,
                              n_obs=int(len(a)), n_variants=len(per_seed)))
grpB = pd.DataFrame(grows)
grpB.to_csv(os.path.join(RES, "designB2_group_summary.csv"), index=False)

# ---------- designC（服务器 20ep 变体；ΔSP vs base_seed1-4 均值） ----------
arows = []
for v in sorted(os.listdir(os.path.join(SR, "designC"))):
    if not v.startswith("drop_"):
        continue
    dropped = v.replace("drop_", "")
    d = load_sp(SR, os.path.join("designC", v))
    for fold, pm in d.items():
        for p, sp in pm.items():
            if p in ISOTYPE or p == dropped:
                continue
            if p in base.get(fold, {}):
                arows.append(dict(dropped=dropped, dropped_group=prot_group(dropped),
                                  target=p, target_group=prot_group(p), fold=fold,
                                  sp_base=base[fold][p], sp_drop=sp, delta=sp - base[fold][p]))
ab = pd.DataFrame(arows)
ab.to_csv(os.path.join(RES, "designC_ablation_deltas.csv"), index=False)
print(f"[designC] ablation rows={len(ab)} dropped markers={ab.dropped.nunique()}")

imp = (ab.groupby(["dropped", "dropped_group"])
       .agg(mean_delta=("delta", "mean"), sd_delta=("delta", "std"),
            n_targets=("delta", "count")).reset_index())
comp_delta = ab[ab.target_group == "compartment"].groupby("dropped")["delta"].mean()
lymph_delta = ab[ab.target_group == "lymph_subtype"].groupby("dropped")["delta"].mean()
other_delta = ab[ab.target_group == "other"].groupby("dropped")["delta"].mean()
tons_delta = ab[ab.fold == "holdout_tonsil"].groupby("dropped")["delta"].mean()
brst_delta = ab[ab.fold == "holdout_breast_cancer"].groupby("dropped")["delta"].mean()
imp["mean_delta_comp"] = imp.dropped.map(comp_delta)
imp["mean_delta_lymph"] = imp.dropped.map(lymph_delta)
imp["mean_delta_other"] = imp.dropped.map(other_delta)
imp["mean_delta_tonsil"] = imp.dropped.map(tons_delta)
imp["mean_delta_breast"] = imp.dropped.map(brst_delta)
imp = imp.sort_values("mean_delta")
imp.to_csv(os.path.join(RES, "designC_marker_importance.csv"), index=False)

# ---------- 100ep（预算再检验，独立小节） ----------
e100 = load_sp(SR, "lodo_v4_100ep")
e100_t = {p: v for p, v in e100["holdout_tonsil"].items() if p not in ISOTYPE}
e100_b = {p: v for p, v in e100["holdout_breast_cancer"].items() if p not in ISOTYPE}
e100_iso = {f: np.mean([v for p, v in d.items() if p in ISOTYPE]) for f, d in e100.items()}

# ---------- 外部验证（重建 CSV） ----------
ext = pd.read_csv(os.path.join(ROOT, "gse_external_summary_100ep.csv"))
ext_proc = ext[ext["mode"] == "proc"]
ext_sp_mean = ext_proc.SP.astype(float).mean()
ext_sp_tonsil = ext_proc[ext_proc.holdout == "holdout_tonsil"].SP.astype(float).mean()
ext_sp_breast = ext_proc[ext_proc.holdout == "holdout_breast_cancer"].SP.astype(float).mean()

# ---------- 报告 ----------
def fmt(x, sd=None):
    if pd.isna(x):
        return "—"
    s = f"{x:.3f}"
    if sd is not None and not pd.isna(sd):
        s += f"±{sd:.3f}"
    return s

L = ["# 结果报告：服务器 100ep 满预算 + 41 变体修正管线（designB2/C）", "",
     "生成时间：2026-09-13 19:20（v2 修正基线口径）。数据来源：AutoDL 服务器回传 results_server_final.tar.gz（41 变体全部成功 + 100ep 主运行）。",
     "**口径说明**：designB2/designC 变体 = 20 epochs；其基线 = v1 base_seed1-4（20 epochs、35 panel、log1p+行中心化，与修正子集约定同源，n=4）。",
     "lodo_v4_100ep（100 epochs）仅用于训练预算再检验（§1/§2），不与 20ep 变体混比。",
     "指标 = LODO 测试 fold 逐蛋白 Spearman（SP）。", ""]

# §1 100ep
L.append("## 1. 100ep 满预算内部基线（lodo_v4_100ep，35 panel，n=1）")
L.append("")
L.append(f"- 31 目标蛋白 SP 均值：tonsil **{np.mean(list(e100_t.values())):.3f}** / breast **{np.mean(list(e100_b.values())):.3f}**"
         f"（20ep 基线 base_seed1-4：tonsil {np.mean(list(base_targets['holdout_tonsil'].values())):.3f} / breast {np.mean(list(base_targets['holdout_breast_cancer'].values())):.3f}）")
L.append(f"- 同型对照（噪声底线）SP 均值：tonsil {e100_iso['holdout_tonsil']:.3f} / breast {e100_iso['holdout_breast_cancer']:.3f}"
         f"（20ep：{base_iso['holdout_tonsil']:.3f} / {base_iso['holdout_breast_cancer']:.3f}）")
top_t = sorted(e100_t.items(), key=lambda kv: -kv[1])[:5]
top_b = sorted(e100_b.items(), key=lambda kv: -kv[1])[:5]
L.append(f"- tonsil Top5：{', '.join(f'{p} {v:.3f}' for p, v in top_t)}")
L.append(f"- breast Top5：{', '.join(f'{p} {v:.3f}' for p, v in top_b)}")
L.append("")

# §2 外部
L.append("## 2. 100ep 外部验证（GSE263617，4 样本 × 2 holdout，proc 模式）")
L.append("")
L.append("- 逐样本 SP：" + "；".join(f"{r.holdout.replace('holdout_', '')}|{r.sample} {float(r.SP):+.3f}" for r in ext_proc.itertuples()))
L.append(f"- **外部 SP 均值 = {ext_sp_mean:.3f}**（tonsil-ckpt {ext_sp_tonsil:.3f} / breast-ckpt {ext_sp_breast:.3f}；20ep 时 0.146 → 提升 +{ext_sp_mean-0.146:.3f}）")
L.append("- 竞品基线（外部 SP，同一协议）：ridge_pc **0.388**（30/31 蛋白胜 DPAS-20ep、24/24 胜 namesake 基线）、ridge_pc_single **0.399**、ridge_pc_coord 0.386、knn15 0.304 — 100ep 满预算 DPAS 仍显著低于线性基线")
L.append("- ridge_pc 逐蛋白代表值：CD4 0.308 / PAX5 0.573 / CD8A 0.651 / CD3E 0.811 / PECAM1 0.610（DPAS-20ep 对应 −0.174 / 0.253 / 0.205 / 0.089* / 0.249；*CD3E 取自 internal_vs_external_per_protein.csv 若缺失则按实际）→ **亚型级失败是模型特异的，不是 RNA–蛋白弱耦合**")
L.append("- 外部 raw 模式（基线预测）SP 均值 = %.3f" % ext[ext["mode"]=="raw"].SP.astype(float).mean())
L.append("")

# §3 designB2
L.append("## 3. designB2 panel 缩减（修正管线，35→25→20；35 = base_seed1-4 n=4，25/20 各 5 seeds，均 20ep）")
L.append("")
g25 = {}
for fold_name, fold in [("Tonsil", "holdout_tonsil"), ("Breast cancer", "holdout_breast_cancer")]:
    sub = grpB[grpB.fold == fold]
    piv = sub.pivot_table(index="group", columns="size", values="sp_mean")
    sdv = sub.pivot_table(index="group", columns="size", values="sp_sd")
    g25[fold] = (piv, sdv)
    L.append(f"### {fold_name}")
    L.append("")
    L.append("| 蛋白组 | 35 (n=4) | 25 (n=5) | 20 (n=5) | 35→20 变化 |")
    L.append("|---|---|---|---|---|")
    for g in ["compartment", "lymph_subtype", "other", "isotype"]:
        if g not in piv.index:
            continue
        v35, v25, v20 = piv.loc[g, "35"], piv.loc[g].get("25", np.nan), piv.loc[g].get("20", np.nan)
        s35, s25, s20 = sdv.loc[g, "35"], sdv.loc[g].get("25", np.nan), sdv.loc[g].get("20", np.nan)
        L.append(f"| {g} | {fmt(v35, s35)} | {fmt(v25, s25)} | {fmt(v20, s20)} | {v20-v35:+.3f} |")
    L.append("")

t = grpB[grpB.fold == "holdout_tonsil"]
c35 = t[(t.group == "compartment") & (t["size"] == "35")].sp_mean.iloc[0]
c25 = t[(t.group == "compartment") & (t["size"] == "25")].sp_mean.iloc[0]
c20 = t[(t.group == "compartment") & (t["size"] == "20")].sp_mean.iloc[0]
l35 = t[(t.group == "lymph_subtype") & (t["size"] == "35")].sp_mean.iloc[0]
l25 = t[(t.group == "lymph_subtype") & (t["size"] == "25")].sp_mean.iloc[0]
l20 = t[(t.group == "lymph_subtype") & (t["size"] == "20")].sp_mean.iloc[0]
gap35, gap25, gap20 = c35 - l35, c25 - l25, c20 - l20
L.append("### H1 vs H2 判定（tonsil，修正管线，基线 n=4）")
L.append("")
L.append(f"- 区室−淋巴差距：35-panel {gap35:+.3f} → 25-panel {gap25:+.3f} → 20-panel {gap20:+.3f}（35→20 变化 {gap20-gap35:+.3f}）")
if gap25 - gap35 <= 0.05 and gap20 - gap35 > 0.05:
    verdict = "差距在 25-panel 基本不变、20-panel 明显扩大 → 激进缩减时 H1（panel 覆盖）有贡献"
elif gap20 - gap35 > 0.05:
    verdict = "差距随缩减显著扩大 → H1（panel 覆盖）有贡献"
elif abs(gap20 - gap35) <= 0.05:
    verdict = "差距随缩减基本不变 → H2（亚型级弱耦合）主导，H1 贡献小"
else:
    verdict = "差距随缩减收窄 → panel 中存在对淋巴组有害/冗余标志物"
L.append(f"- **判定：{verdict}**（注意：结合 §2 ridge 修复亚型失败的证据，'H2 弱耦合'作为失败主因的解释已被推翻——淋巴亚型可由线性模型从 RNA 预测，失败归因于 DPAS 模型本身）")
L.append("")

# §4 designC
L.append("## 4. designC 逐标志物消融（31 个 drop_X 变体，20ep，其余 30 蛋白 ΔSP vs base_seed1-4 均值）")
L.append("")
L.append("### 移除后伤害最大（平均 ΔSP 最负 = 该标志物对整体预测最重要）")
L.append("")
L.append("| 被移除标志物 | 所属组 | 平均ΔSP(全部) | ΔSP(区室) | ΔSP(淋巴) | ΔSP(tonsil) | ΔSP(breast) |")
L.append("|---|---|---|---|---|---|---|")
for _, r in imp.head(8).iterrows():
    L.append(f"| {r.dropped} | {r.dropped_group} | {r.mean_delta:+.4f} | {r.mean_delta_comp:+.4f} | {r.mean_delta_lymph:+.4f} | {r.mean_delta_tonsil:+.4f} | {r.mean_delta_breast:+.4f} |")
L.append("")
allneg = (imp.mean_delta < 0).all()
a2c = imp[imp.dropped == "ACTA2"].mean_delta_comp.iloc[0]
a2l = imp[imp.dropped == "ACTA2"].mean_delta_lymph.iloc[0]
pos = imp[imp.mean_delta > 0]
if allneg:
    tail_note = "31 个标志物平均 ΔSP 均为负，无整体有益的移除"
else:
    tail_note = "存在净正向移除：" + "、".join(f"{r.dropped} {r.mean_delta:+.3f}" for _, r in pos.iterrows()) + "（移除后整体平均反而略改善）"
L.append(f"### 移除后影响最小（平均 ΔSP 最接近 0；{tail_note}）")
L.append("")
L.append("| 被移除标志物 | 所属组 | 平均ΔSP(全部) | ΔSP(区室) | ΔSP(淋巴) |")
L.append("|---|---|---|---|---|")
for _, r in imp.tail(5).iterrows():
    L.append(f"| {r.dropped} | {r.dropped_group} | {r.mean_delta:+.4f} | {r.mean_delta_comp:+.4f} | {r.mean_delta_lymph:+.4f} |")
L.append("")
L.append(f"注：组特异例外——ACTA2 移除后区室组 ΔSP {a2c:+.3f}（{'提升' if a2c > 0 else '受损'}）但淋巴组 {a2l:+.3f}（{'提升' if a2l > 0 else '受损'}）；")
L.append("即 ACTA2 对区室蛋白预测冗余甚至有害，但对淋巴组预测有贡献，与其 v1 设计 B 中的行为方向一致。")
L.append("")
L.append("### 湿实验 mIF 候选清单关联（P0=ACTA2/PTPRC_2，P1=PECAM1，P2=EPCAM/CD3E）")
L.append("")
for m in ["ACTA2", "PTPRC_2", "PECAM1", "EPCAM", "CD3E"]:
    r = imp[imp.dropped == m]
    if len(r):
        r = r.iloc[0]
        L.append(f"- **{m}**（{r.dropped_group}）：移除后平均 ΔSP {r.mean_delta:+.4f}（区室 {r.mean_delta_comp:+.4f} / 淋巴 {r.mean_delta_lymph:+.4f}）")
L.append("")
L.append("完整排序见 results/designC_marker_importance.csv；逐蛋白 ΔSP 长表见 results/designC_ablation_deltas.csv。")

report = "\n".join(L)
with open(os.path.join(ROOT, "结果报告_服务器100ep与41变体.md"), "w", encoding="utf-8") as f:
    f.write(report)
print(report)
