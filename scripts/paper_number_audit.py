# -*- coding: utf-8 -*-
"""论文草稿关键数字一致性校验：自动从存档表格核对 Results/Methods 引用的数字。
输出 PASS/FAIL 清单 -> results/paper_number_audit.md
"""
import os
import glob
import json
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L = []


def chk(name, got, want, tol=0.005):
    ok = got is not None and abs(got - want) <= tol
    L.append((name, want, got, "PASS" if ok else "**FAIL**"))


def chk_range(name, got, lo, hi):
    ok = got is not None and lo <= got <= hi
    L.append((name, f"[{lo},{hi}]", got, "PASS" if ok else "**FAIL**"))


# ---- 1. LODO 训练指标 ----
for ho, rmse_w, pcc_w in [("holdout_tonsil", 0.683, 0.868),
                          ("holdout_breast_cancer", 0.876, 0.783)]:
    d = pd.read_csv(f"runs/lodo_v3/ckpt/{ho}/per_protein_metrics.csv")
    n_spots = d.n_spots.iloc[0]
    # RMSE / spot PCC 存在 json 里，这里用 per_protein 的 pcc 均值交叉核对
    L.append((f"LODO {ho} n_spots", "—", int(n_spots), "info"))

import json as _j
for tag, f in [("tonsil", "runs/lodo_v3/metrics_holdout_tonsil.json"),
               ("breast", "runs/lodo_v3/metrics_holdout_breast_cancer.json")]:
    if os.path.exists(f):
        m = _j.load(open(f))
        L.append((f"LODO {tag} json keys", "—", str(list(m.keys())[:8])[:70], "info"))

# ---- 2. 外部验证 ----
ext = pd.read_csv("results/gse_external_per_protein.csv")
m = ext[(ext.input == "proc") & (~ext.isotype)]
for ck, sp_w, pcc_w in [("holdout_tonsil", 0.145, 0.114),
                        ("holdout_breast_cancer", 0.147, 0.150)]:
    s = m[m.ckpt == ck]
    chk(f"ext {ck} marker Spearman", s.spearman.mean(), sp_w)
    chk(f"ext {ck} marker PCC", s.pcc.mean(), pcc_w)
iso = ext[(ext.input == "proc") & (ext.isotype)]
chk("ext isotype SP (tonsil ckpt)",
    iso[iso.ckpt == "holdout_tonsil"].spearman.mean(), -0.026, tol=0.005)
chk("ext isotype SP (breast ckpt)",
    iso[iso.ckpt == "holdout_breast_cancer"].spearman.mean(), -0.002, tol=0.005)

# 增量/损失蛋白（模型 spear − 基因基线 spear，按蛋白均值后相减）
# 基线逐蛋白值存于 gse_external_per_protein? 若无，用 internal_vs_external_summary 核对聚合值
ivs = pd.read_csv("results/internal_vs_external_summary.csv")
row = ivs[ivs.set.str.contains("proc") | ivs["set"].astype(str).str.contains("GSE")]
L.append(("internal_vs_external rows", "—", ivs.to_dict("records").__len__(), "info"))
print(ivs.to_string())

# ---- 3. niche z 值（scorecard adj 口径，specificity_z 列）----
sc_vals = {("EPCAM", "A"): 2.42, ("EPCAM", "B"): 2.31,
           ("PTPRC_2", "A"): 1.42, ("PTPRC_2", "B"): 1.01,
           ("CD3E", "A"): 1.03, ("CD3E", "B"): 0.97,
           ("CD27", "A"): 1.03, ("CD27", "B"): 0.85,
           ("ITGAX", "A"): 0.86, ("ITGAX", "B"): 0.78,
           ("CD163", "A"): 0.75, ("CD163", "B"): 0.74,
           ("ACTA2", "A"): 1.09, ("ACTA2", "B"): 0.88,
           ("VIM", "A"): 0.93, ("VIM", "B"): 0.87,
           ("PECAM1", "A"): 0.42, ("PECAM1", "B"): 0.78}
sc = {s: pd.read_csv(f"results/niche2_section{s}_scorecard_adj.csv").set_index("protein")
      for s in ["A", "B"]}
for (prot, sec), want in sc_vals.items():
    got = float(sc[sec].loc[prot, "specificity_z"])
    chk(f"niche z (adj) {prot}/{sec}", got, want, tol=0.005)

# ---- 4. 疾病轴 ----
for sec, comp, prot, rho_w in [("B", "immune", "PTPRC_2", 0.359)]:
    d = pd.read_csv(f"results/disease_protein_assoc_{sec}.csv")
    r = d[(d.compartment == comp) & (d.protein == prot) & (~d.isotype)]
    chk(f"assoc B/immune/PTPRC_2", r.spearman.iloc[0], rho_w, tol=0.002)
for sec, comp, fl_w in [("A", "tropho", 0.100), ("B", "immune", 0.349)]:
    d = pd.read_csv(f"results/disease_protein_assoc_{sec}.csv")
    iso_d = d[d.isotype & (d.compartment == comp)]
    fl = np.percentile(np.abs(iso_d.spearman), 95)
    chk(f"floor {sec}/{comp}", fl, fl_w, tol=0.01)

# ---- 5. Visium spots / 签名 AUC ----
dA = pd.read_csv("results/disease_protein_assoc_A.csv")
nA = dA.groupby(["protein", "compartment"]).size().iloc[0]
L.append(("spots A (rows/comp)", "1917", int(len(dA) / 4), "info"))
dB = pd.read_csv("results/disease_protein_assoc_B.csv")
L.append(("spots B (rows/comp)", "1387", int(len(dB) / 4), "info"))
try:
    vs = pd.read_csv("results/visium_eope_score.csv")
    L.append(("visium_eope_score.csv", "—", "exists", "info"))
except Exception:
    pass
aucs = []
for f in glob.glob("results/visium_eope_score_*.csv"):
    d = pd.read_csv(f)
    if "auc" in d.columns:
        aucs += d.auc.dropna().tolist()
if aucs:
    chk_range("6 signatures AUC range min", min(aucs), 0.493, 0.495)
    chk_range("6 signatures AUC range max", max(aucs), 0.499, 0.501)

# ---- 6. 蜕膜 DE ----
ct = pd.read_csv("results/dec_de_celltypes.csv")
L.append(("decidua tested celltypes", "11", len(ct), "PASS" if len(ct) == 11 else "**FAIL**"))
d = pd.read_csv("results/dec_de_dNK1.csv").dropna(subset=["q"])
chk("decidua dNK1 min q (仅dNK1; 全局min=0.099见v2.2项)", d.q.min(), 0.276, tol=0.005)

# ---- 7. 区室蛋白增量（model − baseline，外部 4 样本均值）----
ive = pd.read_csv("results/internal_vs_external_per_protein.csv")
extp = ive[ive["set"] == "外部 GSE"].groupby("protein")[["model_spear", "baseline_spear"]].mean()
extp["delta"] = extp.model_spear - extp.baseline_spear
for prot, want in [("ACTA2", 0.336), ("PECAM1", 0.249), ("PTPRC_2", 0.186),
                   ("CXCR5", 0.177), ("CD14", 0.173), ("CD4", -0.268),
                   ("CR2", -0.265), ("PAX5", -0.117)]:
    chk(f"delta {prot}", extp.loc[prot, "delta"], want, tol=0.005)
# 同口径基线比较（24 个有 namesake 基因的蛋白）
both = extp.dropna()
chk("同口径 model SP (24 蛋白)", both.model_spear.mean(), 0.144, tol=0.002)
chk("同口径 baseline SP (24 蛋白)", both.baseline_spear.mean(), 0.155, tol=0.002)
from scipy.stats import wilcoxon  # noqa: E402

_wp = wilcoxon(both.model_spear, both.baseline_spear).pvalue
chk("namesake 配对 Wilcoxon p", float(_wp), 0.705, tol=0.01)
L.append(("namesake 配对方向 (model 胜/负)", "10/14",
          f"{int((both.delta > 0).sum())}/{int((both.delta < 0).sum())}",
          "PASS" if int((both.delta > 0).sum()) == 10 else "**FAIL**"))
# base seeds 的 31-marker 均值跨 seed SD（Q3 宽松 SD 0.03 的实测依据）
import glob as _glob  # noqa: E402

for _fold, _want in [("holdout_tonsil", 0.024), ("holdout_breast_cancer", 0.016)]:
    _v = []
    for _run in sorted(_glob.glob(f"runs/designB/base_seed*/ckpt/{_fold}/per_protein_metrics.csv")):
        _df = pd.read_csv(_run)
        _v.append(_df[~_df.protein.str.startswith(("mouse_", "rat_"))].spearman.mean())
    chk(f"base seeds 31-marker 均值 SD ({_fold.replace('holdout_', '')})",
        float(pd.Series(_v).std(ddof=1)), _want, tol=0.005)
L.append(("有 namesake 基因的蛋白数", "24", int(len(both)),
          "PASS" if len(both) == 24 else "**FAIL**"))

# ---- 7b. RNA 覆盖诊断（Results §7 / Discussion §2）----
rc = pd.read_csv("results/rna_coverage_diagnosis.csv")
for grp_name, det_w, expr_w in [("lymph", 0.615, 1.33), ("compartment", 0.578, 1.06)]:
    s = rc[rc.group == grp_name]
    chk(f"RNA coverage {grp_name} detect_rate", s.detect_rate.mean(), det_w, tol=0.01)
    chk(f"RNA coverage {grp_name} mean_expr", s.mean_expr.mean(), expr_w, tol=0.01)

# ---- 8. designB2（修正管线 panel 缩减，Results §7）----
b2_runs = glob.glob("results_server_final/runs/designB2/*/ckpt/*/per_protein_metrics.csv")
L.append(("designB2 variants (runs dir)", "10", len(set(p.split(os.sep)[-4] for p in b2_runs)),
          "PASS" if len(set(p.split(os.sep)[-4] for p in b2_runs)) == 10 else "**FAIL**"))
grp = pd.read_csv("results/designB2_group_summary.csv")


def grp_mean(fold, size, group):
    r = grp[(grp.fold == fold) & (grp["size"].astype(str) == str(size)) & (grp.group == group)]
    return float(r.sp_mean.iloc[0]) if len(r) else None


for tag, fold in [("tonsil", "holdout_tonsil"),
                  ("breast", "holdout_breast_cancer")]:
    w = {"tonsil": {"compartment": {"35": 0.193, "25": 0.1677, "20": 0.2066},
                    "lymph_subtype": {"35": 0.089, "25": 0.0569, "20": 0.0725}},
         "breast": {"compartment": {"35": 0.280, "25": 0.0999, "20": 0.1499},
                    "lymph_subtype": {"35": 0.142, "25": 0.1048, "20": 0.0533}}}[tag]
    for gname, sizes in w.items():
        for size, want in sizes.items():
            chk(f"designB2 {tag} {gname} @{size}", grp_mean(fold, size, gname), want)

# 组均值跨 seed SD（正文引用范围）
rows = []
FOLDS2 = [("holdout_tonsil"), ("holdout_breast_cancer")]
GROUPS2 = {
    "compartment": ["ACTA2", "PECAM1", "PTPRC_1", "PTPRC_2", "EPCAM", "KRT5", "VIM",
                    "CD14", "CD68", "CD163", "ITGAM", "ITGAX", "FCGR3A", "HLA_DRA"],
    "lymph_subtype": ["CD3E", "CD4", "CD8A", "CD19", "MS4A1", "PAX5", "CR2", "CXCR5",
                      "CCR7", "PDCD1", "CD27", "SDC1"],
}
for run_root, vmap in [("runs/designB", {"35": [f"base_seed{i}" for i in (1, 2, 3, 4)]}),
                       ("results_server_final/runs/designB2",
                        {"25": [f"size25_seed{i}" for i in (1, 2, 3, 4, 5)],
                         "20": [f"size20_seed{i}" for i in (1, 2, 3, 4, 5)]})]:
    for size, variants in vmap.items():
        for v in variants:
            for fold in FOLDS2:
                fp = os.path.join(run_root, v, "ckpt", fold, "per_protein_metrics.csv")
                if not os.path.exists(fp):
                    continue
                d = pd.read_csv(fp)[["protein", "spearman"]]
                for gname, prots in GROUPS2.items():
                    sub = d[d.protein.isin(prots)]
                    if len(sub):
                        rows.append(dict(size=size, seed=int(v.split("seed")[-1]),
                                         fold=fold, group=gname, gm=sub.spearman.mean()))
sd_long = pd.DataFrame(rows)
sd_stat = sd_long.groupby(["fold", "size", "group"])["gm"].std().reset_index()
t_sd = sd_stat[sd_stat.fold == "holdout_tonsil"]["gm"]
b_sd = sd_stat[sd_stat.fold == "holdout_breast_cancer"]["gm"]
chk_range("designB2 seed-SD tonsil min", float(t_sd.min()), 0.011, 0.013)
chk_range("designB2 seed-SD tonsil max", float(t_sd.max()), 0.115, 0.117)
chk_range("designB2 seed-SD breast min", float(b_sd.min()), 0.008, 0.010)
chk_range("designB2 seed-SD breast max", float(b_sd.max()), 0.180, 0.181)

# ---- 9. 100ep 满预算（Results §6）----
e100_t = pd.read_csv("results_server_final/runs/lodo_v4_100ep/ckpt/holdout_tonsil/per_protein_metrics.csv")
e100_b = pd.read_csv("results_server_final/runs/lodo_v4_100ep/ckpt/holdout_breast_cancer/per_protein_metrics.csv")
ISO = {"mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"}
chk("100ep internal tonsil 31-marker SP", e100_t[~e100_t.protein.isin(ISO)].spearman.mean(), 0.196)
chk("100ep internal breast 31-marker SP", e100_b[~e100_b.protein.isin(ISO)].spearman.mean(), 0.126)
chk("100ep internal tonsil isotype floor", e100_t[e100_t.protein.isin(ISO)].spearman.mean(), 0.178)
chk("100ep internal breast isotype floor", e100_b[e100_b.protein.isin(ISO)].spearman.mean(), 0.049)
base_t = pd.read_csv("runs/designB/base_seed1/ckpt/holdout_tonsil/per_protein_metrics.csv")
b_means = []
for i in (1, 2, 3, 4):
    d = pd.read_csv(f"runs/designB/base_seed{i}/ckpt/holdout_tonsil/per_protein_metrics.csv")
    b_means.append(d[~d.protein.isin(ISO)].spearman.mean())
chk("20ep base tonsil 31-marker SP (n=4 mean)", np.mean(b_means), 0.108)
ext100 = pd.read_csv("gse_external_summary_100ep.csv")
p100 = ext100[ext100["mode"] == "proc"]
chk("100ep external SP mean (proc)", p100.SP.astype(float).mean(), 0.241)
chk("100ep external SP tonsil-ckpt", p100[p100.holdout == "holdout_tonsil"].SP.astype(float).mean(), 0.196)
chk("100ep external SP breast-ckpt", p100[p100.holdout == "holdout_breast_cancer"].SP.astype(float).mean(), 0.286)
chk_range("100ep external per-section min", float(p100.SP.astype(float).min()), 0.169, 0.171)
chk_range("100ep external per-section max", float(p100.SP.astype(float).max()), 0.392, 0.394)

# ---- 10. 竞品基线（Results §6）----
cb = pd.read_csv("results/competitor_benchmark_summary.csv").set_index("model").sp_mean
for m, want in [("ridge_pc", 0.388), ("ridge_pc_single_avg", 0.399),
                ("ridge_pc_coord", 0.386), ("knn15", 0.304)]:
    chk(f"competitor {m}", float(cb[m]), want)
cbp = pd.read_csv("results/competitor_benchmark_per_protein.csv")
ridge_pp = cbp[cbp.model == "ridge_pc"].groupby("protein").spearman.mean()
ive = pd.read_csv("results/internal_vs_external_per_protein.csv")
dpas_pp = ive[ive["set"] == "外部 GSE"].groupby("protein")["model_spear"].mean()
j = pd.concat([ridge_pp.rename("ridge"), dpas_pp.rename("dpas")], axis=1).dropna()
wins = int((j.ridge > j.dpas).sum())
L.append(("ridge wins vs DPAS-20ep", "30/31", f"{wins}/{len(j)}",
          "PASS" if wins == 30 and len(j) == 31 else "**FAIL**"))
chk("ridge VIM (唯一例外)", float(j.loc["VIM", "ridge"]), 0.360)
chk("DPAS-20ep VIM (唯一例外)", float(j.loc["VIM", "dpas"]), 0.448)
jn = pd.concat([ridge_pp.rename("ridge"),
                ive[ive["set"] == "外部 GSE"].groupby("protein")["baseline_spear"].mean()],
               axis=1).dropna()
wins_n = int((jn.ridge > jn.baseline_spear).sum())
L.append(("ridge wins vs namesake", "24/24", f"{wins_n}/{len(jn)}",
          "PASS" if wins_n == 24 and len(jn) == 24 else "**FAIL**"))
for prot, want in [("CD4", 0.308), ("PAX5", 0.573), ("CD8A", 0.651), ("MS4A1", 0.331)]:
    chk(f"ridge_pc {prot}", float(ridge_pp[prot]), want)
for prot, want in [("CD4", -0.174), ("PAX5", 0.253), ("CD8A", 0.205), ("MS4A1", 0.150)]:
    chk(f"DPAS-20ep ext {prot}", float(dpas_pp[prot]), want)

# ---- 11. designC 逐标志物消融（Results §7）----
imp = pd.read_csv("results/designC_marker_importance.csv")
L.append(("designC dropped markers", "31", len(imp),
          "PASS" if len(imp) == 31 else "**FAIL**"))
for prot, want in [("ITGAX", -0.111), ("MS4A1", -0.076), ("PTPRC_2", -0.076),
                   ("PCNA", -0.068), ("PTPRC_1", -0.067), ("ACTA2", 0.015),
                   ("EPCAM", -0.005), ("CD3E", -0.029), ("PECAM1", -0.053)]:
    chk(f"designC meanΔ {prot}", float(imp[imp.dropped == prot].mean_delta.iloc[0]), want)
chk("designC ACTA2 ΔSP(区室)", float(imp[imp.dropped == "ACTA2"].mean_delta_comp.iloc[0]), 0.079)
chk("designC ACTA2 ΔSP(淋巴)", float(imp[imp.dropped == "ACTA2"].mean_delta_lymph.iloc[0]), -0.061)
chk("designC PECAM1 ΔSP(区室)", float(imp[imp.dropped == "PECAM1"].mean_delta_comp.iloc[0]), -0.161)

# ---- 12. 100ep 多 seed（Results "Training budget"、Discussion limitations）----
ms = pd.read_csv("results/lodo100ep_multiseed_summary.csv", index_col="seed")
chk("multiseed s0 spear mean", float(ms.loc[0, "spear_marker_mean"]), 0.241)
chk("multiseed s1 spear mean", float(ms.loc[1, "spear_marker_mean"]), 0.294)
chk("multiseed s2 spear mean", float(ms.loc[2, "spear_marker_mean"]), 0.286)
chk("multiseed 3-seed mean", float(ms["spear_marker_mean"].mean()), 0.274)
chk("multiseed 3-seed SD", float(ms["spear_marker_mean"].std(ddof=1)), 0.029)
chk("multiseed isotype floor mean", float(ms["spear_isotype_mean"].mean()), 0.103)
chk("multiseed isotype floor SD", float(ms["spear_isotype_mean"].std(ddof=1)), 0.004)
chk("ridge-gap in seed SDs", (0.388 - float(ms["spear_marker_mean"].mean()))
    / float(ms["spear_marker_mean"].std(ddof=1)), 3.9, tol=0.15)

# ---- 13. DGAT 同协议对照（Results "Second published graph architecture"）----
dgat_s = pd.read_csv("server_round2/dgat_lodo_gse_summary.csv")
fm = dgat_s.groupby("fold").spearman_mean.mean()
chk("DGAT fold mean (tonsil)", float(fm["holdout_tonsil"]), 0.275)
chk("DGAT fold mean (breast)", float(fm["holdout_breast_cancer"]), 0.408)
chk("DGAT overall mean", float(dgat_s.spearman_mean.mean()), 0.341)
rob = dgat_s[dgat_s.n_spots_eval >= 2000].groupby("fold").spearman_mean.mean()
chk("DGAT robust subset (tonsil)", float(rob["holdout_tonsil"]), 0.305)
chk("DGAT robust subset (breast)", float(rob["holdout_breast_cancer"]), 0.416)
chk("DGAT robust overall", float(rob.mean()), 0.361)
dgat_pp = pd.read_csv("server_round2/dgat_lodo_gse_per_protein.csv").groupby("protein").spearman.mean()
for prot, want in [("PAX5", 0.700), ("CXCR5", 0.679), ("PDCD1", 0.621),
                   ("MS4A1", 0.575), ("CD3E", 0.553), ("CD14", -0.028)]:
    chk(f"DGAT {prot}", float(dgat_pp[prot]), want)
cb2 = pd.read_csv("results/competitor_benchmark_per_protein.csv")
ridge2 = cb2[cb2.model == "ridge_pc"].groupby("protein").spearman.mean()
m2 = pd.merge(dgat_pp.rename("dgat"), ridge2.rename("ridge"),
              left_index=True, right_index=True)
n_win = int((m2.ridge > m2.dgat).sum())
L.append(("ridge wins vs DGAT", "21/31", f"{n_win}/31",
          "PASS" if n_win == 21 else "**FAIL**"))

# ---- 14. v2.2 更正项（2026-09-14 独立复核发现并更正）----
# (1) 内部 LODO 两折在正文中按 Spearman 口径引用（旧稿误用 PCC 0.060/0.233）
intp2 = pd.read_csv("results/internal_vs_external_per_protein.csv")
for _fold, _w in [("tonsil", 0.071), ("breast_cancer", 0.224)]:
    _d = intp2[(intp2["set"] == "内部 LODO") & (intp2["sample"] == _fold)]
    _m = ~_d.protein.str.startswith(("mouse_", "rat_"))
    chk(f"内部 LODO {_fold} 31-marker Spearman", float(_d[_m].model_spear.mean()), _w, tol=0.002)
# (2) 蜕膜全局最小 q（旧稿 0.276 仅为 dNK1；全局为 dNK2 的 0.099）
_allq2 = []
for _f in glob.glob("results/dec_de_*.csv"):
    _d = pd.read_csv(_f)
    if "q" in _d.columns:
        _d = _d.dropna(subset=["q"])
        if len(_d):
            _allq2.append(float(_d.q.min()))
chk("蜕膜全局 min q (11 类, dNK2)", float(np.min(_allq2)), 0.099, tol=0.002)
# (3) niche eta2 中位数（预处理修正后的预测；旧稿 0.101/0.081/0.105/0.083 为修正前旧版）
for _sec, _wi, _wr in [("A", 0.431, 0.395), ("B", 0.524, 0.588)]:
    _kw = pd.read_csv(f"results/niche2_section{_sec}_kw_adj.csv")
    _neg = _kw[_kw.is_negctrl.astype(bool)] if _kw.is_negctrl.dtype == bool else _kw[_kw.is_negctrl.astype(str) == "True"]
    _real = _kw[~(_kw.is_negctrl.astype(bool) if _kw.is_negctrl.dtype == bool else _kw.is_negctrl.astype(str) == "True")]
    chk(f"niche eta2 中位数 对照 {_sec}", float(_neg.eta2.median()), _wi)
    chk(f"niche eta2 中位数 真实 {_sec}", float(_real.eta2.median()), _wr)

# ---- 15. MLP 基线与 RNA 覆盖 Wilcoxon（2026-09-14 扩项）----
_mlp = pd.read_csv("results/mlp_baseline_summary.csv")
_m1 = _mlp[_mlp.model == "mlp_64"].iloc[0]
_m2_ = _mlp[_mlp.model == "mlp_64_32"].iloc[0]
chk("MLP(64) 三 seed 均值", float(_m1.sp_mean), 0.346, tol=0.002)
chk("MLP(64) 三 seed SD", float(_m1.sp_sd), 0.015, tol=0.002)
chk("MLP(64,32) 三 seed 均值", float(_m2_.sp_mean), 0.335, tol=0.002)
chk("MLP(64,32) 三 seed SD", float(_m2_.sp_sd), 0.016, tol=0.002)
_wil = pd.read_csv("results/rna_coverage_wilcoxon.csv")
_p1 = float(_wil[(_wil.scope == "pooled") & (_wil.variable == "detect_rate")].p.iloc[0])
_p2 = float(_wil[(_wil.scope == "pooled") & (_wil.variable == "mean_expr")].p.iloc[0])
chk("RNA覆盖 检出率 Wilcoxon p (pooled)", _p1, 0.66, tol=0.005)
chk("RNA覆盖 表达量 Wilcoxon p (pooled)", _p2, 0.51, tol=0.005)

# ---- 输出 ----
out = ["# 论文数字一致性审计（自动）", "",
       "| 数字 | 草稿引用值 | 实测值 | 判定 |", "|---|---|---|---|"]
fails = 0
for name, w, g, s in L:
    gs = f"{g:.4f}" if isinstance(g, float) else str(g)
    out.append(f"| {name} | {w} | {gs} | {s} |")
    if "FAIL" in s:
        fails += 1
open("results/paper_number_audit.md", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
print(f"\n==== FAIL count: {fails} ====")
