"""同口径对比：内部 LODO 留出集 vs 外部 GSE263617，并给出「同名基因表达」基线。

基线定义：对蛋白 X，直接用其编码基因的 log1p(CP10k) 表达作为预测，
计算与真实蛋白（CLR）的 Spearman。衡量模型的增量价值。
"""
from __future__ import annotations
import os
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as _sp


def _d(X):
    return X.toarray() if _sp.issparse(X) else np.asarray(X)
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

PROT2GENE = {
    "ACTA2": "ACTA2", "BCL2": "BCL2", "CCR7": "CCR7", "CD14": "CD14", "CD163": "CD163",
    "CD19": "CD19", "CD27": "CD27", "CD274": "CD274", "CD3E": "CD3E", "CD4": "CD4",
    "CD40": "CD40", "CD68": "CD68", "CD8A": "CD8A", "CEACAM8": "CEACAM8", "CR2": "CR2",
    "CXCR5": "CXCR5", "EPCAM": "EPCAM", "FCGR3A": "FCGR3A", "HLA_DRA": "HLA-DRA",
    "ITGAM": "ITGAM", "ITGAX": "ITGAX", "KRT5": "KRT5", "MS4A1": "MS4A1", "PAX5": "PAX5",
    "PCNA": "PCNA", "PDCD1": "PDCD1", "PECAM1": "PECAM1", "PTPRC_1": "PTPRC",
    "PTPRC_2": "PTPRC", "SDC1": "SDC1", "VIM": "VIM",
}


def sp(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(spearmanr(a, b).correlation)


def pc(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


rows = []
detail = []

# ---------------- 内部 LODO 留出集 ----------------
for ho, name in [("holdout_tonsil", "tonsil"), ("holdout_breast_cancer", "breast_cancer")]:
    d = ROOT / "runs" / "lodo_v3" / "ckpt" / ho
    Y = np.load(d / "protein_true.npy")
    P = np.load(d / "protein_pred.npy")
    names = [x.strip() for x in open(d / "protein_names.txt") if x.strip()]
    rna = ad.read_h5ad(ROOT / "proc" / "preprocessed" / name / "RNA_proc.h5ad")
    genes = list(rna.var_names)
    Xg = np.asarray(_d(rna.X), dtype=np.float32)
    gi = {g: i for i, g in enumerate(genes)}
    # 真实蛋白是 CLR（log1p + 行中心化）
    mark = [i for i, p in enumerate(names) if not p.startswith(("mouse_", "rat_"))]
    iso = [i for i, p in enumerate(names) if p.startswith(("mouse_", "rat_"))]
    base = []
    for i in mark:
        g = PROT2GENE.get(names[i])
        if g in gi:
            base.append(sp(Xg[:, gi[g]], Y[:, i]))
    rows.append({"set": "内部 LODO", "sample": name, "n_spots": Y.shape[0],
                 "model_pcc": float(np.nanmean([pc(P[:, i], Y[:, i]) for i in mark])),
                 "model_spear": float(np.nanmean([sp(P[:, i], Y[:, i]) for i in mark])),
                 "iso_spear": float(np.nanmean([sp(P[:, i], Y[:, i]) for i in iso])),
                 "baseline_spear": float(np.nanmean(base))})
    for i in mark:
        g = PROT2GENE.get(names[i])
        detail.append({"set": "内部 LODO", "sample": name, "protein": names[i],
                       "model_pcc": pc(P[:, i], Y[:, i]), "model_spear": sp(P[:, i], Y[:, i]),
                       "baseline_spear": sp(Xg[:, gi[g]], Y[:, i]) if g in gi else np.nan})

# ---------------- 外部 GSE ----------------
for s in ["A1LN", "A1TNSL", "D1LN", "D1TNSL"]:
    Yp = ROOT / "proc" / "preprocessed" / f"gse_{s}" / "ADT_clr_clean.h5ad"
    adt = ad.read_h5ad(Yp)
    Y = np.asarray(adt.X, dtype=np.float32)
    names = list(adt.var_names)
    rna = ad.read_h5ad(ROOT / "proc" / "preprocessed" / f"gse_{s}" / "RNA_proc.h5ad")
    Xg = np.asarray(_d(rna.X), dtype=np.float32)
    gi = {g: i for i, g in enumerate(rna.var_names)}
    for ck in ["holdout_tonsil", "holdout_breast_cancer"]:
        P = None
    # 用两个 checkpoint 的平均（外部零样本，取两折模型平均更稳）
    Ps = []
    for ck in ["holdout_tonsil", "holdout_breast_cancer"]:
        f = ROOT / "results" / "gse_external_per_protein.csv"
    e = pd.read_csv(ROOT / "results" / "gse_external_per_protein.csv")
    sub = e[(e["input"] == "proc") & (e["sample"] == s)]
    mark = [i for i, p in enumerate(names) if not p.startswith(("mouse_", "rat_"))]
    iso = [i for i, p in enumerate(names) if p.startswith(("mouse_", "rat_"))]
    base = [sp(Xg[:, gi[PROT2GENE[names[i]]]], Y[:, i]) for i in mark
            if PROT2GENE.get(names[i]) in gi]
    mm = sub.groupby("protein")[["pcc", "spearman"]].mean()
    rows.append({"set": "外部 GSE", "sample": s, "n_spots": Y.shape[0],
                 "model_pcc": float(mm.loc[[names[i] for i in mark], "pcc"].mean()),
                 "model_spear": float(mm.loc[[names[i] for i in mark], "spearman"].mean()),
                 "iso_spear": float(mm.loc[[names[i] for i in iso], "spearman"].mean()),
                 "baseline_spear": float(np.nanmean(base))})
    for i in mark:
        g = PROT2GENE.get(names[i])
        detail.append({"set": "外部 GSE", "sample": s, "protein": names[i],
                       "model_pcc": float(mm.loc[names[i], "pcc"]),
                       "model_spear": float(mm.loc[names[i], "spearman"]),
                       "baseline_spear": sp(Xg[:, gi[g]], Y[:, i]) if g in gi else np.nan})

df = pd.DataFrame(rows)
df["模型-基线(spear)"] = df["model_spear"] - df["baseline_spear"]
pd.set_option("display.width", 240)
print(df.round(4).to_string(index=False))
print()
print("--- 分组均值 ---")
print(df.groupby("set")[["model_pcc", "model_spear", "iso_spear", "baseline_spear", "模型-基线(spear)"]]
      .mean().round(4).to_string())

det = pd.DataFrame(detail)
det.to_csv("results/internal_vs_external_per_protein.csv", index=False)
df.to_csv("results/internal_vs_external_summary.csv", index=False)
print("\n[saved] results/internal_vs_external_{summary,per_protein}.csv")

print("\n--- 外部 GSE 上，模型优于基线的蛋白数 ---")
ex = det[det["set"] == "外部 GSE"]
w = ex.dropna(subset=["baseline_spear"])
print(f"  可比较蛋白 {len(w)} 个，模型胜出 {int((w.model_spear > w.baseline_spear).sum())} 个")
top = w.assign(d=w.model_spear - w.baseline_spear).groupby("protein")["d"].mean().sort_values(ascending=False)
print("  增量最高的 8 个:", top.head(8).round(3).to_dict())
print("  增量最低的 8 个:", top.tail(8).round(3).to_dict())
