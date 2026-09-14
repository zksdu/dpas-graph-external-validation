"""用独立指标判定哪一版 PE 预测更可信：
对每个蛋白，计算「预测蛋白值」与「该蛋白对应基因的表达」的 Spearman 相关。
正确的预处理应显著更高（这是不依赖任何下游假设的外部一致性检验）。
"""
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# 蛋白 -> 编码基因（31 个标志物）
PROT2GENE = {
    "ACTA2": "ACTA2", "BCL2": "BCL2", "CCR7": "CCR7", "CD14": "CD14", "CD163": "CD163",
    "CD19": "CD19", "CD27": "CD27", "CD274": "CD274", "CD3E": "CD3E", "CD4": "CD4",
    "CD40": "CD40", "CD68": "CD68", "CD8A": "CD8A", "CEACAM8": "CEACAM8", "CR2": "CR2",
    "CXCR5": "CXCR5", "EPCAM": "EPCAM", "FCGR3A": "FCGR3A", "HLA_DRA": "HLA-DRA",
    "ITGAM": "ITGAM", "ITGAX": "ITGAX", "KRT5": "KRT5", "MS4A1": "MS4A1", "PAX5": "PAX5",
    "PCNA": "PCNA", "PDCD1": "PDCD1", "PECAM1": "PECAM1", "PTPRC_1": "PTPRC",
    "PTPRC_2": "PTPRC", "SDC1": "SDC1", "VIM": "VIM",
}

OLD = {"A_bc": "runs/pe_sectionA_infer", "A_tonsil": "runs/pe_sectionA_infer_ckpt2",
       "B_bc": "runs/pe_sectionB_infer"}
NEW = {"A_bc": "runs/pe_A_breast_cancer_v2", "A_tonsil": "runs/pe_A_tonsil_v2",
       "B_bc": "runs/pe_B_breast_cancer_v2", "B_tonsil": "runs/pe_B_tonsil_v2"}

rows = []
for key in NEW:
    if key not in OLD:
        continue
    sec = key.split("_")[0]
    a = ad.read_h5ad(f"proc/preprocessed/pe_section{sec}/RNA_proc.h5ad")
    genes = list(a.var_names)
    X = np.asarray(a.X, dtype=np.float32)
    gi = {g: i for i, g in enumerate(genes)}

    for tag, pref in (("old(scale)", OLD[key]), ("new(proc)", NEW[key])):
        P = np.load(f"{pref}.pred.npy")
        names = [x.strip() for x in open(f"{pref}.protein_names.txt") if x.strip()]
        n = min(P.shape[0], X.shape[0])
        rhos = []
        for j, pname in enumerate(names):
            g = PROT2GENE.get(pname)
            if g is None or g not in gi:
                continue
            v = X[:n, gi[g]]
            if v.std() < 1e-8:
                continue
            r = spearmanr(P[:n, j], v).correlation
            if np.isfinite(r):
                rhos.append(r)
        rows.append({"key": key, "version": tag, "n_proteins": len(rhos),
                     "mean_rho": float(np.mean(rhos)), "median_rho": float(np.median(rhos)),
                     "frac_pos": float(np.mean(np.array(rhos) > 0))})

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
print()
piv = df.pivot(index="key", columns="version", values="mean_rho")
piv["delta_new_minus_old"] = piv["new(proc)"] - piv["old(scale)"]
print(piv.round(4).to_string())
df.to_csv("results/pe_input_fix_check.csv", index=False)
print("\n[saved] results/pe_input_fix_check.csv")
