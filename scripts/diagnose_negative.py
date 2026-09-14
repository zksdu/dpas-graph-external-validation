# 诊断虚拟蛋白与 RNA 签名负相关的成因：
# 1) 文库大小效应：虚拟蛋白 vs spot 总 counts
# 2) 组成效应：签名基因占比 vs 虚拟蛋白
# 3) Top-spot 重叠（Jaccard）：高虚拟蛋白 spot 是否也是高 RNA 签名 spot
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import anndata as ad

RUNS = Path(r"D:/workbuddy/0911/pe-virtual-protein/runs")
EXPORT = Path(r"D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/export")
OUT = Path(r"D:/workbuddy/0911/pe-virtual-protein/results")
CKPT = Path(r"D:/workbuddy/0911/pe-virtual-protein/runs/lodo_v3/ckpt/holdout_breast_cancer")
proteins = [x.strip() for x in (CKPT / "common_protein.txt").read_text().splitlines() if x.strip()]

SIGNATURES = {
    "SCT": ["CGB3", "CGB5", "CGB8", "CSH1", "CSH2", "CSHL1", "ERVW-1", "PSG4", "PSG6", "PSG11", "GH2"],
    "HBC": ["CD68", "LYZ", "AIF1", "C1QA", "C1QB", "C1QC", "FCGR3A", "MSR1"],
    "Endo": ["PECAM1", "VWF", "CLDN5", "CDH5", "KDR"],
}
CHECKS = [("CD68", "HBC"), ("CD163", "HBC"), ("EPCAM", "SCT"), ("PECAM1", "Endo")]

for f in ["sectionA", "sectionB"]:
    pred = np.load(RUNS / f"pe_{f}_infer.pred.npy")
    bcs = pd.read_csv(EXPORT / f"{f}_barcodes.txt", header=None)[0].tolist()
    pmat = pd.DataFrame(pred, index=bcs, columns=proteins)
    adata = ad.read_h5ad(rf"D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/h5ad/{f}_RNA_raw.h5ad")
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    gnames = list(adata.var_names)
    lib = X.sum(axis=1)
    print(f"\n===== {f} =====")
    for prot, sig in CHECKS:
        if prot not in pmat.columns:
            continue
        gs = [g for g in SIGNATURES[sig] if g in gnames]
        gi = [gnames.index(g) for g in gs]
        raw_sig = X[:, gi].sum(axis=1)
        # 1) 文库大小
        r_lib, p_lib = spearmanr(pmat[prot].values, lib)
        # 2) 组成（占比）与绝对（counts）两种口径
        frac = raw_sig / np.maximum(lib, 1)
        r_frac, _ = spearmanr(pmat[prot].values, frac)
        r_abs, _ = spearmanr(pmat[prot].values, raw_sig)
        # 3) top-quartile Jaccard
        q = int(0.25 * len(bcs))
        top_v = set(np.argsort(-pmat[prot].values)[:q])
        top_r = set(np.argsort(-frac)[:q])
        jac = len(top_v & top_r) / len(top_v | top_r)
        exp_jac = q / (len(bcs) - q) if len(bcs) > 2 * q else 1.0  # 随机期望上界
        print(f"  {prot:8s}/{sig:5s}: vs libsize rho={r_lib:+.3f} | vs abs-counts rho={r_abs:+.3f} | vs frac rho={r_frac:+.3f} | Jaccard(top25%)={jac:.3f} (random~0.14)")
