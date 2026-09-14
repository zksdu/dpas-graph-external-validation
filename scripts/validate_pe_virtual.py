# PE 虚拟蛋白严格统计验证：
# 1) 置换检验（1000 次标签打乱的零分布）
# 2) RNA 标志物签名验证（实测 RNA 签名分 × 虚拟蛋白，独立于解卷积）
# 3) Moran's I 空间自相关
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

RUNS = Path(r"D:/workbuddy/0911/pe-virtual-protein/runs")
EXPORT = Path(r"D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/export")
OUT = Path(r"D:/workbuddy/0911/pe-virtual-protein/results")
CKPT = Path(r"D:/workbuddy/0911/pe-virtual-protein/runs/lodo_v3/ckpt/holdout_breast_cancer")
proteins = [x.strip() for x in (CKPT / "common_protein.txt").read_text().splitlines() if x.strip()]

# 经典标志物基因签名（ literature-canonical, 用实测 RNA 计算）
SIGNATURES = {
    "SCT": ["CGB3", "CGB5", "CGB8", "CSH1", "CSH2", "CSHL1", "ERVW-1", "ERVFRD-1", "PSG4", "PSG6", "PSG11", "GH2"],
    "VCT": ["TP63", "KRT5", "KRT14", "ITGA6", "EGFR"],
    "EVT": ["HLAG", "ITGA5", "ITGAV", "MMP2", "MMP9", "FN1", "TIMP3"],
    "HBC": ["CD68", "LYZ", "AIF1", "C1QA", "C1QB", "C1QC", "FCGR3A", "MSR1"],
    "Endo": ["PECAM1", "VWF", "CLDN5", "CDH5", "KDR", "SOX17"],
    "Fibro": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
}
PAIRS = [  # (虚拟蛋白, RNA 签名)
    ("EPCAM", "SCT"), ("EPCAM", "VCT"),
    ("CD68", "HBC"), ("CD163", "HBC"), ("HLA-DRA", "HBC"),
    ("PECAM1", "Endo"), ("ACTA2", "EVT"), ("MET", "EVT"),
]

def signature_score(adata_counts_genes, gene_names, sig_genes):
    """简单的均值签名分：签名基因在 panel 内的表达均值（raw counts 归一化后）"""
    gset = [g for g in sig_genes if g in gene_names]
    if len(gset) < 3:
        return None, len(gset)
    idx = [list(gene_names).index(g) for g in gset]
    X = adata_counts_genes[:, idx]
    X = np.log1p(X / np.maximum(X.sum(axis=1, keepdims=True), 1) * 1e4)
    return X.mean(axis=1), len(gset)

def morans_i(vals, coords, k=6):
    n = len(vals)
    d = np.linalg.norm(coords[:, None] - coords[None], axis=2)
    idx = np.argsort(d, axis=1)[:, 1:k+1]
    z = vals - vals.mean()
    num = den = 0.0
    W = 0
    for i in range(n):
        for j in idx[i]:
            num += z[i] * z[j]
            W += 1
    den = (z * z).sum()
    return (n / W) * (num / den) if den > 0 else np.nan

rng = np.random.default_rng(0)
report = []
for f in ["sectionA", "sectionB"]:
    pred = np.load(RUNS / f"pe_{f}_infer.pred.npy")
    bcs = pd.read_csv(EXPORT / f"{f}_barcodes.txt", header=None)[0].tolist()
    pmat = pd.DataFrame(pred, index=bcs, columns=proteins)
    coords = pd.read_csv(EXPORT / f"{f}_coords.csv").set_index("barcode").loc[bcs][["imagecol", "imagerow"]].values

    # 实测 RNA 签名
    import anndata as ad
    adata = ad.read_h5ad(rf"D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/h5ad/{f}_RNA_raw.h5ad")
    X = adata[:, [g for g in adata.var_names if not g.startswith("ENSG")]].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    gnames = list(adata.var_names)

    print(f"\n===== {f} =====")
    report.append(f"===== {f} =====")
    for prot, sig in PAIRS:
        if prot not in pmat.columns:
            continue
        sc_val, n_g = signature_score(X, gnames, SIGNATURES[sig])
        if sc_val is None:
            continue
        r, p = spearmanr(pmat[prot].values, sc_val)
        # 置换检验零分布
        null = [spearmanr(rng.permutation(pmat[prot].values), sc_val)[0] for _ in range(1000)]
        emp_p = (np.abs(null) >= abs(r)).mean()
        mi = morans_i(pmat[prot].values.astype(np.float64), coords.astype(np.float64))
        line = f"{f}\t{prot}\tvs RNA-sig:{sig}({n_g} genes)\trho={r:+.3f}\tperm_p={emp_p:.3f}\tMoransI={mi:.3f}"
        report.append(line)
        print(" ", line.split("\t", 1)[1])

with open(OUT / "strict_validation.tsv", "w", encoding="utf-8") as fh:
    fh.write("\n".join(report) + "\n")
print("\nsaved:", OUT / "strict_validation.tsv")
