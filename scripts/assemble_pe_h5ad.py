# 组装 PE Visium h5ad：raw counts + obsm['spatial'] 像素坐标（DPAS-Graph 推理输入格式）
import numpy as np
import pandas as pd
import scipy.io as sio
import anndata as ad
from pathlib import Path

EXPORT = Path(r"D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/export")
OUT = Path(r"D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/h5ad")
OUT.mkdir(exist_ok=True)
PANEL = [x.strip() for x in open(
    r"D:/workbuddy/0911/pe-virtual-protein/runs/lodo_v3/ckpt/holdout_tonsil/common_gene.txt")] if False else None

for f in ["sectionA", "sectionB"]:
    X = sio.mmread(EXPORT / f"{f}_counts.mtx").T.tocsr()  # spots x genes
    genes = [g.strip() for g in open(EXPORT / f"{f}_genes.txt")]
    bcs = [b.strip() for b in open(EXPORT / f"{f}_barcodes.txt")]
    coords = pd.read_csv(EXPORT / f"{f}_coords.csv").set_index("barcode")
    deconv = pd.read_csv(EXPORT / f"{f}_deconv.csv").set_index("barcode")
    common_bc = [b for b in bcs if b in coords.index and b in deconv.index]
    idx = [bcs.index(b) for b in common_bc]
    X = X[idx, :]
    X = X[:, [i for i, g in enumerate(genes)]]  # keep all genes, panel subset at inference

    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=common_bc),
                       var=pd.DataFrame(index=genes))
    # 补齐 panel 缺失基因（零列，推理脚本要求全 panel 在场）
    panel = [x.strip() for x in open(
        r"D:/workbuddy/0911/pe-virtual-protein/runs/lodo_v3/ckpt/holdout_tonsil/common_gene.txt")]
    missing = [g for g in panel if g not in set(genes)]
    if missing:
        pad = ad.AnnData(X=np.zeros((len(common_bc), len(missing)), dtype=np.float32),
                         obs=pd.DataFrame(index=common_bc), var=pd.DataFrame(index=missing))
        adata = ad.concat([adata, pad], axis=1)
        print(f"  padded {len(missing)} missing panel genes with zeros")
    adata.obsm["spatial"] = coords.loc[common_bc, ["imagecol", "imagerow"]].values.astype(np.float32)
    for c in deconv.columns:
        adata.obs[c] = deconv.loc[common_bc, c].values
    adata.obs["section"] = f
    adata.uns["_dpas_input_domain"] = "raw"

    out = OUT / f"{f}_RNA_raw.h5ad"
    adata.write_h5ad(out)
    # 面板覆盖率
    panel = [x.strip() for x in open(
        r"D:/workbuddy/0911/pe-virtual-protein/runs/lodo_v3/ckpt/holdout_tonsil/common_gene.txt")]
    have = set(genes) & set(panel)
    print(f"{f}: {adata.shape[0]} spots x {adata.shape[1]} genes | panel coverage {len(have)}/4000 "
          f"| missing {sorted(set(panel) - set(genes))[:5]}")
    print(f"  saved: {out}")
