"""按 DPAS-Graph 训练时的真实约定重建 RNA 输入：log1p(CP10k)，**不做 z-score**。

背景（关键 bug 修复）：
  训练时 RNA 走的是 *_proc.h5ad -> _dpas_input_domain="proc" -> 不再做任何变换，
  即模型实际吃的是未 scale 的 log1p(CP10k)。
  但 PE 推理用的是 *_raw.h5ad（整数计数），走 "raw" 分支 ->
  normalize_total + log1p + **sc.pp.scale**，与训练分布不一致。
  GSE263617 提供的 h5ad 也是 z-scored（均值 0、含负值），同样不一致。
  本脚本统一产出 *_proc.h5ad，从根上消除该失配。

输出：
  proc/preprocessed/pe_section{A,B}/RNA_proc.h5ad
  proc/preprocessed/gse_{A1LN,A1TNSL,D1LN,D1TNSL}/RNA_proc.h5ad
  proc/preprocessed/gse_*/ADT_clr_clean.h5ad   （真实蛋白，已是 CLR/列仿射，仅用于评估）
"""
from __future__ import annotations
import gzip
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

CKPT = ROOT / "runs" / "lodo_v3" / "ckpt" / "holdout_breast_cancer"
COMMON_GENE = [x.strip() for x in (CKPT / "common_gene.txt").read_text().splitlines() if x.strip()]
COMMON_PROT = [x.strip() for x in (CKPT / "common_protein.txt").read_text().splitlines() if x.strip()]

# GSE 蛋白名 -> 训练 panel 名
PROT_MAP = {
    "HLA-DRA": "HLA_DRA",
    "PTPRC": "PTPRC_1",
    "PTPRC-1": "PTPRC_2",
}
ISOTYPE = [p for p in COMMON_PROT if p.startswith(("mouse_", "rat_"))]

# mtx 内抗体 feature id -> 训练 panel 名
_PROT_RENAME = {"HLA-DRA": "HLA_DRA"}

# 提交者提供的（z-scored）h5ad，仅用于校验 mtx 列的 spot 顺序
REF_H5 = {
    "A1LN": "GSM8195494_A1_LN.h5ad",
    "A1TNSL": "GSM8195495_A1_TNSL.h5ad",
    "D1LN": "GSM8195496_D1_LN.h5ad",
    "D1TNSL": "GSM8195497_D1_TNSL.h5ad",
}

GSE = ROOT / "data" / "benchmark" / "GSE263617"
GSE_SAMPLES = [
    ("A1LN", "GSM8195494_A1LN", "GSM8195498_A1_LN_Protein.h5ad"),
    ("A1TNSL", "GSM8195495_A1TNSL", "GSM8195499_A1_TNSL_Protein.h5ad"),
    ("D1LN", "GSM8195496_D1LN", "GSM8195500_D1_LN_Protein.h5ad"),
    ("D1TNSL", "GSM8195497_D1TNSL", "GSM8195501_D1_TNSL_Protein.h5ad"),
]


def qc_filter(rna: ad.AnnData, min_genes=500, max_mt=35.0) -> ad.AnnData:
    tmp = rna.copy()
    tmp.var["mt"] = tmp.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(tmp, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    mask = (tmp.obs["n_genes_by_counts"].values >= min_genes) & (tmp.obs["pct_counts_mt"].values <= max_mt)
    return rna[mask].copy()


def to_log_cp10k(rna: ad.AnnData) -> ad.AnnData:
    """严格复现 03_preprocess_v3 的 RNA 变换：normalize_total(1e4) + log1p，无 scale。"""
    rna = rna.copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    if sp.issparse(rna.X):
        rna.X = rna.X.toarray()
    rna.X = np.asarray(rna.X, dtype=np.float32)
    return rna


# ---------------------------------------------------------------- PE
def prep_pe():
    print("=" * 72)
    print("PE Visium：重建为未 scale 的 log1p(CP10k)")
    for sec in ["A", "B"]:
        src = ROOT / f"data/pe_placenta/h5ad/section{sec}_RNA_raw.h5ad"
        rna = ad.read_h5ad(src)
        missing = [g for g in COMMON_GENE if g not in set(rna.var_names)]
        print(f"  section{sec}: {rna.shape}, panel 缺失 {len(missing)} 基因")
        if missing:
            print(f"    缺: {missing[:10]}")
        rna = rna[:, [g for g in COMMON_GENE if g in set(rna.var_names)]].copy()
        # 缺失基因补零列（保持 4000 维顺序）
        if missing:
            add = pd.DataFrame(
                np.zeros((rna.shape[0], len(missing)), dtype=np.float32),
                index=rna.obs_names, columns=missing,
            )
            rna = ad.AnnData(
                X=np.asarray(rna.X, dtype=np.float32),
                obs=rna.obs.copy(), var=rna.var.copy(), obsm=dict(rna.obsm),
            )
            rna = ad.concat([rna, ad.AnnData(X=add.values, obs=rna.obs.copy(),
                                             var=pd.DataFrame(index=add.columns))], axis=1)
            rna = rna[:, COMMON_GENE].copy()
        rna = to_log_cp10k(rna)
        out = ROOT / "proc" / "preprocessed" / f"pe_section{sec}"
        out.mkdir(parents=True, exist_ok=True)
        rna.uns["name"] = f"pe_section{sec}"
        rna.write_h5ad(out / "RNA_proc.h5ad")
        X = np.asarray(rna.X)
        print(f"    -> {out/'RNA_proc.h5ad'}  shape={rna.shape} "
              f"mean={X.mean():.4f} std={X.std():.4f} min={X.min():.3f} max={X.max():.3f}")



def _TOPG(rna, n=300):
    X = np.asarray(np.log1p(rna.X.todense() if sp.issparse(rna.X) else rna.X), dtype=np.float32)
    return np.argsort(-X.mean(0))[:n]


def _morans_i(X, coords, k=6):
    from sklearn.neighbors import NearestNeighbors
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = np.asarray(X, dtype=np.float64)
    X = np.log1p(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    vals = []
    for j in range(X.shape[1]):
        z = Z[:, j]
        if np.isfinite(z).all():
            vals.append(np.corrcoef(z, z[idx].mean(1))[0, 1])
    return float(np.nanmean(vals))


def prep_gse():
    """从 SpaceRanger 原始 mtx 重建 GSE263617：
       - mtx 含 18085 基因 + 35 抗体（原始计数，含 4 个同型对照）
       - RNA -> log1p(CP10k)（与训练一致，不 scale） + 另存原始计数版供对照实验
       - ADT -> 用原始计数做 CLR(log1p + 行中心化)，与训练目标完全同源
    """
    print("=" * 72)
    print("GSE263617：从 SpaceRanger 原始 mtx 重建（基因 + 抗体原始计数）")
    for name, prefix, prot_f in GSE_SAMPLES:
        print(f"-- {name}", flush=True)
        mtx = GSE / f"{prefix}_matrix.mtx.gz"
        feat = GSE / f"{prefix}_features.tsv.gz"
        pos = GSE / f"{prefix}_tissue_positions.csv.gz"

        with gzip.open(feat, "rt") as fh:
            feats = pd.read_csv(fh, sep="\t", header=None)
        feats.columns = ["fid", "symbol", "ftype"]
        is_adt = (feats["ftype"] == "Antibody Capture").values
        print(f"   features {feats.shape}: gene={int((~is_adt).sum())} antibody={int(is_adt.sum())}", flush=True)

        with gzip.open(pos, "rt") as fh:
            pp = pd.read_csv(fh)
        it = pp[pp["in_tissue"] == 1].copy()
        # mtx 的列序 == 条码字典序（已用 Moran's I 验证：字典序 0.46 vs 文件序 0.00）
        it = it.sort_values("barcode", kind="mergesort")
        barcodes = it["barcode"].astype(str).values
        coords = it[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)

        with gzip.open(mtx, "rb") as fh:
            M = sio.mmread(fh).tocsr()          # (F, S)
        print(f"   mtx {M.shape} nnz={M.nnz}", flush=True)
        if M.shape[1] != len(barcodes):
            raise RuntimeError(f"{name}: mtx 列数 {M.shape[1]} != in-tissue spots {len(barcodes)}")

        # ---- 基因矩阵 (S, G)
        Gm = M[~is_adt].T.tocsr().astype(np.float32)
        gsym = feats.loc[~is_adt, "symbol"].astype(str).values
        rna = ad.AnnData(X=Gm,
                         obs=pd.DataFrame(index=pd.Index(barcodes)),
                         var=pd.DataFrame(index=pd.Index(_dedup(gsym))))
        rna.obsm["spatial"] = coords

        # ---- 抗体原始计数 (S, 35)，按训练 panel 重排
        Am = M[is_adt].T.tocsr()
        aname = [_PROT_RENAME.get(x, x) for x in feats.loc[is_adt, "fid"].astype(str).values]
        Am = pd.DataFrame(np.asarray(Am.todense(), dtype=np.float32),
                          index=pd.Index(barcodes), columns=aname)
        if not set(COMMON_PROT).issubset(set(aname)):
            miss = sorted(set(COMMON_PROT) - set(aname))
            raise RuntimeError(f"{name}: mtx 抗体缺失 {miss}")
        Am = Am[COMMON_PROT]

        # ---- 空间自相关自检：坐标与表达必须对齐（Moran's I 应显著 > 0）
        _mi = _morans_i(rna[:, _TOPG(rna)].X, coords)
        print(f"   Moran's I(自检, 高表达基因) = {_mi:.4f}  "
              f"{'OK' if _mi > 0.15 else '*** 坐标可能未对齐 ***'}", flush=True)

        rna = qc_filter(rna)
        Am = Am.loc[rna.obs_names]
        print(f"   after QC: {rna.shape[0]} spots", flush=True)

        missing = [g for g in COMMON_GENE if g not in set(rna.var_names)]
        print(f"   panel 缺失 {len(missing)} / {len(COMMON_GENE)}" + (f" 例:{missing[:8]}" if missing else ""), flush=True)
        cols = [g for g in COMMON_GENE if g in set(rna.var_names)]

        # RNA: 原始计数版
        raw_sub = rna[:, cols].copy()
        if missing:
            z = pd.DataFrame(np.zeros((raw_sub.shape[0], len(missing)), dtype=np.float32),
                             index=raw_sub.obs_names, columns=missing)
            raw_sub = ad.concat([raw_sub, ad.AnnData(X=z.values, obs=raw_sub.obs.copy(),
                                                     var=pd.DataFrame(index=z.columns))], axis=1)
            raw_sub = raw_sub[:, COMMON_GENE].copy()
        raw_sub.X = raw_sub.X.astype(np.float32)
        raw_sub.obsm["spatial"] = rna.obsm["spatial"]
        raw_sub.uns["name"] = name

        # RNA: proc 版（log1p CP10k，不 scale）
        sub = to_log_cp10k(raw_sub)
        sub.obsm["spatial"] = rna.obsm["spatial"]
        sub.uns["name"] = name

        out = ROOT / "proc" / "preprocessed" / f"gse_{name}"
        out.mkdir(parents=True, exist_ok=True)
        sub.write_h5ad(out / "RNA_proc.h5ad")
        raw_sub.write_h5ad(out / "RNA_raw.h5ad")
        Xn = np.asarray(sub.X)
        print(f"   -> RNA_proc.h5ad {sub.shape} mean={Xn.mean():.4f} std={Xn.std():.4f} "
              f"min={Xn.min():.3f} max={Xn.max():.3f}", flush=True)
        print(f"   -> RNA_raw.h5ad  {raw_sub.shape} (原始计数对照)", flush=True)

        # ADT: 原始计数 -> CLR(log1p + 行中心化)，与训练目标同源
        Z = np.asarray(Am.values, dtype=np.float64)
        Z = np.log1p(Z)
        Z = Z - Z.mean(axis=1, keepdims=True)
        adt = ad.AnnData(X=Z.astype(np.float32),
                         obs=pd.DataFrame(index=pd.Index(Am.index)),
                         var=pd.DataFrame(index=pd.Index(COMMON_PROT)))
        adt.obsm["spatial"] = rna.obsm["spatial"].copy()
        adt.uns["name"] = name
        adt.write_h5ad(out / "ADT_clr_clean.h5ad")
        print(f"   -> ADT_clr_clean.h5ad {adt.shape}  "
              f"mean={adt.X.mean():.4f} std={adt.X.std():.4f} "
              f"rowsum_std={adt.X.sum(1).std():.6f}", flush=True)


def _dedup(arr):
    seen, out = {}, []
    for x in arr:
        seen[x] = seen.get(x, 0) + 1
        out.append(x if seen[x] == 1 else f"{x}.{seen[x]}")
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "pe"):
        prep_pe()
    if which in ("all", "gse"):
        prep_gse()
