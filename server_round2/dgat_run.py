"""DGAT LODO 同协议对照 v5（预测段修复版）。

run4 已完成两折训练并保存模型（日志行 123/261）：
  - 12916_gene_31_protein = holdout_tonsil 折（train=breast_cancer）
  - 17434_gene_31_protein = holdout_breast_cancer 折（train=tonsil）
train() 内部落盘的 dgat_work/common_gene_{N}.txt 即训练时基因列表（与 encoder 输入一致）。

v5 修复：预测段 fill_genes 前清空 st.layers（GSE h5ad 自带 layers 触发
"layers[None] and X must be identical"），其余协议不变。
"""
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

ROOT = "/root/pe-virtual-protein"
WORK = os.path.join(ROOT, "dgat_work")
DGAT = os.path.join(ROOT, "DGAT_baseline")
os.makedirs(os.path.join(WORK, "resources"), exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, DGAT)

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse as sp
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader

# torch 2.8 默认 weights_only=True，会拒绝加载 pyg 图缓存（HeteroData）；
# 全部 checkpoint/图缓存均为本次训练自产，显式放行
_orig_torch_load = torch.load
def _torch_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _torch_load

from utils.Preprocessing import preprocess_ST
from utils.Graph_utils import MultiGraphDataset_for_no_protein
from Model.dgat import GATEncoder, Decoder_Protein
from Model import Train_and_Predict as TAP


def fill_genes_safe(st, cg):
    """fill_genes 官方实现与 anndata 0.13 不兼容（layers[None] 校验）；
    语义等价的本地实现：补零基因列 + 重排 var 至 cg 顺序，保留 obsm/uns。"""
    existing = set(st.var_names)
    missing = [g for g in cg if g not in existing]
    X = st.X
    if missing:
        if sp.issparse(X):
            Z = sp.coo_matrix(
                (np.zeros(0, np.float32), (np.zeros(0, np.int64), np.zeros(0, np.int64))),
                shape=(st.n_obs, len(missing)),
            ).tocsr()
            X = sp.hstack([X, Z]).tocsr()
        else:
            X = np.hstack([X, np.zeros((st.n_obs, len(missing)), dtype=np.float32)])
        var = pd.concat([st.var, pd.DataFrame(index=missing)], axis=0)
    else:
        var = st.var.copy()
    st2 = ad.AnnData(X=X, obs=st.obs.copy(), var=var)
    st2.obsm["spatial"] = np.asarray(st.obsm["spatial"])
    st2.uns["name"] = st.uns.get("name", "sample")
    print(f"filled {len(missing)} genes -> {st2.n_vars} vars", flush=True)
    return st2[:, cg].copy()

FOLDS = {"holdout_tonsil": "breast_cancer", "holdout_breast_cancer": "tonsil"}
GSE = ["A1LN", "A1TNSL", "D1LN", "D1TNSL"]

SEED = 2025
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, flush=True)

MODEL_DIR = os.path.join(WORK, "DGAT_models")

# run4 训练产物（日志行 123/261 证据）：fold -> 基因列表文件维度
FOLD_NGENES = {"holdout_tonsil": 12916, "holdout_breast_cancer": 17434}

fold_meta = {}
for fold in FOLDS:
    n = FOLD_NGENES[fold]
    mdir = os.path.join(MODEL_DIR, f"{n}_gene_31_protein")
    need = ["encoder_mRNA.pth", "decoder_mRNA.pth", "encoder_protein.pth", "decoder_protein.pth"]
    assert all(os.path.exists(os.path.join(mdir, f)) for f in need), f"missing model in {mdir}"
    cg = [l.strip() for l in open(f"{WORK}/common_gene_{n}.txt") if l.strip()]
    cp = [l.strip() for l in open(f"{WORK}/common_protein_31.txt") if l.strip()]
    assert len(cg) == n and len(cp) == 31
    fold_meta[fold] = (cg, cp)
    print(f"restored {fold}: genes={len(cg)} proteins={len(cp)} from {mdir}", flush=True)

hidden_dim = TAP.hidden_dim
dropout_rate = TAP.dropout_rate

rows, per_prot = [], []
for fold, (cg, cp) in fold_meta.items():
    mdir = os.path.join(MODEL_DIR, f"{len(cg)}_gene_31_protein")
    print(f"\n########## PREDICT fold={fold} ##########", flush=True)
    enc = GATEncoder(in_channels=len(cg), hidden_dim=hidden_dim, dropout=dropout_rate)
    dec = Decoder_Protein(hidden_dim, cp)
    enc.load_state_dict(torch.load(os.path.join(mdir, "encoder_mRNA.pth"), map_location="cpu"))
    dec.load_state_dict(torch.load(os.path.join(mdir, "decoder_protein.pth"), map_location="cpu"))
    enc.eval().to(DEVICE)
    dec.eval().to(DEVICE)

    for s in GSE:
        d = f"{ROOT}/proc/preprocessed/gse_{s}"
        st = sc.read_h5ad(f"{d}/RNA_raw.h5ad")
        n0 = st.n_obs
        if "spatial" in st.uns:
            del st.uns["spatial"]
        st.uns["name"] = f"gse_{s}_{fold}"
        preprocess_ST(st)                     # QC(min_genes=700)+CP10k+log1p+scale
        st = fill_genes_safe(st, cg)          # 补缺失基因并对齐基因顺序
        ds = MultiGraphDataset_for_no_protein([st], device=DEVICE,
                                              save_dir=f"{WORK}/pyg_pred_{s}_{fold}")
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        with torch.no_grad():
            for data in loader:
                data = data.to(DEVICE)
                z = enc(data["mRNA"].x.to(DEVICE),
                        data[("mRNA", "mRNA_knn", "mRNA")].edge_index.to(DEVICE),
                        data[("mRNA", "mRNA_knn", "mRNA")].edge_attr.to(DEVICE))
                P = dec(z).cpu().numpy().astype(np.float32)
        assert P.shape == (st.n_obs, len(cp)), P.shape

        truth = ad.read_h5ad(f"{d}/ADT_clr_clean.h5ad")
        common = [o for o in st.obs_names if o in set(truth.obs_names)]
        idx = [list(st.obs_names).index(o) for o in common]
        P = P[idx]
        Y = np.asarray(truth[common, cp].X, dtype=np.float32)
        proc = sc.read_h5ad(f"{d}/RNA_proc.h5ad")[common].copy()
        Xg = np.asarray(proc.X, dtype=np.float32)
        gi = {g: i for i, g in enumerate(proc.var_names)}

        sp_l, pc_l, ns_l = [], [], []
        for j, prot in enumerate(cp):
            spv = spearmanr(P[:, j], Y[:, j]).correlation
            pc = pearsonr(P[:, j], Y[:, j])[0]
            sp_l.append(spv); pc_l.append(pc)
            g = prot.split("_")[0]
            ns = spearmanr(Xg[:, gi[g]], Y[:, j]).correlation if g in gi else np.nan
            ns_l.append(ns)
            per_prot.append({"fold": fold, "sample": s, "protein": prot,
                             "spearman": float(spv), "pearson": float(pc),
                             "namesake_spearman": float(ns) if np.isfinite(ns) else np.nan,
                             "n_spots": len(common)})
        spot_pc = [pearsonr(P[k], Y[k])[0] for k in range(P.shape[0])]
        ns_fin = [x for x in ns_l if np.isfinite(x)]
        rows.append({
            "fold": fold, "sample": s, "n_spots_raw": n0, "n_spots_eval": len(common),
            "n_genes_model": len(cg),
            "spearman_mean": float(np.nanmean(sp_l)), "spearman_median": float(np.nanmedian(sp_l)),
            "pearson_mean": float(np.nanmean(pc_l)),
            "spot_pearson_mean": float(np.nanmean(spot_pc)),
            "namesake_spearman_mean": float(np.nanmean(ns_fin)),
        })
        print(f"[{fold:24s}|{s:8s}] n={len(common):5d}/{n0:5d} "
              f"SP={np.nanmean(sp_l):+.3f} PC={np.nanmean(pc_l):+.3f} "
              f"spotPC={np.nanmean(spot_pc):+.3f} base={np.nanmean(ns_fin):+.3f}", flush=True)

os.makedirs(f"{ROOT}/results", exist_ok=True)
df = pd.DataFrame(rows)
pp = pd.DataFrame(per_prot)
df.to_csv(f"{ROOT}/results/dgat_lodo_gse_summary.csv", index=False)
pp.to_csv(f"{ROOT}/results/dgat_lodo_gse_per_protein.csv", index=False)
print("\n=== summary (per fold, mean over 4 samples) ===", flush=True)
print(df.groupby("fold")[["spearman_mean", "pearson_mean", "spot_pearson_mean",
                          "namesake_spearman_mean"]].mean().round(4).to_string(), flush=True)
print("DGAT_RUN_DONE", flush=True)
