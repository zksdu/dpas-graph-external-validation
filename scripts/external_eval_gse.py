"""GSE263617 外部验证：DPAS-Graph 零样本迁移到 4 个完全独立的 CytAssist 样本，
与**真实测得蛋白**对比。同时用该金标准判定 RNA 输入预处理约定孰优
（proc = 未 scale 的 log1p(CP10k)，与训练一致；raw = 触发 sc.pp.scale 的旧路径）。

指标
  - 每个蛋白：预测 vs 真实 的 Pearson / Spearman（跨 spot）
  - 每个 spot：跨 31 个标志物蛋白的 Pearson
  - 基线：用同名基因表达直接当预测（衡量模型的增量价值）
  - 阴性对照：4 个同型对照抗体（真实值来自 mtx 原始计数）
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "DPAS-Graph" / "src"))
sys.path.insert(0, str(ROOT / "DPAS-Graph" / "scripts" / "eval"))

import benchmark_inference as bi  # noqa: E402
from dpas.models.dpas_graph import DualAdaptiveEncoder, Decoder_Protein_MLP  # noqa: E402

SAMPLES = ["A1LN", "A1TNSL", "D1LN", "D1TNSL"]
CKPTS = {"holdout_tonsil": "runs/lodo_v3/ckpt/holdout_tonsil",
         "holdout_breast_cancer": "runs/lodo_v3/ckpt/holdout_breast_cancer"}
VERSIONS = {"proc": "RNA_proc.h5ad", "raw": "RNA_raw.h5ad"}

PROT2GENE = {
    "ACTA2": "ACTA2", "BCL2": "BCL2", "CCR7": "CCR7", "CD14": "CD14", "CD163": "CD163",
    "CD19": "CD19", "CD27": "CD27", "CD274": "CD274", "CD3E": "CD3E", "CD4": "CD4",
    "CD40": "CD40", "CD68": "CD68", "CD8A": "CD8A", "CEACAM8": "CEACAM8", "CR2": "CR2",
    "CXCR5": "CXCR5", "EPCAM": "EPCAM", "FCGR3A": "FCGR3A", "HLA_DRA": "HLA-DRA",
    "ITGAM": "ITGAM", "ITGAX": "ITGAX", "KRT5": "KRT5", "MS4A1": "MS4A1", "PAX5": "PAX5",
    "PCNA": "PCNA", "PDCD1": "PDCD1", "PECAM1": "PECAM1", "PTPRC_1": "PTPRC",
    "PTPRC_2": "PTPRC", "SDC1": "SDC1", "VIM": "VIM",
}


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(spearmanr(a, b).correlation)


def load_model(ckpt_dir, n_genes, n_prot, device):
    enc = DualAdaptiveEncoder(in_channels=n_genes, hidden_dim=bi.hidden_dim,
                              dropout=bi.dropout_rate)
    dec = Decoder_Protein_MLP(hidden_dim=bi.hidden_dim, proteins_or_dim=n_prot)
    ep, dp = bi._resolve_ckpt_paths(ckpt_dir, "best")
    enc.load_state_dict(torch.load(ep, map_location=device, weights_only=False))
    dec.load_state_dict(torch.load(dp, map_location=device, weights_only=False))
    enc.to(device).eval(); dec.to(device).eval()
    return enc, dec


def main():
    device = torch.device("cpu")
    out_rows = []      # 汇总
    per_protein = []   # 每个蛋白明细

    for ver, fname in VERSIONS.items():
        for ck_name, ck_dir in CKPTS.items():
            common_gene, common_prot = bi._load_panel(ck_dir)
            enc = dec = None
            for s in SAMPLES:
                d = ROOT / "proc" / "preprocessed" / f"gse_{s}"
                rna = bi._load_direct_rna(str(d / fname))
                adt = ad.read_h5ad(d / "ADT_clr_clean.h5ad")
                adt = adt[rna.obs_names].copy()
                Y = np.asarray(adt.X, dtype=np.float32)
                names = list(adt.var_names)

                rna2, ds, loader, _ = bi._prepare_graph_dataset(
                    rna, common_gene, str(ROOT / "runs" / "graphs_gse" / f"{s}_{ver}"))
                if enc is None:
                    enc, dec = load_model(ck_dir, len(common_gene), len(common_prot), device)
                with torch.no_grad():
                    P = bi._single_forward(loader, enc, dec, device)   # (S, 35)
                P = np.asarray(P, dtype=np.float32)

                marker_idx = [i for i, p in enumerate(names) if not p.startswith(("mouse_", "rat_"))]
                iso_idx = [i for i, p in enumerate(names) if p.startswith(("mouse_", "rat_"))]

                pcc_m = [pearson(P[:, i], Y[:, i]) for i in marker_idx]
                sp_m = [spear(P[:, i], Y[:, i]) for i in marker_idx]
                pcc_i = [pearson(P[:, i], Y[:, i]) for i in iso_idx]
                sp_i = [spear(P[:, i], Y[:, i]) for i in iso_idx]
                # 逐 spot
                spot_p = [pearson(P[k, marker_idx], Y[k, marker_idx]) for k in range(P.shape[0])]
                rmse = float(np.sqrt(np.mean((P[:, marker_idx] - Y[:, marker_idx]) ** 2)))

                # 基线：同名基因表达
                genes = list(rna2.var_names)
                Xg = np.asarray(rna2.X, dtype=np.float32)
                gi = {g: i for i, g in enumerate(genes)}
                base_sp, gain = [], []
                for i in marker_idx:
                    g = PROT2GENE.get(names[i])
                    if g is None or g not in gi:
                        continue
                    v = Xg[:, gi[g]]
                    b = spear(v, Y[:, i]); m = spear(P[:, i], Y[:, i])
                    if np.isfinite(b) and np.isfinite(m):
                        base_sp.append(b); gain.append(m - b)

                out_rows.append({
                    "input": ver, "ckpt": ck_name, "sample": s, "n_spots": P.shape[0],
                    "pcc_marker_mean": float(np.nanmean(pcc_m)),
                    "pcc_marker_median": float(np.nanmedian(pcc_m)),
                    "spear_marker_mean": float(np.nanmean(sp_m)),
                    "spear_marker_median": float(np.nanmedian(sp_m)),
                    "pcc_isotype_mean": float(np.nanmean(pcc_i)),
                    "spear_isotype_mean": float(np.nanmean(sp_i)),
                    "pcc_spot_mean": float(np.nanmean(spot_p)),
                    "rmse_marker": rmse,
                    "baseline_spear_mean": float(np.nanmean(base_sp)) if base_sp else np.nan,
                    "delta_vs_baseline": float(np.nanmean(gain)) if gain else np.nan,
                    "n_baseline": len(base_sp),
                })
                for i in range(len(names)):
                    per_protein.append({
                        "input": ver, "ckpt": ck_name, "sample": s, "protein": names[i],
                        "isotype": names[i].startswith(("mouse_", "rat_")),
                        "pcc": pearson(P[:, i], Y[:, i]),
                        "spearman": spear(P[:, i], Y[:, i]),
                        "std_true": float(Y[:, i].std()),
                    })
                print(f"[{ver:4s}|{ck_name:22s}|{s:6s}] n={P.shape[0]:5d} "
                      f"PCC={np.nanmean(pcc_m):+.3f} SP={np.nanmean(sp_m):+.3f} "
                      f"spotPCC={np.nanmean(spot_p):+.3f} isoSP={np.nanmean(sp_i):+.3f} "
                      f"base={np.nanmean(base_sp) if base_sp else float('nan'):+.3f}", flush=True)

    df = pd.DataFrame(out_rows)
    pp = pd.DataFrame(per_protein)
    df.to_csv("results/gse_external_summary.csv", index=False)
    pp.to_csv("results/gse_external_per_protein.csv", index=False)

    print("\n" + "=" * 78)
    print("按输入版本汇总（31 个标志物蛋白）")
    print(df.groupby(["input", "ckpt"])[["pcc_marker_mean", "spear_marker_mean",
                                         "pcc_spot_mean", "pcc_isotype_mean",
                                         "baseline_spear_mean", "delta_vs_baseline"]]
          .mean().round(4).to_string())
    print("\n各样本（proc 输入, holdout_tonsil）")
    sub = df[(df.input == "proc") & (df.ckpt == "holdout_tonsil")]
    print(sub[["sample", "n_spots", "pcc_marker_mean", "spear_marker_mean",
               "pcc_spot_mean", "pcc_isotype_mean", "baseline_spear_mean"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
