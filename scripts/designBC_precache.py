# -*- coding: utf-8 -*-
"""designB2 + designC 预生成图缓存：从 lodo_v3 的 35-panel 缓存派生。
protein_target 列 = 面板 h5ad 的 log1p+行中心化值（sorted 列序），
图结构 / X_mRNA 不变。缓存文件名 tag 首段 = 面板蛋白总数（25/20/34）。
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC_GRAPHS = ROOT / "runs" / "lodo_v3" / "graphs"
ISOTYPE = ["mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"]
ALL_MARKERS = ["ACTA2", "BCL2", "CCR7", "CD14", "CD163", "CD19", "CD27", "CD274",
               "CD3E", "CD4", "CD40", "CD68", "CD8A", "CEACAM8", "CR2", "CXCR5",
               "EPCAM", "FCGR3A", "HLA_DRA", "ITGAM", "ITGAX", "KRT5", "MS4A1",
               "PAX5", "PCNA", "PDCD1", "PECAM1", "PTPRC_1", "PTPRC_2", "SDC1", "VIM"]
BASE35 = sorted(ISOTYPE + ALL_MARKERS)
FOLDS = {"holdout_tonsil": ["breast_cancer", "tonsil"],
         "holdout_breast_cancer": ["tonsil", "breast_cancer"]}

manifest = json.loads((ROOT / "proc" / "panel_sub_v2" / "manifest.json").read_text())
n_made = 0
for v in manifest:
    vdir = ROOT / "proc" / "panel_sub_v2" / v["name"]
    keep = [x.strip() for x in (vdir / "dropped.txt").read_text().splitlines() if x.strip()]
    keep_sorted = sorted(p for p in BASE35 if p not in keep)
    col_idx = [BASE35.index(p) for p in keep_sorted]
    for fold, samples in FOLDS.items():
        dst = Path(v["out_root"]) / "graphs" / fold
        dst.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            src = next((SRC_GRAPHS / fold).glob(f"{sample}_4000_*.pth"))
            cfg = src.name.split("_", 3)[3]  # 去掉 sample_4000_35_ 前缀，保留 cfg+后缀
            out = dst / f"{sample}_4000_{len(keep_sorted)}_{cfg}"
            if out.exists():
                continue
            # 面板目标值直接来自 h5ad（log1p+行中心化，sorted 列序）
            sub = ad.read_h5ad(vdir / f"ADT_clr_sub_{sample}.h5ad")
            X = sub.X
            X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
            assert list(sub.var_names) == keep_sorted, "列序必须为 sorted"
            d = torch.load(src, weights_only=False, map_location="cpu")
            assert d["protein_target"].shape[1] == 35
            d["protein_target"] = torch.tensor(X, dtype=torch.float32)
            torch.save(d, out)
            n_made += 1
    print(f"{v['name']}: panel={len(keep_sorted)} cache ok")
print(f"done, {n_made} cache files written")
