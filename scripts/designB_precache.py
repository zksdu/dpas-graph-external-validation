"""设计 B 预生成图缓存：从 runs/lodo_v3 的 35-panel 缓存派生各 panel 变体的
protein_target 列子集（图结构/X_mRNA 不变），避免每变体重建图的内存峰值。
缓存文件名 tag 沿用原 cfg_tag（图配置未变）。
"""
import os
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC_GRAPHS = ROOT / "runs" / "lodo_v3" / "graphs"
ISOTYPE = ["mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"]
ALL_MARKERS = ["ACTA2", "BCL2", "CCR7", "CD14", "CD163", "CD19", "CD27", "CD274",
               "CD3E", "CD4", "CD40", "CD68", "CD8A", "CEACAM8", "CR2", "CXCR5",
               "EPCAM", "FCGR3A", "HLA_DRA", "ITGAM", "ITGAX", "KRT5", "MS4A1",
               "PAX5", "PCNA", "PDCD1", "PECAM1", "PTPRC_1", "PTPRC_2", "SDC1", "VIM"]
BASE35 = sorted(ISOTYPE + ALL_MARKERS)
assert len(BASE35) == 35

FOLDS = {"holdout_tonsil": ["breast_cancer", "tonsil"],      # train, test
         "holdout_breast_cancer": ["tonsil", "breast_cancer"]}

n_made = 0
for vdir in sorted((ROOT / "proc" / "panel_sub").glob("size*")):
    if not (vdir / "specs.json").exists():
        continue
    keep = [x.strip() for x in (vdir / "dropped.txt").read_text().splitlines() if x.strip()]
    keep_sorted = sorted(p for p in BASE35 if p not in keep)
    col_idx = [BASE35.index(p) for p in keep_sorted]
    for fold, samples in FOLDS.items():
        dst = ROOT / "runs" / "designB" / vdir.name / "graphs" / fold
        dst.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            src = next((SRC_GRAPHS / fold).glob(f"{sample}_4000_*.pth"))
            tag = src.name.split("_", 2)[2]  # cfg tag 保持一致
            out = dst / f"{sample}_4000_{tag}"
            if out.exists():
                continue
            d = torch.load(src, weights_only=False, map_location="cpu")
            t = d["protein_target"]
            assert t.shape[1] == 35, t.shape
            d["protein_target"] = t[:, col_idx].clone()
            torch.save(d, out)
            n_made += 1
    print(f"{vdir.name}: panel={len(keep_sorted)} cache ok")

# 基线 seed1-4 直接复用 35-panel 缓存（复制即可）
for seed in [1, 2, 3, 4]:
    for fold in FOLDS:
        dst = ROOT / "runs" / "designB" / f"base_seed{seed}" / "graphs" / fold
        dst.mkdir(parents=True, exist_ok=True)
        for src in (SRC_GRAPHS / fold).glob("*.pth"):
            out = dst / src.name
            if not out.exists():
                shutil.copyfile(src, out)
                n_made += 1
print(f"done, {n_made} cache files written")
