# -*- coding: utf-8 -*-
"""designB2 + designC 统一面板生成（修正目标约定）。

背景：designB 的 size25/20 子集 h5ad 用了 raw 计数行中心化（文件名含 clr 触发
proc 分支，无 log1p），而 base(35) 走 raw 分支 = log1p+行中心化 —— 两组目标
约定不一致。本脚本统一为 **log1p + 行中心化**（与 base/外部真值一致）。

生成：
  proc/panel_sub_v2/size{25,20}_seed{1..5}/  — designB2 重跑面板（10 个）
  proc/panel_sub_v2/drop_<marker>/           — designC 逐标志物消融（31 个）
  proc/panel_sub_v2/manifest.json
所有面板列序 = sorted(4 isotype + 保留标志物)，与缓存/评测的 sorted 约定一致。
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ISOTYPE = ["mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"]
ALL_MARKERS = ["ACTA2", "BCL2", "CCR7", "CD14", "CD163", "CD19", "CD27", "CD274",
               "CD3E", "CD4", "CD40", "CD68", "CD8A", "CEACAM8", "CR2", "CXCR5",
               "EPCAM", "FCGR3A", "HLA_DRA", "ITGAM", "ITGAX", "KRT5", "MS4A1",
               "PAX5", "PCNA", "PDCD1", "PECAM1", "PTPRC_1", "PTPRC_2", "SDC1", "VIM"]
assert len(ALL_MARKERS) == 31

OUT = ROOT / "proc" / "panel_sub_v2"
OUT.mkdir(parents=True, exist_ok=True)

manifest = []
variants = []  # (name, keep_sorted, seed)
for size in (25, 20):
    for seed in range(1, 6):
        rng = np.random.RandomState(1000 * size + seed)
        drop = list(rng.permutation(ALL_MARKERS)[: 31 - (size - len(ISOTYPE))])
        keep = sorted(p for p in ISOTYPE + ALL_MARKERS if p not in drop)
        variants.append((f"size{size}_seed{seed}", keep, seed, drop))
for m in ALL_MARKERS:
    keep = sorted(p for p in ISOTYPE + ALL_MARKERS if p != m)
    variants.append((f"drop_{m}", keep, 1, [m]))

srcs = {}
for ds in ["tonsil", "breast_cancer"]:
    srcs[ds] = ad.read_h5ad(ROOT / f"proc/preprocessed/{ds}/ADT_raw_clean.h5ad")

for name, keep, seed, drop in variants:
    vdir = OUT / name
    vdir.mkdir(parents=True, exist_ok=True)
    specs = []
    for ds in ["tonsil", "breast_cancer"]:
        src = srcs[ds]
        sub = src[:, keep].copy()
        X = sub.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        X = np.asarray(X, dtype=np.float64)
        X = np.log1p(np.clip(X, 0, None))          # 统一约定：log1p
        X = X - X.mean(axis=1, keepdims=True)       # 行中心化（CLR on log 域）
        sub.X = X.astype(np.float32)
        p = vdir / f"ADT_clr_sub_{ds}.h5ad"
        sub.write_h5ad(p)
        specs.append({"name": ds, "batch_id": 0 if ds == "breast_cancer" else 1,
                      "rna": str(ROOT / f"proc/preprocessed/{ds}/RNA_proc.h5ad"),
                      "adt": str(p)})
    (vdir / "specs.json").write_text(json.dumps(specs, indent=1), encoding="utf-8")
    (vdir / "dropped.txt").write_text("\n".join(drop) + "\n", encoding="utf-8")
    root = f"runs/designB2/{name}" if name.startswith("size") else f"runs/designC/{name}"
    manifest.append({"name": name, "kind": "designB2" if name.startswith("size")
                     else "designC", "seed": seed, "out_root": root,
                     "specs": str(vdir / "specs.json")})
    print(f"{name}: keep={len(keep)} drop={drop}")

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
print(f"\n{len(manifest)} variants written")
