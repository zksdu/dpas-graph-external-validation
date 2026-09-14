"""设计 B：in-silico panel 缩减变体生成。
原理：trainer 取训练+测试 ADT var_names 交集为 panel → 只需生成蛋白列子集的
ADT h5ad 与 specs json，无需改任何源码。
抽样：31 标志物中随机移除，4 个同型对照始终保留；子集 CLR（行中心化）与原协议一致。
变体：size ∈ {25, 20} × seed ∈ {1..5}（15 变体）+ 基线 35 面板补 seed 1..4。
用法：python designB_make_panels.py            # 生成全部变体
"""
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ISOTYPE = ["mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"]
ALL_MARKERS = ["ACTA2", "BCL2", "CCR7", "CD14", "CD163", "CD19", "CD27", "CD274",
               "CD3E", "CD4", "CD40", "CD68", "CD8A", "CEACAM8", "CR2", "CXCR5",
               "EPCAM", "FCGR3A", "HLA_DRA", "ITGAM", "ITGAX", "KRT5", "MS4A1",
               "PAX5", "PCNA", "PDCD1", "PECAM1", "PTPRC_1", "PTPRC_2", "SDC1", "VIM"]
assert len(ALL_MARKERS) == 31 and not (set(ISOTYPE) & set(ALL_MARKERS))

OUT = ROOT / "proc" / "panel_sub"
VARIANTS = [(s, r) for s in (25, 20) for r in range(1, 6)]
BASE_SEEDS = [1, 2, 3, 4]  # seed 0 基线已有 runs/lodo_v3

manifest = []
for size, seed in VARIANTS:
    rng = np.random.RandomState(1000 * size + seed)
    drop = list(rng.permutation(ALL_MARKERS)[: 31 - (size - len(ISOTYPE))])
    keep = [p for p in ISOTYPE + ALL_MARKERS if p not in drop]
    assert len(keep) == size
    vdir = OUT / f"size{size}_seed{seed}"
    vdir.mkdir(parents=True, exist_ok=True)
    specs = []
    for ds in ["tonsil", "breast_cancer"]:
        src = ad.read_h5ad(ROOT / f"proc/preprocessed/{ds}/ADT_raw_clean.h5ad")
        sub = src[:, keep].copy()
        X = sub.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        X = np.asarray(X, dtype=np.float64)
        X = X - X.mean(axis=1, keepdims=True)  # 子集上重新 CLR（行中心化）
        sub.X = X.astype(np.float32)
        p = vdir / f"ADT_clr_sub_{ds}.h5ad"
        sub.write_h5ad(p)
        specs.append({"name": ds, "batch_id": 0 if ds == "breast_cancer" else 1,
                      "rna": str(ROOT / f"proc/preprocessed/{ds}/RNA_proc.h5ad"),
                      "adt": str(p)})
        del src
    (vdir / "specs.json").write_text(json.dumps(specs, indent=1), encoding="utf-8")
    (vdir / "dropped.txt").write_text("\n".join(drop) + "\n", encoding="utf-8")
    manifest.append({"kind": "sub", "size": size, "seed": seed,
                     "out_root": f"runs/designB/size{size}_seed{seed}",
                     "specs": str(vdir / "specs.json")})
    print(f"size{size}_seed{seed}: drop {drop}")

for seed in BASE_SEEDS:
    manifest.append({"kind": "base", "size": 35, "seed": seed,
                     "out_root": f"runs/designB/base_seed{seed}",
                     "specs": str(ROOT / "proc/pairs_preprocessed_v3.json")})

(ROOT / "proc" / "panel_sub" / "manifest.json").write_text(
    json.dumps(manifest, indent=1), encoding="utf-8")
print(f"\n{len(manifest)} variants written (14 new runs; base seed0 = runs/lodo_v3)")
