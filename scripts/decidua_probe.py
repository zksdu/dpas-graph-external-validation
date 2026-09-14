# -*- coding: utf-8 -*-
"""探查 Decidua atlas h5ad 的结构：obs 字段、供体/条件构成、细胞类型。"""
import h5py
import numpy as np
import pandas as pd

F = "data/pe_placenta/Decidua_atlas_raw_counts_20230711.h5ad"

with h5py.File(F, "r") as h:
    print("keys:", list(h.keys()))
    obs = h["obs"]
    # anndata 0.8+ 的 obs 是 group-of-groups
    cols = list(obs.keys())
    print("obs cols:", cols)
    for c in cols:
        try:
            if isinstance(obs[c], h5py.Group):
                v = obs[c]["codes"] if "codes" in obs[c] else obs[c]["values"]
                print(f"  {c}: n={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {c}: n={obs[c].shape}, dtype={obs[c].dtype}")
        except Exception as e:
            print(f"  {c}: ERR {e}")
    print("X shape:", h["X/shape"][:] if "X/shape" in h else h["X"].shape)
    if "var" in h:
        vn = h["var/_index"][:]
        print("var n:", len(vn), "head:", [x.decode() if isinstance(x, bytes) else x for x in vn[:5]])
