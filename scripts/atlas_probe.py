#!/usr/bin/env python
"""探查 Placenta snRNA 图谱 h5ad 的结构（h5py 直读，不载入 X）。

用法: python scripts/atlas_probe.py [h5ad路径]
输出: data/pe_placenta/atlas_probe.txt
"""
import sys
import os
import numpy as np
import pandas as pd
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
path = sys.argv[1] if len(sys.argv) > 1 else "data/pe_placenta/Placenta_atlas_raw_counts_20230711.h5ad"
out = ["atlas: " + path]

f = h5py.File(path, "r")
out.append(f"top keys: {list(f.keys())}")


def shape_of(g):
    if "shape" in g.attrs:
        return tuple(g.attrs["shape"])
    if "shape" in g:
        return tuple(np.asarray(g["shape"])[...])
    return None


if "X" in f:
    out.append(f"X: type={type(f['X']).__name__} shape={shape_of(f['X'])}")
    if isinstance(f["X"], h5py.Group):
        out.append(f"   X subkeys: {list(f['X'].keys())}")
        for k in f["X"].keys():
            try:
                out.append(f"   {k}: shape={f['X'][k].shape} dtype={f['X'][k].dtype}")
            except Exception as e:
                out.append(f"   {k}: {e}")
if "layers" in f:
    for k in f["layers"].keys():
        out.append(f"layers/{k}: shape={shape_of(f['layers'][k])} "
                   f"subkeys={list(f['layers'][k].keys()) if isinstance(f['layers'][k], h5py.Group) else ''}")
if "raw" in f:
    out.append(f"raw keys: {list(f['raw'].keys())}")
    if "X" in f["raw"]:
        out.append(f"raw/X shape={shape_of(f['raw']['X'])}")
    if "var" in f["raw"]:
        out.append(f"raw/var shape={shape_of(f['raw']['var'])}")


def read_obs(g):
    """尽量把 obs / var 读成 DataFrame"""
    try:
        from anndata._io.specs.registry import read_elem
        return read_elem(g)
    except Exception as e:
        out.append(f"  (anndata read_elem failed: {e})")
        if "index" in g:
            idx = np.asarray(g["index"]).astype(str)
            return pd.DataFrame(index=idx)
        return None


for key in ["obs", "var"]:
    if key not in f:
        continue
    d = read_obs(f[key])
    out.append(f"\n=== {key} === shape={d.shape}")
    out.append(f"columns: {list(d.columns)}")
    for c in d.columns:
        v = d[c]
        try:
            nu = v.nunique()
        except Exception:
            nu = -1
        if 0 < nu <= 60:
            out.append(f"\n--- {key}['{c}'] ({nu} levels) ---")
            out.append(v.value_counts(dropna=False).head(60).to_string())
        else:
            out.append(f"--- {key}['{c}']: {nu} unique, dtype={v.dtype} ---")

if "obsm" in f:
    out.append(f"\nobsm: {list(f['obsm'].keys())}")
if "uns" in f:
    out.append(f"uns: {list(f['uns'].keys())[:30]}")

txt = "\n".join(str(x) for x in out)
open("data/pe_placenta/atlas_probe.txt", "w", encoding="utf-8").write(txt)
print(txt[:8000])
