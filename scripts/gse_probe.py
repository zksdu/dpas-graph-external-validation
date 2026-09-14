"""探查 GSE263617（4 个配对 CytAssist 样本）的结构与蛋白 panel。"""
import glob
import os
import numpy as np
import pandas as pd
import anndata as ad
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "benchmark", "GSE263617")

SAMPLES = [("A1LN", "GSM8195494_A1_LN.h5ad", "GSM8195498_A1_LN_Protein.h5ad", "GSM8195498_A1LN_isotype_normalization_factors.csv.gz"),
           ("A1TNSL", "GSM8195495_A1_TNSL.h5ad", "GSM8195499_A1_TNSL_Protein.h5ad", "GSM8195499_A1TNSL_isotype_normalization_factors.csv.gz"),
           ("D1LN", "GSM8195496_D1_LN.h5ad", "GSM8195500_D1_LN_Protein.h5ad", "GSM8195500_D1LN_isotype_normalization_factors.csv.gz"),
           ("D1TNSL", "GSM8195497_D1_TNSL.h5ad", "GSM8195501_D1_TNSL_Protein.h5ad", "GSM8195501_D1TNSL_isotype_normalization_factors.csv.gz")]

panels = {}
for name, rna_f, prot_f, iso_f in SAMPLES:
    rp = os.path.join(D, rna_f)
    pp = os.path.join(DP := D, prot_f)
    print("=" * 70)
    print(name)
    a = ad.read_h5ad(rp, backed="r")
    print(f"  RNA  : {a.shape[0]} spots x {a.shape[1]} genes")
    print(f"  obs  : {list(a.obs.columns)[:12]}")
    print(f"  obsm : {list(a.obsm.keys())}")
    print(f"  var  : {list(a.var.columns)[:8]}")
    a.file.close()

    b = ad.read_h5ad(pp)
    print(f"  PROT : {b.shape[0]} spots x {b.shape[1]} proteins")
    print(f"  obs  : {list(b.obs.columns)[:12]}")
    print(f"  layers: {list(b.layers.keys())}")
    panels[name] = list(b.var_names)
    print(f"  panel: {list(b.var_names)}")

    ip = os.path.join(D, iso_f)
    iso = pd.read_csv(ip)
    print(f"  isotype factors: {iso.shape}, cols={list(iso.columns)[:6]}")
    print(iso.head(3).to_string())
    break_only_for_iso = False

print("=" * 70)
print("panel identity across samples:")
base = panels["A1LN"]
for k, v in panels.items():
    print(f"  {k}: n={len(v)} same_as_A1LN={v == base}")

# 与训练 panel 对比
import json, glob as g
cands = g.glob(os.path.join(ROOT, "runs", "**", "*protein_names*"), recursive=True)
print("\ntraining panel files:", cands[:5])
tp = None
for c in cands:
    if "tonsil" in c or "pe_" in c:
        tp = [l.strip() for l in open(c) if l.strip()]
        print("use", c, len(tp))
        break
if tp:
    print("train panel:", tp)
    print("overlap with GSE:", len(set(tp) & set(base)), "/", len(tp))
    print("GSE only:", sorted(set(base) - set(tp)))
    print("train only:", sorted(set(tp) - set(base)))
