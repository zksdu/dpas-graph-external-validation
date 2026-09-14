# -*- coding: utf-8 -*-
"""MLP baseline (reviewer request: non-linear, no-graph baseline).
Protocol mirrors scripts/competitor_benchmark.py exactly:
  - input: top-50 transcriptomic PCs of the two paired training sections
  - targets: log1p(raw ADT) row-centered (training); truth = ADT_clr_clean (external)
  - metric: per-protein Spearman over the 31 markers, isotypes excluded
  - all models fitted on training data only, applied zero-shot to external sections
Architectures: one hidden layer (64,) and two hidden layers (64,32); 3 seeds each.
Output: results/mlp_baseline_per_protein.csv, results/mlp_baseline_summary.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MARKERS = ["ACTA2", "BCL2", "CCR7", "CD14", "CD163", "CD19", "CD27", "CD274",
           "CD3E", "CD4", "CD40", "CD68", "CD8A", "CEACAM8", "CR2", "CXCR5",
           "EPCAM", "FCGR3A", "HLA_DRA", "ITGAM", "ITGAX", "KRT5", "MS4A1",
           "PAX5", "PCNA", "PDCD1", "PECAM1", "PTPRC_1", "PTPRC_2", "SDC1", "VIM"]
SAMPLES = ["A1LN", "A1TNSL", "D1LN", "D1TNSL"]
TRAIN = ["tonsil", "breast_cancer"]
SEEDS = [0, 1, 2]
ARCHS = {"mlp_64": (64,), "mlp_64_32": (64, 32)}


def dense(x):
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float64)


def log_clr(x):
    x = np.log1p(np.clip(x, 0, None))
    return x - x.mean(axis=1, keepdims=True)


# ---------- training data ----------
import anndata as ad

Xs, Ys, genes, prot_order = [], [], None, None
for t in TRAIN:
    rna = ad.read_h5ad(f"proc/preprocessed/{t}/RNA_proc.h5ad")
    adt = ad.read_h5ad(f"proc/preprocessed/{t}/ADT_raw_clean.h5ad")
    if genes is None:
        genes = list(rna.var_names)
        prot_order = list(adt.var_names)
    else:
        assert list(rna.var_names) == genes and list(adt.var_names) == prot_order
    Xs.append(dense(rna.X))
    Ys.append(log_clr(dense(adt.X)))
X_tr = np.vstack(Xs)
Y_tr = np.vstack(Ys)
print("train:", X_tr.shape, Y_tr.shape, flush=True)

pca = PCA(n_components=50, random_state=0)
Z_tr = pca.fit_transform(X_tr)
print("PCA explained var:", round(float(pca.explained_variance_ratio_.sum()), 3), flush=True)

sc = StandardScaler().fit(Z_tr)  # feature scaling for the MLP, fitted on train only
Zs_tr = sc.transform(Z_tr)

# ---------- external sections ----------
ext = {}
for s in SAMPLES:
    rna = ad.read_h5ad(f"proc/preprocessed/gse_{s}/RNA_proc.h5ad")
    adt = ad.read_h5ad(f"proc/preprocessed/gse_{s}/ADT_clr_clean.h5ad")
    Z_te = pca.transform(dense(rna[:, genes].X))
    truth = dense(adt.X)
    truth_cols = list(adt.var_names)
    ext[s] = (sc.transform(Z_te), truth, truth_cols)
    print("ext loaded:", s, Z_te.shape, flush=True)

# ---------- fit + evaluate ----------
rows = []
for arch_name, hidden in ARCHS.items():
    for seed in SEEDS:
        m = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", alpha=1e-3,
                         max_iter=600, early_stopping=True, n_iter_no_change=20,
                         validation_fraction=0.15, random_state=seed)
        m.fit(Zs_tr, Y_tr)
        print(arch_name, "seed", seed, "iters:", m.n_iter_, flush=True)
        for s in SAMPLES:
            Z_te, truth, truth_cols = ext[s]
            Y_hat = m.predict(Z_te)
            for p in MARKERS:
                j_pred = prot_order.index(p)
                j_true = truth_cols.index(p)
                r = spearmanr(Y_hat[:, j_pred], truth[:, j_true])
                rho = r.statistic if hasattr(r, "statistic") else r.correlation
                if np.std(Y_hat[:, j_pred]) < 1e-12:
                    rho = np.nan
                rows.append(dict(model=arch_name, seed=seed, sample=s,
                                 protein=p, spearman=float(rho)))

long = pd.DataFrame(rows)
long.to_csv("results/mlp_baseline_per_protein.csv", index=False)

summ = (long.groupby(["model", "seed"]).spearman.mean().rename("sp_mean").reset_index())
final = (summ.groupby("model").sp_mean.agg(["mean", "std", "count"]).reset_index()
         .rename(columns={"mean": "sp_mean", "std": "sp_sd", "count": "n_seeds"}))
final.to_csv("results/mlp_baseline_summary.csv", index=False)
print(summ.to_string(index=False))
print(final.to_string(index=False))
print("DONE")
