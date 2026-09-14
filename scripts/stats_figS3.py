# -*- coding: utf-8 -*-
"""Fig 3D follow-up + Supplementary Figure S3.
1) Wilcoxon rank-sum (Mann-Whitney U, two-sided) on RNA coverage of namesake
   genes: lymphocyte-subtype (lymph) vs compartment markers, per dataset and
   pooled, for detection rate and mean expression. -> results/rna_coverage_wilcoxon.csv
2) Fig S3: Spearman distributions of 31 marker virtual proteins vs 4 isotype
   controls across truth-based experiments (internal LODO 20ep/100ep, external
   20ep) and placental signature associations. -> results/figS3_isotype_floors.png
   + results/figS3_isotype_floors_data.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ISO = {"mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"}
plt.rcParams.update({"font.family": "Arial", "font.size": 7})

# ---------- 1. Wilcoxon on RNA coverage ----------
rc = pd.read_csv("results/rna_coverage_diagnosis.csv")
print("in_panel values:", rc.in_panel.unique())
rcp = rc[rc.in_panel.astype(str).str.lower().eq("true")].copy()
rows = []
for scope, d in [("pooled", rcp),
                 ("tonsil", rcp[rcp.dataset == "tonsil"]),
                 ("breast_cancer", rcp[rcp.dataset == "breast_cancer"])]:
    for var in ["detect_rate", "mean_expr"]:
        a = d[d.group == "lymph"][var].values
        b = d[d.group == "compartment"][var].values
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(scope=scope, variable=var, n_lymph=len(a), n_comp=len(b),
                         median_lymph=float(np.median(a)), median_comp=float(np.median(b)),
                         mean_lymph=float(np.mean(a)), mean_comp=float(np.mean(b)),
                         U=float(u), p=float(p)))
wil = pd.DataFrame(rows)
wil.to_csv("results/rna_coverage_wilcoxon.csv", index=False)
print(wil.to_string(index=False))

# ---------- 2. Fig S3 data ----------
def pp_metrics(path, label, fold):
    d = pd.read_csv(path)
    d["experiment"] = label
    d["fold"] = fold
    d["kind"] = np.where(d.protein.isin(ISO), "isotype", "marker")
    return d[["experiment", "fold", "protein", "spearman", "kind"]]

parts = []
for fold in ["holdout_tonsil", "holdout_breast_cancer"]:
    parts.append(pp_metrics(f"runs/lodo_v3/ckpt/{fold}/per_protein_metrics.csv",
                            "Internal LODO (20 ep)", fold))
    parts.append(pp_metrics(f"results_server_final/runs/lodo_v4_100ep/ckpt/{fold}/per_protein_metrics.csv",
                            "Internal LODO (100 ep)", fold))
ext = pd.read_csv("results/gse_external_per_protein.csv")
ext = ext[ext.input == "proc"].copy()
ext = ext[["ckpt", "sample", "protein", "spearman"]]
ext["experiment"] = "External GSE263617 (20 ep)"
ext["kind"] = np.where(ext.protein.isin(ISO), "isotype", "marker")
ext = ext.rename(columns={"ckpt": "fold"})
parts.append(ext[["experiment", "fold", "protein", "spearman", "kind"]])

for sec, f in [("A", "results/disease_protein_assoc_A.csv"),
               ("B", "results/disease_protein_assoc_B.csv")]:
    d = pd.read_csv(f)
    d = d[["section", "compartment", "protein", "spearman", "isotype"]].copy()
    d["experiment"] = f"Placenta {sec} (signature assoc.)"
    d["fold"] = d.compartment
    d["kind"] = np.where(d.isotype.astype(bool), "isotype", "marker")
    parts.append(d[["experiment", "fold", "protein", "spearman", "kind"]])

data = pd.concat(parts, ignore_index=True)
data.to_csv("results/figS3_isotype_floors_data.csv", index=False)

order = ["Internal LODO (20 ep)", "Internal LODO (100 ep)",
         "External GSE263617 (20 ep)", "Placenta A (signature assoc.)",
         "Placenta B (signature assoc.)"]

# ---------- 3. render ----------
fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.2), sharey=True)
for ax, exp in zip(axes, order):
    d = data[data.experiment == exp]
    m = d[d.kind == "marker"].spearman.values
    iso = d[d.kind == "isotype"].spearman.values
    bp = ax.boxplot([m, iso], widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=0.8))
    for patch, c in zip(bp["boxes"], ["#b2182b", "#7f7f7f"]):
        patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
    rng = (np.nanmin(d.spearman.values), np.nanmax(d.spearman.values))
    for i, arr in enumerate([m, iso], start=1):
        jit = (np.random.RandomState(0).rand(len(arr)) - 0.5) * 0.12
        ax.scatter(np.full(len(arr), i) + jit, arr, s=2.5, color="black", alpha=0.35, zorder=3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["markers\n(n=%d)" % len(m), "isotype\n(n=%d)" % len(iso)], fontsize=6)
    ax.set_title(exp, fontsize=6.5)
    ax.tick_params(labelsize=6)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
axes[0].set_ylabel("Spearman ρ", fontsize=7)
fig.tight_layout(pad=0.4)
fig.savefig("results/figS3_isotype_floors.png", dpi=600)
print("figS3 saved")
print(data.groupby(["experiment", "kind"]).spearman.agg(["count", "mean", "median"]).to_string())
