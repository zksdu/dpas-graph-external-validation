# -*- coding: utf-8 -*-
"""疾病签名 × 虚拟蛋白关联热图（A/B 双切片）。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COMPS = ["tropho", "immune", "stroma", "endo"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4))
for ax, sec in zip(axes, ["A", "B"]):
    d = pd.read_csv(f"results/disease_protein_assoc_{sec}.csv")
    piv = d.pivot(index="protein", columns="compartment", values="spearman").loc[:, COMPS]
    iso = d[d.isotype].protein.unique()
    mat = piv.values.astype(float)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.45, vmax=0.45, aspect="auto")
    ax.set_xticks(range(len(COMPS)))
    ax.set_xticklabels(COMPS, rotation=30, fontsize=6)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(
        [("* " + p) if p in iso else p for p in piv.index], fontsize=5.5)
    ax.set_title(f"section{sec}: signature score ×\nvirtual protein (Spearman ρ)", fontsize=7.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=5, color="k" if abs(v) < 0.3 else "w")
    cb = cb = plt.colorbar(im, ax=ax, shrink=0.6)
    cb.ax.tick_params(labelsize=5.5)
    cb.ax.tick_params(labelsize=5.5)
plt.tight_layout()
plt.savefig("results/disease_protein_assoc_heatmap.png", dpi=600)
print("saved results/disease_protein_assoc_heatmap.png")
