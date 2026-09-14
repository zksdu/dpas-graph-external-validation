# PE 虚拟蛋白正交验证：pred.npy × 解卷积比例 Spearman + 空间可视化
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(r"D:/workbuddy/0911/pe-virtual-protein/runs")
EXPORT = Path(r"D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/export")
OUT = Path(r"D:/workbuddy/0911/pe-virtual-protein/results")
OUT.mkdir(exist_ok=True)
CKPT = Path(r"D:/workbuddy/0911/pe-virtual-protein/runs/lodo_v3/ckpt/holdout_breast_cancer")

proteins = [x.strip() for x in (CKPT / "common_protein.txt").read_text().splitlines() if x.strip()]

# 生物学预期配对：(虚拟蛋白, 解卷积细胞类型, 方向)
PAIRS = [
    ("PECAM1", "vVEC", +1), ("PECAM1", "PAMM", +1),
    ("ACTA2", "vEVT", +1),
    ("EPCAM", "vVCT", +1), ("EPCAM", "vSCT", +1), ("EPCAM", "vEVT", +1),
    ("KRT5", "vVCT", +1),
    ("PCNA", "APAhi.tropho", +1),
    ("HLA-DRA", "vHBC", +1), ("CD68", "vHBC", +1), ("CD163", "vHBC", +1),
    ("CD3", "vTcell", +1), ("CD8", "vTcell", +1),
    ("CD45", "vTcell", +1),
    ("MUC1", "vSCT", +1),
    ("MET", "vEVT", +1),
]

from scipy.stats import spearmanr

lines = []
results_all = {}
for f in ["sectionA", "sectionB"]:
    pred = np.load(RUNS / f"pe_{f}_infer.pred.npy")  # spots x 35
    deconv = pd.read_csv(EXPORT / f"{f}_deconv.csv").set_index("barcode")
    coords = pd.read_csv(EXPORT / f"{f}_coords.csv").set_index("barcode")
    bcs = pd.read_csv(EXPORT / f"{f}_barcodes.txt", header=None)[0].tolist()
    pmat = pd.DataFrame(pred, index=bcs, columns=proteins)

    print(f"\n===== {f} ({pred.shape[0]} spots) =====")
    for prot, cell, direction in PAIRS:
        if prot not in pmat.columns or cell not in deconv.columns:
            continue
        common = pmat.index.intersection(deconv.index)
        r, p = spearmanr(pmat.loc[common, prot], deconv.loc[common, cell])
        results_all.setdefault(f, []).append((prot, cell, r, p))
        mark = " **" if (direction * r > 0.2 and p < 0.05) else ("  *" if (direction * r > 0.1 and p < 0.05) else "")
        lines.append(f"{f}\t{prot}\t{cell}\t{r:.3f}\t{p:.2e}{mark}")
        print(f"  {prot:10s} vs {cell:14s} rho={r:+.3f} p={p:.1e}{mark}")

    # ---- 空间可视化：4 个代表性虚拟蛋白 ----
    vis = ["PECAM1", "EPCAM", "HLA-DRA", "ACTA2"]
    vis = [v for v in vis if v in pmat.columns]
    fig, axes = plt.subplots(1, len(vis), figsize=(4.2 * len(vis), 4))
    if len(vis) == 1:
        axes = [axes]
    cc = coords.loc[pmat.index]
    for ax, prot in zip(axes, vis):
        v = pmat[prot].values
        order = np.argsort(v)
        s = ax.scatter(cc["imagecol"].values[order], cc["imagerow"].values[order],
                       c=v[order], s=8, cmap="viridis")
        ax.set_title(f"{prot} (virtual)", fontsize=11)
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
        plt.colorbar(s, ax=ax, fraction=0.046)
    fig.suptitle(f"{f}: PE placenta virtual protein spatial maps", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"{f}_virtual_protein_maps.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / f'{f}_virtual_protein_maps.png'}")

with open(OUT / "deconv_validation.tsv", "w", encoding="utf-8") as fh:
    fh.write("section\tprotein\tcelltype\tspearman_rho\tp_value\tnotes\n")
    fh.write("\n".join(l.replace(" **", "\tvalid").replace("  *", "\tweak") for l in lines) + "\n")
print("\nall done")
