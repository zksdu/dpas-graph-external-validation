# 数据驱动判定 sectionA/sectionB 的疾病身份
# PE（尤其早发型）胎盘经典上调：FLT1(sFlt-1)、ENG、PAPPA2、INHA、HTRA4、LEP、CGB 偏高；PLGF(PGF) 下降
import numpy as np
import pandas as pd
from pathlib import Path
import anndata as ad
from scipy.stats import mannwhitneyu

H5AD = Path(r"D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/h5ad")
EXPORT = Path(r"D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/export")
OUT = Path(r"D:/workbuddy/0911/pe-virtual-protein/results")

PE_UP = ["FLT1", "ENG", "PAPPA2", "INHA", "HTRA4", "LEP", "CGB3", "CGB5", "CSH1"]
PE_DOWN = ["PGF", "PLAC1", "PSG1", "TFAP2C", "GCM1"]

def frac_expr(X, names, gs):
    idx = [names.index(g) for g in gs if g in names]
    if not idx:
        return None
    Xi = X[:, idx]
    lib = X.sum(axis=1)
    return (Xi.sum(axis=1) / np.maximum(lib, 1)) * 1e4  # per-10k 归一化

data = {}
for f in ["sectionA", "sectionB"]:
    a = ad.read_h5ad(H5AD / f"{f}_RNA_raw.h5ad")
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    data[f] = (X.astype(np.float32), list(a.var_names))
    print(f"{f}: {X.shape[0]} spots, mean libsize = {X.sum(axis=1).mean():.0f}")

print("\n=== PE 标志基因表达（per-10k 归一化，均值）===")
print(f"{'基因':10s} {'sectionA':>12s} {'sectionB':>12s} {'A/B 比值':>10s} {'方向':>8s}")
for gs, label in [(PE_UP, "PE上调"), (PE_DOWN, "PE下调")]:
    print(f"-- {label} --")
    for g in gs:
        va = frac_expr(data["sectionA"][0], data["sectionA"][1], [g])
        vb = frac_expr(data["sectionB"][0], data["sectionB"][1], [g])
        if va is None or vb is None:
            continue
        ma, mb = va.mean(), vb.mean()
        ratio = ma / mb if mb > 0 else np.inf
        u, p = mannwhitneyu(va, vb)
        star = "***" if p < 1e-10 else ("**" if p < 1e-4 else ("*" if p < 0.05 else ""))
        print(f"{g:10s} {ma:12.3f} {mb:12.3f} {ratio:10.2f} {ratio>1.2 and 'A高' or (ratio<0.83 and 'B高' or '~'):>8s} {star}")

print("\n=== 解卷积细胞类型比例均值（%）===")
for f in ["sectionA", "sectionB"]:
    d = pd.read_csv(EXPORT / f"{f}_deconv.csv")
    print(f"{f}: " + " | ".join(f"{c}={d[c].mean()*100:.1f}" for c in d.columns if c != "barcode"))
