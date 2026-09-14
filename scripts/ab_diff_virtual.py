# A（对照）vs B（推断 eoPE）虚拟蛋白差异分析
# 关键：虚拟蛋白跨切片比较需先做分位数归一化，消除测序深度差异（A 4.3k vs B 12.4k counts）
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(r"D:/workbuddy/0911/pe-virtual-protein/runs")
OUT = Path(r"D:/workbuddy/0911/pe-virtual-protein/results")
CKPT = Path(r"D:/workbuddy/0911/pe-virtual-protein/runs/lodo_v3/ckpt/holdout_breast_cancer")
proteins = [x.strip() for x in (CKPT / "common_protein.txt").read_text().splitlines() if x.strip()]

pred = {}
for f in ["sectionA", "sectionB"]:
    pred[f] = pd.DataFrame(np.load(RUNS / f"pe_{f}_infer.pred.npy"), columns=proteins)

# 注意：不能用秩归一化（会强制两切片边际分布相同，AUC 恒=0.5 无意义）
# 正确做法：直接比较原始虚拟蛋白值，并用“回归掉 spot 文库大小后的残差”做稳健版本
import anndata as _ad

def libsize(f):
    a = _ad.read_h5ad(rf"D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/h5ad/{f}_RNA_raw.h5ad")
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    return np.log1p(X.sum(axis=1)).astype(np.float32)

def resid_on_lib(v, loglib):
    """回归掉文库大小，保留截距（即调整后的均值）"""
    A = np.column_stack([np.ones_like(loglib), loglib])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ coef + coef[0]

libs = {f: libsize(f) for f in ["sectionA", "sectionB"]}

rows = []
for p in proteins:
    a_raw, b_raw = pred["sectionA"][p].values, pred["sectionB"][p].values
    u, pv = mannwhitneyu(b_raw, a_raw)
    auc = u / (len(a_raw) * len(b_raw))
    delta = 2 * auc - 1
    # 深度校正版
    a_adj = resid_on_lib(a_raw, libs["sectionA"])
    b_adj = resid_on_lib(b_raw, libs["sectionB"])
    u2, pv2 = mannwhitneyu(b_adj, a_adj)
    auc2 = u2 / (len(a_adj) * len(b_adj))
    rows.append((p, delta, auc, pv, 2 * auc2 - 1, pv2, a_raw.mean(), b_raw.mean()))

res = pd.DataFrame(rows, columns=["protein", "delta", "auc", "p_value", "delta_adj", "p_adj",
                                  "mean_A", "mean_B"])
res["abs_delta"] = res["delta"].abs()
res = res.sort_values("abs_delta", ascending=False)
# BH 校正
m = len(res)  # 手动 BH 校正
res_sorted = res.sort_values("p_value").reset_index(drop=True)
res_sorted["rank_i"] = np.arange(1, m + 1)
res_sorted["q_value"] = np.minimum(1, res_sorted["p_value"] * m / res_sorted["rank_i"])
res_sorted["q_value"] = res_sorted["q_value"][::-1].cummin()[::-1]
res = res_sorted.sort_values("abs_delta", ascending=False)

print("=== 虚拟蛋白 A(对照) vs B(推断eoPE) 差异 Top 15（分位数归一化后）===")
print(f"{'蛋白':12s} {'delta':>8s} {'AUC':>7s} {'p':>10s} {'q':>8s} {'delta_adj':>10s} {'方向'}")
for _, r in res.head(15).iterrows():
    direction = "B(PE)高" if r.delta > 0 else "B(PE)低"
    print(f"{r.protein:12s} {r.delta:+8.3f} {r.auc:7.3f} {r.p_value:10.2e} {r.q_value:8.3f} {r.delta_adj:+10.3f}   {direction}")

res.to_csv(OUT / "ab_virtual_protein_diff.csv", index=False, encoding="utf-8")

# 可视化：效应量条形图 + 两个代表蛋白的空间分布
fig, ax = plt.subplots(figsize=(8, 5))
top = res.head(15).iloc[::-1]
colors = ["#c0392b" if d > 0 else "#27ae60" for d in top["delta"]]
ax.barh(top["protein"], top["delta"], color=colors)
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("effect size (delta = 2·AUC−1)  |  red: higher in presumed-PE (B)")
ax.set_title("Virtual protein differences: sectionA (control) vs sectionB (inferred eoPE)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "ab_virtual_protein_diff.png", dpi=150)
plt.close(fig)
print("\nsaved:", OUT / "ab_virtual_protein_diff.png", "| csv:", OUT / "ab_virtual_protein_diff.csv")
