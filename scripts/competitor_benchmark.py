# -*- coding: utf-8 -*-
"""增强②：竞品基线对比。同一外部金标准（GSE263617 4 样本）上，与 DPAS-Graph
及 namesake 基因基线同口径对比三个轻量竞品：
  ridge_pc        — 训练配对切片上拟合 PCA(50) + 多输出 Ridge（零样本外推）
  ridge_pc_coord  — 同上 + 空间坐标特征
  knn15           — PC 空间 k=15 逆距离加权迁移
协议与 DPAS 完全一致：训练目标 = log1p(raw ADT) 行中心化；外部真值 = ADT_clr_clean；
指标 = 逐蛋白 Spearman（31 标志物，同型对照排除）。
输出：results/competitor_benchmark_per_protein.csv, results/competitor_benchmark_summary.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MARKERS = ["ACTA2", "BCL2", "CCR7", "CD14", "CD163", "CD19", "CD27", "CD274",
           "CD3E", "CD4", "CD40", "CD68", "CD8A", "CEACAM8", "CR2", "CXCR5",
           "EPCAM", "FCGR3A", "HLA_DRA", "ITGAM", "ITGAX", "KRT5", "MS4A1",
           "PAX5", "PCNA", "PDCD1", "PECAM1", "PTPRC_1", "PTPRC_2", "SDC1", "VIM"]
SAMPLES = ["A1LN", "A1TNSL", "D1LN", "D1TNSL"]
TRAIN = ["tonsil", "breast_cancer"]


def dense(x):
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float64)


def log_clr(x):
    x = np.log1p(np.clip(x, 0, None))
    return x - x.mean(axis=1, keepdims=True)


# ---------- 训练数据 ----------
Xs, Ys, Cs, genes = [], [], [], None
prot_order = None
for t in TRAIN:
    rna = __import__("anndata").read_h5ad(f"proc/preprocessed/{t}/RNA_proc.h5ad")
    adt = __import__("anndata").read_h5ad(f"proc/preprocessed/{t}/ADT_raw_clean.h5ad")
    if genes is None:
        genes = list(rna.var_names)
        prot_order = list(adt.var_names)
    else:
        assert list(rna.var_names) == genes and list(adt.var_names) == prot_order
    Xs.append(dense(rna.X))
    Ys.append(log_clr(dense(adt.X)))
    Cs.append(np.asarray(rna.obsm["spatial"], dtype=np.float64))
X_tr = np.vstack(Xs)
Y_tr = np.vstack(Ys)
C_tr = np.vstack(Cs)
print("train:", X_tr.shape, Y_tr.shape)

# ---------- PCA ----------
pca = PCA(n_components=50, random_state=0)
Z_tr = pca.fit_transform(X_tr)
print("PCA explained var:", round(float(pca.explained_variance_ratio_.sum()), 3))

# alpha 内部 5 折 CV（多输出 MSE）
kf = KFold(n_splits=5, shuffle=True, random_state=0)
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
best_alpha, best_mse = None, np.inf
for a in alphas:
    mse = 0.0
    for tr_i, va_i in kf.split(Z_tr):
        m = Ridge(alpha=a).fit(Z_tr[tr_i], Y_tr[tr_i])
        mse += float(np.mean((m.predict(Z_tr[va_i]) - Y_tr[va_i]) ** 2))
    mse /= kf.get_n_splits()
    print(f"alpha={a}: cv-mse={mse:.4f}")
    if mse < best_mse:
        best_mse, best_alpha = mse, a
print("best alpha:", best_alpha)

# 坐标标准化器（训练拟合）
sc_c = StandardScaler().fit(C_tr)
Zc_tr = np.hstack([Z_tr, sc_c.transform(C_tr) * 0.5])  # 坐标降权，避免坐标主导

models = {
    "ridge_pc": Ridge(alpha=best_alpha).fit(Z_tr, Y_tr),
    "ridge_pc_coord": Ridge(alpha=best_alpha).fit(Zc_tr, Y_tr),
}
# 单组织训练对照（与 DPAS LODO ckpt 同等数据量，零样本到外部）
single_models = {}
for t, (Xi, Yi) in zip(TRAIN, zip(Xs, Ys)):
    Zi = pca.transform(Xi)
    single_models[t] = Ridge(alpha=best_alpha).fit(Zi, Yi)
# 内部 5 折 CV Spearman（判断外部水平的合理性）
from scipy.stats import spearmanr as _spr
cv_sp = []
for tr_i, va_i in kf.split(Z_tr):
    m = Ridge(alpha=best_alpha).fit(Z_tr[tr_i], Y_tr[tr_i])
    P = m.predict(Z_tr[va_i])
    cv_sp += [_spr(P[:, j], Y_tr[va_i, j]).statistic for j in range(Y_tr.shape[1])]
print("internal CV spearman (pooled ridge):", round(float(np.nanmean(cv_sp)), 4))
# knn15 参照
knn_k = 15


def knn_predict(Z_te, Z_ref, Y_ref, k=knn_k):
    out = np.empty((Z_te.shape[0], Y_ref.shape[1]))
    for i in range(Z_te.shape[0]):
        d2 = np.sum((Z_ref - Z_te[i]) ** 2, axis=1)
        idx = np.argsort(d2)[:k]
        w = 1.0 / (np.sqrt(d2[idx]) + 1e-6)
        out[i] = (w[:, None] * Y_ref[idx]).sum(0) / w.sum()
    return out


# ---------- 外部评估 ----------
rows = []
import anndata as ad  # noqa: E402

for s in SAMPLES:
    rna = ad.read_h5ad(f"proc/preprocessed/gse_{s}/RNA_proc.h5ad")
    adt = ad.read_h5ad(f"proc/preprocessed/gse_{s}/ADT_clr_clean.h5ad")
    X_te = dense(rna[:, genes].X)  # 按训练基因序重排
    Z_te = pca.transform(X_te)
    Zc_te = np.hstack([Z_te, sc_c.transform(np.asarray(rna.obsm["spatial"], dtype=np.float64)) * 0.5])
    truth = dense(adt.X)
    truth_cols = list(adt.var_names)
    preds = {"ridge_pc": models["ridge_pc"].predict(Z_te),
             "ridge_pc_coord": models["ridge_pc_coord"].predict(Zc_te),
             "knn15": knn_predict(Z_te, Z_tr, Y_tr)}
    # 单组织模型平均（与 DPAS 两个 LODO ckpt 平均同构）
    sp_mean = np.mean([single_models[t].predict(Z_te) for t in TRAIN], axis=0)
    preds["ridge_pc_single_avg"] = sp_mean
    for model, Y_hat in preds.items():
        # 预测列序 = 训练 ADT var 序；对齐到真值列序
        for p in MARKERS:
            j_pred = prot_order.index(p)
            j_true = truth_cols.index(p)
            r = spearmanr(Y_hat[:, j_pred], truth[:, j_true])
            rho = r.statistic if hasattr(r, "statistic") else r.correlation
            if np.std(Y_hat[:, j_pred]) < 1e-12:
                rho = np.nan
            rows.append(dict(model=model, sample=s, protein=p, spearman=float(rho)))
    print(f"{s} done")

long = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
long.to_csv("results/competitor_benchmark_per_protein.csv", index=False)

# ---------- 汇总（含 DPAS 与 namesake 基线，同口径） ----------
summ = long.groupby("model").spearman.mean().rename("sp_mean").reset_index()
try:
    dpas = pd.read_csv("results/gse_external_per_protein.csv")
    dpas = dpas[(dpas.input == "proc") & (~dpas.isotype)]
    dmean = dpas.groupby("sample").spearman.mean().mean()
    summ = pd.concat([summ, pd.DataFrame([{"model": "DPAS-Graph (mean of 2 ckpts)", "sp_mean": dmean}])],
                     ignore_index=True)
except Exception as e:
    print("DPAS row skip:", e)
summ = summ.sort_values("sp_mean", ascending=False)
summ.to_csv("results/competitor_benchmark_summary.csv", index=False)
print(summ.to_string(index=False))
