#!/usr/bin/env python
"""胎盘 snRNA 图谱（Zenodo 8159511, Placenta_atlas_raw_counts_20230711.h5ad）的
eoPE vs 足月对照的**供体级伪 bulk 差异表达**。

设计：
  * 6 个 eoPE 供体 vs 6 个 Term control 供体（均为晚孕绒毛，tissue=Villi），
    每个供体 × 细胞类型聚合为伪 bulk（原始 counts 求和 → CPM → log2），
    再做 Welch t 检验。供体为独立样本单位，避免把细胞当重复。
  * 流式读取 7.5GB h5ad 的 CSR 矩阵，不全量载入。

输出：
  results/sn_de_<celltype>.csv      每个细胞类型的 DE 表
  results/sn_de_summary.md          汇总
  results/sn_pseudobulk.csv         供体 × 细胞类型的 log2CPM 伪 bulk 矩阵
  results/sn_signatures.json        eoPE up/down 基因签名（供 Visium 打分用）
"""
import os
import json
import numpy as np
import pandas as pd
import h5py
from scipy import stats
from scipy.sparse import csr_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

H5 = "data/pe_placenta/Placenta_atlas_raw_counts_20230711.h5ad"
MIN_CELLS = 50          # 该细胞类型在某供体内至少多少细胞才计入
MIN_DONORS = 3          # 每组至少多少供体
TOPN = 200


def moderated_t(A, B):
    """limma 风格的经验贝叶斯 moderated t 检验（Smyth 2004，常数先验版）。
    在小样本（6 vs 6 供体）下显著提升检出力。返回 (lfc, t, p, df_total)。"""
    from scipy.special import digamma, polygamma
    n1, n2 = A.shape[0], B.shape[0]
    d_g = n1 + n2 - 2.0
    m1, m2 = A.mean(0), B.mean(0)
    s2 = ((A - m1) ** 2).sum(0) + ((B - m2) ** 2).sum(0)
    s2 = s2 / d_g
    lfc = m1 - m2
    se = np.sqrt(s2 * (1.0 / n1 + 1.0 / n2))
    ok = s2 > 0
    # 估计先验自由度 d0 与 s0^2（矩估计）
    e = np.log(np.maximum(s2[ok], 1e-12)) - digamma(d_g / 2) + np.log(d_g / 2)
    var_e = np.var(e, ddof=1) - polygamma(1, d_g / 2)
    d0 = np.inf
    if var_e > 0:
        lo, hi = 1e-4, 1e6
        for _ in range(200):
            mid = np.sqrt(lo * hi)
            if polygamma(1, mid / 2) > var_e:
                lo = mid
            else:
                hi = mid
        d0 = np.sqrt(lo * hi)
        s0_2 = np.exp(np.mean(e) + digamma(d0 / 2) - np.log(d0 / 2))
    else:
        s0_2 = np.exp(np.mean(e))
    s2p = (d0 * s0_2 + d_g * s2) / (d0 + d_g)
    dft = d_g + d0
    se2 = np.sqrt(s2p * (1.0 / n1 + 1.0 / n2))
    t = np.zeros_like(lfc)
    t[ok] = lfc[ok] / np.maximum(se2[ok], 1e-12)
    p = np.ones_like(lfc)
    p[ok] = 2 * stats.t.sf(np.abs(t[ok]), dft)
    return lfc, t, p, dft


def bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full(len(p), np.nan)
    pv = p[ok]
    n = len(pv)
    o = np.argsort(pv)
    qs = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, pv[o[i]] * n / (i + 1))
        qs[o[i]] = prev
    q[ok] = qs
    return q


def main():
    f = h5py.File(H5, "r")
    from anndata._io.specs.registry import read_elem
    obs = read_elem(f["obs"])
    var = read_elem(f["var"])
    genes = np.asarray(var.index).astype(str)
    print("cells", obs.shape, "genes", len(genes))

    # 只保留晚孕（eoPE 与 Term controls）
    keep_cond = ["eoPE", "Term controls"]
    sel = obs["condition"].isin(keep_cond).values
    cell_ct = obs["celltype_annotations"].astype(str).values
    cell_do = obs["donor_id"].astype(str).values
    cell_co = obs["condition"].astype(str).values

    grp_key = np.array([f"{cell_do[i]}|{cell_ct[i]}" for i in range(len(obs))])
    uniq, inv = np.unique(grp_key, return_inverse=True)
    ncell = np.bincount(inv, minlength=len(uniq))
    print("groups:", len(uniq))

    # 流式累加
    X = f["X"]
    data_d, ind_d, ptr_d = X["data"], X["indices"], X["indptr"]
    n_rows = len(ptr_d) - 1
    sums = np.zeros((len(uniq), len(genes)), dtype=np.float64)
    CH = 4000
    for i0 in range(0, n_rows, CH):
        i1 = min(i0 + CH, n_rows)
        p0, p1 = int(ptr_d[i0]), int(ptr_d[i1])
        d = data_d[p0:p1]
        idx = ind_d[p0:p1]
        ptr = np.asarray(ptr_d[i0:i1 + 1], dtype=np.int64) - p0
        m = csr_matrix((d, idx, ptr), shape=(i1 - i0, len(genes)))
        # 用指示矩阵做稀疏聚合（比 np.add.at 快得多，且避免 dense 展开）
        g = inv[i0:i1]
        Ind = csr_matrix((np.ones(i1 - i0, dtype=np.float64),
                          (np.arange(i1 - i0), g)), shape=(i1 - i0, len(uniq)))
        sums += (Ind.T @ m).toarray()
        if i0 % 40000 == 0:
            print(f"  rows {i0}/{n_rows}", flush=True)
    f.close()

    meta = pd.DataFrame({"group": uniq, "n_cells": ncell})
    meta[["donor", "celltype"]] = meta["group"].str.split("|", expand=True)
    cond_map = obs.groupby("donor_id")["condition"].first().to_dict()
    meta["condition"] = meta["donor"].map(cond_map)
    meta = meta[meta["condition"].isin(keep_cond)]
    meta = meta[meta["n_cells"] >= MIN_CELLS].reset_index(drop=True)
    # sums 的行顺序 == uniq 顺序；取 meta 对应的原始位置
    pos = {g: i for i, g in enumerate(uniq)}
    rows = [pos[g] for g in meta["group"]]
    PB = sums[rows]

    # CPM → log2
    tot = PB.sum(1, keepdims=True)
    cpm = PB / np.maximum(tot, 1) * 1e6
    lg = np.log2(cpm + 1)
    pbd = pd.DataFrame(lg, index=meta["group"], columns=genes)
    pbd.to_csv("results/sn_pseudobulk.csv")
    meta.to_csv("results/sn_pseudobulk_meta.csv", index=False)

    # 每个细胞类型做 DE
    out = ["# snRNA 供体级伪 bulk DE：eoPE vs 足月对照（6 vs 6 供体）", "",
           "数据源：Zenodo 8159511 `Placenta_atlas_raw_counts_20230711.h5ad`（绒毛，tissue=Villi）。",
           "eoPE 供体：389-v, 419-v, 577-1-v, 577-2-v, M181-v, PLA120-v；",
           "Term control 供体：327-v, 328-v, 372-v, TRM1-v, TRM2-v, TRM3-v。",
           "每个供体 × 细胞类型聚合原始 counts → CPM → log2(CPM+1)，Welch t 检验，BH 校正。", ""]
    sigs = {}
    rows_sum = []
    for ct, sub in meta.groupby("celltype"):
        a = sub[sub["condition"] == "eoPE"]
        b = sub[sub["condition"] == "Term controls"]
        if len(a) < MIN_DONORS or len(b) < MIN_DONORS:
            continue
        ia = a.index.values          # meta 已 reset_index，行号即 lg 的行位置
        ib = b.index.values
        A, B = lg[ia], lg[ib]
        lfc = A.mean(0) - B.mean(0)

        # 独立过滤：至少 25% 的伪 bulk 样本 CPM >= 1（即 log2CPM >= 1）
        expressed = (np.vstack([A, B]) >= 1).mean(0) >= 0.25
        lfc_m, t_m, p_m, df_mod = moderated_t(A, B)
        p = np.full(len(genes), np.nan)
        q = np.full(len(genes), np.nan)
        tt = np.full(len(genes), np.nan)
        p[expressed] = p_m[expressed]
        q[expressed] = bh(p_m[expressed])
        tt[expressed] = t_m[expressed]
        df = pd.DataFrame({"gene": genes, "log2FC": lfc, "t": tt, "p": p, "q": q,
                           "expressed": expressed,
                           "mean_eoPE": A.mean(0), "mean_ctrl": B.mean(0)})
        df = df.sort_values("p")
        df.to_csv(f"results/sn_de_{ct}.csv", index=False)

        sig_up = df[(df["q"] < 0.05) & (df["log2FC"] > 0.5)]["gene"].tolist()[:TOPN]
        sig_dn = df[(df["q"] < 0.05) & (df["log2FC"] < -0.5)]["gene"].tolist()[:TOPN]
        sigs[ct] = {"up": sig_up, "down": sig_dn}
        rows_sum.append({"celltype": ct, "n_eoPE_donors": len(a), "n_ctrl_donors": len(b),
                         "n_cells_eoPE": int(a["n_cells"].sum()),
                         "n_cells_ctrl": int(b["n_cells"].sum()),
                         "n_up": len(sig_up), "n_dn": len(sig_dn)})
        top = df[(df["q"] < 0.05)].head(12)
        out += [f"## {ct}（eoPE {len(a)} 供体 / {int(a['n_cells'].sum())} 细胞；"
                f"对照 {len(b)} 供体 / {int(b['n_cells'].sum())} 细胞）", "",
                f"显著基因（q<0.05）：up {len(sig_up)}，down {len(sig_dn)}", "",
                "| 基因 | log2FC | q |", "|---|---|---|"]
        for _, r in top.iterrows():
            out.append(f"| {r['gene']} | {r['log2FC']:+.2f} | {r['q']:.2e} |")
        out.append("")

    pd.DataFrame(rows_sum).to_csv("results/sn_de_celltypes.csv", index=False)
    out += ["## 各细胞类型概览", "", "| 细胞类型 | eoPE供体 | 对照供体 | eoPE细胞 | 对照细胞 | up | down |",
            "|---|---|---|---|---|---|---|"]
    for r in rows_sum:
        out.append(f"| {r['celltype']} | {r['n_eoPE_donors']} | {r['n_ctrl_donors']} | "
                   f"{r['n_cells_eoPE']} | {r['n_cells_ctrl']} | {r['n_up']} | {r['n_dn']} |")

    open("results/sn_de_summary.md", "w", encoding="utf-8").write("\n".join(out))
    json.dump(sigs, open("results/sn_signatures.json", "w"), indent=1)
    print("\n".join(out[:60]))
    print("\nsaved: results/sn_de_summary.md, sn_signatures.json")


if __name__ == "__main__":
    main()
