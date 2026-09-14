#!/usr/bin/env python
"""蜕膜 snRNA 图谱（Zenodo 8159511, Decidua_atlas_raw_counts_20230711.h5ad）的
eoPE vs 足月对照的**供体级伪 bulk 差异表达**。

设计：4 eoPE 供体（late_preterm）vs 4 Term control 供体（late_term），
每供体 × 细胞类型聚合原始 counts → CPM → log2，moderated t，BH 校正。

输出：results/dec_de_<celltype>.csv、dec_de_summary.md、dec_signatures.json
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

H5 = "data/pe_placenta/Decidua_atlas_raw_counts_20230711.h5ad"
MIN_CELLS = 50
MIN_DONORS = 3
TOPN = 200


def moderated_t(A, B):
    """limma 风格经验贝叶斯 moderated t（与 snRNA_pseudobulk_de.py 一致）。"""
    from scipy.special import digamma, polygamma
    n1, n2 = A.shape[0], B.shape[0]
    d_g = n1 + n2 - 2.0
    m1, m2 = A.mean(0), B.mean(0)
    s2 = ((A - m1) ** 2).sum(0) + ((B - m2) ** 2).sum(0)
    s2 = s2 / d_g
    lfc = m1 - m2
    ok = s2 > 0
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
    print("cells", obs.shape, "genes", len(genes), flush=True)

    keep_cond = ["eoPE", "Term controls"]
    sel = obs["condition"].isin(keep_cond).values
    cell_ct = obs["celltype_annotations"].astype(str).values
    cell_do = obs["donor_id"].astype(str).values
    cell_co = obs["condition"].astype(str).values

    grp_key = np.array([f"{cell_do[i]}|{cell_ct[i]}" for i in range(len(obs))])
    uniq, inv = np.unique(grp_key, return_inverse=True)
    ncell = np.bincount(inv, minlength=len(uniq))
    print("groups:", len(uniq), flush=True)

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
    pos = {g: i for i, g in enumerate(uniq)}
    rows = [pos[g] for g in meta["group"]]
    PB = sums[rows]

    tot = PB.sum(1, keepdims=True)
    cpm = PB / np.maximum(tot, 1) * 1e6
    lg = np.log2(cpm + 1)
    pd.DataFrame(lg, index=meta["group"], columns=genes).to_csv("results/dec_pseudobulk.csv")
    meta.to_csv("results/dec_pseudobulk_meta.csv", index=False)

    out = ["# 蜕膜 snRNA 供体级伪 bulk DE：eoPE vs 足月对照（4 vs 4 供体）", "",
           "数据源：Zenodo 8159511 `Decidua_atlas_raw_counts_20230711.h5ad`。",
           "eoPE 供体（late_preterm）：073? 否——100-d, 102-d, 274-d, 389-d, 419-d 中筛 4 个；",
           "Term control（late_term）：327-d, 328-d, 372-d, 073-d。",
           "供体 × 细胞类型聚合 → CPM → log2(CPM+1)，moderated t，BH 校正。", ""]
    sigs = {}
    rows_sum = []
    for ct, sub in meta.groupby("celltype"):
        a = sub[sub["condition"] == "eoPE"]
        b = sub[sub["condition"] == "Term controls"]
        if len(a) < MIN_DONORS or len(b) < MIN_DONORS:
            continue
        ia, ib = a.index.values, b.index.values
        A, B = lg[ia], lg[ib]
        lfc = A.mean(0) - B.mean(0)
        expressed = (np.vstack([A, B]) >= 1).mean(0) >= 0.25
        lfc_m, t_m, p_m, _ = moderated_t(A, B)
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
        df.to_csv(f"results/dec_de_{ct}.csv", index=False)
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

    pd.DataFrame(rows_sum).to_csv("results/dec_de_celltypes.csv", index=False)
    out += ["## 各细胞类型概览", "",
            "| 细胞类型 | eoPE供体 | 对照供体 | eoPE细胞 | 对照细胞 | up | down |",
            "|---|---|---|---|---|---|---|"]
    for r in rows_sum:
        out.append(f"| {r['celltype']} | {r['n_eoPE_donors']} | {r['n_ctrl_donors']} | "
                   f"{r['n_cells_eoPE']} | {r['n_cells_ctrl']} | {r['n_up']} | {r['n_dn']} |")

    open("results/dec_de_summary.md", "w", encoding="utf-8").write("\n".join(out))
    json.dump(sigs, open("results/dec_signatures.json", "w"), indent=1)
    print("\n".join(out[:30]))
    print("\nsaved: results/dec_de_summary.md, dec_signatures.json")


if __name__ == "__main__":
    main()
