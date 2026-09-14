#!/usr/bin/env python
"""全 35 个虚拟蛋白 × 6 个 RNA 细胞类型签名的相关矩阵（含同型对照基线）。

与 niche 分析（niche2）互为补充：niche 用解卷积注释做分层，这里用独立的 RNA 签名做连续相关。
核心仍是**同型对照**：只有显著高于同型对照 |rho| 的相关才被视为特异。

输出：
  results/signature_corr_all.csv
  results/signature_corr_summary.md
"""
import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ISOTYPE = ["mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"]

SIGNATURES = {
    "SCT": ["CGB3", "CGB5", "CGB8", "CSH1", "CSH2", "CSHL1", "ERVW-1", "ERVFRD-1",
            "PSG4", "PSG6", "PSG11", "GH2"],
    "VCT": ["TP63", "KRT5", "KRT14", "ITGA6", "EGFR"],
    "EVT": ["HLAG", "ITGA5", "ITGAV", "MMP2", "MMP9", "FN1", "TIMP3"],
    "HBC": ["CD68", "LYZ", "AIF1", "C1QA", "C1QB", "C1QC", "FCGR3A", "MSR1"],
    "Endo": ["PECAM1", "VWF", "CLDN5", "CDH5", "KDR", "SOX17"],
    "Fibro": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
}

# 先验：蛋白 → 应与之正相关的签名
PRIOR = {
    "EPCAM": ["SCT", "VCT"],
    "CD68": ["HBC"], "CD163": ["HBC"], "CD14": ["HBC"],
    "ITGAM": ["HBC"], "ITGAX": ["HBC"], "FCGR3A": ["HBC"],
    "PTPRC_1": ["HBC"], "PTPRC_2": ["HBC"], "HLA_DRA": ["HBC"],
    "CD3E": ["HBC"], "CD4": ["HBC"], "CD8A": ["HBC"],
    "PECAM1": ["Endo"],
    "ACTA2": ["Fibro"], "VIM": ["Fibro"],
}


def main():
    rows = []
    for sec in ["A", "B"]:
        a = ad.read_h5ad(f"data/pe_placenta/h5ad/section{sec}_RNA_raw.h5ad")
        genes = list(a.var_names)
        X = a.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        depth = np.asarray(X.sum(1)).ravel()
        cp10k = X / np.maximum(depth, 1)[:, None] * 1e4
        lg = np.log1p(cp10k)

        meta = pd.read_csv(f"data/pe_placenta/export/section{sec}_metadata.csv", index_col="barcode")
        prots = [x.strip() for x in open(f"runs/pe_section{sec}_infer.protein_names.txt") if x.strip()]
        spots = [x.strip() for x in open(f"runs/pe_section{sec}_infer.spot_names.txt") if x.strip()]
        P = pd.DataFrame(np.load(f"runs/pe_section{sec}_infer.pred.npy"), index=spots, columns=prots)
        common = [s for s in a.obs_names if s in P.index]
        P = P.loc[common]
        gi = {s: i for i, s in enumerate(a.obs_names)}
        idx = np.array([gi[s] for s in common])

        for sig, gl in SIGNATURES.items():
            gset = [g for g in gl if g in genes]
            if len(gset) < 3:
                continue
            cols = [genes.index(g) for g in gset]
            sc = lg[idx][:, cols].mean(1)

            for p in prots:
                r, pv = stats.spearmanr(P[p].values, sc)
                # 深度校正：蛋白与签名都对 log depth 做二次回归取残差
                cov = np.log1p(depth[idx])
                Xd = np.column_stack([np.ones(len(cov)), cov, cov ** 2])
                res_p = P[p].values - Xd @ np.linalg.lstsq(Xd, P[p].values, rcond=None)[0]
                res_s = sc - Xd @ np.linalg.lstsq(Xd, sc, rcond=None)[0]
                ra, pa = stats.spearmanr(res_p, res_s)
                rows.append({"section": sec, "protein": p, "signature": sig,
                             "n_genes": len(gset), "rho": r, "p": pv,
                             "rho_adj": ra, "p_adj": pa,
                             "is_isotype": p in ISOTYPE})
    df = pd.DataFrame(rows)
    df.to_csv("results/signature_corr_all.csv", index=False)

    # 每个切片的同型对照 |rho| 基线（全部 6 个签名）
    base = {}
    for sec in ["A", "B"]:
        for tag in ["rho", "rho_adj"]:
            v = df[(df.section == sec) & (df.is_isotype)][tag].abs()
            base[(sec, tag)] = (v.mean(), v.quantile(0.95))

    lines = ["# 全 35 蛋白 × RNA 细胞类型签名相关（同型对照基线）", "",
             "rho = Spearman(虚拟蛋白, RNA 签名分)；rho_adj = 两者各自对 log1p(depth) 做二次回归取残差后再相关。",
             "同型对照（4 个 IgG）的 |rho| 分布作为非特异性基线。", "",
             "## 同型对照基线", "",
             "| 切片 | 口径 | mean\\|rho\\| | p95\\|rho\\| |", "|---|---|---|---|"]
    for sec in ["A", "B"]:
        for tag, lab in [("rho", "原始"), ("rho_adj", "深度校正")]:
            m, q = base[(sec, tag)]
            lines.append(f"| section{sec} | {lab} | {m:.3f} | {q:.3f} |")

    lines += ["", "## 先验配对的相关（两切片一致性）", "",
              "| 蛋白 | 签名 | rho_A | rho_A(adj) | rho_B | rho_B(adj) | 同向？ | 超过对照p95？ |",
              "|---|---|---|---|---|---|---|---|"]
    for p, sigs in PRIOR.items():
        for sig in sigs:
            r = {}
            for sec in ["A", "B"]:
                for tag in ["rho", "rho_adj"]:
                    sub = df[(df.section == sec) & (df.protein == p) & (df.signature == sig)]
                    r[(sec, tag)] = float(sub[tag].iloc[0]) if len(sub) else np.nan
            same = np.sign(r[("A", "rho")]) == np.sign(r[("B", "rho")])
            over = (abs(r[("A", "rho_adj")]) > base[("A", "rho_adj")][1]) and \
                   (abs(r[("B", "rho_adj")]) > base[("B", "rho_adj")][1])
            lines.append(f"| {p} | {sig} | {r[('A','rho')]:+.3f} | {r[('A','rho_adj')]:+.3f} | "
                         f"{r[('B','rho')]:+.3f} | {r[('B','rho_adj')]:+.3f} | "
                         f"{'✅' if same else '❌'} | {'✅' if over else '❌'} |")

    lines += ["", "## 每个签名上 |rho| 最高的 5 个蛋白（深度校正口径）", ""]
    for sec in ["A", "B"]:
        for sig in SIGNATURES:
            sub = df[(df.section == sec) & (df.signature == sig) & (~df.is_isotype)]
            top = sub.reindex(sub["rho_adj"].abs().sort_values(ascending=False).index).head(5)
            lines.append(f"- **section{sec} / {sig}**：" +
                         "，".join(f"{r.protein} ({r.rho_adj:+.3f})" for r in top.itertuples()))

    open("results/signature_corr_summary.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
