#!/usr/bin/env python
"""双中心化（double-centering）的 niche 特异性分析。

为什么需要双中心化：
  1) 每个 niche 存在一个"全局偏移"——校正深度后，vHBC niche 上几乎所有真实蛋白的 AUC 都
     偏高（~0.7-0.8），而同型对照偏低（~0.33），说明还有未被深度解释的非特异性因子。
  2) 每个蛋白也存在自身的主效应（有的蛋白整体 AUC 偏高）。
因此用双中心化同时移除行（蛋白）与列（niche）的全局效应：
    D'(p,k) = AUC(p,k) − mean_k AUC(p,·) − mean_p AUC(·,k) + mean AUC
这样 D' 衡量的是"蛋白 p 在 niche k 上的**特异性**富集"，同型对照行作为经验零分布。

输出：results/specificity2_*.csv / .png / .md
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ISOTYPE = ["mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"]
EXPECTED = {
    "EPCAM": ["vSCT", "vVCT"],
    "CD68": ["vHBC"], "CD163": ["vHBC"], "CD14": ["vHBC"],
    "ITGAM": ["vHBC"], "ITGAX": ["vHBC"], "FCGR3A": ["vHBC"],
    "PTPRC_1": ["vHBC", "vTcell"], "PTPRC_2": ["vHBC", "vTcell"],
    "HLA_DRA": ["vHBC", "vTcell"],
    "CD3E": ["vTcell"], "CD4": ["vTcell"], "CD8A": ["vTcell"],
    "PECAM1": ["vVEC", "vEB1", "vMC"],
    "ACTA2": ["vMC", "vFB1"], "VIM": ["vFB1", "vMC"],
    "CD19": ["vTcell"], "MS4A1": ["vTcell"], "PAX5": ["vTcell"],
    "CCR7": ["vTcell"], "CXCR5": ["vTcell"], "CD27": ["vTcell"],
}


def dcenter(m):
    a = m.values.astype(float)
    return pd.DataFrame(a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + a.mean(),
                        index=m.index, columns=m.columns)


def main():
    D = {}
    for sec in ["A", "B"]:
        for tag in ["raw", "adj"]:
            m = pd.read_csv(f"results/niche2_section{sec}_aucmat_{tag}.csv", index_col=0)
            D[(sec, tag)] = dcenter(m)
    common = [c for c in D[("A", "adj")].columns if c in D[("B", "adj")].columns]

    # 经验零分布：同型对照行的 |D'|
    null = np.concatenate([D[(s, "adj")].loc[ISOTYPE, common].values.ravel() for s in ["A", "B"]])
    null = null[~np.isnan(null)]
    print(f"isotype null |D'|: n={len(null)}, mean={np.abs(null).mean():.3f}, "
          f"p95={np.percentile(np.abs(null), 95):.3f}")

    rows = []
    for p in D[("A", "adj")].index:
        if p in ISOTYPE:
            continue
        val = {}
        for k in common:
            val[k] = np.nanmean([D[("A", "adj")].loc[p, k], D[("B", "adj")].loc[p, k]])
        best = max(val, key=lambda k: val[k])
        v = val[best]
        # 相对同型对照零分布的 z
        z = (v - null.mean()) / (null.std(ddof=0) + 1e-9)
        exp = EXPECTED.get(p, [])
        rows.append({"protein": p, "best_niche": best, "D": v, "z_vs_isotype": z,
                     "D_A": D[("A", "adj")].loc[p, best], "D_B": D[("B", "adj")].loc[p, best],
                     "expected": "|".join(exp),
                     "match": (best in exp) if exp else None,
                     **{f"d_{k}": round(val[k], 3) for k in common}})
    df = pd.DataFrame(rows).sort_values("D", ascending=False)
    df.to_csv("results/specificity2_combined.csv", index=False)

    # 按 niche 列出 Top 5 蛋白
    lines = ["# 双中心化 niche 特异性（移除蛋白主效应与 niche 全局偏移）", "",
             "D′ = AUC(蛋白, niche) − 行均值 − 列均值 + 总均值（深度校正口径，两切片平均）。",
             f"同型对照经验零分布：n={len(null)}，mean|D′|={np.abs(null).mean():.3f}，"
             f"95 分位 |D′|={np.percentile(np.abs(null), 95):.3f}。", "",
             "## 每个 niche 的 Top 5 特异蛋白", ""]
    for k in common:
        sub = df[[f"d_{k}", "protein"]].sort_values(f"d_{k}", ascending=False).head(5)
        lines.append(f"- **{k}**：" + "，".join(f"{r.protein} ({getattr(r, f'd_{k}'):+.3f})"
                                               for r in sub.itertuples()))
    lines += ["", "## 全表（按跨切片平均 D′ 降序）", "",
              "| 蛋白 | 最佳 niche | D′ | z(对照) | D′_A | D′_B | 先验 | 一致 |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['protein']} | {r['best_niche']} | {r['D']:+.3f} | {r['z_vs_isotype']:+.2f} | "
                     f"{r['D_A']:+.3f} | {r['D_B']:+.3f} | {r['expected']} | "
                     f"{'✅' if r['match'] is True else ('❌' if r['match'] is False else '-')} |")

    # 一致性统计
    ok = df[df["match"].notna()]
    nacc = int((ok["match"] == True).sum())
    lines += ["", f"## 与生物学先验的一致性", "",
              f"有先验的蛋白 {len(ok)} 个，其中最佳 niche 命中先验 **{nacc} / {len(ok)}**。",
              "（随机期望约 1/7 ≈ 14%）"]
    lines.append(f"二项检验 p = {stats.binomtest(nacc, len(ok), 1 / 7, alternative='greater').pvalue:.4f}")

    open("results/specificity2_summary.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
