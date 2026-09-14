#!/usr/bin/env python
"""以同型对照为基线，评估虚拟蛋白 niche 特异性的跨切片可重复性。

背景：niche2 分析显示，4 个同型对照抗体（mouse/rat IgG）与真实蛋白一样在 niche 间显著分层
（eta² 相当），说明虚拟蛋白的 niche 分层含有大量**非特异性**成分。因此不能以 Kruskal-Wallis
的显著性作为证据，而必须以同型对照为基线做**相对**比较：
    ΔAUC(蛋白, niche) = AUC(蛋白, niche) − mean_isotype AUC(niche)
只有在两张切片上 Δ 都明显为正的蛋白-niche 配对，才认为具备可重复的特异性。

输出：
  results/specificity_combined.csv
  results/specificity_combined.png
  results/specificity_summary.md
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ISOTYPE = ["mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"]
NEGCTRL = ISOTYPE + ["KRT5"]

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

TAG = "adj"   # 深度校正口径（主分析）


def main():
    mats, bases = {}, {}
    for sec in ["A", "B"]:
        m = pd.read_csv(f"results/niche2_section{sec}_aucmat_{TAG}.csv", index_col=0)
        mats[sec] = m
        bases[sec] = m.loc[[x for x in ISOTYPE if x in m.index]].mean(axis=0)
    common_niche = [c for c in mats["A"].columns if c in mats["B"].columns]

    rows = []
    prots = [p for p in mats["A"].index if p not in NEGCTRL]
    for p in prots:
        d = {}
        for sec in ["A", "B"]:
            for k in mats[sec].columns:
                d[(sec, k)] = mats[sec].loc[p, k] - bases[sec][k]
        # 跨切片共享 niche 的平均 Δ
        mean_d = {k: np.mean([d[("A", k)], d[("B", k)]]) for k in common_niche}
        best = max(mean_d, key=mean_d.get)
        exp = EXPECTED.get(p, [])
        rows.append({
            "protein": p,
            "best_niche": best,
            "delta_best_mean": mean_d[best],
            "delta_A": d[("A", best)],
            "delta_B": d[("B", best)],
            "expected": "|".join(exp) if exp else "",
            "match_expect": best in exp if exp else "",
            "delta_expected_mean": np.mean([np.mean([d[(s, k)] for s in ["A", "B"]])
                                            for k in exp if k in common_niche]) if exp else np.nan,
            **{f"d_{k}": round(mean_d[k], 3) for k in common_niche},
        })
    df = pd.DataFrame(rows).sort_values("delta_best_mean", ascending=False)
    df.to_csv("results/specificity_combined.csv", index=False)

    # 可重复（两张切片 Δ 均 > 0.1）
    rep = df[(df["delta_A"] > 0.10) & (df["delta_B"] > 0.10)].copy()
    rep.to_csv("results/specificity_reproducible.csv", index=False)

    # 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, sec in zip(axes, ["A", "B"]):
            sub = df.head(14)
            xs = np.arange(len(sub))
            w = 0.35
            ax.bar(xs - w / 2, sub["delta_A"] if sec == "A" else sub["delta_B"], w,
                   label=f"section{sec}", color="#c0392b" if sec == "A" else "#2471a3")
            ax.bar(xs + w / 2, sub["delta_B"] if sec == "A" else sub["delta_A"], w,
                   label="other section", color="#95a5a6", alpha=0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{r.protein}\n({r.best_niche})" for r in sub.itertuples()],
                               rotation=60, ha="right", fontsize=7)
            ax.axhline(0.1, ls="--", c="k", lw=0.8)
            ax.axhline(0, c="k", lw=0.8)
            ax.set_ylabel("ΔAUC vs isotype baseline")
            ax.set_title(f"Top by mean Δ (section{sec} bars)")
            ax.legend(fontsize=8)
        fig.suptitle("Virtual protein niche specificity relative to isotype controls (depth-adjusted)")
        fig.tight_layout()
        fig.savefig("results/specificity_combined.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print("plot failed:", e)

    lines = ["# 虚拟蛋白 niche 特异性：以同型对照为基线的跨切片可重复性", "",
             "ΔAUC = AUC(蛋白, 该 niche vs 其余) − 同型对照在该 niche 的 AUC 均值（4 个 IgG 同型）。",
             "同型对照与真实蛋白在 niche 间的 eta² 相当（A: 0.101 vs 0.105；B: 0.081 vs 0.083，",
             "深度校正后），故只有相对同型对照的**增量**才是特异信号。", "",
             f"跨切片共享 niche：{', '.join(common_niche)}", "",
             "## 可重复的特异性配对（两切片 Δ 均 > 0.10）", "",
             "| 蛋白 | 最佳 niche | Δ_A | Δ_B | 均值 | 是否符合先验 |", "|---|---|---|---|---|---|"]
    for _, r in rep.iterrows():
        lines.append(f"| {r['protein']} | {r['best_niche']} | {r['delta_A']:+.3f} | "
                     f"{r['delta_B']:+.3f} | {r['delta_best_mean']:+.3f} | "
                     f"{'✅' if r['match_expect'] is True else ('❌' if r['match_expect'] is False else '-')} |")
    if len(rep) == 0:
        lines.append("| （无） | | | | | |")
    lines += ["", "## 全部蛋白 Top 20（按跨切片平均 Δ）", "",
              "| 蛋白 | 最佳 niche | 均值 | Δ_A | Δ_B | 先验期望 |", "|---|---|---|---|---|---|"]
    for _, r in df.head(20).iterrows():
        lines.append(f"| {r['protein']} | {r['best_niche']} | {r['delta_best_mean']:+.3f} | "
                     f"{r['delta_A']:+.3f} | {r['delta_B']:+.3f} | {r['expected']} |")
    lines += ["", "## 阴性对照自检（Δ 应接近 0）", ""]
    for sec in ["A", "B"]:
        for p in NEGCTRL:
            if p in mats[sec].index:
                k = (mats[sec].loc[p] - bases[sec]).abs().idxmax()
                lines.append(f"- section{sec} {p}: 最大 |Δ| = "
                             f"{(mats[sec].loc[p] - bases[sec]).abs().max():.3f} @ {k}")
    open("results/specificity_summary.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
