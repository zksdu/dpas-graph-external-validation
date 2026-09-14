#!/usr/bin/env python
"""切片内 niche 分层验证：虚拟蛋白是否按解卷积注释的细胞微环境分层。

动机：跨切片（A vs B）比较受测序深度 2.9x 差异与批次影响，已判定不可行（结果报告 5.6）。
本脚本改为**切片内**比较：用 RDS 自带 SPOTlight 解卷积把每个 Seurat cluster 标注为细胞微环境
（niche），检验 35 个虚拟蛋白是否在 niche 间显著分层。

关键对照设计：
  * 同型对照（mouse_IgG1k / mouse_IgG2a / mouse_IgG2bk / rat_IgG2a）+ 胎盘不表达的 KRT5
    作为**阴性对照**。若这些蛋白也出现同等强度的 niche 分层，说明分层主要来自技术性混杂。
  * 同时给出**深度校正**（对每个蛋白在切片内回归 log1p(nCount_Spatial) 取残差）后的结果。
  * 记分卡：对每一个 niche 计算全部 35 个蛋白的 AUC，用同型对照在该 niche 上的 AUC 分布
    作为零分布，计算真实蛋白的特异性 z 分数。

输出（results/）：
  niche2_<sec>_kw_raw.csv / _kw_adj.csv      Kruskal-Wallis + eta²
  niche2_<sec>_aucmat_raw.csv / _adj.csv     蛋白 × niche 的 AUC 矩阵
  niche2_<sec>_scorecard.csv                 预期 niche 的 AUC + 相对同型对照的特异性 z
  niche2_<sec>_heatmap_adj.png               深度校正后的蛋白 × niche z 热图
  niche2_summary.md                          汇总（含诚实结论）
"""
import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

TYPES = ["vVCT", "vSCT", "vEVT", "vHBC", "vFB1", "vMC", "vTcell", "vVEC", "vEB1"]
ISOTYPE = ["mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"]
NEGCTRL = ISOTYPE + ["KRT5"]

# 蛋白 -> 预期富集的 niche（生物学先验，独立于解卷积注释）
EXPECTED = {
    "EPCAM": ["vSCT", "vVCT"],
    "CD68": ["vHBC"], "CD163": ["vHBC"], "CD14": ["vHBC"],
    "ITGAM": ["vHBC"], "ITGAX": ["vHBC"], "FCGR3A": ["vHBC"],
    "PTPRC_1": ["vHBC", "vTcell"], "PTPRC_2": ["vHBC", "vTcell"],
    "HLA_DRA": ["vHBC", "vTcell"],
    "CD3E": ["vTcell"], "CD4": ["vTcell"], "CD8A": ["vTcell"],
    "PECAM1": ["vVEC", "vEB1"],
    "ACTA2": ["vMC", "vFB1"], "VIM": ["vFB1", "vMC"],
    "CD19": ["vTcell"], "MS4A1": ["vTcell"], "PAX5": ["vTcell"],
    "CCR7": ["vTcell"], "CXCR5": ["vTcell"], "CD27": ["vTcell"],
}


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1))
        q[o[i]] = prev
    return q


def auc_vec(x, mask_in):
    """AUC = P(x_in > x_out)，0.5 表示无区分"""
    a = x[mask_in]
    b = x[~mask_in]
    if len(a) < 10 or len(b) < 10:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return u / (len(a) * len(b)), p


def residualize(mat, cov):
    """对每一列按 cov（含截距）做最小二乘回归，返回残差"""
    X = np.column_stack([np.ones(len(cov)), cov, cov ** 2])
    beta, *_ = np.linalg.lstsq(X, mat, rcond=None)
    return mat - X @ beta


def load(sec):
    a = ad.read_h5ad(f"data/pe_placenta/h5ad/section{sec}_RNA_raw.h5ad")
    obs = a.obs.copy()
    meta = pd.read_csv(f"data/pe_placenta/export/section{sec}_metadata.csv", index_col="barcode")
    prots = [x.strip() for x in open(f"runs/pe_section{sec}_infer.protein_names.txt") if x.strip()]
    spots = [x.strip() for x in open(f"runs/pe_section{sec}_infer.spot_names.txt") if x.strip()]
    pred = pd.DataFrame(np.load(f"runs/pe_section{sec}_infer.pred.npy"),
                        index=spots, columns=prots)
    common = [s for s in obs.index if s in pred.index]
    obs, pred, meta = obs.loc[common], pred.loc[common], meta.reindex(common)

    props = np.nan_to_num(obs[TYPES].astype(float).values, nan=0.0)
    obs["cluster"] = meta["seurat_clusters"].values
    obs["depth"] = meta["nCount_Spatial"].values.astype(float)

    # cluster 级富集比标注 niche（量纲差异大，直接 argmax 会塌缩到 vSCT）
    cl = obs["cluster"].values
    gmean = props.mean(0)
    uniq = pd.unique(cl[~pd.isna(cl)])
    lab = {}
    for c in uniq:
        ratio = props[cl == c].mean(0) / (gmean + 1e-9)
        j = int(np.argmax(ratio))
        lab[c] = TYPES[j] if ratio[j] >= 1.2 else "mixed"
    obs["niche"] = [lab.get(c, "mixed") for c in cl]
    return obs, pred, prots


def analyze(obs, pred, prots, tag, tag2):
    cnt = obs["niche"].value_counts()
    keep = [k for k in cnt.index if cnt[k] >= 20 and k != "mixed"]
    m = np.isin(obs["niche"].values, keep)
    sub = pred[m]
    nic = obs["niche"].values[m]
    n = len(sub)

    # 1) Kruskal-Wallis + eta²
    rows = []
    for p in prots:
        groups = [sub[p].values[nic == k] for k in keep]
        H, pv = stats.kruskal(*groups)
        k = len(keep)
        rows.append({"protein": p, "H": H, "p": pv,
                     "eta2": (H - k + 1) / (n - k) if n > k else np.nan,
                     "is_negctrl": p in NEGCTRL})
    kw = pd.DataFrame(rows)
    kw["q"] = bh(kw["p"].values)
    kw.to_csv(f"results/niche2_{tag}_kw_{tag2}.csv", index=False)

    # 2) 全部蛋白 × niche 的 AUC 矩阵
    mat = pd.DataFrame(index=prots, columns=keep, dtype=float)
    pmat = mat.copy()
    for k in keep:
        msk = nic == k
        for p in prots:
            a_, p_ = auc_vec(sub[p].values, msk)
            mat.loc[p, k] = a_
            pmat.loc[p, k] = p_
    mat.to_csv(f"results/niche2_{tag}_aucmat_{tag2}.csv")
    return kw, mat, keep, nic, sub


def main():
    out = ["# 切片内 niche 分层验证 v2（含同型对照与深度校正）", "",
           "**设计**：Seurat cluster 按 SPOTlight 解卷积富集比标注为细胞微环境（niche）；",
           "对每个虚拟蛋白做 Kruskal-Wallis（eta² 效应量），并计算蛋白 × niche 的 AUC 矩阵。",
           "**阴性对照**：4 个同型对照抗体（mouse/rat IgG）+ 胎盘不表达的 KRT5。",
           "**深度校正**：每个蛋白在切片内对 log1p(nCount_Spatial) 做二次回归取残差。", ""]

    store = {}
    for sec in ["A", "B"]:
        obs, pred, prots = load(sec)
        cnt = obs["niche"].value_counts()
        out += [f"## section{sec}", "",
                "niche spot 数：" + "，".join(f"{k}={v}" for k, v in cnt.items()), ""]

        for tag2, adj in [("raw", False), ("adj", True)]:
            p = pred.copy()
            if adj:
                cov = np.log1p(obs["depth"].values)
                p = pd.DataFrame(residualize(pred.values, cov), index=pred.index, columns=prots)
            kw, mat, keep, nic, sub = analyze(obs, p, prots, f"section{sec}", tag2)

            # 3) 记分卡：预期 niche 的 AUC + 相对同型对照零分布的特异性 z
            rows = []
            for pr, exp in EXPECTED.items():
                e = [x for x in exp if x in keep]
                if not e:
                    continue
                a_, p_ = auc_vec(sub[pr].values, np.isin(nic, e))
                iso = mat.loc[[x for x in ISOTYPE if x in mat.index], e].mean(axis=1).values
                iso = iso[~np.isnan(iso)]
                z = (a_ - iso.mean()) / (iso.std(ddof=0) + 1e-9) if len(iso) > 1 else np.nan
                rows.append({"protein": pr, "expected_niche": "|".join(e),
                             "auc": a_, "p": p_,
                             "isotype_auc_mean": iso.mean() if len(iso) else np.nan,
                             "specificity_z": z})
            sc = pd.DataFrame(rows)
            sc["q"] = bh(sc["p"].values)
            sc = sc.sort_values("auc", ascending=False)
            sc.to_csv(f"results/niche2_section{sec}_scorecard_{tag2}.csv", index=False)

            # 阴性对照自身的 eta²（判断是否非特异性）
            neg = kw[kw["is_negctrl"]]
            real = kw[~kw["is_negctrl"]]
            out += [f"### {'深度校正后' if adj else '原始'}口径", "",
                    f"- Kruskal-Wallis eta²：真实蛋白中位数 **{real['eta2'].median():.3f}**，"
                    f"阴性对照中位数 **{neg['eta2'].median():.3f}**"
                    f"（{'⚠ 对照同样分层，提示技术性混杂' if neg['eta2'].median() > 0.5 * real['eta2'].median() else '✅ 对照分层明显弱于真实蛋白'}）",
                    f"- 显著蛋白数（q<0.05）：{int((kw['q'] < 0.05).sum())} / {len(kw)}", "",
                    "| 蛋白 | 预期 niche | AUC | p | q | 同型对照AUC均值 | 特异性 z |",
                    "|---|---|---|---|---|---|---|"]
            for _, r in sc.iterrows():
                out.append(f"| {r['protein']} | {r['expected_niche']} | {r['auc']:.3f} | "
                           f"{r['p']:.2e} | {r['q']:.2e} | {r['isotype_auc_mean']:.3f} | "
                           f"{r['specificity_z']:+.2f} |")
            out.append("")
            store[(sec, tag2)] = (kw, mat, keep, nic, sub)

    # 热图（深度校正口径）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for sec in ["A", "B"]:
            kw, mat, keep, nic, sub = store[(sec, "adj")]
            z = (sub - sub.mean()) / (sub.std(ddof=0) + 1e-9)
            z["niche"] = nic
            means = z.groupby("niche").mean().T[keep]
            order = kw.sort_values("eta2", ascending=False)["protein"].tolist()
            means = means.reindex([x for x in order if x in means.index])
            fig, ax = plt.subplots(figsize=(3.55, 0.09 * len(means) + 1.6))
            im = ax.imshow(means.values, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
            ax.set_xticks(range(len(keep)))
            ax.set_xticklabels(keep, rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(len(means)))
            ax.set_yticklabels([x + (" *" if x in NEGCTRL else "") for x in means.index], fontsize=5.5)
            ax.set_title(f"section{sec}: virtual protein z by niche\n(depth-adjusted); * = negative control", fontsize=7.5)
            cb = fig.colorbar(im, ax=ax, shrink=0.55)
            cb.ax.tick_params(labelsize=5.5)
            fig.tight_layout()
            fig.savefig(f"results/niche2_section{sec}_heatmap_adj.png", dpi=600)
            plt.close(fig)
    except Exception as e:
        out.append(f"热图生成失败: {e}")

    open("results/niche2_summary.md", "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out[:40]))
    print("...\nwrote results/niche2_summary.md")


if __name__ == "__main__":
    main()
