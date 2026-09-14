#!/usr/bin/env python
"""切片内 niche 分层验证：虚拟蛋白是否按解卷积注释的细胞微环境分层。

动机：跨切片（A vs B）比较受测序深度 2.9x 差异与批次影响，已判定不可行（见结果报告 5.6）。
本脚本改为**切片内**比较：用 RDS 自带的 SPOTlight 解卷积比例把每个 spot 归到主导细胞类型
（niche），检验 35 个虚拟蛋白是否在 niche 间显著分层，并以同型对照（isotype）和不表达蛋白
（KRT5）作为阴性对照。

输出：
  results/niche_<sec>_kw.csv       每个蛋白的 Kruskal-Wallis 检验 + eta^2 效应量
  results/niche_<sec>_means.csv    每个蛋白 × niche 的均值（z-score 口径）
  results/niche_<sec>_auc.csv      每个蛋白在其"预期 niche"上的 AUC（特异性记分卡）
  results/niche_<sec>_heatmap.png  蛋白 × niche 热图
  results/niche_summary.md         汇总
"""
import os
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

TYPES = ["vVCT", "vSCT", "vEVT", "vHBC", "vFB1", "vMC", "vTcell", "vVEC", "vEB1"]

# 蛋白 -> 预期富集的 niche（基于生物学先验，独立于解卷积注释）
EXPECTED = {
    "EPCAM": ["vSCT", "vVCT"],          # 上皮/滋养层
    "CD68": ["vHBC"],                   # 巨噬细胞（Hofbauer cell）
    "CD163": ["vHBC"],
    "CD14": ["vHBC"],
    "ITGAM": ["vHBC"],
    "ITGAX": ["vHBC"],
    "FCGR3A": ["vHBC"],
    "PTPRC_1": ["vHBC", "vTcell"],      # CD45
    "PTPRC_2": ["vHBC", "vTcell"],
    "HLA_DRA": ["vHBC", "vTcell"],
    "CD3E": ["vTcell"],
    "CD4": ["vTcell"],
    "CD8A": ["vTcell"],
    "PECAM1": ["vVEC", "vEB1"],         # 内皮
    "ACTA2": ["vMC", "vFB1"],           # 平滑肌/肌成纤维
    "VIM": ["vFB1", "vMC"],
    "CD19": ["vTcell"],                 # B 细胞（胎盘极少，预期弱）
    "MS4A1": ["vTcell"],
    "PAX5": ["vTcell"],
    "CCR7": ["vTcell"],
    "CXCR5": ["vTcell"],
    "CD27": ["vTcell"],
    # 阴性对照
    "KRT5": None,                       # 胎盘不表达
    "mouse_IgG1k": None,
    "mouse_IgG2a": None,
    "mouse_IgG2bk": None,
    "rat_IgG2a": None,
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


def auc(a, b):
    """AUC = P(x_a > x_b)，使用 Mann-Whitney U"""
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return u / (len(a) * len(b)), p


def main():
    summary = []
    for sec in ["A", "B"]:
        h5 = f"data/pe_placenta/h5ad/section{sec}_RNA_raw.h5ad"
        pred_p = f"runs/pe_section{sec}_infer.pred.npy"
        prot_p = f"runs/pe_section{sec}_infer.protein_names.txt"
        spot_p = f"runs/pe_section{sec}_infer.spot_names.txt"
        meta_p = f"data/pe_placenta/export/section{sec}_metadata.csv"

        a = ad.read_h5ad(h5)
        obs = a.obs.copy()
        meta = pd.read_csv(meta_p, index_col="barcode")

        prots = [x.strip() for x in open(prot_p) if x.strip()]
        spots = [x.strip() for x in open(spot_p) if x.strip()]
        pred = np.load(pred_p)
        print(f"=== section{sec}: pred {pred.shape}, spots {len(spots)}, prots {len(prots)}")

        df = pd.DataFrame(pred, index=spots, columns=prots)
        # 对齐到 h5ad 的 obs 顺序
        common = [s for s in obs.index if s in df.index]
        df = df.loc[common]
        obs = obs.loc[common]
        meta = meta.reindex(common)

        # ---- niche 定义 ----
        # 各类型比例量纲差异大（vSCT 均值 0.30，vHBC 仅 0.013），直接 argmax 会塌缩到 vSCT。
        # 采用两种互补口径：
        #  (a) spot 级：各类型先做跨 spot 的 z-score，再取 argmax（相对富集口径）
        #  (b) cluster 级：对每个 Seurat cluster 计算 "富集比" = 该 cluster 内均值 / 全局均值，
        #     取 argmax，spot 继承其 cluster 的标签（去噪、空间连贯，作为主口径）
        props = obs[TYPES].astype(float).values
        props = np.nan_to_num(props, nan=0.0)
        obs["cluster"] = meta["seurat_clusters"].values

        # (a) spot-level z-argmax
        zt = (props - props.mean(0)) / (props.std(0) + 1e-9)
        obs["niche_spot"] = np.array(TYPES)[np.argmax(zt, axis=1)]

        # (b) cluster-level enrichment-ratio argmax
        cl = obs["cluster"].values
        global_mean = props.mean(0)
        uniq = pd.unique(cl[~pd.isna(cl)])
        clabel = {}
        for c in uniq:
            mk = (cl == c)
            prof = props[mk].mean(0)
            ratio = prof / (global_mean + 1e-9)
            j = int(np.argmax(ratio))
            # 富集比必须 > 1.2 才算明确的 niche，否则标记为 mixed
            clabel[c] = TYPES[j] if ratio[j] >= 1.2 else "mixed"
        obs["niche"] = [clabel.get(c, "mixed") for c in cl]

        for col, tag in [("niche_spot", "spot-z"), ("niche", "cluster-enrich")]:
            cnt = obs[col].value_counts()
            print(f"  [{tag}] {col} counts:\n{cnt.to_string()}")

        # 主口径：cluster 级；仅保留 spot 数 >= 20 的 niche（mixed 也排除）
        cnt = obs["niche"].value_counts()
        keep = [k for k in cnt.index if cnt[k] >= 20 and k != "mixed"]
        print(f"  keep: {keep}")
        m = np.isin(obs["niche"].values, keep)
        sub = df[m]
        nic = obs["niche"].values[m]

        # 1) Kruskal-Wallis + eta^2
        rows = []
        for p in prots:
            groups = [sub[p].values[nic == k] for k in keep]
            H, pv = stats.kruskal(*groups)
            n = len(sub)
            k = len(keep)
            eta2 = (H - k + 1) / (n - k) if n > k else np.nan
            rows.append({"protein": p, "H": H, "p": pv, "eta2": eta2})
        kw = pd.DataFrame(rows)
        kw["q"] = bh(kw["p"].values)
        kw = kw.sort_values("eta2", ascending=False)
        kw.to_csv(f"results/niche_{sec}_kw.csv", index=False)

        # 2) 每个蛋白 × niche 的 z-score 均值
        z = (sub - sub.mean()) / (sub.std(ddof=0) + 1e-9)
        z["niche"] = nic
        means = z.groupby("niche").mean().T
        means = means[keep]
        means.to_csv(f"results/niche_{sec}_means.csv")

        # 3) 预期 niche 的 AUC 记分卡
        auc_rows = []
        for p in prots:
            exp = EXPECTED.get(p, None)
            if not exp:
                continue
            exp = [e for e in exp if e in keep]
            if not exp:
                continue
            inx = sub[p].values[np.isin(nic, exp)]
            outx = sub[p].values[~np.isin(nic, exp)]
            a_, p_ = auc(inx, outx)
            auc_rows.append({"protein": p, "expected_niche": "|".join(exp),
                             "n_in": len(inx), "n_out": len(outx),
                             "auc": a_, "p": p_})
        aucdf = pd.DataFrame(auc_rows)
        aucdf["q"] = bh(aucdf["p"].values)
        aucdf["is_control"] = aucdf["protein"].isin(
            ["KRT5", "mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a"])
        aucdf = aucdf.sort_values(["is_control", "auc"], ascending=[True, False])
        aucdf.to_csv(f"results/niche_{sec}_auc.csv", index=False)

        # 4) 热图
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            order = kw["protein"].tolist()
            mat = means.loc[[p for p in order if p in means.index]]
            fig, ax = plt.subplots(figsize=(1.2 * len(keep) + 3, 0.28 * len(mat) + 2))
            im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
            ax.set_xticks(range(len(keep)))
            ax.set_xticklabels(keep, rotation=45, ha="right")
            ax.set_yticks(range(len(mat)))
            ax.set_yticklabels([p if p not in
                                ("mouse_IgG1k", "mouse_IgG2a", "mouse_IgG2bk", "rat_IgG2a")
                                else p + " *" for p in mat.index], fontsize=7)
            ax.set_title(f"section{sec}: virtual protein z-score by deconvolution niche")
            fig.colorbar(im, ax=ax, shrink=0.6)
            fig.tight_layout()
            fig.savefig(f"results/niche_{sec}_heatmap.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            print("  heatmap failed:", e)

        summary.append({"section": sec, "n_spots": len(sub), "niches": ",".join(keep),
                        "kw": kw, "auc": aucdf, "means": means})

    # ---- 汇总 markdown ----
    lines = ["# 切片内 niche 分层验证（虚拟蛋白 vs 解卷积注释）", "",
             "方法：用 RDS 自带 SPOTlight 解卷积的 9 类细胞比例把 spot 归到主导类型；",
             "对每个虚拟蛋白做 Kruskal-Wallis 检验（eta² 为效应量）并计算其“预期 niche”上的 AUC。",
             "同型对照（mouse/rat IgG）与胎盘不表达蛋白 KRT5 作为阴性对照。", ""]
    for s in summary:
        lines += [f"## section{s['section']}  (n={s['n_spots']} spots, niches: {s['niches']})", ""]
        lines += ["### Kruskal-Wallis 效应量 Top 15", "",
                  "| 蛋白 | H | p | q | eta² |", "|---|---|---|---|---|"]
        for _, r in s["kw"].head(15).iterrows():
            lines.append(f"| {r['protein']} | {r['H']:.1f} | {r['p']:.2e} | {r['q']:.2e} | {r['eta2']:.3f} |")
        lines += ["", "### 预期 niche 的 AUC 记分卡", "",
                  "| 蛋白 | 预期 niche | n_in | n_out | AUC | p | q | 对照 |", "|---|---|---|---|---|---|---|---|"]
        for _, r in s["auc"].iterrows():
            lines.append(f"| {r['protein']} | {r['expected_niche']} | {r['n_in']} | {r['n_out']} | "
                         f"{r['auc']:.3f} | {r['p']:.2e} | {r['q']:.2e} | {'是' if r['is_control'] else ''} |")
        lines.append("")
    open("results/niche_summary.md", "w", encoding="utf-8").write("\n".join(lines))
    print("wrote results/niche_summary.md")


if __name__ == "__main__":
    main()
