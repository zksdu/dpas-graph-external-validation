#!/usr/bin/env python
"""用 snRNA 得到的 eoPE 签名给两张 Visium 切片打分，数据驱动判定切片身份。

思路：
  1. 从 snRNA 伪 bulk DE（6 eoPE 供体 vs 6 足月对照供体）取每个细胞类型的
     up / down 基因（p<0.05 且 |log2FC|>1，按 |t| 取前 50）。
  2. 在每张 Visium 切片内把表达做 CP10K → log1p → **切片内 z-score**，
     再取 score = mean(z_up) − mean(z_dn)。
     切片内 z-score + up/dn 相减，使该分数对测序深度与平台差异基本不敏感，
     从而可以跨切片比较（这是上一轮"绝对量跨切片比较不可行"的规避方案）。
  3. 比较 A / B 两切片的 score 分布（Mann-Whitney + AUC + Cliff's delta）。

输出：
  results/visium_eope_score.csv    每个 spot 的分数
  results/visium_eope_score.png    空间图 + 分布
  results/visium_eope_score.md     汇总
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# 用哪些细胞类型的签名（绒毛主体 + 免疫）
USE_CT = ["vSTB1", "vSTB2", "vSTBjuv", "vCTB", "vHBC", "vVEC"]
TOPN = 50


def load_sig():
    up, dn = [], []
    detail = []
    for ct in USE_CT:
        f = f"results/sn_de_{ct}.csv"
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d = d[d["expressed"] == True] if "expressed" in d.columns else d
        d = d.dropna(subset=["p"])
        u = d[(d["p"] < 0.05) & (d["log2FC"] > 1)].sort_values("t", ascending=False).head(TOPN)
        w = d[(d["p"] < 0.05) & (d["log2FC"] < -1)].sort_values("t").head(TOPN)
        up += u["gene"].tolist()
        dn += w["gene"].tolist()
        detail.append((ct, len(u), len(w)))
    return sorted(set(up)), sorted(set(dn)), detail


def main():
    up, dn, detail = load_sig()
    print("signature sizes:", detail)
    print(f"union up={len(up)} dn={len(dn)}")

    out = ["# 用 snRNA eoPE 签名给 Visium 切片打分（数据驱动的切片身份判定）", "",
           "## 方法", "",
           "1. snRNA 伪 bulk DE：eoPE（6 供体）vs 足月对照（6 供体），每细胞类型取",
           "   p<0.05 且 |log2FC|>1 的前 50 个 up / down 基因。",
           f"   使用细胞类型：{', '.join(USE_CT)}",
           f"2. Visium：CP10K → log1p → **切片内 z-score**，score = mean(z_up) − mean(z_dn)。",
           "   切片内 z-score + up/dn 相减 ⇒ 对测序深度与平台差异基本不敏感，可跨切片比较。",
           "", "各细胞类型贡献的基因数：" +
           "，".join(f"{ct}(up {a}/dn {b})" for ct, a, b in detail), ""]

    # ---- 合并签名打分 ----
    res = {}
    for sec in ["A", "B"]:
        a = ad.read_h5ad(f"data/pe_placenta/h5ad/section{sec}_RNA_raw.h5ad")
        genes = list(a.var_names)
        X = a.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        depth = np.asarray(X.sum(1)).ravel()
        lg = np.log1p(X / np.maximum(depth, 1)[:, None] * 1e4)
        z = (lg - lg.mean(0)) / (lg.std(0) + 1e-9)
        gi = {g: i for i, g in enumerate(genes)}
        uidx = [gi[g] for g in up if g in gi]
        didx = [gi[g] for g in dn if g in gi]
        score = z[:, uidx].mean(1) - z[:, didx].mean(1)
        print(f"  section{sec}: up genes found {len(uidx)}/{len(up)}, dn {len(didx)}/{len(dn)}")
        df = pd.DataFrame({"barcode": a.obs_names, "score": score,
                           "depth": depth, "section": sec})
        df["x"] = a.obsm["spatial"][:, 0]
        df["y"] = a.obsm["spatial"][:, 1]
        res[sec] = df

    # ---- 逐细胞类型的签名分别打分（更敏感）----
    out += ["## 逐细胞类型签名的 A vs B 比较（深度残差后）", "",
            "| 签名来源细胞类型 | up/dn 基因数 | AUC(A>B) | Cliff's δ | p | 更高的一侧 |",
            "|---|---|---|---|---|---|"]
    per_ct_rows = []
    for ct in USE_CT:
        f = f"results/sn_de_{ct}.csv"
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        if "expressed" in d.columns:
            d = d[d["expressed"] == True]
        d = d.dropna(subset=["p"])
        u = d[(d["p"] < 0.05) & (d["log2FC"] > 1)].sort_values("t", ascending=False).head(TOPN)
        w = d[(d["p"] < 0.05) & (d["log2FC"] < -1)].sort_values("t").head(TOPN)
        s_up, s_dn = u["gene"].tolist(), w["gene"].tolist()
        vals = {}
        for sec in ["A", "B"]:
            a = ad.read_h5ad(f"data/pe_placenta/h5ad/section{sec}_RNA_raw.h5ad")
            genes = list(a.var_names)
            X = a.X
            X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            depth = np.asarray(X.sum(1)).ravel()
            lg = np.log1p(X / np.maximum(depth, 1)[:, None] * 1e4)
            z = (lg - lg.mean(0)) / (lg.std(0) + 1e-9)
            gi = {g: i for i, g in enumerate(genes)}
            ui = [gi[g] for g in s_up if g in gi]
            di = [gi[g] for g in s_dn if g in gi]
            if not ui or not di:
                vals[sec] = None
                continue
            sc = z[:, ui].mean(1) - z[:, di].mean(1)
            # 深度残差
            cov = np.log1p(depth)
            Xd = np.column_stack([np.ones(len(cov)), cov, cov ** 2])
            sc = sc - Xd @ np.linalg.lstsq(Xd, sc, rcond=None)[0]
            vals[sec] = sc
            res[sec][f"score_{ct}"] = sc
        if vals.get("A") is None or vals.get("B") is None:
            continue
        uu, pp = stats.mannwhitneyu(vals["A"], vals["B"], alternative="two-sided")
        a_ = uu / (len(vals["A"]) * len(vals["B"]))
        per_ct_rows.append({"celltype": ct, "n_up": len(s_up), "n_dn": len(s_dn),
                            "auc": a_, "delta": 2 * a_ - 1, "p": pp})
        out.append(f"| {ct} | {len(s_up)}/{len(s_dn)} | {a_:.3f} | {2*a_-1:+.3f} | {pp:.3e} | "
                   f"{'A' if a_ > 0.5 else 'B'} |")
    res["A"].to_csv("results/visium_eope_score_A.csv", index=False)
    res["B"].to_csv("results/visium_eope_score_B.csv", index=False)
    out.append("")

    all_df = pd.concat(res.values())
    all_df.to_csv("results/visium_eope_score.csv", index=False)

    A, B = res["A"]["score"].values, res["B"]["score"].values
    u, p = stats.mannwhitneyu(A, B, alternative="two-sided")
    auc = u / (len(A) * len(B))
    # Cliff's delta
    cd = 2 * auc - 1
    out += ["## 结果", "",
            "| 切片 | n spots | score 均值 | 中位数 | 标准差 |",
            "|---|---|---|---|---|"]
    for sec in ["A", "B"]:
        s = res[sec]["score"]
        out.append(f"| section{sec} | {len(s)} | {s.mean():+.3f} | {s.median():+.3f} | {s.std():.3f} |")
    out += ["", f"Mann-Whitney AUC(A>B) = **{auc:.3f}**，Cliff's delta = **{cd:+.3f}**，p = {p:.3e}", "",
            "（AUC>0.5 表示 sectionA 的 eoPE 签名分数更高）", ""]

    hi = "A" if auc > 0.5 else "B"
    out += [f"**判定：section{hi} 的 eoPE 签名分数更高 → 更可能为 eoPE 病例切片。**", "",
            "## 稳健性检查：与深度的相关性", ""]
    for sec in ["A", "B"]:
        r, pv = stats.spearmanr(res[sec]["score"], res[sec]["depth"])
        out.append(f"- section{sec}: Spearman(score, depth) = {r:+.3f} (p={pv:.2e})")
    out.append("")

    # 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for j, sec in enumerate(["A", "B"]):
            d = res[sec]
            ax = axes[0, j]
            sc = ax.scatter(d["x"], d["y"], c=d["score"], cmap="RdBu_r", s=6,
                            vmin=-1.5, vmax=1.5)
            ax.set_title(f"section{sec} eoPE signature score")
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.axis("off")
            fig.colorbar(sc, ax=ax, shrink=0.8)
            axes[1, j].hist(d["score"], bins=60, color="#c0392b" if sec == "A" else "#2471a3")
            axes[1, j].set_title(f"section{sec} score distribution")
            axes[1, j].set_xlabel("eoPE score")
        fig.suptitle("snRNA-derived eoPE signature scored on Visium sections")
        fig.tight_layout()
        fig.savefig("results/visium_eope_score.png", dpi=140)
        plt.close(fig)
    except Exception as e:
        out.append(f"绘图失败: {e}")

    open("results/visium_eope_score.md", "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
