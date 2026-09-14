#!/usr/bin/env python
"""跨图谱合并 eoPE 签名（胎盘 6v6 + 蜕膜 4v4）→ Visium 切片内
疾病签名得分 × 虚拟蛋白的空间关联。

背景：
  * 胎盘/蜕膜两图谱均无空间数据；A/B 切片身份未确定（上轮结论）。
  * 因此只做**切片内**关联：同一张片内，疾病签名高 spots 与低 spots 的
    虚拟蛋白是否系统性不同（不跨片比较，避开深度混杂）。
  * 签名用 moderated t 排序 top-N（蜕膜 4v4 无一过 BH，改用 rank-based），
    跨图谱用 t 值 Stouffer 合并（同 compartment 内）。
  * 阴性对照：4 个同型对照虚拟蛋白 + 随机打乱空间标签的置换 p。

输出：
  results/combined_signatures.json
  results/disease_protein_assoc_<A,B>.csv
  results/disease_protein_assoc_summary.md
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

rng = np.random.default_rng(0)
N_PER_COMP = 100          # 每图谱每 compartment 取 |t| 排序 top-N up / down
N_PERM = 2000

# ---- compartment 映射（两图谱细胞类型 → 4 个区室） ----
COMP_MAP = {
    "placenta": {
        "tropho": ["vSTB1", "vSTB2", "vSTBjuv", "vCTB"],
        "immune": ["vHBC", "vMC", "vTcell"],
        "stroma": ["vFB1"],
        "endo": ["vVEC"],
    },
    "decidua": {
        "tropho": ["dEVT", "dDSTB"],
        "immune": ["dMAC1", "dMAC2", "dNK1", "dNK2", "dTcell", "dMono1"],
        "stroma": ["DSC1"],
        "endo": ["dVEC", "dLEC"],
    },
}


def load_de(atlas):
    """读 results/<atlas>_de_<ct>.csv，返回 {ct: (gene, t, log2FC)}。"""
    out = {}
    pref = "sn_de_" if atlas == "placenta" else "dec_de_"
    for f in sorted(glob.glob(f"results/{pref}*.csv")):
        ct = os.path.basename(f)[len(pref):-4]
        if ct in ("celltypes",):
            continue
        d = pd.read_csv(f).dropna(subset=["t"])
        out[ct] = d.set_index("gene")[["t", "log2FC"]]
    return out


def stouffer_merge(ts, ns):
    """t → 近似 z（正态分位数按符号与秩），再做样本量加权 Stouffer。"""
    from scipy.stats import norm
    zs = []
    for t, n in zip(ts, ns):
        # 大样本 t 近似 z：z = t*(1 - 3/(4df-1))；这里直接用 sign-preserving 秩转换
        r = pd.Series(np.abs(t)).rank(pct=True).values * 2 - 1
        z = norm.ppf(np.clip(r, 1e-6, 1 - 1e-6)) * np.sign(t)
        zs.append(z)
    w = np.array(ns, dtype=float)
    w = w / w.sum()
    return np.sum([z * wi for z, wi in zip(zs, w)], axis=0)


def build_signatures():
    sigs = {}
    report = []
    for comp in ["tropho", "immune", "stroma", "endo"]:
        ts, ns, genes_ref = [], [], None
        for atlas in ["placenta", "decidua"]:
            de = load_de(atlas)
            cts = [c for c in COMP_MAP[atlas][comp] if c in de]
            if not cts:
                report.append(f"{comp}/{atlas}: 无可用细胞类型")
                continue
            for ct in cts:
                d = de[ct]
                ts.append(d["t"].reindex(genes_ref) if genes_ref is not None else d["t"])
                ns.append({"placenta": 12, "decidua": 9}[atlas])  # 供体数
                if genes_ref is None:
                    genes_ref = d.index
                else:
                    genes_ref = genes_ref.union(d.index)
            report.append(f"{comp}/{atlas}: {cts}")
        if not ts:
            continue
        # 对齐基因
        T = pd.concat([t.reindex(genes_ref).fillna(0) for t in ts], axis=1).values
        z = stouffer_merge([T[:, i] for i in range(T.shape[1])], ns)
        zs = pd.Series(z, index=genes_ref)
        up = zs.nlargest(N_PER_COMP).index.tolist()
        dn = zs.nsmallest(N_PER_COMP).index.tolist()
        sigs[comp] = {"up": up, "down": dn,
                      "n_atlases": len(ts), "donors": int(sum(ns))}
        report.append(f"  -> {comp}: up {len(up)} dn {len(dn)}")
    return sigs, report


def score_section(sec, sigs):
    """log1p CP10k → 每基因 z → 签名得分（mean z_up - mean z_dn）。"""
    a = ad.read_h5ad(f"proc/preprocessed/pe_section{sec}/RNA_proc.h5ad")
    X = a.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    genes = list(a.var_names)
    gi = {g: i for i, g in enumerate(genes)}
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    scores, cov = {}, {}
    for comp, s in sigs.items():
        u = [g for g in s["up"] if g in gi]
        d = [g for g in s["down"] if g in gi]
        cov[comp] = (len(u), len(d))
        scores[comp] = Z[:, [gi[g] for g in u]].mean(1) - Z[:, [gi[g] for g in d]].mean(1)
    return a.obs_names.astype(str), scores, cov


def main():
    sigs, rep = build_signatures()
    json.dump(sigs, open("results/combined_signatures.json", "w"), indent=1)
    print("签名构建：")
    for r in rep:
        print(" ", r)

    out_md = ["# 切片内疾病签名 × 虚拟蛋白空间关联", "",
              "签名：胎盘(6v6) + 蜕膜(4v4) moderated-t Stouffer 合并，每 compartment "
              f"top {N_PER_COMP} up/down；Visium 每基因 z 打分；", ""]
    for sec in ["A", "B"]:
        spots, scores, cov = score_section(sec, sigs)
        prots = [x.strip() for x in open(f"runs/pe_{sec}_breast_cancer_v2.protein_names.txt")]
        pred = np.load(f"runs/pe_{sec}_breast_cancer_v2.pred.npy")
        sn = [x.strip() for x in open(f"runs/pe_{sec}_breast_cancer_v2.spot_names.txt")]
        # spot 对齐
        smap = {s: i for i, s in enumerate(spots)}
        idx = np.array([smap.get(s, -1) for s in sn])
        ok = idx >= 0
        print(f"section{sec}: spots={len(sn)} matched={ok.sum()}")
        P = pred[ok]
        C = np.array([[scores[c][idx[i]] for c in scores] for i in np.where(ok)[0]],
                     dtype=float)
        comps = list(scores.keys())
        iso = [i for i, p in enumerate(prots) if p.startswith(("mouse_", "rat_"))]
        rows = []
        for pi, pn in enumerate(prots):
            for ci, cn in enumerate(comps):
                rho, _ = spearmanr(C[:, ci], P[:, pi])
                # 置换：打乱签名得分
                null = np.empty(N_PERM)
                for k in range(N_PERM):
                    null[k] = spearmanr(C[:, ci], P[rng.permutation(len(P)), pi])[0]
                rows.append({"section": sec, "protein": pn,
                             "isotype": pn in [prots[i] for i in iso],
                             "compartment": cn, "spearman": rho,
                             "perm_p": float((np.abs(null) >= abs(rho)).mean())})
        d = pd.DataFrame(rows)
        d.to_csv(f"results/disease_protein_assoc_{sec}.csv", index=False)
        iso_floor = d[d.isotype].groupby("compartment")["spearman"].apply(
            lambda s: np.percentile(np.abs(s), 95))
        real = d[~d.isotype]
        out_md += [f"## section{sec}（spots={int(ok.sum())}）", "",
                   f"同型对照 |rho| 95 分位（噪声地板）："
                   f"{ {k: round(v, 3) for k, v in iso_floor.items()} }", "",
                   "| 蛋白 | compartment | rho | perm_p | 超地板 |", "|---|---|---|---|---|"]
        for _, r in real.iterrows():
            fl = iso_floor.get(r.compartment, np.nan)
            tag = "**是**" if abs(r.spearman) > fl and r.perm_p < 0.05 else ""
            if r.perm_p < 0.05:
                out_md.append(f"| {r.protein} | {r.compartment} | {r.spearman:+.3f} | "
                              f"{r.perm_p:.3f} | {tag} |")
        out_md.append("")
    open("results/disease_protein_assoc_summary.md", "w", encoding="utf-8").write(
        "\n".join(out_md))
    print("\n".join(out_md))


if __name__ == "__main__":
    main()
