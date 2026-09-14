"""100ep 多 seed 聚合（风险②）：seed0（log 重建版）+ seed1 + seed2 → mean ± SD。

口径与论文一致：proc 输入、31 marker Spearman（4 样本平均后跨 fold 平均）、
isotype 地板、namesake 基线。输出 results/lodo100ep_multiseed_summary.csv。

seed0 文件为 ext_eval_100ep.log 重建版（列 mode/holdout/sample/n/PCC/SP/spotPCC/isoSP/base），
seed1/2 为 external_eval_gse_100ep.py 原生 pandas 输出（input/ckpt/sample/...），此处归一化。
"""
import numpy as np
import pandas as pd

COLMAP = {"PCC": "pcc_marker_mean", "SP": "spear_marker_mean",
          "spotPCC": "pcc_spot_mean", "isoSP": "spear_isotype_mean",
          "base": "baseline_spear_mean"}
KEY_COLS = ["spear_marker_mean", "pcc_marker_mean", "pcc_spot_mean",
            "spear_isotype_mean", "baseline_spear_mean"]


def load_seed(seed):
    if seed == 0:
        df = pd.read_csv("gse_external_summary_100ep.csv")
        df = df[df["mode"] == "proc"]  # 16 行含 raw 对照版，只取 proc
        df = df.rename(columns={"holdout": "ckpt", **COLMAP})
        for c in KEY_COLS:
            df[c] = df[c].astype(str).str.replace("+", "", regex=False).astype(float)
    else:
        df = pd.read_csv(f"server_round2/gse_external_summary_100ep_s{seed}.csv")
        df = df[df.input == "proc"]
    return df.groupby("ckpt")[KEY_COLS].mean()


rows = []
for seed in (0, 1, 2):
    agg = load_seed(seed)
    rows.append({"seed": seed, **{k: agg[k].mean() for k in KEY_COLS}})

out = pd.DataFrame(rows).set_index("seed")
summary = pd.DataFrame({
    "mean": out.mean().round(4),
    "sd": out.std(ddof=1).round(4),
    **{f"seed{s}": out.loc[s].round(4) for s in (0, 1, 2)},
})
print(summary.to_string())
out.round(4).to_csv("results/lodo100ep_multiseed_summary.csv")
print("\nsaved: results/lodo100ep_multiseed_summary.csv")
