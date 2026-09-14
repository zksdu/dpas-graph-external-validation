# -*- coding: utf-8 -*-
"""Fig1 拼版（SCI 最终印刷尺寸）：A/B（基准与外部验证）、C（panel 缩减）、D（预算×基线）。
三行全宽布局，输出 results/fig1_composite.png（600 dpi，宽 ~7.4in）。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fig = plt.figure(figsize=(7.4, 9.6))
gs = fig.add_gridspec(3, 1, height_ratios=[2.9, 2.7, 3.9], hspace=0.07)
for i, f in enumerate(["results/fig1_benchmark_external.png",
                       "results/fig1c_panel_reduction.png",
                       "results/fig1d_budget_vs_baselines.png"]):
    ax = fig.add_subplot(gs[i])
    ax.imshow(mpimg.imread(f))
    ax.axis("off")

plt.savefig("results/fig1_composite.png", dpi=600, bbox_inches="tight")
print("saved fig1_composite.png")
