"""RNA 覆盖诊断：检验"亚型标记失败源于训练侧 RNA 覆盖不足"假设。
结论（2026-09-12）：假设被反驳——淋巴亚型标记基因在训练数据中的检出率/表达量
不低于区室标记；外部验证中亚型级失败更可能源于 RNA-蛋白耦合强度在亚型级较弱，
或 35-plex 抗体在亚型间的交叉反应。输出 results/rna_coverage_diagnosis.csv"""
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
d = lambda X: X.toarray() if sp.issparse(X) else np.asarray(X)
COMP = ['EPCAM', 'PECAM1', 'ACTA2', 'VIM', 'PTPRC', 'CD14', 'ITGAX', 'HLA_DRA', 'CD163', 'CD68']
LYMPH = ['CD3E', 'CD3D', 'CD4', 'CD8A', 'MS4A1', 'CD19', 'PAX5', 'CR2', 'CXCR5', 'CD27', 'SDC1']

rows = []
for ds in ['tonsil', 'breast_cancer']:
    a = ad.read_h5ad(ROOT / f'proc/preprocessed/{ds}/RNA_proc.h5ad')
    names = list(map(str, a.var_names))
    X = d(a.X)
    for g in sorted(set(COMP + LYMPH)):
        if g in names:
            v = X[:, names.index(g)]
            rows.append({'dataset': ds, 'gene': g, 'in_panel': True,
                         'detect_rate': float((v > 0).mean()), 'mean_expr': float(v.mean())})
        else:
            rows.append({'dataset': ds, 'gene': g, 'in_panel': False,
                         'detect_rate': np.nan, 'mean_expr': np.nan})

t = pd.DataFrame(rows)
t['group'] = np.where(t.gene.isin(LYMPH), 'lymph', 'compartment')
t.to_csv(ROOT / 'results/rna_coverage_diagnosis.csv', index=False)
agg = t.groupby('group')[['detect_rate', 'mean_expr']].mean().round(3)
agg.to_csv(ROOT / 'results/rna_coverage_diagnosis_groupmean.csv')
print(agg.to_string())
print('\nnot in panel:', sorted(set(COMP + LYMPH) - set(t[t.in_panel].gene.unique())))
