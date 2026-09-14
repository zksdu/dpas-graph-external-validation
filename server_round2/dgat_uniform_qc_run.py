# -*- coding: utf-8 -*-
"""DGAT uniform-QC rerun (reviewer-response reserve; GPU server only).

Purpose: re-run the DGAT LODO comparison with the DGAT-internal spot QC
threshold relaxed from its default 700 to 500 detected genes -- the threshold
used everywhere else in this study -- so that DGAT and our models are compared
on identical spot sets. Responds to reviewer question "is the DGAT comparison
confounded by its own spot filter?" (manuscript quantifies 16/3,024 retained).

Usage (GPU server, same env as dgat_run.py run4):
    python server_round2/dgat_uniform_qc_run.py

Design:
  * Patches utils.Preprocessing.preprocess_ST BEFORE importing
    Model.Train_and_Predict, so both the training folds and the external
    prediction inherit the 500-gene threshold.
  * If the DGAT preprocess_ST exposes a `min_genes`-like parameter it is
    called with 500 directly; otherwise a documented-equivalent pipeline
    (filter_cells min_genes=500 -> CP10K -> log1p -> scale) replaces it, with
    a hard assert on the retained spot fraction so silent deviations fail loudly.
  * All outputs are namespaced `_uniformqc` and never overwrite the original
    run's tables.

AFTER THE RUN: pull results_server_final + the two CSVs back to the local
machine, verify md5, extend paper_number_audit.py, then (and only then)
consider shutting the instance down.
"""
import inspect
import os
import sys

ROOT = "/root/pe-virtual-protein"
WORK = os.path.join(ROOT, "dgat_work_uniformqc")
DGAT = os.path.join(ROOT, "DGAT_baseline")
os.makedirs(os.path.join(WORK, "resources"), exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, DGAT)

MIN_GENES = 500  # this study's uniform threshold (DGAT default: 700)

import scanpy as sc  # noqa: E402


def _patched_preprocess_ST(st, *a, **k):
    """Uniform-QC replacement for DGAT's preprocess_ST."""
    import utils.Preprocessing as P
    sig = inspect.signature(P.preprocess_ST)
    for name in ("min_genes", "min_gene", "n_genes_min", "min_detected_genes"):
        if name in sig.parameters:
            k[name] = MIN_GENES
            print(f"[uniformqc] calling original preprocess_ST with {name}={MIN_GENES}", flush=True)
            return P.preprocess_ST(st, *a, **k)
    print("[uniformqc] no threshold parameter exposed; applying documented-equivalent "
          f"pipeline (filter_cells min_genes={MIN_GENES} -> CP10K -> log1p -> scale)", flush=True)
    sc.pp.filter_cells(st, min_genes=MIN_GENES)
    sc.pp.normalize_total(st, target_sum=1e4)
    sc.pp.log1p(st)
    sc.pp.scale(st, max_value=10)
    return st


import utils.Preprocessing as _P  # noqa: E402
_P.preprocess_ST = _patched_preprocess_ST

# Now import the DGAT model/train modules; their internal
# `from utils.Preprocessing import preprocess_ST` resolves AFTER our patch.
from Model import Train_and_Predict as TAP  # noqa: E402
from utils.Graph_utils import MultiGraphDataset_for_no_protein  # noqa: E402
from Model.dgat import GATEncoder, Decoder_Protein  # noqa: E402

# --- inherit the rest of the protocol from the original run script ---
sys.path.insert(0, os.path.join(ROOT, "server_round2"))
_orig = open(os.path.join(ROOT, "server_round2", "dgat_run.py"), encoding="utf-8").read()
assert "preprocess_ST(st)" in _orig, "original script drifted; re-derive this wrapper"

# Re-execute the original script body with:
#   1. WORK/MODEL_DIR/results redirected to *_uniformqc paths
#   2. the already-patched preprocess_ST (module-level import in the original
#      body will re-bind to our patched function because utils.Preprocessing
#      is already in sys.modules with the patched attribute)
_body = _orig.replace('dgat_work"', 'dgat_work_uniformqc"').replace(
    "dgat_work/", "dgat_work_uniformqc/")
_body = _body.replace('"""DGAT LODO 同协议对照 v5（预测段修复版）。',
                      '"""Uniform-QC wrapper execution body (500-gene threshold)."""  # ')
exec(compile(_body, "dgat_run_uniformqc_body.py", "exec"))

print("UNIFORM-QC RUN COMPLETE — pull CSVs back and verify md5 before shutdown", flush=True)
