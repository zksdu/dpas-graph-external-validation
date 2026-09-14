#!/bin/bash
# DPAS-Graph 满预算重训（100 epochs，默认超参），内存门限 13.5GB
cd /d/workbuddy/0911/pe-virtual-protein
PY=/c/Users/Administrator/.workbuddy/binaries/python/envs/dpas/Scripts/python.exe
OUT=/c/workbuddy_archive/lodo_v4_100ep
LOGD=runs/lodo_v4_100ep_logs
mkdir -p "$LOGD" "$OUT"
while true; do
  free=$("$PY" -c "import psutil;print(psutil.virtual_memory().available/2**30)")
  echo "[$(date +%H:%M)] waiting, available=${free}GB"
  if awk -v a="$free" 'BEGIN{exit !(a>=13.5)}'; then break; fi
  sleep 600
done
echo "[$(date +%H:%M)] start 100ep training"
"$PY" -u DPAS-Graph/scripts/train/run_lodo.py \
  --specs_json proc/pairs_preprocessed_v3.json --names "tonsil,breast_cancer" \
  --out_root "$OUT" --epochs 100 --lr_scale 1.0 \
  --log_mode compact --seed 0 > "$LOGD/train.log" 2>&1
rc=$?
echo "[$(date +%H:%M)] rc=$rc"
