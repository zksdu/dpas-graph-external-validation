#!/bin/bash
# 设计 B 全部变体训练（串行）。内存门限：available >= 12 GB 才启动下一个训练，
# 否则每 10 分钟重查（训练 commit 峰值 ~11 GB，本机无页面文件）。
cd /d/workbuddy/0911/pe-virtual-protein
PY=/c/Users/Administrator/.workbuddy/binaries/python/envs/dpas/Scripts/python.exe
LOGD=runs/designB/logs
mkdir -p "$LOGD"

need_mem_gb() {
  "$PY" -c "import psutil;print(psutil.virtual_memory().available/2**30)"
}

run_one() {  # $1=variant_name  $2=specs_path  $3=seed
  local v=$1 specs=$2 seed=$3
  if [ -f "$LOGD/${v}.done" ]; then echo "[skip] $v done"; return 0; fi
  rm -rf "runs/designB/${v}/ckpt"
  while true; do
    free=$(need_mem_gb)
    echo "[$(date +%H:%M)] $v waiting, available=${free}GB"
    if awk -v a="$free" 'BEGIN{exit !(a>=13.5)}'; then break; fi
    sleep 600
  done
  echo "[$(date +%H:%M)] $v start"
  "$PY" -u DPAS-Graph/scripts/train/run_lodo.py \
    --specs_json "$specs" --names "tonsil,breast_cancer" \
    --out_root "runs/designB/${v}" --epochs 20 --lr_scale 1.0 \
    --log_mode compact --seed "$seed" > "$LOGD/${v}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then touch "$LOGD/${v}.done"; echo "[$(date +%H:%M)] $v OK"; else echo "[$(date +%H:%M)] $v FAIL rc=$rc"; fi
  return $rc
}

for seed in 1 2 3 4 5; do
  run_one "size25_seed${seed}" "proc/panel_sub/size25_seed${seed}/specs.json" "$seed"
done
for seed in 1 2 3 4 5; do
  run_one "size20_seed${seed}" "proc/panel_sub/size20_seed${seed}/specs.json" "$seed"
done
for seed in 1 2 3 4; do
  run_one "base_seed${seed}" "proc/pairs_preprocessed_v3.json" "$seed"
done
echo "[$(date +%H:%M)] ALL DONE"
