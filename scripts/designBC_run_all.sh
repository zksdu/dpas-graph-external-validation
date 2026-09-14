#!/bin/bash
# 主链：等 100ep 训练完成 → designB2/designC 面板生成 → 预缓存 → 41 个变体串行训练
# 完成信号：C:/workbuddy_archive/lodo_v4_100ep/ckpt/holdout_breast_cancer/evaluation_summary.json
cd /d/workbuddy/0911/pe-virtual-protein
PY=/c/Users/Administrator/.workbuddy/binaries/python/envs/dpas/Scripts/python.exe
LOGD=runs/designBC/logs
mkdir -p "$LOGD"

DONE100="/c/workbuddy_archive/lodo_v4_100ep/ckpt/holdout_breast_cancer/evaluation_summary.json"
echo "[$(date +%H:%M)] waiting for 100ep training..."
for i in $(seq 1 120); do
  [ -f "$DONE100" ] && break
  sleep 300
done
if [ ! -f "$DONE100" ]; then echo "[$(date +%H:%M)] 100ep timeout (10h) — abort"; exit 1; fi
echo "[$(date +%H:%M)] 100ep finished, starting panels"

"$PY" scripts/designBC_make_panels.py > "$LOGD/make_panels.log" 2>&1 || { echo "make_panels FAIL"; exit 1; }
"$PY" scripts/designBC_precache.py   > "$LOGD/precache.log" 2>&1  || { echo "precache FAIL"; exit 1; }

run_one() {
  local v=$1 specs=$2 out=$3 seed=$4
  if [ -f "$LOGD/${v}.done" ]; then echo "[skip] $v"; return 0; fi
  while true; do
    free=$("$PY" -c "import psutil;print(psutil.virtual_memory().available/2**30)")
    echo "[$(date +%H:%M)] $v waiting, available=${free}GB"
    if awk -v a="$free" 'BEGIN{exit !(a>=13.5)}'; then break; fi
    sleep 600
  done
  echo "[$(date +%H:%M)] $v start"
  "$PY" -u DPAS-Graph/scripts/train/run_lodo.py \
    --specs_json "$specs" --names "tonsil,breast_cancer" \
    --out_root "$out" --epochs 20 --lr_scale 1.0 \
    --log_mode compact --seed "$seed" > "$LOGD/${v}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then touch "$LOGD/${v}.done"; echo "[$(date +%H:%M)] $v OK"
  else echo "[$(date +%H:%M)] $v FAIL rc=$rc"; fi
}

for v in $("$PY" -c "import json;m=json.load(open('proc/panel_sub_v2/manifest.json'));print(' '.join(x['name'] for x in m))"); do
  specs="proc/panel_sub_v2/${v}/specs.json"
  case "$v" in
    size*) out="C:/workbuddy_archive/designB2/${v}"; seed="${v##*seed}";;
    *)     out="C:/workbuddy_archive/designC/${v}";  seed=1;;
  esac
  run_one "$v" "$specs" "$out" "$seed"
done
echo "[$(date +%H:%M)] ALL DONE"
