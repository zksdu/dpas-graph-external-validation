#!/bin/bash
# 服务器端一键队列：环境自备(AutoDL镜像自带torch+cuda) → 缓存 → 100ep → designB2(10)+designC(31) → 打包
# 用法: nohup bash scripts/server_run_all.sh > server_run.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOGD=runs/server_logs
mkdir -p "$LOGD"

echo "=== [1/4] GPU check ==="
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== [2/4] panels + graph caches ==="
if [ ! -f proc/panel_sub_v2/manifest.json ]; then
  "$PY" scripts/designBC_make_panels.py 2>&1 | tee "$LOGD/make_panels.log"
fi
if [ ! -f "$LOGD/precache.done" ]; then
  "$PY" scripts/designBC_precache.py 2>&1 | tee "$LOGD/precache.log" && touch "$LOGD/precache.done"
fi

echo "=== [3/4] DPAS 100-epoch full budget (GPU) ==="
if [ ! -f runs/lodo_v4_100ep/ckpt/holdout_breast_cancer/evaluation_summary.json ]; then
  "$PY" -u DPAS-Graph/scripts/train/run_lodo.py \
    --specs_json proc/pairs_preprocessed_v3.json --names "tonsil,breast_cancer" \
    --out_root runs/lodo_v4_100ep --epochs 100 --lr_scale 1.0 \
    --log_mode compact --seed 0 2>&1 | tee "$LOGD/train100ep.log"
fi

echo "=== [4/4] 41 variants (designB2 x10 + designC x31) ==="
for v in $("$PY" -c "import json;m=json.load(open('proc/panel_sub_v2/manifest.json'));print(' '.join(x['name'] for x in m))"); do
  specs="proc/panel_sub_v2/${v}/specs.json"
  case "$v" in
    size*) out="runs/designB2/${v}"; seed="${v##*seed}";;
    *)     out="runs/designC/${v}";  seed=1;;
  esac
  if [ -f "$LOGD/${v}.done" ]; then echo "[skip] $v"; continue; fi
  echo "[$(date +%H:%M)] $v start (seed=$seed)"
  if "$PY" -u DPAS-Graph/scripts/train/run_lodo.py \
      --specs_json "$specs" --names "tonsil,breast_cancer" \
      --out_root "$out" --epochs 20 --lr_scale 1.0 \
      --log_mode compact --seed "$seed" > "$LOGD/${v}.log" 2>&1; then
    touch "$LOGD/${v}.done"; echo "[$(date +%H:%M)] $v OK"
    # 节省磁盘：删除模型权重(每变体~800MB)，保留 per_protein_metrics.csv/npy/summary
    find "$out" -name "*.pth" -delete
  else
    echo "[$(date +%H:%M)] $v FAIL"
  fi
done

echo "=== packaging results ==="
tar czf results_server.tar.gz runs/lodo_v4_100ep/ckpt/holdout_*/evaluation_summary.json \
  runs/lodo_v4_100ep/ckpt/holdout_*/per_protein_metrics.csv \
  $(find runs/designB2 runs/designC -name "evaluation_summary.json" -o -name "per_protein_metrics.csv" 2>/dev/null | tr '\n' ' ') \
  runs/server_logs 2>/dev/null || true
ls -lh results_server.tar.gz
echo "ALL DONE — download results_server.tar.gz"
