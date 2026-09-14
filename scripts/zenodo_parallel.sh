#!/bin/bash
# Zenodo 大文件多连接分段并行下载器
# 用法: zenodo_parallel.sh <file_key> <total_bytes> <n_parts> <outfile>
KEY="$1"; TOTAL="$2"; N="$3"; OUT="$4"
URL="https://zenodo.org/api/records/8159511/files/${KEY}/content"
DIR="${OUT}.parts"
mkdir -p "$DIR"
CHUNK=$(( (TOTAL + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  START=$((i * CHUNK))
  END=$((START + CHUNK - 1))
  if [ $END -ge $TOTAL ]; then END=$((TOTAL - 1)); fi
  if [ $START -ge $TOTAL ]; then break; fi
  PART="$DIR/part_$(printf '%03d' $i)"
  (
    expected=$((END - START + 1))
    for attempt in 1 2 3 4 5 6; do
      have=0
      [ -f "$PART" ] && have=$(stat -c %s "$PART")
      if [ "$have" -ge "$expected" ]; then break; fi
      s=$((START + have))
      curl -sS --ssl-no-revoke -L --retry 2 --retry-delay 5 --speed-limit 5000 --speed-time 120 \
        -r ${s}-${END} -o - "$URL" >> "$PART" 2>/dev/null
    done
  ) &
done
wait
# 合并
> "$OUT"
for i in $(seq 0 $((N-1))); do
  PART="$DIR/part_$(printf '%03d' $i)"
  [ -f "$PART" ] && cat "$PART" >> "$OUT"
done
have=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
echo "downloaded=$have expected=$TOTAL"
if [ "$have" = "$TOTAL" ]; then
  rm -rf "$DIR"
  echo "OK"
else
  echo "INCOMPLETE"
fi
