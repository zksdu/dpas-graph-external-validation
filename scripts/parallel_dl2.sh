#!/bin/bash
# 大文件多连接分段并行下载器 v2（修正版）
# 用法: parallel_dl2.sh <URL> <total_bytes> <n_parts> <outfile> [md5]
# 特性: 单实例锁、分片大小精确校验（超限则重下）、合并后 size+md5 校验
URL="$1"; TOTAL="$2"; N="$3"; OUT="$4"; MD5="$5"
DIR="${OUT}.parts"
mkdir -p "$DIR"
if ! mkdir "$DIR/.lock" 2>/dev/null; then
  echo "ANOTHER INSTANCE RUNNING (lock at $DIR/.lock)"; exit 1
fi
trap 'rmdir "$DIR/.lock" 2>/dev/null' EXIT
CHUNK=$(( (TOTAL + N - 1) / N ))
echo "url=$URL total=$TOTAL n=$N chunk=$CHUNK"
for i in $(seq 0 $((N-1))); do
  START=$((i * CHUNK))
  END=$((START + CHUNK - 1))
  if [ $END -ge $TOTAL ]; then END=$((TOTAL - 1)); fi
  if [ $START -ge $TOTAL ]; then break; fi
  PART="$DIR/part_$(printf '%03d' $i)"
  (
    expected=$((END - START + 1))
    for attempt in $(seq 1 300); do
      have=0
      [ -f "$PART" ] && have=$(stat -c %s "$PART")
      [ "$have" -eq "$expected" ] && exit 0
      if [ "$have" -gt "$expected" ]; then rm -f "$PART"; have=0; fi
      s=$((START + have))
      need=$((expected - have))
      tmp="$PART.tmp.$$"
      curl -sS --ssl-no-revoke -L --speed-limit 1000 --speed-time 180 \
        -r ${s}-${END} -o "$tmp" "$URL" 2>/dev/null
      got=0
      [ -f "$tmp" ] && got=$(stat -c %s "$tmp")
      if [ "$got" -gt 0 ] && [ "$got" -le "$need" ]; then
        cat "$tmp" >> "$PART"
        rm -f "$tmp"
      else
        rm -f "$tmp"
        echo "part_$(printf '%03d' $i) retry a=$attempt got=$got need=$need" >> "$DIR/../dl2_warn.log"
      fi
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
if [ "$have" != "$TOTAL" ]; then
  echo "INCOMPLETE"
  exit 2
fi
if [ -n "$MD5" ]; then
  echo "computing md5 ..."
  m=$(md5sum "$OUT" | awk '{print $1}')
  echo "md5=$m expected=$MD5"
  if [ "$m" = "$MD5" ]; then
    echo "OK-MD5"
    rm -rf "$DIR"
  else
    echo "MD5-MISMATCH"
    exit 3
  fi
else
  echo "OK"
  rm -rf "$DIR"
fi
