#!/usr/bin/env bash

set -euo pipefail

RAW="/your path/xxx.ARW"
OUT="output.jpg"
CONTRAST="-2"
SHARPEN="0.5"
DUMP=0

while getopts "r:o:c:s:d" opt; do
  case "$opt" in
    r) RAW="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    c) CONTRAST="$OPTARG" ;;
    s) SHARPEN="$OPTARG" ;;
    d) DUMP=1 ;;
    *) echo "[run_isp] unknown parameter：-$OPTARG" && exit 1 ;;
  esac
done
shift $((OPTIND - 1))

CMD=(python3 ISPdemo.py "$RAW" "$OUT" --contrast "$CONTRAST" --sharpen "$SHARPEN")
if [[ "$DUMP" -eq 1 ]]; then
  CMD+=(--dump)
fi

echo "[run_isp] under implementation：${CMD[*]}"
"${CMD[@]}"

echo "[run_isp] Processing complete, output file：$OUT"