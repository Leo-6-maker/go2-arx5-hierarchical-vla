#!/usr/bin/env bash
set -euo pipefail
root=/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2/R3
view="$root/models/G6000/arms/PI0260/R3/G/checkpoint-6000"
pid="$root/eval/G_6000_run.pid"
test ! -e "$pid"
test ! -e "$root/eval/G_6000"
test -f "$root/model_bindings/G_6000.json"
test -f "$view/model.safetensors.index.json"
test "$(wc -l < "$root/eval/attempts.jsonl")" -eq 1302
test "$(wc -l < "$root/eval/head_forwards.jsonl")" -eq 5208
cd "$root"
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python -u pi0260_r3_eval.py \
  --point G_6000 --model-view "$view" --binding "$root/model_bindings/G_6000.json" \
  > "$root/eval/G_6000_run.log" 2>&1 < /dev/null &
echo $! > "$pid"
sleep 2
kill -0 "$(cat "$pid")"
printf 'G_6000_EVAL_PID=%s\n' "$(cat "$pid")"
