#!/usr/bin/env bash
set -euo pipefail

model="${1:?expected R0 or R2}"
case "$model" in R0|R2) ;; *) exit 2 ;; esac

root=/var/cache/users/liuyu-nfs/pi0246-p1-corelab-stage-20260924
python=/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/runtime/venv/bin/python
mkdir -p "$root/logs"
trap 'printf "%s\n" "$?" > "$root/logs/p1-$model.exit"' EXIT
exec > "$root/logs/p1-$model.log" 2>&1

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1

"$python" -u "$root/pi0245_p1_stock.py" \
  --r0-root "$root/r0" \
  --run-card "$root/r0/pi0243_run_card.json" \
  --path-overlay "$root/pi0246_p1_path_overlay_corelab_20260924.json" \
  --p0-binding "$root/p0/effective_processor_xyz_v2.json" \
  --noise-root "$root/noise" \
  --model-root "$root/models/$model" \
  --model-name "$model" \
  --output-root "$root/p1_corelab" \
  --source-index 0 --noise-name N0 --device cuda:0 --all
