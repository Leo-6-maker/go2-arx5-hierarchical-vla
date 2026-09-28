#!/usr/bin/env bash
set -euo pipefail
task="${1:?A-D required}"
case "$task" in A|B|C|D);; *) exit 2;; esac
root=/home/liuyu-nfs/Projects/go2-arx5/PI0252
inputs=/home/liuyu-nfs/Projects/go2-arx5/PI0250/inputs/frozen_v1
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=$root/runtime/tmp XDG_CACHE_HOME=$root/runtime/cache MPLCONFIGDIR=$root/runtime/matplotlib
exec /home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python "$root/runtime/pi0252_g_model_worker.py" \
 --stage10-root /home/liuyu-nfs/Projects/isaac-gr00t --base-model-path "$inputs/processor_base" \
 --model-path "$root/models/G2000/inference_view_step_2000" --dataset-dir "$inputs/train_dataset" \
 --modality-config "$inputs/modality/go2_arx5_modality.py" --cosmos-snapshot "$inputs/cosmos_snapshot" \
 --mailbox "$root/pre_full_sim/G/$task/mailbox" --device cuda:0 --seed 22301 --poll-period-s .05 \
 --local-files-only --action-contract DENSE_H40_V1
