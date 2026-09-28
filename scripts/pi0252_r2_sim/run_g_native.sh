#!/usr/bin/env bash
set -euo pipefail
task="${1:?A-D required}"

case "$task" in
 A) letter=a; xyz=(0.30 0.10 0.58);;
 B) letter=b; xyz=(0.30 -0.10 0.58);;
 C) letter=c; xyz=(0.30 0.05 0.64);;
 D) letter=d; xyz=(0.30 -0.05 0.63);;
 *) exit 2;;
esac
root=/home/liuyu-nfs/Projects/go2-arx5/PI0252
runtime=$root/runtime
native=/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/stage13c/runtime_env
wbc=/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare
code=/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/reference-317a
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=2
export TMPDIR=$runtime/tmp XDG_CACHE_HOME=$runtime/cache MPLCONFIGDIR=$runtime/matplotlib
export TORCH_EXTENSIONS_DIR=$runtime/torch_extensions
export PYTHONPATH="$runtime/PI0226_pytorch3d_shim:$wbc:$native/public_native/isaacgym/python:$code/scripts:$code/src"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH="$native/public_native/isaacgym/python/isaacgym/_bindings/linux-x86_64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$native/public_native/python_env/lib/python3.8/site-packages/torch/lib:$native/native_py38_4080/lib:$native/public_native/python_env/lib"
export PATH="$native/native_py38_4080/bin:$native/public_native/python_env/bin:/usr/local/cuda/bin:/usr/bin:/bin"
exec "$native/native_py38_4080/bin/python" "$runtime/pi0252_g_one_chunk.py" \
 --source-root "$wbc" --trajectory "/home/liuyu-nfs/pi0223_goal_b_20260918/refs_r00/train-${letter}_c0_continuous_reference.npz" \
 --arm "$wbc/runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018/checkpoints_arm/ac_weights_039999.pt" \
 --dog "$wbc/runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018/checkpoints_dog/ac_weights_039999.pt" \
 --mailbox "$root/pre_full_sim/G/$task/mailbox" --output-dir "$root/pre_full_sim/G/$task/output" --label "TRAIN-${task}_C0_G2000" --checkpoint-dir "$root/models/G2000/inference_view_step_2000" --response-timeout-s 300 --target-position-world-m "${xyz[@]}" --seed 22301 --sim-device cuda:0
