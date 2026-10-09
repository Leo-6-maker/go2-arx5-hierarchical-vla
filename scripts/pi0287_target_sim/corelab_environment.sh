#!/bin/bash
set -euo pipefail
root=/home/liuyu-nfs/Projects/go2-arx5/PI0287
native=/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/stage13c/runtime_env
wbc=/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MAX_JOBS=2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$root/project-src/src:$root/project-src:/home/liuyu-nfs/Projects/go2-arx5/PI0252/runtime/PI0226_pytorch3d_shim:$wbc:$native/public_native/isaacgym/python"
export TORCH_EXTENSIONS_DIR=$root/runtime/torch_extensions
export TMPDIR=$root/runtime/tmp XDG_CACHE_HOME=$root/runtime/cache
export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH="$native/public_native/isaacgym/python/isaacgym/_bindings/linux-x86_64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$native/public_native/python_env/lib/python3.8/site-packages/torch/lib:$native/native_py38_4080/lib:$native/public_native/python_env/lib"
export PATH="$native/native_py38_4080/bin:$native/public_native/python_env/bin:/usr/local/cuda/bin:/usr/bin:/bin"
exec "$native/native_py38_4080/bin/python" "$@"
