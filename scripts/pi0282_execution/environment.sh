root=/scratch/e1192548/Projects/go2-arx5/experiments/PI0282
venv=/scratch/e1192548/stage22_h200_runtime_deb920_20260902/runtime/venv
ff=/scratch/e1192548/Projects/go2-arx5/archive/stage_runs/STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1__9243a7880b06ac3984ddf4a1a667b685441ec0f34d6d529fa0f96c3c894e10cf
export HF_HOME=/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PI0282_OFFICIAL_DIR=/scratch/e1192548/stage22_h200_runtime_deb920_20260902/src/isaac-gr00t
export PI0282_COSMOS_DIR="$HF_HOME/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
export PYTHONPATH="$root/code:$PI0282_OFFICIAL_DIR:$root/project-src/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_CORETYPE=Prescott
export LD_LIBRARY_PATH="$ff/lib:$ff/compat:$venv/lib/python3.12/site-packages/torchcodec:$venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
