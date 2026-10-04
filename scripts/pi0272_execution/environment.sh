root=/scratch/e1192548/Projects/go2-arx5/experiments/PI0272
venv=/scratch/e1192548/stage22_h200_runtime_deb920_20260902/runtime/venv
ff=/scratch/e1192548/Projects/go2-arx5/archive/stage_runs/STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1__9243a7880b06ac3984ddf4a1a667b685441ec0f34d6d529fa0f96c3c894e10cf
export HF_HOME=/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_CORETYPE=Prescott
export LD_LIBRARY_PATH="$ff/lib:$ff/compat:$venv/lib/python3.12/site-packages/torchcodec:$venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root/code:/scratch/e1192548/Projects/go2-arx5/experiments/PI0269/code:/scratch/e1192548/Projects/go2-arx5/experiments/PI0267/code:/scratch/e1192548/Projects/go2-arx5/experiments/PI0264/code:/scratch/e1192548/Projects/go2-arx5/experiments/PI0260/R3/execution:/scratch/e1192548/pi0244_fixed16_10k_v1/scripts:/scratch/e1192548/pi0244_fixed16_10k_v1/src:/scratch/e1192548/pi0247_generation_to_sim_20260926/execution_code_m64_39b3718a0a1e/scripts:/scratch/e1192548/pi0247_generation_to_sim_20260926/execution_code_m64_39b3718a0a1e/src:/scratch/e1192548/stage22_h200_runtime_deb920_20260902/src/isaac-gr00t:${PYTHONPATH:-}"
