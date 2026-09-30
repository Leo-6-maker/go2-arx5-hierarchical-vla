#!/usr/bin/env bash
# Hopper routes CPU-only PBS requests to a GPU queue; archive on the login host at low priority.
set -euo pipefail
root=/scratch/e1192548/Projects/go2-arx5/experiments/PI0261
venv=/scratch/e1192548/stage22_h200_runtime_deb920_20260902/runtime/venv
arm=${1:?arm}
step=${2:?step}
test "$arm" = C -o "$arm" = P
test "$step" = 1000 -o "$step" = 2000
export HF_HOME="$root/hf_home"
export HF_TOKEN="$(cat /home/svu/e1192548/.config/huggingface/token)"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
exec nice -n 15 "$venv/bin/python" -u "$root/code/pi0261_archive_trainer.py" --arm "$arm" --step "$step"
