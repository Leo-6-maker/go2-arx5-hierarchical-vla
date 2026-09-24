#!/usr/bin/env bash
set -euo pipefail

: "${PI0246_T_RUN_CARD:?set the sealed Corelab child-card path}"
: "${PI0246_T_RUNTIME_ROOT:?set the bound runtime root}"
: "${PI0246_T_RUNTIME_WRAPPER:?set this wrapper path}"
: "${PI0246_T_RUNTIME_BINDING:?set the runtime binding path}"
: "${PI0246_T_RUNTIME_BINDING_SHA256:?set the sealed runtime-binding SHA256}"
: "${PI0246_T_TRAINER:?set the staged trainer path}"
: "${PI0246_T_PYTHON:?set the exact Stage10 Python path}"

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec "$PI0246_T_PYTHON" "$PI0246_T_TRAINER" --train --run-card "$PI0246_T_RUN_CARD"
