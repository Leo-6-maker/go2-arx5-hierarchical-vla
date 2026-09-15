#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/runtime/groot-site${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/.singularity.d/libs:/runtime/groot-site/torch/lib:/runtime/groot-site/torchcodec${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec /usr/bin/python3.12 "$@"
