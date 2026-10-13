#!/usr/bin/env bash
set -euo pipefail
entry_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
profile="${1:-}"
if [[ -z "$profile" ]]; then
  echo 'Choose: amd-management | 4080-simulation | 4080-cpu-data | hopper-neural' >&2
  exit 2
fi
shift
entry_json="$(/usr/bin/python3 "$entry_dir/doctor.py" "$profile" --resolve-only)" || { printf '%s\n' "$entry_json"; exit 2; }
if [[ "$#" -eq 0 ]]; then
  exec /usr/bin/python3 "$entry_dir/doctor.py" "$profile"
fi
if [[ "$profile" == hopper-neural && -z "${PBS_JOBID:-}" ]]; then
  echo 'PBS_REQUIRED: use the existing legal neural job environment.' >&2
  exit 2
fi
entry_python="$(printf '%s' "$entry_json" | /usr/bin/python3 -c 'import sys,json;print(json.load(sys.stdin)["python"])')"
entry_cwd="$(printf '%s' "$entry_json" | /usr/bin/python3 -c 'import sys,json;print(json.load(sys.stdin)["cwd"])')"
export PYTHONPATH="$entry_cwd/src:$entry_cwd/../isaac-gr00t"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GR00T_WBC_PYTHON="$entry_python"
cd -- "$entry_cwd"
if [[ "$1" == python ]]; then shift; exec "$entry_python" "$@"; fi
exec "$@"
