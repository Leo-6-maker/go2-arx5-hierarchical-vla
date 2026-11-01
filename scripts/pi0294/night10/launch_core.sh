#!/usr/bin/env bash
# Execute ONLY from an authorized CoreLab user session after native bindings.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'usage: launch_core.sh BOUND_PLAN_JSON NEW_NODE_LOCAL_RUN_ROOT USER_UNIT_NAME' >&2
  exit 2
fi
plan=$(realpath -e "$1")
root="$2"
unit="$3"
[[ "$root" = /* && "$unit" =~ ^core-night10-[A-Za-z0-9-]+$ ]] || { echo 'absolute new run root and safe unit name required' >&2; exit 2; }
[[ ! -e "$root" ]] || { echo 'run root already exists: reconcile; never restart blindly' >&2; exit 2; }
[[ -d "$(dirname "$root")" ]] || { echo 'missing approved node-local parent' >&2; exit 2; }
python=$(command -v python)
"$python" -m scripts.pi0294.night10.supervisor check --plan "$plan"
systemctl --user show-environment >/dev/null
# Does not enable linger or edit any systemd unit definition.
systemd-run --user --collect --unit="$unit" --same-dir \
  "$python" -u -m scripts.pi0294.night10.supervisor run --arm --plan "$plan" --output "$root"
printf '%s\n' 'SUBMITTED_USER_TRANSIENT_UNIT_NOT_READY_PROOF' "$unit" "$root"
echo 'Verify unit PID/starttime, two independent heartbeats and the first native receipt before declaring READY.'
