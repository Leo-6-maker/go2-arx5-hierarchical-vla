# HPC launcher index

The files in `hpc/` are historical, gate-bound launch surfaces. They are kept
in place for reproducibility and are not submitted by the repository
consolidation PR.

## Contents

- `pbs/` contains stage-specific PBS templates for the historical Stage22,
  Stage28–Stage40 campaigns.
- `stage30_p0_import_origin.sh` is the Stage30 import/provenance wrapper.
- `stage40_corelab_ct3_runtime.sh` is the Corelab RTX runtime-preparation and
  hold wrapper. It records explicit paths and GPU UUIDs but is not executed by
  this cleanup.

Each launcher binds its own stage SHA, gate marker, artifact root, and runtime
inputs. Do not generalize a variable or copy a scheduler/cache assumption from
Vanda A40 into Corelab RTX. Do not edit a wrapper solely to make historical
reports look uniform.

## Current resource boundary

Under `PI_GATE_STATE_0071`:

```text
repository maintenance: allowed
GR00T forward: 0
WBC rollout: 0
reference update: 0
training/backward/optimizer: 0
new data generation: 0
Vanda A40: HOLD
Hopper H200: HOLD
hardware/ROS/ZMQ/P1M: LOCKED
```

No PBS submission, GPU inspection, model construction, simulator/WBC launch,
or hardware action is part of this branch. Future execution must use a new PI
gate and the exact source/evidence identities named by that gate.

## Private runtime inputs

Launcher environment variables point to node-local model, WBC, dataset, cache,
and output roots. Keep those values out of committed configuration and logs;
use the public manifests/reports to record identity hashes only. Never commit
tokens, credentials, checkpoints, videos, datasets, USD assets, or scheduler
output.
