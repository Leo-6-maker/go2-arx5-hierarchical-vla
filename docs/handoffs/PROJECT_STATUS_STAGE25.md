# Project Status — Stage25

Date: 2026-09-03 (Asia/Singapore)

## Current branch

```text
stage25/o1-parameter-delta-gate
base: c5ded6e119df920402451497aacdc69274ca8fc1
```

## Current gate

```text
STAGE25_PREPARATION: PASS
BASE_TO_O1_NUMERICAL_DELTA: PREPARED_NOT_EXECUTED
NEXT_GATE: GPT_PI_REVIEW_AFTER_STAGE25_CORELAB_PROVENANCE_SCOPE_PATCH
```

The exact Stage23 O1@200 historical path, run manifest, trainer state,
checkpoint index, and per-file seal were recovered from the immutable Phase-A
relay. The relay contains no full O1 safetensor shards. The missing bytes are
not substituted and do not invalidate the preparation.

The Stage23 `checkpoint_file_seal.tsv` remains immutable with SHA256
`7aa07a84d62a5052bb9522ba0a04f8d1b711dac0f6b448db3e696135a663f806`. The O1
root scope is now two-layered: the exact `candidate_200` sealed core plus
exactly five regular, non-symlink supplemental paths:
`experiment_cfg/conf.yaml`, `experiment_cfg/config.yaml`,
`experiment_cfg/dataset_statistics.json`,
`experiment_cfg/final_model_config.json`, and
`experiment_cfg/final_processor_config.json`. Supplemental metadata is
`PRESENT_DAY_UNSEALED_METADATA` and is excluded from tensor identity; any
missing, extra, symlink, or escaping path fails closed before tensor reads.

## Prepared audit

`scripts/stage25_o1_parameter_delta_audit.py` has two boundaries:

- `prepare` verifies the Stage23 archive/payload seals and freezes the O1
  provenance and comparison schema, including the two-layer root scope;
- `audit` requires exact BASE/O1 trees and the O1 per-file seal, then performs
  a CPU-only streamed safetensor comparison.

The audit covers all 14 category-specific projector keys, rows 4–7/10/11/24–27,
the W1 active 10D versus inactive 122D partition, and aggregate deltas for
all serialized model keys. Expected-frozen group changes are fail-closed.

## Prohibited until next authorization

```text
Hopper/H200/PBS: false
new model construction: false
forward/inference: false
training/fine-tuning: false
WBC/native rollout: false
validation e / holdout f: false
hardware/ROS/ZMQ/P1M: false
```

See `reports/STAGE25_FINAL_REPORT.md` for the Corelab return and
`reports/STAGE25_O1_PARAMETER_DELTA_PROTOCOL.md` for the execution contract.
