# Stage22 Resource Gate

Date: 2026-09-01 (Asia/Singapore)

## Scope

Stage22 is simulation-only. The planned R0/R1 comparison uses the existing
Stage10 official N1.7 runtime and the materialized native same-scene expert
fixture. No hardware, ROS, ZMQ, P1M, WBC training, or real data path is used.

## Observed preflight

| Node | GPU state at preflight | Stage10 runtime | Decision |
|---|---|---|---|
| `corelab-4080s` | GPU0 RTX 4080 SUPER 32.7 GiB had another user's workload; GPU1 16.4 GiB was mostly free | native WBC runtime only; Stage10 GR00T venv absent | native expert/oracle only; no GR00T training |
| `corelab-5080-1` | RTX 5080 16.3 GiB had another user's workload and another user's policy process | Stage10 path absent | not used |
| `corelab-5080-2` | RTX 5080 16.3 GiB had another user's Isaac training using about 10.7 GiB | Stage10 venv and HF auth available | not used while occupied |

The other users' processes were not killed, signalled, reprioritized, or
attached to. A fresh preflight at `2026-09-01T19:16:50+08:00` found the same
occupancy. The machine-readable gate was then run against 5080-2 and returned
`BLOCKED_OCCUPIED_GPU` before model construction; its node-local output is
`/var/cache/users/liuyu-nfs/go2-arx5-vla-stage22/r1_preflight_occupied_20260901`.
Known 4090 aliases were also checked read-only but were not usable in that
check (one public-key authorization failure and one host-key verification
failure), so they were not treated as available resources.

The Stage22 launcher has a mandatory R1 one-step preflight before any full R1
run. No model training or GPU inference has been run while no compatible free
GPU/runtime pair was available.

## Available CPU evidence

The Stage22 expert collection, dataset materialization, official loader,
collator, and N1.7 processor qualification were completed on CPU in the
Stage10-compatible environment. This does not substitute for model training
or GPU inference.

## Gate status

```text
R1_1_STEP_MEMORY_QUALIFICATION: PENDING_FREE_COMPATIBLE_GPU
R0_PROFILE_EVALUATION: PENDING_MODEL_CHECKPOINT
R1_PROFILE_EVALUATION: PENDING_MODEL_CHECKPOINT
```

When a compatible GPU is free, re-run a fresh `nvidia-smi`/process/disk/RAM
preflight, then invoke `scripts/stage22_train_recipes.py` with a new node-local
output root. Do not use a busy device merely because it reports some free
bytes.
