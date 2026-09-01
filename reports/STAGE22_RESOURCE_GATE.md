# Stage22 Resource Gate

Date: 2026-09-01 (Asia/Singapore)

## Scope

Stage22 is simulation-only. The planned R0/R1 comparison uses the existing
Stage10 official N1.7 runtime and the materialized native same-scene expert
fixture. No hardware, ROS, ZMQ, P1M, WBC training, or real data path is used.

## Observed preflight

| Node | GPU state at preflight | Stage10 runtime | Decision |
|---|---|---|---|
| `corelab-4080s` | GPU0 RTX 4080 SUPER 32.7 GiB had another user's workload; GPU1 16.4 GiB was mostly free | initially native WBC runtime only; a node-local validated Stage10 copy plus private FFmpeg compatibility path was later staged for the preflight | native expert/oracle only; R1 preflight attempted on GPU0, no full training |
| `corelab-5080-1` | RTX 5080 16.3 GiB had another user's workload and another user's policy process | Stage10 path absent | not used |
| `corelab-5080-2` | RTX 5080 16.3 GiB initially had another user's Isaac training using about 10.7 GiB; later clean recheck had only 90 MiB used | Stage10 venv and HF auth available | shared preflight attempted after clean recheck; no full training |

The other users' processes were not killed, signalled, reprioritized, or
attached to. The original empty-GPU policy was superseded by explicit user
authorization to share a GPU when free VRAM remains. The preflight now accepts
`--allow-shared-gpu`, but still requires no OOM and at least
`max(15% total, 2 GiB)` device-level headroom.

The first 5080-2 shared attempt returned `CUDA_OOM` during `model.to(device)`:
peak device usage was `15,810/16,303 MiB`, with `493 MiB` headroom. It did not
reach an optimizer step. Node-local output:
`/var/cache/users/liuyu-nfs/go2-arx5-vla-stage22/r1_preflight_shared_20260901_1930`.

The 4080S GPU0 shared attempt used a staged copy of the already validated
Stage10 runtime/model/checkpoint and a private FFmpeg-6 compatibility library
path (the target system lacked two FFmpeg libraries). Torchcodec then imported
successfully, but R1 reached `optimizer.step()` and returned `CUDA_OOM`:
peak device usage was `32,163/32,760 MiB`, with `597 MiB` headroom versus the
`4,914 MiB` requirement. Node-local output:
`/var/cache/users/liuyu-nfs/go2-arx5-vla-stage22/r1_preflight_4080_shared_ffmpeg6_20260901`.
In both attempts the external post-run residual was `0 MiB` and the other
user's process remained unchanged.

After the later clean recheck on `corelab-5080-2`, the same one-step preflight
was rerun with the existing Stage10 `env.sh` and canonical modality module.
There was no pre-existing compute app (`90 MiB` baseline), but R1 still
reached `optimizer.step()` and failed with `CUDA_OOM` while allocating a 20 MiB
Adam state tensor. Peak device usage was `15,792/16,303 MiB`, leaving `511 MiB`
versus the `2,445 MiB` requirement; PyTorch peak allocated/reserved were
`14,849.7/15,362.0 MiB`. The machine-readable result records
`failure_classification=CUDA_OOM`, `optimizer_step_observed_in_traceback=true`,
and `optimizer_step_completed=false`:
`/var/cache/users/liuyu-nfs/go2-arx5-vla-stage22/r1_preflight_5080_2_clean_env_20260901`.
The external post-run residual returned to `0 MiB`.

A subsequent 4080S GPU0 recheck found substantial apparent free memory
(`31,603/32,760 MiB` free at the read-only check, `0%` utilization), with an
existing non-Stage22 `nxnode` allocation of `266 MiB`. Using the staged
Stage10 environment, private FFmpeg compatibility path, and matching modality
module, the R1 preflight constructed the model, loaded the train-only fixture,
and entered `optimizer.step()`, but failed in multi-tensor Adam while trying
to allocate `288 MiB`. Peak device usage was `28,146/32,760 MiB`, leaving
`4,614 MiB` versus the `4,914 MiB` requirement; PyTorch peak
allocated/reserved were `30,314.7/31,120.0 MiB`. The result records
`failure_classification=CUDA_OOM`,
`optimizer_step_observed_in_traceback=true`, and
`optimizer_step_completed=false`:
`/var/cache/users/liuyu-nfs/go2-arx5-vla-stage22/r1_preflight_4080_clean_20260901`.
The post-run device state returned to the `627 MiB` baseline and the external
process was unchanged.

Known 4090 aliases were also checked read-only but were not usable in that
check (one public-key authorization failure and one host-key verification
failure), so they were not treated as available resources.

The Stage22 launcher has a mandatory R1 one-step preflight before any full R1
run. The shared attempts did not qualify, including one clean 16 GiB attempt
and the later nearly-free 32 GiB 4080S attempt; no full R0/R1 adaptation or
model-profile inference has been run.

## Available CPU evidence

The Stage22 expert collection, dataset materialization, official loader,
collator, and N1.7 processor qualification were completed on CPU in the
Stage10-compatible environment. This does not substitute for model training
or GPU inference.

## Gate status

```text
R1_1_STEP_MEMORY_QUALIFICATION: BLOCKED_INSUFFICIENT_HEADROOM_OR_OOM_ON_TESTED_GPUS
R0_PROFILE_EVALUATION: PENDING_MODEL_CHECKPOINT
R1_PROFILE_EVALUATION: PENDING_MODEL_CHECKPOINT
```

When any compatible GPU has enough remaining headroom, re-run a fresh
`nvidia-smi`/process/disk/RAM preflight and invoke
`scripts/stage22_r1_preflight.py --allow-shared-gpu` with a new node-local
output root. Do not start full training from a preflight with OOM or
insufficient headroom.
