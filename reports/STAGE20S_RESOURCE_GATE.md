# Stage20S Resource and Safety Gate

Date: 2026-09-01 (Asia/Singapore)

## Scope

Stage20S was executed as a simulation-only integration stage. No physical
Go2/ARX5 connection, motor command, teleoperation, hardware ROS, ZMQ,
PolicyServer, or real data recording was started.

## Git and source anchors

- parent Stage19 commit: `33760f738029dda225ec91f49fb26fa2e18965db`;
- working branch: `stage20s/sim-vla-wbc-closed-loop`;
- private origin: `git@github.com:Leo-6-maker/go2-arx5-hierarchical-vla.git`;
- Stage10 GR00T source: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`;
- node-local output root: `/var/cache/users/liuyu-nfs/go2-arx5-vla-stage20s/`.

The Stage20S branch contains only project-owned contracts, fixtures, launch
helpers, tests, and reports. Model checkpoints, videos, parquet datasets,
native WBC source, private lab archives, and credentials remain node-local or
external.

## Runtime selection

| Workload | Node/runtime | Result |
|---|---|---|
| GR00T inference and bounded adaptation | `corelab-5080-2`, Python 3.12.3, PyTorch 2.9.0+cu128, CUDA 12.8, RTX 5080 | PASS; peak model allocation about 6089 MiB |
| native IsaacGym WBC | `corelab-4080s`, native Python 3.8, PyTorch 2.3.1, native IsaacGym | PASS for known-good orchestration and executed model candidate |
| login node | not used for GPU-heavy work | PASS |

The 4080S native run used the delivered WBC checkpoints at the Stage13
external paths. The checkpoint hashes were checked against the Stage16
integrity record. The runtime used the existing native IsaacGym installation;
`flash_attn` was unavailable and the GR00T runtime used its documented SDPA
fallback. No installation or modification of the shared runtime was made.

The native node had unrelated user processes visible during preflight. They
were not killed, signalled, reniced, or inspected beyond resource accounting.
No Stage20S process remained after each run.

## Authentication

Stage10 Hugging Face authentication on `corelab-5080-2` was verified with
`hf auth whoami` as user `leoyuer`. The token was not requested, printed,
copied, inspected, or stored by Stage20S.

## Resource policy

No H200 was used: the bounded projector/state/action-encoder adaptation fit on
the available 5080. The training recipe is explicitly resource-bounded and is
not presented as the full-memory official adaptation recipe.
