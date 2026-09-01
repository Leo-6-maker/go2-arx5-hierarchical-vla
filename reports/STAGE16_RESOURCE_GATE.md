# Stage16 Resource Gate

Date: 2026-09-01 (Asia/Singapore)

## Git and execution gate

- Canonical repository: `/home/liuyu-nfs/Projects/go2-arx5-hierarchical-vla`
- Starting branch: `stage15/native-reference-binding`
- Stage15 commit: `e676c150a7e7aa406ba44aa07752441bab33b0fb`
- Stage16 branch: `stage16/native-wbc-benchmark`
- Remote: no Git remote configured; recorded as `GITHUB_REMOTE_PENDING`.
- The Stage15 worktree was clean before Stage16 changes. Stage15 reports were not modified.

The first local shell attempt returned `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted` before SSH or the remote command started. It was treated as a local command-sandbox failure, not as evidence about the compute node. A later network-capable shell completed the bounded preflight.

## Selected compute node

Native IsaacGym was selected on `corelab-4080s` because its validated runtime was present and the GPU was idle at the gate. `corelab-5080-1` and `corelab-5080-2` were inspected but were not used: the Stage13 native runtime was absent there and other-user Python workloads were active. No other-user process was killed, signaled, reprioritized, or attached to.

Observed selected-node facts:

| Item | Observation |
|---|---|
| Hostname | `corelab-4080s` |
| Python | 3.8.20 (`native_py38_4080/bin/python`) |
| PyTorch / CUDA | torch 2.3.1 / CUDA 12.1 |
| IsaacGym | native import succeeded |
| GPU | 2 x RTX 4080 SUPER; selected visible device was idle at gate |
| RAM | 62 GiB total, approximately 56 GiB available |
| Disk | approximately 1.8 T total, approximately 1.3 T available |
| Other-user load | no active compute application on selected GPU at gate |
| Stage16 output | `/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage16` |

The run environment used the supplied native IsaacGym Python path, compatibility path, native Python environment, `LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6`, and `CUDA_VISIBLE_DEVICES=1`. No camera, writer, ROS, ZMQ, PolicyServer, P1M, or GR00T process was started.

## Immutable input gate

The following hashes were checked on the selected compute node before experiments and again during final integrity checks:

| Input | SHA256 |
|---|---|
| `WBCTrackingShare_20260831.zip` | `b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167` |
| native `go2_x5.urdf` | `96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06408462e0e49a8a2067e25` |
| arm `ac_weights_039999.pt` | `99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a` |
| dog `ac_weights_039999.pt` | `51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670` |
| native `wbc.py` | `56132d4ff9b866ce196bbc1b1353900b967c54f0f48d2d4e4b82c3555769a0bb` |
| P1M canonical USD | `92d64f697d3f5ea6f17bbeaab72142607019f0309b5068c47f37a623499b9a53` |

Two intentionally aborted preparation attempts supplied an incorrect config/checkpoint path. The hash gate rejected both before output generation; no immutable file or experiment result was changed.

## Verification status

- Stage16 Python modules/scripts: `py_compile` passed.
- Patch whitespace: `git diff --check` passed.
- Pytest was not available in the selected Python (`No module named pytest`); no package was installed. Runtime smoke checks and native benchmark gates were used instead.
- Node-local outputs, logs, manifests, NPZ files, checkpoints, and source archives remain outside Git.

The shared Isaac-GR00T checkout was outside the Stage16 scope. Its final read-only check showed pre-existing modifications in `CLAUDE.md` and `getting_started/GR00T_inference.ipynb`; Stage16 did not touch those files. The expected `/home/liuyu-nfs/Projects/sonic-core` checkout was not present, and no SONIC process was started.
