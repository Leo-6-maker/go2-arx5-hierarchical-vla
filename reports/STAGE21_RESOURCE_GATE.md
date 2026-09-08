# Stage21 Resource and Integrity Gate

Date: 2026-09-01 (Asia/Singapore)

## Scope

Stage21 used only simulation. No hardware, ROS, ZMQ, PolicyServer, P1M,
training, finetuning, or WBC source/checkpoint modification was used.

## Nodes

| Node | Runtime/resource observation | Stage21 use |
|---|---|---|
| `corelab-4080s` | Native IsaacGym runtime; Python 3.8.20; PyTorch 2.3.1; RTX 4080 SUPER devices | Native WBC, same-scene camera, GT and model receding runners; `CUDA_VISIBLE_DEVICES=1` |
| `corelab-5080-2` | Stage10 runtime and HF auth; RTX 5080; Python 3.12.3; PyTorch 2.9.0+cu128 | Actual N1.7 inference/worker |
| `corelab-5080-1` | Not used; an unrelated user workload was present | Skipped |

The native control timestep measured in the runner was
`0.01999999955 s` (approximately 50 Hz). The Stage21 chunk period remained
the explicit `0.05 s` `TEST_ONLY_STAGE21_SIMULATION` value; it was not
promoted to a production action frequency.

At the final read-only check, all used Stage21 processes had exited. Existing
processes belonging to other users were not killed, signaled, reniced, or
attached. Ports 5550, 5556, and 5557 had no Stage21 listener. The final
process check found no `stage21`, `groot_worker`, or Stage21 mailbox process.

## Artifact roots

Raw runtime artifacts remain node-local:

```text
/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage21/
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage21/
/home/liuyu-nfs/Projects/.go2-arx5-vla-stage21-mailbox/
```

Only manifests, source, tests, and reports are intended for Git. The WBC
archive/source/checkpoints, GR00T model weights, output NPZ files, mailbox
contents, and credentials were not staged.

## Git preflight

The Stage21 branch was created from clean Stage20T tip
`58fe006a4986386b923bcef2b7a43d6f6559bbff`:

```text
stage21/same-scene-receding-horizon
origin = git@github.com:Leo-6-maker/go2-arx5-hierarchical-vla.git
```

The source tree scan found no token, private-key, password, or credential
literal. The final staged-artifact audit is repeated immediately before the
Stage21 commit.
