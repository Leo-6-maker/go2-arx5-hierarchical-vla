# Stage17 Actual GR00T Inference

Date: 2026-09-01 (Asia/Singapore)

## Result

`PASS`

Exactly one approved full custom N1.7 inference was run through the validated
Stage10 custom-embodiment path. No simulator, WBC, camera service, ROS,
transport, or task-success evaluator was started by this run.

## Model and processor provenance

- host: `corelab-5080-2`
- GR00T repository SHA: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`
- model identity: `nvidia/GR00T-N1.7-3B`
- safetensors index SHA256: `407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59`
- processor config SHA256: `85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be`
- custom modality: `NEW_EMBODIMENT`, `eef_9d[9] + gripper_position[1]`
- device: CPU; the GPU was intentionally not used because another user's
  workload was active
- Python / PyTorch / CUDA build: `3.12.3` / `2.9.0+cu128` / `12.8`

## Input provenance

- dataset: `/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/datasets/go2_arx5_synthetic`
- episode: `0`, first frame
- language: `synthetic straight reaching test trajectory`
- state: `[0.28, -0.05, 0.32, 1, 0, 0, 0, 1, 0, 0.15]` (stored float values
  are preserved in the node-local manifest)
- external RGB and wrist RGB: first frames from the Stage10 synthetic MP4s;
  there was no live camera
- image SHA256: external
  `a5705aaeaec3d789fb716edbb2fcc72f24cd49e459a7e3e5bcaee6e536a43706`, wrist
  `1360e966e06a141391d7f0b90cc286ee76275b1de6e4782579ae92909ee5e169`
- input shapes: external/wrist `[1,96,128,3]`, state `[1,1,10]`

## Output and performance

| Item | Result |
|---|---:|
| raw universal output | `[1,40,132]` |
| decoded physical output | `[1,40,10]` |
| raw output SHA256 | `c4ecb7459fa5ccfad2be6054f38e64fdf21b3b044c26fdbc2c3d96ebe38f49f6` |
| decoded output SHA256 | `a361e4fea21c5ad817167e10226fe46a936cdd630b4f509baf2c9301b23865dc` |
| model load | `200.957005 s` |
| inference latency | `27.130034 s` |
| peak GPU allocation/reservation | `0 / 0 MiB` |

The decoded output is stored only as the node-local artifact
`/var/cache/users/liuyu-nfs/go2-arx5-vla-stage17/actual_groot/actual_groot_output.npz`.
The Git repository receives no model output or model-private files; only
provenance manifests and reports are tracked.

## Interpretation

The model output was finite and had the expected universal and custom physical
shapes. This is a runtime closure result, not task evidence. The trajectory
was subsequently audited and rejected by the Stage17 safety gate under
explicit test-only speed thresholds; it was never executed through WBC.

`task_performance_evaluated: false`
