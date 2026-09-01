# Stage18 Resource and Git Audit

Date: 2026-09-01 (Asia/Singapore)

## Git preflight

- Canonical repository: `/home/liuyu-nfs/Projects/go2-arx5-hierarchical-vla`.
- Promotion source: `main` merge commit
  `a016dcba0a1afa727a084aa746a8ae725068e417`.
- Working branch: `stage18/robot-data-readiness`.
- Origin: private `git@github.com:Leo-6-maker/go2-arx5-hierarchical-vla.git`.
- Stage15, Stage16, and Stage17 history was preserved; no reset or history
  rewrite was used.
- The pre-implementation worktree was clean. D3 data, WBC source/archive,
  checkpoints, images, videos, and credentials were not copied into the
  repository. Local secret-pattern audit found no staged credential material.

## Compute/runtime preflight

The read-only audit ran on `corelab-5080-2`, not the login server:

- Python 3.12.3 in the existing Stage10 runtime;
- PyTorch 2.9.0+cu128, CUDA available, GR00T import passed;
- RTX 5080, 16,303 MiB total, about 90 MiB used and 0% utilization at audit;
- 62 GiB RAM with about 59 GiB available;
- node-local filesystem had about 692 GiB free;
- D3 root exists at `/var/cache/users/liuyu-nfs/go2-arx5-data-d3`;
- no project-owned Stage18/WBC/ROS/ZMQ/PolicyServer workload was started.

The Stage10 virtual environment has no pytest executable. CPU tests therefore
used the system `/usr/bin/pytest` 7.4.4 with NumPy 1.26.4; no package was
installed.

## Authentication boundary

With the complete Stage10 startup (`env.sh` plus its venv),
`hf auth whoami` returned `leoyuer` and the gated Cosmos resource required by
the official processor was accessible. No token was requested, printed,
copied, or inspected. An earlier smoke invocation that omitted `env.sh` was
not treated as evidence; it used the wrong HF cache context and returned 401.
Authentication does not authorize model inference or real data collection.
