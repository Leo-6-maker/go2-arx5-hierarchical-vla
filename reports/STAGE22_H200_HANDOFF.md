# Stage22 H200 Handoff

Date: 2026-09-01 (Asia/Singapore)

> Historical packaging snapshot: this handoff records the original frozen
> `deb920a` transfer package. H1 has since passed on H200 and H2 is
> running/dispatched; current status and direct-launch audit rules are tracked
> in `reports/STAGE22_H2_TOOLING.md` and the Stage22 PR ledger.

## Frozen handoff record

```text
ROLE: CODEX_CORELAB
STAGE: Stage22 H200 handoff packaging
GIT_SHA: deb920a070141e6a32a88cd0da6719828647e1e6
GR00T_SHA: 51d4c89f72fda44cbf77285c6a8114b52676b8a1
CPU_TESTS: 76 passed in 0.24s
HANDOFF_ROOT: /var/cache/users/liuyu-nfs/go2-arx5-vla-stage22/h200_handoff_deb920a_20260901
PBS_TEMPLATES_PREPARED: YES; NOT SUBMITTED
TRAIN_SPLIT: red_target_train_a,b,c,d
VALIDATION_SPLIT: red_target_holdout_e
LOCKED_TEST_SPLIT: red_target_holdout_f
RESULT: CODE_DATA_INPUT_FROZEN_FOR_H200
BLOCKER: NONE_FOR_CORELAB_PACKAGING; HOPPER_LOGIN_AND_PROJECT_DISCOVERY_PENDING_CODEX_HOPPER
NEXT_GATE_REQUEST: CODEX_HOPPER_R1_1STEP_PREFLIGHT
```

## Input artifact SHA256

The directory hashes are deterministic hashes over sorted relative file paths,
file sizes, and per-file SHA256 values. The handoff manifest is the authority.

```text
stage22_repo_deb920a07014.tar.gz
  ed96ff0302ea5f9b85a52b4b071fc641f9d93cbfeac6433bc4fc06dd70bf7fbc
train fixture tree
  2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789
GR00T-N1.7-3B tree
  8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4
go2_arx5_modality.py
  e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031
STAGE22_TARGET_SPLIT_LOCK_V1.json
  7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246
```

The complete per-file checksum list is node-local at
`manifests/SHA256SUMS.sha256` under `HANDOFF_ROOT`; it was verified with
`sha256sum -c` after packaging.

## Transfer files

```text
stage22_repo_deb920a07014.tar.gz
payload/lerobot_train/
payload/GR00T-N1.7-3B/
payload/modality/go2_arx5_modality.py
pbs/stage22_h200_r1_preflight.pbs
pbs/stage22_h200_r0_r1_50.pbs
manifests/STAGE22_H200_HANDOFF_V1.json
manifests/SHA256SUMS.sha256
```

The package contains the locked train-only fixture (24 episodes, 960 frames)
and the validated base model. It does not contain the Stage10 runtime itself;
Hopper runtime construction remains a separate step.

## What was executed

- Verified the clean Stage22 branch and frozen Git SHA.
- Verified the train-only split sidecar and target split lock.
- Created a Git archive containing tracked source only.
- Copied the train fixture, base model, modality module, and PBS templates to
  node-local scratch.
- Recorded Git, GR00T, tree, file, and template hashes.
- Verified every transfer-file checksum after packaging.
- Scanned packaged names for credentials/private keys and confirmed no
  symlinks.

## What was not executed

- No Hopper login or `amgr login`/`hpc project` command.
- No PBS submission or compute-node allocation.
- No GPU training, R0/R1 optimizer step, or inference.
- No WBC execution, hardware, ROS, ZMQ, P1M, or real-data path.
- No password, HF token, private SSH key, WBC archive/source/checkpoint, or
  real-robot artifact was copied.

The next executor must obtain the real Hopper project ID interactively and
perform only the H1 one-step R1 preflight before considering the 50-step R0/R1
comparison.
