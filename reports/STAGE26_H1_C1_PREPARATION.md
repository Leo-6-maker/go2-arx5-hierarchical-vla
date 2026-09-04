# Stage26 H1 C1 Preparation

Date: 2026-09-04 (Asia/Singapore)

## Gate and scope

This is the Corelab preparation deliverable for PI_GATE_STATE_0002 in Issue
#11. It creates the stage26/h1-fixed-noise-local-flow branch from the
Stage25-closed base and prepares one source-mapped fixed-input local-flow
runner. It does not report a scientific H1 result.

The runner is ready for the conditional Hopper authorization only after the
static checks in this branch pass:

H1_C1_EXECUTION_READY=true
SCIENTIFIC_CONTRACT_CHANGED=false
DOWNSTREAM_AUTHORIZATION_STILL_VALID=true

The runtime remains fail-closed on every identity or source mismatch.

## Frozen identities

Stage26 base SHA:       4005a4dd1ab6c82807985ef856d340fa50acaa6c
GR00T source SHA:       51d4c89f72fda44cbf77285c6a8114b52676b8a1
dataset:                TINY1_H2STATS
dataset tree SHA256:    541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157
dataset files/bytes:    13 / 152958
stats.json SHA256:      fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23
relative_stats SHA256:  8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1
modality SHA256:        e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031
base processor SHA256:  85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be
base statistics SHA256: c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9
Cosmos revision:        9ce19a195e423419c349abfc86fd07178b230561
Cosmos cache tree:      1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9
selected snapshot:      stage22_expert_red_target_train_a/expert-0000.json

The runtime verifies the selected snapshot relative path, exact source
episode/example IDs, H40 finite label arrays, six-file train-a order, and
records the selected snapshot SHA256. It reads no validation or holdout
snapshot directory. The TINY fixture manifest rejects symlinks and binds the
single train episode to stage22_expert_red_target_train_a:expert-0000.

## Exact C1 input contract

The runner calls the existing Stage23 Phase-C static protocol and refuses any
probe-grid drift:

seeds:       2223, 2224, 2225
quantiles:   0.10, 0.50, 0.90
t values:    0.06776241796562847, 0.36966943557751086, 0.7837719744658148
t buckets:   67, 369, 783
noise:       CPU torch.Generator, shape [1,40,132], float32
trajectory:  x_t=(1-t)*noise+t*expert_action
target:      v_target=expert_action-noise
mask:        first 10 active; last 122 remain in input and have mask zero
pairing:     one canonical noise and one t reused by BASE and O1@200

The action sample is obtained through the official
LeRobotEpisodeLoader -> extract_step_data -> Gr00tN1d7Processor -> collator
path. The processor-normalized H40x132 action is decoded back to the exact
selected source snapshot label before either checkpoint is constructed.

## Source equivalence

The direct helper mirrors only these pinned N1.7 training-forward regions:

| Operation | Pinned source | Runner behavior |
|---|---|---|
| checkpoint load | gr00t/model/gr00t_n1d7/setup.py:75-89 | official AutoModel.from_pretrained, local-only |
| inference precision | Stage22 profile evaluator:358-364 | eval then model.to(device, dtype=torch.bfloat16) |
| visual conditioning | gr00t/model/gr00t_n1d7/gr00t_n1d7.py:598-599 | one backbone pass per checkpoint |
| state conditioning | gr00t/model/gr00t_n1d7/gr00t_n1d7.py:209-217 | reshape and category-specific state encoder |
| local flow input | gr00t/model/gr00t_n1d7/gr00t_n1d7.py:237-239 | fixed noisy action and fixed discrete time |
| DiT path | gr00t/model/gr00t_n1d7/gr00t_n1d7.py:241-270 | position embedding, fusion, exact branch, hidden-state return |
| predicted velocity | gr00t/model/gr00t_n1d7/gr00t_n1d7.py:272-278 | action decoder and final H40 slice |

The helper does not call the complete model forward, the action-head
free-running sampler, Euler integration, backward, or an optimizer. The
backbone conditioning pass is shared by the nine action-head probes for each
checkpoint. The scientific count is exactly two checkpoints x nine direct
teacher-forced flow probes = 18.

## Output and interpretation boundary

Each of the 18 rows records canonical noise/noisy-action/target hashes, the
model-input hash, raw predicted-velocity hash and shape, active masked MSE,
inactive prediction/target energy, implied universal-action hash, active
model-action hash, decoded physical-action hash, profile metrics, SafetyGate
transition data, and finite flags.

The paired summary reports BASE and O1 MSE for all nine probes, O1/BASE ratios,
improved count, median ratio, and geometric-mean ratio. These are descriptive
measurements only. No new numerical scientific PASS threshold is introduced;
GPT/PI classifies the result as local-flow improvement, persistent local-flow
failure, or regime-specific improvement.

Decoded profiles use GR00TTrajectoryProvider and TrajectorySafetyGateV1 with
the already-qualified NATIVE_WBC_RELATIVE_REFERENCE /
x5_link6_gripper_center contract. No WBC or hardware call exists in the
runner.

## Execution command contract

The default mode is non-executing:

python scripts/stage26_h1_c1_fixed_noise_local_flow.py --self-test

The conditional Hopper invocation must supply the exact node-local paths and
an explicit CUDA device:

python scripts/stage26_h1_c1_fixed_noise_local_flow.py \
  --execute \
  --device cuda:0 \
  --stage10-root /var/cache/users/liuyu-nfs/go2-arx5-vla-stage10 \
  --groot-source /var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/src/isaac-gr00t \
  --base-checkpoint /scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B \
  --o1-checkpoint /scratch/e1192548/stage23_o1_20260903/runs/597966.hopper-m-02/candidate_200/checkpoint-200 \
  --dataset-dir /var/cache/users/liuyu-nfs/go2-arx5-vla-stage23/action_pipeline_20260903_p0/tiny1_h2stats_lerobot_v2 \
  --snapshot-root /var/cache/users/liuyu-nfs/go2-arx5-wbc-stage22/expert_dataset_v1/snapshots \
  --modality-config /scratch/e1192548/stage22_handoff_deb920a_20260902/payload/modality/go2_arx5_modality.py \
  --output-dir /var/cache/users/liuyu-nfs/go2-arx5-vla-stage26/h1_c1

This command is not run by Corelab preparation. Hopper may run it only under
the separate conditional forward-only gate. It must stop after writing
stage26_h1_c1_result.json and stage26_h1_c1_hash_manifest.json.

## Forbidden operations

training: false
backward: false
optimizer_step: false
PBS: false
iterative_sampler: false
C2: false
WBC: false
hardware: false
validation_e_read: false
holdout_f_read: false
inactive_noise_intervention: false
architecture_pivot: false
