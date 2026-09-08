# Stage45A FULL24 adaptation protocol

Date: 2026-09-08 (Asia/Singapore)

```text
ROLE: CODEX_CORELAB
STAGE: Stage45A
BRANCH: stage45/data-expansion-design
BASE: main @ bfa14bf1b74eae908c0922575a4db671c903b9b8
PI_GATE: PI_GATE_STATE_0072
STATUS: DESIGN_ONLY_READY_FOR_PI
```

## Purpose and boundary

Stage44R accepted the narrow diagnosis
`STAGE44_TINY1_SINGLE_SAMPLE_COVERAGE_LIMITATION_SUPPORTED`: the exact
`train_a/expert-0000` cell is target-directed, while the broader Stage43
windows first fail at the decoded GR00T output. Stage45A turns the next
coverage experiment into a pre-registered, CPU/read-only protocol. It does
not train a model or run a model forward.

The return token for this design package is:

```text
CORELAB_STAGE45_FULL24_ADAPTATION_PROTOCOL_READY
```

The following remain frozen at zero for this branch:

```text
GR00T forwards: 0
WBC steps: 0
reference updates: 0
new data generation: 0
training/backward/optimizer steps: 0
GPU/simulator/PBS execution: none
Hopper H200: HOLD
Vanda A40: HOLD
Validation-E: LOCKED
Holdout-F: LOCKED
```

The machine-readable design is
`manifests/STAGE45A_FULL24_ADAPTATION_DESIGN_V1.json`. Its unresolved
`structured_action_statistics` hash is intentional: the hash must be computed
from the existing 24 training labels by a CPU preflight and sealed before any
model construction. A missing or mismatched hash blocks execution; it is not
a license to substitute TINY1 statistics or to start H200 work.

## Frozen data ladder

The ladder is deliberately narrow:

```text
Level 0  TINY1
         1 window, current accepted diagnostic baseline
             ↓
Level 1  FULL24_EXISTING_TRAIN   PRIMARY NEXT CANDIDATE
         4 train targets × 6 replanning snapshots = 24 windows
             ↓ only if a later PI gate admits it
Level 2  MULTIINIT120             PROTOCOL ONLY
         4 targets × 5 base initial states × 6 snapshots = 120 windows
```

`FULL24_EXISTING_TRAIN` is the already materialized Stage22 train-only
fixture. It is not a new simulator collection and must not be rebuilt during
Stage45A. Its authoritative Stage22 handoff identity is:

```text
canonical tree method: STAGE22_PREPARE_H200_HANDOFF_TREE_DIGEST_V1
tree SHA256:           2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789
regular files:         85
total bytes:           919661
materialized episodes: 24 one-window episodes
materialized frames:   960 (24 × H40)
source expert episodes: 4 (train_a/b/c/d)
snapshots per source episode: 6
```

The split lock is tracked at
`manifests/STAGE22_TARGET_SPLIT_LOCK_V1.json` with SHA256
`7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246`. Its
train IDs are exactly `red_target_train_a`, `red_target_train_b`,
`red_target_train_c`, and `red_target_train_d`; `red_target_holdout_e` remains
validation-only and `red_target_holdout_f` remains final-test-only.

The canonical window order is target-major, then snapshot-major:

| windows | target | world target (m) | source snapshot IDs |
| ---: | --- | --- | --- |
| 0–5 | `red_target_train_a` | `[0.30, 0.10, 0.58]` | `stage22_expert_red_target_train_a:expert-0000` … `expert-0005` |
| 6–11 | `red_target_train_b` | `[0.30, -0.10, 0.58]` | `stage22_expert_red_target_train_b:expert-0000` … `expert-0005` |
| 12–17 | `red_target_train_c` | `[0.30, 0.05, 0.64]` | `stage22_expert_red_target_train_c:expert-0000` … `expert-0005` |
| 18–23 | `red_target_train_d` | `[0.30, -0.05, 0.63]` | `stage22_expert_red_target_train_d:expert-0000` … `expert-0005` |

The manifest expands this table into all 24 rows and requires the materialized
`meta/episodes.jsonl` order to match it. A mismatch, extra holdout marker,
missing snapshot, non-40 horizon, or changed tree hash is fail-closed.

`target_position_world_m` is metadata/evaluation provenance only. The model
input remains body RGB, wrist RGB, StateV1, and the fixed language
`move the end effector to the red target`; target XYZ is not an input tensor.

## Interface and representation freeze

Stage45A does not change any accepted Stage38–Stage44R interface:

```text
StateV1                         [10]
body RGB / wrist RGB            [96,96,3] uint8
StructuredActionV1              K16 / T16 / H40 / D132
raw model output                [1,40,132]
decoded physical action         [40,10] = EEF 9D + gripper 1D
active loss dimensions          10
masked universal padding        122
source frame                    NATIVE_WBC_RELATIVE_REFERENCE
TCP semantic                    x5_link6_gripper_center
EEF transform                   T_reference^-1 @ T_absolute_target
reference                       last observed StateV1 eef_9d
rotation                        first two rotation rows, flattened
gripper                         absolute separate channel
```

The exact upstream identities remain bound to the Stage40/Stage44R records:

```text
exact BASE model tree:          8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4
STRUCTURED@1000 tree:           30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137
Stage22 stats.json:             fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23
Stage22 relative_stats.json:    8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1
Stage44 Structured stats:       0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206
GR00T source:                   51d4c89f72fda44cbf77285c6a8114b52676b8a1
Cosmos revision:                9ce19a195e423419c349abfc86fd07178b230561
Cosmos materialized tree:       1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9
modality config:                e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031
```

### Normalization rule

The primary FULL24 run has two distinct statistics domains:

1. The official Stage22 `stats.json` and `relative_stats.json` bytes are
   copied/read-only and must match their hashes above. They are not regenerated
   by this protocol.
2. Stage38-D representation-specific `StructuredActionV1` statistics are
   computed from the 24 FULL24 training labels only, with
   `MINMAX_TO_NEG1_POS1`, `use_percentiles=true`, q01/q99 selected bounds,
   inactive-mask exclusion, and no padding contribution. The CPU preflight
   writes the resulting JSON, computes its SHA256, and puts that hash in the
   execution manifest before model construction.

The Stage44 `0553436d…` statistics are a provenance reference, not an
automatic FULL24 substitute. A byte-identical result may be reused only if
the CPU preflight proves the hash and the label population are exactly the
same; otherwise the newly computed FULL24 hash is authoritative. No
outcome-dependent stats swap is allowed.

## Two pre-registered starting-checkpoint options

Both options use the same FULL24 bytes, recipe, seed, budget, and fixed
evaluation. Only the starting weights differ:

| option | starting point | optimizer/scheduler | role | interpretation |
| --- | --- | --- | --- | --- |
| A `WARM_START_FULL24` | exact `STRUCTURED@1000` tree `30b14d36…` | fresh AdamW/cosine state; `resume_from_checkpoint=false` | secondary sensitivity comparator | includes the prior 1000-step TINY1 exposure plus 2000 FULL24 steps; not a clean coverage-only claim |
| B `FRESH_BASE_FULL24` | exact BASE tree `8717e322…` | fresh AdamW/cosine state; `resume_from_checkpoint=false` | **primary clean coverage candidate** | tests FULL24 coverage from the same exact base without TINY1 pre-exposure |

Option A is weights-only warm-start by protocol. A literal Trainer resume
that carries optimizer/scheduler/RNG state is a different intervention and is
not silently substituted. Option B is the only result eligible for the first
coverage interpretation; A is retained to quantify the exposure confound if a
later PI gate separately admits it.

## Frozen Stage38 recipe and budget

The Stage38-D/Stage28 recipe is copied as a design contract, not executed:

```yaml
seed: 2222
max_steps: 2000
save_steps: [1000, 2000]
save_total_limit: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
optimizer: adamw_torch
learning_rate: 0.0001
weight_decay: 0.00001
lr_scheduler_type: cosine
warmup_steps: 10
warmup_ratio: 0.0
max_grad_norm: 1.0
bf16: true
fp16: false
tf32: true
dataloader_num_workers: 0
eval_strategy: no
resume_from_checkpoint: false
save_only_model: false
tune_projector: true
tune_diffusion_model: true
tune_visual: false
tune_llm: false
tune_vlln: false
use_relative_action: true
use_percentiles: true
num_gpus: 1
```

`checkpoint-2000` is the fixed primary endpoint. `checkpoint-1000` is a
retained descriptive checkpoint, not a loss-selected model. There is no
adaptive extension, early stopping, learning-rate change, stats replacement,
precision change, quantization, CPU offload, clipping, smoothing, or
alternate checkpoint selection after seeing an outcome.

## Evaluation and later Stage43 comparison

After a future PI gate authorizes training, the first evaluation remains
train-only and offline. At checkpoints 1000 and 2000, report actual runtime
counters and, for all 24 windows, finite/masked loss, physical action RMSE,
endpoint error, target-direction cosine/progress, path-length and speed
ratios, roughness, and initial continuity. Validation-E and Holdout-F are not
read and cannot select a checkpoint.

Only after a separate post-training gate may the selected fixed endpoint be
used for the pre-registered Stage43 comparison:

```text
old STRUCTURED@1000 vs new FRESH_BASE_FULL24 checkpoint-2000
conditions: nominal, +X 5 cm, -X 5 cm, +Y 5 cm, -Y 5 cm
same language, target, WBC, timing, seed policy, and no handoff teleport
```

The comparison reports continuous initial error, minimum error, final error,
progress, GR00T action deltas, and closed-loop integrity. It does not turn
five conditions into a general robustness claim, and it does not train on
those Stage43 conditions. A/B FULL24 results cannot be declared task success
without a separately frozen success definition.

## MULTIINIT120 is protocol-only

The future Level 2 design is recorded for completeness only:

```text
targets: red_target_train_a/b/c/d
base states: [0,0], [+0.05,0], [-0.05,0], [0,+0.05], [0,-0.05] m
snapshots per target/base state: 6
windows: 4 × 5 × 6 = 120
```

No base perturbation is generated, no simulator is opened, and no new image
or label is materialized by Stage45A. The Level 2 entry is a future protocol
placeholder only.

## H200 planning estimate (not a measurement)

These are conservative scheduling inputs derived from accepted historical
identities, not a present-time H200 observation:

```text
historical H200 total VRAM identity: 150110273536 bytes (~139.8 GiB)
largest accepted structured checkpoint tree: 20889596357 bytes (~19.5 GiB)
FULL24 source fixture: 919661 bytes (~0.9 MiB)
planned checkpoints: checkpoint-1000 and checkpoint-2000, full trainer state
storage planning reserve: >=64 GiB free before launch (checkpoint copies,
  optimizer/scheduler/RNG state, logs, and cache overhead)
PBS walltime planning upper bound: 24:00:00, inherited from Stage38 template
```

The future runtime must record actual peak allocated/reserved VRAM, host,
GPU UUID, walltime, checkpoint inventories, and optimizer state. An OOM or
identity mismatch stops the run. It may not be “fixed” by changing precision,
quantizing, offloading, reducing image resolution, changing H40/K16/T16, or
substituting a checkpoint.

## Execution manifest and seal format

Before any future model construction, a node-local preflight manifest must
conform to `CORELAB_STAGE45A_FULL24_ADAPTATION_EXECUTION_MANIFEST_V1` and bind:

```text
repository branch + exact HEAD
PI gate and predecessor Stage44R identity
FULL24 tree/hash, file count/bytes, split-lock hash
all 24 mapping rows and source snapshot hashes
BASE or STRUCTURED@1000 tree/hash (one explicit option)
Stage22 stats/relative-stats hashes
new FULL24 structured-statistics JSON/hash
GR00T source, Cosmos revision/tree, modality hash
StructuredActionV1 contract and immutable recipe
seed, max/save steps, trainable/frozen module inventory
Validation-E/Holdout-F exclusion flags
resource boundary and zero-before-execution counters
```

After a future run, the immutable result seal must conform to
`CORELAB_STAGE45A_FULL24_ADAPTATION_RESULT_SEAL_V1` and bind every result,
checkpoint inventory, runtime identity, observed optimizer-step counter,
metric table, and offline-evaluation artifact by SHA256. A declared budget is
not evidence: counters must come from the actual runtime, and any missing or
unbound artifact is `BLOCKED`.

The current branch contains only this protocol and its machine-readable
design/tests. It stops here for PI review; it does not authorize H200.
