# Stage23 Action-Pipeline Contract Audit

Date: 2026-09-03 (Asia/Singapore)

## Final status

```text
STAGE23_ACTION_PIPELINE_AND_EVALUATOR_SANITY_PASS
```

The Stage23 action contract is internally consistent on the frozen Stage22
simulation train fixture. The audit did not train, fine-tune, run model
inference, execute WBC, read validation target `e`, or read locked final-test
target `f`.

The official processor qualification initially exposed two environment issues:
the local Cosmos snapshot was not selected by the default loader call, and
the Stage10 shell did not expose a compatible FFmpeg library path for
`torchcodec`. The final qualification used the already-frozen Stage10 Cosmos
snapshot with `local_files_only=true` and the existing Stage13 public native
FFmpeg libraries. No token, network download, or source/model modification
was used.

## Frozen physical contract

```text
StateV1                 = EEF XYZ 3 + EEF row-rot6d 6 + gripper 1 = 10D
physical action         = EEF XYZ + row-rot6d + gripper = 10D
EEF action              = absolute future EEF pose, processed as relative
gripper                 = absolute, separate 1D channel
universal action        = [H=40, D=132]
active loss dimensions  = 10
masked padding          = 122
production action rate  = TBD_BY_ADAPTATION_DATASET
simulation period       = 0.05 s, TEST_ONLY_STAGE22_SIMULATION
```

The action period is metadata/provenance for this simulation fixture. It is
not a model input and is not promoted to a production Go2+ARX5 action-rate
decision.

## Transform trace

For each action window, the reference is the last observed `state.eef_9d`
sample in the current observation window. The first future action is therefore
relative to that observation reference, not relative to the first future
action and not a frame-to-frame delta.

For reference pose `T_ref` and absolute future pose `T_abs[k]`, the official
operation is:

```text
T_rel[k] = inverse(T_ref) @ T_abs[k]
```

Equivalently, for position and rotation:

```text
p_rel[k] = R_ref.T @ (p_abs[k] - p_ref)
R_rel[k] = R_ref.T @ R_abs[k]
```

The same fixed `T_ref` is used for every one of the 40 points. The official
`delta_chunking` implementation exists as a separate frame-to-frame utility,
but it is not the configured EEF training path.

The rotation representation is the first two rows of the 3x3 rotation matrix,
flattened row-major as six values. The official decoder normalizes the first
row, Gram-Schmidt orthogonalizes the second row, and forms the third row with
a cross product. The gripper is not included in this SE(3) transform.

## Normalization and loss trace

The EEF relative action uses the generated `relative_action[eef_9d]`
statistics. The gripper uses the ordinary absolute-action statistics. The
official path uses min/max normalization unless a configured mean/std embedding
is selected, then applies the existing outlier clipping policy. Stage23 now
records normalization-statistics provenance explicitly. The primary
future O1 fixture, `TINY1_H2STATS`, copies the exact frozen Stage22 train-fixture
`stats.json` and `relative_stats.json` bytes. The retained
`TINY1_TINYSTATS` and `TINY6_TINYSTATS` variants deliberately regenerate
statistics from their respective tiny materializations; they are controlled
diagnostics, not the primary O1 dataset.

The official N1.7 processor concatenates the two physical action keys in
modality order, producing 10 columns, then pads to 132 columns and horizon
40. Its action mask has the same shape. The model loss is elementwise MSE
multiplied by this mask and divided by the mask sum.

The numerical audit recovered:

| quantity | result |
|---|---:|
| universal shape before batch | `[40,132]` |
| universal shape after processor batch | `[1,40,132]` |
| decoded physical shape | `[1,40,10]` |
| active physical dimensions | 10 |
| physical active elements in one H=40 window | 400 |
| padding dimensions | 122 |
| active padding elements | 0 |
| padding dimensions contributing to loss | 0 |

Thus the 122 padding dimensions do not contribute to the training loss under
the audited modality and processor path.

## Time and ordering semantics

The configured action indices are ordered `0..39`. The materialized parquet
timestamp column is retained as source provenance and has strictly increasing
values with median period `0.05 s` in every selected episode. The timestamp is
not included as a state/action model tensor by the Stage22 modality config.

## Immutable source evidence

All source evidence below is read-only and bound to GR00T SHA
`51d4c89f72fda44cbf77285c6a8114b52676b8a1`.

| semantic item | source | evidence |
|---|---|---|
| reference is last observed state | `gr00t/data/state_action/state_action_processor.py` | lines 365–397 |
| relative EEF conversion | same file | lines 639–664 |
| relative normalization statistics | same file | lines 179–204 |
| min/max or mean/std normalization | same file | lines 396–417 |
| fixed-reference action chunk | `gr00t/data/state_action/action_chunking.py` | lines 82–109 |
| frame-to-frame delta is separate | same file | lines 111–141 |
| relative-to-absolute composition | same file | lines 624–658 |
| row-packed rot6d | `gr00t/data/state_action/pose.py` | lines 426–468 |
| relative statistics loader | `gr00t/data/stats.py` | lines 294–360, 415–435 |
| physical concatenation and padding mask | `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` | lines 580–640 |
| masked loss | `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | lines 272–278 |
| project horizon, representation, and timing | `scripts/stage20s_groot_modality.py` | lines 13–41 |

The project modality file is a Stage22 simulation fixture contract. The
official repository and source paths above remain external runtime inputs and
are not copied into this repository.

## Stage22 source identity and stats lock

The source fixture was rehashed with the exact Stage22 H200 handoff method,
not with the older Stage23 local manifest method:

```text
method: STAGE22_PREPARE_H200_HANDOFF_TREE_DIGEST_V1
entries: payload/lerobot_train/<relative>\0<bytes>\0<file_sha256>\n
files: 85
bytes: 919661
symlinks: 0
canonical Stage22 SHA256: 2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789
matches frozen identity: true
```

The earlier `c371b07183d238ffe60b748ab50dbf811b676cd8d572760e934598e053893ad2`
value is retained as a Stage23 local-manifest diagnostic. It uses a different
field order and no `payload/lerobot_train/` prefix, so it is not comparable to
the Stage22 handoff digest and is not used as the frozen source identity.

The frozen source statistics are:

```text
stats.json:          fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23
relative_stats.json: 8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1
```

The P0 rerun produced the following explicit modes:

| fixture | statistics mode | qualification | stats identity |
|---|---|---|---|
| `TINY1_H2STATS` | `STAGE22_FROZEN_COPIED` | PASS | byte-identical to Stage22 source |
| `TINY1_TINYSTATS` | `TINY_FIXTURE_REGENERATED` | PASS | intentionally different tiny-fixture stats |
| `TINY6_TINYSTATS` | `TINY_FIXTURE_REGENERATED` | PASS | intentionally different tiny-fixture stats |

## Conclusion

The diagnostic found no evidence that the Stage22 profile blocker was caused
by a mistaken absolute/relative transform, wrong rotation ordering, gripper
concatenation, timestamp reordering, or active universal padding. These
findings do not prove that GR00T can learn the task. They establish that a
future learnability experiment would be testing model/data learnability rather
than an already-detected action-contract mismatch.
