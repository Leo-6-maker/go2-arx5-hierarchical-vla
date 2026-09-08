# Stage22 Expert Dataset Materialization

The six-episode native snapshot collection was materialized through the
existing RawEpisode → Synchronizer → logical training example → LeRobot-v2
builder path. Only small derived simulation artifacts were transferred to the
model node; WBC source/checkpoints and model weights remain node-local.

## Split and counts

```text
locked train targets:      red_target_train_a/b/c/d
locked validation target:  red_target_holdout_e
locked final-test target:  red_target_holdout_f
historical all-target fixture: 36 one-window episodes (not for training)
locked train-only fixture:     24 one-window episodes
locked train-only frames:      960
valid H=40 windows in train:   24
state ABI:          StateV1 [10]
desired action:     [40,10] = EEF [40,9] + gripper [40,1]
achieved as label:  false
target in model input: false
```

The snapshot-expanded horizon is explicit metadata. Repeating one exact
native observation over a 40-point fixture is a loader/materialization test;
it is not claimed to be a continuous native video recording.

The tracked split lock is
`manifests/STAGE22_TARGET_SPLIT_LOCK_V1.json`. The training fixture carries
`meta/stage22_split_lock.json` with `dataset_role=TRAIN`, and the Stage22
training launcher refuses the historical all-target fixture.

## Official loader path

The Stage10-compatible official Isaac-GR00T path passed on CPU:

```text
official episode loader: LeRobotEpisodeLoader
official training dataset: ShardedSingleStepDataset
official processor: Gr00tN1d7Processor
episode boundaries: preserved
single-step finite: true
physical action: [40,10]
universal action: [1,40,132]
decoded physical action: [1,40,10]
round-trip max abs error: 3.584885e-08
training/finetuning performed: false
```

The optional native `lerobot` package itself was not used for package-level
dataset construction; the official GR00T loader/collator/processor path over
the materialized v2 fixture was qualified. This distinction remains explicit.
