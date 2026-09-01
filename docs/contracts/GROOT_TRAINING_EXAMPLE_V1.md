# GR00TTrainingExampleV1

`GR00TTrainingExampleV1` is a derived model-facing view of a
`RawEpisodeV1`.

## Primary ABI

```text
StateV1 = current EEF XYZ (3)
         + current EEF GR00T_FIRST_TWO_ROWS rot6d (6)
         + current gripper (1)
         = 10D
```

The label is:

```text
future desired absolute EEF XYZ + rot6d rows [H,9]
+ future desired gripper [H,1]
+ future point_timestamps [H]
```

`H` is a dataset/configuration value. The D3 fixture uses `H=40` only as a
`TEST_ONLY_D3_PIPELINE_FIXTURE` value. It is not a production action-rate
claim. The derived record can be represented as a LeRobot-compatible logical
row with a first action plus an explicit `future_action` trajectory; the full
raw episode remains authoritative.

The achieved EEF and full whole-body state stay in the raw record for audit,
quality analysis, and later label review. They are not silently used as the
desired training label.
