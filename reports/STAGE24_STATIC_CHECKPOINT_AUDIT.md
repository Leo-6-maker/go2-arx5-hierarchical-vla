# Stage24 A0/A1 — Exact Checkpoint and Static Coupling Audit

Date: 2026-09-03 (Asia/Singapore)

## Result

```text
A0_EXACT_CHECKPOINT: PASS
A1_STATIC_10D_122D_AUDIT: PASS_WITH_STATIC_CLAIM_BOUNDARY
```

The exact Stage10 base checkpoint is now readable from `corelab-5080-2`.
The audit used CPU `safe_open` reads of the exact safetensor rows only. It did
not construct a model, open CUDA, run forward/inference, train, or modify the
checkpoint. Machine-readable evidence is in
`reports/STAGE24_STATIC_CHECKPOINT_AUDIT.json`.

## Frozen identity

| item | value |
|---|---|
| checkpoint | `/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/assets/GR00T-N1.7-3B` |
| index SHA256 | `407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59` |
| full tree SHA256 | `8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4` |
| tree files / bytes | `82 / 6,927,373,529` |
| shard 1 SHA256 | `8a1a1d8a33c99103c7c80c136073c5bb8bfe9ca8f7a970c93c033ea89742906d` |
| shard 2 SHA256 | `c3f61940deb2007ba1ad7743013b57f0f8462356151db9655175d7aca2d40661` |
| GR00T source | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` |

Both the index and every file in the expected checkpoint tree matched the
frozen identity. The older A0 recovery report remains historical evidence of
the earlier access blocker; it is superseded by this report.

## ID10 row classification

`NEW_EMBODIMENT` is category ID 10 in the exact checkpoint. Across all five
audited category-specific blocks, row 10 is initializer-like and closer in
RMS to unmapped/unused-looking rows than to the known pretrained reference
rows 24–27:

| block | ID10 RMS | named rows median RMS | unmapped rows median RMS | vote |
|---|---:|---:|---:|---|
| encoder W1 | 0.020018 | 0.037143 | 0.020007 | unused-like |
| encoder W2 | 0.019996 | 0.030228 | 0.019998 | unused-like |
| encoder W3 | 0.020024 | 0.021764 | 0.020000 | unused-like |
| decoder layer1 | 0.019992 | 0.021112 | 0.020002 | unused-like |
| decoder layer2 | 0.019970 | 0.020462 | 0.020013 | unused-like |

The conservative result is:

```text
ID10_CLASSIFICATION: ID10_UNUSED_LIKE
```

This is a static weight-shape classification, not proof of the slot's
training history. It does not mean the row is absent: row 10 is present in the
checkpoint, but its audited weights are indistinguishable from the
initializer-like comparison cohort.

## Active versus inactive action columns

The physical action is the first 10 dimensions of the universal 132D action.
For the ID10 action encoder input (`W1`, split across its 132 input columns)
and decoder output (`layer2.W`, split across its 132 output columns):

| block | active 10D RMS | inactive 122D RMS | inactive / active RMS | inactive energy fraction |
|---|---:|---:|---:|---:|
| encoder W1 | 0.020060 | 0.020015 | 0.997756 | 0.923927 |
| decoder layer2 | 0.019959 | 0.019971 | 1.000627 | 0.924330 |

The 92.4% energy fraction is primarily the dimensionality effect of 122
columns versus 10; it is not by itself evidence that the inactive noise caused
the trajectory oscillation. The important static finding is that ID10 does not
show a trained-versus-suppressed separation between its physical and inactive
column blocks. A causal inactive-channel intervention remains unrun.

## Boundary

```text
new model forward: false
new inference: false
new training: false
CUDA/GPU computation: false (CPU tensor reads only)
checkpoint modified: false
validation e read: false
holdout f read: false
WBC/hardware/ROS/ZMQ/P1M: false
```
