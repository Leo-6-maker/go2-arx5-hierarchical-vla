# Stage22 H2 Direct-Launch Audit Contract V1

The authoritative H2 experiment does not depend on the newer Stage22 wrapper.
Hopper invokes the frozen Stage20S launcher from
`deb920a070141e6a32a88cd0da6719828647e1e6` directly, once for R0 and once for
R1, with the same H200, runtime, base checkpoint, and train-only fixture.

The Corelab adapter consumes:

```text
R0 STAGE20S_GR00T_TRAINING_RUN_V1
R1 STAGE20S_GR00T_TRAINING_RUN_V1
STAGE22_H2_DIRECT_EXECUTION_PROVENANCE_V1
STAGE22_VALIDATION_SNAPSHOT_LOCK_V1
```

It never reruns either launch. A passing audit is an integrity gate, not a
profile-quality or task-success claim.

## Required execution provenance

The provenance manifest must record:

- direct frozen-launch commit and command for each recipe;
- training seed `2222`, 50 optimizer steps, save interval 25, learning rate
  `1e-4`, H=40, and the test-only `0.05 s` action timing;
- exact base-model tree SHA256
  `8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4`;
- train-fixture tree SHA256, split-lock SHA256, modality hash, GR00T SHA;
- Cosmos repository, resolved revision, and cache-tree SHA256 from the passed
  H1 gate; H1 must record `status=PASS` and
  `optimizer_step_completed=true`;
- one GPU UUID shared by R0 and R1 plus matching runtime identity;
- identical base-model, train-fixture, and modality paths in both direct run
  manifests; the shared base tree hash is checked separately;
- validation snapshot-lock SHA256 and the exact inference seeds
  `2223/2224/2225`;
- SHA256 of each direct run manifest and module-inventory file;
- a checkpoint-50 lineage contract listing the required hashes, even before a
  checkpoint is selected.

## Requires-grad inventory

Each recipe supplies a
`STAGE22_TRAINABLE_MODULE_INVENTORY_V1` JSON file with one row for each role:

```text
projector, state, action, diffusion, visual, llm, vlln
```

Each row must contain `name`, `all_requires_grad`,
`trainable_parameter_count`, and `parameter_count`. R0 enables only
projector/state/action; R1 additionally enables diffusion. Visual, LLM, and
VLLN must be frozen in both. R1 total trainable parameters must equal
`1,419,078,784`.

## Command

```text
python scripts/stage22_audit_h2_direct_launch.py \
  --r0-run /path/to/r0/run_manifest.json \
  --r1-run /path/to/r1/run_manifest.json \
  --execution-provenance /path/to/h2_execution_provenance.json \
  --validation-snapshot-manifest manifests/STAGE22_VALIDATION_SNAPSHOT_LOCK_V1.json \
  --output /path/to/h2_direct_audit.json
```

The audit emits explicit gate labels:

```text
DIRECT_LAUNCH_AUDIT_SUPPORTED
EXACT_SEED_GATE_2223_2224_2225
VALIDATION_SNAPSHOT_COMPLETENESS_GATE
BASE_TREE_HASH_GATE
TRAINABLE_MODULE_INVENTORY_GATE
SAME_GPU_UUID_GATE
CHECKPOINT50_LINEAGE_GATE
```
