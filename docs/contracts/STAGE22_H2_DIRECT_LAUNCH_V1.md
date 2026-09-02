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
- exact train-fixture tree SHA256
  `2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789`;
- exact split-lock SHA256
  `7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246`;
- exact modality-config SHA256
  `e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031`;
- exact GR00T SHA
  `51d4c89f72fda44cbf77285c6a8114b52676b8a1`;
- Cosmos repository, resolved revision, and cache-tree SHA256 from the passed
  H1 gate; H1 must record `status=PASS` and
  `optimizer_step_completed=true`;
- one GPU UUID shared by R0 and R1 plus matching runtime identity;
- an independently captured, SHA256-bound `STAGE22_GPU_EVIDENCE_V1` artifact
  for each recipe, containing hostname, capture time, UUID, device name, and
  total VRAM; top-level string equality alone is not sufficient;
- identical base-model, train-fixture, and modality paths in both direct run
  manifests; the shared base tree hash is checked separately;
- validation snapshot-lock SHA256 and the exact inference seeds
  `2223/2224/2225`;
- SHA256 of each direct run manifest and module-inventory file;
- a checkpoint-50 lineage contract listing the required hashes, even before a
  checkpoint is selected.

The direct-launch run template in
`hpc/pbs/stage22_h200_r0_r1_50.pbs` is intentionally a thin launcher: it
invokes `stage20s_train_gr00t.py` from the frozen source directly for R0 and
R1, after H1 has passed. It does not invoke `stage22_train_recipes.py` and it
does not repeat H1.

## Requires-grad inventory

Each recipe supplies a
`STAGE22_TRAINABLE_MODULE_INVENTORY_V1` JSON file with one row for each role:

```text
projector, state, action, diffusion, visual, llm, vlln
```

Each row must contain `name`, `all_requires_grad`,
`trainable_parameter_count`, and `parameter_count`. R0 enables only
projector/state/action; R1 additionally enables diffusion. Visual, LLM, and
VLLN must be frozen in both. The file must be emitted by the independent
read-only model inventory probe using live `requires_grad` values, and must
include its parameter-inventory digest and frozen-launcher construction
identity. R1 total trainable parameters must equal `1,419,078,784`.

Capture one read-only GPU identity artifact immediately before each direct
launch (with a distinct output path):

```text
python scripts/stage22_capture_gpu_evidence.py \
  --output /scratch/h2/r0_gpu_evidence.json
python scripts/stage22_capture_gpu_evidence.py \
  --output /scratch/h2/r1_gpu_evidence.json
```

The H2 executor must also run the read-only live inventory probe once for each
recipe, using the same frozen launcher, base model, dataset, and modality
config. The probe constructs the model only; it must not call forward,
backward, inference, an optimizer step, or a training loop:

```text
python scripts/stage22_probe_trainable_modules.py \
  --stage10-root /path/to/stage10 \
  --base-model-path /path/to/GR00T-N1.7-3B \
  --dataset-dir /path/to/lerobot_train \
  --modality-config /path/to/go2_arx5_modality.py \
  --launcher /path/to/frozen/stage20s_train_gr00t.py \
  --recipe R1 \
  --output /scratch/h2/r1_module_inventory.json \
  --local-files-only
```

The resulting GPU evidence and inventory files are node-local artifacts. The
execution-provenance manifest records their paths and SHA256 values; neither
the evidence nor model-private files belong in Git.

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
EXACT_FROZEN_INPUT_IDENTITY_GATE
BASE_TREE_HASH_GATE
TRAINABLE_MODULE_INVENTORY_GATE
ARTIFACT_BACKED_GPU_EVIDENCE_GATE
SAME_GPU_UUID_GATE
CHECKPOINT50_LINEAGE_GATE
```
