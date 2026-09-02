# STAGE22 Checkpoint Provenance Contract V1

This contract seals a future candidate checkpoint for hash verification during
`Hopper -> local -> Corelab` transfer. It is a provenance operation, not a
model-selection or deployment operation.

## Rules

- only a successful simulation-only Stage22 run manifest may be referenced;
- the requested recipe must agree with `tune_diffusion_model` in that run;
- `action_horizon=40` and the recorded test-only timing semantics are retained;
- every regular file is SHA256 hashed; symlink targets must remain inside the
  checkpoint root and are hashed by resolved content;
- a deterministic tree SHA256 covers paths, kinds, sizes, content hashes, and
  symlink targets;
- explicit credential-like paths cause sealing to fail;
- `selection_status` defaults to `CANDIDATE_NOT_SELECTED`;
- `transfer_performed` is always `false` in this tool;
- sealing requires independent `global_step=50` evidence inside the checkpoint;
- sealing requires a passing direct-launch H2 lineage manifest and binds the
  base tree, train-fixture tree, split lock, modality, GR00T, Cosmos, source
  run-manifest, and checkpoint tree hashes;
- the manifest and checkpoint remain node-local and are never added to Git.

## Commands

Seal a candidate after an H2 run has produced a checkpoint:

```text
python scripts/stage22_checkpoint_provenance.py seal \
  --checkpoint /scratch/checkpoint-50 \
  --run-manifest /scratch/r0/run_manifest.json \
  --recipe R0 \
  --global-step-evidence /scratch/checkpoint-50/trainer_state.json \
  --lineage-manifest /scratch/h2_direct_audit.json \
  --output /scratch/checkpoint-50.provenance.json
```

Verify a copy after transport without changing it:

```text
python scripts/stage22_checkpoint_provenance.py verify \
  --checkpoint /scratch/checkpoint-50-copy \
  --manifest /scratch/checkpoint-50.provenance.json
```

The resulting status only establishes byte-level provenance. It does not say
that the checkpoint is selected, task-capable, real-robot-safe, or ready for
deployment.
