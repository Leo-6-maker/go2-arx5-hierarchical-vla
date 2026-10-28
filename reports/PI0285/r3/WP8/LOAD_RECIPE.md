# PI0285 source policy load and delivery recipe

This package binds the actual PLACE4096-1024 model view and full Trainer identities. Its current qualification is **DEVELOPMENT_ONLY_NOT_CONFIRMED**. There are no completed learned TRAIN/DEV/CONFIRM panels in this receipt. CPU bundle readiness does not establish task capability. Target selected/deployment remain NONE.

The Linux source evaluator uses the existing PI0250 native environment, PI0285 code, PI0283 native helpers, pinned official GR00T code and project source. Set PI0285_ROOT to the owner-bound source root and retain its original PYTHONPATH. Native dependencies and code hashes are in DELIVERY_MANIFEST.json. The model uses one row10, one BF16 K4 weight set and common frozen processor/statistics. The source Panda simulation is NON_REALTIME.

## Owner load/evaluate entry points

Run only the next already admitted fixed panel, from the evaluator's existing sealed environment. These commands execute model/physics calls and charge the original PI0285 ledger; they were **not run by WP7/WP8**.

```sh
# Corelab: the first eligible point, current PLACE4096-1024, after its real model service is ready.
python "$PI0285_ROOT/code/policy.py" --point PLACE4096-1024 --panel TRAIN
# Then recompute the actual recorded outputs using CPU only.
CUDA_VISIBLE_DEVICES='' python "$PI0285_ROOT/code/audit_source.py" --split PLACE4096-1024_TRAIN
CUDA_VISIBLE_DEVICES='' python "$PI0285_ROOT/code/qualification.py" --point PLACE4096-1024
```

DEV runs only when the unchanged TRAIN >=7/8 gate permits it. The first TRAIN >=7/8 and DEV >=13/16 candidate is frozen by the existing qualification entry point, then CONFIRM_A285300..285331 runs once. Failure closes learned experimentation. Missing or partial panels cannot qualify. Existing valid outputs, including quality failures, are reused.

For an owner-authorized resident load/parity call, the existing `model_worker.py --point POINT` reads one JSON object per stdin line. It loads and checks the full bound BF16 tensor identities before printing `{"ready":true,...}`. A fresh load that makes a prediction uses the inherited load/parity allowance. Never use a new arbitrary query as another quality screen.

The request schema is:

```json
{
  "input": "ABSOLUTE_OWNER_SEALED_ROOT/input.npz",
  "output": "ABSOLUTE_OWNER_SEALED_ROOT/output.npz",
  "condition": {"task": "T1", "layout": 285100},
  "query_index": 0
}
```

The NPZ contains only `actual_state` float32[10], `base_raw` and `wrist_raw` uint8[128,128,3], and one exact valid task `instruction`. No action target, teacher phase, object pose or task-ID tensor enters the model. `condition` and `query_index` identify the fixed noise packet and ledger; the native model receives only observation/language. Outputs are native BF16 K4 H40x132 with active10, decoded FP32 H40x10. Commands execute first8 at20Hz, then a new current pre-command observation; physics is100Hz, task duration32s. XYZ is canonical world/task frame, rotation is row-rot6d, gripper output is independent commanded opening while input is measured opening. Preserve the production clipping and decoder.

## Immutable manifests

The prospective helper files are isolated. The owner may install them into an isolated helper directory, preserving the healthy runtime and its code seal. Put that directory before existing code in PYTHONPATH for the helper commands only. Set R3_HELPER_DIR to that directory and POINT to the actual frozen point; ROOT must be the host's actual PI0285_ROOT.

```sh
# CPU only, Corelab, after immutable transfer/reconstruction. Store a new versioned path.
CUDA_VISIBLE_DEVICES='' python "$R3_HELPER_DIR/r3_delivery.py" build --root "$PI0285_ROOT" --point "$POINT" --output "$PI0285_ROOT/r3/WP8/$POINT/DELIVERY_MANIFEST.json"
CUDA_VISIBLE_DEVICES='' python "$R3_HELPER_DIR/r3_delivery.py" qualify --root "$PI0285_ROOT" --point "$POINT" --output "$PI0285_ROOT/r3/WP8/$POINT/QUALIFICATION.json"
```

The current repository manifest is a frozen development snapshot. Build the final confirmed manifest into a new `confirmed` directory after the unique confirmation passes; do not overwrite the development manifest. Bind copied source panels, CPU panels, CONFIRM_A_CANDIDATE and MODEL_POINTS by exact hashes before invoking protection on Hopper. These are small evidence files; no optimizer moves to Corelab.

## Independent protection before shared learning

After actual independent CONFIRM_A >=26/32, the protector copies the complete audited Trainer to a **new** owner-controlled Hopper `protected` directory, including optimizer, scheduler, RNG, cursor, model shards and processor files. It checks every file hash/size, link count, distinct file identity and all required recovery members; partial copies fail closed. It also binds the independently reconstructed Corelab inference manifest. Its copied full checkpoint is a useful exact recovery asset, rather than a metadata marker. Pristine, original/raw evidence and the current resume checkpoint remain active preserved references.

```sh
# Hopper CPU only. Copy the final Corelab delivery/evidence receipts by verified hashes first.
CUDA_VISIBLE_DEVICES='' python "$R3_HELPER_DIR/r3_delivery.py" protect --root "$PI0285_ROOT" --point "$POINT" --training-audit "$PI0285_ROOT/evidence/${POINT}_TRAINING_CPU.json" --view-manifest "$CONFIRMED_CORELAB_MANIFEST" --destination "$PI0285_ROOT/protected/PLACE_V1_FULL_TRAINER"
CUDA_VISIBLE_DEVICES='' python "$R3_HELPER_DIR/r3_delivery.py" verify-protection --destination "$PI0285_ROOT/qualification/PLACE_V1_PROTECTED.json"
```

No protected receipt exists in this preparation because confirmation has not run. Shared optimization stays gated. After protection, the owner runs the single T2 DEV16 acquisition reference on Corelab. A source-bound measured reserve receipt must declare `status=OWNER_MEASURED_SHARED_TRAIN_EVALUATE_AND_HANDOFF_RESERVE`, `admitted=true`, `remaining_measured_seconds>=required_seconds>=4500`, and a still-valid `valid_until_epoch` no later than1791498600 (Oct9 06:30 SGT). Only then may the CPU preparer produce the inherited full-Adam SHARED4096 card; it launches nothing.

```sh
# Hopper CPU only. The acquisition source+CPU receipts must be copied by exact hashes.
CUDA_VISIBLE_DEVICES='' python "$R3_HELPER_DIR/r3_shared.py" prepare --root "$PI0285_ROOT" --protected "$PI0285_ROOT/qualification/PLACE_V1_PROTECTED.json" --export "$PI0285_ROOT/delivery/$POINT/EXPORT.json" --audit "$PI0285_ROOT/evidence/${POINT}_TRAINING_CPU.json" --reserve "$OWNER_MEASURED_RESERVE"
```

Shared initialization restores exact qualified model+Adam moments/global steps from the independent copy; only its declared phase scheduler resets to cosine4096/warmup205/interface1e-4/DiT1e-5. The frozen plan is32768 TRAIN-only draws, T1/T2 exactly16384 each. A-confirmation layouts never enter training or development selection. Shared qualification requires each TRAIN7/8,DEV13/16,pairedDEV12/16, then unique CONFIRM_B each26/32 and paired24/32. Preserve qualified T1 through all successor outcomes.

Paired reporting reuses actual independent CPU panels and their same-reset RGB/state/camera proofs:

```sh
CUDA_VISIBLE_DEVICES='' python "$R3_HELPER_DIR/r3_shared.py" paired --root "$PI0285_ROOT" --point "$SHARED_POINT" --split DEV --old-point "$PLACE_V1_POINT" --output "$PI0285_ROOT/r3/WP7/$SHARED_POINT/PAIRED_DEV.json"
```

It reports both task scores, paired-both, opposite-goal41-sample hold outcomes and old T1 retention at the same fixed denominator. Wrong-goal hold alone does not assert a full historical grasp/lift success. Actual model video references are included only when original video files exist; no teacher video is labeled as model qualification footage. Current learned-video list is empty.
