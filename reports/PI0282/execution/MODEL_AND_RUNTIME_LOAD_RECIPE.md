# PI0282 actual model and runtime load recipe

## Fixed model identities

| Point | Public HF revision | Exact export prefix | Local availability |
|---|---|---|---|
| ACTION512 | ff03f42e5178a367cb99fac054ec498412f1fc09 | evidence/PI0282/R2-delivery/ACTION512 | HF model-only recipe; local view retired after valid TRAIN0/4 |
| ACTION1024 | 60027054d3f62e5ae0b2b3e70be54e76fe9a2980 | evidence/PI0282/R2-delivery/ACTION1024 | Complete Hopper Trainer and Core inference model |
| ACTION2048 | 4f32c9fd680a1db69ef1fdc0f1411e16a801ccdc | evidence/PI0282/R2-delivery/ACTION2048 | Complete Hopper Trainer and Core inference model |

Repo: `leoyuer/pi0246-gate-pass-campaign` (PUBLIC). These exports contain whole changed tensors, not tensor residuals; recover the mixed full master exactly from the unchanged original base plus export. Optimizer is not part of an inference export.
Original base: `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`. Official code `51d4c89f72fda44cbf77285c6a8114b52676b8a1`; Cosmos `9ce19a195e423419c349abfc86fd07178b230561`. These dependencies already exist on both hosts.

## Existing complete recovery

Hopper root `/scratch/e1192548/Projects/go2-arx5/experiments/PI0282`.
Complete baseline recoveries: `learning/ACTION2048/main/checkpoint-1024` and `checkpoint-2048`. `HOP_TRAINING_FINAL_CPU_AUDIT.json` inventories every file with byte size/SHA256 and independently validates full model tensors, other31 packed rows, Adam moments and all step counts. Config-only directories and the HF-only512 point are NOT exact Trainer recoveries. Last recovery is2048, cursor16384; cumulative actual work2080, not2048.

## Existing Core inference bundles

Core root `/home/liuyu-nfs/Projects/go2-arx5/PI0282`.
`models/ACTION1024` and `models/ACTION2048` link to actual local NVMe folders `/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0282/models/POINT`.
Their `MATERIALIZATION.json` and `design/MODEL_POINTS.json` bind every serialized shard and all1030 effective native BF16 parameter hashes. The sole omitted original serialization alias is tied `backbone.model.lm_head.weight`; its equality to the frozen embedding was proved.
Processor/statistics/mapping are in `data/processor_config.json`, `data/statistics.json`, `data/embodiment_id.json`, also preserved in each checkpoint and HF export. Decode and packet semantics are the actual runtime `code/codec.py`, `code/inference.py`, `code/source.py` and source-card bindings. TRAIN-only whole-population min/max; actual world TCP XYZ+first2 rotation rows+measured opening; native full132 noise/S4/H40; original gate, native source PD and32s task.

## Exact original command forms

These preserve the already executed contract. Finished quality panels must be inspected/reused, not rerun. Every new model/physical query still enforces the absolute R2 cutoff; the commands do not grant execution after authority expiry.

```bash
root=/home/liuyu-nfs/Projects/go2-arx5/PI0282
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/hf_home
export PYTHONPATH="$root/code:/home/liuyu-nfs/Projects/go2-arx5/PI0280/official:/home/liuyu-nfs/Projects/go2-arx5/PI0276/project-src"
/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python -u "$root/code/inference.py" --root "$root" --point ACTION2048
# Ordered stdin request contains input/output paths under root, condition and query_index.
# Packet file has actual_state, base_raw, wrist_raw ONLY; no teacher/future action.

# Original finite panel form (already consumed TRAIN results are not relaunched):
/home/liuyu-nfs/Projects/go2-arx5/PI0276/venv/bin/python -u "$root/code/source_launch.py" ACTION2048 TRAIN

# CPU-only saved-output recheck, no new model/physics:
export CUDA_VISIBLE_DEVICES=''
/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python "$root/preparation/closed_loop_audit.py" ACTION2048
```

To reconstruct a locally retired inference view, download only the six listed model-export files from its fixed PUBLIC revision into a goal-owned directory, verify every byte size/SHA from the retained HF receipt, then run the actual alias-aware `model_view.py` with the existing original base. Do not download a new pretrained model or transfer an optimizer for inference. A fresh scientific launch still requires applicable authority.

## Scope and quality

All policies are source-Panda EEF+gripper policies. They are not target WBC policies, not real-time systems, not physical-transfer approvals, not deployment selections. `selected/deployment=NONE`. Model availability and CPU integrity PASS are not task PASS. Refer to the fixed per-condition quality report for the eventual verdict.
