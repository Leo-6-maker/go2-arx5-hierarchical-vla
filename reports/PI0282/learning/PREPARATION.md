# PI0282 native learning preparation

Authority: release `8fdaab73b8b6686409317d9562833235c2839acc`, canonical `6043060523`. This preparation has **zero actual model forwards, training updates, PBS submissions or physics starts**. GPU admission and source evaluation belong to the owner.

## Original files and runtime

Hopper original model: `/scratch/e1192548/Projects/go2-arx5/archive/stage_runs/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B`. Both weight shards plus config, processor, statistics and embodiment mapping match the PI0280 revision hashes and sizes exactly. The effective model load will still check all 1,030 original parameter hashes and all four loading-information error lists before changing scope.

Official clean source: `/scratch/e1192548/stage22_h200_runtime_deb920_20260902/src/isaac-gr00t`, commit `51d4c89f72fda44cbf77285c6a8114b52676b8a1`. Existing runtime venv: `/scratch/e1192548/stage22_h200_runtime_deb920_20260902/runtime/venv`; observed Python 3.12.3, Torch 2.9.0+cu128, Transformers 4.57.3, Accelerate 1.14.0 and Safetensors 0.8.0. The old environment's overlay-rich `PYTHONPATH` must be replaced with only this official tree and PI0282 code. FFmpeg, loader libraries and existing offline Cosmos cache can be reused as location bindings.

## Scope and objective

Headers confirm 14 packed BF16 projector tensors: 325,783,680 scalars, with exactly 10,180,740 in row10. Shared DiT has 1,091,722,240 scalars. Both arms load independently from the same original files. No row initialization, source-row copying or new network architecture is used.

The official complete model forward and masked flow-matching loss remain unchanged. Packed row10 gradients are masked explicitly, with zero projector decay and fresh Adam state. Other rows' parameters and Adam moments are checked exactly. ACTION enables only the additional native shared DiT; INTERFACE leaves it frozen under native evaluation behavior. The native head forward does not wrap the frozen DiT in `no_grad`, preserving the encoders' input gradient path. The VLM, vlln, VL self-attention and positions remain frozen. ACTION's native dropout training mode and INTERFACE's frozen DiT evaluation mode are recorded as part of the scope comparison.

Trainable masters are FP32; native Trainer autocast is BF16, with fp16 and TF32 disabled. LR, decay, clipping and the 103-step/2048-step cosine schedule are the prospective protocol settings. Full 132-dimensional native action noise remains. The passive observer records the actual original `torch.randn` output and native `sample_time` return, without modifying either distribution. Local CPU checks establish mask/zero-decay/moment properties and unchanged passive RNG capture; real BF16 native gradients and paired CPU/CUDA randomness remain canary evidence to be obtained.

## Consumption, recovery and budget

The ordered two-worker pinned loader consumes DATA's actual `FrozenDataset.rows` from `PLAN16384.jsonl`. Only the training step records consumption; prefetched rows do not advance the cursor. The owner freezes `design/TRAINING_LAYOUT.json` with common `microbatch` and `accumulation` before either arm. Preferred layout is 4×2, effective batch8.

Each arm uses three distinct processes: continuous stop8; split stop4; fresh split resume4 stop8. Full HF checkpoint4/8 saves model, optimizer, scheduler, RNG, Trainer state and cursor. The CPU comparator checks all consumed inputs and actual noise/time exactly, frozen rows exactly, updated FP32 tensors at the published tolerance, and bitwise equality separately. The owner supplies one fixed native S4 canary observation at both 8-update endpoints per arm, with identical actual inference noise and original gate; four calls total. The comparator also verifies other-row optimizer moments are zero.

Main training starts only after its canary passes. Formal checkpoints128 (safe health), 1024 and 2048 use the same full Trainer format; both 1024/2048 directories remain. Latest/previous are references to complete directories. The ledger checks before the next update and records every completed optimizer step, including discarded or rolled-back work: formal4096, canary32, prospective memory32, technical64, total4224. Native training-forward attempts have a separate journal. Buffered journals flush every16 entries and fsync at close; the maximum unflushed crash boundary is15 entries, while every completed optimizer step fsyncs its ledger immediately.

Payload estimates per complete Trainer are 9.47 GiB for INTERFACE and 19.64 GiB for ACTION, including FP32 masters and full packed Adam moments. The four formal endpoint/milestone payloads total about58.2 GiB, before small metadata and atomic peaks. These estimates are not quota measurements. The owner permits retiring only this goal's redundant consumed canary payloads after complete hash/restore/RNG/prediction audit PASS, retaining receipts; old paused recoveries are protected.

## Entry and outstanding admission

`train.py --root <PI0282-Hopper-root> --base <original-model> --data-root <new-data-root> --arm ACTION2048 --phase continuous --stop 8` starts the first canary only after owner publication, legal PBS allocation, common layout, and four successful original expert codec replay receipts. `--phase split --stop 4`, then a fresh process with `--phase split --resume 4 --stop 8`, closes recovery. `canary.py --root ... --arm ...` consumes actual comparison artifacts. ACTION precedes INTERFACE. Formal `--phase main --stop 1024` and fresh `--resume 1024 --stop 2048` preserve the common schedule and cursor.

Pending: DATA's native roundtrip/cache/batch/ordered-loader receipts, expert codec admission, legal scheduler/resource admission, actual pristine effective loading, real native gradients and full recovery canaries. VPN loss prevented additional remote CPU checks; it did not authorize a different model/runtime. No policy is selected or deployed.
