# C0 full Trainer archival readiness

Online preflight passed for existing PUBLIC model repository leoyuer/pi0246-gate-pass-campaign. Fixed new prefix: arms/PI0286/C0_REPLAY1024/checkpoint-1024.

The wrapper reuses the exact mature PI0284 full14-member upload/fixed-revision streamed byte-readback implementation (SHA126880ca6a0aab20ad556417335500f5b18f26c1f9fbbfdb81027b62049a064c), with root/campaign binding to PI0286 and upload concurrency4. It is staged only under Hopper PI0286/archive/c0_full_trainer; healthy code and seal are untouched. Online environment is child-only and separate from offline training. Token stays on server and is never printed.

One actual CPU worker is waiting for the complete C0 checkpoint1024 and PASS native TRAINING_CPU audit. Before upload it requires all14 exact members, phase step1024/max1024, cursor8192,2048 microbatches/8192 consumed chunks, and parent Adam6144 card. No scientific result is claimed by this readiness receipt.

CPU affinity<=4, address-space limit16GiB, low scheduling priority. No model/neural/teacher/physics calls. Full Trainer is not copied to PC/Core; public readback streams each full file into SHA256 and size counters. Unique local Trainer remains protected after archival. Online archival may continue after14:00, independent of science cutoff. No training retry or source/code change on archival error.

Expected final receipts: Hopper evidence/C0_REPLAY1024/checkpoint-1024.archive_receipt.json, checkpoint-1024.full_readback.json, archive/c0_full_trainer/COMPLETE.json. Fixed HF revision and tree are unobserved until actual upload/readback PASS.
