# Actual Complete Engineering Checkpoint Delivery

The existing local-PC single owner completed private HF delivery of the actual
split/resumed engineering M4 checkpoint. It is **not a formal dual-task policy**,
does not pass the failed strict numerical smoke gate, and is not selected.

## Verified Artifact

- Existing private dataset repository: `leoyuer/pi0291-arx5-fixed-base`.
- Final immutable revision: `9cbba4da7623b1563d7c2029e6b4e8c0c171ee71`.
- Prefix: `PI0294/20261013T142457Z-localpc-pi0294/checkpoints/engineering-M4-5e9772b929d1`.
- Payload revision before READY: `b7c70f00c62e0f1492f9870942e41cee6a94c41e`.
- Complete checkpoint: 25,540,953,259 bytes, three safetensors shards, 1030 tensor
  headers, optimizer, scheduler, RNG, Trainer state, sampler, statistics, and the
  saved official processor/config/embodiment mapping.
- Checkpoint manifest SHA256:
  `5e9772b929d1efc497e0fc2423f8de082f37da656eb276bdf5ab77b7f4a3aa6c`.
- READY SHA256: `d68eddd9f2dd5c1edf90c397f419ad910d008cdb718646c9020e8f03f1587c8e`.

The exact remote file set, every file size, and every SHA256 were verified at the
fixed payload and final revisions. Large LFS files were verified without another
full download; small files were read back. READY was committed last. Local
checkpoint bytes were checked again after upload and remain intact.

The saved global step is 4 / 128 consumed samples in the restored split path.
Actual engineering updates across all paths remain 8; formal updates remain 0.
This artifact used the original PI0292 legacy bank/statistics for engineering
smoke, not the new paired v3.0 bank. File completeness is not an independent
numerical restore PASS or a task-success result.

## Actual Execution

Origin source `d92d94e1d026c630291d9273e7203605bf22b361`, native own job
`664432.hopper-m-02`, original request SHA and allocation SHA are preserved in the
HF manifest. Delivery source `2fd28e004c7cf1b9eb1d0db2f23d84931f005a91` was pushed
normally, four changed files read back exactly, and CI 38440570313 completed
successfully. Local old/new science contracts passed 172 tests / 177 subtests;
the separate full repository suite passed 758, with 14 explicit skips.

Hopper deployed the 219-file capsule at a verified idle boundary before dispatch.
The bounded delivery used one CPU, no model load, no optimizer pickle execution,
no physical work, and no new PBS/GPU allocation. Whole-parent plus reaped-child
accounting was 181.494410 CPU seconds / 263.781555 wall seconds. Inner publisher
CPU seconds are not added again. The exact process group was verified terminal;
CPU reservation returned to zero. Failed checkpoints and old sources were not
deleted. No second full local checkpoint archive was created.

At receipt time the owned run used 102,237,930,614 bytes. This is a final usage
snapshot, not a measurement of the exact intermediate peak. The delivery guard
retained the original 180 GiB cap, reserved at most 32 GiB temporary growth, and
enforced the original deadline. There is no claim of reclaimed model space.

## Remaining Science

The Hopper ledger remains PBS 3/4, 489 allocated GPU seconds, engineering 8/16,
main 0, pending 0, GPU reserved 0. Whole recorded Hopper CPU is 4385.828502 seconds.
The strict numerical resume comparison still failed; new deterministic GPU
qualification has not run. Formal v3.0 official qualification/HF receive,
main learning, true dual-SSH Teacher/model feedback, DEV/confirmation/regression,
and policy videos remain outstanding.

CoreLab's original SSH gateway continues timing out. The local campus tunnel and
route are present, so an absent VPN cannot be claimed as the diagnosed cause.
The gateway/server/network condition still requires external verification. No
collector restart, ID replay, new source deployment, or termination was attempted
on that unobservable host. Its current first-attempt status remains UNKNOWN.

The requested complete two-skill goal is active and incomplete. Engineering HF
delivery is concrete artifact progress, not a replacement for the requested
policy behavior. Selected/deployment remain NONE; WBC, Go2, and hardware were
not executed. The original 09:00 cutoff / 10:00 SGT deadline do not move.
