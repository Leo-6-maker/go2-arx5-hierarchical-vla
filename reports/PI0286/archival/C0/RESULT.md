# C0 full Trainer PUBLIC archival — actual PASS

## Local receipt copy deviation and repair

The first C1 receipt collection accidentally wrote generic C1 receipt filenames into the local C0 folder because its copied download command retained the C0 destination. This was a local report-copy error. The earlier claim that the C1 local receipt set had been copied correctly is retracted. The incorrect local bytes were preserved before repair in `../receipt_copy_deviation_20261009_1428/`, with `ERROR_COPY_INVENTORY.json` and `REPAIR_RESULT.json`.

Both arms were subsequently fetched from their original explicit Hopper paths with `fetch_trainer_archive_receipts.py`. C0 now binds revision `b0b8c65de8d5bbdf0946ccdda9ecb23348cf16c9`; all 12 copied receipt hashes and all 14 native/upload/readback entries and the campaign tree passed. Corrected C0 index SHA256: `c4eb263635e81cb02f26ea84b0885cdf4dbf947223509ca9c5c1d7f0babb86af`. Remote original receipts were unchanged; the repair reran zero models, uploads, or model byte readbacks.

Completed 2026-10-09 13:23:42 SGT. Repository `leoyuer/pi0246-gate-pass-campaign`, prefix `arms/PI0286/C0_REPLAY1024/checkpoint-1024`, fixed revision `b0b8c65de8d5bbdf0946ccdda9ecb23348cf16c9`.

All14 files, 20,468,176,040 bytes, fully read back anonymously at that revision. Every size/SHA equals the native TRAINING_CPU audit and upload manifest. Campaign tree recomputed: `e61e2d5288dab9a0408a67bd88583d022e3c85d42aef5ebfbe699957bcb3d7ab`.

`ACTUAL_RECEIPTS_INDEX.json` records original Hopper paths, sizes, and SHA256 for all12 receipts, fetched byte-for-byte. Key hashes:

- upload receipt: `66f7678bbccf390c3dba2661f1c6b1b9a31df830d8546a442a02cced8d0f2630`
- complete readback: `daba4bae104b639a672a6b15721cd76f0e00aa546cde3b8dcd5f6ad25eb40169`
- native-audit/archive binding: `5175556e93b7824459dabdf4c4dab7ae2619a54db779e7d631bf65d58993519a`
- native CPU audit: `5f41404bd6463ee2edfed76b0efc8ec04f3c23e9cf332898f74b06e50f61face`

The mature helper uploaded once and verified seven whole files, including both model shards. Actual optimizer download measured9.07MB/s over30.34s. Authorized archival repair reused those whole-file PASS members and the existing revision, then read the remaining optimizer through four bounded64MiB HTTP ranges at a time. Every response required exact206 Content-Range/length. Bytes were hashed in original file order; the complete optimizer SHA matched. Remaining metadata also received full-file byte checks. No second upload occurred.

CPU affinity4,16GiB address-space limit, four network worker threads plus hash/main thread. Scientific queries, optimizer updates, physics calls, and full-Trainer transfers to PC/Core: all0. Archive code stayed outside training code/seal. The completed CPU worker exited0; archival-only failures/stops and source/logs are preserved.

The local full14 Trainer was present when the independent proof was written. Root subsequently reported exact C0 HF-only retirement of the two model shards and optimizer at13:25, under its separate `execution/C0_FULL_HF_ONLY_RETIREMENT.json` receipt. This is historical retained-at-completion evidence. Local C0 full Trainer is no longer resumable; do not rerun the verifier that requires all local members. Reuse the closed archival proof plus the separate retirement receipt. Configs, journals, manifests, and readback evidence remain preserved.
