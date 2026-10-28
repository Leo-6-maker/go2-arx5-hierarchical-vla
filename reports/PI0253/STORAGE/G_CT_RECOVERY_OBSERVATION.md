# G complete content readback and shared C_T recovery companion

G2000 now has a strict anonymous complete-content receipt: all nine full Trainer objects / 20882646394 bytes streamed and SHA256/length checked at revision efecea485ae77000599d19171b02491f8d514f42, tree cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226. Bytes were discarded after hashing; no whole Trainer copy was created. The four FULL strict streams remain in progress in the same healthy process. Mature LFS-only receipts remain preserved alongside the companion receipts.

The unchanged qualified C_T runtime now has a small explicit restoration mapping:

- processor_base/processor_config.json and processor_base/statistics.json reuse the complete baseline input archive at revision f95e26f2abdcc5197aeb2f8259bb187aa0802521; their bytes were not reuploaded.
- Actual train_dataset/meta/stats.json, relative_stats.json and the fixed modality source are archived with CT_RECOVERY_COMPANION.json at revision f85ab92db7a6dec3765553aceeb4f45e7f42729b, prefix evidence/PI0253/ct-recovery-companion-20260929, repo leoyuer/pi0246-gate-pass-campaign.
- Four new small files / 96312 bytes; tree 0fe1e803f0654308fc94f6bc91e323cebb427358944903c89214aa9bb9451a47. Authenticated complete readback and separate anonymous complete readback of all six required objects PASS (4496675 total bytes across the two revisions).
- The original model-base statistics differ from the historical T parent. This known frozen binding is recorded explicitly; no substitution of baseline serialized statistics for TRAIN new-embodiment/relative statistics was made.

This companion is processor/statistics/source metadata for the qualified runtime, not a model or a Trainer resume checkpoint. Exact hashes match the current FULL card's C_T bindings. No model query, gradient query, optimizer update, physics episode, new deletion or healthy-worker modification was made for this supplement. Local G initializer remains protected; candidate inference views remain the active CALIB workset.
