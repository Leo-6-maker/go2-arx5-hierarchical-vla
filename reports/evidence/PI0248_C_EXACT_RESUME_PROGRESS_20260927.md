# PI0248 C exact-resume progress — 2026-09-27

## Current state

- The original C run completed updates 1,000/2,000. Job `634134.hopper-m-02` failed only in its archive callback; the C step-1000 checkpoint was uploaded to public HF and fully read back.
- Exact-resume attempt `634250.hopper-m-02` restored the pinned step-1000 Trainer state, then stopped before a resumed optimizer update because the recovery attempt lacked the 1,000-row S4 objective-audit prefix. No update was repeated.
- The restored local checkpoint was re-hashed against the durable source record and retired through the project helper. Scratch returned to 39.7 GB free before the new job.
- The C resume adapter now copies and validates the exact 1,000-row audit prefix. Its continuation preflight preserves the original empty-prefix receipt and proves the current C prefix contains exactly the nine checkpoint files plus the verified manifest. It reports `target_prefix_empty=false` and `prefix_was_empty_before_first_publish=true`.
- C resume preflight: `PASS_PI0248_HF_RESUME_PREFIX`; run-card SHA-256 `638e9f3ea4bc3e195a23afe64701c4da6b6874577642faeb025477c433c6914a`; step-1000 tree SHA-256 `8c0cfdd6116cdfc0e295091900539b438fcd56d87423a4cbcc7a28a5fc8c5e46`; pinned checkpoint revision `3d817c29f972735475c2e5fde3b244aede6f6d79`.
- R preflight: `PASS`, its independent prefix is empty, and authenticated public-HF write/readback passed.
- Job `634387.hopper-m-02` is running on one Hopper H200 with `PI0248_RESUME_AFTER_STEP=1000`; PBS `Mail_Points=n`. The latest query showed it running; final completion and update count are not yet available.

## Scientific boundary

Both arms retain the sealed 2,000-update recipe, fixed exposure table, S4 objective, fresh optimizer, LR 1e-5, and M64-final5000 initializer. C resumes only updates 1,001–2,000. R remains required regardless of C's later verdict. Only final-2000 checkpoints proceed to the two-noise, four-gate evaluation; no scientific gate has been evaluated from C step 1,000, and `selected_checkpoint=NONE`.

## Next

Complete C to 2,000, verify/archive both checkpoint states and close training integrity read-only; then complete R and evaluate both arms at final-2000 on both noise groups. Stop on the first paired protocol verdict; do not add steps, seeds, losses, or branches.

## Recovery retry update — 2026-09-27, 06:32 SGT

This update supersedes the earlier statement that job 634387 was running. Job 634387.hopper-m-02 exited before a resumed optimizer update: the inherited GR00T TrainingConfig requires a boolean resume_from_checkpoint during config serialization, while the exact-resume adapter supplied the checkpoint path string. Its logs and receipt remain preserved.

Commit 0a37e542f3d653964c894bb550d428557282462f fixes only that serialization boundary: it temporarily supplies the required boolean while GR00T writes the config, then restores the verified checkpoint path for Trainer resume. Frozen PI0247/GR00T source and the science card fields are unchanged.

- New sealed C card SHA-256: 77de414d3f0ea4fd5a377f309fd1ea91f98fd02ab8a364f2c3f3600c6256e9a0.
- C HF resume-prefix preflight passed again. It verified the exact nine public-HF checkpoint payloads plus manifest, pinned revision 3d817c29f972735475c2e5fde3b244aede6f6d79, and tree SHA-256 8c0cfdd6116cdfc0e295091900539b438fcd56d87423a4cbcc7a28a5fc8c5e46.
- R card SHA-256: 181b724d39ba3a41ed1abebe11d9a9c3c3358ad81a7a82d87d3e515c6ca70c6b. Its online public-HF write/readback preflight passed with the R prefix empty.
- C job 634402.hopper-m-02 is running on one Hopper H200 with Mail_Points=n. Exact restore passed from the pinned step-1000 archive; the optimizer state and remaining 1,000-row continuation are bound by resume_record.json. Logs reached GR00T Starting training and Resuming from checkpoint. The resumed update count is pending read-only verification.

The sealed 2,000-update recipe, 64-row table, S4 objective, fresh optimizer, LR 1e-5, and final-2000-only paired gates remain unchanged. R remains required after C regardless of C's verdict; selected_checkpoint=NONE. No gate verdict is assigned yet.
