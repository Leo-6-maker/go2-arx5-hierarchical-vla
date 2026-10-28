# Storage content-readback correction

The bounded deletion inventory and measured reclamation remain unchanged. Job 637051 has now completed 10000 updates, with PBS job_state=F / Exit_status=0; the finite original CALIB process remains active on Corelab. No restart or additional deletion was made for this supplement.

The mature PASS_PT_FULL_HF_READBACK receipts bind all full Trainer files to the fixed revision and source inventory, but their large-file method is hf_lfs_sha256. That method is LFS metadata verification, not complete streamed readback of the object. Earlier storage text describing those receipts as complete content readback overstates their method. Those historical receipts and hashes are preserved.

To meet the storage authorization's exact content requirement, G2000 and FULL2500/5000/7500/10000 are undergoing anonymous, fixed-revision streaming of every Trainer file, discarding payload bytes after SHA256/length checks. Completed per-file companion receipts are immutable and reusable. This is CPU/network verification only, with zero model queries, optimizer updates or physical episodes. No weights are downloaded to the PC, no whole Trainer package is stored on Corelab, and the healthy scientific worker is unchanged. Until each companion receipt completes, its strict full-content storage status is STREAMING_PENDING, not PASS. Existing recovery identities and candidate protection remain in force.

Input manifest and exact verification program are adjacent. The storage purge's 114350417486 logical bytes, measured per-host allocated/free changes, UNKNOWN skips and quota UNKNOWN remain as reported in CLOSEOUT.md.
