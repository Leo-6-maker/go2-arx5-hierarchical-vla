# PI0253 storage preview

Authority eb2ac1b76338fc5b3486809f6bb28e80529ed4b1 / canonical5874472882. Explicit files only; published preview precedes unlink. No model queries, optimizer updates or candidate-set changes.

|Host|Files to unlink|Logical bytes planned|Preview SHA256|
|---|---:|---:|---|
|hopper-l-01|10|47646006520|87981b10cbb44c528934477036797c336d6316ec3f1f184b3d34b76d8b72c9a2|
|corelab-4080s|26|66704410966|b2d32265c25b03436a9b161a967347f82ce3e0cdaaf2a59abd47c06f0891d8e2|
|corelab-amd-server|0|0|015fa9b782d74636cedbc50ae9a63d6317d8569e94c038245c816236d66ca907|

## Protection and retirement

- Hopper: current 637051 fileset and G initializer protected, including shared G hardlinks. Retire C/X initializer copies, terminal PI0243 FAIL and verified global_step=max_steps=1 smoke copies.
- Corelab GPU/shared NFS: retire terminal K/X/C/E weights and exclusive model symlinks; G/A/FULL1000 unused model views become HF-only with exact prior full-file readback plus fresh anonymous fixed-revision object reachability. No new large-file readback claimed.
- FULL2500 remains formal candidate and local working model. FULL5000/7500/10000 local working directories protected. FULL1000 is model-only, never a resume/candidate.
- Corelab AMD: legacy baseline/qualification roles UNKNOWN, all skipped; shared NFS counted on GPU only.
- Required U milestone/baseline/data/processor/statistics/small evidence/runtime and existing HF/Git history retained; this round does not retire them.
- Unknown old assets skipped individually. Only owned explicit regular files/symlinks; inode/link/identity/open-FD recheck immediately before unlink. Inaccessible PAM/SSH/systemd login session daemons recorded separately; no project worker exempted. Compute-node process inspection unavailable; all live card-bound inputs and active root protected.
- Weight tombstones mark no local reload/resume. Tiny configs are metadata, not Trainer state. No global cache prune, recursive delete, unrelated process interruption or additional broad cleanup.
- Quota headroom UNKNOWN. Final report separately records logical unlinks, allocated inode bytes and measured free changes; concurrent training/download changes are not credited as cleanup.
