# Stopped655971 engineering repair

## Actual failure and repair

Job655971.hopper-m-02: finished Exit1, scheduler reported17 seconds, H200 hopper-17. Failure was authority data member comparison of login metadata against compute mount metadata, before model/canary/optimizer work. Completed optimizer updates0, checkpoint directories0, quality ledger empty. Actual scheduler stime10:46:37 and obittime10:47:11 span34 seconds; retain both scheduler fields and reported walltime17, without inferring utilization.

Only stopped Hopper authority.py and its Hopper CODE_SEAL entry changed. Previous source/seal are retained under evidence/engineering655971. Old allocation and process history remain unchanged. Core source/seal were untouched.

Expert arrays and parent14 now require byte size and full SHA256. Mount-specific dev/inode/ctime are retained as historical metadata, and current-process localstat is only a cache key. SHA checks cache once per current local file version/process. Actual CPU expert+parent+active C0 card full-byte readback PASS; synthetic different historical mount metadata passes, changed same-size bytes and wrong-size metadata reject.

Inspected callers: authority data_admission/admitted and stage_card were the only cross-mount stamp assertions; prepare_training merely records historical stamps. Native phase.restore_model/restore_optimizer and training_audit validate tensor/file bytes rather than those stamps. Core storage_recovery local file-closure checks remain unchanged.

Authority SHA before dc76988d95e9dbc14c06082c6e9ee7158f1bfe9864042547c65d6b73a2d81355; after2d3fff1c2d06fa2f8760fa6f7f57bb8babc5250139595d49bfc7585ad4b13154.

## Explicit cap and prospective recovery

Protocol section5 limits at most one accepted H200 lease; section7 allows at most two technical training recoveries, but does not explicitly waive that lease cap. No replacement submission is authorized by the current literal cap. Root requested explicit narrow user amendment. engineering_recovery.py default preparation records NOT_AUTHORIZED/launchfalse; no qsub or record handoff occurred.

After explicit user approval, root may publish active amendment with status USER_APPROVED_SEQUENTIAL_PREMODEL_ENGINEERING_REPLACEMENT, launch_allowed true, previous_PBS_JOBID655971.hopper-m-02, maximum_accepted_total2, replacement_leases_allowed1, literal explicit_user_approval, approval_received_epoch, canonical6072813602, releaseeae6dbc1f9281b204a4e28e5f7c100d705837318. Root refreshes capacity conservatively using actual quota and units, then invokes engineering_recovery.py --activate. That verifies exact failed/stopped/zero-work boundary and retains old current records as exact failed engineering history. Root invokes original submit.py afterward. Closed GPU process ledger and all scientific caps remain intact; count prior scheduler occupancy, no old budget transfer.

## Timing evidence

Historical DOSE formal2048 native microbatch timings: p50 .8873s per summed two microbatches; p90 .9104s.1024 at p90 times1.25 is1165.3s (19.42min), excluding optimizer/checkpoint/load/audit. Prior actual carry canary continuous/split4/resume4 used224.75/193.39/206.78s; fixed inference54.78/53.21s. Process-start to retirement754.76s (12.58min), excluding initial load before process receipt. This is one actual proof, not a p90 canary sample. Prudent C0 planning requires roughly35–45min after actual replacement start, then transfer/Core gates and at least45min confirmation reserve. Scientific time gates remain11:30 latest arm,12:10 updates,12:55 all science. Queue delay remains unknown.

## Evidence

655971_MOUNT_STAMP_FAILURE.json;655971_BYTE_REPAIR_CPU_READBACK.json;ENGINEERING_RECOVERY_PREPARATION.json;PRIOR_NATIVE_STEP_LATENCIES.json;PRIOR_NATIVE_PROCESS_LATENCIES.json. No worker qsub/model/optimizer/physics calls.
