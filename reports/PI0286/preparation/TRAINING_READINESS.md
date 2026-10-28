# PI0286 isolated training preparation

## Observed

- Hopper isolated training stage passed CPU import, syntax and counter checks; no model, optimizer or PBS submission by this worker.
- Sealed DOSE6144 parent: all fourteen live files reread and SHA256 matched fixed HF revision `75432eb91a2c602930a9e1e979aaebbeaebff02b`, tree `fb17b65effc3ae8dcbefa784adc8f7e6fab916821d29e4b2539e5be685ed8469`.
- Original 64 T1 TRAIN285100–285163 expert array members reread and matched source bank SHA256. Immutable namespace aliases preserved per episode; old codec FAIL label preserved.
- C0 plan: 8192 original expert chunks, native1024/warm52, micro4/accum2, peak row5e-5/DiT5e-6. Adam carry6144 and separate phase step0.
- Paired native epsilon/time schedule fixed before results at lineage6144 for both arms.

## Runtime interfaces

`stage_training.py` writes only isolated PI0286 train-owned files and normalizes LF. `prepare_training.py` binds actual parent and data, then writes prospective card. `submit.py` requires owner active C0 card and capacity admission, records the only request before qsub, and adopts any existing request without another submission.

`launch.py` takes one actual H200 allocation and runs C0 then waits in intervals at most30 seconds for an independently admitted C1 before11:30. No second allocation. Each arm independently restores full DOSE6144 parent, performs actual continuous8 versus fresh4+resume4 canary16, compares full states and fixed K4 outputs, retires only consumed new canary payload, then trains native1024. Nested model processes share GPU interval union. Full final state persists in shared scratch.

C1 card needs actual correction admission path/hash with status `PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC`, requested32, valid>=24, parent point `PI0285_REPAIR_A2048-2048`, layouts285108–285123, branch times[3,12], and exact4096 suffix rows in8192. Same full parent and paired RNG hash required. C1 PBS template is not submitted separately; the accepted C0 driver owns both arms.

## Not observed

The prospective C0 card has launch_allowed=false. Owner promotion and capacity/reserve admission precede qsub. Actual new parent-carry canary, full model training, qualification, archival and C1 correction admission remain unobserved by this worker. No historical canary PASS is reused. No full model loaded on Core or PC. Archived PI0285 runtime is unchanged.

Update stop12:10, latest arm start11:30; formal2048/allcompleted2112/canary32/technicalrollback32; no old budget transfer. Technical exact recovery is owner coordinated; this new driver rejects an existing allocation or canary proof rather than silently starting again.

## Evidence

- `TRAIN_CODE_STAGED.json`: isolated CPU stage.
- `C0_TRAINING_PREPARATION.json`: actual parent/data readback and prospective card.
- Hopper `design/PARENT_READBACK.json`, `data/EXPERT_DATA_ADMISSION.json`, `design/PAIRED_NATIVE_RNG.json`, `design/C0_REPLAY1024_CARD.prospective.json`.

## Conditional C1 handoff helper

`prepare_c1_transfer.py` is a local owner-invoked CPU relay, syntax checked and not executed. It requires the actual correction bank and fixed sampler PASS on Core, derives a finite named member list, rereads SHA256 on source and target, and transfers only fresh owned arrays plus correction receipts/plan. It uses existing SSH identities through a PC byte stream; no whole dataset copy, qsub, code edit or CODE_SEAL change. It writes a prospective C1 card with launch_allowed=false. Owner reserve/admission promotion remains required. Actual relay is unobserved until admitted corrections exist.

Owner accepted sole job655971.hopper-m-02 at10:36:11. Worker read-only checks found queueQ/Insufficientngpus, allocation absent and no training logs. No further submission.
