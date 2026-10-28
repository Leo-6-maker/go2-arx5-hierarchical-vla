# PI0291 Actual Closeout

Status: **ARX5_DATA_PARTIAL_NATIVE_CPU_AND_HF_VERIFIED**.
The explicitly authorized existing56-episode CPU-only closeout is complete.
The original80-attempt scientific target is NOT complete. No first-training
READY, model capability, mobile Go2, original WBC or real-robot success claim.

## Authority And Identity

- Original release:15cc7957488fb879f5f050e3602327592da7036b.
- Frozen physical/teacher/scorer/camera/data source:57a02b4923e5001df5fd4b3b5958c372b9cff541.
- Run:20261013T023717-localpc-arx5; domain FIXED_BASE_ARX5_TASK_SIM.
- Original physics T0/cutoff/deadline unchanged. A separate direct-human
  CPU-only closeout GO, #11 comment6135957173, runs01:12:42Z to02:12:42Z
  on2026-10-13; no new/retried physics was authorized or performed.
- All121 sealed source hashes rechecked equal after delivery.

## Actual Bank

Native six-axis ARX5 finite torque and opposed contact fingers qualified in
a fixed-base MuJoCo scene; original dual RGB saved. Pilot8/8, qualification
new layouts4/4. No modification or execution of the original WBC.

| Cohort | Planned | Attempted | Owner-Admitted Formal BC | Unattempted |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 64 | 56 | 56 | 8 |
| DEV | 16 | 0 | 0 | 16 |

Commissioning4 and pilot8 remain diagnostics, never formal BC. Overall
physical starts68, integrations1,088,000, paired observations43,588. Original
queue persisted child exit120 and terminal/current-null state; its PID was
absent after reconnection. Exit root cause is NOT_PROVEN. Every attempted
formal episode is complete/successful and independently raw-audited; there
are no formally attempted physical failures. The24 genuinely unattempted
IDs remain in the fixed denominator. Last disconnect-era UNKNOWN reports
are historical observations, not replaced by retrospective guesses.

Producer admission SHA256:8bf151625272114d667986618be4a0949a9a1e3bd4bced9aca3feaddc2315cf6.
Producer92-row attempt ledger SHA256:946f8d7a01f252a7fe6b5d862acf2b42281ae33d7c2ce400eb97f4bdd061bf2d.
Public lossless decimal-seed projections and separate producer/public hashes
are in `closeout-raw/`. Original bytes remain private.

## Actual CPU Data Qualification

Frozen export:33,656 TRAIN H40 windows,0 DEV. Each episode has641 observations,
640 genuinely committed absolute EEF10 commands and601 full H40 windows.
The original instruction is preserved; official relative conversion occurs
once, anchored at actual observed TCP. Privileged object/contact/teacher and
joint execution evidence do not become Student input.

Actual DatasetLoader -> fixed official processor -> official collator PASS:
real dual RGB, two different episodes/multiple stages, single/mixed batches,
variable-text diagnostic, deterministic0/2-worker equality,10 valid and132
padded dimensions, five rejected privileged/future/time-order negatives.
All33,656 TRAIN windows actually encode/decode; normalized float32 maxima
XYZ8.5735081e-9m, SO3 2.1073424e-8rad, gripper0. Pre-float32 action preclip35,
stock inverse float32 clip0, state preclip0 and encoder clipping0. Zero DEV
windows mean NOT_TESTED, not a zero-error DEV success. Actual CPU618.029364s.

TRAIN-only frozen statistics:33,656 state-anchor rows and1,346,240 equally
weighted window-action timestep rows. No DEV/pilot/failed/legacy fit,
percentiles or outlier clipping. Statistics SHA256:
43469f3818119c075774d91c5b08bbc2a46a0323a572a8e58614daa48155fd2d.
The immutable exporter manifest retains its at-export NOT_RUN field;
the final actual authority is `closeout-native/DATALOADER_SMOKE.json`.

## Actual Private HF Handoff

- Private dataset:leoyuer/pi0291-arx5-fixed-base.
- Prefix:PI0291/20261013T023717-localpc-arx5/v1-57a02b49.
- Data commit:373a792bd495e215b5f919307fb5a9d5f74a5241.
- READY-last fixed revision:c25eeadf9be0aa7d46bb6f7b65e260cbf3a305ea.
- Complete remote file set81, aggregate3,589,423,752 bytes; every size/SHA
  verified at the immutable revision. Private access rechecked.
- Corelab freshly downloaded a complete T291000 archive and the full native
  export shard:1,295 raw members and63 native members hash-verified. Original
  TRAIN statistics preserved, six real stage batches and all601 windows
  exercised through the frozen actual loader/official codec; PASS.
- Hopper separately read/hash-verified18,382-byte READY at that same revision
  using existing auth; private repo verified. No full dataset/model download,
  model load, neural forward or PBS.
- No robot meshes, private WBC tree, weights/tokenizer or credentials uploaded.
  Asset-free physical replay and third-party mesh redistribution NOT_CLAIMED.

Actual transfer/readback/Hopper proofs are in `closeout-final/`. Uploaded
DATASET_CARD/TRAINING_HANDOFF are upload-time metadata, so their HF-readback
pending fields are historical. Post-readback public summaries cite the exact
immutable transfer/readback receipts; no uploaded producer bytes are rewritten.

## Resources And Maintenance

Final measured-or-explicitly-bounded CPU ledger33,690.082884s, separate
bootstrap upper600s. The ledger includes previously disclosed lost-monitor,
maintenance and failed-launch upper bounds, plus a60s administrative reserve,
not a claim of perfectly measured utilization. Conservative closeout delta
876.546189s, including earlier uncharged receipts/reserve, is below14,400s.
Final owned logical8,059,518,979B; unique-inode allocated8,183,476,224B;
external-output upper268,435,456B. Combined logical bound remains below40GiB.
Observed SSD free566,368,034,816B is not personal quota; quota remains UNKNOWN.

Phase2 maintenance is PARTIAL_WITH_SKIPS:22/24 copy-only candidates,2,206
files, sources retained;2 dependency skips,9 cache skips, no proven eligible
extra model view. Phase2 deleted0/freed0; previous40.994GiB is separate.
Unqualified failed/partial checkpoints remain protected. Exact legacy process
retirement was already reported, not repeated. #47 comment6130933143 and
the maintenance receipt directory remain authoritative.

## Remaining Scientific And Engineering Gates

No DEV data and no complete80-attempt bank; insufficient basis for independent
model evaluation. Official Trainer adapter is NOT_QUALIFIED:stock factory
expects another disk format and constructs a model before data. CPU loader
PASS does not create a truthful runnable training command. Pretrained
candidate is unselected and full-weight identity not freshly qualified.
Future adapter/model work requires separate scientific GO and legal Hopper
PBS. New data/reliability work is only a NOT_AUTHORIZED draft.

All new model load/forward/train/PBS/original WBC/hardware/deletion are0.
Selected/deployment remain NONE. No background model service, tunnel or
physical relay test was opened. This completes only the bounded existing-data
closeout and stops for scientific review.
