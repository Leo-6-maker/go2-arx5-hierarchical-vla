# PI0244 R2/XYZ4 Engineering Attempts — 2026-09-23

## CPU regression attempt 1

- Source commit: `9fd52f9ba639edec7078730b0d58b396e7302ce6`
- Runtime: Hopper Stage10 Python 3.12.3, CUDA disabled
- Model weights loaded: 0
- GR00T model forwards: 0
- Optimizer/scheduler steps: 0
- Result: failed in the self-test, before any model load or scientific run.
- Cause: the weight-one parity check called the fixed XYZ×4 loss and compared it with the native unweighted loss.
- Observed assertion: expected `2.1787754513324087`, got `2.1290663833316708`; absolute delta `0.04970906800073793`.
- Repair: parameterize only the local loss helper's XYZ weight so the CPU control calls weight 1, while the sealed training invocation retains weight 4. No trainer recipe, data, model, precision, mask, or gate changed.
- Disposition: preserve this failed test record; rerun the exact-runtime CPU regression after the repair.

## HF archive identity check

- The sealed R0 card names `leoyuer/pi0234-stage-a-admitted-v1`; authenticated Hub `model_info` returned 404 for that repository.
- The preserved R0 receipts themselves contain mixed `durable_path` repository prefixes: checkpoint-2500 names `leoyuer/pi0234-stage-a-admitted-v1`, while 5000/7500/10000 name `leoyuer/pi0243-stage-a-admitted-v1`. The original receipts and R0 card are unchanged.
- The configured token identifies account `leoyuer`. Its existing `leoyuer/pi0243-stage-a-admitted-v1` repository is public and readable; the proposed `pi0244x-xyz4-m16-10k-v1/` prefix had zero files at the live check.
- R2 therefore binds its new family to that existing public repository and the unused R2-only prefix. This is storage routing only; the R0 paths remain recorded verbatim and no R0 checkpoint was rewritten or uploaded.

## Project_Storage capacity proof attempt 1

- Prepared card SHA256: `808f84bf05669fcb0154804c664aa800336790dd7eebb31a56fc5d838daa88bc`.
- The mature `posix_fallocate` proof requested `125348740672` bytes on the exact output fileset. It failed closed when one approximately 10 GiB segment reported only `106496` allocated bytes. The helper removed all temporary reservation segments; no checkpoint or scientific output was written.
- Kernel I/O accounting showed approximately `125348773888` bytes written before cleanup, so this is specifically an NFS allocation-accounting mismatch, not evidence that capacity admission passed.
- A same-fileset 128 MiB dense-random write/fsync/readback probe passed: file size `134217728`, allocated blocks `179560448`, matching SHA256 `8f6c6eaec27e7ca7b36eebc76e48808fc114e9805eeb2116a27a0952db10f014`; available bytes changed from `495034261504` to `494900043776` during the probe and returned to `495034261504` after cleanup.
- Disposition at that attempt: R2 remained unsubmitted. The recovery and final admission are recorded below; the failed card/root are retained unchanged.

## Final R2 preparation and capacity admission

- Runtime implementation is pinned to execution commit `b80855c186e13e7539dabbf9f5365ee55d664e26`; the exact remote checkout and sealed run card both identify this commit. The later report-only commit, if any, is not part of the training runtime.
- The mature reservation helper gained a narrow opt-in dense-write fallback; R2 explicitly requests it. The original helper behavior for other callers remains unchanged. A same-fileset 128 MiB invocation through the actual helper passed fsync, allocated-block, and full readback-hash checks.
- Exact Stage10 CPU self-test passed: weight-one value/gradient parity, masked inactive slots, finite loss, and XYZ-to-other gradient ratio 4.0. It loaded zero GR00T weights, made zero model forwards, and performed zero optimizer steps.
- Exact Stage10 native Trainer/action-head runtime preflight passed with GR00T commit `51d4c89f72fda44cbf77285c6a8114b52676b8a1`; it found native `loss`, `action_loss`, and `action_mask`, with weights loaded=0, forwards=0, optimizer steps=0.
- The final run-card verification passed: card SHA256 `72283a8039f1518d93e7274c35eb81d633dcf75cca4e2ba1f1d74922fd010969`, execution binding `3014f6cdd323a72120e825a4de37d44d5357e3481a92cf19b7d19fda5011e7d1`, and unchanged 10,000-row schedule SHA256 `c9c8f80cc8ea3549ccb0066582eeae1854018f578175836d2cb6bb6b26a3caa9`.
- Final same-fileset capacity proof is `PASS_NONSPARSE_FSYNC_RESERVATION`, SHA256 `bbfbc9f87d4ca5a70f670bc1fbbcda2469fdf96a2b96db4898947a52e6213554`. It reserved `125348740672` bytes across 12 dense-write/fsync/readback parts and verified `167565606912` allocated bytes. Reservation files were removed by the mature helper; available bytes returned to the recorded preflight value. The PBS quota command remains `UNKNOWN_NOT_EXPOSED`, not PASS.
- The run card binds the original PI0226 adapted checkpoint-1000 file inventory, fixed 16-window cohort, C_T stats/modality, original exposure schedule, seed, native precision/config, sole XYZ×4 loss intervention, public HF repo `leoyuer/pi0243-stage-a-admitted-v1`, and unused R2-only prefix `pi0244x-xyz4-m16-10k-v1/`. R0 evidence and its failed archive identity remain untouched.
- Live `qstat -u e1192548` showed no active job; the prior R0 screen job `628527.hopper-m-02` is terminal (`Exit_status=0`, `Stageout_status=1`).
- Current disposition: all R2 preparation/admission checks pass and the authorized training is ready for one submission. At this evidence point, no R2 training job has been submitted; model weights loaded=0, training/backward/optimizer/scheduler steps=0, and `selected_checkpoint=NONE`.

## R2 attempt 628673 — compute path access diagnosis

- `628673.hopper-m-02` ran on `hopper-26`, then exited at PBS wrapper line 37 after 2 seconds, before `nvidia-smi`, Python, model load, HF sidecar, or Trainer. Preserved PBS output reports 4,488 KiB memory, zero CPU time, and no GPU-related information.
- `qstat -xf` confirms the resolved launch bound `PI0243_RUN_CARD`, `PI0243_PROJECT_ROOT`, `PI0243_METADATA_MIRROR_ROOT`, `PI0243_OUTPUT_MIRROR_ROOT`, and `PI0243_ARCHIVE_ROOT` directly to `/Project_Storage/...`; the submitted arguments had `-P CFP03-CF-004` but no PBS `group_list`.
- Login-node `stat` shows `/Project_Storage/CFP-03/CFP03-CF-004` is mode `2770`, owned by `nobody:HPC_Users_3104`; the user is a member of `HPC_Users_3104`, but the failed PBS job did not request that execution group. The run-card leaf is readable to all after traversal, so missing group context at the parent is a specific, evidence-supported explanation for `test -r` failing. Compute-node effective groups/mount remain unobserved; this is not proof that the whole filesystem is absent.
- The prior 627454 Hopper training launch used `/scratch/...` for project, card, metadata, output, and archive roots. The mature `hpc/pbs/pi0243_stage.pbs` explicitly documents compute-local scratch to avoid heterogeneous Project_Storage mounts. Its qstat record shows it ran on `hopper-22` for 16m28s; it is precedent, but not by itself proof that R2's Project_Storage path is inaccessible on every compute node.
- R2 cannot simply move the output to the currently available scratch: login `df` showed 22 GiB free there, while the sealed local-retention plan reserves 125,348,740,672 bytes. The original card, capacity proof, and R0 evidence are therefore preserved; no scratch-only substitution is made.
- Minimal launch repair: request the already-authorized `HPC_Users_3104` group in the R2 PBS wrapper and emit compute `id`, mount source, and path-chain diagnostics before continuing. Login confirms the account's group membership and the relevant directory modes; the local wrapper passes `bash -n` and `git diff --check`; PBS `man qsub` documents `-W group_list=<group list>`. The wrapper change is not part of the sealed trainer code root and does not alter the run card, binding, capacity proof, or scientific recipe.
- The next single authorized R2 training submission must use this wrapper from a compute-visible submit path. Its early access checks run before GPU query/model setup; if they pass, the same job proceeds with the sealed recipe. No replacement job has yet been submitted; all R2 model/forward/backward/update/HF/checkpoint counts remain zero and `selected_checkpoint=NONE`.

## R2 attempt 628777 — Project_Storage absent on assigned node

- `628777.hopper-m-02` was assigned `hopper-23` with one GPU, 12 CPUs, 225 GB, and `group_list=HPC_Users_3104`; it terminated `F / Exit_status=2` after two seconds. The submitted PBS wrapper SHA256 was `2c585990aaf6556bdca1caa6426a54a14d2e5f84bc551b825e0c3cd8bfe846a4`.
- Retained stdout records `uid=22182(e1192548)`, primary group `HPC_Users_3104`, and that group in the effective supplementary groups. The wrapper's `namei -l` output ends with `/Project_Storage - No such file or directory` while checking the sealed run card. This narrows the cause on this assigned node to an absent mount/path, not missing `HPC_Users_3104` membership. Do not generalize this single-node observation to every Hopper node.
- The early run-card guard stopped before `nvidia-smi`, Python, model construction/load, HF sidecar, Trainer, or any optimizer update. PBS reports 0 CPU time, 4,908 KiB memory, and no GPU-related process information. R2 scientific counters remain zero; sealed run-card SHA256 `72283a8039f1518d93e7274c35eb81d633dcf75cca4e2ba1f1d74922fd010969`, execution binding `3014f6cdd323a72120e825a4de37d44d5357e3481a92cf19b7d19fda5011e7d1`, capacity-proof SHA256 `bbfbc9f87d4ca5a70f670bc1fbbcda2469fdf96a2b96db4898947a52e6213554`, and XYZ4 recipe are unchanged.
- On Hopper login, `df` shows Project_Storage mounted with 462 GB available, while `/scratch` has 22 GB available. The sealed 125,348,740,672-byte R2 reservation is on Project_Storage; the current scratch fileset is not an admissible substitute. This is consistent with the existing PI0243 storage guidance about heterogeneous compute-node mounts.
- The PBS `.OU`, `.ER`, and submitted wrapper were copied to `/Project_Storage/CFP-03/CFP03-CF-004/pi0244x_xyz4_m16_10k_v1_densecap/engineering_attempts/628777/` and verified against scratch SHA256s: `.OU` `b2b06eed7c9f1619541590856a0bea4e7058fef8f50cd6359c7780211e5f1b70`; `.ER` `203dd9c475e5939c4efb13ddc3f094e7bac51175594f873bf8a6fdf5e8263124`; wrapper `2c585990aaf6556bdca1caa6426a54a14d2e5f84bc551b825e0c3cd8bfe846a4`.
- No second PBS job or CPU probe was submitted. Current disposition: `EXTERNAL_INFRA_PATH_BLOCKER` pending a supported compute-visible durable storage route or site resolution for an eligible node class. Do not rewrite the sealed card, binding, capacity proof, or redirect outputs to undersized scratch. Preserve 628673, 628725, and 628777 records/logs. `selected_checkpoint=NONE`.
