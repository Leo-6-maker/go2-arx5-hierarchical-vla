# PI0248 C startup failure and engineering recovery

Date: 2026-09-27 (Asia/Singapore)

## Failed PBS attempt

- Job: `634051.hopper-m-02`; PBS terminal state `F`; exit status `1`.
- Allocated H200 wall time was 3 seconds; CPU time was 0 seconds; memory use was 21,608 KiB.
- The traceback reaches `pi0247_lu_train.train()` at `_verify_code` and ends in `RecursionError`. The PI0248 adapter replaced the legacy verifier before capturing it, so the wrapper called itself.
- The attempt stopped before model construction, any S4 forward, or an optimizer update. `/C/training` and `training_manifest.json` are absent. No checkpoint or science query was produced.
- Original C run card SHA256: `bb93895a115193c38b21900510b69427d6e5948f644330fecea88166031aec68`. Original R card SHA256: `8b9242d715fe62615d4a47f106db29e7d9e6ea274317c1ac568bb3aafa20e857`. Both remain unchanged.

Raw PBS files copied from Hopper and retained here:

- `634051.ER` SHA256 `d196f5de61e77530bc5a07f8b17957d406c3c062fa1043842a337880d8497ed2`
- `634051.OU` SHA256 `0ba9448592bdeb841b2dae1317eb3b6cf285f8ed719900006cd178ed87579ed3`

## Repair and sealed recovery cards

Commit `744f1265b061b418c38d6561de45de103b08a6e7` fixes the root cause by capturing legacy verifier functions before installing PI0248 adapters. It also adds the PI0248-specific checkpoint screen runner; the frozen PI0247 input builder, decoder, and safety gate remain the evaluation basis.

A code-only call through the installed adapter verified the original verifier binding and returned `PASS_PI0248_ADAPTER_CODE_BINDING_NO_TRAINING`. Python compilation and `git diff --check` passed. No model was loaded during this check.

New sealed recovery cards and outputs are under:

`/scratch/e1192548/pi0248_m64_pose_repair_20260927/recovery_engineering_fix_744f126/`

- C card SHA256: `93c65de7cfa183e363f0c2a5822d117a45f4809b69166b94b57a294edacae104`
- R card SHA256: `c3ba70840764e961a27eb897c5f615b114f9937661ee4fc837e843abd34b1ca1`
- Shared schedule SHA256: `339978cc507a55d69044532b4786f0365690b477aec3927bfa4c489f3baf65fc`, identical to the original sealed schedule.
- Both cards retain 2,000 updates, checkpoints at 1,000/2,000, LR `1e-5`, S4, the same C/R objectives, and `selected_checkpoint=NONE`.
- Public HF preflight passed for both arms with identity verified and each target prefix empty. C probe commit `2f72538fd69ca29e16af98dffe5cd4842aaab51e`; R probe commit `6eb1c2c36f9a7a16e36a1edc0cebd27954b29bd1`.

No recovery training job has been submitted yet. The next action is one C submission using the recovery card; R remains serially gated on C completion. The 2,000-update budget and 512-screen budget are unchanged.
