# PI0248 G0 mechanism report

**Protocol:** `PI0248_M64_POSE_REPAIR_PAIRED_PILOT_V1.md`
**Authorized commit:** `81e3faa4eb23e6ce0ed521e6b8dd5b7206506771`
**Issue #11 authority:** `5847759659`
**Status:** G0 read-only analysis passed. No new model queries were made.

## Scope and identity

G0 reused the sealed 64-source cohort: the original 16 and the 48 additions. The 64 sources span 61 unique rollout bases (16 for old16 and 46 for new48). Existing D2 screens and fixed processor/source identities were reused; D1, D2, and M64 training were not rerun.

The processor mirror is bound to frozen (C_T) statistics from `train_dataset/meta/stats.json` (SHA-256 `c24d465bc4acd594beb041b42f2e0f4ea955a67c7714e178ffdb313a61adb3f4`). The model-side base statistics differ from this parent binding, so the continuation must explicitly retain (C_T); it must not silently substitute the model sidecar.

## Training exposure reconstruction

The historical M64 training manifest and actual consumed sample rows reconcile exactly at global step 5000:

- Schedule SHA-256: `29b16f7a885617d4200db1126ee09517c147b4ceaaad1709e116466710c278da`.
- All 5000 planned rows match actual consumption, including the 1000-row prefix and 4000-row exact-resume suffix.
- old16 received 1250 updates; new48 received 3750.
- 56 sources received 78 updates and 8 sources received 79 updates.
- Forbidden reads/operations in the run record are all zero.

This is the historical M64 exposure, not the new C/R 2000-update schedule.

## Pose and position findings

The original 16 have 32 paired U-versus-M64-final cells. All 32 preserve the same scoring input and noise identity.

| old16 paired median | U baseline | M64 final | M64 minus U |
|---|---:|---:|---:|
| XYZ RMSE | 0.01319 m | 0.01771 m | +0.00601 m |
| SO(3) RMSE | 0.02154 rad | 0.11232 rad | +0.09493 rad |

Across old16 B0 screens, SO(3) error rose from 0.01918 rad at step 0 to 0.09184 at step 1000, 0.10784 at step 2500, and 0.11438 at step 5000. Position error rose early and later recovered to 0.01670 m. The final B1 medians are 0.01970 m and 0.11175 rad.

For new48 B0, position RMSE improved from 0.11550 m at step 0 to 0.02578 m at step 5000, while SO(3) remained around 0.10–0.12 rad (0.11849 to 0.10836). At step 2500 the medians were 0.04262 m and 0.10409 rad.

Final M64 gate counts:

| Cohort | Noise | Accept | Warn | Reject | Median XYZ | Median SO(3) |
|---|---|---:|---:|---:|---:|---:|
| old16 | B0 | 1 | 15 | 0 | 0.01670 m | 0.11438 rad |
| old16 | B1 | 1 | 15 | 0 | 0.01970 m | 0.11175 rad |
| new48 | B0 | 2 | 45 | 1 | 0.02578 m | 0.10836 rad |
| new48 | B1 | 2 | 46 | 0 | 0.02550 m | 0.11173 rad |

The pooled position-tail (steps 35–39) SSE fraction medians are 0.1831/0.1758 for old16 B0/B1 and 0.1433/0.1545 for new48 B0/B1. These position trends do not explain away the persistent pose error.

## CPU decoder and gradient verification

The differentiable rotation mirror matches the pinned decoder on all 128 final-M64 B0/B1 cells across H40. Maximum physical rotation-matrix absolute difference is (2.98\times10^{-7}), below the (10^{-5}) protocol tolerance; all 128 anchors are nontrivial.

The check also verified clipping/min-max inverse, row-vector 6D convention, relative-to-absolute composition, known-angle chordal scaling, masking, exact zero-coefficient value and gradient parity, finite differences, and restoration of model mode. The finite-difference gradient was (-0.188662), agreeing with the analytic value (-0.188670). Status: `PASS_CPU_DECODER_AND_GRADIENT`.

## Interpretation and next action

The read-only evidence identifies a persistent orientation regression alongside improving position fit; it does not establish why training caused that regression. The narrow C/R paired pilot is therefore the next authorized test: C retains the original U objective, and R adds only the protocol-defined physical rotation term. It will use the exact M64-final-5000 revision, the fresh shared 64/2000 exposure table, full S4 rollout gradients, and separate B0/B1 final-2000 gates. No padding intervention or additional loss/step search is justified by G0.

## Evidence

- `pi0248_m64_breakdown_v3.json` — SHA-256 `c777db45a642ed836fdd4c426d8f9f5506d2416d2738b97f1c40089532d67933`.
- `pi0248_m64_exposure_audit.json` — SHA-256 `cc4ea9636a7fb496490915e3dce9a479fb2328e355104bb48cc8a8c796c09fe2`.
- `pi0248_rotation_cpu_check.json` — SHA-256 `9f533ad139191d788472d3323030b2ba7f6a6ae96f8d2f6f4ae2d233a8bdd56`.
