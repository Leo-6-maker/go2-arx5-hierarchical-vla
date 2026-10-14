# PI0294 Dual-Task Dataset Card

Status: v3.0 CPU-qualified and delivered; v3.1 frozen export and complete CPU
qualification passed. Neither dataset proves model skill. The first full
codec attempt timed out at its original 2400-second stage limit; its failure
remains. A distinct frozen-bank attempt passed without export or stats refit.

All new scenes were fixed before rollout. T1 pick-and-place and T2 pushing
share robot/object/target/camera/light initial conditions; only language and
audit identity differ. Family F0 is baseline, F1 spatial variation, F2 visual
variation and F3 combined variation. Physics and safety gates were not relaxed.

| Bank | Planned New Attempts | New Admitted | Preserved Rejected | Reused Legacy | Native Episodes | TRAIN / DEV H40 Windows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v3.0 | 80 | 78 | 2 | 80 | 158 | 75726 / 19232 |
| v3.1 | 224 | 215 | 9 | 80 | 295 | 149048 / 28247 |

v3.1 includes v3.0, not 224 additional episodes. Its extension produced
144 first attempts, 137 admitted and 7 additional rejections.
The legacy 64 TRAIN/16 DEV episodes retain their original identity and bytes.

| Bank | T1 TRAIN | T1 DEV | T2 TRAIN | T2 DEV | Planned New Groups | Groups With Both Tasks Admitted |
| --- | --- | --- | --- | --- | ---: | ---: |
| v3.0 | 32/32 | 8/8 | 30/32 | 8/8 | 40 | 38 |
| v3.1 | 93/96 | 16/16 | 91/96 | 15/16 | 112 | 106 |

v3.1 has 109 groups with at least one task admitted. Six static field-of-view
failures affected three TRAIN pairs: F0/010, F1/028 and F3/008. They performed
zero integrations and are preserved in the attempted denominator. Three T2
episodes physically succeeded safely but failed the independent reconstruction
limit of 1e-5: TRAIN F1/011, TRAIN F2/003 and DEV F3/000. None was replaced,
renumbered or repaired by relaxing the audit.

TRAIN/DEV group intersections are empty in both actual exported manifests.
Including legacy groups, v3.0 has 96 TRAIN/24 DEV groups; v3.1 has 157/32.
These counts include single-task legacy groups and are not counts of fully
admitted new task pairs. The independent confirmation set remains sealed.

Each admitted episode has 641 observed states, 640 committed commands,
601 full H40 windows, dual RGB and calibration. Input is RGB/language/measured
EEF and gripper; targets are committed future absolute EEF10 commands.
Privileged object/contact/stage/task/success information is audit-only.
The official processor applies the relative conversion once. Each bank fits
statistics from its own TRAIN only; DEV does not refit those statistics.

v3.0 all-window official codec and native 0/2/4 loader checks passed. Its DEV
float32 inverse clipping affected 27 components, with 190 state preclip
components; maximum decoded XYZ error was 0.000310253 m and SO3 error
0.000137690 rad. These are disclosed, not silently removed or refitted away.
v3.1 checked all 149048 TRAIN and 28247 DEV windows. TRAIN decoded maximum
XYZ error was 9.929381e-9 m and SO3 error 4.214685e-8 rad, with zero float32
inverse clips. Four preclip components before float32 conversion are retained
in the report. DEV had five float32 inverse-clip components and 175 state
preclip components, maximum XYZ error 0.000173080 m and SO3 4.214685e-8 rad.
The official 0/2/4-loader consumption/resume checks passed. The warning that
four worker processes share three reserved cores is preserved; the short CPU
throughput measurements are not GPU training benchmarks.

M512 trained only on v3.0. The new v3.1 statistics were not hot-added.
Fixed hashes and private HF revision are in DATASET_MANIFEST.json; private raw
paths and the full native member inventory remain in the registered runmap.
v3.1 is not yet HF-delivered, and HF private capacity also blocks M512 delivery.
