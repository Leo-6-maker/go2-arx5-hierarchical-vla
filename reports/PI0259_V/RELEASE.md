# PI0259 Agent V release and handoff

**PI0257_V2_RELEASE_FROZEN.** Bound source commit
`df5517d89561ffc32e41ae2fe6b315460853c8a5`, original source manifest, ZIP,
portable QA, Corelab QA and fixed HF receipts. Targeted identity checks passed;
all 183 local HTML targets exist. Normal-speed versus 0.25x media are distinct,
raw H40 versus sampled native metrics remain labelled, and physical execution
remains `GUARD_STUB_ONLY`. No new substantive defect or v3 was introduced.

The frozen archive is 88,268,120 bytes, SHA256
`8a64517851b101750f8afbc0873ba54d124bfe7b1ef77f1d05a704906a263d3e`.
Existing 96-cell CPU replay, 22 full-video decodes and 213 anonymous HF file
readbacks were reused, not repeated. See `PI0257_V2_RELEASE_FROZEN.json` for
the finite binding list and exact receipt identities.

[Fixed HF v2 content](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/88a9dc144cd0bb3263279f952fcf2f6fa8a0b4dd/evidence/PI0257/corelab-video-review-v2-20260929).
Receipt revision: `0e02ca5be5a74e65471f0b4f2dd6b2b825a19fd0`.
PC index: `D:/Users/GR00T project/Projects/go2-arx5/PI0257/v2/pc_bundle/delivery/index.html`.
This is a Windows PC location, not a Corelab path.

**DATA_VISUAL_QA_TOOL_READY; H0 CORRECTNESS_BLOCKED.**
Entry and exact supported bindings: `scripts/visualization/pi0259_review.py`
and `PI0259_REVIEW.md`. CPU negatives cover target/command changes, quarter
coverage, frames, paired data, collator leakage, seals, readonly paths,
forbidden imports and the physical guard. H0/FK/mask cards and H2 saved-output
cards execute on explicitly synthetic test fixtures. No fixture is presented
as experimental evidence. Automatic numeric reviews remain named partials
until the packet-specific source/semantic checks and natural RGB review close.

Agent M delivered H0_v1 during tool preparation. Its 491 explicitly listed
files (30,289,637 bytes) were copied read-only and verified against manifest
`94954516f9b1e9159e7fe468c04d7155e6cc20eb4d21c601b8389e8ab2dcac62`.
Only those sealed paths were read; no producer directory was modified.
The actual layout adapter `pi0259_h0.py` reviewed 144 original image pairs,
18 exact common-start repeats, raw segmentation/depth, full camera matrices,
URDF FK, StateV1 and saved reference geometry. All 144 source-labelled cards
are available in the PC H0 review index. H1/H2 have not been delivered.

The independent result is `CORRECTNESS_BLOCKED` for common-rig observability,
not a media-layout block. Rig0/2 support only M3 at the fixed common starts.
Rig1 supports M2+M3 at common starts but cannot complete the registered poses:

| Task | Saved later poses / 48 | Visibility failures among saved | Missing geometry poses |
|---|---:|---:|---:|
|M1|40|9|8|
|M2|36|0|12|
|M3|32|14 (black endpoint <4 pixels)|16|

Independent FK-prefix qualification reproduces M1 10/12, M2 0/12, M3 8/12.
Saved collision contacts are explicitly AS_RECORDED; V did not rerun FCL.
Initial requested pose equals actual FK/old nominal/identity; no old-start
defect is claimed. Maximum saved FK error 2.22e-16; StateV1 error 2.11e-8;
full camera rotation matrix delta 2.88e-7. Depth backprojection on 4466
visible spherical cue/camera samples has maximum surface residual 0.194 mm,
consistent with the saved camera/geometry bindings. This residual is not a
new admission threshold. The native destructor's later exit does not erase
the hash-verified observations or independently justify discarding every cell.

H0 machine receipt SHA256:
`1e16c79ad6d7c1f98318cf8d151378b27b050dcaeb2068c3874b815c8fabcde7`.
PC: `D:/Users/GR00T project/Projects/go2-arx5/PI0259_V/H0_review_final/index.html`.
Raw packet public source reported by M: HF revision
`6c0d169637f836190eeb7250eee018d9a392bb43`, prefix
`evidence/PI0259/H0-observable-fk-calibration-v1-20260929`.
V's independent byte source is the explicitly delivered Corelab manifest,
not an independently repeated anonymous HF readback. Private URDF binding
was checked locally and is excluded from Git/review deliverables.

Waiting for any subsequent sealed correction/H1/H2 is a finite handoff state.
No monitoring service, background polling or scientific retry was started.

Finite Corelab resource refresh: corelab-4080s node cache 946 GiB free;
GPU memory 8139/32760 MiB and 7341/16376 MiB. Agent V used neither GPU.
Quota command unavailable: quota UNKNOWN. Local D: had 246.77 GiB free;
this CPU tool/report write is below 1 GiB with more than 10 GiB reserve.
No process changes, producer mutation or cleanup.

Agent V new model loads/queries/training/renderer scenes/physics/WBC/Hopper
submissions: **all 0**. Historical FAIL/NONE and selected/deployment NONE
remain. Imported v2 includes its historical inference entry for provenance;
it was not invoked by Agent V and is outside this authorization.
