# PI0265 G1 — mechanism readout and scientific stop

Overall **PARTIAL / PI0265_CONTROLLER_ENVELOPE_BLOCKED**. All four C0 and sixteen fixed DEV cases are retained. No confirmation, FEEDBACK_TRAIN collection, R/D optimization, GR00T query or learned-policy episode followed. Historical PI0264 FAIL and formal selected/deployment NONE remain unchanged.

## Reference versus actuation

Both C0 routes have byte-identical reset/RNG and installed native tables within each representation. On family000, canonical terminal positions differ from requested terminals by at most 2.99e-7 m; terminal orientation differences are at most 2.25e-7 rad. G0 additionally checked actual native current/heading-preview transformations on nonidentity rotations. These results support the tested coordinate/installation contract; they do not prove every possible WBC reference or controller state.

Family000 M1 achieved final positions are already close under LEGACY (6.1/7.8 mm), but attitude errors are 0.674/0.680 rad. Under ROW6D, the same two fixed commands finish at 3.1/4.0 mm and 0.042/0.055 rad and satisfy the holding and successful-episode tracking criteria. C0 improves in the same limited comparison. This is evidence for a useful conditioning intervention in these cases, not a unique or universal root cause. Installed training-time PyTorch3D remains UNKNOWN; weights, native bounded quaternion-vector feature and dynamics were not changed.

## Endpoint hits are not M3 completion

Family000 ROW6D M3 endpoints finish at 3.35/4.19 mm and 0.032/0.041 rad, but the pre-frozen full-path checks do not pass. Command0 never satisfies start admission and receives zero ordered bins. Command1 receives all twelve bins but records reverse/gap flags under the frozen strict angle implementation. Its SO3 tracking RMS is 0.107 rad; command0's is 0.157 rad, both above the 0.10-rad dynamic qualification limit.

Those strict path flags are reported as the implementation's flags; twelve credited bins must not be described as a missing arc. No post-outcome path tolerance or holding rule was changed. Even a more generous interpretation of these two endpoint/path observations could not meet the registered 7/8 gate, because family001 contributes four safety failures to each representation.

## Independent startup collision

The normal registered reset for family001 (seed828632358) produces measured-FK/URDF self-intersection between `base` and `x5_link8`, penetration approximately 0.002706 m, after the counted 0.02-s implicit reset step. All four task/command cases per representation stop before an actor forward. This is a forbidden-pair geometric check on the actual moving-root/joint state with the preserved fixed/adjacent collision exclusions, not a fixed-base IK veto or a native force-termination claim. The shared reset failure cannot be attributed to the choice of 6D packing. No substitute reset, scene, seed, teleport or settling search was attempted.

## Accounting and durable evidence

Seven initial zero-actor starts exposed an added diagnostic's support-contact classification defect. Their raw receipts were preserved and individually marked ENGINEERING_INVALID; the source-bound repair excludes ordinary ground support from the added self/obstacle diagnostic and leaves native ground termination unchanged. These are separate from the subsequent eight genuine startup self-collision failures. Total accounting: 27 starts, 9027 control ticks, 36108 simulator calls, 18000 arm/dog actor forwards, 180.539996 simulated seconds; zero GR00T queries/optimizer updates.

Independent CPU world-FK/metric replay covers every valid case. The actual-camera video contains twenty chapters, 660 decoded frames, 66 seconds at10fps with explicitly marked4x playback and paused last frames. The public fixed-revision archive includes all raw valid/invalid attempts, actual noise packets, source receipts and code, with anonymous full-file and all21902-member readback. See `PUBLIC_RECEIPT.json`, `CONTROL_QUALITY.csv`, `CPU_REPLAY.json` and `VIDEO_RECEIPT.json`.

The registered controller envelope is not admitted: LEGACY0/8 and ROW6D2/8, M3 zero qualifying successes. The scientific dependency chain stops here; this is not a completed feedback-training or learned-policy closed-loop pilot.
