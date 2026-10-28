# Fixed expert pilot result and interpretation

All36 prescribed expert-only pilots completed once, with0 successes and0 invalid engineering starts. M1/M2/M3 each0/12, below10/12. There are no task-phase observations/labels or TRAIN/MT-VAL demonstrations. The downstream collection and multitask training-admission chain stops.

Families000/001/004 produce settling orientation gate rejections; families002/003/005 terminate physically while settling. Task/command siblings share each family's actual reset/image/RNG identity. This pattern precedes the task intervention and supports a common preparation/controller-reference dependency; it does not identify a unique cause, establish intrinsic task infeasibility, or measure a learned model. No expert profile, gains, gate threshold, robot parameter or seed is changed to pursue a pass.

The first gate example rejects at actual0.219999995s with initial rotation discontinuity1.5764893137820062rad against the original1.5rad threshold. Registered settling orientation is constant. Original cached desired/achieved quaternion evidence reproduces that criterion within1e-5, but original full handoff arrays were not saved. Complete arrays added prospectively in recording V2 permit full numerical gate replay for12 later episodes; original24 results remain unchanged with their evidence deviation.

Same-tick clean settling desired/achieved XYZ/SO3 summaries, actual durations, gate/termination classes, hashes, runtime versions and paired reset fields are in PILOT/pilot_stage_summary.json and PILOT/pilot_cpu_replay.json. Native auto-reset can occur on termination; post-step terminal state/camera may already be reset. These rows are excluded from clean tracking and remain in raw evidence with termination events. Do not call a post-reset TCP the failure pose or reference quality task success.

Four videos use recorded synthetic body/wrist cameras, not extra simulator rollouts. Actual timestamps/raw NPZ hashes are in PILOT/video_manifest.json; playback timing is presentation only. Original pre-execution freeze, recording fix and count-correction comments5885426386,5885987788,5886006072 remain available.

The next scientific decision must address the frozen preparation/reference/platform dependency and unproven task observability under a new explicit version. Existing pilot FAILs cannot become engineering retries or successful imitation data. Contact/manipulation gaps remain separate; no deployment/generalization result is claimed.
