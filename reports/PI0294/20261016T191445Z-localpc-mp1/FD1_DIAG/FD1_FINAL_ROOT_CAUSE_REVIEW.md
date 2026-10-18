# FD1 Final Root Cause Review

2026-10-19, Asia/Singapore. **Diagnostic execution complete with one explicitly unqualified geometry case. No training GO.**

## Answer First

The dominant measured immediate failure mechanism is unsafe student reference execution near the table, not an initial-approach label and not a demonstrated WBC defect. Twenty of21 original collision vetoes have exact first pad/table geometry; one remains unqualified. This identifies the original S1 trigger, not a unique neural or demonstration root cause.

All40 original conditions and42 attempts remain: T1 10/20, T2 4/20,23 S1 vetoes,3 complete object-goal failures,0 actual native safety faults. Original14/20 targets remain unmet. Selected/deployment remain NONE.

## What Is Observed

- All14 T2 vetoes occur at12-28.8s after real object contact and motion;11 already exceed10cm displacement. Observer AUDIT_ONLY_approach is not actual initial approach. Exact original500Hz endpoints refine the phase matrix; the original intermediate20Hz snapshot is not substituted for the final native tick.
- Four coarse event classes change when the actual final500Hz tick replaces the last20Hz pose (which can precede rejection by50ms). Final T2 veto classes are7 contact-active transport,5 separated/in-zone and2 separated/out-of-zone after displacement. These event classes do not declare task success.
- All22 qualified reconstructions have original highrate error0, EEF observation error0 and full native IK/control copy error0, below the unchanged1e-10 gate. Original S1 category and exact offset tick match. Model XML and26 delivered asset members match their original SHA256SUMS; all15 frozen numerical files are unchanged.
- First-collision geometry is20 pad/table cases (9T1,11T2), using ARM_TABLE_OR_FLOOR. Their signed contact distances span -7.8826802e-05 to -6.47285075e-08m. These are not4mm penetration-threshold events. Rejected prefixes have0 actual live ticks.
- At the active request point of all20 pad/table vetoes, requested Z is below the predecision TCP. In11/20 passive static projections, requested XYZ alone produces negative pad-table vertical clearance; SO3-only and gripper-only projections do so in0/20. This supports a translation/clearance mechanism but is NOT a same-state dynamic expert-channel ablation; servo history and combined channels still matter.
- Realized safe-history minimum pad vertical clearance in the11 qualified T2 collision cases spans about0.114-7.240mm. Nominal expert4mm is a static design value, not this student distribution and not an expert dynamic margin measurement.
- The noncollision vetoes reproduce TCP speed0.402211746m/s against the original0.4m/s rule, and JOINT_LIMIT. All arm margins in the latter are positive; frozen rule inspection locates the finger-validity branch, but the exact offending finger side/position was not logged. No arm-limit claim is invented.
- During the terminal41 observations, T1 complete failure has release0/41 and TCP-clear0/41. T2 complete failures have highrate lift7.909mm and7.559mm, first exceeding5mm at18.496s and20.070s; terminal zone0/41 despite released/separated41/41. One briefly enters the zone at17.65s before leaving. Motion alone is not success.

## What Is Not Proven

The original ordinary expert dynamically succeeded on30 old TRAIN scenes with all8 stages, but dynamic feasibility and signed margin on these new layouts were never demonstrated. Static qualification cannot establish that. Family confounds geometry and appearance; paired noise/action differences precede state divergence, but do not prove a unique long-horizon cause. No new same-state expert intervention, channel substitution, inference, label or training was authorized or performed.

The four discordant scene pairs have identical initial state/pixels, different first H40 actions at t=0, and first recorded actual camera/TCP differences at0.05s. Original model-query inputs only recur every0.4s; this is not the first physical divergence time. The full10-scene paired denominators, both banks and all failures are retained.

One spent T2 F0/000 bank29201 replacement case remains RESTORE_UNQUALIFIED: the initial diagnostic extra snapshot changed the reconstructed-state fingerprint. Its history/measurement attempt costs are included; it was not retried. The read-side observer was changed to use the already captured state; all21 previously unattempted independent cases then completed under a new cold source. The missing collision pair is not filled from the other bank or another model.

Original hidden IK/Python/RNG snapshots were not recorded. Complete deterministic history and observed-state/native-copy parity are measured; bitwise identity of unrecorded original state is not claimed. This distinction and the one missing geometry case are retained in the evidence-grade matrix.

## Single Next Decision

Recommend only a **new TRAIN-only dynamic expert-clearance pilot**, original4mm reference versus one preregistered larger-margin reference, with WBC/plant/gripper/S1/score unchanged. First determine whether the nominal expert reference itself has fragile dynamic margin and whether a larger-margin task path remains safe and task-successful. This is a falsifiable qualification proposal, not a decision to retrain or proof that expert clearance caused the policy failures.

The proposal is NOT_AUTHORIZED and does not reuse opened confirmation geometries for labels or selection. No capacity change, FlowDAgger, EEF13 or WBC compensation is selected in this scope. No next experiment auto-starts.

## Evidence And Closure

[42-attempt provenance](FD1_42_ATTEMPT_COVERAGE.json), [40-cell physical phases](FD1_40_CELL_ACTUAL_PHASE_MATRIX.json), [first geometry](FD1_S1_FIRST_GEOMETRY.json), [native parity](FD1_SHADOW_NATIVE_PARITY.json), [paired/terminal analysis](FD1_EXPLANATORY_TABLE.json), [evidence grades](FD1_EVIDENCE_GRADE_MATRIX.json), [resource final](FD1_RESOURCE_FINAL.json).

Actual FD1 shadow use:177340ticks,45engine starts, each original case at most once. No new live episode, GPU/PBS, model generation, Teacher demonstration, optimizer update or training label. CoreCPU includes measured workers plus explicit conservative control uppers; local coordination uses an explicit3core-hour upper, not a utilization claim. FD1 lease and all workers are closed; oldN1 lease bytes, clocks, scores, data, models and STOPs remain. TF-R3 STOP stays; fixed-base source success is not Go2/WBC or real-robot proof.
