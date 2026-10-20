# Successor Draft: TRAIN-Only Action-Range Comparison

Status: **NOT_AUTHORIZED**. No T0, budget, resource lease, collection, fitting,
training, GPU, evaluation or model selection is authorized by this draft.

## One Question

Can the unchanged EEF10 relative pose codec express the saved, physically
successful TRAIN commands across the intended geometric coverage under a
predeclared TRAIN-only normalization range?

The actual motivation is AP0/-Y/NEAR/T2: its Teacher completes the object task,
but original stats/decoder produce a first SO3 error0.0001381733rad versus the
frozen0.0001rad gate at anchor209/action31. Channels4/6 are clipped there;
this is evidence for a range/roundtrip investigation, not proof that stats are
the sole cause, that student failures are fixed, or that a new model is needed.

## Controlled Design For A Future GO

Predeclare an eligible archive of actual safe, complete, raw-audited TRAIN
parents and a geometry-disjoint TRAIN fit/diagnostic split before inspecting
comparative errors. Include the saved codec-rejected physical successes in
diagnostics rather than selecting only easy currently admitted windows. Keep
all candidates/exclusions and the original frozen-stats baseline.

Compare original statistics with one TRAIN-fit range under the identical
codec/relative transform/inverse algorithm, same float64-to-float32 path,
same committed targets, all601 windows, SO3/XYZ/gripper thresholds and same
reader/collator. Record per-channel out-of-range values, first failing window,
full error distribution and actual event join. Do not assume relative
coordinate2 is world Z. A bypassed/clipped label is not a corrected reference.

The first study is saved-data CPU only, not new Teacher physics or GR00T
training. A fitted candidate is an explicitly new numerical version, never a
silent update of PV1 or the closed DATA scope. It must be disclosed even if
used only in offline qualification. Any later policy comparison requires its
own scientific GO and matched end-to-end evaluation.

WBC/plant/gripper PD/Teacher/cameras/language/S1/scoring/labels remain fixed.
DEV/CONFIRM/LOCKBOX data cannot fit or calibrate the candidate. Do not combine
this single intervention with new sampling, recovery, objects,12mm Teacher or
world-model changes. If the unchanged range is already sufficient in a
controlled reconstruction, reject the stats explanation and report the actual
numeric/interface discrepancy instead of manufacturing a new normalizer.

This is the single proposed successor for this failed data gate. PV1's existing
B0/matched A-B critical path remains independent and higher priority.
