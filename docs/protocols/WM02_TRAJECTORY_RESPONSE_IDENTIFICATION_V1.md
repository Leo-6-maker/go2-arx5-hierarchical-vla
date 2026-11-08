# WM02 one-direction trajectory response identification V1

This is a new planned data/model contract, not an executed experiment. No
existing XY response data or old parameter/normalizer/threshold is rebranded.
The slow full-geometry trajectory must first pass its own complete native
qualification; earlier graded passes do not grant the full domain. Every
rejection and partial run remains evidence. The offline modules start no
worker, query, physics, Actor or optimizer.

The narrow hypothesis is whether observed tracking lag/history changes the
response to a signed X reference pulse, beyond a static gain and an additive
same-history lag/velocity forecast. It is not a full XYZ/rotation/gripper
Jacobian, exact counterfactual experiment, or claim of native correction gain.

## Frozen prospective panel

`fusion_trajectory_response_contract.make_panel(physical_parents=...,
source_sha=..., query_pins=...)` constructs the exact
`WM02_TRAJECTORY_RESPONSE_PANEL_V1` schema. The source is a full 40-character
SHA; query pins are full hashes for checkpoint, spec, result, action,
observation and frame map. The nominal recipe is explicitly
`FULL_H40_RIGID_RELOCATE_SCALE_WORKING_GRIP_INTEGER_RETIME_2X_V2`, geometry scale
1 and time stretch factor 2. The numerical probe has one common clock for all
levels. No theta-specific retiming, clipping or gripper/rotation change occurs.

There are four new TRAIN and two new DEV physical parents, two contexts C0/C1,
levels `[-1,0,+1]`, and two independent restarts per cell: **72 physical runs**,
48 TRAIN and 24 DEV. The first 12-run single-TRAIN-parent pilot occupies its
original cells and may be reused when the complete frozen panel exists; it
cannot fit this model or be counted as 72 independent examples. Execution
order is frozen and balanced/randomized prospectively by the original owner;
there is no new scheduler here and no retries to select successful outcomes.

C0 holds the original INIT reference. C1 rises +0.5mm in target-world X during
the first eight 50ms atoms from step2503, holds until step3503, then returns
quintically over the last eight atoms to original INIT at step3703. Both
planned decision endpoints are INIT. Actual decision TCP and recent histories
are measured separately; the original steady screen remains unchanged. A
context label does not prove that actual states were sufficiently separated.

At step3703, after sealing the actual prefix, the signed pulse is first
installed: +/−0.5mm X, quintic rise8, hold through atom16, return by atom24.
The pulse changes only the reference; desired nominal outcomes remain fixed.
`ContextProgram` and `ResponseProgram` expose complete immutable paths and
full-H40 suffix previews. All three final candidates still require the
original native certifier. Diagnostics do not guarantee joint acceleration,
collision, contact, TTL, consumption or domain qualification.

Each exact `WM02_TRAJECTORY_RESPONSE_EPISODE_V1` contains the panel hash,
recipe, scale, query pins, run, scene_id, parent, split, context_id,
theta_level, restart_index and common planned_end_atom. The new native
collector authenticates source/admission/predecessor files separately.

## Replay and causal input

`fusion_trajectory_response_replay.response_window(record, panel)` accepts
authenticated episode, native result, full collection report, sealed decision,
original raw-H40 NumPy array and five file hashes (native result, report,
episode, decision, raw action). File/lease authentication is the caller's
responsibility; the pure function grants none. It rejects old phases, missing
consumption, incomplete native/trace/shutdown evidence and raw dtype/byte
changes. It joins the nine prefix states and eight past references back to
their actual predecision records.

`build_replay(records, panel)` preserves every actual installed H40 from every
atom, all original raw40 values/bytes identities, complete nominal/corrected/
desired paths and source pins. V1 learning uses **one predeclared H16 probe
per physical run**. Dense overlapping windows are not independent examples.
An audit-only incomplete replay can be made with `require_complete=False`;
all fitting and noise estimation still require the complete 72-cell cohort.

Inputs are nine actual 106-vectors, eight past consumed EEF10+BASE7 references,
the full known nominal H40, declared pulse basis and candidate theta, plus
the actual current source-age/TTL. State-vector rotations use columns; native
EEF10 uses rows. Future TCP is label-only. IDs, context labels, future
observations, desired outcomes and absolute world translation/native clock
are not regression features. All 40 nominal rows enter fixed temporal bins.

TTL remains in seconds and is converted explicitly: `.4s` equals eight
continuous 50ms atoms but only seven complete live atoms under the original
strict expiry. H16 prediction crosses renewals and is explicitly
`PREDECLARED_PROGRAM_CONDITIONAL`; it does not use future observed renewal
clocks or claim that an uninstalled suffix has already been executed.

## Models, fair comparators and freeze

`TrajectoryResponseStudy` predicts direct H16 XYZ displacement anchored at
current actual TCP. All learned regressors use fixed NumPy ridge=1:

- TRAIN equal-parent/context/restart static response curves for each theta.
- Nominal-only residual over those curves.
- `ADDITIVE_LAG_VELOCITY`: the same known nominal plus current XYZ lag, last
  actual velocity and mean past tracking error, without history×theta.
- `CONDITIONAL`: the same features plus the nine history scalars×theta.
- `CORRECTION_BLIND`: retains nominal/history and the equal-level TRAIN prior,
  removing theta from both features and base. Corrected path differences are
  excluded too; poisoned future correction bytes cannot affect its forecast.

Deterministic current-lag transport, constant velocity and current HOLD
forecasts remain visible. They are forecasting comparators, not evidence that
ordinary feedback control was physically evaluated. A later native comparison
must give ordinary feedback the same history, candidates and opportunities.

Every TRAIN leave-one-parent-out fold refits its static curves and own
normalizers without the held parent. Threshold provenance is new TRAIN only:
`train_noise_threshold(..., fraction=..., floor_mm=...)` estimates single-run
coordinate scatter as `sqrt(mean(same_cell_restart_difference_mm**2)/2)` and
returns `max(predeclared_floor_mm, predeclared_fraction*scatter)`. Fraction and
new-domain floor must be frozen without DEV; they are required arguments, not
chosen by this module. The exact TRAIN manifest and method are saved. The old
XY threshold 0.08124448952310876mm is not a new-domain threshold.

Conditional is selected only if its held-TRAIN gain exceeds that threshold
against both static and same-history additive models. Otherwise the additive
model must beat static meaningfully, or selection falls back to static. All
four fits and negative CV results remain saved. This uses 20 closed-form
ridge solves, five static fits, 20 normalizer fits and zero optimizer updates.

`save` creates an exclusive portable artifact. `load` reads it without pickle;
`verify_reload` requires a distinct file-reloaded object, exact arrays and
exact TRAIN forecasts for all three candidates before enabling DEV forecasts.
A refit invalidates that verification. No DEV label enters fit, normalizers,
noise threshold, selection or the saved freeze. Report H1/4/8/16 XYZ and XY
separately, physical-parent/run counts and independent-restart scatter. Even
positive exposed-DEV prediction results do not establish control benefit or
generalization to another nominal trajectory, phase, correction direction,
scale, rotation, gripper/contact condition, or exact hidden-state fork.
