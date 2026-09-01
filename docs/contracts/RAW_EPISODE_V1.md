# RawEpisodeV1

`RawEpisodeV1` is the immutable, source-rate recording contract for a future
Go2+ARX5 episode. It is the audit authority; a LeRobot table or a GR00T
processor input is never the only copy of an episode.

## Required semantics

- Preserve native timestamps and clock-domain metadata exactly as received.
- Retain body RGB, wrist RGB where available, language/task metadata, raw
  whole-body state, ARX5 joint state/commands, desired EEF target, achieved
  EEF, gripper command/measured values, calibration ID, and lifecycle/safety
  events.
- Keep desired and achieved EEF streams in separate fields. Achieved EEF is an
  audit signal and must not silently become a training target.
- Store EEF pose as absolute XYZ plus `GR00T_FIRST_TWO_ROWS` rot6d. The layout
  is explicit because the repository also has a distinct native-WBC wire
  layout.
- Arrays are copied and read-only at the Python contract boundary. The source
  JSON/image files remain outside this repository.

## Implementation

`go2arx5_vla.data.RawEpisodeV1.from_d3_json()` is a non-destructive adapter for
the D3 simulated fixture. It maps `state`, `action_target`, and
`action_achieved` without collapsing them, retains the 50 Hz source timeline,
and records the source JSON hash and simulation provenance.

The D3 fixture is `SYNTHETIC/ISAAC_RENDER_ONLY`, not a real robot recording. It
does not supply physical `T_Go2_ARX5`, `T_flange_TCP`, joint-zero calibration,
or clock-sync calibration; those remain `REAL_TBD`.
