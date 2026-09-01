# ADR-008: Reference Completion vs Fixed-Step Rollout

## Status

Accepted for Stage16.

## Context

The native environment advances a time-law reference, computes forward-projected progress, and may reset an environment immediately after a termination decision. A rollout that contains fewer than 100 logged steps is therefore ambiguous: it may have completed its reference, timed out, fallen, become invalid, hit a joint limit, or aborted on tracking/timing. In addition, `traj_s` can saturate at `traj_L` before the reference time-law reaches its end.

## Decision

Native benchmark completion is decided by the explicit reference/episode lifecycle. The harness records reference duration, progress, timing/lateral diagnostics, timeout/fall/invalid/joint-limit signals, and a pre-reset snapshot. It assigns one of the Stage16 termination codes, with `REFERENCE_COMPLETE` reserved for the completed reference interval and `TRACKING_ABORT` reserved for an explicit native early-abort condition.

The benchmark does not pad or force every rollout to 100 control steps, and it does not infer failure from `steps < 100`.

## Consequences

Normal reference completion is not counted as tracking failure. Post-reference idle samples are excluded from tracking metrics. A reset event without a preserved pre-reset snapshot is not sufficient evidence for a precise root-cause claim.
