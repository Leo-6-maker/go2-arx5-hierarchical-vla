# Stage33 dual-track bottleneck map

This draft protocol is bound to `PI_GATE_STATE_0015` and starts at accepted
Stage32 commit `440dae61758c90211fb98239abcfac8778124cba`. It does not modify
or rerun the sealed Stage32 job.

## MA: paired exposure robustness

The exact locked TINY1 sample is evaluated only with the stock four-step
N1.7 Euler sampler on H2@1000 and H2@2000. Historical seeds 2223--2225 are
reused from the immutable Stage31/Stage32 trace artifacts. New seeds are
exactly 2226--2238, one paired call per checkpoint, for 26 new sampler calls;
BASE calls are zero. Initial noise is captured and must match exactly within
each pair. The predeclared positive classification is at least 12/16 lower
H2@2000 active10 MSE and geometric-mean paired ratio below one; otherwise the
result is mixed.

## MB: inactive-122 nuisance diagnostic

The nine Stage26 direct-flow probes are reused unchanged at H2@1000 and
H2@2000. Each probe has four conditions with identical active10 noisy state,
target, conditioning, and timestep: stock plus three deterministic CPU
resamples of inactive dimensions 10:132. A fifth diagnostic condition zeros
only those inactive dimensions. The campaign has exactly 90 direct
action-head calls and no official sampler or training call. The material
criterion is the frozen median induced-active-MSE/stock-oracle-MSE threshold
of 0.10 together with at least eight H2@2000 zero-inactive improvements.

## MC and WA

MC is a CPU/static DROID semantic/projector mapping audit. It proposes copying
only compatible first-10 relative-EEF/absolute-gripper semantics and
neutralizing DROID joint/padded dimensions; it never edits a checkpoint or
trains a DROID-initialized candidate. WA first audits the exact node-local
Stage16 support. On this Hopper host the protected Stage16 native support root
is absent, so the zero-simulation return is
`NO_VALID_SUPPORT_AWARE_CONTINUOUS_COHORT`; no WBC rollout is fabricated.

All output is written under a new scratch-backed Stage33 root. HOME, `/tmp`,
and historical Stage22--Stage32 trees remain untouched. Any post-first-call
failure preserves partial records and stops without a replacement PBS job.
