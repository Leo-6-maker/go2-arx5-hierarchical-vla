# Stage24 A4/A5 — Adaptation Recipe, Exposure, and Upstream Precedent

Date: 2026-09-03 (Asia/Singapore)

## Exact O1 recipe reconstructed

The recipe was reconstructed from the sealed Phase-A run manifest and its
exact `checkpoint_200/experiment_cfg/config.yaml`. It is not inferred from a
checkpoint directory name.

| item | value |
|---|---|
| GR00T SHA | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` |
| base index SHA | `407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59` |
| embodiment | `new_embodiment` / `NEW_EMBODIMENT` |
| physical action / universal action | `10D` / `[H=40, D=132]` |
| representation | relative EEF XYZ+rot6d plus absolute gripper |
| model dtype | bfloat16 |
| diffusion inference steps | 4 Euler steps |
| batch / accumulation | 1 / 1 |
| optimizer / LR | AdamW / `1e-4` |
| scheduler / warmup | cosine / `0.05` |
| weight decay / grad cap | `1e-5` / `1.0` |
| O1 maximum | 200 optimizer steps, fresh BASE→N runs |
| effective trainable flags | projector=true, diffusion=true, visual=false, LLM=false, VLLN=false |
| effective state dropout | `0.0` in the Stage23 launcher |
| timing | `0.05 s`, `TEST_ONLY_STAGE22_SIMULATION` |
| resource | one H200; no OOM in sealed Phase-A evidence |

The Stage23 source audit and Phase-A config bind the same H40 dense relative-
EEF design. The complete source/config/hash evidence is in
`reports/STAGE24_RECIPE_UPSTREAM_AUDIT.json`.

## Exposure comparison

O1 used `B=1`, `N=200`, and gradient accumulation 1: 200 nominal example
presentations and 200 stochastic flow noise/time draws. The pinned upstream
SO100 custom example uses `B=32`, `N=2000`, or 64,000 nominal presentations.
The nominal ratio is therefore:

```text
64,000 / 200 = 320x
```

These are presentation counts, not unique windows. The upstream fixture has
multiple episodes/windows, so the ratio is an exposure comparison and not a
claim that the official example supplies 320× more unique data.

## Dense H40 relative-EEF precedent

The pinned upstream source contains:

```text
tag:             OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
type:            EEF
format:          XYZ_ROT6D
representation:  RELATIVE
action horizon:  40
```

This establishes that dense H40 relative EEF is an upstream N1.7 precedent. It
does not establish that the Go2+ARX5 distribution matches that pretraining
distribution. The upstream SO100 custom example is a distinct H16 joint-
relative recipe and is not silently substituted for the current ABI.

## A6 decision

The one prepared next Hopper candidate is:

```text
H2_LARGER_STOCHASTIC_FLOW_EXPOSURE_SAME_SAMPLE_H40_OVERFIT
```

Rationale:

1. Exact ID10 weights are present but initializer-like across all five blocks,
   so custom-slot exposure remains a direct unresolved factor.
2. The H4/H8 prefix audit already shows the profile defect, so a shorter
   evaluated horizon is not currently the leading explanation.
3. Translation statistics are nondegenerate after the structural first point
   and show no catastrophic scale mismatch; fixed rotation/gripper cells are
   fixture design facts.
4. Dense H40 relative EEF has an upstream precedent, while O1 exposure is
   320× below the pinned SO100 nominal presentation budget.

The proposed future protocol keeps `TINY1_H2STATS`, H40, the current 10D/132D
ABI, stochastic flow, training seed 2222, B1, and the four-step sampler, and
extends the bounded same-sample run to a proposed `N=2000`. This is prepared
only, not authorized or executed in Stage24. Increasing `max_steps` also
changes the cosine schedule's time scale; that consequence must be reported
explicitly and prevents calling the future run a perfectly isolated exposure
intervention.

H1 fixed-noise flow, H3 true H16, and H4 inactive-channel intervention remain
unexecuted alternatives. No candidate was selected from validation `e` or
locked test `f`.
