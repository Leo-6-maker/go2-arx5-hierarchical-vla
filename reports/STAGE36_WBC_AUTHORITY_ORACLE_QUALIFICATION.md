# Stage36 — delivered-WBC authority re-baseline and oracle qualification

Stage36 begins with a CPU-only, read-only authority audit. It creates the new
identity `DELIVERED_WBC_AUTHORITY_V2` only when the delivered source, policy
checkpoints, native configuration, robot assets, reach table, runtime
modules, and simulator identity can all be bound to unique hashable paths.

The audit never relabels a new artifact as historical Stage16. Historical
hashes are reported as `MATCH_OLD` or `DIFFERENT_NEW_BASELINE`; missing or
ambiguous inputs stop the stage with
`GPT_PI_REVIEW_WBC_ARTIFACT_REDELIVERY_REQUIRED`.

The sealed Stage35 candidate corpus is consumed in place. Its exact tree hash
is `b98960f9dc4bfd04b5a0de0f693ddea4f50c8849ef80da83428280c1e2e42414` and its
manifest hash is
`c1e4021a77817af851ec8c1ba5d0c4b5f116558c16f03ff0e24457e1fd6e0642`.
Cases are not regenerated or edited. Native-support admission is distinct
from the Stage35 analytic witness and cannot pass without the native sealed
reach table.

This branch contains no GR00T import, model construction, simulator/WBC
rollout, PBS submission, checkpoint write, validation-E read, holdout-F read,
hardware action, or PR merge. Stage36-C/D/E remain locked until the authority
and native corpus predicates pass.
