# Corelab Stage40 CT2 Static Takeover

Classification: `NON_SCIENTIFIC_DEVELOPMENT_EVIDENCE`

Under `PI_GATE_STATE_0064`, CT2 selectively reimplements the portable
evidence/admission concepts identified in the Vanda audit. No Vanda commit was
merged or cherry-picked, and no A40 scheduler, account, container, filesystem,
or cache assumption was imported. Existing Corelab RTX paths, GPU-UUID
selection, native runtime repair, clean-room compatibility layer, and cache
isolation remain the execution mechanics for the later cutover.

The new standard-library-only layer requires separate `RTX-runtime`, `RTX-Q0`,
`RTX-bind`, `M1`, and `M2` bundles and canonical seals. Every non-root stage
must verify its immediate predecessor's sidecar hash, seal schema, inner result
schema, PASS state, stage identity, development seed, repository/executable
lineage, execution class, counters, and forbidden-operation flags.

Raw evidence is represented by role-specific regular files under a bounded
evidence root. Their sizes and SHA256 values form an evidence manifest; the
manifest hash is embedded in the canonical result; the result and execution
bundle hashes are embedded in the canonical seal. Missing roles, symlinks,
path escape, evidence mutation, sidecar mutation, or identity drift fails
closed.

Counters cannot be established by assigning result values. A sealed stage must
include hash-bound before/after monotonic observations, one unit delta per
GR00T request, one unit delta per reference-only update, and positive WBC
segment deltas for both M1 segments or all four M2 segments. M1 therefore
requires two observed GR00T forwards, WBC work before and after the handoff,
and exactly one reference update. M2 requires four observed forwards, four
positive WBC segments, and exactly three updates.

The existing mailbox worker now reports its monotonic counter before and after
each actual completed model call. The native wrapper already owned a monotonic
WBC-step counter and now also exposes a Stage40 bookkeeping-only reference
update counter. These additions do not alter model inputs/outputs, WBC policy,
controller behavior, physics, or shared executor semantics.

Static negative tests cover missing predecessor seals, bad seal SHA, wrong
stage, wrong seed, wrong HEAD, wrong execution class, evidence tampering,
counter forgery, and M2 admission from a non-M1 seal. The existing Stage40
prefix/lifecycle regression continues to enforce initial bind before the first
WBC step without changing `RecedingHorizonExecutorV1`.

CT2 performs no GPU work, model construction, GR00T forward, WBC step,
reference update, training, formal-seed access, Validation-E/Holdout-F access,
hardware action, or H200 action. Actual bundles are frozen only from a clean
pushed HEAD immediately before each authorized execution class.
