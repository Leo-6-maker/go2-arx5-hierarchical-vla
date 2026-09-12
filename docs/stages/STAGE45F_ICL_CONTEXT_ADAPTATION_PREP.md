# Stage45F ICL ContextV1 preparation and PI0145 hardening

This page records the CPU/static preparation authorized by PI_GATE_STATE_0145.
It is a protocol and provenance surface only; it is not a model or training
result.  The preparation branch is stacked on the accepted Corelab C10
compatibility branch and does not modify the canonical Vanda branch.

## Frozen upstream and output ABI

The source contract is bound to GR00T N1.7 commit
`51d4c89f72fda44cbf77285c6a8114b52676b8a1`.  ContextV1 is pathless: records
refer to immutable artifact IDs and SHA256 values, never filesystem paths.
The downstream output remains StructuredActionV1 with shape `[40, 132]` and
190 active scalars; no decoder or action semantics are changed here.

## ContextV1 causal contract

Each record carries a query time and four explicit, independently auditable
sections:

1. task demonstration context;
2. interaction history;
3. failure/outcome context;
4. local dynamics context.

Task demonstrations satisfy `demo_time_range[1] <= query_time`.  Every
transition entry also carries an information-availability time and must satisfy
`event_time <= available_time <= query_time`.  The latter is the time when the
latest included response/outcome/state-after became observable, not merely the
command start time.  Local dynamics additionally requires
`event_time + dt <= available_time`, so a claimed early `state_after` cannot
close a later transition.  Commanded actions and executed/observed responses are
separate fields.  A response that was not observed is represented by explicit
`null`; it is never inferred.  A local dynamics entry must include immutable
state-before, state-after, and delta digests plus a positive `dt`, so the
builder cannot fabricate a physical transition.

The nested interaction timing payload is an exact model-visible ABI: only
`{"dt": <positive seconds>}` is allowed, with float32/seconds semantics and
no missing value.  Failure outcomes use the declared finite vocabulary
(`no_progress_observed`, `progress_observed`, `recovered`, `unrecovered`, or
`unknown`).  Metadata/provenance/target/query fields are recursively rejected
inside model-visible payloads, including timing.

The model-visible whitelist is limited to digest-resolved payloads (and the
task-demo language).  Window/target IDs, source IDs/SHAs, demo/attempt IDs,
timestamps, provenance and builder identities are metadata-only.  In
particular, `target_id` can never become a model shortcut.  The independent
`ICL_CONTEXTTENSORV1_PLAN` records the plan-only tensor ABI: float32 event
slots `[B,N,F]`, int32 language tokens `[B,N_text]`, boolean event/section and
missingness masks, causal section order, source-declared frame/units,
explicit normalization identity, and the immutable digest resolver.  Its
resolver is tied to immutable `SEALED_TRAIN_DESCRIPTOR` / `TRAIN24`
membership and carries `available_time` from that source; callers cannot
override membership or availability.

The CPU builder rejects target XYZ, future observations or expert labels/
actions, Validation-E/Holdout-F material, path-like fields, unknown fields,
source identity drift, non-finite values, and missing-vs-null response
distinctions.  Canonical section ordering is one causal law for every section:
event time, then availability time, then model-visible payload identity;
metadata IDs cannot reorder distinguishable history.

## Integration families and experiment ladder

The integration families remain neutral and unselected: multi-frame
multimodal prompt, extended state history, **C1 action-side context tokens**
(`ContextEncoder -> [B,K,1536]`, zero gate `alpha_0=0`, concatenated between
state and action features with an action-side isolated residual or
mask-preserving bypass and the final 40-action slice unchanged), **C2
VL-memory context tokens** (`[B,K,2048]`, explicit `attention_mask=true` and
`image_mask=false`), and deeper explicit memory/cross-attention.  C1 and C2
are separate architectures; C2 is not an alias of the deeper family.

The C1 zero gate is a parity obligation, not a claim that concatenating zero
context values preserves attention.  At zero gate the original state/action
positions and attention denominator must remain unchanged, while the isolated
context residual still receives a learnable gradient.  The CPU scalar
counterexample in `c1_zero_gate_attention_counterexample()` records a
one-key output changing from `1` to `1/2` when an unmasked zero context key is
appended; no model forward is used.

The ladder distinguishes `I0_EXISTING_INTERFACE_CONTEXT_PROBE` (only existing
checkpoint-compatible multimodal/image/language inputs) from
`I0_ADAPTER_ZERO_GATE_PARITY` (a static/future C1 parity contract whose actual
forward remains locked).  Learning starts at `I1_CONTEXT_ADAPTER_ONLY`, then
I2 joint post-training and I3 history-conditioned DAgger.  All are plan-only;
the nine context variants and both future Validation-E placement options are
frozen, with no automatic winner or route selection.

## Reproducible CPU surface

`scripts/stage45f_icl_context_v1.py` provides the explicit builder, validator,
canonical index, static manifest audit, and synthetic mutation regression.
`tests/test_stage45f_icl_context_v1.py` covers causal timestamps, target and
locked-split leakage, path leakage, explicit null responses, source/schema
drift, nested timing/scalar leakage, equal-event availability ordering,
transition closure, the C1 scalar parity counterexample, deterministic order
invariance, and the zero-execution boundary.

The associated JSON manifests are the machine-readable authority.  They keep
model construction/forward, training, GPU/PBS, Validation-E/Holdout-F,
simulator/WBC, and hardware at zero.  This stage stops at
`STAGE45F_CORELAB_PI0145_CONTEXT_AND_C10_HARDENING_READY_FOR_PI_REVIEW`.
