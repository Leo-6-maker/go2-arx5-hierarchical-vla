# ADR-029: BASE to O1@200 Parameter-Delta Gate

Date: 2026-09-03

Status: Accepted for preparation; numerical execution requires a separate
CPU/static gate authorization.

## Context

Stage23 demonstrated a strong O1@200 free-running profile failure, but its
Phase-A relay intentionally excluded the full checkpoint weights. Stage24
therefore audited the exact BASE checkpoint and classified the
`NEW_EMBODIMENT` slot as initializer-like, while correctly leaving the
BASE→O1 parameter delta unknown.

The missing question is cheaper and more informative than immediately
starting another H200 training run: did the executed O1@200 run change the
custom projector and shared diffusion modules relative to BASE, and were any
modules expected to be frozen changed unexpectedly?

## Decision

1. Compare exact BASE and O1@200 safetensor trees directly, without model
   construction, CUDA, forward, inference, or training.
2. Require the frozen BASE index/tree identity, the O1 index identity, and the
   immutable O1 per-file seal before reading model tensors.
3. Compare all 14 category-specific projector keys and rows 4–7, 10, 11, and
   24–27. Do not assume non-ID10 rows are byte-identical without auditing the
   pinned optimizer/weight-decay semantics.
4. For the 132D-facing action encoder W1, report active 10D versus inactive
   122D columns with per-column-normalized metrics; total energy alone is not
   sufficient.
5. Reconcile serialized keys exactly, except for the single pinned
   `backbone.model.lm_head.weight` BASE-only tied alias when
   `backbone.model.model.language_model.embed_tokens.weight` is present in
   both indexes and the BASE shard mapping is exact. Every other missing or
   extra key fails closed; common-key shape and dtype remain exact gates.
6. Aggregate every compared serialized key into the frozen recipe groups. Any
   delta in an expected-frozen group stops interpretation pending source
   explanation.
7. Keep the O1 bytes outside Git. Git tracks the harness, protocol, hashes,
   scalar evidence, and provenance only.

## Consequences

The preparation result can be completed even when the Phase-A relay contains
only O1 metadata/seals. The numerical result remains explicitly unavailable
until the exact O1 shards are relayed. This prevents a newer checkpoint,
directory naming, or partial artifact from being mistaken for O1@200.

The one key-count difference observed on the exact Hopper checkpoint is a
known tied-weight serialization alias under the pinned Transformers save
path. The explicit reconciliation does not change any tensor input or
scientific comparison; it only prevents a serialized alias from blocking the
same CPU/static gate.

## Evidence

```text
reports/STAGE25_O1_PARAMETER_DELTA_PROTOCOL.md
reports/STAGE25_O1_PARAMETER_DELTA_PROTOCOL.json
reports/STAGE24_STATIC_CHECKPOINT_AUDIT.md
reports/STAGE23_PHASE_B_ATTRIBUTION_AUDIT.md
```
