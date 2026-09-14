# Post-Validation harness architecture proposal

This is a design-only proposal produced under PI0168. It does not implement or invoke a post-Validation harness, and it does not change PR #71, the frozen evaluator, the admission policy, or the runtime source.

## Design goals

The harness should make the process boundary and evidence lineage explicit without baking a particular PI number into reusable mechanics. A run-specific gate, pointer, branch, and owner belong under `authority_metadata`; generic code should consume a schema version and immutable identities.

The proposed flow is:

```text
RunSpecV1
  -> stdlib preflight / Git identity / fresh-root check
  -> optional isolated verifier child (payload-facing, if authorized)
  -> frozen admission in a clean parent
  -> model/runtime process (only after admission)
  -> raw artifact seal
  -> CPU-only post-hoc decode/metric/aggregate
  -> PosthocSealV1
```

The PI0168 preparation itself stops before the optional verifier and before any model/runtime process.

## Proposed machine-readable contracts

### RunSpecV1

```json
{
  "schema": "RUN_SPEC_V1",
  "run_id": "opaque-stable-run-id",
  "authority_metadata": {
    "issue": {"number": 11, "comment_id": 0},
    "pr": {"number": 0, "comment_id": 0},
    "owner": "vanda",
    "gate_token": "opaque-gate-token"
  },
  "source_identity": {
    "repository": "owner/name",
    "branch": "exact-branch",
    "commit": "64-character-sha",
    "source_manifest_sha256": "64-character-sha"
  },
  "protocol_identity": {
    "protocol_schema": "named-protocol-schema",
    "protocol_sha256": "64-character-sha",
    "checkpoint_arms": 0,
    "windows_per_arm": 0,
    "noise_identity_sha256": "64-character-sha"
  },
  "process_policy": {
    "admission_parent_must_be_runtime_clean": true,
    "payload_verifier_process": "none|isolated-child",
    "model_process": "separate-after-admission",
    "posthoc_process": "cpu-only",
    "automatic_retry": false
  },
  "resource_policy": {
    "model_load_ceiling": 0,
    "model_side_call_ceiling": 0,
    "training_allowed": false,
    "holdout_f_allowed": false
  },
  "path_policy": {
    "input_namespaces": ["opaque-allowlisted-path"],
    "output_root_must_be_fresh": true,
    "raw_artifact_must_be_preserved_before_posthoc": true
  }
}
```

`authority_metadata` is the only location for gate/comment identifiers. The generic mechanics do not contain a PI-number constant.

### EvidenceReusePlanV1

```json
{
  "schema": "EVIDENCE_REUSE_PLAN_V1",
  "run_id": "opaque-stable-run-id",
  "source_evidence": {
    "raw_manifest_path": "allowlisted-relative-or-bound-path",
    "raw_manifest_sha256": "64-character-sha",
    "inventory_path": "allowlisted-relative-or-bound-path",
    "inventory_sha256": "64-character-sha",
    "source_run_id": "opaque-source-run-id"
  },
  "cells": [
    {
      "arm_id": "opaque-arm-id",
      "window_id": "opaque-window-id",
      "relative_raw_path": "raw/cell.npy",
      "file_sha256": "64-character-sha",
      "array_sha256": "64-character-sha",
      "shape": [0, 0],
      "dtype": "float32",
      "source_model_calls": 0,
      "reuse_mode": "byte-preserved"
    }
  ],
  "read_policy": {
    "payload_read_authorized": false,
    "source_is_immutable": true,
    "missing_or_hash_mismatch": "fail-closed"
  }
}
```

The plan records identity and provenance, not a new scientific interpretation. A child verifier may produce this plan only when the active gate authorizes payload-facing work.

### ModelCallLedgerV1

```json
{
  "schema": "MODEL_CALL_LEDGER_V1",
  "run_id": "opaque-stable-run-id",
  "source_identity": {"commit": "64-character-sha", "protocol_sha256": "64-character-sha"},
  "events": [
    {
      "event_id": "opaque-event-id",
      "process": "admission-parent|model-child|posthoc-child",
      "phase": "named-phase",
      "arm_id": "opaque-arm-id",
      "window_id": "opaque-window-id",
      "operation": "model-load|conditioning|action-head|decode|aggregate",
      "attempted": false,
      "completed": false,
      "model_identity": null,
      "raw_artifact_path": null,
      "raw_artifact_sha256": null,
      "error_class": null
    }
  ],
  "totals": {
    "model_load_attempts": 0,
    "model_loads_completed": 0,
    "conditioning_attempts": 0,
    "action_head_calls": 0,
    "raw_artifacts": 0,
    "training_steps": 0,
    "backward_steps": 0,
    "optimizer_steps": 0
  }
}
```

The ledger separates attempts from completions and binds every model-side event to a process and immutable run identity. Reused raw cells have zero new model events and are linked through `EvidenceReusePlanV1`.

### PosthocSealV1

```json
{
  "schema": "POSTHOC_SEAL_V1",
  "status": "PASS|PARTIAL|BLOCKED",
  "run_id": "opaque-stable-run-id",
  "authority_metadata": {"gate_token": "opaque-gate-token", "issue_comment_id": 0},
  "source_identity": {"commit": "64-character-sha", "protocol_sha256": "64-character-sha"},
  "evidence": {
    "raw_manifest_sha256": "64-character-sha",
    "derived_manifest_sha256": "64-character-sha",
    "cpu_summary_sha256": "64-character-sha",
    "ledger_sha256": "64-character-sha",
    "inventory_sha256": "64-character-sha"
  },
  "cardinality": {"arms": 0, "windows_per_arm": 0, "raw_rows": 0, "derived_rows": 0},
  "boundary_flags": {
    "raw_before_derived": true,
    "training": false,
    "holdout_f_read": false,
    "automatic_retry": false,
    "scientific_contract_changed": false
  },
  "interpretation": {
    "descriptive_only": true,
    "winner_selected": false,
    "iid_claim": false
  },
  "failure": null
}
```

## Required invariants

1. The clean parent checks runtime-module absence immediately before frozen admission; a later helper cannot retroactively make an admission pass.
2. Any payload-facing NumPy verification happens in an explicitly identified isolated child, whose only parent output is validated JSON and hashes.
3. Raw artifacts are written and sealed before decode, derived metrics, aggregation, or interpretation.
4. Fresh output roots, exact source identity, immutable protocol identity, and no automatic retry are mandatory.
5. `RunSpecV1`, `EvidenceReusePlanV1`, `ModelCallLedgerV1`, and `PosthocSealV1` carry gate IDs as metadata; reusable code has no PI-number constants.
6. A partial or blocked seal remains a truthful engineering outcome and is never relabeled as scientific success or failure.

## Consolidation boundary

Portable candidates are the schema validators, hash/counter accounting, raw-first ordering, and process-boundary assertions. Device-specific runtime loaders, Train24 assumptions, Validation-E six-window assumptions, H200/PBS mechanics, and model/processor imports remain separate adapters until a PI-approved contract proves they are equivalent.
