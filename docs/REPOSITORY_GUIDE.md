# Repository guide

The repository is intentionally split between reusable contracts, historical
experiment runners, and evidence indexes. The split is organizational only:
stage-specific files remain at their original paths so sealed reports and
Issue #11 references continue to resolve.

## Directory map

| Directory | Role | Maintenance rule |
| --- | --- | --- |
| `src/` | Reusable library code: contracts, adapters, metrics, data readiness, runtime lifecycle, and native-WBC boundaries. | Changes require semantic review; do not alter a frozen ABI or controller behavior as part of cleanup. |
| `scripts/` | Active and legacy stage runners, auditors, finalizers, and offline analyzers. | Stage-specific names are provenance anchors, not a stable public API. Preserve paths and invocation semantics. |
| `tests/` | CPU contract, lifecycle, parser, provenance, and deterministic fixture tests. | Prefer standard-library fixtures; avoid importing model/WBC runtimes in consolidation tests. |
| `reports/` | Human-readable stage reports and the consolidation audit. | Reports are historical evidence; append corrections rather than rewriting accepted results. |
| `manifests/` | Small machine-readable identity, input, and execution manifests. | Keep hashes and schemas exact; large payloads stay outside Git. |
| `artifact_manifests/` | Logical references to external checkpoints, WBC archives, datasets, and runtime assets. | Record identity/path metadata without committing private bytes. |
| `docs/` | Architecture, contracts, ADRs, handoffs, timeline, baseline, and governance documentation. | Describe accepted behavior only; do not promote unproven task claims. |
| `hpc/` | Historical PBS and remote-runtime wrappers. | Treat as archived launch surfaces; no submission is part of repository consolidation. |
| `configs/` | Public configuration examples and schemas. | Local path/config files are ignored and must not contain credentials. |
| `experiments/` | Small public experiment metadata and entry-point notes. | Node-local runs/outputs are ignored and external. |

## Evidence topology

Git stores source, compact manifests, reports, and tests. Model checkpoints,
WBC source/archives, datasets, videos, simulator assets, logs, and raw rollout
arrays belong on the authorized node-local filesystem. Reports may mention an
external path or SHA so another operator can locate the evidence; that mention
does not make the path portable or suitable for committing.

The accepted Stage44R source identity is:

```text
stage44r/corelab-attribution-repair
bd0f02934e680789cf38adacb838e1ea555c707d
```

The consolidation branch starts at that exact tip. It must not rebase,
squash, cherry-pick a Vanda branch, or change the Stage40–Stage44R evidence
lineage.

## Branch and history policy

- `main` is the public integration baseline; the consolidation PR targets it.
- `repo-consolidation/stage44r-baseline` is a one-PR documentation/audit
  branch rooted at the accepted Stage44R tip.
- Historical stage branches and PRs are evidence records. Do not close,
  delete, or merge them individually during this cleanup.
- If PI approves the PR, merge it normally to preserve the 183 scientific
  commits. Do not squash the lineage.
- The archival recommendation is recorded separately in
  `docs/HISTORICAL_ARCHIVAL_RECOMMENDATION.md` and is not an action list for
  this branch.

## Validation without scientific exposure

Use the following read-only/static checks before opening a consolidation PR:

```bash
git diff --check
python -m compileall -q src scripts tests
python -m unittest discover -s tests -p 'test*.py'
python -m pytest -q                 # when pytest is available
```

The consolidation suite must also inspect tracked file extensions, symlinks,
zero-byte files, secret-like patterns, and large blobs. No validation command
in this workflow may construct GR00T, invoke a simulator/WBC rollout, train,
generate data, submit PBS, or access Validation-E/Holdout-F.

## Contract guardrails

The following are frozen across this cleanup: `StructuredActionV1` and D1
decode, StateV1, model input keys, SafetyGate thresholds, WBC/controller
semantics, reference-only handoff/timing, training recipe, checkpoint
selection, frame/TCP conventions, and evidence hashes. Any proposed change to
one of those belongs in a new PI-authorized scientific stage, not this PR.
