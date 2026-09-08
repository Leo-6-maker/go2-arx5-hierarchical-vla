# Repository consolidation audit (RC0)

Audit date: 2026-09-08 (Asia/Singapore)

Status: `PASS`

This is a read-only inventory performed before the documentation changes in
the `repo-consolidation/stage44r-baseline` branch. The machine-readable record
is [`REPOSITORY_CONSOLIDATION_AUDIT.json`](REPOSITORY_CONSOLIDATION_AUDIT.json).

## Baseline and lineage

```text
accepted branch: stage44r/corelab-attribution-repair
accepted HEAD:   bd0f02934e680789cf38adacb838e1ea555c707d
parent:          f6c4759cdbd3be513d20d6b3934ebdbbdeedd5f4
main:            237be6fc37c20cf44ccccf5f0cfbd3cce629760f
ahead / behind:  183 / 0
merge-base:      237be6fc37c20cf44ccccf5f0cfbd3cce629760f
```

The local checkout matched the remote Stage44R tip and was clean. The
accepted lineage is therefore a forward-only descendant of `main`; there is
no need to merge the historical stack one PR at a time.

## GitHub inventory

GitHub reports 38 stage branches. PRs #1 and #2 are closed. The open historical
PR set is #3–#10 and #12–#26 (23 PRs); Issue #11 is the governance surface and
Issue #27 is this consolidation task. The complete names are in the JSON
companion and the proposed post-merge handling is in
[`docs/HISTORICAL_ARCHIVAL_RECOMMENDATION.md`](../docs/HISTORICAL_ARCHIVAL_RECOMMENDATION.md).

Two stage branches diverge from the accepted Stage44R tip:

- `stage39-vis/offline-mp4` — presentation evidence branch;
- `stage40-dev/a40-bringup` — Vanda/A40 engineering branch.

They are retained for explicit review. All other inventoried stage branches
are ancestors of the accepted tip and are not deleted by this PR.

## Tracked-tree inventory

At the RC0 tip the repository contained 501 tracked files:

```text
docs 66       hpc 19          manifests 10   reports 145
scripts 121   src 68          tests 55       configs 5
artifact_manifests 7          .github 1      experiments 1
```

There were 108 stage-prefixed scripts and 22 tracked JSON files. Historical
stage runners and reports are referenced by exact path from 53 and 15 tracked
files respectively. Those paths are provenance-bearing and were not moved,
renamed, or deleted.

## Secret, private-path, and large-file audit

- Conservative credential patterns (private-key headers, cloud/API key
  prefixes, password/secret assignments, and common HF/W&B token assignments)
  matched zero tracked credential-like values.
- Node-local path strings occur in reports and launch wrappers by design; they
  are references to external evidence, not committed data. Governance marker
  tokens are identifiers, not credentials.
- No tracked symlinks or zero-byte files were found.
- No checkpoint, dataset, video, archive, or other binary-like extension is
  tracked.
- No current tracked file exceeds 1 MiB. The largest current file is the
  historical `reports/STAGE24_PREFIX_AUDIT.json` at 530,891 bytes. Six
  historical text blobs are at least 100 KiB; none exceeds 1 MiB and none is a
  private model/data binary.

## Resource boundary

The consolidation audit and documentation work consumed no scientific
resources:

```text
GR00T forwards:            0
WBC steps:                 0
reference updates:         0
training/backward/opt:     0
new data generation:       0
GPU/simulator execution:   false
PBS submissions:           0
Vanda A40:                 HOLD
Hopper H200:               HOLD
Validation-E/Holdout-F:    LOCKED
hardware/ROS/ZMQ/P1M:      LOCKED
```

RC0 therefore passes. The only changes in this consolidation branch are
navigation, audit, archival-recommendation, and CPU/static-regression files;
scientific/runtime semantics remain untouched.
