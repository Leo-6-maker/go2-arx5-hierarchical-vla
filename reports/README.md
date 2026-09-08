# Reports index

Reports are the human-readable layer of the evidence chain. At the RC0
baseline there were 145 tracked report files, including stage reports,
protocols, resource gates, and Corelab handoffs. This index and the
consolidation audit are additive; historical reports remain at their original
paths.

## How to read the evidence

1. Start with the stage row in [the research timeline](../docs/RESEARCH_TIMELINE.md).
2. Read the cited report and its machine-readable manifest/seal, when present.
3. Use the Issue #11 comment cited by the row for the governing decision and
   claim boundary.
4. Treat node-local paths in reports as external evidence references, not as
   files available from a fresh clone.

## Main navigation points

- Stage19–Stage25: `STAGE19_FINAL_REPORT.md` through
  `STAGE25_FINAL_REPORT.md`, plus their protocol and provenance companions.
- Stage26–Stage32: preparation/audit reports and the corresponding stage
  runners; some later final returns are retained in PR-linked evidence rather
  than duplicated under a new filename.
- Stage33–Stage39: `STAGE33_DUAL_TRACK_BOTTLENECK_MAP.md`,
  `STAGE34_WBC_ALIGNED_MOTION_API.md`,
  `STAGE35_MOTION_API_AUTHORITY_CLOSURE.md`, and
  `STAGE36_WBC_AUTHORITY_ORACLE_QUALIFICATION.md`; Stage37–39 final returns
  remain linked through their PR comments and scripts.
- Stage40–Stage41: `CORELAB_STAGE40_CT1_PROVENANCE.md`,
  `CORELAB_STAGE40_CT2_STATIC_TAKEOVER.md`,
  `CORELAB_STAGE40_CT3_RUNTIME_PREP_20260908.md`, and
  `CORELAB_STAGE41_M2_BASELINE_PROFILE.md`.
- Stage42–Stage44R: the canonical runners and sealed node-local result roots
  are linked from Issue #11 comments `5580348321`, `5581418722`,
  `5582857275`, and `5583158968`. The repository does not fabricate duplicate
  reports for those external result roots.
- `REPOSITORY_CONSOLIDATION_AUDIT.md` and its `.json` companion: this PR's
  RC0 inventory and resource-boundary check.

## Preservation rule

Do not rename, move, or remove a stage report merely to make the tree look
tidier. Historical reports are often referenced by exact path from a sealed
manifest, an Issue #11 return, or a stage runner. Add an index or a clearly
labelled correction layer instead.
