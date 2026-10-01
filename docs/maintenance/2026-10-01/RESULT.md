# Repository consolidation — executed result

Status: **ADMINISTRATIVE_CONSOLIDATION_COMPLETE**

- Branch names retired after identical tag preservation: **163**.
- Historical PRs closed unmerged: **29**.
- Superseded issues closed: **[48, 50]**; #11 and #47 remain.
- After counts: `{"branches": 8, "open_prs": 0, "open_issues": 2}` (maintenance PR may be added later).
- Every pre-change source head retained: **True**.
- No experiment, runtime source, model, HF evidence, private input or scientific result was changed.
- Main code integration is not claimed; #47 records remaining tested extraction work.

## Retained branch names

- `main`
- `pi0257/corelab-video-evidence-workflow-v1`
- `pi0263/storage-subagent-model-audit-go`
- `pi0263/train-template-counterbalancing`
- `pi0264/language-diversity-transfer`
- `pi0265/new-scene-feedback`
- `pi0266/state-conditioned-feedback`
- `repo-maintenance/consolidation-20261001`

## Closed historical PRs

#46, #49, #51, #52, #53, #54, #55, #56, #57, #58, #59, #60, #61, #62, #63, #64, #65, #66, #67, #68, #69, #71, #72, #73, #74, #75, #76, #77, #78

## Explicit skips

```json
[]

```

## Restore historical source

The original branch name is now a compatibility tag at the exact old SHA. It is not a release or a result promotion.
```bash
git fetch origin tag <historical-name>
git switch -c <new-working-name> refs/tags/<historical-name>
```

Original branch heads, PR descriptions and issue bodies: [BEFORE.json](BEFORE.json).
Verified archive before retirement: [ARCHIVE_PREPARED.json](ARCHIVE_PREPARED.json).
Machine-readable after-state: [AFTER.json](AFTER.json).
