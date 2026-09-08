# Historical PR and branch archival recommendation

This page retains the RC0 recommendation and records the authorized RC8
execution. Under `PI_GATE_STATE_0072`, PR #28 was normally merged into `main`
at `bfa14bf1b74eae908c0922575a4db671c903b9b8`; PRs #4–#10 and #12–#26 were
closed unmerged as superseded, with comments and evidence preserved. PR #3 was
already merged before RC8 and remains merged; it was not rewritten or
force-closed. No historical branch was deleted. The accepted successor anchor
is `bd0f02934e680789cf38adacb838e1ea555c707d`.

The table below is the original RC0 snapshot and proposed-action record. The
RC8 execution status above is authoritative for the current GitHub state.

## Historical PRs

GitHub RC0 found PRs #1–#26. PRs #1 and #2 are already closed; PRs #3–#10
and #12–#26 remain open historical stage records. PR #11 is the governance
issue, not a PR. The recommended post-merge action for the open PRs is
`close-unmerged as superseded`, while preserving their comments, files, and
links to sealed evidence.

| PR | Stage record | Current RC0 state | Proposed action after consolidation review |
| --- | --- | --- | --- |
| #1 | Stage15–17 native/WBC and GR00T interface | closed | keep comments and merge history |
| #2 | Stage18 data-readiness contracts | closed | keep comments and merge history |
| #3 | Stage19 dataset materialization | open | close-unmerged as superseded; preserve evidence |
| #4 | Stage20S simulation-only adaptation | open | close-unmerged as superseded; preserve evidence |
| #5 | Stage20T timing qualification | open | close-unmerged as superseded; preserve evidence |
| #6 | Stage21 same-scene receding horizon | open | close-unmerged as superseded; preserve evidence |
| #7 | Stage22 action-profile adaptation | open | close-unmerged as superseded; preserve evidence |
| #8 | Stage23 action-pipeline learnability | open | close-unmerged as superseded; preserve evidence |
| #9 | Stage24 NEW_EMBODIMENT audit | open | close-unmerged as superseded; preserve evidence |
| #10 | Stage25 BASE→O1 parameter delta | open | close-unmerged as superseded; preserve evidence |
| #12 | Stage26 fixed-noise local flow | open | close-unmerged as superseded; preserve evidence |
| #13 | Stage27 H2 optimization semantics | open | close-unmerged as superseded; preserve evidence |
| #14 | Stage28 fresh BASE→2000 preparation | open | close-unmerged as superseded; preserve evidence |
| #15 | Stage29 sampler transfer | open | close-unmerged as superseded; preserve evidence |
| #16 | Stage30 temporal residual decomposition | open | close-unmerged as superseded; preserve evidence |
| #17 | Stage31 on-trajectory flow audit | open | close-unmerged as superseded; preserve evidence |
| #18 | Stage32 exposure interpolation | open | close-unmerged as superseded; preserve evidence |
| #19 | Stage33 bottleneck map | open | close-unmerged as superseded; preserve evidence |
| #20 | Stage34 WBC-aligned Motion API | open | close-unmerged as superseded; preserve evidence |
| #21 | Stage35 Motion API authority closure | open | close-unmerged as superseded; preserve evidence |
| #22 | Stage36 delivered-WBC authority/oracle | open | close-unmerged as superseded; preserve evidence |
| #23 | Stage37 geometry/structured oracle | open | close-unmerged as superseded; preserve evidence |
| #24 | Stage38 StructuredAction/offline | open | close-unmerged as superseded; preserve evidence |
| #25 | Stage39 sealed native replay | open | close-unmerged as superseded; preserve evidence |
| #26 | Stage40 live GR00T→WBC | open | close-unmerged as superseded; preserve evidence |

## Branches

The 38 stage branches found at RC0 fall into three preservation classes:

| Branch set | Relationship to accepted Stage44R | Recommendation |
| --- | --- | --- |
| `stage15/...` through `stage38/...`, `stage39/structured-wbc-qualification`, `stage40-b1r/...`, `stage40-c/...`, `stage40-dev/rtx4080s-bringup`, `stage40-rlq/...`, `stage40/...`, `stage41-dev/...`, `stage42-dev/...`, `stage43-dev/...`, `stage44-dev/...` | Ancestors of `bd0f029...` | Retain through PR review; after a normal merge, tag or retain the accepted anchors and delete only branches confirmed obsolete. |
| `stage39-vis/offline-mp4` | Diverged presentation branch | Retain branch/tag until its visualization evidence is explicitly archived; do not infer it is redundant from the consolidation merge. |
| `stage40-dev/a40-bringup` | Diverged Vanda/A40 engineering branch | Retain as an independent historical/fallback record; no A40 execution is authorized by this task. |
| `stage44r/corelab-attribution-repair` | Accepted source tip | Retain as the Stage44R provenance anchor, or replace only with a PI-approved tag after the consolidation merge. |

The complete branch names and SHA inventory are machine-recorded in
`reports/REPOSITORY_CONSOLIDATION_AUDIT.json`. No branch deletion, history
rewrite, or evidence removal was performed by RC8. Future branch cleanup
requires a separate explicit PI authorization after the corresponding evidence
has been independently archived.
