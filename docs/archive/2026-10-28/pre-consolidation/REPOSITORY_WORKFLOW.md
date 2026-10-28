# Repository workflow — after the 2026-10-01 consolidation

## One navigation surface, explicit execution sources

`main` is the maintained accepted baseline and entry point. An active experiment
uses the exact execution SHA in its own manifest. The latest experimental runtime
currently lives on `pi0266/state-conditioned-feedback`; it is not silently
promoted into main by administrative cleanup.

Use **Issue #11** for scientific GO/STOP decisions and concise stage returns.
Use **Issue #47** for tested reusable-code integration that is still outstanding.
Do not create a new general-purpose issue or use an unrelated historical PR as
another mandatory ledger. Technical PRs should represent a bounded code change.

## Branch lifecycle

1. Keep one execution branch per currently authorized scientific goal, with one
   main owner. Optional storage/help work may be separate when its write scope
   is truly separate; it is not a required signoff stage.
2. Put plan amendments and engineering fixes on the appropriate current branch,
   preserving immutable release/run SHAs. A routine comment, pointer or repair
   does not automatically require another permanent branch.
3. On experiment closure, preserve source and exact evidence, and explicitly
   distinguish execution completion from scientific quality PASS/FAIL/PARTIAL.
4. Archive a retired branch only after verifying an exact recoverable source ref
   and checking current PR/consumer dependencies. Do not infer irrelevance from
   age, a FAIL label, or a name alone.
5. The October 1 migration uses a **same-name lightweight compatibility tag** at
   each retired branch HEAD. Tags are immutable by project policy; they are not
   GitHub Releases or proof of deployment qualification. No server-side tag
   protection rule is claimed to have been configured by this task.

The same-name tags keep historical ref-based file/tree locators useful, but
branch-only clients can require an explicit tag fetch. Existing local worktrees
are not deleted or rewritten. Never force-push an archival tag.

## Recover historical source

Consult [INVENTORY.md](maintenance/2026-10-01/INVENTORY.md) or
[BEFORE.json](maintenance/2026-10-01/BEFORE.json) for the old branch name and exact SHA.

```bash
# Run only in your own intended checkout, and preserve uncommitted work.
git fetch origin tag <historical-branch-name>
git switch -c <new-authorized-working-branch> refs/tags/<historical-branch-name>
```

If the original remote branch name itself is required by a resumed, authorized
workflow, recreate it at the recorded SHA rather than moving the archive tag.
A new branch is not authorization for a model run, dataset read or simulator step.

For a non-mutating local view of retired remote-tracking refs use
`git remote prune origin --dry-run`. Actual local pruning is optional and must
not delete uncommitted/local-only work; this cleanup did not run it on PC or HPC.

## PR lifecycle and integration

Historical provenance PRs were closed **unmerged** with a recovery pointer.
Their titles, bodies, comments and code remain. Closing means the old review
surface is not an active queue; it does not mean the experiment passed or that
its source was accepted into main.

For future reusable code, create a narrow PR with exact source lineage, a clear
diff and appropriate CPU tests. Preserve old frozen execution wrappers when
required by existing manifests. Do not octopus-merge alternative losses,
normalizations, experiments or runtime branches merely to reduce branch count.

During this housekeeping, only navigation, archival metadata and isolated
administration utilities are eligible for merge. Scientific runtime integration
remains explicit work under #47; tests and compatibility must be demonstrated
for each bounded extraction. A docs/administration merge is not a new baseline
qualification for GR00T or WBC.

## Evidence and assets

Maintain separate identities for protocol, execution code, checkpoint,
processor/statistics, dataset split, actual random tensors, raw predictions and
metric implementation. Preserve historical negative results and original
comment URLs. New summaries link to originals rather than rewriting them.

Keep large models, raw datasets and videos out of Git. Use authorized fixed
archive revisions and manifests; never publish private WBC, robot/vendor assets,
credentials or machine secrets to a public hub. Storage cleanup is a distinct
user-authorized task, not an implied side effect of closing a PR.

## Concurrency and safety

The migration snapshots every branch HEAD, keeps explicit active dependencies,
creates and reads back tags before any closure, refreshes live PR dependencies,
and deletes refs using an atomic push with an expected-SHA lease for each ref.
A concurrently moved, protected, unknown or newly dependent branch is kept.
No remote job is killed, checkpoint moved or scientific branch force-updated.

The one-shot repository-writing workflow is removed after execution. Its script,
tests and immutable request/receipts remain for audit; they are not an unattended
scheduler. Further administrative writes require a new explicit request and
fresh inventory rather than replaying the old apply request.

## Actual audit

- [Executed result](maintenance/2026-10-01/RESULT.md)
- [Before state and original issue/PR descriptions](maintenance/2026-10-01/BEFORE.json)
- [Verified archives created before retirement](maintenance/2026-10-01/ARCHIVE_PREPARED.json)
- [After state and explicit skips](maintenance/2026-10-01/AFTER.json)

The audit records administrative completeness only. Current scientific
`selected_checkpoint` and `deployment_selected_checkpoint` remain NONE unless
an explicit subsequent scientific decision changes them.
