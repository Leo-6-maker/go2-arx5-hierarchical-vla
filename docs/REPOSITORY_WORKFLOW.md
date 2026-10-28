# Repository workflow · main promotion, 2026-10-28

## One integration branch; pinned execution sources

The user explicitly requested consolidation and merge to main. PRs #81–#85 preserve the full experimental ancestry, main's independent changes, and unique legacy protocol/evidence commits. **`main` is now the integration and publication entry**, not just an older navigation snapshot. This supersedes older branch-routing restrictions for repository management only.

An experiment still consumes the full source SHA/seal in its own manifest. Main is not automatically deployed, and its merged historical modules are not automatically qualified. Keep the existing `~/Projects/GR00T-WBC` entry, owner, profiles, immutable capsules, private registry and current permissions. Do not pull/checkout an in-use source or force-reset a divergent checkout.

Read [START_HERE](../START_HERE.md), [status](../CURRENT_SCIENCE_STATUS.md), [route](EXPERIMENT_ROADMAP.md), [code map](CODE_MAP.md), and the [integration ref change](../infrastructure/workspace/INTEGRATION_REF_CHANGE_20261028.json).

## Branch lifecycle

New bounded changes use short-lived feature/ops branches from current main and ordinary merge PRs. The owner fetches and reconciles concurrent commits before publication. Do not re-create a separate permanent branch for every checkpoint or report.

`pi0267/executor-decoupled-recovery` is temporarily retained for the prior publisher/branch-bound tooling migration. Its reviewed head is merged. Future publication targets main; if this old ref advances, compare and merge the new commits normally before any retirement. No archival tag or main merge authorizes active server changes.

Six exact obsolete heads are candidates in [BRANCH_RETIREMENT.json](maintenance/2026-10-28/BRANCH_RETIREMENT.json). **No branch refs were deleted in this connector session.** Existing GitHub tools do not expose ref deletion; the guarded helper is supplied for the authorized management owner, with default dry-run.

```bash
# From the consolidated management checkout, with Git and gh already authenticated:
python scripts/maintenance/retire_merged_branches_20261028.py
# Only after inspecting the dry-run; this applies only the six frozen candidates:
python scripts/maintenance/retire_merged_branches_20261028.py --apply
```

The helper uses a temporary bare Git repository, checks the exact repository and candidate HEADs, current main ancestry, branch protection and open PR dependencies, creates/readbacks same-name lightweight recovery tags, and deletes only the approved refs in an atomic expected-SHA transaction. It never checks out or deletes the user's worktree, main, pi0267, raw data, models, server leases or credentials. A moved/protected/dependent candidate aborts the apply rather than widening scope. Failed writes are not automatically repeated.

The lease option in the helper guards **deletion of exact old refs only**; it never force-updates any surviving branch or tag. Tags are archival identities, not a scientific release. The helper has not been executed or validated by local tests in this session; inspect it and run its dry-run before apply, then publish its actual JSON result to #47. Do not replay the October1 consolidation script, whose authority and candidate set were different.

## Migration of a management checkout

Save local dirty/untracked/stash and local-only history first. Fetch origin and verify that the former integration HEAD is an ancestor of origin/main. At a verified no-active-consumer boundary, switch the management worktree to the main integration line using normal Git operations; resolve any local divergence rather than reset/force. Do not change immutable running capsules, scientific sources, datasets/statistics, registered paths or clocks as a side effect.

The existing ENTRY_POLICY_V1 remains a historical deployment contract. The separate October28 ref-change record supersedes its publication branch only; actual host registry/entry adoption still requires a real receipt. The current publisher remains the existing root owner; no new subagent or paused automation is started by this migration.

## Evidence and validation

Keep Issue #11 for scientific decisions/results and #47 for operations. Write concise current status plus links to immutable dated evidence; do not repeatedly append full logs to the root entry. Historical contracts, negative results and archived issue bodies remain unchanged.

No large model/raw/video/vendor/private asset is newly added. Only already-authorized fixed-revision artifact routes are used for future execution; public HF model delivery does not make private assets public.

CPU workflow is the original richer experimental version, including PI0292/PI0294 checks. No new Actions run was obtained for this consolidation and local execution tools were unavailable. Report this as **NOT_RUN**, not CI PASS. GitHub mergeability, history and file readbacks validate repository integration only. Deploying any new runtime source needs the existing CPU/native gates.

See the [consolidation report](maintenance/2026-10-28/CONSOLIDATION_REPORT.md). Original October1/12/13 navigation remains under [the byte-preserved archive](archive/2026-10-28/pre-consolidation/).
