# Stage14 repository consolidation

Status: `PASS WITH DEVIATION`

## Canonical repository

- path: `/home/liuyu-nfs/Projects/go2-arx5-hierarchical-vla`
- local synchronized path: `/mnt/LocalHDD1/HomeNFS/liuyu-nfs/Projects/go2-arx5-hierarchical-vla`
- branch: `main`
- initial consolidation commit: `b9b58f857a21f6d5b7426ce43ca4007bc0e6e224`
- GitHub remote: not configured; `gh` is not installed on the login environment

The repository contains maintained project-owned contracts, trajectory math,
the offline GR00T-to-WBC adapter, provider interfaces, metrics, native-run
orchestration, tests, manifests, ADRs, and handoff documentation. It does not
contain the delivered lab source, checkpoints, videos, datasets, or USD files.

## Artifact boundary

The five committed manifests are logical references only:

1. delivered WBC archive, SHA256
   `b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167`;
2. Isaac-GR00T repository, commit
   `51d4c89f72fda44cbf77285c6a8114b52676b8a1`;
3. immutable P1M asset, SHA256
   `92d64f697d3f5ea6f17bbeaab72142607019f0309b5068c47f37a623499b9a53`;
4. E0/E1/E2 benchmark source and node-local reconstruction manifests;
5. D1/D2/D3 data-stage artifact references.

`.gitignore` excludes private checkpoints, archives, videos, USD, arrays,
datasets, run logs, W&B state, local path overrides, and credentials.

## Validation

- full CPU pytest: `17 passed`;
- dependency-light CPU contract smoke: `CPU_CONTRACT_SMOKE_PASS`;
- artifact manifest validator: `validated 5 artifact manifests`.

The native runs were executed against the consolidation commit above; their
machine-readable manifests preserve that exact integration SHA. Report-only
changes after the runs must not be confused with the code SHA bound to a run.

## Remote creation

No remote was created because authenticated GitHub CLI access was unavailable.
After installing/authenticating `gh`, the intended private-remote operation is:

```text
gh repo create go2-arx5-hierarchical-vla --private \
  --source /home/liuyu-nfs/Projects/go2-arx5-hierarchical-vla \
  --remote origin --push
```

This is deliberately left as a user-authorized follow-up and does not block
the local project repository.
