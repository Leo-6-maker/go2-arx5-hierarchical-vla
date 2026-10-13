# Corelab Workspace Consolidation

Status: **ENTRY_READY_RETIREMENT_DEFERRED**. A real shared workspace and a
canonical current-branch checkout have been created and verified. Legacy
directories are NOT all retired;Projects does NOT physically contain only
one directory yet. This is a new maintenance task, not renewed physics or
training authority and not an expansion of previous Phase2 deletion scope.

## Actual Entry

Login to Corelab AMD and enter `~/Projects/GR00T-WBC`. It contains:

```text
GR00T-WBC/
  README.md
  code/vla/
  code/isaac-gr00t -> existing clean shared NFS worktree
  assets/ASSET_INDEX.json
  datasets/DATASET_INDEX.json
  datasets/PI0291/
  experiments/PI0291/
  infrastructure/
```

code/vla is a new independent Git checkout on
pi0267/executor-decoupled-recovery, pinned at creation to
e3cdb22a1802fcae1d97e670487d5bc60b5e5c0b, not main or an old stage45f branch.
Existing legacy branches/worktrees remain unchanged. Official shared
reference51d4c89f72fda44cbf77285c6a8114b52676b8a1 is a navigation alias only;
the two modified files in the original official checkout are protected.
The actual SSD processor/runtime path remains unchanged.

Server HTTPS clone initially lacked authentication. A credential-free exact
Git bundle from the authenticated local checkout created the independent
canonical checkout;no credential/helper/key was exported. The user then
confirmed public-key setup. Local-PC and AMD SSH authenticated successfully
as Leo-6-maker. Only the new canonical origin was switched to GitHub SSH,
and its live branch read verified;old remotes and global Git settings were
not changed. Public-key authentication is not itself a proof of branch write
rights;the report publication uses the existing local authenticated owner.

## Actual Verification

- Original four audits plus path checker pinned against the original artifact
  hash manifest;all448 JSON/CSV items structurally reconciled. No fresh
  1.81-million-file walk. This lightweight inventory is not complete global
  static-reference/ignored-file recovery qualification.
-160 top-level identity observations across AMD/4080S;mount/realpath/inode,
  Git SHA/worktree/normal untracked status, own process/FD metadata observed.
-27 new entry files read back with identical SHA and shared inode on both
  nodes;15 role paths either verified on their owner or returned HOST_REQUIRED.
  AMD local SSD absence does not mean4080S data loss. No cross-node SSD
  symlink was added and no raw data was copied back to NFS.
-18 pre-existing CPU engineering tests passed on each host. Four local
  resolver unit tests passed;canonical package imported in the existing4080S
  CPU environment. No new AMD environment or package installation. Fixed
  official pose source parses and matches existing SSD source bytes. These
  are path/code checks, not new model or task qualification.
- Shared original Git HEAD/status preserved. Four pre-existing worktrees
  fail Git queries on4080S because their gitdir metadata points to an AMD-only
  /mnt namespace:pi0240,pi0241,VLA-rtx4080s andGR00T-rtx4080s. These failures
  existed before this task;preserved,not repaired/pruned/deleted. New canonical
  checkout is independent and passes on both hosts.
- Hopper existing source path and pinned SHA were read;pretrained index
  metadata only,no full weight hash/load. Own qstat read succeeded. Registered
  alias maps to observed hopper-l-01;other hosts must not assume that identity.

## Data And Protected Assets

PI0291 remains PARTIAL:56 TRAIN,0 DEV,33,656 full H40 windows,official CPU
data chain and private HF readback verified. Dataset metadata binds fixed
revisionc25eeadf9be0aa7d46bb6f7b65e260cbf3a305ea and its file/SHA set, not mutable
HEAD. That archive proves only its covered data. It does NOT back up every
Home model, private WBC, environment or historical experiment. Trainer
adapter remains unqualified;no training-ready command was added.

Native robot/private WBC,PI0250/PI0276/PI0290/PI0291 raw and environments,
unique Full Trainer components,failed scientific evidence,historical archives,
legacy home venv and private _handoff remain protected. Home venv's dead
interpreter is not deletion permission. The new workspace contains indices
and pointers,not copies of large runtime data or robot meshes.

## Retirement And Space

RETIRE_CANDIDATES.json records24 project candidate observations across two
hosts and44 Home review entries. None has a complete independent recovery,
ignored/untracked/local-object uniqueness,worktree-safe retirement and global
consumer closure proof. Approved exact delete list is empty.

Actual old-file deletions0,old-directory moves0,new reclaimed allocated
bytes0. No rm/rmtree,worktree remove/prune/repair,process signal,cache unlink
or checkpoint deletion occurred. Prior Phase1 44,017,266,688 allocated bytes
(40.994GiB) is historical and not counted again;Phase2 delete/free0 remains.

The new checkout,bundle and indices add593,643,745 logical bytes and
607,711,232 unique-inode allocated bytes at the measured snapshot,about
0.57GiB. This is added storage,not reclaimed space. Btrfs exclusive/reflink
space and personal quota were not measured. Remaining administration/local
bundle CPU is not fully metered;known probe/deploy/readback usage is separate,
not perfect utilization. The initial failed HTTPS attempt was bounded and
changed no old assets. No old scientific budget/clock is reused.

## Next Exact Gate

First review the four cross-host Gitdir cases and each candidate's absolute
path consumers. A later plan must preserve ignored/untracked/stashed/local
objects,prove complete independent restoration and resolve two-node
process/FD/lease/static references. Only then retire an exact approved list
using proper Git worktree tooling and expected file/inode checks;automatically
skip unknowns. Do not turn directory names,clean status or old small copies
into deletion proofs. The aim of one physical Projects directory remains
pending until those gates pass.

The working entry is ready now. Existing local-PC remains the sole scheduler;
Corelab keeps simulation/data/CPU/video,Hopper keeps future neural work under
legal PBS. No simulation,model load/forward/train,new PBS,WBC,hardware,service
or tunnel started. selected=NONE;deployment=NONE.

Exact private directory/FD/runmap observations stay in the private workspace
and local owner;GitHub receives only sanitized summaries. #47 records actual
maintenance outcomes;#11 scientific protocol/PI0291 conclusions are unchanged.
