# Actual CoreLab September retirement

Authority: Issue #47 comment 6140769226. Maintenance run
`20261013T074558Z-localpc-september-retirement`, independent of the original
PI0292 science clock and budgets.

Actual deletion and read-back, not a dry run:

| Storage | Deleted last-link file allocation accounting | Observed filesystem available increase |
| --- | ---: | ---: |
| CoreLab shared NFS | 93.30 GiB | 93.01 GiB |
| CoreLab 4080S local SSD | 76.03 GiB | 75.86 GiB |

The file figures count `st_blocks` at the final regular-file link, once per
storage. They are not a unique-extent release proof on Btrfs. Filesystem space
is measured separately; shared writers, filesystem metadata and retained small
receipts prevent attributing the entire observed delta solely to this actor.

The named Home PI0209/PI0240/PI0246/PI0215/PI0220/PI0223 and stage13 staging,
175 explicitly September-dated SSD candidates, September PI0250-PI0263 cohort,
and qualified cold September archive copies were actually retired. Exact
candidate roots are absent; all 11 retirement actors are terminal and absent.
Earlier syntax/FD/permission/hardlink holds and their repairs remain recorded.
The stale September PI0256 NFS alias was removed after target absence and fresh
consumer checks, with zero additional bytes claimed.

Current PI0291/PI0292 data and source passed byte-level verification: 80 native
numeric files, 96,160 unique RGB files, 64 TRAIN / 16 DEV, unchanged bank manifest
and TRAIN statistics, both physical source seals, and all 30 active PI0292
operations seal files. AMD management, 4080S simulation and CPU-data entry
doctors passed again after the Hopper navigation metadata update. No model,
physics, WBC, hardware, GPU or PBS work was added by maintenance.

Current October experiments and the mixed original WBC root remain protected.
This is not a claim that all historical storage of any date has disappeared.
Another pre-existing maintenance actor owns Hopper deletion; its releases are
not counted as this CoreLab actor's work. Credentials, exact paths, process
identities and detailed inode manifests remain private.

See `RETIREMENT_RECEIPT.json` for exact byte counts and bounded accounting.
