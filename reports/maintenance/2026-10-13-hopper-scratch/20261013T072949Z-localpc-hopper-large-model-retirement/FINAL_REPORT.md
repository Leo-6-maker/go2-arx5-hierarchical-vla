# Hopper September Scratch Retirement — Final Report

- Run: `20261013T072949Z-localpc-hopper-large-model-retirement`
- Fixed GO: `33bbc2933e8bd6554d49b04f536519c10d0b962f`
- Status: **complete for the user-authorized September 2026 scope**.

## Actual deletion and quota

The verified receipts cover **282 retired roots**, with **193,073 regular files**, **796 symlink objects inside those roots**, **10 additional personal symlink aliases**, and **24,055 directories** verified absent. The Stage22 set, including its 196,000/200,000 recovery state, was permanently discarded. No full-size backup or HF upload was made.

Hopper’s personal quota report fell from **940.7 GB / 1.0 TB** at 15:43 SGT to **707.4 GB / 1.0 TB** at 17:53 SGT, an actual reported reduction of **233.3 GB (about 217 GiB)**. The ≤850 GB soft target was met. Per-path allocation was not used as a release claim because some paths shared hard-linked inodes.

Deleted candidates included September Stage22, 28 PI experiment roots, 183 archived stage runs, 41 repository mirrors, 15 archive remainders, one personal legacy-storage root, and one miscellaneous experiment root. The `archive/stage_runs` directory is now empty. An additional exact-manifest pass removed a September Xet temporary cache (15 files, 7 directories; 175,456,256 allocated bytes); the 0.1 GB-resolution quota display remained 707.4 GB.

## Protected PI0292 dependencies

The seven protected N1.7 files matched their expected sizes and hashes. The pinned official source commit remained clean; the fixed 10-file Cosmos snapshot was present; and the PI0292 processor constructed with `new_embodiment` at projector index 10 and PI0292 statistics attached to the input and state-action processors. No model was loaded and CUDA was untouched.

The Python runtime check remains incomplete: `uv pip check` reports seven missing declared packages (listed in `PROTECTED_HASH_READBACK.json`). No package installation, training, GPU work, or PBS submission was attempted.

## Scope boundary and remaining conditions

No September candidate root remains from the audited manifest. The remaining experiment roots were dated **October 1–9** (including PI0286 on October 9), so they were outside this September-only authorization and remain untouched. Home was also untouched; its quota remained 20.52 GB / 40 GB. A Home top-level after-list has 63 entries, but a task-time Home before-list was not captured. The Projects root changed from 12 entries to 7; five personal symlink aliases were removed.

The small Hugging Face snapshot cache still contains only configuration/index/provenance files (three blobs totaling 107,154 bytes; no weight tensors), retained as audit metadata. Other generic cache namespaces have top-level dates in May and August, predate September, and were outside scope; their recursive size was not accounted.

The final quota is sufficient against the prior 80 GiB peak plus 20 GiB margin estimate, but the old PI0292 preparation still says `selected: NONE` and `authorizes_execution: false`. This cleanup does not refresh project authority or restore the expired compute project. The user must confirm a current valid allocation before any future job.
