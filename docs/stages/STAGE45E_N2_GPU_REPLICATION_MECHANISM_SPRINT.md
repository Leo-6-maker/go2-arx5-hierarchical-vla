# Stage45E N2 GPU replication and mechanism sprint

This branch is the Corelab execution lane continued under PI0125 (Issue #11
comment 5635727857).  It starts at `a476a581f4d9a5ad104e691d4037762c6f1685ba`
and leaves the frozen PR #53 seed-2222 history untouched.

The only scientific intervention is the accepted N2 XYZ normalization:
the three XYZ selected bounds use the unchanged robust centers and common
half-range `0.107349471 m`.  FULL24 data, non-XYZ statistics, the T16 mask
(190 active scalars), decoder, model, optimizer, and training recipe remain
unchanged.  The two replication seeds (2223 and 2224) each start from the
exact BASE and run to steps 1000 and 2000.

The accepted legacy FULL24 checkpoint bundle is relayed byte-for-byte into a
fresh Corelab staging root and independently rehashed before any comparator
model import.  After the exact relay gate, the existing GPU queue continues on
the same RTX 4080S through G3 teacher-forced/local-flow and G4
teacher-forced-versus-free-running probes.  PI0125 then adds G5, limited to
the three already-trained N2 @1000 checkpoints under the same D1/D2A contracts;
no @2000 calls are duplicated.  The phase budgets and flow equations are
machine-readable in
`manifests/STAGE45E_N2_GPU_REPLICATION_MECHANISM_SPRINT_V1.json`.

All raw arrays are sealed before derived metrics.  CPU aggregation is deferred
until the GPU queue ends.  A missing identity, nonfinite result, frozen graph
OOM, or missing read-only legacy comparator is fail-closed; no precision,
batch, horizon, sharding, offload, retry, or checkpoint substitution is
permitted.  Validation-E, Holdout-F, WBC, simulator, hardware, PBS, new data,
and merge remain locked.  Broad CPU aggregation is deferred until G5 has
sealed.
