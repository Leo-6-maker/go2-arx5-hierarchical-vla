# Stage45E N2 Corelab lane

Stage45E N2 is a single-factor normalization intervention on the accepted
FULL24 TRAIN24 cohort.  It preserves the four robust XYZ centres and replaces
only selected dimensions `0,1,2` with the common half-range
`0.107349471 m`.  The T16 mask, StructuredActionV1 codec, K16/H40 timing,
dataset, model graph and FULL24 fresh-BASE recipe are unchanged.

The frozen identity and boundary are in
`manifests/STAGE45E_N2_BALANCED_XYZ_NORMALIZATION_V1.json`.

## CPU/static artifact

`scripts/stage45e_n2_statistics.py` validates the accepted runtime-V2 stats
outer/payload hashes and the FULL24 tree, rebuilds all 24 labels from the
materialized TRAIN parquet, and writes a new N2 stats artifact.  The artifact
does not contain a source scratch path.  On the Corelab mirror the resulting
audit records 4,560 active scalars and 1,080 XYZ scalars before clipping.

The Stage38 factory consumes the artifact only when
`STAGE45E_N2_STATS_ARTIFACT` is explicitly set.  It recomputes TRAIN
statistics first and accepts the bridge only if all non-intervention fields and
all non-XYZ selected bounds match (up to the explicit sub-micro-unit
parquet/JSON round-off tolerance).  The normalizer then uses the N2 XYZ bounds.

## Corelab execution boundary

`scripts/stage45e_n2_corelab.py --mode static` is CPU-only.  GPU modes require
explicit BASE, FULL24, GR00T, Cosmos, modality, config, runtime-stats, N2
artifact and GPU-UUID paths.  The runner checks identities before importing
torch/GR00T, binds the requested 32GB RTX 4080 SUPER UUID, and never kills or
reprioritizes another process.

The canary performs one model action generation and one training-batch
forward/backward.  It does not call an optimizer, scheduler or checkpoint
writer.  Only a passing canary permits the one fresh-BASE 2,000-step run with
checkpoints at 1,000 and 2,000.  Resource failure is reported without changing
precision, batch, horizon, offload, graph or module-freezing policy.

No Validation-E, Holdout-F, WBC, simulator, hardware or PBS path is touched.
