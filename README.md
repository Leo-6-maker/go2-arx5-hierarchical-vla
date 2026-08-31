# Go2 + ARX5 Hierarchical Whole-Body VLA

This repository is the source of truth for the project's integration code,
contracts, adapters, benchmark orchestration, and reproducibility documents.
Large or private inputs remain outside Git and are referenced through
`artifact_manifests/`.

## Current system boundary

```text
GR00T N1.7 (future runtime)
  -> timestamped EEF trajectory chunk (XYZ + SO(3), gripper separate)
  -> WBC reference adapter / nine-point SE(3) preview
  -> delivered native WBC (internal arm + body/gait coordination)
  -> native IsaacGym Go2+ARX5 embodiment
```

The external VLA interface is EEF-only at this stage. The delivered WBC's
structured 9D body/gait signal is an internal controller output, not a GR00T
label. The native IsaacGym embodiment is the controller baseline; the P1M
Isaac-Sim asset is a separate target-embodiment transfer case.

Stage15 adds an explicit `TrajectoryShapeV1` + `PlacementProfileV1` /
`EpisodeReferenceBindingV1` boundary. This prevents an E1/E2 absolute pose
from being mistaken for a native-episode reference. The binding is validated
against 12 native reset traces; the current aligned diagnostic set is not yet
a broad native benchmark.

## CPU checks

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

GPU/Isaac execution is intentionally not part of the CPU test suite. Native
WBC source, checkpoints, videos, datasets, and USD assets are private external
artifacts and must not be committed here.

## Stage15 diagnostic tools

The static/native audit entry points are kept in `scripts/`:

- `native_reference_trace.py` records native anchor and preview semantics
  without loading a policy checkpoint;
- `audit_native_reference_binding.py` checks the independent binding against
  those traces;
- `reinstantiate_stage15_benchmark.py` derives shape objects and native
  placement classifications;
- `prepare_stage15_native_inputs.py` materializes reset-bound external inputs.

Their large outputs belong under the node-local Stage15 artifact root and are
referenced by manifests and reports rather than committed to Git.
