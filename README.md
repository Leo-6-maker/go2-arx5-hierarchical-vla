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

## CPU checks

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

GPU/Isaac execution is intentionally not part of the CPU test suite. Native
WBC source, checkpoints, videos, datasets, and USD assets are private external
artifacts and must not be committed here.
