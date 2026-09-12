# PI0149 native C10 fixture delivery

This is a CPU-only, synthetic-test-only handoff. It is not a model result and
does not release H200, PBS, model forward, training, Validation-E, Holdout-F,
WBC, simulation, hardware, or merge.

## Producer and consumer pins

- Gate: Issue #11 comment `5646608846` (`PI_GATE_STATE_0149`)
- Producer PR: #66, branch `stage45e/h200-unified-replication-campaign`
- Producer source HEAD: `757f91b6e1702f256be55889ccdcf22391561e2f`
- Producer script SHA256: `735c2fba5867b1ad7ae87bd2124e4f6b02d9962a408d56b096180856a9201e58`
- Contract: `manifests/STAGE45F_PI0149_VANDA_C10_PRODUCER_CONSUMER_V2.json`
- Contract SHA256: `4c8479abb4ef2de90557eea604bd5bbc18a8311c4cc5b11b1de75aaba2564f8f`
- Consumer PR #65 HEAD: `d080f4e0011e9c2bed4827dc1c6988c761b2ed0f`
- Consumer entrypoint: `scripts/stage45f_corelab_evidence_convergence.py::build_packet`

## Transport artifact

- Archive: `artifacts/pi0149_native_fixture_757f91b.tar.gz`
- Archive SHA256: `33d6cb0d2aa3cef204db2f125636cfec4981c34f9470444fdd87c697ccc250f5`
- Archive bytes: `1784826`
- Archive entries: `251`
- Transport: committed to the PR #66 branch for Corelab to fetch by exact
  commit/path; the prior PI0147 archive remains unchanged.
- Prior PI0147 archive SHA256: `f11f9aec6e8ae9d09d30b3a02864d56a277eca4687412124e1e0ad28989caa78`

The archive is unpacked under its preserved root
`stage45e_pi0149_native_producer_fixture/`. `PATH_RELOCATION_MAP.json` is a
resolution-only map for absolute provenance paths; sealed JSON bytes must not
be rewritten or rehashed after unpacking.

## Native package hashes

- `fixture_manifest.json`: `3581692d2a464cba8990243e226f55c1a2d6d09d6b96e65ee5097536a94e74a7`
- `PATH_RELOCATION_MAP.json`: `7968012c001bf64d7af7a19332ff27db2d59e26c4dc1e4f16c79a6eb76493877`
- `vanda_producer/NATIVE_PRODUCER_POINTER.json`: `aeea2d3c288dcf0231d6bae96e76eba5ee86397e6fd9f924a429535df8996a9e`
- `vanda_producer/final_result.json`: `472542701927d2d129da71672ce1aa9e57b46cce951d3004606a81b546d7a0ed`
- `vanda_producer/cpu_summary.json`: `d22b2da09d9fe6b8289265e8eb40137afc3694a74df30b2cfe70b680100fdd73`
- `vanda_producer/CPU_EVIDENCE_INVENTORY.json`: `352fcfb2f610a67098df375af86155c052250a010452e815186dcf756a7a1200`
- `vanda_producer/C10_HANDOFF.json`: `2a009d12e5149a3ec4278da72d7ffe0b07ddad37f2120ba2ac627318b7c4c645`
- `vanda_producer/canonical_row_index.json`: `d1f12d186ccb02141f6eda9ab2b999cca50d268709b39de6ea1b3e360cc14592`
- `vanda_producer/c10_metric_index/canonical_e4_metric_index.json`: `1d721b528cadcbaf5e7ecebbacb8b529ffcfb737c6ffc172a2680b37621532e1`

## Observable fixture content

- 18 arms, 24 windows per arm
- 336 current E4 source rows + 96 historical E4 source rows = 432 physical metric rows
- 27 same-seed physical metric pairings
- 1,296 phase-reference index entries, explicitly not physical metric rows
- 8 historical bindings preserved as read-only references
- D1/D2A rows retain their phase-specific probe semantics; they are not
  reclassified as E4 metrics
- target XYZ input: false; raw-before-derived: true
- GPU/PBS/model forward/training: 0

## CPU consumer verification

After a fresh unpack and running from the preserved archive root, the pinned
Corelab `build_packet` returned:

```text
status=PASS_DESCRIPTIVE_H200_TRAIN24_ONLY
canonical_e4_row_count=432
paired_comparison_count=27
validation_e_read=false
holdout_f_read=false
winner_declared=false
```

This is a producer/consumer CPU fixture check only. Independent Corelab
receiver receipt is still pending; no receipt is claimed by this file.
