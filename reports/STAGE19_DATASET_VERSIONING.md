# Stage19 Dataset Versioning and Split Policy

## Version identity

Every derived dataset must record:

- immutable raw episode IDs and source hashes;
- calibration bundle ID/version and frame/TCP semantics;
- synchronizer and quality-gate policy versions;
- source timestamp authority and clock-domain metadata;
- state/action ABI version and orientation layout;
- materializer, processor, and dataset-loader code identities;
- horizon and explicit action timestamp provenance;
- dataset configuration hash and generated file manifest.

`RawEpisodeV1` is immutable audit input. A materialized LeRobot dataset is a
derived view and receives a new version when labels, timing, calibration,
modality, processor, or split policy changes.

## Split rules

- split by episode, never by adjacent frames or future windows;
- keep all windows from an episode in one split;
- record the split seed and counts;
- keep D3 simulation fixtures out of real-robot adaptation training by
  default;
- keep E1/E2 WBC benchmark trajectories out of VLA training unless a future
  study explicitly defines and audits that use;
- keep achieved EEF in raw/audit fields, not as an unannounced label;
- preserve source timestamps and never repair timing by changing the data
  silently.

## Source mapping

| Source | Stage19 role |
|---|---|
| D1 | VLA representation/pre-adaptation analysis |
| D2/D3 | simulated pipeline fixture and stress qualification |
| E1/E2 | native WBC benchmark only |
| WBC generator | controller support/distribution analysis |
| future real Go2+ARX5 demos | primary robot-specific adaptation data |

The Stage19 fixture is a materialization qualification artifact, not a claim
that the base GR00T checkpoint learned Go2+ARX5 behavior.
