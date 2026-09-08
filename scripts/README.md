# Scripts index

`scripts/` contains both reusable audit helpers and historical stage runners.
The stage names are part of the provenance contract: reports, manifests, and
Issue #11 comments refer to them by path. They are not interchangeable public
APIs, and this consolidation does not move or delete them.

## Reusable boundaries

- `native_reference_trace.py` and `audit_native_reference_binding.py` — native
  reference/placement checks.
- `native_wbc_benchmark.py` and `analyze_stage16.py` — native benchmark and
  descriptive analysis surfaces.
- `stage40_corelab_evidence.py` and `stage40_corelab_phase.py` — sealed
  execution-bundle and lifecycle helpers.
- `stage40_d1_offline_parity.py` and `stage40_structured_decode.py` —
  CPU/read-only interface parity and decode checks.

## Stage runner families

| Family | Representative paths | Boundary |
| --- | --- | --- |
| Stage19–24 | `stage19_*`, `stage20s_*`, `stage20t_*`, `stage21_*`, `stage22_*`, `stage23_*`, `stage24_*` | Data readiness, adaptation diagnostics, and provenance audits; some historical runners contain gated GPU/PBS entry points. |
| Stage25–32 | `stage25_*` through `stage32_*` | Checkpoint/optimizer/sampler diagnostics and posthoc analyses; use only the exact stage gate and external artifact identities recorded in the reports. |
| Stage33–39 | `stage33_*` through `stage39_*` | Bottleneck, Motion API, oracle, codec, offline, and sealed native replay stages. Historical PBS wrappers are under `hpc/`. |
| Stage40 | `stage40_*` | Runtime/evidence/lifecycle code for the accepted GR00T→WBC engineering baseline. The consolidation task does not invoke any of these runners. |
| Stage41–44R | `stage41_*`, `stage42_*`, `stage43_*`, `stage44_*`, `stage44r_corelab_attribution_repair.py` | Long-horizon, causality, controlled-cohort, attribution, and zero-forward repair evidence. Stage44R is the accepted source tip. |

## Stage44R boundary

`stage44r_corelab_attribution_repair.py` is a zero-forward posthoc analyzer.
It consumes sealed Stage43/Stage44 artifacts and must not be repurposed for a
new model call, WBC step, reference update, training run, or data-generation
job. Its accepted result is the narrow diagnosis
`STAGE44_TINY1_SINGLE_SAMPLE_COVERAGE_LIMITATION_SUPPORTED`.

## Running a script

Read the corresponding report and gate before invoking any historical runner.
Most scripts require node-local model/WBC/data paths that are intentionally
not present in this checkout. For repository maintenance, limit execution to
`--help`, static parsing, or explicit CPU/self-test modes that do not import a
model or simulator.
