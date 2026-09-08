# Accepted baselines and claim boundaries

This page separates engineering acceptance from scientific task claims. A
stage can be accepted as a useful, sealed baseline while still leaving the
next scientific question open. The source of authority is the stage evidence
and the latest Issue #11 PI gate, not this summary.

## Accepted layers

| Layer | Accepted baseline | What it establishes | What it does not establish |
| --- | --- | --- | --- |
| Data and interface readiness | Stage19–Stage23, with their documented deviations | Versioned raw/derived data contracts, official loader/processor boundaries, StructuredActionV1 semantics, and action-profile diagnostics are reproducible at their stated scope. | Physical data collection, arbitrary dataset generalization, or a clean model task result. |
| Native/WBC infrastructure | Stage34–Stage39 | Motion-API contracts, provenance-aware native authority handling, structured oracle qualification, and the sealed StructuredActionV1 native replay boundary are explicitly recorded. | A universal WBC guarantee; the Stage36 authority/tracking limitation is not erased by later stages. |
| Closed-loop integration | Stage40 | The exact GR00T → StructuredActionV1 → SafetyGate → reference-only handoff → native WBC loop closes on the frozen TINY1 development fixture, including the accepted M2 schedule. | General robustness, task success, realtime operation, or hardware readiness. |
| Long-horizon engineering | Stage41 | Ten reference-only replans and a fresh same-seed LH10 repeat remain lifecycle-stable and sealed on the same development fixture. | Adaptation to unseen tasks, target relocation, or statistical robustness. |
| Feedback causality | Stage42 | GR00T outputs change under the bounded, already-sealed observation counterfactuals; `endpoint_reached` is reference/trajectory completion, not task success. | Equal importance of modalities, target grounding, or closed-loop task success. |
| Controlled initial-condition characterization | Stage43 | Five fixed pre-episode ±5 cm base translations are characterized with exact integrity counters and fresh observations; the cohort is a characterization, not a general robustness result. | A success threshold, positive target progress, or generalization beyond the five conditions. |
| Stage44R attribution | Stage44R | Exact TINY1 source-cell fit is target-directed, while broader outputs and Stage43 references expose a model-side coverage/generalization limitation first at decoded GR00T output. | Model training failure, binding/WBC root cause, task success, or a data-expansion recipe already being authorized. |

## Stage44R accepted diagnosis

```text
STAGE44_TINY1_SINGLE_SAMPLE_COVERAGE_LIMITATION_SUPPORTED
```

The narrow evidence chain is:

```text
24/24 expert labels target-progress positive
        ↓
exact STRUCTURED@1000 TINY1 source cell
        ↓
predicted progress +0.113261 m
endpoint target-vector cosine +0.997154
endpoint error 0.009431 m
        ↓
broader Stage43 decoded references 20/20 negative
installed references 20/20 negative
achieved EEF 20/20 negative
```

This supports the diagnosis that the frozen checkpoint can reproduce its
exact training cell but does not cover the broader state/target windows. The
negative Stage43 behavior appears in the decoded GR00T reference before the
binding or WBC tracking layer. Stage44R performed no new forward, WBC step,
reference update, training, or data generation.

## Claims deliberately not established

The accepted baselines do not establish any of the following:

- binary task success or a validated centimeter success threshold;
- general target or initial-condition robustness;
- general manipulation capability or visual scene grounding beyond the
  tested fixture and counterfactual boundaries;
- realtime/asynchronous latency qualification;
- physical Go2 + ARX5 hardware readiness or safety;
- that the current TINY1 checkpoint is suitable for H200 fine-tuning without a
  new, explicitly authorized data/training gate.

Stage45 data expansion design and H200 fine-tuning remain `HOLD` until the
repository-consolidation PR is reviewed and a new PI gate authorizes them.
