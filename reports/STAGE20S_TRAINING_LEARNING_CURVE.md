# Stage20S Training Ladder and Learning Curve

## Recipe

The official Isaac-GR00T experiment entry point was used with a
simulation-only, resource-bounded recipe. The projector/state/action encoder
path was trainable; the large language/visual backbone and diffusion
transformer were frozen. `tune_diffusion_model=false`, `tune_llm=false`, and
`tune_visual=false` are recorded in every run manifest. No real data or
hardware path was imported.

## 10/25/50/100 ladder

Each ladder point ran 50 optimizer steps, with H=40 and the explicit
test-only 0.05 s sample period. The values below are the final logged step
losses, not validation success metrics.

| Episodes | Rows | H=40 windows | Steps | First logged loss | Final logged loss | Status |
|---:|---:|---:|---:|---:|---:|---|
| 10 | 610 | 220 | 50 | 1.0204 | 0.9110 | PASS |
| 25 | 1525 | 550 | 50 | 0.9618 | 0.9527 | PASS |
| 50 | 3050 | 1100 | 50 | 0.9934 | 1.0170 | PASS |
| 100 | 6100 | 2200 | 50 | 1.0459 | 1.0505 | PASS |

The short ladder is a resource and pipeline sanity check. It does not support
a monotonic data-scaling claim: the final losses vary with the short-step
optimization and seed.

## Overfit sanity run

The 10-episode fixture was also trained for 500 steps. The first logged loss
was `1.0204`; the step-500 logged loss was `0.1179`. The run remained
simulation-only and used the same frozen-backbone policy. The checkpoint was
used only for the subsequent controlled visual-reach inference and native-WBC
simulation test.

## Interpretation boundary

This demonstrates that the data path can drive a robot-specific custom
embodiment adaptation run and that an overfit sanity checkpoint can emit a
stable test trajectory. It does not establish general visual-language task
performance, real-robot readiness, or the full official large-memory recipe.
