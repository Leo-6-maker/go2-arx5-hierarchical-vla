# PI0159 conditioning-mechanism result

Status: **COMPLETED (CPU reconstruction from sealed raw)**

Science classification: **STATE_AND_VISION_REACH_LOCAL_FIELD**

Both single-modality substitutions changed the active teacher-forced local field at the sampled flow times; standardized-T tails show the corresponding downstream trajectory change without treating anchor p0 mechanics as learned state response.

The original model attempt completed one model construction, four conditioning
passes, and twelve teacher-forced action-head calls. Its raw bundles were
sealed before the aggregation bug. This packet performs no model import,
model forward, sampler call, training, WBC, simulator, or closed-loop retry.

The fixed local-flow algebra is `x_t=(1-t)x0+t*x1`, `v_target=x1-x0`, and
`x1_hat=x_t+(1-t)*v_hat`; the exact N2 runtime-V2 decoder is applied only
after raw verification. State clipping is reported per condition. The
standardized-T anchor is the primary cross-condition tail view, while the
actual-anchor view is retained as decoder mechanics. TRAIN world-target and
task-success metrics remain unverified.

## Single recommendation

Authorize one fixed-budget independent-expert / visited-state local-field
supervision diagnostic before any ICL or training decision, using the same
raw-first ABI on post-policy states. Do not start it automatically.
