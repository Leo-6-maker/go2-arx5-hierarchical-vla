# PI0257 video evidence, review revision v2

This revision changes presentation and package verification only. It adds zero
model queries, training, backward calls, physics steps or WBC actor calls.
The fixed v1 cohort, predictions, noise, original videos and FAIL/NONE outcomes
are unchanged. v1 presentation and its original capability labels are retained
as historical files in legacy_v1, not endorsed as current capabilities.

## Commands

Use the existing Corelab PI0250 Python runtime, or a local Python with NumPy,
OpenCV and FFmpeg. FFprobe is optional; FFmpeg showinfo supplies actual frame
timestamps if it is unavailable. Set PI0257_FFMPEG / PI0257_FFPROBE for custom
paths. An already-installed imageio-ffmpeg binary is also detected.

The default root is the task-local Corelab PI0257_v2 directory. After extracting
the ZIP, pass --root to the directory containing delivery/.

    python eef_video_workflow.py plan
    python eef_video_workflow.py render --root /path/to/bundle
    python eef_video_workflow.py render --root /path/to/bundle --kind actual --playback 1
    python eef_video_workflow.py render --root /path/to/bundle --kind preparation --playback 1
    python eef_video_workflow.py verify-portable --root /path/to/bundle
    python eef_video_workflow.py verify-corelab --root /path/to/bundle
    python eef_video_workflow.py package --root /path/to/bundle

No arguments means a read-only plan. render --kind also accepts predictions or
all. The prediction video keeps its complete 32 chapters / 64 seconds; actual
recording speeds are explicit and keep the same source timestamps.

verify / verify-portable needs only the extracted package and CPU media/math
dependencies. It checks file size/SHA, full media decode, timestamp order,
saved input/noise hashes, 96 saved-trajectory metric recomputations, initial
rotation-jump values and preparation telemetry. It does NOT reconstruct the
processor or run the full gate implementation. Its receipt is outside delivery,
so verification does not invalidate the sealed archive.

verify-corelab additionally reconstructs the frozen processor, raw-action decoder
and full gate chain. This requires the original Corelab runtime and workspace.
It is deliberately separate from portable QA. Neither verifier calls a model.

## Presentation and capability boundaries

The main prediction video has fixed numeric scales across all chapters and
models, XY/XZ views and XYZ-error curves. HTML provides 32 fixed-scale triptychs,
32 explicitly marked local zoom triptychs and 96 model-cell zoom cards.
SafetyGate legality, WBC tracking and target completion are separate judgments.

01_metrics.json is unchanged. 01_metrics_v2.json appends explicitly named
raw_h40_* versus sampled_native_reference_* values, ranges and origins.
Historical PI0252 TCP telemetry is not a cache-corrected physical measurement.

run-episode is GUARD_STUB_ONLY: a rejecting entry, with the actual physical
runner / receding dispatcher NOT_CONNECTED. Separate authority plus actual
observer non-interference and prepared-state admission would be needed before
a future executor could be connected. Nothing here enables new physics.

The original export and infer-recorded adapters remain Corelab-bound for
provenance. This revision does not invoke them. Scientific identity hashes
exclude presentation settings, and completed valid cells are never rerun.

evidence_so3.py is the unmodified public project's NumPy-only trajectory/so3.py.
portable_numeric.screen_metrics uses the same admitted pi0243 scorer formulas,
independently applied to saved decoded arrays. Private WBC/asset code, weights,
tokenizers and credentials are excluded.
