# Fresh target observation at an actual native pause V2

This is an independently callable capture seam, not a new execution phase,
model server, scheduler, scope, or native qualification. It starts no worker.
The V1 capture, archived-query reader, inference entry and sealed receipts stay
unchanged. The helper is `world_model.fusion_paused_observation`.

`capture_paused_observation(adapter, binding=..., task=...,
task_sha256=..., query_uuid=..., capture_index=0)` requires the current admitted
owner's `TargetSequentialAdapter`, its already prepared `ActualState` reader,
and a `TargetShadowBinding` pinned to this run's execution-spec and compiled
scene bytes. The caller first reaches a real paused boundary through the
original native lifecycle. It does not pass a `paused=True` assertion.

The helper calls the adapter's actual `observe()` before and after capture,
compares measured qpos/qvel/ctrl with the exact current `state.bin`, reads the
original `pause_ack.json`, checks the exact current `step_quota`, and binds the
original `quota.binding` to this native output and stage identity. The adapter's
original `capture_loaded_identity()` identifies its actual owned simulator and
loaded executable. The pause ACK must advance while physics time, FSM motion
counter, leg-history count and gait indices remain fixed. Raw measured bytes
and quota must remain identical. This does not restore complete MPC state.

The original `FrameRecorder.save` renders both 256x256 target cameras from a
copy of that same measured packet, without stepping it. Original `ActualState`
and `TargetShadowBinding.bind` check packet, camera FK, calibration, PNG bytes,
explicit rigid frame map, TCP, original simulation clock and measured gripper.
The original sealed shadow ABI is retained unchanged inside a new
`WM02_FUSION_LIVE_PAUSED_OBSERVATION_V2` envelope. Its old `shadow_only` and
unqualified camera/embodiment statements are not rewritten. The outer envelope
records the actual live paused producer, task binding, query/snapshot UUID,
owner/runtime identity, before/after pause evidence, and file hashes. Camera
domain shift is an experimental condition; it does not prohibit a new,
explicitly defined target-task diagnostic.

Before constructing a renderer, the caller's launch environment must have
`CUDA_VISIBLE_DEVICES=''`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`,
`LIBGL_ALWAYS_SOFTWARE=1`, and
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json`, with
no `MUJOCO_EGL_DEVICE_ID`. The helper does not change the environment or system
configuration. After the real save it reuses the existing renderer-identity
query and requires actual Mesa `llvmpipe`, `softpipe`, or software-rasterizer
identity; the result is retained in the envelope. Empty CUDA visibility alone
is insufficient evidence of CPU OpenGL rendering.

The task document has exactly these fields, with its canonical JSON hash
supplied by the owning preregistered condition:

```json
{
  "schema": "WM02_FUSION_PAUSED_TARGET_TASK_V2",
  "task_id": "a declared target short-skill identity",
  "language": "the exact instruction sent to the frozen processor",
  "scene_sha256": "the actual compiled scene SHA256",
  "score_contract_sha256": "the separately frozen score definition SHA256",
  "language_provenance": "NEW_TARGET_SHORT_SKILL_PROMPT"
}
```

`EXACT_SOURCE_TASK_PROMPT` is the other allowed language provenance. The helper
binds the declaration and scene; it does not prove visual task interpretation,
reachability, score validity or task success. Score geometry, object truth,
teacher stage and goals are not added to the observation payload.

The fresh directory is
`native-output/PAUSED_OBSERVATIONS/query-uuid/`. Its capture intent is retained
before rendering. Existing query directories, including incomplete attempts,
are never reused. `paused_inference_payload(record)` verifies the full V2
envelope and nested original binding, then returns the unchanged observation
ABI for the official processor. It neither loads nor calls a model.

After a separately admitted inference returns, the caller must call
`verify_still_paused(adapter, record)` before considering installation. That
check rereads actual feedback, unchanged native bytes/quota, runtime identity,
same owned simulator PID/executable, and a fresh native pause ACK. It checks
neither the model result nor execution eligibility. The caller must separately
bind checkpoint/processor/seed/noise/action hashes to this observation and
retain original full-preview certification, BASE hold, gripper packets,
installed ACKs, source-relative TTL, consumption evidence and native safety.

The old target query is already complete. It used the source instruction:
“Pick up the red cube, place it inside the blue square, release it, and move
the gripper away.” Its F2T06 producer was a TRAIN calibration HOLD/open
episode, not a registered execution/scoring of that instruction. The new task
must be declared before a fresh query, rather than relabeling that old draw.
The available local target lineage names `task_cube`, `task_cube_free`,
`task_cube_geom` and four `task_goal` geoms; `nq=34` alone is not evidence for
a book object. A book task requires actual book-scene evidence of its own.

A concrete new diagnostic is an approach step with the instruction “Move the
open gripper a little toward the red cube, then hold without touching it.” A
separately frozen score can report actual reduction in distance to an
object-relative precontact point, signed progress toward it, lateral drift,
actual opening, object movement, contact and original safety. Precontact
arrival and a complete pick/place remain separate outcomes. Thresholds and
any required hold duration belong to the new score contract and must be fixed
before evaluation. The helper implements no score or threshold.

A single fresh query followed by execution/renewal of its fixed proposal is
`FRESH_TARGET_QUERY_WITH_NATIVE_PROPOSAL_REPLAY`. Reusing that proposal across
independent direct/WM restarts can isolate a correction experiment, but does
not establish sequential VLA feedback. A later feedback comparison must query
each branch's own newly captured RGB and measured state after its own actual
prefix, using the same frozen model, processor, adaptation rule and
preregistered seed schedule. It must not copy the other branch's pixels or use
the archived T06 observation as its new observation.

Multiple-chunk control is not implemented here. For a future version, a new
query's reference anchor must be the current installed reference at its actual
execution boundary, while its desired anchor is the current measured TCP.
Repeatedly supplying the original2503 reference to every new query would rewind
the reference. Within a query, ordinary renewals must remain exact persistent
suffixes and must never reanchor to achieved motion. Query cadence must account
for the bridge and retained-source-knot retiming; renewing a pose preview every
seven atoms is not itself a request for another VLA draw.
