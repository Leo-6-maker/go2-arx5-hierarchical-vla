# Episode Replay Viewer V1

A local, read-only viewer of saved PI0294 Teacher and Student Eval facts. It reuses the field layouts in `scripts/pi0294/episode.py`, `task_state.py` and `night_r1_worker.py`; it never creates an Engine, calls `mj_step`, runs a model/WBC, generates demonstrations or recalculates task success.

## Run on the PC

From a repository checkout with Python 3.10+ (NumPy is needed only when opening `highrate.npz`):

```bash
python -m scripts.pi0294.replay_viewer.server --config /absolute/private/replay-config.json --check-only
python -m scripts.pi0294.replay_viewer.server --config /absolute/private/replay-config.json
```

Open `http://127.0.0.1:8765`. The service binds only to loopback and rejects foreign Host/Origin values. It has no upload, write, delete, SSH or remote fetch endpoint. Stop it with Ctrl+C. Source paths stay private and are not sent to the browser; sample values and original audit content are visible locally.

A clearly marked synthetic UI demonstration can be created in a new local directory:

```bash
python -m scripts.pi0294.replay_viewer.demo --output /absolute/private/new-synthetic-demo
python -m scripts.pi0294.replay_viewer.server --config /absolute/private/new-synthetic-demo/config.json
```

Synthetic four-example fixtures are UI checks, not the requested four actual scientific records.

## Private input configuration

The management owner supplies already-authorized local copies of four existing episodes: T1 Teacher success, T2 Teacher success, C50 T1 success, and C50 S1 rejection or full task failure. Do not relaunch collection or evaluation to obtain replay footage. Do not publish the configuration, robot assets, raw images or copied source folders to source Git or public HF.

```json
{
  "episodes": [{
    "id": "local-t1-teacher",
    "source_kind": "teacher",
    "task": "T1",
    "fixture": false,
    "format": "fixed_base",
    "root": "/absolute/private/episode",
    "samples_sha256": "<original samples.json SHA256>",
    "audit_path": "INDEPENDENT_CPU_AUDIT.json",
    "audit_sha256": "<original audit SHA256>",
    "highrate_sha256": "<optional original highrate.npz SHA256>",
    "provenance": {"source_run": "<recorded identity>", "source_commit": "<original full SHA>"}
  }]
}
```

Set `source_kind` explicitly to `teacher`, `student_eval` or `offline_prediction`; a model output is never silently reclassified as a physical fact. `format` is `fixed_base` (`samples.json`) or `native_body` (`samples.jsonl`). This version does not parse or execute ROS bags. Native-body frames need an explicit `native_camera_index`, keyed by decimal sample index and camera name; each entry gives `path`, original `sha256` and exact `timestamp_s`. Missing channels are visibly missing.

The original audit document is shown verbatim with its SHA. Its status is not interpreted as a new success decision. A saved negative witness can therefore be shown without pretending it was a completed task. Metadata/sample identities are verified on loading. RGB identities are verified by the server on every request and independently by the browser before display. A changed image is refused; stale images are removed when seeking.

The slider advances saved observations with no interpolation. Optional high-rate joint state is the latest record at or before the observation; `state_time_s` is shown separately, and a missing initial state stays missing. Rows use the original `time + qpos + qvel + finite_joint_motor_ctrl + wrench6 + safety` layout, with `nq`/`nv` from the archive. No future joint state is substituted.

## Three-dimensional view and legal assets

Default mode displays recorded actual TCP, actual object and desired/committed EEF trajectories in an orbitable 3D plot. It does **not** draw an invented robot skeleton. This fallback is useful with original RGB while assets remain unavailable.

An optional `kinematic_model` binds a local `.mjb` or the original `scene.xml.json` by SHA. It is loaded only with `--allow-local-kinematics`. For XML, provide a list of `assets` with explicit relative VFS `name`, private local `path` and SHA; includes/plugins and undeclared files are refused. Scenes containing asset-directory directives require a separate explicit, identity-bound VFS export rather than a hidden path rewrite. No asset is copied, downloaded or disclosed by this code. Asset provenance/licensing must be established before this option is used.

Each state is reconstructed in a fresh `MjData` using recorded `qpos`, `qvel`, `ctrl` and recorded state time, followed only by `mujoco.mj_forward`. The current minimal renderer shows body-parent skeleton edges, not full textured robot meshes. It neither integrates dynamics nor reevaluates contacts or score. Complete mesh rendering can be added after legal assets and an actual local replay have been validated.

## Offline prediction overlays

An optional `offline_predictions` object points to a local JSON file plus its SHA. Rows must identify `frame_index`, exact `observation_time_s`, `label`, `model_identity`, `coordinate_frame: "world"`, `xyz` and `executed: false`. They appear as separate purple dashed trajectories labelled “未执行”. Coordinate conversion must be explicit before export; predictions cannot overwrite actual TCP, joint states, object trajectories, recorded commands or original audit results. E0/E1/E2 predictions that do not yet exist must not be synthesized as scientific evidence.

## Validation boundary

The accompanying tests use synthetic files and a mock `mj_forward` to check read-only behavior, identity refusal, camera clocks, preceding-state selection, independent prediction overlays, no dynamics call, HTTP routes and loopback protections. They are not actual Teacher/Student playback, native model/physics qualification, a CI certificate or a scientific task result. Four actual episodes and browser inspection with their real images remain required for full user acceptance.

## Derived keyframe views and existing local image caches

A sparse local preview is a separate view, never a rewrite of the original episode. Set `samples_path` to a new file containing unchanged selected original sample objects. Its `samples_sha256` identifies that **derived file**. Add `recorded_view` with `kind: "recorded_keyframe_subset"`, `source_frame_count`, ordered `source_frame_indices`, the **original** `original_samples_sha256`, and a SHA-bound `selection_receipt` local relative file. The receipt repeats those source fields and contains `selected_row_canonical_sha256`, one SHA per selected object encoded with `json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)`. Preserve the native readback/selection provenance privately.

Existing images may be read without copying by listing explicit private `image_roots` and an `image_path_map`: each original camera path maps to `{ "root": "cache_name", "path": "relative_cache_filename.png" }`. Mapped files remain confined to that named local directory and must match the original sample's image SHA and timestamp. The original camera path stays visible in the frame data. This option does not authorize traversing unrelated folders or downloading new images.

The UI explicitly labels `关键帧子集 N/M · 不插值`, retains original source frame indices and uses dashed connections between sample points. Those connections are a visual guide, not an interpolated or complete physical trajectory. Extra RGB records without associated saved sample rows must stay unmatched in provenance; no frame or state is manufactured to fill a mismatch. Compact numeric cards show five decimal places and xyz labels; the original full-precision values remain in the current-frame JSON.
