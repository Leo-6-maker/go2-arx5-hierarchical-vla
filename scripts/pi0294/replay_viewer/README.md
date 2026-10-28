# PC Episode Replay · Rerun + thin Python adapter

A local, read-only viewer of saved PI0294 Teacher and Student Eval facts. It reuses the field layouts in `scripts/pi0294/episode.py`, `task_state.py` and `night_r1_worker.py`; it never creates an Engine, calls `mj_step`, runs a model/WBC, generates demonstrations or recalculates task success.

## Default: official Rerun viewer

The default PC workflow now uses the mature **Rerun SDK / Viewer**, with a small Python adapter over the same read-only `Episode` loader. Playback, time scrubbing, camera panels, plots and 3D interaction belong to Rerun. The previous HTML prototype is retained below for historical compatibility; do not extend it into another 3D engine, timeline or player.

Use a separate PC Python 3.11+ environment and the exact tested SDK:

```bash
python -m pip install -r scripts/pi0294/replay_viewer/requirements-rerun.txt
python -m scripts.pi0294.replay_viewer.episode_to_rerun \
  --config /absolute/private/replay-config.json \
  --output /absolute/private/new-rerun-export
rerun /absolute/private/new-rerun-export/local-t1-teacher/local-t1-teacher.rrd
```

Each configured episode produces its own `.rrd`, an official `.rbl` blueprint and a private export receipt. The blueprint is also embedded in the recording, so normal opening uses the RRD alone. Output directories must be new; old exports and source episodes are never overwritten. The RRD contains private original RGB and audit details: keep it local, outside source Git and public HF.

The blueprint presents original BODY/WRIST RGB, world TCP/object/desired/accepted/committed positions, sample-point histories, saved gripper/error signals, original audit/provenance, full-precision current frame and Teacher/S1 logs. Source seconds and original frame index are separate timelines. Events with no recorded second timestamp are not assigned a fabricated time; their original sequence and raw JSON remain available. Sparse views retain the 41/641 declaration and export observed points only. Between sparse samples Rerun may hold the latest saved observation; this is **not** simulated/interpolated robot motion. Offline predictions remain purple entities labelled `NOT_EXECUTED`, never actual motion or task success.

No model or URDF is downloaded. An optional existing identity-bound local model can be supplied only through the documented legal-asset option; `mj_forward` reconstruction stays separate from Rerun's renderer. Without such assets, the export contains recorded poses/points and original RGB, not an invented robot skeleton or mesh.

Official API references: [operating modes / file save](https://rerun.io/docs/reference/sdk/operating-modes), [encoded original images](https://rerun.io/docs/reference/types/archetypes/encoded_image), [programmatic blueprints](https://rerun.io/docs/howto/visualization/build-a-blueprint-programmatically). The PC dependency is pinned to `rerun-sdk==0.40.0`; do not install it into immutable Core/Vanda runtime environments.

## Historical HTML prototype (compatibility only)



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

## Shared private input configuration

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

The original audit document is shown verbatim with its SHA. Its status is not interpreted as a new success decision. A saved negative witness can therefore be shown without pretending it was a completed task. Metadata/sample identities are verified on loading. The Rerun exporter verifies each original image before recording its encoded bytes; exported bytes were independently decoded and SHA-checked. The historical HTML service separately verifies images on request and in its browser. A changed image is refused.

Rerun uses its native timeline; the historical HTML slider advances saved observations with no interpolation. Optional high-rate joint state is the latest record at or before the observation; `state_time_s` is shown separately, and a missing initial state stays missing. Rows use the original `time + qpos + qvel + finite_joint_motor_ctrl + wrench6 + safety` layout, with `nq`/`nv` from the archive. No future joint state is substituted.

## Three-dimensional view and legal assets

Default mode displays recorded actual TCP, actual object and desired/committed EEF trajectories in an orbitable 3D plot. It does **not** draw an invented robot skeleton. This fallback is useful with original RGB while assets remain unavailable.

An optional `kinematic_model` binds a local `.mjb` or the original `scene.xml.json` by SHA. It is loaded only with `--allow-local-kinematics`. For XML, provide a list of `assets` with explicit relative VFS `name`, private local `path` and SHA; includes/plugins and undeclared files are refused. Scenes containing asset-directory directives require a separate explicit, identity-bound VFS export rather than a hidden path rewrite. No asset is copied, downloaded or disclosed by this code. Asset provenance/licensing must be established before this option is used.

Each state is reconstructed in a fresh `MjData` using recorded `qpos`, `qvel`, `ctrl` and recorded state time, followed only by `mujoco.mj_forward`. The optional adapter exports body-parent skeleton edges to Rerun, not full textured robot meshes. It neither integrates dynamics nor reevaluates contacts or score. Complete mesh rendering can be added after legal assets and an actual local replay have been validated.

## Offline prediction overlays

An optional `offline_predictions` object points to a local JSON file plus its SHA. Rows must identify `frame_index`, exact `observation_time_s`, `label`, `model_identity`, `coordinate_frame: "world"`, `xyz` and `executed: false`. They appear as separate purple saved prediction points labelled `NOT_EXECUTED` in Rerun; the historical HTML view uses dashed trajectories. Coordinate conversion must be explicit before export; predictions cannot overwrite actual TCP, joint states, object trajectories, recorded commands or original audit results. E0/E1/E2 predictions that do not yet exist must not be synthesized as scientific evidence.

## Validation boundary

The accompanying tests use synthetic files and a mock `mj_forward` to check read-only behavior, identity refusal, camera clocks, preceding-state selection, independent prediction overlays, no dynamics call, HTTP routes and loopback protections. They are not actual Teacher/Student playback, native model/physics qualification, a CI certificate or a scientific task result. Actual pinned-SDK roundtrip tests are separate from these nine synthetic reader contracts: eleven focused tests pass in the PC environment. Three actual sparse records (T1/T2 each41/641, C50 S1 two/224) have official RrdReader byte/semantic readback and root official Web Viewer inspection. C50 success, full continuous replay and legal full robot reconstruction remain incomplete; no new scientific result is inferred.

## Derived keyframe views and existing local image caches

A sparse local preview is a separate view, never a rewrite of the original episode. Set `samples_path` to a new file containing unchanged selected original sample objects. Its `samples_sha256` identifies that **derived file**. Add `recorded_view` with `kind: "recorded_keyframe_subset"`, `source_frame_count`, ordered `source_frame_indices`, the **original** `original_samples_sha256`, and a SHA-bound `selection_receipt` local relative file. The receipt repeats those source fields and contains `selected_row_canonical_sha256`, one SHA per selected object encoded with `json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)`. Preserve the native readback/selection provenance privately.

Existing images may be read without copying by listing explicit private `image_roots` and an `image_path_map`: each original camera path maps to `{ "root": "cache_name", "path": "relative_cache_filename.png" }`. Mapped files remain confined to that named local directory and must match the original sample's image SHA and timestamp. The original camera path stays visible in the frame data. This option does not authorize traversing unrelated folders or downloading new images.

Rerun explicitly labels `KEYFRAME SUBSET N/M — NOT CONTINUOUS REPLAY`, retains original source frame indices and logs sample point histories without connecting missing motion. The historical HTML prototype uses dashed visual guides between points; those are not interpolated physical states. Extra RGB records without associated saved sample rows must stay unmatched in provenance; no frame or state is manufactured to fill a mismatch. Compact numeric cards show five decimal places and xyz labels; the original full-precision values remain in the current-frame JSON.
