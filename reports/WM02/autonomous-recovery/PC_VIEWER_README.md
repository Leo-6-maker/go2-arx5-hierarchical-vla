# Private PC native-mesh replay

实时接入另见 [WM02 本体场景实时显示](../../../docs/operations/WM02_LIVE_VIEWER.md)。新的 `watch_native_live` 使用独立的只读原生状态通道；本页旧入口和 recorded-file bridge 仍只用于回放。

The implementation loads the **original same-episode compiled MuJoCo3.2.7 MJB** and original GRTSTATE journal, checks their SHA256 and ABI34/32/20, applies recorded qpos/qvel/ctrl/time, calls `mj_forward`, and synchronizes the native passive viewer. No interpolation, new physics, controller or safety qualification. The old failures are labelled `TARGET_FAILED_REPLAY`.

On the current PC the private variables are already resolved by the operator. A future review command (new output directory each time):

```bash
PYTHONPATH=src:. "$WM02_VIEWER_PYTHON" -m scripts.wm02.play_native_recording \
  --config "$WM02_REPLAY_CONFIG" \
  --output "$WM02_PRIVATE/replay-review-new" --max-wall 300 --rate 0.5
```

This entry opens recorded geometry on the PC and does not require reopening the finished experimental lease. The private config binds `scene.path/sha256`, `journal.path/sha256`, `episode_id`, `original_native_source_sha`, and label. Assets/configs/screenshots stay outside Git. Never substitute a different model for the recording.

- Native mouse: orbit, pan and zoom. Separate transport window: Pause/Play, drag original-time slider, playback speed. Arrow keys: one saved packet; Home/End: first/last packet. End of recording holds the last pose.
- T1:2700 original packets,2ms..5.4s. T2:27 packets,2ms..54ms. T1 was a failed task; T2 includes the original limit/STOP failure. These are not new successful robot motions.
- Offscreen render check:120 actual frames at960×640,102.495FPS for T1. Native passive UI:29.78–29.85FPS when capped at30; do not describe the capped display as a measured≥30FPS pass.
- Separate finite transport check actually executed pause→saved-packet seek→rate change/resume. A private replay producer/read-only file subscriber received141 updates, reached sequence2700, then held its last frame with49 stale display frames. This is a recorded-file bridge, **not live target**. No ROS/socket/action publisher exists.
- Native step guard plus Python forbidden-step assertions:0 integration entry calls. The guard is extra validation; it is not a substitute for the read-only implementation. Actual data and costs, including one missing-receipt correction, are in `PC_VIEWER_ACTUAL.json`.

Scientific workers used registered immutable source capsules and finite5/8/20s endpoints; both PC leases are now CLOSED. For a new registered bridge experiment, `scripts.wm02.replay_native_mesh --mode bridge-replay` requires a fresh admitted PC replay spec. No native target scope is inferred from GUI availability. Actual live target streaming awaits an independently admitted owner process and a separately bound read-only source.
