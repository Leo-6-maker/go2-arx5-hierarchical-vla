"""Seal two causal, synchronized sensor anchors from a closed H0 episode."""

import argparse
import json
from pathlib import Path

import numpy as np

from source_custody import pin, require
from sea_fruit_camera import FruitBodyRenderer, FruitCameraSession
from sea_three_view import calibration, native_sample_identity
from policy_query import tensor_sha
from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.native_replay import forbid_integration

EVENTS = ("before_close", "first_query_after_grasp")


def history_ticks(tick):
    require(type(tick) is int and tick >= 4000, "event has no complete 8 second Memory history")
    return [0, *range(tick - 4000, tick + 1, 500)]


def samples(native, events):
    wanted = {t for event in events.values() if event is not None for t in history_ticks(event["tick"])}
    if not wanted:
        return {}
    with (native / "initialization.jsonl").open() as f:
        init = json.loads(next(f))
    require(init["time"] == 0. and init["kind"] == "actual_startup_V1", "native reset missing")
    first = {k: np.asarray(init[k]) for k in ("qpos", "qvel", "ctrl")}
    selected = {0: dict(first, step=0, time_s=0.)}
    previous = 0
    for pkt in packets(native / "state.bin.journal"):
        t = pkt["step"]
        require(t == previous + 1, "event capture requires every earlier native tick")
        if t in wanted:
            selected[t] = pkt
        if t == max(wanted):
            break
        previous = t
    require(set(selected) == wanted, "event has missing archived causal samples")
    return selected


def measured_hand(writer):
    from dense_replay import pose

    m, d = writer.model, writer.data
    base, flange, tool = m.body("x5_base_link").id, m.body("x5_link6").id, m.site("x5_ee").id
    fingers = [m.jnt_qposadr[m.joint(f"task_{side}_slide").id] for side in ("left", "right")]
    return dict(armbase=pose(d.xpos[base], d.xmat[base]), link6=pose(d.xpos[flange], d.xmat[flange]),
                tcp=pose(d.site_xpos[tool], d.site_xmat[tool]), finger_slide_m=d.qpos[fingers].copy())


def capture(case):
    import mujoco
    from OpenGL import GL

    def read(path):
        return json.loads(path.read_bytes())

    require(read(case / "control/LEASE.json")["status"] == "CLOSED", "active physics owner")
    execute = read(case / "NATIVE_REQUEST.json")
    result = read(case / "RESULT.private.json")
    spec = read(case / "EXECUTION_SPEC.json")
    native = case / execute["episode"]
    require(native.name == result["episode"] and result["absent"], "unclosed episode or live children")
    events = read(case / "H0_OBSERVATION_EVENTS.private.json")
    require(set(events) == set(EVENTS), "different registered event roster")
    selected = samples(native, events)
    scene = Path(spec["model_path"])
    digest = spec["frozen_files"][str(scene)]
    pin(scene, digest)
    destination = case / "THREE_VIEW_H0.private"
    destination.mkdir(mode=0o700)
    report = {name: {"status": "EVENT_MISSING"} for name in EVENTS}
    if selected:
        session = FruitCameraSession(scene, digest)
        history = FruitBodyRenderer(scene)
        try:
            renderer = GL.glGetString(GL.GL_RENDERER).decode()
            require(any(x in renderer.lower() for x in ("llvmpipe", "softpipe")), "CPU Mesa required")
            with forbid_integration(mujoco) as calls:
                for name in EVENTS:
                    event = events[name]
                    if event is None:
                        continue
                    tick = event["tick"]
                    packet = selected[tick]
                    require(abs(event["time_s"] - packet["time_s"]) <= 1e-8
                            and event["last_consumed"]["gripper"]["native_step"] == tick,
                            "RGB anchor does not match actual consumption")
                    times = history_ticks(tick)
                    head = [history.render(selected[t], audit_masks=False)[0] for t in times[:-1]]
                    body, wrist, auxiliary, body_wrist_cal = session.capture_three(native.name, packet)
                    require(body.any() and wrist.any(), "missing native body or wrist RGB")
                    head = np.stack([*head, body])
                    state = measured_hand(session.render_copy)
                    np.savez_compressed(destination / (name + ".npz"), head=head, wrist=wrist,
                                        aux=auxiliary.frame.rgb, history_ticks=np.array(times),
                                        **state)
                    report[name] = dict(
                        status="SEALED", tick=tick, time_s=packet["time_s"], history_ticks=times,
                        scene_sha256=digest, current_packet_sha256=native_sample_identity(native.name, packet),
                        history_frame_sha256=[tensor_sha(f) for f in head],
                        wrist_sha256=tensor_sha(wrist), aux_sha256=tensor_sha(auxiliary.frame.rgb),
                        aux_calibration=calibration(), body_wrist_calibration=body_wrist_cal,
                        last_consumed=event["last_consumed"], renderer=renderer,
                        actual_target_kinematics_only=True,
                    )
            require(not calls, "event recorder attempted integration")
        finally:
            try:
                history.close()
            finally:
                session.close()
    (destination / "MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("case", type=Path)
    capture(cli.parse_args().case.resolve())
