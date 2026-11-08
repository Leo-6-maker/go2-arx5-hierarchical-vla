"""Synthetic native/capture seam doubles; no renderer, model, GPU or physics."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pi0292.relay_protocol import digest as document_hash, sealed, unseal
from scripts.pi0294.task_state import ActualState, parse_state
from go2arx5_vla.world_model import fusion_paused_observation as p
from go2arx5_vla.world_model.fusion_observation import TargetShadowBinding
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_fusion_observation import fixture as observation_fixture, png

QUERY = "12345678-1234-4234-8234-123456789abc"


def write(path, value):
    path.write_text(json.dumps(value))


def capture_fixture(tmp_path, monkeypatch, *, branch="branch", color=20):
    for key, value in dict(CUDA_VISIBLE_DEVICES="", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                          LIBGL_ALWAYS_SOFTWARE="1",
                          __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json").items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    old_binding, args, measured, transform = observation_fixture()
    root = tmp_path / branch
    output = root / "native-01"
    output.mkdir(parents=True)
    scene, native = root / "scene.mjb", root / "native-binary"
    scene.write_bytes(b"not a MuJoCo model: synthetic seam fixture")
    native.write_bytes(b"not an executable: synthetic seam fixture")
    spec = dict(registered_root=str(root), model_path=str(scene), native_binary=str(native),
                stage_identity_sha256="a" * 64, max_ticks=5003)
    write(root / "EXECUTION_SPEC.json", spec)
    (output / "state.bin").write_bytes(args["native_state_bytes"])
    pause = dict(step=3703, time_s=3703 * .002, running=False, clock_paused=True,
                 ack_sequence=1, motiontime=3900, leg_history_steps=120, gait_indices=.3)
    write(output / "pause_ack.json", pause)
    (output / "step_quota").write_text("3703\n")
    (output / "quota.binding").write_text("\n".join(("GRTQUOTA2", output.name,
        spec["stage_identity_sha256"], output.name, "5003", "5", "32")) + "\n")
    binding = TargetShadowBinding("ACTUAL_TARGET_WORLD", digest(scene),
        digest(root / "EXECUTION_SPEC.json"), transform, old_binding.expected_transform_sha256)
    task = dict(schema=p.TASK_SCHEMA, task_id="RED_CUBE_SMALL_APPROACH_V2", scene_sha256=digest(scene),
        language="Move the open gripper a little toward the red cube, then hold without touching it.",
        score_contract_sha256="b" * 64, language_provenance="NEW_TARGET_SHORT_SKILL_PROMPT")
    events = SimpleNamespace(saves=[], closes=0, observe_calls=0, load_calls=0,
                             during_save=None, pid=12345, fresh_ack=True,
                             renderer=dict(GL_VENDOR="Mesa", GL_RENDERER="llvmpipe SYNTHETIC FIXTURE", backend="egl_mesa_cpu"))
    class Adapter:
        def guard(self): pass
        def observe(self):
            events.observe_calls += 1
            return parse_state((self.output / "state.bin").read_bytes())
        def capture_loaded_identity(self):
            events.load_calls += 1
            return dict(sim=dict(pid=events.pid, executable_sha256=digest(native)))
        def spin(self, seconds, predicate):
            if events.fresh_ack:
                current = json.loads((self.output / "pause_ack.json").read_text())
                current["ack_sequence"] += 1
                write(self.output / "pause_ack.json", current)
            return predicate()
    adapter = Adapter()
    adapter.output, adapter.spec, adapter.actual_state = output, spec, args["actual_state"]
    class Recorder:
        def __init__(self, model_path, folder):
            assert model_path == scene
            self.folder = folder
        def save(self, index, packet):
            events.saves.append((index, packet["step"]))
            if events.during_save:
                events.during_save(adapter)
            for name, raw in dict(body_rgb=png(color), wrist_rgb=png(color + 40)).items():
                (self.folder / "images" / f"{index:06d}-{name}.png").write_bytes(raw)
            (self.folder / "images" / f"{index:06d}-calibration.json").write_bytes(args["calibration_bytes"])
            return dict(index=index, native_step=packet["step"], time_s=packet["time_s"])
        def close(self): events.closes += 1
    monkeypatch.setattr(p, "FrameRecorder", Recorder)
    def renderer_identity():
        assert events.saves  # The identity check follows an actual save call.
        return events.renderer
    monkeypatch.setattr(p, "software_renderer_identity", renderer_identity)
    monkeypatch.setattr(ActualState, "update", lambda self, packet: dict(measured,
        native_step=packet["step"], time_s=packet["time_s"]))
    return adapter, binding, task, events


def capture(adapter, binding, task, **kwargs):
    return p.capture_paused_observation(adapter, binding=binding, task=task,
        task_sha256=document_hash(task), query_uuid=QUERY, **kwargs)


def test_fresh_capture_binds_actual_pause_task_raw_bytes_and_unchanged_shadow_abi(tmp_path, monkeypatch):
    adapter, binding, task, events = capture_fixture(tmp_path, monkeypatch)
    raw = (adapter.output / "state.bin").read_bytes()
    record = capture(adapter, binding, task, capture_index=3)
    body = unseal(record)
    shadow = unseal(body["original_shadow_ABI"])
    payload = p.paused_inference_payload(record)
    assert body["schema"] == p.SCHEMA and body["input_mode"] == p.INPUT_MODE
    assert body["native_step"] == 3703 and body["native_time_s"] == 3703 * .002
    assert payload["language"] == task["language"] and payload["snapshot_uuid"] == QUERY
    assert shadow["provenance"]["shadow_only"] is True
    assert shadow["provenance"]["online_closed_loop"] is False
    assert body["provenance"]["actual_current_owned_world"] is True
    assert body["renderer"] == events.renderer
    assert body["provenance"]["native_execution_authorized"] is False
    assert body["provenance"]["first_draw_replay_is_sequential_feedback"] is False
    assert body["pause_after"]["ack_sequence"] > body["pause_before"]["ack_sequence"]
    assert events.saves == [(3, 3703)] and events.closes == 1 and events.observe_calls >= 2
    folder = adapter.output / "PAUSED_OBSERVATIONS" / QUERY
    assert (folder / "STATE.private.bin").read_bytes() == raw
    assert (adapter.output / "state.bin").read_bytes() == raw
    assert all(digest(folder / name) == pin for name, pin in body["files"].items())
    assert "PRIVILEGED" not in json.dumps(record)
    assert set(payload) == {"snapshot_uuid", "language", "state_eef10", "images", "frame", "tcp"}
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("fault", ["running", "unpaused", "ack_step", "quota", "quota_identity", "clock", "reader"])
def test_caller_pause_flag_cannot_replace_actual_native_barrier(tmp_path, monkeypatch, fault):
    adapter, binding, task, events = capture_fixture(tmp_path, monkeypatch)
    ack = json.loads((adapter.output / "pause_ack.json").read_text())
    if fault == "running": ack["running"] = True
    elif fault == "unpaused": ack["clock_paused"] = False
    elif fault == "ack_step": ack["step"] -= 25
    elif fault == "clock": ack["time_s"] = 0.
    elif fault == "quota": (adapter.output / "step_quota").write_text("3728\n")
    elif fault == "quota_identity": (adapter.output / "quota.binding").write_text("caller says paused")
    elif fault == "reader": adapter.actual_state = object()
    write(adapter.output / "pause_ack.json", ack)
    with pytest.raises(ValueError): capture(adapter, binding, task)
    assert not events.saves


@pytest.mark.parametrize("fault", ["step", "same_step_state", "quota", "leg_history", "no_new_ack", "scene_bytes"])
def test_native_advance_or_changed_hidden_clock_during_render_is_rejected(tmp_path, monkeypatch, fault):
    adapter, binding, task, events = capture_fixture(tmp_path, monkeypatch)
    def mutate(a):
        if fault in ("step", "same_step_state"):
            raw = bytearray((a.output / "state.bin").read_bytes())
            if fault == "step": struct.pack_into("<Qd", raw, 8, 3728, 3728 * .002)
            else: struct.pack_into("<d", raw, 36, .001)
            (a.output / "state.bin").write_bytes(raw)
        elif fault == "quota": (a.output / "step_quota").write_text("3728\n")
        elif fault == "leg_history":
            ack = json.loads((a.output / "pause_ack.json").read_text())
            ack["leg_history_steps"] += 1
            write(a.output / "pause_ack.json", ack)
        elif fault == "scene_bytes": Path(a.spec["model_path"]).write_bytes(b"changed scene during render")
        else: events.fresh_ack = False
    events.during_save = mutate
    with pytest.raises(ValueError): capture(adapter, binding, task)
    folder = adapter.output / "PAUSED_OBSERVATIONS" / QUERY
    assert (folder / "CAPTURE_FAILED.private.json").is_file()
    assert not (folder / "PAUSED_OBSERVATION.private.json").exists()
    assert events.closes == 1
    with pytest.raises((FileExistsError, ValueError)): capture(adapter, binding, task)
    assert len(events.saves) == 1


def test_after_inference_barrier_rechecks_actual_world_packet_and_live_ack(tmp_path, monkeypatch):
    adapter, binding, task, events = capture_fixture(tmp_path, monkeypatch)
    record = capture(adapter, binding, task)
    checked = p.verify_still_paused(adapter, record)
    assert checked["native_step"] == 3703 and checked["model_result_checked"] is False
    assert not checked["native_execution_authorized"]
    events.pid += 1
    with pytest.raises(ValueError, match="SAME_ACTUAL_OWNED_SIMULATOR"):
        p.verify_still_paused(adapter, record)
    events.pid -= 1
    raw = bytearray((adapter.output / "state.bin").read_bytes())
    struct.pack_into("<d", raw, 36, .005)
    (adapter.output / "state.bin").write_bytes(raw)
    with pytest.raises(ValueError, match="QUERY_RETURN"):
        p.verify_still_paused(adapter, record)


@pytest.mark.parametrize("fault", ["language", "task_scene", "task_hash", "outer_step", "files", "grant"])
def test_complete_task_clock_and_nested_abi_sidecar_survives_inference_transport(tmp_path, monkeypatch, fault):
    adapter, binding, task, events = capture_fixture(tmp_path, monkeypatch)
    record = capture(adapter, binding, task)
    body = unseal(record)
    if fault == "language": body["task"]["language"] = "A different task"
    elif fault == "task_scene": body["task"]["scene_sha256"] = "0" * 64
    elif fault == "task_hash": body["task_sha256"] = "0" * 64
    elif fault == "outer_step": body["native_step"] += 25
    elif fault == "files": body["files"]["STATE.private.bin"] = "0" * 64
    else: body["provenance"]["native_execution_authorized"] = True
    with pytest.raises(ValueError): p.paused_inference_payload(sealed(body))
    assert events.saves == [(0, 3703)]


def test_duplicate_query_is_not_rerendered_and_each_branch_uses_own_pixels(tmp_path, monkeypatch):
    first, binding, task, events = capture_fixture(tmp_path, monkeypatch, branch="direct", color=20)
    record1 = capture(first, binding, task)
    with pytest.raises(FileExistsError): capture(first, binding, task)
    assert events.saves == [(0, 3703)]
    second, binding, task, events2 = capture_fixture(tmp_path, monkeypatch, branch="corrected", color=90)
    record2 = p.capture_paused_observation(second, binding=binding, task=task,
        task_sha256=document_hash(task), query_uuid="12345678-1234-4234-8234-123456789abd")
    assert p.paused_inference_payload(record1)["images"][0]["sha256"] != p.paused_inference_payload(record2)["images"][0]["sha256"]
    assert unseal(record1)["world"]["registered_root"] != unseal(record2)["world"]["registered_root"]
    assert events2.saves == [(0, 3703)]


def test_module_import_does_not_load_neural_ros_or_physics_runtime():
    source = ("import sys; from go2arx5_vla.world_model import fusion_paused_observation; "
              "assert not any(x in sys.modules for x in ['torch','gr00t','mujoco','rclpy','transformers'])")
    completed = subprocess.run([sys.executable, "-B", "-c", source], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("key,value", [("CUDA_VISIBLE_DEVICES", "0"), ("CUDA_VISIBLE_DEVICES", None),
    ("MUJOCO_GL", "osmesa"), ("PYOPENGL_PLATFORM", "glx"), ("LIBGL_ALWAYS_SOFTWARE", "0"),
    ("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"),
    ("MUJOCO_EGL_DEVICE_ID", "0")])
def test_cpu_renderer_environment_is_required_before_constructing_recorder(tmp_path, monkeypatch, key, value):
    adapter, binding, task, events = capture_fixture(tmp_path, monkeypatch)
    if value is None: monkeypatch.delenv(key)
    else: monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="CPU_EGL_MESA_ENVIRONMENT"):
        capture(adapter, binding, task)
    assert not events.saves and events.load_calls == 0


@pytest.mark.parametrize("renderer", [dict(GL_VENDOR="NVIDIA", GL_RENDERER="NVIDIA RTX"),
    dict(GL_VENDOR="Mesa", GL_RENDERER="NVIDIA RTX"), dict(GL_VENDOR="NVIDIA", GL_RENDERER="llvmpipe"), {}])
def test_actual_post_save_renderer_must_be_mesa_software(tmp_path, monkeypatch, renderer):
    adapter, binding, task, events = capture_fixture(tmp_path, monkeypatch)
    events.renderer = renderer
    with pytest.raises(ValueError, match="ACTUAL_POST_SAVE_CPU_SOFTWARE_RENDERER"):
        capture(adapter, binding, task)
    assert events.saves == [(0, 3703)] and events.closes == 1
    assert not (adapter.output / "PAUSED_OBSERVATIONS" / QUERY / "PAUSED_OBSERVATION.private.json").exists()
