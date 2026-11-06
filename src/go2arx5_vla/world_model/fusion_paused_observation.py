"""Capture one fresh observation from an owned, actually paused native world.

This helper is not an inference server, native admission route, or controller.
It preserves the existing shadow binding unchanged inside a new provenance
envelope. Rendering uses the original recorder's measured-state copy only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from scripts.pi0292.relay_protocol import digest as document_hash, sealed, unseal, uuid_string
from scripts.pi0294.task_clock import FROZEN_FIELDS, require_pause
from scripts.pi0294.task_native_worker import FrameRecorder
from scripts.pi0294.task_state import ActualState, parse_state
from scripts.wm02.capture_fusion_observation import software_renderer_identity
from .fusion_observation import TargetShadowBinding, shadow_inference_payload
from .storage import digest, write_json

SCHEMA = "WM02_FUSION_LIVE_PAUSED_OBSERVATION_V2"
TASK_SCHEMA = "WM02_FUSION_PAUSED_TARGET_TASK_V2"
INPUT_MODE = "ACTUAL_OWNED_TARGET_LIVE_PAUSED_CAPTURE"


def _need(condition, reason):
    if not condition:
        raise ValueError("FUSION_PAUSED_OBSERVATION_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _cpu_renderer_environment():
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
          and os.environ.get("MUJOCO_GL") == "egl"
          and os.environ.get("PYOPENGL_PLATFORM") == "egl"
          and os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1"
          and os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES") == "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
          and "MUJOCO_EGL_DEVICE_ID" not in os.environ,
          "EXPLICIT_EXISTING_CPU_EGL_MESA_ENVIRONMENT")


def _task(value, expected_hash, scene_hash):
    keys = {"schema", "task_id", "language", "scene_sha256", "score_contract_sha256",
            "language_provenance"}
    _need(isinstance(value, dict) and set(value) == keys
          and value["schema"] == TASK_SCHEMA and document_hash(value) == expected_hash,
          "PREDECLARED_TASK_DOCUMENT_AND_HASH")
    _need(all(type(value[k]) is str and value[k] for k in ("task_id", "language"))
          and len(value["language"]) <= 4096 and value["scene_sha256"] == scene_hash
          and value["language_provenance"] in ("EXACT_SOURCE_TASK_PROMPT", "NEW_TARGET_SHORT_SKILL_PROMPT")
          and type(value["score_contract_sha256"]) is str
          and len(value["score_contract_sha256"]) == 64
          and all(c in "0123456789abcdef" for c in value["score_contract_sha256"]),
          "SCENE_LANGUAGE_AND_EXTERNAL_SCORE_CONTRACT_BINDING")
    return copy.deepcopy(value)


def _pause(adapter):
    """Read actual adapter feedback and its original pause/quota files together."""
    adapter.guard()
    observed = adapter.observe()
    raw = (Path(adapter.output) / "state.bin").read_bytes()
    packet = parse_state(raw)
    _need(observed["step"] == packet["step"] and observed["time_s"] == packet["time_s"]
          and all(np.array_equal(observed[k], packet[k]) for k in ("qpos", "qvel", "ctrl")),
          "ADAPTER_OBSERVE_AND_CURRENT_PACKET_MUST_AGREE")
    pause = json.loads((Path(adapter.output) / "pause_ack.json").read_text())
    quota = (Path(adapter.output) / "step_quota").read_bytes()
    _need(pause.get("clock_paused") is True and pause.get("running") is False
          and type(pause.get("step")) is int and pause["step"] == packet["step"]
          and int(quota.decode().strip()) == packet["step"]
          and type(pause.get("ack_sequence")) is int and pause["ack_sequence"] >= 0
          and all(k in pause for k in FROZEN_FIELDS)
          and type(pause["time_s"]) in (int, float) and math.isfinite(pause["time_s"])
          and abs(pause["time_s"] - packet["time_s"]) <= 1e-8
          and abs(packet["time_s"] - packet["step"] * .002) <= 1e-8,
          "ORIGINAL_NATIVE_PAUSE_ACK_AND_EXACT_CURRENT_STEP_QUOTA")
    return dict(packet=packet, raw=raw, pause=pause, quota=quota)


def _same_pause(first, last):
    _need(last["raw"] == first["raw"] and last["quota"] == first["quota"],
          "MEASURED_PACKET_OR_QUOTA_CHANGED_WHILE_PAUSED")
    require_pause(first["pause"], last["pause"], first["packet"]["step"])


def _after_pause(adapter, first):
    # The original native pause ACK continues to acknowledge its unchanged
    # physics/FSM/leg history. A caller's `paused=True` is never accepted.
    def fresh():
        value = json.loads((Path(adapter.output) / "pause_ack.json").read_text())
        return value.get("ack_sequence", -1) > first["pause"]["ack_sequence"]
    _need(adapter.spin(1., fresh), "FRESH_ACTUAL_NATIVE_PAUSE_ACK_REQUIRED")
    last = _pause(adapter)
    _same_pause(first, last)
    return last


def _runtime(adapter, binding):
    _need(isinstance(binding, TargetShadowBinding), "ORIGINAL_TARGET_SHADOW_BINDING_REQUIRED")
    root, output = Path(adapter.spec["registered_root"]), Path(adapter.output)
    spec_path = root / "EXECUTION_SPEC.json"
    _need(output.resolve().is_relative_to(root.resolve())
          and json.loads(spec_path.read_text()) == adapter.spec
          and digest(spec_path) == binding.target_runtime_sha256
          and digest(adapter.spec["model_path"]) == binding.scene_sha256,
          "CURRENT_OWNER_SPEC_AND_EXACT_COMPILED_SCENE_BYTES")
    quota_binding = (output / "quota.binding").read_bytes()
    lines = quota_binding.decode().splitlines()
    _need(len(lines) == 7 and lines[:5] == ["GRTQUOTA2", output.name,
          adapter.spec["stage_identity_sha256"], output.name, str(adapter.spec["max_ticks"])],
          "ORIGINAL_RUN_BOUND_NATIVE_QUOTA_IDENTITY")
    loaded = adapter.capture_loaded_identity()
    simulator = loaded.get("sim", {})
    _need(type(simulator.get("pid")) is int and simulator["pid"] > 0
          and simulator.get("executable_sha256") == digest(adapter.spec["native_binary"]),
          "ACTUAL_OWNED_SIMULATOR_EXECUTABLE_IDENTITY")
    _need(isinstance(getattr(adapter, "actual_state", None), ActualState),
          "OWNER_PREPARED_ORIGINAL_ACTUAL_STATE_READER_REQUIRED")
    return dict(registered_root=str(root), native_output=str(output),
                execution_spec_sha256=digest(spec_path), scene_sha256=binding.scene_sha256,
                native_binary_sha256=simulator["executable_sha256"],
                simulator_pid=simulator["pid"], quota_binding_sha256=_sha(quota_binding))


def capture_paused_observation(adapter, *, binding, task, task_sha256, query_uuid, capture_index=0):
    """Retain a current dual-RGB input; never query a model or advance physics.

    The caller supplies its already admitted TargetSequentialAdapter and reader.
    This creates a fresh query-named directory below that adapter's output.
    An existing directory is never reused, including after an incomplete capture.
    Call ``verify_still_paused`` again after inference, before using its result.
    """
    uuid_string(query_uuid)
    _need(type(capture_index) is int and capture_index >= 0, "NONNEGATIVE_CAPTURE_INDEX")
    _cpu_renderer_environment()
    declared_task = _task(task, task_sha256, binding.scene_sha256)
    first = _pause(adapter)
    world = _runtime(adapter, binding)
    parent = Path(adapter.output) / "PAUSED_OBSERVATIONS"
    parent.mkdir(exist_ok=True)
    output = parent / query_uuid
    output.mkdir(exist_ok=False)
    (output / "images").mkdir()
    write_json(output / "CAPTURE_INTENT.private.json", dict(
        schema=SCHEMA, query_uuid=query_uuid, snapshot_uuid=query_uuid,
        input_mode=INPUT_MODE, task_sha256=task_sha256, world=world,
        native_step=first["packet"]["step"], native_time_s=first["packet"]["time_s"],
        native_state_sha256=_sha(first["raw"]), model_queries_requested=0))
    started = time.monotonic()
    recorder = None
    try:
        recorder = FrameRecorder(Path(adapter.spec["model_path"]), output)
        capture = recorder.save(capture_index, first["packet"])
        _need(capture == dict(index=capture_index, native_step=first["packet"]["step"],
                             time_s=first["packet"]["time_s"]), "ORIGINAL_RECORDER_SAME_PACKET")
        renderer = software_renderer_identity()
        _need("mesa" in renderer.get("GL_VENDOR", "").lower()
              and any(name in renderer.get("GL_RENDERER", "").lower() for name in
                      ("llvmpipe", "softpipe", "software rasterizer")),
              "ACTUAL_POST_SAVE_CPU_SOFTWARE_RENDERER")
        state_path = output / "STATE.private.bin"
        with state_path.open("xb") as stream:
            stream.write(first["raw"])
            stream.flush()
            os.fsync(stream.fileno())
        calibration = output / "images" / f"{capture_index:06d}-calibration.json"
        images = {camera: output / "images" / f"{capture_index:06d}-{camera}.png"
                  for camera in ("body_rgb", "wrist_rgb")}
        shadow = binding.bind(actual_state=adapter.actual_state,
            native_state_bytes=state_path.read_bytes(), expected_state_sha256=_sha(first["raw"]),
            capture=capture, calibration_bytes=calibration.read_bytes(),
            expected_calibration_sha256=digest(calibration),
            image_bytes={k: p.read_bytes() for k, p in images.items()},
            expected_image_sha256={k: digest(p) for k, p in images.items()},
            language=declared_task["language"], snapshot_uuid=query_uuid)
        shadow_inference_payload(shadow)  # Require an explicit mapped policy view.
        write_json(output / "BOUND_SHADOW_ABI.private.json", shadow)
        recorder.close()
        recorder = None
        last = _after_pause(adapter, first)
        _need(digest(adapter.spec["model_path"]) == world["scene_sha256"]
              and digest(Path(adapter.spec["registered_root"]) / "EXECUTION_SPEC.json") == world["execution_spec_sha256"]
              and _sha((Path(adapter.output) / "quota.binding").read_bytes()) == world["quota_binding_sha256"],
              "SCENE_RUNTIME_OR_QUOTA_BINDING_CHANGED_DURING_CAPTURE")
        files = {p.relative_to(output).as_posix(): digest(p) for p in
                 [state_path, calibration, *images.values(), output / "BOUND_SHADOW_ABI.private.json"]}
        record = sealed(dict(schema=SCHEMA, input_mode=INPUT_MODE,
            query_uuid=query_uuid, snapshot_uuid=query_uuid, capture=capture,
            native_step=first["packet"]["step"], native_time_s=first["packet"]["time_s"],
            task=declared_task, task_sha256=task_sha256, world=world,
            renderer=renderer,
            pause_before=first["pause"], pause_after=last["pause"],
            step_quota_sha256=_sha(first["quota"]), files=files,
            original_shadow_ABI=shadow,
            provenance=dict(actual_current_owned_world=True, simulation_paused_during_capture=True,
                native_clock_rebased=False, task_language_bound_to_declared_scene=True,
                camera_equivalence=False, target_task_success=False,
                native_execution_authorized=False, source_outcomes_supplied=False,
                VLA_inferences=0, physics_integrations=0,
                first_draw_replay_is_sequential_feedback=False,
                online_closed_loop="REQUIRES_EACH_NEW_BRANCH_OBSERVATION_QUERY_AND_ACTUAL_PREFIX_ACK"),
            capture_wall_s=time.monotonic() - started))
        write_json(output / "PAUSED_OBSERVATION.private.json", record)
        return record
    except BaseException as error:
        write_json(output / "CAPTURE_FAILED.private.json", dict(
            schema=SCHEMA, query_uuid=query_uuid, failure=type(error).__name__ + ": " + str(error),
            retry_authorized=False, partial_capture_retained=True))
        raise
    finally:
        if recorder is not None:
            recorder.close()


def paused_inference_payload(record):
    """Read an observation ABI while retaining the V2 provenance alongside it."""
    body = unseal(record)
    _need(body.get("schema") == SCHEMA and body.get("input_mode") == INPUT_MODE,
          "VERSIONED_LIVE_PAUSED_ENVELOPE")
    task = _task(body["task"], body["task_sha256"], body["world"]["scene_sha256"])
    uuid_string(body["query_uuid"])
    shadow = unseal(body["original_shadow_ABI"])
    payload = shadow_inference_payload(body["original_shadow_ABI"])
    target = shadow["target_observation"]
    index = body["capture"]["index"]
    _need(body["snapshot_uuid"] == body["query_uuid"] == payload["snapshot_uuid"]
          and body["capture"] == shadow["capture"]
          and body["native_step"] == shadow["native_step"] == body["capture"]["native_step"]
          and body["native_time_s"] == shadow["native_time_s"] == body["capture"]["time_s"]
          and payload["language"] == task["language"]
          and shadow["target_observation"]["scene_sha256"] == body["world"]["scene_sha256"]
          and shadow["target_observation"]["target_runtime_sha256"] == body["world"]["execution_spec_sha256"],
          "SAME_QUERY_TASK_SCENE_CLOCK_AND_UNCHANGED_SHADOW_ABI")
    _need(body["files"]["STATE.private.bin"] == target["native_state_sha256"]
          and body["files"][f"images/{index:06d}-calibration.json"] == target["calibration_sha256"]
          and all(body["files"][f"images/{index:06d}-{camera}.png"] == target["image_sha256"][camera]
                  for camera in ("body_rgb", "wrist_rgb")), "ORIGINAL_STATE_CALIBRATION_AND_IMAGE_FILE_BINDINGS")
    require_pause(body["pause_before"], body["pause_after"], body["native_step"])
    _need(body["provenance"]["native_execution_authorized"] is False
          and body["provenance"]["VLA_inferences"] == 0
          and body["provenance"]["physics_integrations"] == 0,
          "CAPTURE_IS_NOT_AN_INFERENCE_OR_EXECUTION_GRANT")
    return payload


def verify_still_paused(adapter, record):
    """Call after a separate inference; reject any native advance or new world."""
    paused_inference_payload(record)
    body = unseal(record)
    current = _pause(adapter)
    world = body["world"]
    _need(str(Path(adapter.output)) == world["native_output"]
          and digest(Path(adapter.spec["registered_root"]) / "EXECUTION_SPEC.json") == world["execution_spec_sha256"]
          and digest(adapter.spec["model_path"]) == world["scene_sha256"]
          and _sha((Path(adapter.output) / "quota.binding").read_bytes()) == world["quota_binding_sha256"]
          and _sha(current["raw"]) == body["files"]["STATE.private.bin"]
          and _sha(current["quota"]) == body["step_quota_sha256"], "QUERY_RETURN_MUST_MATCH_CURRENT_WORLD_AND_PACKET")
    original = dict(current, pause=body["pause_after"])
    last = _after_pause(adapter, original)
    loaded = adapter.capture_loaded_identity()
    _need(loaded["sim"]["pid"] == world["simulator_pid"]
          and loaded["sim"]["executable_sha256"] == world["native_binary_sha256"],
          "SAME_ACTUAL_OWNED_SIMULATOR_AFTER_QUERY")
    return dict(schema=SCHEMA, query_uuid=body["query_uuid"], native_step=body["native_step"],
                native_time_s=body["native_time_s"], pause_after_query=last["pause"],
                unchanged_measured_packet_sha256=_sha(last["raw"]),
                native_execution_authorized=False, model_result_checked=False)
