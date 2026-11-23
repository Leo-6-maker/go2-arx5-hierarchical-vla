"""Run one preregistered contact-free card inside the fresh owner's critical section."""

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from clock_guard import new_receiver_output
from go2arx5_vla.world_model.fusion_trajectory_adapter import (
    MECHANICAL_TRAVEL_M,
    WORKING_OPEN_M,
)
from go2arx5_vla.world_model.init_trace_reader import audit_trace
from go2arx5_vla.world_model.native_live_control_v3 import live_path
from go2arx5_vla.world_model.native_replay import packets
from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
from identity import read, sha
from scripts.wm02 import mobile_boundary_runner_v1 as mobile

from scripts.pi0291_arx5_data.native_dataset import pose10
from scripts.pi0292.durable import write
from scripts.pi0292.eval_contract import matrix10


class TechnicalAdapter(TargetSequentialAdapter):
    def launch(self, argv, name, env):
        if name == "planner":
            original = (
                Path(self.spec["source_root"]) / "scripts/pi0294/native_receiver.py"
            )
            pin = self.spec["source_files"]["scripts/pi0294/native_receiver.py"]
            wrapper = Path(__file__).with_name("b2_receiver.py")
            if (
                Path(argv[2]) != original
                or sha(original) != pin
                or sha(wrapper) != self.spec["b2_code"][wrapper.name]
            ):
                raise ValueError("receiver launch mismatch")
            argv = [*argv[:2], str(wrapper), *argv[3:]]
            env = dict(env, B2_CARD=self.spec["b2_card"], B2_RECEIVER_PIN=pin)
        super().launch(argv, name, env)


def fixed(sample, mount):
    row = pose10(sample["tcp"], WORKING_OPEN_M / MECHANICAL_TRAVEL_M)
    return np.repeat(row[None], 40, axis=0), np.repeat(mount[None], 41, axis=0)


def paused(adapter, tick):
    ack = read(live_path(adapter, "pause_ack.json"))
    if (
        ack["step"] != tick
        or ack["running"] is not False
        or ack["clock_paused"] is not True
        or adapter.observe()["step"] != tick
    ):
        raise ValueError("expected a confirmed native pause")
    return ack


def episode(adapter, card, result):
    out = adapter.output
    adapter.prepare_dynamic()
    ready = read(out / "GUARD_READY.json")
    if ready != {
        "episode": out.name,
        "pid": next(p.pid for name, p in adapter.processes if name == "planner"),
    }:
        raise ValueError("unmatched live receiver")
    mount = mobile.arm(adapter)
    sample = adapter.measured_task_state()
    action, base = fixed(sample, mount)
    for i in range(128):
        if i % 7 == 0:
            mobile.install(adapter, action, base)
        sample, mount = mobile.consume(adapter, action, sequence=i)
    paused(adapter, 5703)
    action, base = fixed(sample, mount)
    atoms = 20 if card == "T0_STATIC_HOLD" else 39
    for i in range(atoms):
        if i % 2 == 0:
            mobile.install(adapter, action, base)
        sample, _ = mobile.consume(adapter, action, sequence=128 + i)
        if not sample["safety_valid"] or sample["native_grasp"]:
            raise RuntimeError("technical card contact/safety qualification failed")
    tick = 5703 + 25 * atoms
    paused(adapter, tick)
    if (out / "GUARD_TERMINAL.json").exists():
        raise RuntimeError("unrequested receiver clock fault")
    if card == "T0_STATIC_HOLD":
        result["cancel"] = adapter.cancel_reference()
    else:
        count = adapter.counts.get("mode", 0)
        result["fault_boundary_tick"] = tick
        write(out / "B2_REWIND_REQUEST.json", {"episode": out.name, "step": tick})

        def acknowledged():
            return (
                (out / "GUARD_TERMINAL.json").exists()
                and adapter.counts.get("mode", 0) > count
                and adapter.latest["mode"].data == "HOLD"
            )

        if not adapter.spin(10, acknowledged):
            raise RuntimeError(
                "clock fault receipt or independent HOLD acknowledgement missing"
            )
        receipt = read(out / "GUARD_TERMINAL.json")
        if (
            receipt["episode"] != out.name
            or receipt["fault"]["reason"] != "rewind"
            or receipt["fault"]["stream"] != "mpc"
            or not receipt["reference_invalidated"]
        ):
            raise RuntimeError("incorrect clock fault receipt")
        result.update(
            receiver_fault=receipt["fault"],
            hold_ack_tick=adapter.observe()["step"],
            hold_request_published=receipt["hold_request_published"],
        )
    result["pause_step_before_seal"] = paused(adapter, tick)["step"]
    adapter.seal_trace(tick)
    if card == "T1_CLOCK_FAULT_STOP":
        adapter.spin(0.25)
        result["new_target_publications_after_fault"] = sum(
            r["kind"] == "installed" and r["monotonic_s"] >= receipt["monotonic_s"]
            for r in adapter.reference_events()
        )
        result["confirmed_terminal_native_tick"] = paused(adapter, tick)["step"]
        if result["new_target_publications_after_fault"]:
            raise RuntimeError("target publication after terminal fault")


def samples(path):
    if not path.exists():
        return {}
    with path.open() as f:
        rows = [r for s in f if (r := json.loads(s))["native_start_step"] >= 5703]
    if not rows:
        return {}
    pos, ang = [], []
    for row in rows:
        target = matrix10(row["target_endpoint_EEF10"])
        tcp = np.array(row["actual_TCP"])
        pos.append(1000 * np.linalg.norm(target[:3, 3] - tcp[:3, 3]))
        cos_angle = (np.trace(target[:3, :3].T @ tcp[:3, :3]) - 1) / 2
        ang.append(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
    return {
        "atoms": len(rows),
        "position_rms_mm": float(np.sqrt(np.mean(np.square(pos)))),
        "position_peak_mm": float(max(pos)),
        "rotation_rms_deg": float(np.sqrt(np.mean(np.square(ang)))),
        "rotation_peak_deg": float(max(ang)),
        "distinct_installs": len({r["installed_reference_sha256"] for r in rows}),
        "atoms_with_exact_solver_echo": sum(
            bool(r["solver_exact_installed_targets"]) for r in rows
        ),
        "arm_requests": sum(r["arm_requests"] for r in rows),
        "base_commands": sum(r["base_commands"] for r in rows),
        "valid_no_grasp_atoms": sum(
            bool(r["safety_valid"]) and not r["native_grasp"] for r in rows
        ),
    }


def run_card(spec, spec_path):
    root = Path(spec["registered_root"])
    mobile.registered(spec, spec_path)
    out = new_receiver_output(root)
    adapter = TechnicalAdapter(spec, out)
    card = spec["b2_card"]
    value = {
        "card": card,
        "native_requests": 1,
        "cap": spec["max_ticks"],
        "completed": False,
        "episode": out.name,
    }
    write(
        root / "NATIVE_REQUEST.json",
        {"pid": os.getpid(), "epoch": time.time(), "episode": out.name},
    )
    try:
        adapter.start()
        episode(adapter, card, value)
    except BaseException:
        value["failure"] = traceback.format_exc()
    finally:
        if adapter.node is not None:
            try:
                if adapter.processes and all(
                    p.poll() is None for _, p in adapter.processes
                ):
                    adapter.capture_loaded_identity()
                if not adapter.trace_sealed:
                    adapter.seal_trace(adapter.observe()["step"])
            except BaseException:
                value["evidence_error"] = traceback.format_exc()
        try:
            adapter.close()
        except BaseException:
            value["close_error"] = traceback.format_exc()
    journal = out / "state.bin.journal"
    integrations = (
        len(packets(journal)) if journal.exists() and journal.stat().st_size else 0
    )
    value.update(
        journal_integrations=integrations,
        owned_exits=adapter.exit_records,
        sealed=adapter.trace_sealed,
        measured=samples(out / "MOBILE_SAMPLES.private.jsonl"),
    )
    try:
        trace = audit_trace(out / "init.trace", max_native_ticks=spec["max_ticks"])
        value["native_trace"] = {
            k: trace[k]
            for k in (
                "status",
                "native_step_records",
                "STOP_observed",
                "integrations_after_first_STOP",
                "errors",
            )
        }
    except BaseException:
        value["trace_error"] = traceback.format_exc()
    value["mobile_guard_stop"] = (out / "MOBILE_SAFETY_STOP.private.json").exists()
    physics = out / "physics.jsonl"
    value["native_safety_stops"] = (
        sum(json.loads(line)["kind"] == "stop" for line in physics.open())
        if physics.exists()
        else None
    )
    if "fault_boundary_tick" in value:
        value["integrations_after_fault_boundary"] = (
            integrations - value["fault_boundary_tick"]
        )
    expected = 6203 if card == "T0_STATIC_HOLD" else 6678
    m = value["measured"]
    value["completed"] = bool(
        not any(key.endswith("error") or key == "failure" for key in value)
        and value["sealed"]
        and integrations == expected == value["native_trace"]["native_step_records"]
        and value["native_trace"]["status"] == "RECORDED_TRACE_CHECKS_PASS"
        and not value["mobile_guard_stop"]
        and value["native_safety_stops"] == 0
        and len(adapter.exit_records) == 4
        and all(r["exit_code"] == 0 for r in adapter.exit_records)
        and m.get("atoms")
        == m.get("atoms_with_exact_solver_echo")
        == m.get("valid_no_grasp_atoms")
        == ((expected - 5703) // 25)
        and m["arm_requests"] > 0
        and m["base_commands"] > 0
        and (
            card == "T0_STATIC_HOLD"
            or value.get("hold_ack_tick")
            == value.get("confirmed_terminal_native_tick")
            == expected
        )
    )
    write(root / "RESULT.private.json", value)
    return value
