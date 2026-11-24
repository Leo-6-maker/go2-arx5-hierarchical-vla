"""Audit native compact H0 results, measured contacts and three-view custody."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from sea_fruit_h0_cold import pin, read, require, sha
from sea_fruit_h0_capture import EVENTS, history_ticks, samples
from sea_fruit_h0_native import verify_fixed_base
from sea_fruit_h0_owner import COMPACT_CAMPAIGN, CAP
from sea_fruit_h0_score import StationaryHoldScore
from sea_stance_attribution import DT, trace_window
from sea_three_view import native_sample_identity
from d1_factorial_reference_matrix import held_part
from policy_query import tensor_sha
from go2arx5_vla.world_model.init_trace_reader import audit_trace
from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.native_replay import forbid_integration


def published_reference(native, end):
    with (native / "planner.jsonl").open() as f:
        rows = list(map(json.loads, f))
    accepted = {r["ref_id"]: json.loads(r["raw"]) for r in rows if r["kind"] == "accept"}
    if not accepted:
        return 0
    actual = [e for e in rows if e["kind"] == "installed" and e.get("ref_id") in accepted]
    fixed = held_part(next(iter(accepted.values())), "base")
    verify_fixed_base(actual, fixed)
    cancel = next((i for i, r in enumerate(rows) if r.get("reason") == "CANCEL"), len(rows))
    require(all(r["_pi0294"]["source_mpc_s"] < end*DT for r in accepted.values())
            and all(r["bound_start"] < end*DT for r in actual)
            and not any(r["kind"] in ("accept", "installed") for r in rows[cancel:]),
            "task reference after the final pause")
    return len(actual)


def capture_evidence(case, native, events):
    folder = case / "THREE_VIEW_H0.private"
    manifest = folder / "MANIFEST.json"
    if not manifest.is_file():
        failure = case / "CAPTURE_FAILURE.private.json"
        if failure.is_file():
            return dict(status="TECHNICAL_FAILURE", evidence_sha256=sha(failure))
        return {}
    rows = read(manifest)
    require(set(rows) == set(events) == set(EVENTS), "unexpected three-camera roster")
    selected = samples(native, events)
    for name, obs in rows.items():
        event = events[name]
        if obs["status"] != "SEALED":
            require(event is None and obs["status"] == "EVENT_MISSING", "unexpected missing anchor")
            continue
        tick = event["tick"]
        require(obs["tick"] == tick and obs["time_s"] == event["time_s"]
                and obs["last_consumed"] == event["last_consumed"]
                and obs["history_ticks"] == history_ticks(tick)
                and obs["current_packet_sha256"] == native_sample_identity(native.name, selected[tick]),
                "three-camera anchor differs from the closed episode")
        artifact = folder / (name + ".npz")
        with np.load(artifact, allow_pickle=False) as state:
            body, wrist, aux = (state[k] for k in ("head", "wrist", "aux"))
            fingers = state["finger_slide_m"]
            require(body.shape == (10, 256, 256, 3) and wrist.shape == (256, 256, 3)
                    and aux.shape == (480, 640, 3) and state["history_ticks"].tolist() == obs["history_ticks"]
                    and [tensor_sha(f) for f in body] == obs["history_frame_sha256"]
                    and tensor_sha(wrist) == obs["wrist_sha256"] and tensor_sha(aux) == obs["aux_sha256"],
                    "three-camera bytes differ from their actual custody manifest")
            require(fingers.shape == (2,) and np.array_equal(
                fingers, obs["last_consumed"]["gripper"]["two_measured_finger_slide_m"]
            ), "finger observation differs from real native ACK")
            obs["measured_go2_inner_pad_gap_mm"] = 1.048 + 1000*float(fingers.sum())
        obs["artifact_sha256"] = sha(artifact)
    return dict(manifest_sha256=sha(manifest), events=rows)


def score_archive(native, case, spec, result, end):
    import mujoco
    import yaml

    journal = native / "state.bin.journal"
    score = result.get("score")
    path = Path(spec["model_path"])
    entry = result.get("entry_tick")
    if score is None:
        require(sum(1 for _ in packets(journal)) == end, "technical journal count")
        return None
    model = mujoco.MjModel.from_binary_path(str(pin(path, spec["frozen_files"][str(path)])))
    replay = StationaryHoldScore(model, spec["multitask_geometry"])
    coordinates = {}
    with forbid_integration(mujoco) as calls:
        for i, packet in enumerate(packets(journal), 1):
            require(packet["step"] == i, "native journal went backward or missed a tick")
            if i < entry:
                continue
            replay.update_packet(packet)
            if i in (entry, end):
                coordinates[i] = replay.state.data.site("x5_ee").xpos.copy()
    require(not calls and i == end and replay.last == end and replay.result() == score, "independent HOLD score differs")
    cfg, = [Path(p) for p in spec["frozen_files"] if Path(p).name == "config.yaml"]
    cfg = next(iter(yaml.safe_load(pin(cfg, spec["frozen_files"][str(cfg)]).read_text()).values()))
    steps = [r for r in trace_window(native / "init.trace", entry, end) if r[3] == 3]
    scale = np.array(cfg["dog_commands_scale"][:3])
    with (case / "HOLD_CONSUMPTION.private.jsonl").open() as f:
        atoms = [json.loads(x) for x in f]
    return dict(score=score, actual_TCP_net_displacement_mm=float(1000*np.linalg.norm(coordinates[end]-coordinates[entry])),
                consumed_50ms_atoms=len(atoms), last_stage=atoms[-1]["stage"] if atoms else None,
                gait_cycles=sum(r[10+45] > 0 for r in steps),
                maximum_compiled_base_command_norm=max(
                    (float(np.linalg.norm(np.asarray(r[49:52])/scale)) for r in steps), default=None))


def row(case):
    result = read(case / "RESULT.private.json")
    verdict = read(case / "AUDIT.json")
    lease = read(case / "control/LEASE.json")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    native = case / result["episode"]
    require(lease["status"] == "CLOSED" and not result["survivors"], "run owner is still active")
    identity = native / "LOADED_EXECUTOR_00.private.json"
    pids = {item["pid"] for item in result["exits"]}
    if identity.is_file():
        loaded = read(identity)
        pids.update(item["pid"] for item in loaded.values())
        for role, name in (("sim", "native_binary"), ("solver", "solver_binary"),
                           ("controller", "controller_binary")):
            require(loaded[role]["executable_sha256"] == spec["frozen_files"][spec[name]],
                    "loaded native executable differs from its frozen pin")
    require(all(not Path("/proc", str(p)).exists() for p in pids), "native processes not closed")
    trace = native / "init.trace"
    end = verdict["ticks"]
    proof = audit_trace(trace, max_native_ticks=CAP, state_abi=(41, 38, 20)) if trace.is_file() else None
    world = None
    if proof is not None:
        require(proof["native_step_records"] == end and proof["writer"]["dropped"] == 0
                and proof["integrations_after_first_STOP"] in (None, 0), "terminal native accounting differs")
        world = score_archive(native, case, spec, result, end)
    pause = None
    if result.get("cancel") is not None:
        pause = read(native / "pause_ack.json")
        require(result["cancel"]["native_step"] == pause["step"] == end and result["cancel"]["mode"] == "HOLD"
                and pause["clock_paused"] and not pause["running"]
                and proof["writer"]["sealed_at_explicit_pause"], "terminal pause not closed")
    efile = native / "planner.jsonl"
    error = (result.get("interrupt") or "").splitlines()
    return dict(
        outcome=result["outcome"], qualified=verdict["qualified"], completed=verdict.get("completed", False),
        physical_ticks=end, actual_stage_transitions=result.get("stage_transitions"), world=world,
        fixed_world_BASE_publications=published_reference(native, end) if efile.is_file() else 0,
        capture=capture_evidence(case, native, result.get("observation_events")),
        stop_observed=proof["STOP_observed"] if proof else None,
        terminal_pause={key: pause[key] for key in ("step", "running", "clock_paused")} if pause else None,
        controller_cancel=result.get("cancel"), first_error=error[-1] if error else None,
        lease_closed=True, processes_absent=True, evidence_sha256={p.name: sha(p) for p in (
            case / "RESULT.private.json", case / "AUDIT.json", case / "control/LEASE.json", trace,
            native / "state.bin.journal", native / "pause_ack.json", identity, efile) if p.is_file()},
    )


def run(site):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU readout only")
    folder = site / "v19" / COMPACT_CAMPAIGN
    panel_path = folder / "PANEL_CLOSE.json"
    panel = read(panel_path)
    rows = {slot: row(folder / slot) for slot in panel["started"]}
    lease_ids = {read(folder / slot / "control/LEASE.json")["run_uuid"] for slot in rows}
    ticks = sum(r["physical_ticks"] for r in rows.values())
    require(set(rows)^set(panel["unstarted"]) == {"G00-H0", "G01-H0"} and not panel["missing_verdict"]
            and ticks == panel["observed_integrations"] and len(lease_ids) == len(rows)
            and panel["frozen_model_forwards"] == panel["training_updates"] == 0
            and all(not (folder / s / "NATIVE_REQUEST.json").exists() for s in panel["unstarted"]),
            "panel denominator differs")
    return dict(status="COMPACT_NATIVE_H0_PANEL_CLOSED", allocated=2, started=len(rows),
                unstarted=panel["unstarted"], qualified=sum(r["qualified"] for r in rows.values()),
                physical_ticks=ticks, rows=rows, frozen_model_forwards=0, training_updates=0,
                campaign_sha256={name: sha(folder / name) for name in
                                 ("PANEL_CLOSE.json", "BINDING.json", "ALLOCATION.json")})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("site", type=Path)
    print(json.dumps(run(cli.parse_args().site.resolve()), indent=2))
