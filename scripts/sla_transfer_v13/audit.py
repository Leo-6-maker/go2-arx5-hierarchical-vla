"""Join sealed target arrays to original command receipts and the commissioned V1.2 kernel."""

import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from retime import DT, rates, twist


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pinned(path, expected, manifest):
    observed = sha(path)
    if observed != expected:
        raise ValueError(f"Changed input {Path(path).name}")
    manifest[str(path)] = observed
    return path


def record(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def load_episode(root, case_id, row, manifest):
    from scripts.pi0294.task_state import write_gripper
    from go2arx5_vla.world_model import fusion_trajectory_adapter as limits
    import inspect

    p = Path(row["episode_pin"]["path"])
    pinned(p, row["episode_pin"]["sha256"], manifest)
    ep = read(p)
    assert ep["split"] == "TRAIN" and ep["id"] == case_id
    c = p.parents[3] / "cases" / case_id
    n = c / "native-01"
    spec = read(c / "EXECUTION_SPEC.json")
    seal = read(root / "v12/DENSE" / case_id / "receipt.json")
    pinned(c / "EXECUTION_SPEC.json", seal["source_spec_sha256"], manifest)
    pinned(c / "RESULT.private.json", ep["native_result_sha256"], manifest)
    for k in (
        "native_binary",
        "controller_binary",
        "solver_binary",
        "outer_library",
        "trace_library",
        "observer_library",
        "mobile_safety_library",
        "model_path",
    ):
        path = Path(spec[k])
        pinned(path, spec["frozen_files"][str(path)], manifest)
    assert spec["native_backend"] == "MUJOCO_ONLY" and spec["physics_dt_s"] == 0.002
    src = Path(spec["source_root"])
    for key, expected in read(root / "v12/ABI_CPU_PREFLIGHT.json")["frozen_native_modules"].items():
        pinned(src / key, expected, manifest)
    for modulefile in [inspect.getfile(limits), inspect.getfile(write_gripper)]:
        f = Path(modulefile)
        pinned(f, spec["source_files"][str(f.relative_to(src))], manifest)
    assert "source+.4" in inspect.getsource(write_gripper).replace(" ", "")
    ci = read(root / "v12/CONTROL_INDEX.json")[case_id]
    samples = n / "MOBILE_SAMPLES.private.jsonl"
    queries = n / "LIVE_DATA/QUERIES.private.json"
    commands = n / "COMMANDS.private.jsonl"
    for f in [samples, queries, n / "control.jsonl"]:
        pinned(f, ci["archive_sha256"][f.name], manifest)
    pinned(commands, seal["source_files"]["COMMANDS.private.jsonl"], manifest)
    cs = {x["native_start_step"]: x for x in map(json.loads, samples.open())}
    receipts = {}
    for x in map(json.loads, commands.open()):
        h = x.get("wire_sha256")
        if h is None:
            continue
        receipts.setdefault(h, {}).setdefault(x["phase"], []).append(x)
    control_path = root / "v12/CONTROL_BINDINGS" / (case_id + ".jsonl")
    pinned(control_path, ci["sidecar_sha256"], manifest)
    controls = {x["tick"]: x for x in map(json.loads, control_path.open())}
    return spec, read(queries), cs, receipts, controls


def diagnostic(mats, dt, limits, full=False):
    result = {}
    for name, vec in rates(mats, dt).items():
        values = np.linalg.norm(vec, axis=1)
        threshold = limits[name]
        indices = np.arange(len(values))

        def region(mask):
            vals = values[mask]
            ratios = vals / threshold
            return dict(
                n=len(vals),
                maximum=float(vals.max()),
                threshold=threshold,
                ratios_med_p95_max=np.quantile(ratios, [0.5, 0.95, 1.0]).tolist(),
                exceedance_indices=indices[mask][ratios > 1 + 1e-10].tolist(),
            )

        parts = {"all": np.ones(len(values), bool), "first": indices == 0}
        if full:
            acc = name.endswith("acceleration")
            parts.update(
                interior=(indices >= 1) & (indices <= (31 if acc else 31)),
                suffix=(indices >= 32),
                endpoint=indices == indices[-1],
            )
        else:
            parts["interior"] = (indices > 0) & (indices < len(values) - 1)
        result[name] = dict(
            values=values.tolist(), regions={k: region(v) for k, v in parts.items()}
        )
    dp = np.linalg.norm(np.diff(mats[:, :3, 3], axis=0), axis=1)
    result["length_m"] = float(dp.sum())
    result["step_median_p95_max_m"] = np.quantile(dp, [0.5, 0.95, 1.0]).tolist()
    result["turn_rad"] = float(np.linalg.norm(twist(mats[:-1], mats[1:], 1)[1], axis=1).sum())
    return result


def make_context(root, w, old, loaded):
    from scripts.pi0291_arx5_data.native_dataset import matrix_from10
    from scripts.wm02.mfree_c20_query_audit_v3 import pose7

    spec, queries, cmd, receipts, controls = loaded
    t = w["tick"]
    q = queries[w["anchor"]]
    a = q["actual_state"]
    prior = cmd[t - 25]
    before = cmd[t - 50]
    prev = q["previous_confirmed_command"]
    assert "EXPERT" in spec["role"] and "EXPERT" in spec["scope"], "Unregistered owner population"
    assert (
        t == q["native_step"] == prior["native_end_step"]
        and prior["ticks"] == before["ticks"] == 25
        and prior["safety_valid"]
    )
    assert (
        before["native_end_step"] == prior["native_start_step"]
        and abs(0.044 * prior["installed_EEF10"][9] - prev["width_m"]) < 1e-12
    )
    assert (
        old["previous_sequence"] == prior["gripper_command"]["sequence"]
        and prev == old["previous_confirmed_command"]
    )
    bank = receipts[prior["installed_reference_sha256"]]
    ref = q["recent_reference"]
    assert (
        ref["exact_receiver_knot"]
        and ref["accepted_wire_sha256"] == prior["installed_reference_sha256"]
    )
    interval = queries[w["anchor"] - 1]["intervals"][-1]
    assert (
        interval["end_native_step"] == t
        and interval["confirmed_ticks"] == 25
        and interval["installed_reference_sha256"] == ref["accepted_wire_sha256"]
    )
    assert bank["accepted"] and bank["installed"] and prior["solver_exact_installed_targets"]
    assert prior["gripper_command"]["source_native_s"] == prev["source_native_s"]
    mode = controls[t]
    assert mode["last_fully_consumed_atom"]["sequence"] == old["previous_sequence"]
    actual = np.asarray(a["tcp"])
    reference = pose7(ref["accepted_pose7"])
    # Start command and receiver-confirmed end of the same actually consumed 25-tick interval.
    start = matrix_from10(np.asarray(prior["installed_EEF10"]))
    assert np.allclose(
        reference, matrix_from10(np.array(prior["target_endpoint_EEF10"])), atol=5e-7
    )
    incoming = twist(start, reference, DT)
    issued = twist(matrix_from10(np.array(before["installed_EEF10"])), start, DT)
    # Read achieved positions at consecutive sealed queries or at the exact start of this atom from dense replay.
    frames = root / "v12/DENSE" / w["episode"] / "frames.jsonl"
    pair = {r["tick"]: r for r in map(json.loads, frames.open()) if r["tick"] in (t - 25, t)}
    assert np.allclose(actual, pair[t]["T_W_TCP"], atol=1e-10)
    achieved = twist(np.array(pair[t - 25]["T_W_TCP"]), actual, DT)
    bm = np.array(a["base_pose"])
    base7 = np.r_[bm[:3, 3], Rotation.from_matrix(bm[:3, :3]).as_quat()]
    ctx = (
        np.array(spec["policy_from_world"]),
        np.array(q["state_eef10"]),
        actual,
        reference,
        prior,
        base7,
        prev,
    )
    first = old["student"]["working_width_m"][0]
    last = prev["width_m"]
    measured = a["gripper_fraction"] * 0.044
    contact = bool(
        a["robot_object_contact"]
        or a["left_object_contact"]
        or a["right_object_contact"]
        or a["native_grasp"]
    )
    ack = bank["accepted"][-1]["evidence"][0]
    installed = bank["installed"][-1]["evidence"][0]
    s2 = dict(
        teacher_owner=spec["owner"],
        role=spec["role"],
        student_owner_transfer_receipt=None,
        student_owned_reset=False,
        state_mode=mode["arm_mode"],
        fsm=mode["fsm"],
        running=mode["running"],
        controller_callback=mode["controller_call"],
        delivered_sequence=prior["gripper_command"]["sequence"],
        source_native_s=prev["source_native_s"],
        query_native_s=prev["query_native_s"],
        ttl_s=0.4,
        expiry_native_s=prev["source_native_s"] + 0.4,
        elapsed_consumption_ticks=25,
        consumed_at_tick=t,
        accepted_wire=ref["accepted_wire_sha256"],
        ack_monotonic_s=ack.get("monotonic_s"),
        install_monotonic_s=installed.get("monotonic_s"),
        solver_receipt_count=len(prior["solver_exact_installed_targets"]),
        receiver_ack_scope="EEF path confirmed and 25-tick paired gripper emission; independent gripper ownership ACK not exported",
        controller_selected_reference=ref["controller_selected_reference"],
        contact_valid=a["contact_valid"],
        contact=contact,
        grasp=a["native_grasp"],
        prohibited_contact=a["illegal_contact"],
        actual_takeover_flag=a["takeover"],
        measured_m=measured,
        previous_command_m=last,
        new_first_m=first,
        tracking_gap_m=abs(measured - last),
        first_measured_gap_m=abs(first - measured),
        seam_speed_m_s=abs(first - last) / DT,
        threshold_m_s=0.040,
        transition_lower_bound_s=abs(first - last) / 0.040,
        minimum_50ms_intervals=int(np.ceil(abs(first - last) / (0.040 * DT) - 1e-10)),
        G0="NOT_STUDENT_OWNED_RESET",
        G1="HANDOVER_BLOCKED",
        reason="CONTACT_OR_HOLD_AND_NO_OWNER_TRANSFER" if contact else "NO_OWNER_TRANSFER_ACK",
    )
    return ctx, incoming, issued, achieved, s2, q, spec
