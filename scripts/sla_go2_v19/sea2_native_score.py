"""Independent SEA calibration scoring from the standing receiver and state ledgers."""

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from sea2_native_profile_preflight import NATIVE_WINDOW_S, known_intent
from sea_reference import DT, OPEN, ReferenceSegment, motion_projection
from demonstrated_grip import require, rows
from known_curve_scoring import shape_score
from known_dynamic_readout import errors
from reach_pair import paired_audit
from scripts.pi0292.eval_contract import matrix10

STEP = .002
ATOM = round(DT / STEP)
MOVING_DWELL = round(NATIVE_WINDOW_S / STEP)
TERMINAL_REFUSALS = {
    "WRONG_EPISODE", "WRONG_CLOCK", "UNVERIFIED_REFERENCE", "CONTACT_OR_STOP", "GRIPPER_OWNERSHIP",
    "WRONG_PAUSED_OBSERVATION", "CROSS_OWNER_INTENT", "STALE_OR_CROSS_EPISODE_INTENT",
    "ORACLE_OR_UNBOUND_PROVENANCE", "INVALID_WORLD_POSE", "SOURCE_PHASE_WITHOUT_EXACT_CONSUMPTION",
}


def publication_poses(records, ledger, episode):
    """Join SHA-verified submissions to the original 41-point solver installations."""
    expected = {x["wire"]: x["tick"] for x in ledger}
    require(len(expected) == len(ledger), "duplicate publication wire")
    accepted, installed = set(), {}
    for row in records:
        wire = row.get("ref_id")
        if wire not in expected:
            continue
        tick = expected[wire]
        if row["kind"] == "accept":
            require(hashlib.sha256(row["raw"].encode()).hexdigest() == wire,
                    "accepted wire hash changed")
            meta = json.loads(row["raw"])["_pi0294"]
            require(meta["episode"] == episode and meta["operation"] == "install"
                    and abs(meta["source_mpc_s"] - tick*STEP) <= 1e-8
                    and meta["ttl_s"] == .4, "accepted publication clock or owner")
            accepted.add(wire)
        elif row["kind"] == "installed" and len(row["times"]) == 41:
            poses = np.asarray(row["poses"])
            require(wire in accepted and wire not in installed
                    and poses.shape == (41, 14) and np.isfinite(poses).all()
                    and abs(row["bound_start"] - tick*STEP) <= 1e-8
                    and np.allclose(row["times"], tick*STEP + np.arange(41)*DT, atol=1e-8, rtol=0),
                    "original solver installation does not match publication")
            out = np.repeat(np.eye(4)[None], len(poses), axis=0)
            out[:, :3, 3] = poses[:, :3]
            out[:, :3, :3] = Rotation.from_quat(poses[:, 3:7]).as_matrix()
            installed[wire] = out
    require(set(installed) == accepted == set(expected), "missing publication admission")
    return installed


def endpoints(atom):
    return tuple(matrix10(np.asarray(atom[key])) for key in ("installed_EEF10", "target_endpoint_EEF10"))


def consume_poses(atoms, ledger, installed, start, end):
    """Read exact consumed endpoints before sampling their known 2 ms reference."""
    require(len(atoms)*ATOM == end-start and len(ledger)*2 == len(atoms),
            "wrong number of publications or consumed atoms")
    pieces = []
    for i, atom in enumerate(atoms):
        pub = ledger[i//2]
        tick = start+i*ATOM
        wire = pub["wire"]
        require(
            atom["native_start_step"] == tick and atom["native_end_step"] == tick+ATOM
            and atom["ticks"] == ATOM and pub["tick"] == start + i//2*2*ATOM
            and atom["installed_reference_sha256"] == wire and atom["solver_exact_installed_targets"]
            and atom["safety_valid"] and not atom["native_grasp"]
            and atom["installed_EEF10"][9] == atom["target_endpoint_EEF10"][9] == OPEN
            and abs(atom["gripper_command"]["EEF_plan_source_s"] - pub["tick"]*STEP) <= 1e-8
            and abs(atom["gripper_command"]["source_native_s"] - tick*STEP) <= 1e-8,
            "unverified consumed atom",
        )
        k = i % 2
        prior, next_pose = endpoints(atom)
        require(np.allclose([prior, next_pose], installed[wire][k:k+2], atol=2e-7, rtol=0),
                "consumed pose differs from the solver installation")
        if pieces:
            require(np.allclose(pieces[-1][1], prior, atol=2e-7, rtol=0), "discontinuous consumed seam")
        pieces.append((prior, next_pose))
    return np.asarray([p[0] for p in pieces] + [pieces[-1][1]])


def interpolate(knots):
    atom_phase = np.arange((len(knots)-1)*ATOM+1)/ATOM
    nodes = np.arange(len(knots))
    out = np.repeat(np.eye(4)[None], len(atom_phase), axis=0)
    out[:, :3, 3] = np.column_stack([np.interp(atom_phase, nodes, p) for p in knots[:, :3, 3].T])
    out[:, :3, :3] = Slerp(nodes, Rotation.from_matrix(knots[:, :3, :3]))(atom_phase).as_matrix()
    return out


def read_capture(case, result, spec):
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from sea2_preflight import frozen_model
    from standing_reach import StateMonitor

    capture = case / result["episode"]
    start = json.loads((case / "REACH_ENTRY.json").read_text())["tick"]
    end = result["end"]
    require(result["cancel"]["native_step"] == end, "terminal cancel tick mismatch")
    atoms = list(rows(capture / "MOBILE_SAMPLES.private.jsonl"))
    prior = next(a for a in atoms if a["native_end_step"] == start)
    ledger = [r for r in rows(case / "REACH_WIRES.jsonl") if start <= r["tick"] < end]
    installed = publication_poses(rows(capture / "planner.jsonl"), ledger, result["episode"])
    interval = [a for a in atoms if start <= a["native_start_step"] < end]
    executed = consume_poses(interval, ledger, installed, start, end)
    model = frozen_model(spec)
    monitor = StateMonitor(model)
    achieved, clearances = [], []
    frames = packets(capture / "state.bin.journal")
    for packet in frames:
        if packet["step"] < start:
            continue
        require(packet["step"] == start + len(achieved), "discontinuous actual trajectory")
        x, r, _, _, _, clearance, _ = monitor.observe(packet)
        if not achieved:
            monitor.entry = monitor.stable[-1]
        p = np.eye(4)
        p[:3, :3], p[:3, 3] = r, x
        achieved.append(p)
        clearances.append(float(clearance)*1000)
    actual = np.asarray(achieved)
    last_pose, held_pose = endpoints(prior)
    require(len(actual) == end-start+1 and np.allclose(prior["actual_TCP"], actual[0], atol=2e-7, rtol=0)
            and np.allclose(executed[0], held_pose, atol=2e-7, rtol=0),
            "actual TCP or consumed entry does not match prior atom")
    for i, a in enumerate(interval):
        require(np.allclose(a["actual_TCP"], actual[(i+1)*ATOM], atol=2e-7, rtol=0),
                "atom actual TCP differs from independent state")
    return start, actual, executed, last_pose, held_pose, min(clearances)


def shape(actual, reference):
    def axes(a, b):
        return {axis: shape_score(x[:, None], y[:, None]) if np.ptp(y) > 1e-10 else None
                for axis, x, y in zip("xyz", a.T, b.T)}

    def rotvec(p):
        return Rotation.from_matrix(p[:, :3, :3] @ p[0, :3, :3].T).as_rotvec()

    return dict(xyz=axes(actual[:, :3, 3], reference[:, :3, 3]),
                rotation_vector=axes(rotvec(actual), rotvec(reference)))


def fixed_clock_score(actual, issued, segment, *, hold):
    t = np.arange(len(actual))*STEP
    _, _, phase = segment.at(t)
    nominal = segment.curve.at(30*phase)
    motion = np.ones(len(t), dtype=bool) if hold else (t >= segment.lead-1e-9) & (t <= segment.end+1e-9)
    require(len(actual) == len(issued) and np.sum(motion) >= 2 and t[-1] >= segment.end-1e-9,
            "registered motion phase is incomplete")
    a, n, i = actual[motion], nominal[motion], issued[motion]
    return dict(
        source_samples=int(motion.sum()), source_span_s=[float(t[motion][0]), float(t[motion][-1])],
        achieved_to_intent=errors(a, n), issued_to_intent=errors(i, n),
        achieved_to_issued=errors(a, i), shapes_to_intent=shape(a, n),
        shapes_to_issued=shape(a, i), terminal_hold=errors(actual[-501:], nominal[-501:]),
    )


def participation(actual, knots, segment, start):
    """Credit the first eligible signed increment separately for each motion channel."""
    _, _, phase = segment.at(np.arange(len(knots))*DT)
    begin = None
    found = {"translation": None, "rotation": None}
    for i in range(1, len(knots)):
        if (i-1)*DT < segment.lead-1e-9 or phase[i] <= phase[i-1]:
            continue
        if begin is None:
            begin = i
            continue
        values = motion_projection(actual[begin*ATOM], knots[begin], actual[i*ATOM], knots[i])
        for channel in found:
            if found[channel] is None and values[channel]:
                found[channel] = dict(tick=start+i*ATOM, **values)
    return found


def score_case(case, result, spec, card, limits):
    physical = paired_audit(case, result, spec)
    status = dict(
        completed=True, ticks=physical["ticks"], selected_MRT_policy_id=None,
        legacy_driver_outcome=result["outcome"], science_qualified=False,
    )
    if result["outcome"] == "A_REFERENCE_REFUSED":
        refusal = physical["refusal"]
        terminal = refusal["reason"] in TERMINAL_REFUSALS
        return dict(status, status="TERMINAL_REFUSAL" if terminal else "REFERENCE_REFUSED",
                    completed=not terminal, refusal=refusal)
    start, actual, knots, previous, held, clearance = read_capture(case, result, spec)
    require(len(actual) == MOVING_DWELL+1, "complete calibration window required")
    logs = list(rows(case / "SEA_REFERENCE.jsonl"))
    inputs = [r for r in logs if r["kind"] == "KNOWN_PHYSICAL_INPUT"]
    admitted = [r for r in logs if r["kind"] == "RETURN_ADMITTED"]
    require(len(inputs) == len(admitted) == result["student"]["accepted_returns"] == 1
            and admitted[0]["tick"] == inputs[0]["tick"] == start,
            "known physical intent or observation clock changed")
    received_tcp = np.asarray(inputs[0]["actual_tcp"])
    require(np.allclose(received_tcp, actual[0], atol=2e-7, rtol=0),
            "provider TCP differs from independent native state")
    intent = known_intent(result["episode"], start, received_tcp, card[2], card[3])
    require(admitted[0]["input"] == inputs[0]["input"] == intent.observation_sha256,
            "physical intent did not survive exact byte readback")
    segment = ReferenceSegment(intent.curve, received_tcp, held, previous, limits)
    expected = segment.at(np.arange(len(knots))*DT)[0]
    require(np.allclose(knots, expected, atol=2e-7, rtol=0),
            "consumed command differs from independently compiled SEA reference")
    evaluations = fixed_clock_score(actual, interpolate(knots), segment, hold=card[0]=="SEA2-HOLD")
    seen = participation(actual, knots, segment, start)
    motion = result["student"]["source_motion"]
    first = min((p["tick"] for p in seen.values() if p is not None), default=None)
    require(first == (None if motion is None else motion["tick"]), "live motion credit differs on replay")
    required_channels = {
        "SEA2-HOLD": (), "SEA2-XYZ": ("translation",),
        "SEA2-SO3": ("rotation",), "SEA2-MIXED": ("translation", "rotation"),
    }[card[0]]
    supported = all(seen[k] is not None for k in required_channels)
    qualified = evaluations["achieved_to_intent"]["in_tube"] == 1 and supported
    return dict(
        status, status="PROFILE_PASS" if qualified else "PROFILE_FAIL",
        science_qualified=qualified, actual_clearance_mm=clearance,
        source_motion=motion, independent_motion=seen, evaluations=evaluations,
    )
