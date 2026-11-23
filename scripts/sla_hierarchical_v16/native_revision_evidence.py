"""Measure installed overlap, solver lookahead and published arm commands on sealed native motion."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from receiver_fault_replay import read, sha
from sealed_panel import configure_site

DT = 0.002
FIRST = 5703
LAST = FIRST + 1500


def check(value, reason):
    if not value:
        raise ValueError(reason)


def stream(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def bounds(values):
    if not len(values):
        return None
    return dict(
        zip(
            ("min", "median", "p95", "max"),
            np.percentile(values, [0, 50, 95, 100]).tolist(),
            strict=True,
        )
    )


def installed_overlap(first, second):
    """Align real world knots, with sign-independent orientation and separate mount checks."""
    from scipy.spatial.transform import Rotation

    result = {}
    shift = second["_pi0294"]["source_mpc_s"] - first["_pi0294"]["source_mpc_s"]
    check(abs(shift - 0.1) < 1e-7, "unexpected installation period")
    for field in ("ee", "base"):
        data = []
        for doc in (first, second):
            points = doc[field]["waypoints"]
            t = np.array([p["time"] for p in points])
            check(
                len(t) == 41
                and np.allclose(t, np.arange(41) * 0.05, atol=1e-8, rtol=0),
                "malformed 41-point wire grid",
            )
            data.append(
                (
                    np.asarray([p["position"] for p in points]),
                    Rotation.from_quat([p["quaternion_xyzw"] for p in points]),
                )
            )
        (x, r), (y, s) = data
        dx = np.linalg.norm(x[2:] - y[:-2], axis=1)
        da = (r[2:] * s[:-2].inv()).magnitude()
        changed = np.flatnonzero((dx > 1e-6) | (da > 1e-5))
        result[field] = dict(
            overlap_knots=len(dx),
            changed=len(changed),
            first_change_s=float(changed[0] * 0.05) if len(changed) else None,
            maximum_translation_mm=float(1000 * max(dx)),
            maximum_angle_deg=float(np.degrees(max(da))),
            at_new_origin_mm=float(1000 * dx[0]),
            at_new_origin_deg=float(np.degrees(da[0])),
        )
    return result


def audit_case(folder, rr, pinned_ros):
    from go2arx5_vla.world_model.target_reference import trajectory_identity
    from d2_policy_content_replay import interpolate, stamp

    card = rr["card"]
    res = folder / "RESULT.private.json"
    check(sha(res) == rr["result_sha256"], "case seal")
    out = folder / read(res)["episode"]
    check(read(folder / "control/LEASE.json")["status"] == "CLOSED", "unclosed world")
    spec_path = folder / "EXECUTION_SPEC.json"
    check(
        sha(spec_path) == read(folder / "control/LEASE.json")["spec_sha256"],
        "spec identity",
    )
    spec = read(spec_path)
    controller = Path(spec["controller_binary"])
    solver = Path(spec["solver_binary"])
    for role, binary in (("solver", solver), ("controller", controller)):
        check(
            sha(binary)
            == spec["frozen_files"][str(binary)]
            == read(res)["loaded"][role],
            "frozen binary",
        )
    argv = next(
        x["argv"]
        for x in read(out / "PROCESS_COMMANDS.private.json")
        if x["name"] == "controller"
    )
    overrides = {v.split(":=", 1)[0] for v in argv if ":=" in v}
    check(
        "--params-file" not in argv
        and overrides <= {"use_sim_time", "urdfFile", "libFolder", "taskFile"},
        "preview overrides",
    )
    ros = out / "ROS_CONSUMPTION.private.jsonl"
    check(sha(ros) == pinned_ros, "changed ROS receipt")
    atom_file = out / "MOBILE_SAMPLES.private.jsonl"
    atoms = [r for r in stream(atom_file) if FIRST <= r["native_start_step"] < LAST]
    check(
        [r["native_start_step"] for r in atoms] == list(range(FIRST, LAST, 25)),
        "missing native command atoms",
    )
    keys = [a["installed_reference_sha256"] for a in atoms]
    check(
        keys[::2] == keys[1::2] and len(set(keys)) == 30,
        "10Hz ownership does not match 50ms atoms",
    )
    wires = keys[::2]
    events = out / "planner.jsonl"
    echo_map = defaultdict(list)
    docs = {}
    needed = set(wires)
    installed_wall = {}
    # Preserve warmup targets too: a prior policy may drive a post-boundary arm request.
    for e in stream(events):
        wire = e.get("ref_id")
        if e["kind"] == "accept" and wire in needed:
            check(
                hashlib.sha256(e["raw"].encode()).hexdigest() == wire, "accept identity"
            )
            doc = json.loads(e["raw"])
            check(
                doc["_pi0294"]["operation"] == "install"
                and doc["_pi0294"]["episode"] == out.name,
                "installation provenance",
            )
            docs[wire] = doc
        if (
            e["kind"] == "installed"
            and (
                wire in needed
                or (
                    e.get("observation_time_s") is not None
                    and FIRST * DT - 1.2 <= e["observation_time_s"] <= LAST * DT + 0.2
                )
            )
            and wire
        ):
            target = trajectory_identity(e["times"], e["poses"], e["inputs"])
            echo_map[target].append((wire, e["monotonic_s"]))
            if wire in needed:
                installed_wall[wire] = min(
                    e["monotonic_s"], installed_wall.get(wire, float("inf"))
                )
    check(
        set(wires) == set(docs) == set(installed_wall),
        "incomplete acceptance or receiver installation",
    )
    sources = [docs[k]["_pi0294"]["source_mpc_s"] for k in wires]
    check(
        np.allclose(sources, np.arange(FIRST, LAST, 50) * DT, rtol=0, atol=1e-8),
        "wire clock",
    )
    # The existing native atom join has already matched both its ROS target and its solver.
    for i, a in enumerate(atoms):
        echoes = a["solver_exact_installed_targets"]
        check(
            a["native_end_step"] == FIRST + 25 * (i + 1)
            and a["safety_valid"]
            and echoes
            and all(any(w == keys[i] for w, _ in echo_map[h]) for h in echoes),
            "unjoined native atom",
        )
    overlap = [
        dict(
            old=wires[i],
            new=wires[i + 1],
            tick=FIRST + 50 * (i + 1),
            **installed_overlap(docs[wires[i]], docs[wires[i + 1]]),
        )
        for i in range(29)
    ]
    requests = []
    policies = []
    observed_solver_echoes = set()
    for r in stream(ros):
        m = r["message"]
        wall = r["received_monotonic_s"]
        if r["topic_key"] == "arm_request":
            t = stamp(m)
            if FIRST * DT + 1e-8 < t <= LAST * DT + 1e-8:
                check(
                    m["joint_names"] == ["x5_joint" + str(i) for i in range(1, 7)]
                    and len(m["points"]) == 1,
                    "arm request ABI",
                )
                p = m["points"][0]
                requests.append((t, wall, p["positions"], p["velocities"]))
        elif r["topic_key"] == "policy":
            src = m["init_observation"]["time"]
            grid = np.asarray(m["time_trajectory"])
            if not len(grid) or grid[-1] < FIRST * DT or src > LAST * DT + 1e-8:
                continue
            target = m["plan_target_trajectories"]
            h = trajectory_identity(
                target["time_trajectory"],
                [v["value"] for v in target["state_trajectory"]],
                [v["value"] for v in target["input_trajectory"]],
            )
            ids = frozenset(w for w, v in echo_map[h] if v <= wall)
            observed_solver_echoes.update((ref, h) for ref in ids)
            q = np.asarray([v["value"] for v in m["state_trajectory"]])
            ff = np.asarray([v["data"] for v in m["data"]])
            check(
                m["controller_type"] == 1
                and q.shape == ff.shape == (len(grid), 12)
                and np.all(np.diff(grid) > 0),
                "unknown low-level policy",
            )
            policies.append(
                (
                    src,
                    grid,
                    wall,
                    q[:, 6:],
                    ff[:, 6:],
                    ids,
                    target["time_trajectory"][-1] - src,
                )
            )
    check(requests and policies, "empty motion interval")
    for atom in atoms:
        check(
            all(
                (atom["installed_reference_sha256"], h) in observed_solver_echoes
                for h in atom["solver_exact_installed_targets"]
            ),
            "native atom echo absent in solver journal",
        )
    tt = np.array([v[0] for v in requests])
    qout = np.array([v[2] for v in requests])
    vout = np.array([v[3] for v in requests])
    bins = [[] for _ in requests]
    near = np.zeros(len(requests), int)
    for n, (src, grid, _, q, ff, _, _) in enumerate(policies):
        ix = np.flatnonzero(
            (tt >= src - 1e-8)
            & (tt + 0.01 >= grid[0] - 1e-8)
            & (tt + 0.01 <= grid[-1] + 1e-8)
        )
        if not len(ix):
            continue
        qp = np.clip(interpolate(grid, q, tt[ix] + 0.15), -3.2, 3.7).astype(np.float32)
        vp = (0.4 * interpolate(grid, ff, tt[ix] + 0.01)).astype(np.float32)
        eq = np.max(abs(qp - qout[ix]), axis=1)
        ev = np.max(abs(vp - vout[ix]), axis=1)
        for j in ix[(eq == 0) & (ev == 0)]:
            bins[j].append(n)
        near[ix[(eq <= 2e-6) & (ev <= 2e-6)]] += 1
    counts = Counter()
    within = Counter()
    old_age = []
    for record, match in zip(requests, bins, strict=True):
        t, wall, *_ = record
        observed = [w for w in wires if installed_wall[w] <= wall]
        counts["requests"] += 1
        if len(match) != 1:
            counts["ambiguous_content" if match else "missing_content"] += 1
            continue
        counts["unique_content"] += 1
        src, _, arrival, _, _, ids, _ = policies[match[0]]
        if len(ids) != 1:
            counts["ambiguous_or_unjoined_target"] += 1
            continue
        wire = next(iter(ids))
        if not observed:
            counts["unobserved_current"] += 1
            continue
        current = observed[-1]
        if arrival > wall:
            counts["observed_solver_after_request"] += 1
        if wire == current:
            counts["latest_target"] += 1
        elif (
            wire not in installed_wall or installed_wall[wire] < installed_wall[current]
        ):
            counts["older_target"] += 1
            if arrival <= wall:
                counts["older_target_observed_solver_in_order"] += 1
            old_age.append((t - docs[current]["_pi0294"]["source_mpc_s"]) * 1000)
        else:
            counts["ordering_conflict"] += 1
    for v in policies:
        within["total"] += 1
        if len(v[5]) == 1:
            within["one_echo_wire"] += 1
    return dict(
        card=card,
        atoms=len(atoms),
        installs=len(wires),
        overlap=overlap,
        arm=dict(counts),
        close_tolerance_requests_with_match=int(sum(near > 0)),
        solver=dict(within),
        older_target_relative_to_latest_install_ms=bounds(old_age),
        optimizer_state_horizon_s=bounds([v[1][-1] - v[0] for v in policies]),
        echoed_reference_horizon_s=bounds([v[6] for v in policies]),
        ros_sha256=pinned_ros,
        planner_sha256=sha(events),
        atomic_record_sha256=sha(atom_file),
        selected_policy_id_exported=any(
            r["controller_selected_reference"]
            != "DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED"
            for r in atoms
        ),
    )


def run(root, repo):
    from d2_policy_content_replay import SOURCE
    from d1_readiness import baseline

    src = Path(baseline(root)["native_root"]) / "src"
    for name, expected in SOURCE.items():
        check(sha(next(src.rglob(name))) == expected, "solver replay source " + name)
    folder = repo / "reports/SLA_HIERARCHICAL_V16"
    hashes = read(folder / "V0_EVIDENCE_MATRIX_ACTUAL.json")["source_hashes"]
    ros = {
        r["card"]: r["ros_sha256"]
        for r in read(folder / "V3_DYNAMIC_MARGIN_ACTUAL.json")["records"]
    }
    records = []
    for phase in ("F1", "F2"):
        ledger = root / "feedback" / (phase + "_VERIFIED.json")
        check(sha(ledger) == hashes[phase], "unsealed population")
        (campaign,) = (root / "feedback").glob("sla-feedback-" + phase.lower() + "-*")
        for row in read(ledger)["records"]:
            result = {"card": row["card"], "eligible": row["eligible"]}
            if row["eligible"]:
                result.update(audit_case(campaign / row["card"], row, ros[row["card"]]))
            records.append(result)
    check(
        len(records) == 18 and sum(v["eligible"] for v in records) == 8,
        "startup denominator",
    )
    return dict(
        records=records, new_native_starts=0, new_physics_ticks=0, new_gpu_forwards=0
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    sys.path[:0] = [
        str(repo / "scripts/sla_transfer_v15"),
        str(repo / "scripts/sla_transfer_v14"),
    ]
    result = run(root, repo)
    with Path(sys.argv[2]).open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
