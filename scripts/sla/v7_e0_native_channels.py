"""Export the observed articulation and base-command evidence before the fixed PLACE stops."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v7_e0_contact_geometry import ARM
from v7_place_first_divergence import CELLS, ROOTS, physics_rows


DIVERGENCE = "ae851efe8c5122972f7c17578f15507966fa1417c599d2f3ece23ad3f50e76c0"
REQUEST_MANIFEST = "74f1573534dbd9f6604961798142f0dddadf154662350a2cf4d0d6b2c122ed42"
TICK_SECONDS = .002


def controller_receipts(path, expected, ready, stop):
    """Retain receive-time windows without assigning an unstamped command to a solver tick."""
    base_commands, arm_requests, observation = [], [], None
    counts = Counter()
    digest = hashlib.sha256()

    def save_window(current, right_receive):
        step, left_receive, arm, base = current
        if ready <= step <= stop:
            counts[f"{len(arm)}_arm_{len(base)}_base"] += 1
            base_commands.extend((step, received, left_receive, right_receive, len(base), len(arm), *values)
                                 for received, values in base)
            arm_requests.extend((step, received, left_receive, right_receive, ns, *positions, *velocities)
                                for received, ns, positions, velocities in arm)

    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not any(marker in line for marker in (b'"observation"', b'"base_command"', b'"arm_request"')):
                continue
            row = json.loads(line)
            key = row["topic_key"]
            received = row["received_monotonic_s"]
            if key == "observation":
                time = row["message"]["time"]
                step = round(time / TICK_SECONDS)
                require(abs(time - step * TICK_SECONDS) < 1e-6, "Unexpected solver observation clock")
                if observation is not None:
                    save_window(observation, received)
                observation = [step, received, [], []]
            if observation is None:
                continue
            if key == "arm_request":
                msg = row["message"]
                require(tuple(msg["joint_names"]) == tuple(ARM) and len(msg["points"]) == 1,
                        "Unexpected arm request layout")
                stamp = msg["header"]["stamp"]
                ns = stamp["sec"] * 1_000_000_000 + stamp["nanosec"]
                point = msg["points"][0]
                positions, velocities = point["positions"], point["velocities"]
                require(len(positions) == len(velocities) == len(ARM)
                        and np.isfinite(positions + velocities).all(), "Invalid arm request values")
                observation[2].append((received, ns, positions, velocities))
            if key == "base_command":
                values = row["message"]["data"]
                require(len(values) == 6 and np.isfinite(values).all(), "Expected the original six base-command fields")
                observation[3].append((received, values))
    require(digest.hexdigest() == expected, "Archived controller receipts changed")
    if observation is not None and ready <= observation[0] <= stop:
        save_window(observation, -1.)
    return (np.asarray(base_commands, dtype=np.float64).reshape(-1, 12),
            np.asarray(arm_requests, dtype=np.float64).reshape(-1, 5 + 2 * len(ARM)), dict(counts))


def physical(case, terminal_sha, archive_sha, ready, stop):
    pin(case / "RESULT.private.json", terminal_sha)
    lease = read(case / "control/LEASE.json")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    require(lease["status"] == "CLOSED", "Only the original terminal archives may be exported")
    found = sha(case / "native-01/physics.jsonl")
    require(archive_sha is None or found == archive_sha, "Archived native physics changed")
    rows = list(physics_rows(case, ready, stop))
    steps = [row["step"] for row in rows]
    require(steps and all(right > left for left, right in zip(steps, steps[1:])), "Unordered or absent native samples")
    return spec, found, {row["step"]: row for row in rows}


def run(base, site, divergence_path, request_root, output):
    original = read(pin(divergence_path, DIVERGENCE))
    pin(request_root / "OUTPUTS.sha256", REQUEST_MANIFEST)
    checksums = {}
    for line in (request_root / "OUTPUTS.sha256").read_text().splitlines():
        digest, path = line.split(None, 1)
        checksums[Path(path.strip()).name] = digest
    require(tuple(row["cell"] for row in original["cases"]) == CELLS, "Wrong registered contact cohort")
    roots = [base / "R4-DEV-0200-PLACE-NATIVE-01", site / "RECOVERY1000-NATIVE-PLACE-01",
             site / "RECOVERY3000-NATIVE-PLACE-01"]
    summary = []
    for row in original["cases"]:
        case, ready, stop = (row[key] for key in ("cell", "ready_tick", "stop_tick"))
        arrays, specs, states, traffic = {}, [], [], {}
        parent_sha, final_sha = row["R4_vs_3K"]["direct_physics_on_native_ticks"]["archive_hashes"]
        for root, role, terminal, archive in zip(roots, ROOTS, row["result_sha256"], (parent_sha, None, final_sha), strict=True):
            case_root = root / f"{role}_{case}"
            spec, physics_sha, state = physical(case_root, terminal, archive, ready, stop)
            witness = f"{role}-{case}.json"
            record, = read(pin(request_root / witness, checksums[witness]))["records"]
            require(record["terminal_sha256"] == terminal and not record["independent_controller_witness"],
                    "The source evidence scope changed")
            commands, requests, windows = controller_receipts(case_root / "native-01/ROS_CONSUMPTION.private.jsonl",
                                                             record["ros_sha256"], ready, stop)
            arrays[role + "_base_delivery"] = commands
            arrays[role + "_arm_delivery"] = requests
            traffic[role] = dict(windows=windows, captured_base_commands=len(commands), captured_arm_requests=len(requests),
                                 unique_arm_stamps=len(np.unique(requests[:, 4])), input_sha256=record["ros_sha256"],
                                 physics_sha256=physics_sha, native_samples=len(state))
            specs.append(spec)
            states.append(state)
        require(all(spec["frozen_files"][spec["model_path"]] == row["scene_sha256"]
                    and spec["comparison_policy_seed"] == row["seed"] for spec in specs), "Scene or policy seed changed")
        ticks = sorted(states[0].keys() & states[1].keys() & states[2].keys())
        require(len(ticks) >= 2, "The three-way original physical sample intersection is empty")
        arrays["native_ticks"] = np.asarray(ticks)
        for role, rows in zip(ROOTS, states, strict=True):
            for key in ("qpos", "qvel", "tcp_xyz", "tcp_R"):
                arrays[f"{role}_{key}"] = np.asarray([rows[tick][key] for tick in ticks], dtype=np.float64)
        scene = Path(specs[0]["model_path"])
        model = mujoco.MjModel.from_binary_path(str(pin(scene, row["scene_sha256"])))
        qpos, qvel = ([int(getattr(model.joint(name), field)[0]) for name in ARM] for field in ("qposadr", "dofadr"))
        arrays["arm_joint_qpos_columns"] = np.asarray(qpos)
        arrays["arm_joint_qvel_columns"] = np.asarray(qvel)
        before_contact = arrays["native_ticks"] < stop
        difference = arrays["RECOVERY3000_qpos"] - arrays["R4_DEV_0200_qpos"]
        maximum = np.max(np.abs(np.rad2deg(difference[before_contact][:, qpos])), axis=0).tolist()
        path = output / f"{case}-OBSERVED-NATIVE-CHANNELS.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        summary.append(dict(cell=case, ready_tick=ready, stop_tick=stop, common_native_samples=len(ticks), last_common_tick=ticks[-1],
                            sample_gap_ticks=max(np.diff(ticks).tolist()), arm_joint_names=ARM,
                            r4_3k_max_absolute_arm_difference_deg=maximum, controller_delivery=traffic,
                            file=path.name, sha256=sha(path)))
    write(output / "ACTUAL.json", dict(scope="ARCHIVED_DIRECT_PHYSICS_AND_CONTROLLER_RECEIPTS", divergence_sha256=DIVERGENCE,
          base_delivery_columns=["previous_observation_tick", "received_monotonic_s", "left_observation_receive_s",
                                 "right_observation_receive_s", "interval_base_count", "interval_arm_count",
                                 *[f"message_data_{j}" for j in range(6)]],
          arm_delivery_columns=["previous_observation_tick", "received_monotonic_s", "left_observation_receive_s",
                                "right_observation_receive_s", "header_stamp_ns", *[f"q_{name}" for name in ARM],
                                *[f"qdot_{name}" for name in ARM]],
          base_command_native_stamp_available=False, cases=summary, new_native_integrations=0, new_model_forwards=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "site", "divergence_path", "request_root", "output"):
        p.add_argument(name, type=Path)
    run(**vars(p.parse_args()))
