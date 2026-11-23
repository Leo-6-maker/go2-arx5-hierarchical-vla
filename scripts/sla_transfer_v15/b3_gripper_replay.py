"""Recompute teacher finger motor inputs from original commands and archived 2 ms states."""

import argparse
from pathlib import Path

from d1_readiness import configure, pinned, read, record, rows, sha
from gripper_ownership import assess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_episode(spec_path, dense, windows, pins):
    import mujoco
    import numpy as np
    from go2arx5_vla.world_model import multitask_state_v6 as native

    from scripts.pi0294 import task_state

    cfg = read(spec_path)
    for module, name in (
        (native, "src/go2arx5_vla/world_model/multitask_state_v6.py"),
        (task_state, "scripts/pi0294/task_state.py"),
    ):
        pinned(Path(module.__file__), cfg["source_files"][name])
    compiled = cfg["compiled_outer_result"]
    pinned(Path(compiled["path"]), compiled["sha256"])
    build = read(Path(compiled["path"]))["outer_candidate"]
    library = Path(cfg["outer_library"])
    pinned(library, build["library_sha256"])
    pinned(library.with_suffix(".cpp"), build["source_sha256"])
    pinned(library, cfg["frozen_files"][str(library)])
    pinned(Path(dense["receipt_path"]), dense["receipt_sha256"])
    receipt = read(Path(dense["receipt_path"]))
    require(
        receipt["id"] == spec_path.parent.name and receipt["split"] == "TRAIN",
        "episode binding",
    )
    pinned(spec_path, receipt["source_spec_sha256"])
    model_path = Path(cfg["model_path"])
    pinned(model_path, receipt["scene_sha256"])
    model = mujoco.MjModel.from_binary_path(str(model_path))
    motors = [
        model.actuator("task_" + side + "_motor").id for side in ("left", "right")
    ]
    joints = [model.joint("task_" + side + "_slide").id for side in ("left", "right")]
    require(
        motors == [18, 19]
        and np.array_equal(model.actuator_trnid[motors, 0], joints)
        and np.all(model.actuator_gear[motors, 0] == 1)
        and model.opt.timestep == 0.002,
        "finger model ABI",
    )
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    observer = native.TwoObjectState(model, cfg["task_instruction_id"].split("-")[1])
    directory = spec_path.parent / "native-01"
    samples = directory / "MOBILE_SAMPLES.private.jsonl"
    pinned(samples, pins[str(samples)])
    commands = list(rows(samples))
    journal = directory / "state.bin.journal"
    pinned(journal, receipt["source_files"][journal.name])
    chosen = {w["consumed_at_tick"]: w for w in windows}
    require(len(chosen) == len(windows), "duplicate selected window")
    result = []
    maximum, sum_sq = 0.0, 0.0
    changes, discriminated = 0, 0
    step = 0
    first = commands[0]["native_start_step"]
    with journal.open("rb") as stream:

        def take():
            nonlocal step
            pkt = native.read_journal_packet(stream)
            step += 1
            require(pkt and pkt["step"] == step, "journal gap")
            return pkt

        while step < first - 2:
            take()
        prev = take()  # state before the last initialization integration
        start = take()  # first command is issued at this paused boundary
        u = start["ctrl"][motors]
        require(np.all(abs(u) < 8), "cannot invert saturated initial finger force")
        target = prev["qpos"][qa] + (u + 3 * prev["qvel"][va]) / 400
        last_width = None
        for index, row in enumerate(commands):
            cmd = row["gripper_command"]
            require(
                row["ticks"] == 25
                and row["native_start_step"] == step
                and cmd["sequence"] == index
                and cmd["source_native_s"] == start["time_s"],
                "issued command order or clock",
            )
            fraction = row["installed_EEF10"][9]
            require(
                0.002 / 0.044 <= fraction <= 0.042 / 0.044, "command width out of range"
            )
            width = 0.044 * fraction
            held_target = target.copy()
            max_residual = 0.0
            max_held_error = 0.0
            for _ in range(25):
                end = take()
                q, dq = start["qpos"][qa], start["qvel"][va]
                target += np.clip(width - target, -0.00008, 0.00008)
                expected = np.clip(400 * (target - q) - 3 * dq, -8, 8)
                actual = end["ctrl"][motors]
                error = actual - expected
                max_residual = max(max_residual, float(np.max(abs(error))))
                sum_sq += float(error @ error)
                if last_width is not None and width != last_width:
                    held_target += np.clip(last_width - held_target, -0.00008, 0.00008)
                    held_input = np.clip(400 * (held_target - q) - 3 * dq, -8, 8)
                    max_held_error = max(
                        max_held_error, float(np.max(abs(actual - held_input)))
                    )
                start = end
            require(
                step == row["native_end_step"]
                and np.array_equal(row["qpos"], end["qpos"])
                and np.array_equal(row["qvel"], end["qvel"]),
                "issued window/journal join",
            )
            maximum = max(maximum, max_residual)
            changed = last_width is not None and width != last_width
            changes += int(changed)
            discriminated += int(
                changed and max_held_error > 1e-6 and max_residual <= 1e-6
            )
            last_width = width
            if step in chosen:
                w = chosen[step]
                measured = float(np.mean(end["qpos"][qa]))
                observed = observer.update(end)
                require(
                    w["delivered_sequence"] == index
                    and w["source_native_s"] == cmd["source_native_s"]
                    and w["query_native_s"] == end["time_s"]
                    and abs(w["previous_command_m"] - width) < 1e-12
                    and abs(w["measured_m"] - measured) < 1e-12
                    and row["native_grasp"] == w["grasp"]
                    and observed["native_grasp"] == w["grasp"]
                    and observed["robot_object_contact"] == w["contact"]
                    and observed["safety_valid"]
                    and observed["contact_valid"]
                    and row["safety_valid"]
                    and w["contact_valid"]
                    and not w["prohibited_contact"],
                    "selected window does not match the archive",
                )
                result.append(
                    {
                        "id": w["id"],
                        "status": assess(w),
                        "sequence": index,
                        "tick": step,
                        "prior_issued_width_m": width,
                        "achieved_width_m": measured,
                        "new_first_width_m": w["new_first_m"],
                        "grasp": w["grasp"],
                        "contact": w["contact"],
                        "seam_speed_m_s": abs(w["new_first_m"] - width) / 0.05,
                        "changed": changed,
                        "max_force_error_N": max_residual,
                        "hold_previous_force_error_N": max_held_error
                        if changed
                        else None,
                    }
                )
    require(len(result) == len(chosen), "selected windows missing")
    count = len(commands) * 25
    return {
        "episode": dense["id"],
        "commands": len(commands),
        "replay_ticks": count,
        "selected": result,
        "max_force_error_N": maximum,
        "rms_force_error_N": (sum_sq / (2 * count)) ** 0.5,
        "changed_commands": changes,
        "changes_distinguished_from_keeping_previous": discriminated,
        "journal_sha256": sha(journal),
        "commands_sha256": sha(samples),
        "model_sha256": sha(model_path),
        "receiver_source_sha256": build["source_sha256"],
        "receiver_library_sha256": build["library_sha256"],
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    root = args.root.resolve()
    manifest = root / "v13/run01/S0_ACTUAL.json"
    pinned(manifest, "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9")
    pins = read(manifest)["inputs"]
    source = root / "v13/run01/S2_ACTUAL.json"
    pinned(source, "e53b719ef09742b623a15d337a6ad1fc45aefd33156cee373b1f956a5c4749d2")
    windows = read(source)["windows"]
    dense = root / "v12/A2_O2_DENSE_CAPTURE.json"
    pinned(dense, pins[str(dense)])
    specs = {
        p.parent.name: p for k in pins if (p := Path(k)).name == "EXECUTION_SPEC.json"
    }
    for p in specs.values():
        pinned(p, pins[str(p)])
    configure(read(next(iter(specs.values()))))
    episodes = read(dense)["episodes"]
    require(
        len(windows) == len({w["id"] for w in windows}) == 16
        and {w["episode"] for w in windows}
        == {e["id"] for e in episodes}
        == set(specs),
        "fixed panel join",
    )
    out = [
        audit_episode(
            specs[e["id"]], e, [w for w in windows if w["episode"] == e["id"]], pins
        )
        for e in episodes
    ]
    record(
        args.output,
        {
            "scope": "Sealed TRAIN teacher gripper motor replay. No native receiver owner or sequence ACK in state journal.",
            "tolerance_N": 1e-6,
            "episodes": out,
            "native_requests": 0,
            "new_physics_ticks": 0,
            "model_forwards": 0,
            "optimizer_updates": 0,
            "student_handoffs": 0,
        },
    )


if __name__ == "__main__":
    main()
