"""Exercise the frozen single-cube contact receiver with archived joint-state substitutions."""

import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys

import numpy as np

from demonstrated_grip import pin, read, require
from sealed_panel import configure_site


def frames_at(archive, requested):
    """Return only the registered binary joint states and their source-frame digests."""
    found = {}
    with archive.open("rb") as f:
        while sizes := f.read(4):
            (n,) = struct.unpack("<I", sizes)
            require(36 <= n <= 2**21, "frame size")
            raw = f.read(n)
            require(len(raw) == n and raw[:8] == b"GRTSTATE", "short state")
            tick, clock, nq, nv, nu = struct.unpack_from("<QdIII", raw, 8)
            if tick in requested:
                require(
                    abs(clock - tick * 0.002) < 1e-8 and (nq, nv, nu) == (41, 38, 20),
                    "source state ABI",
                )
                values = np.frombuffer(raw, dtype="<f8", count=nq + nv, offset=36)
                found[tick] = (
                    values[:nq].copy(),
                    values[nq:].copy(),
                    hashlib.sha256(raw).hexdigest(),
                )
            if tick >= max(requested):
                break
    require(found.keys() == requested, "missing native source state")
    return found


def make_archive(path, base, replacement):
    """Keep the non-integrating reader's original clock, controls and all other frames."""
    cut = None
    with base.open("rb") as src, path.open("xb") as dst:
        for step in range(1, 6454):
            size = src.read(4)
            (n,) = struct.unpack("<I", size)
            raw = bytearray(src.read(n))
            require(
                len(raw) == n
                and struct.unpack_from("<Q", raw, 8)[0] == step
                and struct.unpack_from("<III", raw, 24) == (34, 32, 20),
                "replay ABI",
            )
            if step == 6439:
                if replacement is not None:
                    raw[36 : 36 + 66 * 8] = (
                        np.r_[replacement[0][:34], replacement[1][:32]]
                        .astype("<f8")
                        .tobytes()
                    )
                cut = np.frombuffer(raw, dtype="<f8", count=66, offset=36).copy()
            dst.write(size)
            dst.write(raw)
    return cut


def decoded_contacts(model, values):
    import mujoco
    from go2arx5_vla.world_model.native_replay import forbid_integration

    with forbid_integration(mujoco):
        d = mujoco.MjData(model)
        d.qpos[:] = values[:34]
        d.qvel[:] = values[34:]
        mujoco.mj_forward(model, d)
        return [
            dict(
                geoms=[model.geom(c.geom1).name, model.geom(c.geom2).name],
                distance_m=float(c.dist),
            )
            for c in d.contact
        ]


def main(root, repo, out):
    import mujoco

    archive = root / "v16/v3-fault-reader01"
    test = pin(
        archive / "fault_probe",
        "117768796d5decdd0219ca2fc1b2a437415dea445c68b34acc041ce5dae8470c",
    )
    reset = pin(
        archive / "reset.bin",
        "5e2ee8eb42281d9cc21baadd0782168322b121c21dcbd6b991a94bd79683a718",
    )
    reports = repo / "reports/SLA_HIERARCHICAL_V16"
    old = read(
        pin(
            reports / "V3_RECEIVER_EXPIRY_REPLAY.json",
            "1c375c347a88aff11f45b5e09d922066ed747947ee5a2962698e512e871a2cd0",
        )
    )
    reader = pin(archive / "librecorded_step.so", old["test_step_sha256"])
    receiver = pin(root / "v16/v2-build03/student_grip.so", old["receiver_sha256"])
    case = root / "feedback/sla-feedback-f2-a1-20261123-01/F2-01"
    spec = read(
        pin(
            case / "EXECUTION_SPEC.json",
            read(case / "control/LEASE.json")["spec_sha256"],
        )
    )
    scene = Path(spec["model_path"])
    model = mujoco.MjModel.from_binary_path(
        str(pin(scene, spec["frozen_files"][str(scene)]))
    )
    n = case / read(case / "RESULT.private.json")["episode"]
    base = pin(n / "state.bin.journal", old["archived_state_sha256"])
    require((model.nq, model.nv, model.nu) == (34, 32, 20), "wrong receiver scene")
    destinations = [
        (
            model.joint(i).name or "",
            int(model.jnt_type[i]),
            int(model.jnt_qposadr[i]),
            int(model.jnt_dofadr[i]),
        )
        for i in range(model.njnt)
    ]
    pins = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]
    p = root / "v12/A2_O2_DENSE_CAPTURE.json"
    meta = {r["id"]: r for r in read(pin(p, pins[str(p)]))["episodes"]}
    call = read(
        pin(
            reports / "H1_REVISION_ACTUAL.json",
            "855860708527c0a2b0d86b5ad710ea94da7201ba89e02bca5c41e08518ca206c",
        )
    )["calls"]
    roster = {
        r["episode"]: r
        for r in read(
            pin(
                root / "evidence/P1_REGISTRATION.json",
                "1d496553b347a43c6f5c45c99d736896060ca9d71bf6cd5a72fef3e767b1b900",
            )
        )["target_windows"]
    }
    treatments = {"ORIGINAL": None}
    for k, selection in (("T14", {18953, 19515}), ("T16", {21453})):
        ep = call[k]["episode"]
        g = roster[ep]["episode_pin"]
        node = pin(Path(g["path"]), g["sha256"]).parents[3] / "cases" / ep / "native-01"
        m = meta[ep]
        rec = read(pin(Path(m["receipt_path"]), m["receipt_sha256"]))["source_files"]
        with pin(node / "physics.jsonl", rec["physics.jsonl"]).open() as f:
            header = json.loads(f.readline())
        src = [
            (x["name"], x["type"], x["qposadr"], x["dofadr"]) for x in header["joints"]
        ]
        require(
            src[:-1] == destinations and src[-1] == ("task_blue_cube_free", 0, 34, 32),
            "joint names, types or addresses differ",
        )
        for tick, state in frames_at(
            pin(node / "state.bin.journal", rec["state.bin.journal"]), selection
        ).items():
            treatments[f"{k}-{tick}"] = state
    out.mkdir(mode=0o700)
    results = {}
    for name, state in treatments.items():
        d = out / name
        d.mkdir(mode=0o700)
        joint = make_archive(d / "reader.bin", base, state)
        pairs = decoded_contacts(model, joint)
        token = hashlib.sha256(("noncontact:" + name).encode()).hexdigest()
        (d / "binding").write_text(
            "GRTQUOTA2\nVALID\n" + token + "\nVALID\n6503\n5\n128\n"
        )
        env = dict(
            os.environ,
            GR00T_TASK_GRIPPER_COMMAND=str(d / "unused_legacy.bin"),
            GR00T_TASK_STATE=str(d / "record.bin"),
            GR00T_TASK_STEP_QUOTA=str(d / "quota"),
            GR00T_TASK_QUOTA_BINDING=str(d / "binding"),
            GR00T_TASK_RUN_UUID="VALID",
            GR00T_TASK_CARD_SHA256=token,
            GR00T_TASK_RESERVATION_ID="VALID",
            GR00T_TASK_ERROR=str(d / "reject.txt"),
            V16_GRIP_DIR=str(d),
            V16_GRIP_EPOCH=token,
            V16_GRIP_STOP_TICK="6503",
            V3_REPLAY_JOURNAL=str(d / "reader.bin"),
        )
        proc = subprocess.run(
            [str(test), str(receiver), str(reader), str(scene), str(reset), "VALID"],
            env=env,
            cwd=d,
            umask=0o077,
            capture_output=True,
            timeout=90,
        )
        require(
            [Path(s).resolve() for s in (d / "route.txt").read_text().splitlines()]
            == [receiver.resolve(), reader.resolve()],
            "wrong nonintegrating call route",
        )
        delivered = np.fromfile(d / "delivered_ticks.bin", dtype="<u8")
        require(
            np.array_equal(delivered, np.arange(1, len(delivered) + 1))
            and len(delivered) in (6439, 6453),
            "replay count failure",
        )
        stopped = len(delivered) == 6439
        with (d / "receiver.csv").open() as f:
            stops = [r for r in csv.reader(f) if "STOP" in r[0]]
        last = next(reversed((d / "pause_calls.txt").read_text().splitlines()))
        require(
            int((d / "quota").read_text()) == 6453
            and int(last) == (6428 if stopped else 6453),
            "grant mismatch",
        )
        require(
            (
                proc.returncode == -signal.SIGINT
                and len(stops) == 1
                and stops[0][0] == "PROHIBITED_CONTACT_STOP"
                and int(stops[0][2]) == 6439
                and (d / "reject.txt").read_text().splitlines()
                == ["V16_NO_CONTACT_VIOLATION"]
            )
            if stopped
            else (proc.returncode == 0 and not stops),
            "unexplained receiver exit",
        )
        results[name] = dict(
            source_frame_sha256=state[2] if state is not None else None,
            reconstructed_contacts=pairs,
            exit_code=proc.returncode,
            contact_stop_tick=6439 if stopped else None,
            granted_to=6453,
            last_replayed_tick=int(delivered[-1]),
            previous_pause_or_final_tick=int(last),
            replayed_after_contact_stop=int((delivered > 6439).sum())
            if stopped
            else None,
        )
    require(
        results["ORIGINAL"]["last_replayed_tick"] == 6453, "unmodified control failed"
    )
    with (out / "ACTUAL.json").open("x") as f:
        json.dump(
            dict(
                conditions=results,
                receiver_sha256=old["receiver_sha256"],
                replay_step_sha256=old["test_step_sha256"],
                frozen_scene_sha256=spec["frozen_files"][str(scene)],
                native_starts=0,
                integrations=0,
            ),
            f,
            indent=2,
            allow_nan=False,
        )
        f.write("\n")


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    main(root, repo, Path(sys.argv[2]).resolve())
