"""Read-only verification of the official checkpoint, recorded forwards and sensor clocks."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def check(ok, label):
    if not ok:
        raise ValueError(label)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def pin(path, expected):
    actual = digest(path)
    check(actual == expected, str(path))
    return actual


def blob(path):
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def events(folder, name):
    return [
        r
        for line in (folder / "calls.jsonl").read_text().splitlines()
        if (r := json.loads(line))["event"] == name
    ]


def checkpoint(root):
    path = root / "evidence/SOURCE_AND_WEIGHT_MANIFEST.actual.json"
    manifest_hash = pin(
        path, "2e473d45a29f624f8c0242bf60389214613e3a060446e6845cd1a95322f2610c"
    )
    m = read(path)
    repo = root / "vendor/XPolicyLab"
    commit = subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip()
    check(
        commit == m["source_commit"] == "c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454",
        "upstream",
    )
    for f in m["source_files"]:
        p = repo / f["path"]
        pin(p, f["sha256"])
        check(blob(p) == f["git_blob_sha1"], str(p))
    for episode in m["episodes"]:
        p = root / "data" / Path(episode["path"]).name
        pin(p, episode["published_lfs_sha256"])
        check(p.stat().st_size == episode["bytes"], "source episode size")
    cfg_path = root / "evidence/P0_EFFECTIVE_CONFIG.json"
    pin(cfg_path, m["effective_config_sha256"])
    cfg = read(cfg_path)
    check(
        cfg == m["effective_config"]
        and cfg["extra_frames_first"] is False
        and (
            cfg["train_config_name"],
            cfg["action_type"],
            cfg["attention_impl"],
            cfg["env_cfg_type"],
        )
        == ("train_robodojo_real", "ee", "xla", "arx_x5"),
        "effective config",
    )
    base = repo / "policy/SLA_Memory/checkpoints"
    selected = repo / "policy/SLA_Memory" / cfg["ckpt_name"]
    check(selected == base / "SLA_Memory_robodojo_real_ckpt", "load path")
    files = m["verified_checkpoint_files"]
    check(
        {str(p.relative_to(base)) for p in selected.rglob("*") if p.is_file()}
        == {f["path"] for f in files},
        "checkpoint member set",
    )
    for f in files:
        p = base / f["path"]
        actual = pin(p, f["local_content_sha256"])
        check(
            p.stat().st_size == f["bytes"]
            and (
                actual == f["published_lfs_sha256"]
                if f["published_lfs_sha256"]
                else blob(p) == f["git_blob_sha1"]
            ),
            str(p),
        )
    check(
        m["model_revision"] == "7f6880569c3be777291e646fad1b4951245e8553", "HF revision"
    )
    return {
        "source_commit": commit,
        "source_files": len(m["source_files"]),
        "model": m["model_repository"],
        "revision": m["model_revision"],
        "checkpoint_files": files,
        "bytes": sum(f["bytes"] for f in files),
        "manifest_sha256": manifest_hash,
        "config": cfg,
        "norm_sha256": digest(selected / "assets/train_robodojo_real/norm_stats.json"),
    }


def source(root):
    report = read(root / "P0_REAL_GPU0_O0_REPORT.json")
    for f, h in report["files"].items():
        pin(root / f, h)
    d = root / "evidence/O0_PANEL_01"
    primary = {}
    rows = []
    metrics = {
        r["call"]: r["arms"]["right"]
        for r in read(root / "evidence/P0_SOURCE_ACTION_METRICS.json")["calls"]
    }
    for r in events(d, "FORWARD_RETURNED"):
        key = r["episode"], r["frame"]
        p = d / r["output_file"]
        pin(p, r["file_sha256"])
        with np.load(p, allow_pickle=False) as z:
            a, expert = z["native_absolute_arms"], z["expert_absolute_arms"]
            check(a.shape == expert.shape == (50, 16) and np.isfinite(a).all(), str(p))
            rmse = (
                np.sqrt(np.mean(np.sum((a[:, 8:11] - expert[:, 8:11]) ** 2, axis=1)))
                * 1000
            )
            hold = (
                np.sqrt(
                    np.mean(np.sum((z["state"][8:11] - expert[:, 8:11]) ** 2, axis=1))
                )
                * 1000
            )
            check(
                np.isclose(rmse, metrics[r["call"]]["position_rmse_mm"])
                and np.isclose(
                    hold, metrics[r["call"]]["hold_measured_state_position_rmse_mm"]
                ),
                "expert comparison",
            )
        inp = read(d / f"call_{r['call']:02d}_input.json")
        t = r["frame"]
        check(
            inp["head_frames"] == [0, *[max(0, t - k * 30) for k in range(8, 0, -1)], t]
            and inp["control_host_pixel_hash_match"] is True,
            str(key),
        )
        if r["mode"] == "primary":
            check(key not in primary, "duplicate primary")
            primary[key] = a
        else:
            check(np.array_equal(primary[key], a), "source repeat")
        rows.append(
            {
                "call": r["call"],
                "episode": r["episode"],
                "frame": t,
                "mode": r["mode"],
                "right_position_vector_rms_mm": float(rmse),
                "hold_current_state_rms_mm": float(hold),
                "output_sha256": r["file_sha256"],
            }
        )
    check(len(primary) == 6 and len(rows) == 8, "source denominator")
    return rows


def target(root):
    from transfer_frames import projector

    v = root / "v12"
    s0 = root / "v13/run01/S0_ACTUAL.json"
    pin(s0, "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9")
    pins = read(s0)["inputs"]
    cap = v / "A2_O2_DENSE_CAPTURE.json"
    reg = v / "TARGET_REGISTRATION.json"
    for p in (cap, reg):
        pin(p, pins[str(p)])
    m, reg = read(cap), read(reg)
    roster = {w["id"]: w for w in reg["windows"]}
    frames = {}
    for e in m["episodes"]:
        receipt = Path(e["receipt_path"])
        pin(receipt, e["receipt_sha256"])
        log = receipt.with_name("frames.jsonl")
        pin(log, read(receipt)["frames_sha256"])
        frames[e["id"]] = {
            r["tick"]: r for r in map(json.loads, log.read_text().splitlines())
        }
    left = read(v / "A1_SOURCE_01/call_02_input.json")["state"][:8]
    primary, rows = {}, []
    for folder in (v / "O2_GPU_01", v / "P2_EXTRA_GPU_01"):
        (launch,) = events(folder, "LOAD_ENTERED")
        pin(v / "target_forward.py", launch["runner_sha256"])
        pin(cap, launch["input_manifest_sha256"])
        check(
            launch["registered"] == "7f6880569c3be777291e646fad1b4951245e8553",
            "target revision",
        )
        entered = {r["call"]: r for r in events(folder, "FORWARD_ENTERED")}
        returned = events(folder, "FORWARD_RETURNED")
        check(len(entered) == len(returned), "incomplete forward")
        for r in returned:
            key = r["id"]
            w = m["windows"][key]
            check(
                all(r[k] == roster[key][k] for k in ("episode", "tick", "seed")),
                "target identity",
            )
            log = frames[w["episode"]]
            t = w["tick"]
            ticks = [0, *[max(0, t - k * 500) for k in range(8, 0, -1)], t]
            check(
                ticks == w["head_ticks"]
                and all(abs(log[k]["sim_time_s"] - 0.002 * k) < 1e-8 for k in ticks),
                "wrong historical timestamps",
            )
            p = v / "STUDENT_INPUT" / (key + ".npz")
            pin(p, w["input_sha256"])
            check(
                r["input_sha256"]
                == entered[r["call"]]["input_sha256"]
                == w["input_sha256"]
                and r["masks"] == [True] * 3,
                "target input binding",
            )
            actual, origin, offset = projector(log[t], reg)
            with np.load(p, allow_pickle=False) as z:
                images = [*z["head"], z["right_wrist"]]
                hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in images]
                check(
                    hashes
                    == w["rgb_sha256"]
                    == [log[k]["rgb_sha256"][0] for k in ticks]
                    + [log[t]["rgb_sha256"][1]],
                    "sensor pixel mismatch",
                )
                check(
                    np.array_equal(
                        z["state"], np.asarray([*left, *actual], dtype=np.float32)
                    )
                    and np.allclose(origin, w["T_W_source_origin"])
                    and np.allclose(offset, w["T_link6_TCP"]),
                    "target geometry",
                )
            p = folder / r["output_file"]
            pin(p, r["file_sha256"])
            a = np.load(p, allow_pickle=False)
            check(a.shape == (50, 16) and np.isfinite(a).all(), "decoded action shape")
            if r["repeat"]:
                check(np.array_equal(primary[key], a), "target repeat")
            else:
                check(key not in primary, "duplicate target primary")
                primary[key] = a
            rows.append(
                {
                    "id": key,
                    "tick": t,
                    "head_ticks": ticks,
                    "repeat": r["repeat"],
                    "input_sha256": r["input_sha256"],
                    "output_sha256": r["file_sha256"],
                }
            )
    check(
        set(primary) == set(roster) and len(primary) == 16 and len(rows) == 18,
        "target denominator",
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "v12"))
    report = {
        "checkpoint": checkpoint(root),
        "source": source(root),
        "target": target(root),
        "target_real_cameras": ["body", "right_wrist"],
        "virtual_left": "black image + fixed source arm",
        "archived_output": "official post-inverse absolute EEF, both arms",
        "new_gpu_forwards": 0,
        "new_physics_ticks": 0,
        "training_updates": 0,
    }
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
