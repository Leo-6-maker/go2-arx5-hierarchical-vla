#!/usr/bin/env python3
"""Stage39 visualization-only renderer from sealed native-WBC state logs.

This file never constructs or runs the WBC.  It renders the already sealed
``native_external_rollout.npz`` rows as a transparent URDF-frame skeleton,
EEF/base/preview traces, and metadata overlays.  The skeleton is deliberately
used instead of re-running the simulator: this keeps the visualization
separate from Stage39 scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_RESULT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage39_structured_wbc_qualification_20260906/"
    "stage39_eval_603560.hopper-m-02/stage39_result.json"
)
DEFAULT_URDF = Path(
    "/scratch/e1192548/leon-storage/wbc-authority/"
    "stage36_authority_v2_20260906_clean_v1/WBCTrackingShare/"
    "resources/robots/go2_x5_v3/urdf/go2_x5.urdf"
)
DEFAULT_OUTPUT = Path(
    "/scratch/e1192548/leon-storage/runs/stage39_vis_offline_mp4_20260906"
)
FPS = 30
PADDING_S = 0.25


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    r, p, y = map(float, rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # URDF fixed-axis roll-pitch-yaw convention.
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12 or abs(float(angle)) < 1e-12:
        return np.eye(3)
    x, y, z = axis / n
    c, s = math.cos(float(angle)), math.sin(float(angle))
    C = 1.0 - c
    return np.array(
        [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
         [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
         [z * x * C - y * s, z * y * C + x * s, c + z * z * C]],
        dtype=float,
    )


def quat_matrix_xyzw(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = map(float, q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
        dtype=float,
    )


def transform(R: np.ndarray, p: Sequence[float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = np.asarray(p, dtype=float)
    return out


class URDFFrames:
    """Small stdlib URDF FK sufficient for transparent visualization."""

    def __init__(self, path: Path):
        root = ET.parse(path).getroot()
        self.children: Dict[str, List[dict]] = {}
        self.parent_links = set()
        self.child_links = set()
        for node in root.findall("joint"):
            parent = node.find("parent").attrib["link"]
            child = node.find("child").attrib["link"]
            origin = node.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            axis_node = node.find("axis")
            axis = np.fromstring(axis_node.attrib.get("xyz", "0 0 0"), sep=" ") if axis_node is not None else np.zeros(3)
            self.children.setdefault(parent, []).append(
                {"name": node.attrib["name"], "type": node.attrib.get("type", "fixed"),
                 "child": child, "R": rpy_matrix(rpy), "p": xyz, "axis": axis}
            )
            self.parent_links.add(parent)
            self.child_links.add(child)
        self.root = next(iter(self.parent_links - self.child_links))

    def positions(self, base_pos: Sequence[float], base_quat: Sequence[float], q: Sequence[float]) -> Dict[str, np.ndarray]:
        q = np.asarray(q, dtype=float).reshape(-1)
        # IsaacGym's delivered asset order is the explicit leg order followed by x5.
        qmap = {
            **{name: float(q[i]) for i, name in enumerate((
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint")) if i < len(q)},
            **{name: float(q[12 + i]) for i, name in enumerate((
                "x5_joint1", "x5_joint2", "x5_joint3", "x5_joint4", "x5_joint5", "x5_joint6")) if 12 + i < len(q)},
            "x5_gripper_joint": float(q[18]) if len(q) > 18 else 0.0,
            "x5_joint8": float(q[19]) if len(q) > 19 else 0.0,
        }
        world = transform(quat_matrix_xyzw(base_quat), base_pos)
        out: Dict[str, np.ndarray] = {self.root: world}

        def visit(parent: str) -> None:
            parent_T = out[parent]
            for joint in self.children.get(parent, []):
                jt = transform(joint["R"], joint["p"])
                value = qmap.get(joint["name"], 0.0)
                if joint["type"] == "revolute" or joint["type"] == "continuous":
                    jt = jt @ transform(axis_angle(joint["axis"], value), (0, 0, 0))
                elif joint["type"] == "prismatic":
                    jt[:3, 3] += joint["axis"] * value
                out[joint["child"]] = parent_T @ jt
                visit(joint["child"])

        visit(self.root)
        return {name: T[:3, 3].copy() for name, T in out.items()}


def interp_numeric(src_t: np.ndarray, src: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    src = np.asarray(src)
    if src.ndim == 1:
        return np.interp(dst_t, src_t, src)
    flat = src.reshape(len(src), -1)
    out = np.stack([np.interp(dst_t, src_t, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
    return out.reshape((len(dst_t),) + src.shape[1:])


def normalize_quat(values: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(n, 1e-12)


def resample(data: Mapping[str, np.ndarray], fps: int = FPS) -> Dict[str, np.ndarray]:
    source_t = np.asarray(data["sim_time"], dtype=float)
    source_t = source_t - source_t[0]
    duration = float(np.asarray(data["reference_duration"]).reshape(-1)[0])
    count = int(round(duration * fps)) + 1
    target_t = np.linspace(0.0, duration, count)
    out: Dict[str, np.ndarray] = {"sim_time": target_t}
    for key in ("base_pos", "desired_pos", "achieved_pos", "preview_pos", "dog_q", "arm_q", "dof_pos", "traj_s", "traj_s_ref"):
        if key in data:
            out[key] = interp_numeric(source_t, np.asarray(data[key]), target_t)
    # Orientation is not used for the skeleton camera, but retain normalized logs.
    for key in ("base_quat", "desired_quat", "achieved_quat"):
        if key in data:
            out[key] = normalize_quat(interp_numeric(source_t, np.asarray(data[key]), target_t))
    if "preview_pos" in out:
        out["preview_pos"] = out["preview_pos"].reshape(len(target_t), 9, 3)
    out["duration"] = np.asarray(duration)
    return out


def row_metric(row: Mapping[str, object]) -> float:
    return float(row["metrics"]["eef_tracking_xyz_m"]["mean"])


def select_rows(result: Mapping[str, object]) -> List[Tuple[str, Mapping[str, object]]]:
    rows = list(result["rows"])
    learned = [r for r in rows if r["row_id"].startswith("L")]
    accepts = sorted((r for r in learned if r["pre_wbc_safety_gate"]["decision"] == "ACCEPT"), key=row_metric)
    warns = sorted((r for r in learned if r["pre_wbc_safety_gate"]["decision"] == "WARN"), key=row_metric)
    if len(accepts) != 3 or len(warns) != 13:
        raise ValueError("sealed Stage39 gate counts changed")
    median_accept = accepts[len(accepts) // 2]
    median_warn = warns[len(warns) // 2]
    by_id = {r["row_id"]: r for r in rows}
    return [
        ("V0_C0_expert", by_id["C0_dense_expert_tiny1"]),
        ("V1_C1_structured_oracle", by_id["C1_structured_oracle_k16t16"]),
        ("V2_learned_ACCEPT_median", median_accept),
        ("V3_learned_WARN_median", median_warn),
        ("V4_learned_STRESS_max_error", max(learned, key=row_metric)),
    ]


def self_check() -> None:
    assert np.allclose(rpy_matrix((0, 0, 0)), np.eye(3))
    assert np.allclose(axis_angle(np.array([0, 0, 1]), math.pi / 2) @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-7)
    assert np.allclose(normalize_quat(np.array([[0, 0, 0, 2.0]])).ravel(), [0, 0, 0, 1])
    print("SELF_CHECK=PASS")


def render_one(label: str, row: Mapping[str, object], frames: URDFFrames, urdf_sha: str, out_path: Path, dpi: int = 100) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from imageio_ffmpeg import get_ffmpeg_exe

    npz_path = Path(str(row["output_dir"])) / "native_external_rollout.npz"
    source_sha = sha256_file(npz_path)
    raw = dict(np.load(npz_path, allow_pickle=False))
    data = resample(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=FPS, codec="libx264", quality=8, macro_block_size=1, ffmpeg_log_level="error")
    _ = get_ffmpeg_exe()  # fail early if the local imageio ffmpeg runtime is absent
    fig = plt.figure(figsize=(19.2, 10.8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    desired = data["desired_pos"]
    achieved = data["achieved_pos"]
    base = data["base_pos"]
    preview = data["preview_pos"]
    all_xyz = np.concatenate([desired, achieved, base, preview.reshape(-1, 3)], axis=0)
    lo = all_xyz.min(axis=0) - np.array([0.12, 0.12, 0.08])
    hi = all_xyz.max(axis=0) + np.array([0.12, 0.12, 0.12])
    span = max(float(np.max(hi - lo)), 0.7)
    center = (lo + hi) / 2
    lo = center - span / 2
    hi = center + span / 2
    segment_paths = [
        ["base", "FL_hip", "FL_thigh", "FL_calf", "FL_foot"],
        ["base", "FR_hip", "FR_thigh", "FR_calf", "FR_foot"],
        ["base", "RL_hip", "RL_thigh", "RL_calf", "RL_foot"],
        ["base", "RR_hip", "RR_thigh", "RR_calf", "RR_foot"],
        ["base", "x5_base_link", "x5_link1", "x5_link2", "x5_link3", "x5_link4", "x5_link5", "x5_link6", "gripper_center"],
    ]
    leg_colors = ["#2f80ed", "#56ccf2", "#6fcf97", "#27ae60"]
    pad_frames = max(1, round(PADDING_S * FPS))
    total = len(data["sim_time"]) + 2 * pad_frames
    gate = str(row["pre_wbc_safety_gate"]["decision"])
    gate_color = "#16a085" if gate == "ACCEPT" else "#d68910"
    frame_meta = {"frames": total, "fps": FPS, "reference_duration_s": float(data["duration"]), "padding_s_each_side": PADDING_S}

    for frame_idx in range(total):
        i = min(max(frame_idx - pad_frames, 0), len(data["sim_time"]) - 1)
        ax.clear()
        state = frames.positions(data["base_pos"][i], data["base_quat"][i], data["dof_pos"][i])
        for path_idx, path in enumerate(segment_paths):
            points = np.array([state[name] for name in path if name in state])
            color = "#d35400" if path_idx == 4 else leg_colors[path_idx]
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=3.0 if path_idx == 4 else 2.0, marker="o", markersize=3)
        ax.plot(base[: i + 1, 0], base[: i + 1, 1], base[: i + 1, 2], color="#7f8c8d", linewidth=1.5, linestyle="--", label="base trajectory")
        ax.plot(desired[:, 0], desired[:, 1], desired[:, 2], color="#27ae60", linewidth=1.5, linestyle=":", alpha=0.75, label="desired EEF")
        ax.plot(achieved[: i + 1, 0], achieved[: i + 1, 1], achieved[: i + 1, 2], color="#c0392b", linewidth=2.5, label="achieved EEF")
        ax.plot(preview[i, :, 0], preview[i, :, 1], preview[i, :, 2], color="#8e44ad", linewidth=1.5, linestyle="--", marker=".", markersize=5, label="9-point WBC preview")
        ax.scatter(*desired[i], color="#27ae60", marker="*", s=110, depthshade=False, label="current desired target")
        ax.scatter(*achieved[i], color="#c0392b", marker="o", s=45, depthshade=False)
        ax.scatter(*base[i], color="#7f8c8d", marker="s", s=35, depthshade=False)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)"); ax.set_zlabel("world z (m)")
        ax.view_init(elev=22, azim=-62)
        ax.set_box_aspect((1, 1, 0.8))
        ax.set_title("Stage39-VIS | OFFLINE STATE-LOG RENDER | SCIENTIFIC_EVIDENCE=false\nGo2 + ARX5 URDF-frame skeleton")
        sim_t = float(data["sim_time"][i])
        s = float(np.asarray(data["traj_s"])[i]) if "traj_s" in data else float("nan")
        s_ref = float(np.asarray(data["traj_s_ref"])[i]) if "traj_s_ref" in data else float("nan")
        seed = row.get("seed") if row.get("seed") is not None else "n/a"
        termination = "REFERENCE_COMPLETE" if i == len(data["sim_time"]) - 1 else "RUNNING"
        info = (f"{label}\nrow={row['row_id']}  seed={seed}  class={row['reference_class']}\n"
                f"SafetyGate={gate}  termination={termination}\n"
                f"sim_time={sim_t:.3f}s  traj_s={s:.3f}  traj_s_ref={s_ref:.3f}\n"
                f"source_npz_sha256={source_sha[:16]}…  URDF_sha256={urdf_sha[:16]}…")
        ax.text2D(0.015, 0.02, info, transform=ax.transAxes, fontsize=10, family="monospace", color=gate_color,
                  bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": gate_color, "boxstyle": "round,pad=0.45"})
        ax.legend(loc="upper left", fontsize=8)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
        writer.append_data(frame)
    writer.close()
    plt.close(fig)
    return {"label": label, "row_id": row["row_id"], "seed": row.get("seed"), "gate": gate,
            "reference_class": row["reference_class"], "source_npz": str(npz_path), "source_npz_sha256": source_sha,
            "output_mp4": str(out_path), "output_mp4_sha256": sha256_file(out_path), "frames": total,
            "fps": FPS, **frame_meta}


def combine_main(videos: Sequence[Mapping[str, object]], output: Path) -> dict:
    """Concatenate the four frozen presentation representatives only."""
    import imageio.v2 as imageio
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=FPS, codec="libx264", quality=8, macro_block_size=1, ffmpeg_log_level="error")
    frames = 0
    for video in videos[:4]:
        reader = imageio.get_reader(str(video["output_mp4"]))
        for frame in reader:
            writer.append_data(frame)
            frames += 1
        reader.close()
    writer.close()
    return {"label": "MAIN_V0_V1_V2_V3", "output_mp4": str(output),
            "output_mp4_sha256": sha256_file(output), "frames": frames, "fps": FPS,
            "segments": [video["label"] for video in videos[:4]],
            "scientific_evidence": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    result = json.loads(args.result.read_text(encoding="utf-8"))
    frames = URDFFrames(args.urdf)
    selected = select_rows(result)
    if args.skip_stress:
        selected = selected[:4]
    args.output.mkdir(parents=True, exist_ok=True)
    urdf_sha = sha256_file(args.urdf)
    manifest = {"schema": "STAGE39_VIS_OFFLINE_MP4_V1", "scientific_evidence": False,
                "scientific_head": "a7f91bfe88ab0e6d593a0b40af254126ccc1841c",
                "stage39_result": str(args.result), "stage39_result_sha256": sha256_file(args.result),
                "urdf": str(args.urdf), "urdf_sha256": urdf_sha, "fps": FPS, "padding_s_each_side": PADDING_S,
                "selection_rule": "V0 C0; V1 C1; V2 median ACCEPT XYZ mean; V3 median WARN XYZ mean; V4 max-error stress",
                "videos": []}
    for label, row in selected:
        out = args.output / (label.lower() + ".mp4")
        print("RENDER", label, row["row_id"], flush=True)
        manifest["videos"].append(render_one(label, row, frames, urdf_sha, out))
    manifest["main_video"] = combine_main(manifest["videos"], args.output / "stage39_vis_main_v0_v1_v2_v3.mp4")
    manifest_path = args.output / "stage39_vis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "manifest_sha256": sha256_file(manifest_path),
                      "videos": [(v["label"], v["output_mp4_sha256"]) for v in manifest["videos"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
