"""Offline native progressive-relocation cohort construction.

The Stage15 shape files and placement profiles are immutable inputs.  This
module only creates derived references: a smooth reset-to-placement lead-in
is prepended to a parent bound trajectory, while the parent's challenge
segment is copied byte-for-byte in pose values.  The provenance and hashes
make the distinction auditable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from ..trajectory.so3 import geodesic_angle, quat_to_matrix, quat_slerp


SE3_LAMBDA_M_PER_RAD = 0.15


def _array(value: Any, dtype=np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _summary(values: Any) -> dict[str, float]:
    x = _array(values).reshape(-1)
    if x.size == 0:
        return {"mean": 0.0, "p01": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(x)),
        "p01": float(np.percentile(x, 1)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def quaternion_geodesic(q0: Any, q1: Any) -> np.ndarray:
    """Return SO(3) geodesic angle for xyzw quaternions."""

    a = _array(q0)
    b = _array(q1)
    return geodesic_angle(quat_to_matrix(a), quat_to_matrix(b))


def trajectory_descriptor(timestamp: Any, position: Any, quaternion: Any) -> dict[str, Any]:
    """Calculate timing, speed, orientation, and curvature descriptors."""

    ts = _array(timestamp)
    p = _array(position)
    q = _array(quaternion)
    if ts.ndim != 1 or p.shape != (len(ts), 3) or q.shape != (len(ts), 4):
        raise ValueError("trajectory arrays must be [T], [T,3], and [T,4]")
    if len(ts) < 2 or np.any(np.diff(ts) <= 0.0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    dt = np.diff(ts)
    dp = np.diff(p, axis=0)
    step = np.linalg.norm(dp, axis=-1)
    angular_step = quaternion_geodesic(q[:-1], q[1:])
    speed = step / np.maximum(dt, 1e-12)
    angular_speed = angular_step / np.maximum(dt, 1e-12)
    if len(p) > 2:
        unit = dp / np.maximum(step[:, None], 1e-12)
        valid_turn = (step[:-1] > 1e-10) & (step[1:] > 1e-10)
        turn = np.zeros(len(step) - 1, dtype=np.float64)
        turn[valid_turn] = np.arccos(
            np.clip(
                np.sum(unit[:-1][valid_turn] * unit[1:][valid_turn], axis=-1),
                -1.0,
                1.0,
            )
        )
        curvature = float(np.sum(turn) / max(float(np.sum(step)), 1e-9))
    else:
        curvature = 0.0
    orientation_range = quaternion_geodesic(np.broadcast_to(q[0], q.shape), q)
    se3_step = np.sqrt(step * step + (SE3_LAMBDA_M_PER_RAD * angular_step) ** 2)
    return {
        "duration_s": float(ts[-1] - ts[0]),
        "path_length_m": float(np.sum(se3_step)),
        "translation_path_length_m": float(np.sum(step)),
        "net_displacement_m": float(np.linalg.norm(p[-1] - p[0])),
        "translation_speed_mps": _summary(speed),
        "angular_speed_radps": _summary(angular_speed),
        "orientation_range_rad_max": float(np.max(orientation_range)),
        # This matches Stage13B's reconstructed generator descriptor:
        # accumulated tangent turn divided by translation path length.
        "curvature_proxy_rad_per_m": curvature,
        "curvature_proxy_max": curvature,
        "workspace_min": p.min(axis=0).tolist(),
        "workspace_max": p.max(axis=0).tolist(),
        "point_count": int(len(ts)),
    }


def _table_query(table: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Bilinear lookup matching Stage15's native 2-D reach table."""

    table = _array(table)
    u = _array(direction)
    az = np.arctan2(u[..., 1], u[..., 0])
    el = np.arcsin(np.clip(u[..., 2], -1.0, 1.0))
    n_az, n_el = table.shape

    def index(value, lo, span, n, wrap):
        x = (value - lo) / span * n - 0.5
        i0f = np.floor(x)
        w = np.clip(x - i0f, 0.0, 1.0)
        i0 = i0f.astype(np.int64)
        i1 = i0 + 1
        if wrap:
            i0 %= n
            i1 %= n
        else:
            i0 = np.clip(i0, 0, n - 1)
            i1 = np.clip(i1, 0, n - 1)
        return i0, i1, w

    a0, a1, wa = index(az, -np.pi, 2.0 * np.pi, n_az, True)
    e0, e1, we = index(el, -np.pi / 2.0, np.pi, n_el, False)
    return (
        table[a0, e0] * (1.0 - wa) * (1.0 - we)
        + table[a1, e0] * wa * (1.0 - we)
        + table[a0, e1] * (1.0 - wa) * we
        + table[a1, e1] * wa * we
    )


def load_reach_table(path: Path) -> np.ndarray:
    """Load the native ``go2_x5_2d.pt`` table without importing IsaacGym."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only on lean hosts
        raise RuntimeError("loading a .pt reach table requires torch") from exc
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # torch versions before the weights_only keyword
        payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, Mapping):
        for key in ("table", "reach_table", "R_max"):
            if key in payload:
                return _array(payload[key])
    return _array(payload)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trajectory_hash(timestamp: Any, position: Any, quaternion: Any, metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for value in (timestamp, position, quaternion):
        digest.update(np.asarray(value, dtype="<f8").tobytes())
    digest.update(json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def initial_reference_continuity(
    position: Any,
    quaternion: Any,
    reset_position: Any,
    reset_quaternion: Any,
) -> dict[str, float]:
    """Compute the first reference point's discontinuity from reset EEF."""

    p = _array(position)
    q = _array(quaternion)
    return {
        "initial_position_discontinuity_m": float(np.linalg.norm(p[0] - _array(reset_position))),
        "initial_rotation_discontinuity_rad": float(quaternion_geodesic(q[0], reset_quaternion)),
    }


def _reach_ratio(
    target_position: np.ndarray,
    shoulder_position: np.ndarray,
    shoulder_quaternion: np.ndarray,
    reach_table: np.ndarray,
) -> np.ndarray:
    delta = target_position - shoulder_position[None, :]
    radius = np.linalg.norm(delta, axis=-1)
    direction_world = delta / np.maximum(radius[:, None], 1e-12)
    shoulder_rotation = quat_to_matrix(shoulder_quaternion)
    direction_shoulder = np.einsum("ij,nj->ni", shoulder_rotation.T, direction_world)
    rmax = _table_query(reach_table, direction_shoulder)
    return radius / np.maximum(rmax, 1e-3)


def analyze_candidate(
    entry: Mapping[str, Any],
    bound_entry: Mapping[str, Any],
    shape_data: Mapping[str, Any],
    bound_data: Mapping[str, Any],
    reset_trace: Mapping[str, Any],
    reach_table: np.ndarray,
) -> dict[str, Any]:
    """Combine Stage15 provenance, continuity, and pointwise reach analysis."""

    timestamp = _array(shape_data["timestamp"])
    shape_position = _array(shape_data["position"])
    shape_quaternion = _array(shape_data["quaternion"])
    bound_position = _array(bound_data["position"]) + _array(reset_trace["anchor_position"])[None, :]
    bound_quaternion = _array(bound_data["quaternion"])
    reset_position = _array(bound_entry["reset_tcp_position_world"])
    reset_quaternion = _array(bound_entry["reset_tcp_quaternion_xyzw"])
    native = dict(entry["native_classification"])
    shoulder_position = _array(reset_trace["shoulder_position"])
    shoulder_quaternion = _array(reset_trace["shoulder_quaternion_xyzw"])
    fixed_rho = _reach_ratio(bound_position, shoulder_position, shoulder_quaternion, reach_table)
    static_offset = _array(native.get("best_static_base_xy_offset", [0.0, 0.0]))
    static_shoulder = shoulder_position + np.asarray([static_offset[0], static_offset[1], 0.0])
    static_rho = _reach_ratio(bound_position, static_shoulder, shoulder_quaternion, reach_table)
    unreachable = fixed_rho > 1.0
    first = int(np.flatnonzero(unreachable)[0]) if np.any(unreachable) else None
    continuity = initial_reference_continuity(
        bound_position, bound_quaternion, reset_position, reset_quaternion
    )
    result = {
        "trajectory_id": str(entry["trajectory_id"]),
        "parent_id": str(entry.get("parent_id", entry["trajectory_id"])),
        "source": entry.get("source"),
        "family": entry.get("family"),
        "original_class": entry.get("original_class"),
        "shape_file": entry.get("shape_file"),
        "shape_sha256": entry.get("shape_sha256"),
        "bound_file": bound_entry.get("path"),
        "native_class": native.get("native_class"),
        "placement_profile": entry.get("placement_profile", {}),
        "native_classification": native,
        "descriptor": trajectory_descriptor(timestamp, shape_position, shape_quaternion),
        "continuity": continuity,
        "fixed_base_max_rho_pointwise": float(np.max(fixed_rho)),
        "fixed_base_feasible_fraction": float(np.mean(fixed_rho <= 1.0)),
        "first_fixed_base_unreachable_index": first,
        "first_fixed_base_unreachable_time_s": None if first is None else float(timestamp[first]),
        "fixed_base_rho_p95": float(np.percentile(fixed_rho, 95)),
        "best_static_rho_max_pointwise": float(np.max(static_rho)),
        "best_static_feasible_fraction": float(np.mean(static_rho <= 1.0)),
        "minimum_static_base_translation_m": float(np.linalg.norm(static_offset)),
        "minimum_static_base_yaw_rad": None,
        "fixed_base_rho": fixed_rho.tolist(),
        "best_static_rho": static_rho.tolist(),
        "bound_position": bound_position.tolist(),
        "bound_quaternion": bound_quaternion.tolist(),
        "reset_tcp_position_world": reset_position.tolist(),
        "reset_tcp_quaternion_xyzw": reset_quaternion.tolist(),
    }
    return result


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3.0 - 2.0 * x)


def make_progressive_relocation(
    timestamp: Any,
    position: Any,
    quaternion: Any,
    reset_position: Any,
    reset_quaternion: Any,
    *,
    parent_id: str,
    parent_hash: str,
    sample_hz: Optional[float] = None,
    max_linear_speed_mps: float = 0.25,
    max_angular_speed_radps: float = 0.65,
    minimum_lead_in_s: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Prepend a bounded C1 lead-in from reset EEF to parent challenge start."""

    ts = _array(timestamp)
    p = _array(position)
    q = _array(quaternion)
    p0 = _array(reset_position)
    q0 = _array(reset_quaternion)
    if len(ts) < 2 or np.any(np.diff(ts) <= 0.0):
        raise ValueError("parent trajectory timestamps must be strictly increasing")
    parent_dt = float(np.median(np.diff(ts)))
    if sample_hz is None:
        # Keep the parent challenge samples byte/value-identical and make the
        # complete derived trajectory uniform.  The native loader's
        # ``to_canonical`` takes one dt, so mixing a 50 Hz lead-in with a 100
        # Hz parent silently shortens the reference time law.
        dt = parent_dt
        effective_sample_hz = 1.0 / dt
    else:
        effective_sample_hz = float(sample_hz)
        dt = 1.0 / effective_sample_hz
        if not np.allclose(np.diff(ts), dt, rtol=1e-5, atol=1e-8):
            raise ValueError("explicit lead-in sample_hz must match the parent uniform sample interval")
    distance = float(np.linalg.norm(p[0] - p0))
    angle = float(quaternion_geodesic(q0, q[0]))
    lead_duration = max(
        float(minimum_lead_in_s),
        1.5 * distance / max(float(max_linear_speed_mps), 1e-9),
        1.5 * angle / max(float(max_angular_speed_radps), 1e-9),
    )
    lead_count = max(2, int(np.ceil(lead_duration / dt)))
    lead_duration = lead_count * dt
    lead_ts = np.linspace(0.0, lead_duration, lead_count + 1)
    fraction = _smoothstep(lead_ts / lead_duration)
    lead_position = p0[None, :] + fraction[:, None] * (p[0] - p0)[None, :]
    lead_quaternion = np.asarray([quat_slerp(q0, q[0], value) for value in fraction])

    # The first challenge sample is the exact join point.  Drop it from the
    # appended suffix so the derived reference has no duplicate timestamp.
    challenge_ts = lead_duration + (ts - ts[0])
    out_ts = np.concatenate([lead_ts, challenge_ts[1:]])
    out_p = np.concatenate([lead_position, p[1:]], axis=0)
    out_q = np.concatenate([lead_quaternion, q[1:]], axis=0)
    metadata = {
        "derived_type": "DERIVED_PROGRESSIVE_RELOCATION",
        "parent_trajectory_id": str(parent_id),
        "parent_trajectory_hash": str(parent_hash),
        "lead_in_duration_s": float(lead_duration),
        "lead_in_sample_count": int(len(lead_ts)),
        "lead_in_path_length_m": distance,
        "lead_in_rotation_change_rad": angle,
        "join_time_s": float(lead_duration),
        "join_position_error_m": float(np.linalg.norm(out_p[lead_count] - p[0])),
        "join_rotation_error_rad": float(quaternion_geodesic(out_q[lead_count], q[0])),
        "max_linear_speed_mps": float(max_linear_speed_mps),
        "max_angular_speed_radps": float(max_angular_speed_radps),
        "sample_hz": float(effective_sample_hz),
        "challenge_start_index": int(lead_count),
        "challenge_source_values_unchanged": True,
    }
    metadata["derived_trajectory_hash"] = trajectory_hash(out_ts, out_p, out_q, metadata)
    return (
        {"timestamp": out_ts, "position": out_p, "quaternion": out_q},
        metadata,
    )


def select_progressive_ids(
    analyses: Sequence[Mapping[str, Any]], *, count: int = 8, preferred_ids: Optional[Iterable[str]] = None
) -> list[str]:
    """Select a deterministic, family-diverse subset of native static cases."""

    candidates = [
        row
        for row in analyses
        if row.get("native_class") == "NATIVE_STATIC_BASE_RELOCATION_SUFFICIENT"
    ]
    by_id = {str(row["trajectory_id"]): row for row in candidates}
    selected: list[str] = []
    for trajectory_id in preferred_ids or ():
        if trajectory_id in by_id and trajectory_id not in selected:
            selected.append(trajectory_id)
    # First preserve distinct motion families, preferring longer and more
    # informative challenge segments within each family.
    ranked = sorted(
        candidates,
        key=lambda row: (
            str(row.get("family", "")),
            -float(row.get("descriptor", {}).get("path_length_m", 0.0)),
            str(row["trajectory_id"]),
        ),
    )
    seen_families = set()
    for row in ranked:
        family = str(row.get("family", "UNKNOWN"))
        if family not in seen_families and len(selected) < count:
            selected.append(str(row["trajectory_id"]))
            seen_families.add(family)
    for row in sorted(candidates, key=lambda item: str(item["trajectory_id"])):
        if len(selected) >= count:
            break
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id not in selected:
            selected.append(trajectory_id)
    return selected[:count]
