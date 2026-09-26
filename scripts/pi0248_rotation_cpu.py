#!/usr/bin/env python3
"""PI0248 differentiable mirror of the pinned physical rotation decoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


THETA_REF = 0.05
LAMBDA_ROT = 0.1


def row_rot6d_to_matrix(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror EndEffectorPose._rot6d_to_matrix, including its zero-norm domain."""
    if value.shape[-1] != 6 or not torch.isfinite(value).all():
        raise ValueError("row rot6d must be finite with six values")
    row1_raw, row2_raw = value[..., :3], value[..., 3:]
    norm1 = torch.linalg.vector_norm(row1_raw, dim=-1, keepdim=True)
    if torch.any(norm1 == 0):
        raise ValueError("pinned row rot6d decoder is undefined for a zero first row")
    row1 = row1_raw / norm1
    row2_orth = row2_raw - (row1 * row2_raw).sum(dim=-1, keepdim=True) * row1
    norm2 = torch.linalg.vector_norm(row2_orth, dim=-1, keepdim=True)
    if torch.any(norm2 == 0):
        raise ValueError("pinned row rot6d decoder is undefined for collinear rows")
    row2 = row2_orth / norm2
    row3 = torch.cross(row1, row2, dim=-1)
    return torch.stack((row1, row2, row3), dim=-2), norm1.squeeze(-1), norm2.squeeze(-1)


def decode_relative_rotation(
    normalized_rot6d: torch.Tensor,
    state_rot6d: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Clip/min-max-decode, row-rot6d decode, then compose R_state @ R_relative."""
    if normalized_rot6d.ndim != 3 or normalized_rot6d.shape[1:] != (40, 6):
        raise ValueError("rotation action must have shape [B,40,6]")
    if state_rot6d.shape != (normalized_rot6d.shape[0], 6):
        raise ValueError("state anchor must have shape [B,6]")
    if minimum.shape != (40, 6) or maximum.shape != (40, 6):
        raise ValueError("pinned rotation min/max must have shape [40,6]")
    with torch.autocast(device_type=normalized_rot6d.device.type, enabled=False):
        normalized = normalized_rot6d.float()
        lo, hi = minimum.to(normalized.device, torch.float32), maximum.to(normalized.device, torch.float32)
        relative = (normalized.clamp(-1.0, 1.0) + 1.0) * 0.5 * (hi - lo) + lo
        r_relative, n1, n2 = row_rot6d_to_matrix(relative)
        r_state, state_n1, state_n2 = row_rot6d_to_matrix(state_rot6d.float())
        absolute = r_state[:, None] @ r_relative
    return absolute, {"relative_row1_norm": n1, "relative_row2_orth_norm": n2,
                      "state_row1_norm": state_n1, "state_row2_orth_norm": state_n2,
                      "relative_physical_rot6d": relative}


def rotation_chordal_loss(
    generated: torch.Tensor,
    expert: torch.Tensor,
    action_mask: torch.Tensor,
    state_eef: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    *,
    theta_ref: float = THETA_REF,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if generated.shape != expert.shape or generated.shape != action_mask.shape:
        raise ValueError("generated, expert, and mask shapes must match")
    if generated.ndim != 3 or generated.shape[1:] != (40, 132) or state_eef.shape != (generated.shape[0], 9):
        raise ValueError("rotation endpoint loss requires [B,40,132] and [B,9] state")
    valid = (action_mask[..., 3:9] > 0).all(dim=-1)
    count = int(valid.sum().item())
    if count == 0:
        raise ValueError("rotation endpoint loss has no valid rotation points")
    with torch.autocast(device_type=generated.device.type, enabled=False):
        predicted, pred_details = decode_relative_rotation(
            generated[..., 3:9].float(), state_eef[:, 3:9].float(), minimum, maximum)
        target, target_details = decode_relative_rotation(
            expert[..., 3:9].float(), state_eef[:, 3:9].float(), minimum, maximum)
        per_step = (predicted - target).square().sum(dim=(-2, -1)) / (2.0 * float(theta_ref) ** 2)
        loss = (per_step * valid.float()).sum() / valid.sum()
    if not torch.isfinite(loss):
        raise ValueError("rotation endpoint loss is nonfinite")
    orthogonal = predicted @ predicted.transpose(-1, -2)
    eye = torch.eye(3, dtype=predicted.dtype, device=predicted.device)
    return loss, {
        "valid_rotation_points": count,
        "saturated_rotation_values": int((generated[..., 3:9].detach().abs() > 1.0).sum().item()),
        "relative_row1_norm_min": float(pred_details["relative_row1_norm"].detach().min().item()),
        "relative_row2_orth_norm_min": float(pred_details["relative_row2_orth_norm"].detach().min().item()),
        "rotation_orthogonality_max_abs": float((orthogonal - eye).detach().abs().max().item()),
        "target_row1_norm_min": float(target_details["relative_row1_norm"].detach().min().item()),
        "target_row2_orth_norm_min": float(target_details["relative_row2_orth_norm"].detach().min().item()),
    }


def add_rotation_term(base: torch.Tensor, rotation: torch.Tensor, coefficient: float) -> torch.Tensor:
    # Keep lambda=0 exactly the existing U objective and its original gradient.
    return base if coefficient == 0.0 else base + float(coefficient) * rotation


def _numpy_rows(value: np.ndarray) -> np.ndarray:
    import torch.nn.functional as F

    a = np.asarray(value, dtype=np.float64)
    row1, row2 = a[..., :3], a[..., 3:]
    row1 = F.normalize(torch.from_numpy(row1), dim=-1).numpy()
    row2 = row2 - np.sum(row1 * row2, axis=-1, keepdims=True) * row1
    row2 = F.normalize(torch.from_numpy(row2), dim=-1).numpy()
    row3 = np.cross(row1, row2)
    return np.stack((row1, row2, row3), axis=-2)


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rotation_self_check(minimum: torch.Tensor, maximum: torch.Tensor,
                         raw: np.ndarray, state: np.ndarray) -> dict:
    from go2arx5_vla.trajectory.so3 import rot6d_to_matrix

    # Noncommuting anchor and relative rotations detect composition/order and row/column swaps.
    ax, ay = 0.61, 0.47
    rx = np.asarray([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ry = np.asarray([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    packed = lambda r: r[:2].reshape(6)
    rx_rows, ry_rows = packed(rx), packed(ry)
    torch_rx, _, _ = row_rot6d_to_matrix(torch.from_numpy(rx_rows).float())
    torch_ry, _, _ = row_rot6d_to_matrix(torch.from_numpy(ry_rows).float())
    compose = torch_rx @ torch_ry
    order_delta = float((compose - torch_ry @ torch_rx).abs().max().item())
    column_delta = float(np.max(np.abs(rot6d_to_matrix(rx_rows) - rx)))
    if order_delta < 0.1 or column_delta < 0.1:
        raise AssertionError("row convention or relative composition order was not distinguished")

    theta = 0.37
    rz = np.asarray([[math.cos(theta), -math.sin(theta), 0],
                     [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    rref = torch.from_numpy(rx).float()
    rhat = rref @ torch.from_numpy(rz).float()
    rstar = rref
    observed = float((rhat - rstar).square().sum().item() / (2 * THETA_REF ** 2))
    expected = 4.0 * math.sin(theta / 2.0) ** 2 / THETA_REF ** 2
    if not math.isclose(observed, expected, rel_tol=2e-6, abs_tol=2e-6):
        raise AssertionError("known-angle chordal identity failed")

    # Exercise the actual 40-row bounds and the official clip path.
    test_raw = np.zeros((1, 40, 132), dtype=np.float32)
    test_raw[..., 3] = 2.0
    import pi0247_m64_screen as m64
    card, fit, ctx = m64.inputs()
    import pi0242_train_infer_bridge as bridge
    official = bridge._decode_physical(ctx["processor"], ctx["embodiment"], test_raw, state)
    mirror, _ = decode_relative_rotation(torch.from_numpy(test_raw[..., 3:9]),
        torch.from_numpy(state[None, 3:9]), minimum, maximum)
    native = torch.from_numpy(_numpy_rows(official[0, :, 3:9])).float()
    clip_max = float((mirror[0] - native).abs().max().item())
    if clip_max > 1e-5:
        raise AssertionError(f"clipped-rotation decoder parity failed: {clip_max}")

    # Prediction equals expert gives zero; the mask excludes exactly one horizon.
    source = torch.from_numpy(raw.astype(np.float32)).unsqueeze(0)
    state_t = torch.from_numpy(state[None, :9].astype(np.float32))
    mask = torch.ones((1, 40, 132), dtype=torch.float32)
    same, _ = rotation_chordal_loss(source, source, mask, state_t, minimum, maximum)
    if float(same.item()) != 0.0:
        raise AssertionError("equal prediction/target must have zero rotation loss")
    mask[:, 11, 3:9] = 0
    shifted = source.clone()
    shifted[:, 11, 4] += 0.03
    masked_loss, masked_info = rotation_chordal_loss(shifted, source, mask, state_t, minimum, maximum)
    full_mask = torch.ones_like(mask)
    full_loss, _ = rotation_chordal_loss(shifted, source, full_mask, state_t, minimum, maximum)
    if masked_info["valid_rotation_points"] != 39 or not torch.isfinite(masked_loss) or not full_loss > masked_loss:
        raise AssertionError("rotation mask does not exclude only the masked horizon")

    # Finite difference through min-max, Gram-Schmidt, and anchor composition.
    target = source.clone()
    coordinate = next(((h, d) for h in range(40) for d in range(3, 9)
                       if abs(float(source[0, h, d])) < 0.8), None)
    if coordinate is None:
        raise AssertionError("no interior normalized rot6d coordinate for finite difference")
    h, d = coordinate
    target[0, h, d] += 0.04
    x = source.clone().requires_grad_(True)
    fd_mask = torch.ones_like(mask)
    loss, _ = rotation_chordal_loss(x, target, fd_mask, state_t, minimum, maximum)
    grad = float(torch.autograd.grad(loss, x)[0][0, h, d].item())
    delta = 1e-3
    with torch.no_grad():
        plus, minus = source.clone(), source.clone()
        plus[0, h, d] += delta
        minus[0, h, d] -= delta
        f_plus = float(rotation_chordal_loss(plus, target, fd_mask, state_t, minimum, maximum)[0].item())
        f_minus = float(rotation_chordal_loss(minus, target, fd_mask, state_t, minimum, maximum)[0].item())
    finite_difference = (f_plus - f_minus) / (2 * delta)
    if not math.isclose(grad, finite_difference, rel_tol=2e-2, abs_tol=2e-3):
        raise AssertionError(f"rotation loss finite difference mismatch: autograd={grad}, fd={finite_difference}")

    scalar = torch.tensor(0.3, requires_grad=True)
    base = scalar.square()
    extra = (scalar * 7).square()
    zero_total = add_rotation_term(base, extra, 0.0)
    base_grad = torch.autograd.grad(base, scalar, retain_graph=True)[0]
    zero_grad = torch.autograd.grad(zero_total, scalar)[0]
    if not torch.equal(base, zero_total) or not torch.equal(base_grad, zero_grad):
        raise AssertionError("lambda_rot=0 changed original loss value or gradient")

    return {"noncommuting_composition_order_max_delta": order_delta,
            "row_vs_column_rotation_max_delta": column_delta,
            "known_angle_chordal_observed": observed, "known_angle_chordal_expected": expected,
            "clipped_decoder_matrix_max_abs": clip_max, "synthetic_clipped_values": 40,
            "equal_action_rotation_loss": float(same.item()),
            "mask_valid_points": masked_info["valid_rotation_points"],
            "finite_difference_coordinate": [h, d], "finite_difference_autograd": grad,
            "finite_difference_centered": finite_difference,
            "lambda_zero_value_and_gradient_exact": True}


def run_cpu_check(output: Path) -> None:
    import pi0247_m64_screen as m64
    import pi0247_d1_screen as d1
    card, fit, ctx = m64.inputs()
    import pi0242_train_infer_bridge as bridge
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    processor = ctx["processor"]
    embodiment = ctx["embodiment"].value
    params = processor.state_action_processor.norm_params[embodiment]["action"]["eef_9d"]
    min64 = np.asarray(params["min"], dtype=np.float64)
    max64 = np.asarray(params["max"], dtype=np.float64)
    if min64.shape != (40, 9) or max64.shape != (40, 9) or not np.isfinite(min64).all() or not np.isfinite(max64).all():
        raise AssertionError("pinned horizon-specific min/max do not match H40 eef_9d")
    minimum = torch.from_numpy(min64[:, 3:9].astype(np.float32))
    maximum = torch.from_numpy(max64[:, 3:9].astype(np.float32))
    if torch.any(maximum <= minimum):
        raise AssertionError("pinned rotation range contains a zero/negative-width coordinate")

    report_path = m64.M64 / "screen_step5000/screen_report.json"
    screen = _read(report_path)
    assert len(screen["cells"]) == 128
    max_error = 0.0
    worst = None
    clips = 0
    anchors = 0
    row1_min = float("inf")
    row2_min = float("inf")
    horizon_error = np.zeros(40, dtype=np.float64)
    source_set = set()
    for item in screen["cells"]:
        assert item["status"] == "VALID"
        record_path = Path(item["source_cell"]) / "record.json"
        record = d1.collect_record(record_path, ctx["p1"])
        with np.load(record_path.parent / "tensors.npz", allow_pickle=False) as z:
            raw, saved, state = z["raw_scored"].copy(), z["decoded"].copy(), z["state"].copy()
        assert raw.shape == (1, 40, 132) and saved.shape == (1, 40, 10) and state.shape == (10,)
        actual = bridge._decode_physical(processor, ctx["embodiment"], raw, state)
        rhat, details = decode_relative_rotation(torch.from_numpy(raw[..., 3:9]),
            torch.from_numpy(state[None, 3:9].astype(np.float32)), minimum, maximum)
        official = torch.from_numpy(rot6d_rows_to_matrix(actual[0, :, 3:9])).float()
        saved_matrix = torch.from_numpy(rot6d_rows_to_matrix(saved[0, :, 3:9])).float()
        per_horizon = (rhat[0] - official).abs().amax(dim=(-2, -1)).numpy()
        error = float(per_horizon.max())
        if error > max_error:
            max_error, worst = error, {"source_episode_id": item["source_episode_id"], "noise": item["noise_name"]}
        horizon_error = np.maximum(horizon_error, per_horizon)
        native_saved_error = float((official - saved_matrix).abs().max().item())
        if native_saved_error > 1e-5:
            raise AssertionError(f"pinned CPU decoder differs from saved screen output: {native_saved_error}")
        clips += int((np.abs(raw[..., 3:9]) > 1.0).sum())
        anchor = _numpy_rows(state[None, 3:9])[0]
        anchors += int(np.max(np.abs(anchor - np.eye(3))) > 1e-3)
        row1_min = min(row1_min, float(details["relative_row1_norm"].min().item()))
        row2_min = min(row2_min, float(details["relative_row2_orth_norm"].min().item()))
        source_set.add(item["source_episode_id"])
    if max_error > 1e-5 or len(source_set) != 64 or anchors == 0:
        raise AssertionError(f"CPU decoder mirror failed: max={max_error}, sources={len(source_set)}, anchors={anchors}")

    first = screen["cells"][0]
    record_path = Path(first["source_cell"]) / "record.json"
    with np.load(record_path.parent / "tensors.npz", allow_pickle=False) as z:
        raw = z["raw_scored"].copy()[0]
        state = z["state"].copy()
    check = _rotation_self_check(minimum, maximum, raw, state)
    stats = fit["processor_identity"]["frozen_C_T"]
    result = {
        "schema": "PI0248_ROTATION_DECODER_CPU_CHECK_V1", "status": "PASS_CPU_DECODER_AND_GRADIENT",
        "run_card_sha256": m64.sha(m64.START / "run_card.json"),
        "m64_screen_report_sha256": m64.sha(report_path),
        "processor_identity": stats,
        "pinned_statistics": {"embodiment": embodiment, "use_percentiles": bool(processor.use_percentiles),
            "clip_outliers": bool(processor.clip_outliers), "use_relative_action": bool(processor.use_relative_action),
            "min_shape": list(min64.shape), "max_shape": list(max64.shape),
            "rotation_minmax_sha256": {"minimum": _hash_array(min64[:, 3:9]), "maximum": _hash_array(max64[:, 3:9])},
            "rotation_min_by_dim": min64[:, 3:9].min(axis=0).tolist(),
            "rotation_max_by_dim": max64[:, 3:9].max(axis=0).tolist(),
            "horizon_rows_used": 40},
        "cpu_decoder_parity": {"cells": len(screen["cells"]), "sources": len(source_set),
            "noise_groups": ["B0", "B1"], "matrix_max_abs": max_error,
            "matrix_max_abs_by_horizon": horizon_error.tolist(), "worst_cell": worst,
            "saved_official_decode_max_abs": 1e-5, "actual_normalized_rotation_clipped_values": clips,
            "nontrivial_anchor_cells": anchors,
            "relative_first_row_norm_min": row1_min, "relative_second_row_orthogonal_norm_min": row2_min},
        "self_check": check, "model_queries_added": 0,
    }
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to replace conflicting rotation CPU evidence: {output}")
    output.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output),
        "cells": len(screen["cells"]), "matrix_max_abs": max_error,
        "rotation_bounds_sha256": result["pinned_statistics"]["rotation_minmax_sha256"]}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run_cpu_check(args.output)


if __name__ == "__main__":
    main()
