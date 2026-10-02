"""CPU-only constructed checks, not a replay of PI0267 data or learned models.

1. A reversible unclamped affine codec may have a very large forward gain.
2. PI0267's six +/- paired translations and rotations span three local
   pose-perturbation directions, not six independent SE(3) directions.

The toy rotation bounds below are deliberately constructed from yaw-only
rotations. They are NOT the project's measured statistics. NumPy is the only
non-standard dependency. No network, renderer, torch, or model loader is used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def rotation(axis: int, angle: float) -> np.ndarray:
    if axis not in (0, 1, 2) or not np.isfinite(angle):
        raise ValueError('A finite angle and axis 0, 1, or 2 are required.')
    v = np.eye(3)[axis]
    k = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    return np.eye(3) + np.sin(angle)*k + (1-np.cos(angle))*(k@k)


def main() -> dict:
    # Constructed yaw-only bounds; this is NOT the fixed TRAIN statistics.
    lo = np.array([np.cos(.3), -np.sin(.3), 0., -np.sin(.3), np.cos(.3), 0.])
    hi = np.array([1., np.sin(.3), 0., np.sin(.3), 1., 0.])
    width = np.maximum(hi-lo, 1e-6)
    rows = []
    for axis in range(3):
        for sign in (1, -1):
            raw = rotation(axis, sign*.060)[:2, :].reshape(6)
            encoded = 2*(raw-lo)/width-1
            stored = encoded.astype(np.float32)
            decoded = (stored.astype(np.float64)+1)*width/2+lo
            error = float(np.max(np.abs(decoded-raw)))
            assert error < 1e-7
            rows.append(dict(axis='xyz'[axis], sign=sign,
                             raw_max_abs=float(np.max(np.abs(raw))),
                             normalized_max_abs=float(np.max(np.abs(encoded))),
                             float32_roundtrip_max_component_error=error))
    coupled = []
    independent = []
    for axis in range(3):
        e = np.eye(3)[axis]
        for sign in (1, -1):
            coupled.append(np.r_[.010*sign*e, .060*sign*e])
            independent.append(np.r_[.010*sign*e, np.zeros(3)])
            independent.append(np.r_[np.zeros(3), .060*sign*e])
    # Unit-normalized local displacement: dp/.01 and Log(R'R^T)/.06.
    scale = np.array([.01]*3+[.06]*3)
    coupled = np.asarray(coupled)/scale
    independent = np.asarray(independent)/scale
    rank_c = int(np.linalg.matrix_rank(coupled))
    rank_i = int(np.linalg.matrix_rank(independent))
    assert rank_c == 3 and rank_i == 6
    # Correct physical states and labels are still necessary; rank alone does
    # not establish valid recovery examples or learned recovery capability.
    return dict(
        status='PASS_CONSTRUCTED_CPU_FIXTURES',
        scope='No project statistics, original samples, model, renderer, WBC, or training replay.',
        normalization=dict(rule='2*(x-lo)/max(hi-lo,1e-6)-1',
            toy_bounds='Constructed yaw-only rotations with yaw range +/-0.3 rad; NOT measured project bounds.',
            floor_forward_gain=2e6, rows=rows,
            interpretation='Accurate inverse roundtrip does not establish a well-scaled learning problem.'),
        local_pose_support=dict(coupled_six_state_rank=rank_c,
            independent_twelve_state_rank=rank_i,
            coupled_singular_values=np.linalg.svd(coupled, compute_uv=False).tolist(),
            caution='Rank is per-anchor perturbation support, not the rank of the whole corpus. It does not by itself explain R failures on the combined variants.'),
        actual_model_queries=0, optimizer_updates=0, policy_frame_pairs=0,
        physics_starts=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    result = main()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
