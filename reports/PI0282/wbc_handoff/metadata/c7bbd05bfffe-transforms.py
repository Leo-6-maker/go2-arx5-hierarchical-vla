"""Minimal PyTorch3D transform compatibility for the frozen WBC runtime."""

from __future__ import annotations

import torch


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Match pytorch3d.transforms.quaternion_to_matrix (w, x, y, z)."""
    r, i, j, k = torch.unbind(quaternions, dim=-1)
    two_s = 2.0 / (quaternions * quaternions).sum(dim=-1)
    return torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    ).reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Match pytorch3d.transforms.matrix_to_rotation_6d flattening."""
    return matrix[..., :, :2].clone().reshape(matrix.shape[:-2] + (6,))
