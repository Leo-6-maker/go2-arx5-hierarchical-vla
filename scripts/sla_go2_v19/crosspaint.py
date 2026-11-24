"""Same-view robot appearance transfer within an observed support mask."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion


@dataclass(frozen=True)
class PaintResult:
    image: np.ndarray
    replaced: np.ndarray
    unresolved: np.ndarray


def replace_appearance(observed, target_rgb, source_rgb, target_mask, source_mask):
    """Use only robot renders in the observed camera. Preserve all other bytes."""
    if any(
        x.shape != observed.shape or x.ndim != 3 or x.shape[-1] != 3 or x.dtype != np.uint8
        for x in (observed, target_rgb, source_rgb)
    ) or any(x.shape != observed.shape[:2] or x.dtype != bool for x in (target_mask, source_mask)):
        raise ValueError("RGB and robot masks must share one calibrated raster")
    overlap = binary_erosion(target_mask & source_mask, structure=np.ones((3, 3), dtype=bool))
    looks_like_robot = (observed == target_rgb).all(axis=-1)
    supported = overlap & looks_like_robot
    image = np.where(supported[..., None], source_rgb, observed)
    return PaintResult(image, supported, (target_mask | source_mask) & ~supported)
