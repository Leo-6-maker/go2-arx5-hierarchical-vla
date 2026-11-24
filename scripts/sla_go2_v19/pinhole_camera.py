"""Render a calibrated rectified raster using the camera's focal lengths and principal point."""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if (min(self.width, self.height) <= 1 or not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all()
                or min(self.fx, self.fy) <= 0):
            raise ValueError("invalid pinhole intrinsics")

    @classmethod
    def centered(cls, width, height, fovy_deg):
        if not 0 < fovy_deg < 180:
            raise ValueError("invalid camera field of view")
        f = height / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
        return cls(width, height, f, f, (width - 1) / 2, (height - 1) / 2)

    @contextmanager
    def bind(self, model, view, renderer):
        """Bind only the temporary render camera. Pixel centers use OpenCV coordinates."""
        if (renderer.width, renderer.height) != (self.width, self.height):
            raise ValueError("calibrated projection and render raster differ")
        index = model.camera(view).id
        if model.cam_orthographic[index]:
            raise ValueError("pinhole projection requires a perspective camera")
        size, intrinsic, resolution = (model.cam_sensorsize[index], model.cam_intrinsic[index],
                                       model.cam_resolution[index])
        original = (size.copy(), intrinsic.copy(), resolution.copy())
        # MuJoCo uses centered physical sensor coordinates (+x right, +y up).
        # OpenCV indexes pixel centers from the top left. A common scale
        # cancels from the camera-ray equations.
        scale = 1e-5
        try:
            resolution[:] = [self.width, self.height]
            size[:] = scale * np.array([self.width, self.height])
            intrinsic[:] = scale * np.array([
                self.fx, self.fy, self.width / 2 - self.cx - .5, self.cy + .5 - self.height / 2,
            ])
            yield
        finally:
            size[:], intrinsic[:], resolution[:] = original
