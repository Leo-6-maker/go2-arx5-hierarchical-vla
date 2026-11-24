"""Map a rectified Twin render to the observed raster with an explicit Brown lens model."""

from dataclasses import dataclass

import numpy as np

from pinhole_camera import PinholeCamera


def brown(x, y, coefficients):
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
    rr = x*x + y*y
    numerator = 1 + rr*(k1 + rr*(k2 + rr*k3))
    denominator = 1 + rr*(k4 + rr*(k5 + rr*k6))
    scale = numerator / denominator
    slope = ((k1 + rr*(2*k2 + rr*3*k3)) - scale*(k4 + rr*(2*k5 + rr*3*k6))) / denominator
    xd = x*scale + 2*p1*x*y + p2*(rr + 2*x*x)
    yd = y*scale + p1*(rr + 2*y*y) + 2*p2*x*y
    xx = scale + 2*x*x*slope + 2*p1*y + 6*p2*x
    yy = scale + 2*y*y*slope + 6*p1*y + 2*p2*x
    xy = 2*x*y*slope + 2*p1*x + 2*p2*y
    valid = (denominator > 0) & (numerator > 0) & (xx > 0) & (yy > 0)
    return xd, yd, xx, yy, xy, valid


@dataclass(frozen=True)
class WarpedLayer:
    rgb: np.ndarray
    robot: np.ndarray
    valid: np.ndarray


class LensRemap:
    """A same-pose mapping from a rectified pinhole raster to the captured raster."""

    def __init__(self, rectified: PinholeCamera, captured: PinholeCamera, distortion):
        d = np.asarray(distortion, dtype=float)
        if d.shape not in ((4,), (5,), (8,)) or not np.isfinite(d).all():
            raise ValueError("expected OpenCV Brown coefficients in k1,k2,p1,p2[,k3[,k4,k5,k6]] order")
        d = np.pad(d, (0, 8-len(d)))
        self.rectified = rectified
        self.captured = captured
        v, u = np.indices((captured.height, captured.width), dtype=float)
        xd = (u - captured.cx) / captured.fx
        yd = (v - captured.cy) / captured.fy
        x, y = xd.copy(), yd.copy()
        valid = np.ones(x.shape, dtype=bool)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for _ in range(12):
                px, py, xx, yy, xy, positive = brown(x, y, d)
                det = xx*yy - xy*xy
                valid &= positive & (det > 0) & np.isfinite(det)
                safe = np.where(valid, det, 1.)
                ex, ey = px - xd, py - yd
                x = np.where(valid, x - (yy*ex - xy*ey) / safe, 0.)
                y = np.where(valid, y - (xx*ey - xy*ex) / safe, 0.)
            px, py, xx, yy, xy, positive = brown(x, y, d)
            error = np.hypot((px-xd)*captured.fx, (py-yd)*captured.fy)
            valid &= positive & (xx*yy - xy*xy > 0) & np.isfinite(error) & (error <= 1e-5)
        self.inverse_error_px = np.where(valid, error, np.nan)
        sx, sy = x * rectified.fx + rectified.cx, y * rectified.fy + rectified.cy
        # Preserve known integer correspondences across the normalized-ray calculation.
        sx = np.where(abs(sx-np.rint(sx)) <= 1e-10, np.rint(sx), sx)
        sy = np.where(abs(sy-np.rint(sy)) <= 1e-10, np.rint(sy), sy)
        self.valid = valid & (sx >= 0) & (sx <= rectified.width-1) & (sy >= 0) & (sy <= rectified.height-1)
        self.x = np.where(self.valid, sx, 0.)
        self.y = np.where(self.valid, sy, 0.)
        if not self.valid.any():
            raise ValueError("capture rays have no valid rectified support")

    def sample(self, image):
        """Bilinear sampling in the rectified raster. Return the four donor weights as well."""
        image = np.asarray(image)
        if image.shape[:2] != (self.rectified.height, self.rectified.width):
            raise ValueError("rectified input raster does not match its intrinsics")
        x0, y0 = np.floor(self.x).astype(np.intp), np.floor(self.y).astype(np.intp)
        x1, y1 = np.minimum(x0+1, self.rectified.width-1), np.minimum(y0+1, self.rectified.height-1)
        dx, dy = self.x-x0, self.y-y0
        return ((image[y0, x0], (1-dx)*(1-dy)), (image[y0, x1], dx*(1-dy)),
                (image[y1, x0], (1-dx)*dy), (image[y1, x1], dx*dy))

    def warp_rgb(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.shape != (self.rectified.height, self.rectified.width, 3) or rgb.dtype != np.uint8:
            raise ValueError("expected rectified uint8 RGB matching the lens map")
        interpolated = sum(color * weight[..., None] for color, weight in self.sample(rgb))
        return np.where(self.valid[..., None], np.rint(interpolated).clip(0, 255).astype(np.uint8), 0)

    def robot_layer(self, rgb, mask):
        mask = np.asarray(mask)
        if mask.shape != (self.rectified.height, self.rectified.width) or mask.dtype != bool:
            raise ValueError("expected rectified boolean robot mask matching the lens map")
        taps = self.sample(mask)
        all_robot = np.logical_and.reduce([value | (weight == 0) for value, weight in taps])
        all_background = np.logical_and.reduce([~value | (weight == 0) for value, weight in taps])
        valid = self.valid & (all_robot | all_background)
        return WarpedLayer(self.warp_rgb(rgb), self.valid & all_robot, valid)
