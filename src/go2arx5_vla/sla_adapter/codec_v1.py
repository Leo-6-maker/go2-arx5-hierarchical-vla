"""EEF10 policy frame <-> episode-fixed mount frame <-> SLA EEF7.

T_policy_from_mount is obtained from the original post-hold native packet and
the calibrated robot model. No object state or extra TCP offset is used here.
The returned physical trajectory stays in the existing policy/consumer frame.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10

HORIZON = 40
LABEL_TIMES_S = np.arange(HORIZON, dtype=np.float64) * .05
TOLERANCES = dict(xyz_m=1e-6, so3_rad=1e-5, gripper=1e-6)


def rigid(value):
    a = np.array(value, dtype=np.float64, copy=True)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
            or not np.allclose(a[:3, :3] @ a[:3, :3].T, np.eye(3), rtol=0, atol=1e-8)
            or not np.isclose(np.linalg.det(a[:3, :3]), 1, rtol=0, atol=1e-8)):
        raise ValueError('MEASURED_CALIBRATED_RIGID_FRAME_REQUIRED')
    return a


def eef(value, shape):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('FINITE_CANONICAL_EEF10_SHAPE_REQUIRED')
    # The existing canonical codec orthogonalizes valid row-rot6d. Reject
    # degenerate rows before calling it; never turn NaNs into plausible poses.
    x, y = a[..., 3:6], a[..., 6:9]
    if (np.linalg.norm(x, axis=-1) < 1e-8).any() or (np.linalg.norm(np.cross(x, y), axis=-1) < 1e-8).any():
        raise ValueError('NONDEGENERATE_ROT6D_ROWS_REQUIRED')
    return a


def pad32(value):
    a = np.asarray(value)
    if a.shape[-1] != 7 or not np.isfinite(a).all():
        raise ValueError('SEVEN_FINITE_ACTIVE_DIMENSIONS_REQUIRED')
    return np.pad(a, [(0, 0)] * (a.ndim-1) + [(0, 25)])


@dataclass(frozen=True)
class Codec:
    T_policy_from_mount: object

    def __post_init__(self):
        a = rigid(self.T_policy_from_mount)
        a.setflags(write=False)
        object.__setattr__(self, 'T_policy_from_mount', a)

    def encode(self, state10, nominal_h40):
        state = eef(state10, (10,))
        action = eef(nominal_h40, (40, 10))
        if ((action[:, 9] < 0) | (action[:, 9] > 1)).any():
            raise ValueError('NOMINAL_GRIPPER_COMMAND_RANGE_REQUIRED')
        mount_from_policy = np.linalg.inv(self.T_policy_from_mount)
        s = mount_from_policy @ matrix_from10(state)
        a = mount_from_policy @ matrix_from10(action)
        state7 = np.r_[s[:3, 3], Rotation.from_matrix(s[:3, :3]).as_rotvec(), state[9]]
        delta = a[:, :3, :3] @ s[:3, :3].T
        action7 = np.c_[a[:, :3, 3]-s[:3, 3], Rotation.from_matrix(delta).as_rotvec(), action[:, 9]]
        return state7, action7

    def decode(self, state10, action32):
        state = eef(state10, (10,))
        a = np.asarray(action32, dtype=np.float64)
        if a.shape != (40, 32) or not np.isfinite(a).all():
            raise ValueError('FULL_SLA_H40_32_NO_TRUNCATION_REQUIRED')
        s = np.linalg.inv(self.T_policy_from_mount) @ matrix_from10(state)
        result = np.broadcast_to(np.eye(4), (40, 4, 4)).copy()
        result[:, :3, 3] = s[:3, 3] + a[:, :3]
        result[:, :3, :3] = Rotation.from_rotvec(a[:, 3:6]).as_matrix() @ s[:3, :3]
        # No clipping, joint repair or endpoint basis is hidden in this codec.
        return pose10(self.T_policy_from_mount @ result, a[:, 6])


def roundtrip_error(expected, actual):
    x, y = matrix_from10(expected), matrix_from10(actual)
    return dict(xyz_m=float(np.abs(x[..., :3, 3]-y[..., :3, 3]).max()),
                so3_rad=float(Rotation.from_matrix(x[..., :3, :3] @ np.swapaxes(y[..., :3, :3], -1, -2)).magnitude().max()),
                gripper=float(np.abs(np.asarray(expected)[..., 9]-np.asarray(actual)[..., 9]).max()))


@dataclass(frozen=True)
class Quantiles:
    q01: object
    q99: object

    def __post_init__(self):
        low, high = [np.array(v, dtype=np.float64, copy=True) for v in (self.q01, self.q99)]
        if low.shape != (7,) or high.shape != (7,) or not np.isfinite([low, high]).all() or (high < low).any():
            raise ValueError('TRAIN_ONLY_SEVEN_DIMENSION_QUANTILES_REQUIRED')
        low.setflags(write=False)
        high.setflags(write=False)
        object.__setattr__(self, 'q01', low)
        object.__setattr__(self, 'q99', high)

    @classmethod
    def fit(cls, values):
        a = np.asarray(values)
        if a.ndim != 2 or a.shape[1] != 7 or not len(a) or not np.isfinite(a).all():
            raise ValueError('NONEMPTY_FINITE_TRAIN_VALUES_REQUIRED')
        return cls(*np.quantile(a, [.01, .99], axis=0))

    def normalize(self, value):
        # Identical epsilon and no clipping, as pinned OpenPI quantile transform.
        return (np.asarray(value)-self.q01)/(self.q99-self.q01+1e-6)*2-1

    def unnormalize(self, value):
        return (np.asarray(value)+1)/2*(self.q99-self.q01+1e-6)+self.q01

    def summary(self, values):
        a = np.asarray(values)
        return dict(q01=self.q01.tolist(), q99=self.q99.tolist(), epsilon=1e-6,
                    constant_dimensions=np.flatnonzero(self.q01 == self.q99).tolist(),
                    outside_quantiles_fraction=((a < self.q01) | (a > self.q99)).mean(0).tolist(),
                    implicit_clip=False, fitted_active_dimensions=7, padding_after_normalization=25)
