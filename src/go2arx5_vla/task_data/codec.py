from __future__ import annotations

import numpy as np
from ..trajectory.so3 import rotation_matrix_to_rot6d_rows,rot6d_rows_to_matrix


def pose10(pose,opening):
    pose=np.asarray(pose,dtype=np.float64)
    return np.r_[pose[:3,3],rotation_matrix_to_rot6d_rows(pose[:3,:3]),float(opening)]


def decode(action,anchor):
    action=np.asarray(action,dtype=np.float64)
    T=np.eye(4);T[:3,3]=action[:3];T[:3,:3]=rot6d_rows_to_matrix(action[3:9])
    return np.asarray(anchor)@T,float(action[9])


def relative_action(targets,anchor):
    inverse=np.linalg.inv(np.asarray(anchor,dtype=np.float64))
    return np.stack([pose10(inverse@np.asarray(pose),opening) for pose,opening in targets])
