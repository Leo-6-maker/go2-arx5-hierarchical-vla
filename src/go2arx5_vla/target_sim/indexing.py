"""Explicit DOMAIN_SIM indices; gathered state must be refreshed each step."""

from dataclasses import dataclass
from typing import Mapping, Sequence
from types import MappingProxyType

import numpy as np


def _indices(value, shape=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or (shape is not None and raw.shape != shape):
        raise ValueError("indices must be integers with the declared shape")
    result = raw.astype(np.int64, copy=True)
    if (result < 0).any() or np.unique(result).size != result.size:
        raise ValueError("indices must be nonnegative and unique")
    result.setflags(write=False)
    return result


def _device_indices(state, indices):
    if isinstance(state, np.ndarray):
        return indices
    import torch
    return torch.as_tensor(indices.copy(), dtype=torch.long, device=state.device)


@dataclass(frozen=True)
class TargetActorIndexMap:
    actor_indices: Mapping[str, Sequence[int]]
    robot_body_indices: np.ndarray
    robot_dof_indices: np.ndarray
    body_names: Sequence[str]
    dof_names: Sequence[str]
    feet_names: Sequence[str]
    tcp_name: str
    leg_dof_names: Sequence[str]
    arm_dof_names: Sequence[str]
    gripper_dof_names: Sequence[str]
    domain: str = "DOMAIN_SIM"

    def __post_init__(self):
        if self.domain != "DOMAIN_SIM":
            raise ValueError("only explicit DOMAIN_SIM indices are accepted")
        if set(self.actor_indices) != {"robot", "table", "object"}:
            raise ValueError("robot, table and object actor rows are required")
        actors = {name: _indices(rows) for name, rows in self.actor_indices.items()}
        n = actors["robot"].size
        if not n or any(rows.shape != (n,) for rows in actors.values()):
            raise ValueError("actor rows must be nonempty [num_envs]")
        _indices(np.concatenate(list(actors.values())))
        bodies = _indices(self.robot_body_indices, (n, len(self.body_names)))
        dofs = _indices(self.robot_dof_indices, (n, len(self.dof_names)))
        if not self.body_names or not self.dof_names:
            raise ValueError("body and DOF names are required")
        for names in (self.body_names, self.dof_names):
            if len(set(names)) != len(names):
                raise ValueError("duplicate body or DOF name")
        if not self.feet_names or any(name not in self.body_names for name in (*self.feet_names, self.tcp_name)):
            raise ValueError("feet/TCP must resolve to robot rigid bodies")
        groups = (*self.leg_dof_names, *self.arm_dof_names, *self.gripper_dof_names)
        if any(not group for group in (self.leg_dof_names, self.arm_dof_names, self.gripper_dof_names)):
            raise ValueError("leg, arm and gripper named DOFs are required")
        if len(set(groups)) != len(groups) or set(groups) != set(self.dof_names):
            raise ValueError("leg/arm/gripper groups must partition robot DOFs")
        object.__setattr__(self, "actor_indices", MappingProxyType(actors))
        object.__setattr__(self, "robot_body_indices", bodies)
        object.__setattr__(self, "robot_dof_indices", dofs)
        for name in ("body_names", "dof_names", "feet_names", "leg_dof_names", "arm_dof_names", "gripper_dof_names"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def num_envs(self):
        return self.actor_indices["robot"].size

    def env_ids(self, env_ids=None):
        ids = np.arange(self.num_envs) if env_ids is None else _indices(env_ids)
        if ids.ndim != 1 or (ids >= self.num_envs).any():
            raise ValueError("env_ids must be unique valid environment IDs")
        return ids

    def validate_counts(self, actor_count, body_count, dof_count):
        for rows, count in ((np.concatenate(list(self.actor_indices.values())), actor_count),
                            (self.robot_body_indices, body_count), (self.robot_dof_indices, dof_count)):
            if rows.max() >= count:
                raise ValueError("DOMAIN_SIM index exceeds global tensor size")

    def body_indices(self, names):
        return self.robot_body_indices[:, [self.body_names.index(name) for name in names]]

    def dof_indices(self, names):
        return self.robot_dof_indices[:, [self.dof_names.index(name) for name in names]]

    @property
    def feet_indices(self):
        return self.body_indices(self.feet_names)

    @property
    def tcp_indices(self):
        return self.body_indices((self.tcp_name,))[:, 0]

    @property
    def leg_indices(self):
        return self.dof_indices(self.leg_dof_names)

    @property
    def arm_indices(self):
        return self.dof_indices(self.arm_dof_names)

    @property
    def gripper_indices(self):
        return self.dof_indices(self.gripper_dof_names)

    def _rows(self, rows, env_ids):
        return rows[self.env_ids(env_ids)]

    def _gather(self, state, rows, env_ids):
        indices = self._rows(rows, env_ids)
        if indices.size and indices.max() >= state.shape[0]:
            raise ValueError("global tensor is smaller than DOMAIN_SIM indices")
        # ponytail: explicit fresh gathers; callers keep native buffers by copy_ each refresh.
        return state[_device_indices(state, indices)]

    def _scatter(self, state, rows, values, env_ids):
        indices = self._rows(rows, env_ids)
        expected = indices.shape + tuple(state.shape[1:])
        if tuple(values.shape) != expected:
            raise ValueError(f"scatter values must have shape {expected}")
        if indices.size and indices.max() >= state.shape[0]:
            raise ValueError("global tensor is smaller than DOMAIN_SIM indices")
        state[_device_indices(state, indices)] = values

    def gather_actor(self, state, actor="robot", env_ids=None):
        return self._gather(state, self.actor_indices[actor], env_ids)

    def gather_robot_bodies(self, state, env_ids=None):
        return self._gather(state, self.robot_body_indices, env_ids)

    def gather_robot_dofs(self, state, env_ids=None):
        return self._gather(state, self.robot_dof_indices, env_ids)

    def scatter_actor(self, state, values, actor="robot", env_ids=None):
        self._scatter(state, self.actor_indices[actor], values, env_ids)

    def scatter_robot_dofs(self, state, values, env_ids=None):
        self._scatter(state, self.robot_dof_indices, values, env_ids)
