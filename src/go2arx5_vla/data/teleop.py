"""Robot-independent desired task-space command contract for future pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .readiness import GROOT_EEF_ROT6D_LAYOUT, _finite, _shape


@dataclass(frozen=True)
class TeleopCommandV1:
    """A desired EEF+gripper command; this is a label, not an actuation API."""

    command_id: str
    timestamp: float
    eef_9d: np.ndarray
    gripper: np.ndarray
    frame: str
    tcp_semantic_id: str
    source: str
    rot6d_layout: str = GROOT_EEF_ROT6D_LAYOUT

    def __post_init__(self) -> None:
        if not self.command_id or not self.frame or not self.tcp_semantic_id or not self.source:
            raise ValueError("command ID, frame, TCP ID, and source are required")
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("command timestamp must be finite")
        eef = _shape("TeleopCommandV1.eef_9d", self.eef_9d, (9,))
        gripper = _shape("TeleopCommandV1.gripper", self.gripper, (1,))
        _finite("TeleopCommandV1.eef_9d", eef)
        _finite("TeleopCommandV1.gripper", gripper)
        if self.rot6d_layout != GROOT_EEF_ROT6D_LAYOUT:
            raise ValueError(f"TeleopCommandV1 requires {GROOT_EEF_ROT6D_LAYOUT}")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "eef_9d", eef)
        object.__setattr__(self, "gripper", gripper)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "TeleopCommandV1",
            "command_id": self.command_id,
            "timestamp": self.timestamp,
            "eef_9d": self.eef_9d.tolist(),
            "gripper": self.gripper.tolist(),
            "frame": self.frame,
            "tcp_semantic_id": self.tcp_semantic_id,
            "source": self.source,
            "rot6d_layout": self.rot6d_layout,
        }
