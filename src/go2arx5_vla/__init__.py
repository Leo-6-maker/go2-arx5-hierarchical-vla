"""Canonical interfaces for the Go2+ARX5 hierarchical VLA project."""

from .contracts.trajectory import GripperCommandV1, TrajectoryChunkV1, WBCPreviewV1

__all__ = ["TrajectoryChunkV1", "WBCPreviewV1", "GripperCommandV1"]
