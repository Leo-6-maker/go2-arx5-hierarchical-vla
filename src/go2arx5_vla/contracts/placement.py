"""Embodiment-specific placement intent contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


PLACEMENT_INTENTS = (
    "CENTERED",
    "COMFORTABLE",
    "WORKSPACE_EDGE",
    "STATIC_BASE_RELOCATION_TARGET",
    "CONTINUOUS_BASE_COORDINATION_TARGET",
    "BOUNDARY",
    "EXPECTED_INFEASIBLE",
)


@dataclass(frozen=True)
class PlacementProfileV1:
    """A placement intent, not an embodiment-independent feasibility label."""

    profile_id: str
    intent: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    original_class: Optional[str] = None
    native_class: Optional[str] = None
    status: str = "UNCLASSIFIED_FOR_EMBODIMENT"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.intent not in PLACEMENT_INTENTS:
            raise ValueError(f"unknown placement intent: {self.intent}")
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "provenance", dict(self.provenance))

    def classified(self, native_class: str) -> "PlacementProfileV1":
        return PlacementProfileV1(
            profile_id=self.profile_id,
            intent=self.intent,
            parameters=self.parameters,
            original_class=self.original_class,
            native_class=native_class,
            status="CLASSIFIED_FOR_NATIVE_EMBODIMENT",
            provenance=self.provenance,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the placement intent without collapsing it into a shape label."""
        return {
            "schema": "PlacementProfileV1",
            "profile_id": self.profile_id,
            "intent": self.intent,
            "parameters": dict(self.parameters),
            "original_class": self.original_class,
            "native_class": self.native_class,
            "status": self.status,
            "provenance": dict(self.provenance),
        }
