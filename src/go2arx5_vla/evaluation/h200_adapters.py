"""Versioned, allowlisted model-runtime adapter registry.

The generic execution engine imports this module without importing a model
framework.  A model package is imported only by the selected, exact registry
entry after static runtime admission has passed.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable, Mapping


class AdapterRegistryError(ValueError):
    """Raised when an adapter is not an approved versioned entry."""


@dataclass(frozen=True)
class AdapterSpec:
    """The only import information the generic engine is allowed to use."""

    adapter_id: str
    module: str
    factory: str
    version: str


_ALLOWLIST: dict[str, AdapterSpec] = {
    "groot_n1d7_structured_action_v1": AdapterSpec(
        adapter_id="groot_n1d7_structured_action_v1",
        module="go2arx5_vla.evaluation.groot_n1d7_adapter",
        factory="build_adapter",
        version="v1",
    ),
}


def registered_adapter_ids() -> tuple[str, ...]:
    """Return the immutable, deterministic adapter allowlist."""

    return tuple(_ALLOWLIST)


def adapter_spec(adapter_id: str) -> AdapterSpec:
    """Resolve an exact adapter ID without importing its implementation."""

    if type(adapter_id) is not str or not adapter_id.strip():
        raise AdapterRegistryError("runtime.adapter must be a non-empty exact ID")
    try:
        return _ALLOWLIST[adapter_id]
    except KeyError as exc:
        raise AdapterRegistryError(f"adapter is not allowlisted: {adapter_id}") from exc


def resolve_adapter(adapter_id: str, context: Any) -> Any:
    """Import and construct one approved adapter after admission only.

    ``importlib`` is intentionally confined to this function.  The module and
    factory names come only from ``_ALLOWLIST``; a manifest cannot provide an
    arbitrary import path.
    """

    spec = adapter_spec(adapter_id)
    module = importlib.import_module(spec.module)
    factory = getattr(module, spec.factory, None)
    if not callable(factory):
        raise AdapterRegistryError(
            f"allowlisted adapter factory is unavailable: {spec.adapter_id}"
        )
    adapter = factory(context)
    if adapter is None or not callable(getattr(adapter, "infer", None)):
        raise AdapterRegistryError(
            f"allowlisted adapter did not provide infer(): {spec.adapter_id}"
        )
    if getattr(adapter, "adapter_id", None) != spec.adapter_id:
        raise AdapterRegistryError(
            f"adapter identity is not the allowlisted ID: {spec.adapter_id}"
        )
    if getattr(adapter, "adapter_version", None) != spec.version:
        raise AdapterRegistryError(
            f"adapter version is not the allowlisted version: {spec.adapter_id}/{spec.version}"
        )
    return adapter


__all__ = [
    "AdapterRegistryError",
    "AdapterSpec",
    "adapter_spec",
    "registered_adapter_ids",
    "resolve_adapter",
]
