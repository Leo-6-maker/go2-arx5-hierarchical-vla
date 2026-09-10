"""Immutable, admission-owned inputs for the allowlisted H200 adapter.

The generic engine verifies the ordinary manifest artifacts, while an adapter
may need a few additional materialized inputs to construct its processor.  A
``AdapterRuntimeBindingV1`` is the single handoff for those values.  It is
constructed from verified artifacts, copied into the admission document, and
then frozen before the adapter is resolved or imported.

This module is deliberately model-framework free.  It may therefore be used
by CPU/static admission tests without importing Torch, GR00T, Transformers, or
CUDA.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..experiments.reusable import (
    ReusableManifestError,
    require_mapping,
    verify_artifact_identity,
)


ADAPTER_RUNTIME_BINDING_SCHEMA = (
    "GO2ARX5_REUSABLE_ADAPTER_RUNTIME_BINDING_V1"
)
ADAPTER_RUNTIME_BINDING_VERSION = "v1"
ADAPTER_RUNTIME_BINDING_FIELDS = (
    "schema",
    "adapter_id",
    "adapter_version",
    "immutable",
    "dataset",
    "groot_source",
    "cosmos_cache",
    "modality_config",
    "processor_checkpoint",
    "base_checkpoint",
    "cuda_device",
    "checkpoint",
    "statistics",
    "noise_bank",
    "language",
)
_OPTIONAL_FIELDS = ("transformers_cache",)


class AdapterRuntimeBindingError(ReusableManifestError):
    """Raised when adapter inputs cannot be bound to admitted identities."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterRuntimeBindingError(message)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    try:
        return require_mapping(value, field=field)
    except ReusableManifestError as exc:
        raise AdapterRuntimeBindingError(str(exc)) from exc


def _string(value: Any, *, field: str) -> str:
    _require(
        type(value) is str and bool(value.strip()),
        f"{field} must be a non-empty JSON string",
    )
    return value.strip()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


@dataclass(frozen=True)
class AdapterRuntimeBindingV1(Mapping[str, Any]):
    """Read-only versioned adapter handoff carried by admission."""

    document: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "document", _freeze(self.document))

    def __getitem__(self, key: str) -> Any:
        return self.document[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.document)

    def __len__(self) -> int:
        return len(self.document)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible detached copy for evidence writing."""

        return _thaw(self.document)


def _public_artifact(actual: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(actual["resolved_path"]),
        "sha256": str(actual["sha256"]),
        "kind": str(actual.get("kind", "file")),
    }


def _verify_artifact(
    value: Any, *, base_dir: Path, field: str
) -> dict[str, Any]:
    try:
        actual = verify_artifact_identity(
            _mapping(value, field=field), base_dir=base_dir, field=field
        )
    except ReusableManifestError as exc:
        raise AdapterRuntimeBindingError(str(exc)) from exc
    return _public_artifact(actual)


def _same_artifact(
    left: Mapping[str, Any], right: Mapping[str, Any], *, field: str
) -> None:
    _require(
        str(left.get("kind", "file")) == str(right.get("kind", "file")),
        f"{field}.kind differs from the admitted identity",
    )
    _require(
        str(left.get("path")) == str(right.get("path")),
        f"{field}.path differs from the admitted identity",
    )
    _require(
        str(left.get("sha256")) == str(right.get("sha256")),
        f"{field}.sha256 differs from the admitted identity",
    )


def _bound_artifact(
    candidate: Any,
    expected: Any,
    *,
    base_dir: Path,
    field: str,
) -> dict[str, Any]:
    actual = _verify_artifact(candidate, base_dir=base_dir, field=field)
    expected_actual = _verify_artifact(
        expected, base_dir=base_dir, field=f"{field}.admitted"
    )
    _same_artifact(actual, expected_actual, field=field)
    return actual


def _artifact_map(
    candidate: Any,
    expected: Any,
    *,
    labels: list[str],
    base_dir: Path,
    field: str,
) -> dict[str, dict[str, Any]]:
    candidate_map = {
        str(key): value for key, value in _mapping(candidate, field=field).items()
    }
    expected_map = {
        str(key): value
        for key, value in _mapping(expected, field=f"{field}.admitted").items()
    }
    _require(
        set(candidate_map) == set(labels),
        f"{field} must contain exactly the manifest checkpoint labels",
    )
    _require(
        set(expected_map) == set(labels),
        f"{field}.admitted must contain exactly the manifest checkpoint labels",
    )
    result: dict[str, dict[str, Any]] = {}
    for label in labels:
        result[label] = _bound_artifact(
            candidate_map[label],
            expected_map[label],
            base_dir=base_dir,
            field=f"{field}.{label}",
        )
    return result


def _validate_shape(
    binding: Mapping[str, Any], *, adapter_id: str, adapter_version: str
) -> None:
    _require(
        binding.get("schema") == ADAPTER_RUNTIME_BINDING_SCHEMA,
        "adapter runtime binding schema mismatch",
    )
    _require(
        binding.get("adapter_id") == adapter_id,
        "adapter runtime binding adapter_id differs from the manifest",
    )
    _require(
        binding.get("adapter_version") == adapter_version,
        "adapter runtime binding adapter_version is not the allowlisted version",
    )
    _require(
        binding.get("immutable") is True,
        "adapter runtime binding must be explicitly immutable",
    )
    allowed = set(ADAPTER_RUNTIME_BINDING_FIELDS) | set(_OPTIONAL_FIELDS)
    _require(
        set(binding).issubset(allowed),
        "adapter runtime binding contains an unversioned field",
    )
    missing = [field for field in ADAPTER_RUNTIME_BINDING_FIELDS if field not in binding]
    _require(not missing, f"adapter runtime binding is missing: {', '.join(missing)}")


def _validate_binding_against_runtime(
    binding: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    runtime: Mapping[str, Any],
    checkpoint_expected: Mapping[str, Any],
    statistics_expected: Mapping[str, Any],
    noise_expected: Mapping[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    adapter_id = _string(manifest["runtime"].get("adapter"), field="manifest.runtime.adapter")
    _validate_shape(binding, adapter_id=adapter_id, adapter_version=ADAPTER_RUNTIME_BINDING_VERSION)
    labels = [str(item["label"]) for item in manifest["checkpoints"]]

    canonical: dict[str, Any] = {
        "schema": ADAPTER_RUNTIME_BINDING_SCHEMA,
        "adapter_id": adapter_id,
        "adapter_version": ADAPTER_RUNTIME_BINDING_VERSION,
        "immutable": True,
    }
    canonical["dataset"] = _bound_artifact(
        binding["dataset"],
        runtime["dataset_path"],
        base_dir=base_dir,
        field="adapter_runtime_binding.dataset",
    )
    canonical["groot_source"] = _bound_artifact(
        binding["groot_source"],
        runtime["groot_source"],
        base_dir=base_dir,
        field="adapter_runtime_binding.groot_source",
    )
    canonical["cosmos_cache"] = _bound_artifact(
        binding["cosmos_cache"],
        runtime["cosmos_cache_root"],
        base_dir=base_dir,
        field="adapter_runtime_binding.cosmos_cache",
    )
    for field in ("modality_config", "processor_checkpoint", "base_checkpoint"):
        canonical[field] = _verify_artifact(
            binding[field], base_dir=base_dir, field=f"adapter_runtime_binding.{field}"
        )

    device = _string(
        binding["cuda_device"], field="adapter_runtime_binding.cuda_device"
    )
    _require(
        device == _string(runtime["gpu_device"], field="runtime.gpu_device"),
        "adapter CUDA device differs from the admitted GPU device",
    )
    canonical["cuda_device"] = device

    language = _string(binding["language"], field="adapter_runtime_binding.language")
    contract = _mapping(manifest.get("contract"), field="manifest.contract")
    _require(
        language == _string(contract.get("language"), field="manifest.contract.language"),
        "adapter language differs from the manifest contract",
    )
    canonical["language"] = language
    adapter_input_identities = _mapping(
        _mapping(manifest.get("runtime"), field="manifest.runtime").get(
            "adapter_input_identities"
        ),
        field="manifest.runtime.adapter_input_identities",
    )
    for field in ("modality_config", "processor_checkpoint", "base_checkpoint"):
        _require(
            field in adapter_input_identities,
            f"manifest.runtime.adapter_input_identities.{field} is required",
        )
        _same_artifact(
            canonical[field],
            _verify_artifact(
                adapter_input_identities[field],
                base_dir=base_dir,
                field=f"manifest.runtime.adapter_input_identities.{field}",
            ),
            field=f"adapter_runtime_binding.{field}",
        )
    canonical["checkpoint"] = _artifact_map(
        binding["checkpoint"],
        checkpoint_expected,
        labels=labels,
        base_dir=base_dir,
        field="adapter_runtime_binding.checkpoint",
    )
    canonical["statistics"] = _artifact_map(
        binding["statistics"],
        statistics_expected,
        labels=labels,
        base_dir=base_dir,
        field="adapter_runtime_binding.statistics",
    )
    canonical["noise_bank"] = _bound_artifact(
        binding["noise_bank"],
        noise_expected,
        base_dir=base_dir,
        field="adapter_runtime_binding.noise_bank",
    )
    if "transformers_cache" in binding:
        canonical["transformers_cache"] = _verify_artifact(
            binding["transformers_cache"],
            base_dir=base_dir,
            field="adapter_runtime_binding.transformers_cache",
        )
    return canonical


def build_adapter_runtime_binding(
    manifest: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    *,
    observed_runtime: Mapping[str, Any],
    observed_sampler: Mapping[str, Any],
    verified_artifacts: Mapping[str, Any],
    manifest_path: str | Path,
) -> AdapterRuntimeBindingV1:
    """Materialize and freeze the adapter binding from verified identities."""

    base_dir = Path(manifest_path).expanduser().resolve().parent
    raw = _mapping(
        runtime_config.get("adapter_runtime_binding"),
        field="runtime_config.adapter_runtime_binding",
    )
    verified_checkpoints = {
        str(item["label"]): item["identity"]
        for item in verified_artifacts["checkpoints"]
    }
    verified_statistics = {
        str(item["label"]): item["statistics"]
        for item in verified_artifacts["checkpoints"]
    }
    canonical = _validate_binding_against_runtime(
        raw,
        manifest=manifest,
        runtime=observed_runtime,
        checkpoint_expected=verified_checkpoints,
        statistics_expected=verified_statistics,
        noise_expected=observed_sampler["noise_bank"],
        base_dir=base_dir,
    )
    return AdapterRuntimeBindingV1(canonical)


def validate_adapter_runtime_binding(
    manifest: Mapping[str, Any],
    admission: Mapping[str, Any],
    *,
    manifest_path: str | Path,
) -> AdapterRuntimeBindingV1:
    """Re-validate the materialized binding after H200 admission passes."""

    base_dir = Path(manifest_path).expanduser().resolve().parent
    binding = _mapping(
        admission.get("adapter_runtime_binding"),
        field="admission.adapter_runtime_binding",
    )
    runtime = _mapping(
        admission.get("runtime_late_bound"), field="admission.runtime_late_bound"
    )
    checkpoint_identities = _mapping(
        {str(item["label"]): item["identity"] for item in admission.get("checkpoint_identities", [])},
        field="admission.checkpoint_identities",
    )
    statistics = _mapping(
        {str(item["label"]): item["statistics"] for item in admission.get("checkpoint_identities", [])},
        field="admission.checkpoint_statistics",
    )
    sampler_value = admission.get("sampler")
    if isinstance(sampler_value, Mapping):
        noise = sampler_value.get("noise_bank", admission.get("noise_bank"))
    else:
        # Older preflight documents exposed the verified noise identity at the
        # top level.  Keep this fallback only for a detached admission object;
        # the current validator always emits the complete sampler mapping.
        noise = admission.get("noise_bank")
    _require(noise is not None, "admission must carry the verified noise identity")
    canonical = _validate_binding_against_runtime(
        binding,
        manifest=manifest,
        runtime=runtime,
        checkpoint_expected=checkpoint_identities,
        statistics_expected=statistics,
        noise_expected=noise,
        base_dir=base_dir,
    )
    return AdapterRuntimeBindingV1(canonical)


__all__ = [
    "ADAPTER_RUNTIME_BINDING_FIELDS",
    "ADAPTER_RUNTIME_BINDING_SCHEMA",
    "ADAPTER_RUNTIME_BINDING_VERSION",
    "AdapterRuntimeBindingError",
    "AdapterRuntimeBindingV1",
    "build_adapter_runtime_binding",
    "validate_adapter_runtime_binding",
]
