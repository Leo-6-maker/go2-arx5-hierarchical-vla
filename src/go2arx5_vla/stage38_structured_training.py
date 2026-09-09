"""Stage38-D structured-action adapter for the pinned GR00T training path.

The accepted Stage38 contract changes only the model-facing action labels.  The
official GR00T processor is still used for observations, language, and state;
this module replaces the processor-produced dense action tensor *in memory*
with a normalized :class:`StructuredActionV1` tensor and its frozen mask.

This module is imported by the Stage38-D launcher only after the CPU A/B/C
qualification.  Imports of pandas, torch, and GR00T therefore remain inside
the runtime factory.  In particular, the custom factory deliberately does not
call GR00T's ``generate_stats`` helpers: those helpers write sidecar files into
the immutable TINY1 source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .structured_action import (
    DEFAULT_MOTION_INTENT_V2_FRAME,
    DEFAULT_MOTION_INTENT_V2_TCP,
    STRUCTURED_ACTION_DIM,
    STRUCTURED_ACTION_HORIZON,
    STRUCTURED_ACTIVE_DENSE_DIM,
    STRUCTURED_TIMING_MODE,
    StructuredActionV1,
    encode_structured_action,
    reconcile_stage45b_label_anchor,
    structured_action_contract_summary,
    structured_action_mask,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _as_float_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise ValueError(f"{name} must contain exactly one finite scalar")
    return array


def _column_matrix(frame: Any, column: str, indices: Sequence[int], *, name: str) -> np.ndarray:
    if column not in frame.columns:
        raise KeyError(f"{column!r} is absent from the training frame")
    rows = []
    for index in indices:
        value = np.asarray(frame[column].iloc[int(index)], dtype=np.float64).reshape(-1)
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values at row {index}")
        rows.append(value)
    if not rows:
        raise ValueError(f"{name} cannot be empty")
    width = rows[0].size
    if any(row.size != width for row in rows):
        raise ValueError(f"{name} has inconsistent row widths")
    return np.stack(rows, axis=0)


def _timestamp_vector(frame: Any, *, name: str) -> np.ndarray:
    if "timestamp" not in frame.columns:
        raise KeyError("the authoritative parquet timestamp column is missing")
    values = []
    for row in frame["timestamp"].tolist():
        values.append(float(_as_float_vector(row, name=name)[0]))
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size < STRUCTURED_ACTION_HORIZON:
        raise ValueError(
            f"{name} must contain at least {STRUCTURED_ACTION_HORIZON} timestamps, got {result.size}"
        )
    if not np.isfinite(result).all() or np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return result


def _structured_stats(labels: Sequence[StructuredActionV1], *, use_percentiles: bool) -> dict[str, Any]:
    """Compute action statistics from training labels only.

    Statistics are per universal action dimension.  Samples that are masked
    inactive for a dimension are excluded from that dimension's population;
    padding therefore cannot influence normalization.  This mirrors GR00T's
    min/max-to-[-1,1] convention while keeping the representation-specific
    statistics separate from the historical dense statistics.
    """

    if not labels:
        raise ValueError("at least one structured training label is required")
    values = np.stack([label.values for label in labels], axis=0)
    masks = np.stack([label.mask for label in labels], axis=0).astype(bool)
    if values.shape[1:] != (STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTION_DIM):
        raise ValueError(f"unexpected structured label shape: {values.shape}")

    fields: dict[str, list[float]] = {}
    for name in ("min", "max", "mean", "std", "q01", "q99"):
        fields[name] = []
    active_counts: list[int] = []
    for dim in range(STRUCTURED_ACTION_DIM):
        observed = values[:, :, dim][masks[:, :, dim]]
        if observed.size == 0:
            observed = np.asarray([0.0], dtype=np.float64)
        if not np.isfinite(observed).all():
            raise ValueError(f"structured training statistics contain non-finite dim {dim}")
        fields["min"].append(float(np.min(observed)))
        fields["max"].append(float(np.max(observed)))
        fields["mean"].append(float(np.mean(observed)))
        fields["std"].append(float(np.std(observed, ddof=0)))
        fields["q01"].append(float(np.percentile(observed, 1.0)))
        fields["q99"].append(float(np.percentile(observed, 99.0)))
        active_counts.append(int(observed.size))

    lower = np.asarray(fields["q01"] if use_percentiles else fields["min"], dtype=np.float64)
    upper = np.asarray(fields["q99"] if use_percentiles else fields["max"], dtype=np.float64)
    return {
        "schema": "STAGE38_STRUCTURED_ACTION_STATISTICS_V1",
        "normalization": "MINMAX_TO_NEG1_POS1",
        "use_percentiles": bool(use_percentiles),
        "selected_lower_bound": "q01" if use_percentiles else "min",
        "selected_upper_bound": "q99" if use_percentiles else "max",
        "timing_mode": STRUCTURED_TIMING_MODE,
        "action_horizon": STRUCTURED_ACTION_HORIZON,
        "universal_action_dim": STRUCTURED_ACTION_DIM,
        "active_mask": structured_action_mask(STRUCTURED_TIMING_MODE).tolist(),
        "active_scalar_count": int(np.count_nonzero(structured_action_mask(STRUCTURED_TIMING_MODE))),
        "label_count": int(len(labels)),
        "active_observation_count_by_dim": active_counts,
        "statistics": fields,
        "selected_lower": lower.tolist(),
        "selected_upper": upper.tolist(),
        "source": "STAGE38_D_TRAINING_DATA_ONLY",
    }


@dataclass(frozen=True)
class StructuredActionNormalizer:
    """Representation-specific min/max normalizer used at the model boundary."""

    metadata: Mapping[str, Any]
    clip_outliers: bool = True

    def __post_init__(self) -> None:
        lower = np.asarray(self.metadata["selected_lower"], dtype=np.float64)
        upper = np.asarray(self.metadata["selected_upper"], dtype=np.float64)
        if lower.shape != (STRUCTURED_ACTION_DIM,) or upper.shape != (STRUCTURED_ACTION_DIM,):
            raise ValueError("structured normalization bounds must have 132 dimensions")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("structured normalization bounds must be finite")
        if np.any(upper < lower):
            raise ValueError("structured normalization upper bounds must not be below lower bounds")
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(self, "_upper", upper)

    def normalize(self, label: StructuredActionV1) -> np.ndarray:
        values = np.asarray(label.values, dtype=np.float64)
        if values.shape != (STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTION_DIM):
            raise ValueError(f"structured label shape mismatch: {values.shape}")
        normalized = np.zeros(values.shape, dtype=np.float64)
        varying = ~np.isclose(self._upper, self._lower)
        if np.any(varying):
            normalized[:, varying] = 2.0 * (
                values[:, varying] - self._lower[varying]
            ) / (self._upper[varying] - self._lower[varying]) - 1.0
        if self.clip_outliers:
            normalized = np.clip(normalized, -1.0, 1.0)
        normalized[np.asarray(label.mask) == 0.0] = 0.0
        if not np.isfinite(normalized).all():
            raise ValueError("normalized structured label contains non-finite values")
        return normalized.astype(np.float32, copy=False)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.metadata)


def build_structured_gr00t_dataset_factory():
    """Return a GR00T-compatible dataset factory with on-the-fly labels.

    The imports are intentionally local.  Importing this project package on a
    login node must remain possible without the pinned GR00T runtime.
    """

    from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
    from gr00t.data.dataset.sharded_single_step_dataset import (
        ShardedSingleStepDataset,
        extract_step_data,
    )
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType

    class StructuredShardedSingleStepDataset(ShardedSingleStepDataset):
        """Official shard layout plus a read-only timestamp/label side channel."""

        def __init__(
            self,
            *args: Any,
            timing_mode: str,
            use_percentiles: bool,
            stage45b_label_reconciliation: bool = False,
            **kwargs: Any,
        ):
            modality_configs = kwargs.get("modality_configs")
            if modality_configs is None and len(args) >= 3:
                modality_configs = args[2]
            if modality_configs is None:
                raise ValueError("modality_configs are required")
            action_indices = list(modality_configs["action"].delta_indices)
            expected_indices = list(range(STRUCTURED_ACTION_HORIZON))
            if action_indices != expected_indices:
                raise ValueError(
                    "Stage38-D requires the frozen dense H40 action indices "
                    f"{expected_indices}, got {action_indices}"
                )
            if timing_mode != STRUCTURED_TIMING_MODE:
                raise ValueError(f"Stage38-D timing mode is fixed to {STRUCTURED_TIMING_MODE}")
            self.timing_mode = timing_mode
            self.use_percentiles = bool(use_percentiles)
            self.stage45b_label_reconciliation = bool(stage45b_label_reconciliation)
            self._timestamp_cache: dict[int, np.ndarray] = {}
            self._label_cache: dict[tuple[int, int], StructuredActionV1] = {}
            super().__init__(*args, **kwargs)
            labels = self._collect_training_labels()
            self.structured_action_labels = labels
            self.structured_action_statistics = _structured_stats(
                labels, use_percentiles=self.use_percentiles
            )
            self.structured_action_normalizer = StructuredActionNormalizer(
                self.structured_action_statistics,
                clip_outliers=True,
            )

        def _episode_id(self, episode_index: int) -> int:
            return int(self.episode_loader.episodes_metadata[int(episode_index)]["episode_index"])

        def _timestamps(self, episode_index: int) -> np.ndarray:
            episode_index = int(episode_index)
            if episode_index in self._timestamp_cache:
                return self._timestamp_cache[episode_index].copy()
            # pandas is available in the pinned training runtime.  Read only
            # the authoritative timestamp column from the original parquet.
            import pandas as pd

            episode_id = self._episode_id(episode_index)
            filename = self.episode_loader.data_path_pattern.format(
                episode_chunk=episode_id // self.episode_loader.chunk_size,
                episode_index=episode_id,
            )
            path = Path(self.dataset_path) / filename
            frame = pd.read_parquet(path, columns=["timestamp"])
            timestamps = _timestamp_vector(
                frame, name=f"{self.dataset_path}:episode-{episode_id}:timestamp"
            )
            expected_length = self.episode_loader.get_episode_length(episode_index)
            if timestamps.size != expected_length:
                raise ValueError(
                    f"timestamp length mismatch for episode {episode_id}: "
                    f"{timestamps.size} != {expected_length}"
                )
            self._timestamp_cache[episode_index] = timestamps
            return timestamps.copy()

        def _label_from_frame(
            self,
            frame: Any,
            episode_index: int,
            step_index: int,
        ) -> StructuredActionV1:
            key = (int(episode_index), int(step_index))
            cached = self._label_cache.get(key)
            if cached is not None:
                return cached
            action_indices = [int(step_index) + i for i in range(STRUCTURED_ACTION_HORIZON)]
            state_indices = [int(step_index)]
            action_eef = _column_matrix(
                frame, "action.eef_9d", action_indices, name="action.eef_9d"
            )
            action_gripper = _column_matrix(
                frame, "action.gripper_position", action_indices, name="action.gripper_position"
            )
            state_eef = _column_matrix(frame, "state.eef_9d", state_indices, name="state.eef_9d")[0]
            state_gripper = _column_matrix(
                frame, "state.gripper_position", state_indices, name="state.gripper_position"
            )[0]
            if action_eef.shape[1] != 9 or action_gripper.shape[1] != 1:
                raise ValueError("Stage38-D expects action dimensions 9+1")
            physical_action = np.concatenate([action_eef, action_gripper], axis=-1)
            state_v1 = np.concatenate([state_eef, state_gripper], axis=-1)
            reconciliation: dict[str, Any] | None = None
            if self.stage45b_label_reconciliation:
                state_v1, reconciliation = reconcile_stage45b_label_anchor(
                    physical_action, state_v1
                )
            timestamps = self._timestamps(episode_index)[action_indices]
            provenance: dict[str, Any] = {
                "stage": "Stage38-D",
                "dataset_path": str(self.dataset_path),
                "episode_index": int(episode_index),
                "source_episode_index": self._episode_id(episode_index),
                "step_index": int(step_index),
                "source_timestamp_authority": "ORIGINAL_PARQUET_TIMESTAMP_COLUMN",
                "training_label_generation": "ON_THE_FLY_READ_ONLY",
                "stage45b_label_reconciliation_enabled": self.stage45b_label_reconciliation,
            }
            if reconciliation is not None:
                provenance["stage45b_label_anchor_reconciliation"] = reconciliation
            label = encode_structured_action(
                physical_action,
                timestamps,
                state_v1,
                timing_mode=self.timing_mode,
                chunk_id=f"stage38-d:{Path(self.dataset_path).name}:episode-{self._episode_id(episode_index)}:step-{step_index}",
                source_frame=DEFAULT_MOTION_INTENT_V2_FRAME,
                tcp_semantic_id=DEFAULT_MOTION_INTENT_V2_TCP,
                observation_timestamp=float(timestamps[0]),
                provenance=provenance,
            )
            self._label_cache[key] = label
            return label

        def _collect_training_labels(self) -> list[StructuredActionV1]:
            labels: list[StructuredActionV1] = []
            # Use the loader's parquet-only method so CPU statistics collection
            # never decodes video and never writes to the dataset.
            for episode_index in range(len(self.episode_loader)):
                frame = self.episode_loader._load_parquet_data(self._episode_id(episode_index))
                for step_index in range(self.get_effective_episode_length(episode_index)):
                    labels.append(self._label_from_frame(frame, episode_index, step_index))
            if not labels:
                raise ValueError("the exact training split produced no valid Stage38 labels")
            return labels

        def get_datapoint(
            self,
            episode_data: Any,
            step_index: int,
            episode_index: int | None = None,
        ) -> dict[str, Any]:
            if episode_index is None:
                raise ValueError("Stage38-D datapoints require an explicit episode index")
            if self.processor is None:
                raise AssertionError("Processor must be set before getting datapoints")
            vla_step_data = extract_step_data(
                episode_data,
                step_index,
                self.modality_configs,
                self.embodiment_tag,
                self.allow_padding,
            )
            processed = self.processor(
                [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            )
            label = self._label_from_frame(episode_data, episode_index, step_index)
            processed["action"] = self.structured_action_normalizer.normalize(label)
            processed["action_mask"] = label.mask.astype(np.float32, copy=True)
            return processed

        def get_shard(self, idx: int) -> list:
            episodes = self.sharded_episodes[idx]
            datapoints = []
            for episode_index, step_indices in episodes:
                episode_data = self.episode_loader[int(episode_index)]
                for step_index in step_indices:
                    datapoints.append(
                        self.get_datapoint(
                            episode_data,
                            int(step_index),
                            episode_index=int(episode_index),
                        )
                    )
            return datapoints

    class StructuredDatasetFactory:
        """Drop-in replacement for GR00T's writing DatasetFactory."""

        def __init__(self, config: Any):
            self.config = config

        def build(self, processor: Any):
            if self.config.training.eval_strategy != "no":
                raise ValueError("Stage38-D forbids validation/evaluation dataset construction")
            all_datasets = []
            all_weights = []
            for dataset_spec in self.config.data.datasets:
                for dataset_path in dataset_spec.dataset_paths:
                    if str(dataset_path) != str(dataset_spec.dataset_paths[0]):
                        raise ValueError("Stage38-D only admits the single frozen TINY1 dataset")
                    dataset = StructuredShardedSingleStepDataset(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(dataset_spec.embodiment_tag),
                        modality_configs=self.config.data.modality_configs[dataset_spec.embodiment_tag],
                        shard_size=self.config.data.shard_size,
                        episode_sampling_rate=self.config.data.episode_sampling_rate,
                        seed=self.config.data.seed,
                        allow_padding=self.config.data.allow_padding,
                        timing_mode=STRUCTURED_TIMING_MODE,
                        use_percentiles=self.config.model.use_percentiles,
                        stage45b_label_reconciliation=(
                            os.environ.get("STAGE45B_LABEL_ANCHOR_RECONCILIATION", "0")
                            == "1"
                        ),
                    )
                    all_datasets.append(dataset)
                    all_weights.append(float(dataset_spec.mix_ratio))
            if len(all_datasets) != 1:
                raise ValueError("Stage38-D requires exactly one dataset specification")

            mixture = ShardedMixtureDataset(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=True,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            )
            dataset = all_datasets[0]
            mixture.structured_action_statistics = dataset.structured_action_statistics
            mixture.structured_action_contract = structured_action_contract_summary(
                STRUCTURED_TIMING_MODE
            )
            mixture.structured_action_label_count = len(dataset.structured_action_labels)

            # This is the project-owned sidecar consumed by the later offline
            # evaluator.  It is outside the source dataset and is written only
            # by rank zero.  The normal GR00T statistics sidecar remains the
            # historical raw/state statistics needed by the stock processor.
            if os.environ.get("RANK", "0") == "0":
                output_dir = Path(self.config.training.output_dir)
                _write_json(
                    output_dir / "experiment_cfg" / "structured_action_statistics.json",
                    {
                        "schema": "STAGE38_D_STRUCTURED_TRAINING_SIDE_CAR_V1",
                        "contract": mixture.structured_action_contract,
                        "statistics": mixture.structured_action_statistics,
                        "dataset_path": str(dataset.dataset_path),
                        "dataset_label_count": len(dataset.structured_action_labels),
                        "source_dataset_untouched_by_factory": True,
                        "upstream_generate_stats_called": False,
                    },
                )
            return mixture, None

    return StructuredDatasetFactory


def install_structured_dataset_factory() -> type:
    """Patch only the imported GR00T setup module's factory symbol."""

    from gr00t.model.gr00t_n1d7 import setup as gr00t_setup

    factory = build_structured_gr00t_dataset_factory()
    gr00t_setup.DatasetFactory = factory
    return factory


__all__ = [
    "StructuredActionNormalizer",
    "build_structured_gr00t_dataset_factory",
    "install_structured_dataset_factory",
]
