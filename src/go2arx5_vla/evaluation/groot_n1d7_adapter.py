"""Allowlisted GR00T N1.7 adapter for the reusable inference engine.

The module itself is import-safe on CPU: Torch, Transformers, and GR00T are
loaded only from ``prepare`` after the engine has completed admission.  All
checkpoint and window iteration remains owned by the generic engine; this
adapter owns model construction, observation preparation, and the versioned
source-mapped action-head sampler.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


ADAPTER_ID = "groot_n1d7_structured_action_v1"
ADAPTER_VERSION = "v1"
SOURCE_EQUIVALENCE = {
    "adapter_id": ADAPTER_ID,
    "version": ADAPTER_VERSION,
    "sampler_semantics": "source_mapped_action_head_euler",
    "decoder_semantics": "manifest_bound_cpu_decoder",
    "model_input_keys": ("body_rgb", "wrist_rgb", "state"),
    "target_xyz_in_model_input": False,
}


class GrootN1d7AdapterError(RuntimeError):
    """Raised when the versioned GR00T adapter cannot bind its inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GrootN1d7AdapterError(message)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _strict_integer(value: Any, *, field: str) -> int:
    _require(type(value) is int, f"{field} must be a JSON integer")
    return value


def _path(value: Any, *, field: str, directory: bool | None = None) -> Path:
    result = Path(str(value)).expanduser()
    _require(result.is_absolute(), f"{field} must be an absolute path")
    if directory is True:
        _require(result.is_dir() and not result.is_symlink(), f"{field} is not a directory")
    elif directory is False:
        _require(result.is_file() and not result.is_symlink(), f"{field} is not a file")
    return result.resolve()


def _load_frameworks() -> tuple[Any, Any]:
    """Import model frameworks only from the post-admission preparation call."""

    import torch
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401
    from transformers import AutoModel

    return torch, AutoModel


def _build_processor(config: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    """Build the accepted processor through the pinned runtime helper."""

    # This import is deliberately inside the adapter, after admission.  The
    # generic engine has no dependency on a model-specific processor module.
    from scripts.stage26_h1_c1_fixed_noise_local_flow import _build_runtime_processor

    return _build_runtime_processor(
        groot_source=_path(config["groot_source"], field="adapter_config.groot_source", directory=True),
        base_checkpoint=_path(
            config["base_checkpoint"],
            field="adapter_config.base_checkpoint",
            directory=True,
        ),
        dataset_dir=_path(
            config["dataset_dir"], field="adapter_config.dataset_dir", directory=True
        ),
        modality_path=_path(
            config["modality_config"],
            field="adapter_config.modality_config",
            directory=False,
        ),
        cosmos_snapshot=_path(
            config["cosmos_snapshot"],
            field="adapter_config.cosmos_snapshot",
            directory=True,
        ),
    )


class GrootN1d7StructuredAdapter:
    """One versioned GR00T action-head adapter instance."""

    adapter_id = ADAPTER_ID
    adapter_version = ADAPTER_VERSION

    def __init__(self, context: Any) -> None:
        self._initial_context = context
        self._context: Any = None
        self._config: Mapping[str, Any] | None = None
        self._torch: Any = None
        self._auto_model: Any = None
        self._processor: Any = None
        self._embodiment: Any = None
        self._modality_configs: Any = None
        self._loader: Any = None
        self._model: Any = None
        self._model_label: str | None = None
        self.runtime_flags: dict[str, Any] = {
            "adapter_id": ADAPTER_ID,
            "adapter_version": ADAPTER_VERSION,
            "model_imported": False,
            "model_construction": False,
            "model_forward": False,
        }

    def prepare(self, context: Any) -> None:
        """Import frameworks and construct processor state after admission."""

        config = _mapping(
            context.runtime_config.get("adapter_config"),
            field="runtime_config.adapter_config",
        )
        for key in (
            "device",
            "groot_source",
            "base_checkpoint",
            "dataset_dir",
            "modality_config",
            "cosmos_snapshot",
        ):
            _require(key in config, f"runtime_config.adapter_config.{key} is required")
        device = str(config["device"]).strip()
        _require(device.startswith("cuda"), "GR00T adapter requires an admitted CUDA device")
        self._context = context
        self._config = config
        self._torch, self._auto_model = _load_frameworks()
        self.runtime_flags["model_imported"] = True
        self._processor, self._embodiment, self._modality_configs = _build_processor(config)
        self._loader = None

    def _window_spec(self, window: Mapping[str, Any]) -> Mapping[str, Any]:
        value = _mapping(window.get("adapter_input"), field="window.adapter_input")
        _strict_integer(value.get("dataset_index"), field="window.adapter_input.dataset_index")
        for key in ("observations", "state_v1", "expert_physical_action"):
            _require(key in value, f"window.adapter_input.{key} is required")
        return value

    def _prepare_model_inputs(
        self, window: Mapping[str, Any], spec: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Prepare one observation through the pinned GR00T data boundary."""

        if self._loader is None:
            from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

            _require(self._config is not None, "adapter was not prepared")
            self._loader = LeRobotEpisodeLoader(
                _path(self._config["dataset_dir"], field="adapter_config.dataset_dir", directory=True),
                self._modality_configs,
            )
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.types import MessageType, VLAStepData

        index = _strict_integer(spec["dataset_index"], field="window.adapter_input.dataset_index")
        episode = self._loader[index]
        step_data = extract_step_data(
            episode,
            0,
            self._modality_configs,
            self._embodiment,
            allow_padding=False,
        )
        inference_step = VLAStepData(
            images=step_data.images,
            states=step_data.states,
            actions={},
            text=str(self._config.get("language", "")),
            embodiment=self._embodiment,
        )
        processed = self._processor(
            [{"type": MessageType.EPISODE_STEP.value, "content": inference_step}]
        )
        batch = self._processor.collator([processed])
        model_inputs = _mapping(batch.get("inputs"), field="processor.inputs")
        _require("action" not in model_inputs, "expert action leaked into model input")
        _require("action_mask" not in model_inputs, "action mask leaked into model input")
        for forbidden in ("target_position_world_m", "target_xyz", "target_world_m"):
            _require(forbidden not in model_inputs, f"target data leaked into model input: {forbidden}")
        state = model_inputs.get("state")
        if state is not None:
            _require(bool(self._torch.isfinite(state).all().item()), "processed state is nonfinite")
        return model_inputs

    def _model_for(self, checkpoint: Mapping[str, Any]) -> Any:
        label = str(checkpoint.get("label", ""))
        if self._model is not None and self._model_label == label:
            return self._model
        if self._model is not None:
            del self._model
            self._model = None
        identity = _mapping(checkpoint.get("identity"), field="checkpoint.identity")
        checkpoint_path = _path(
            identity.get("resolved_path", identity.get("path")),
            field=f"checkpoint[{label}].identity.path",
            directory=True,
        )
        _require(self._config is not None, "adapter was not prepared")
        loading_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
        }
        cache_dir = self._config.get("transformers_cache_dir")
        if cache_dir:
            loading_kwargs["cache_dir"] = str(
                _path(cache_dir, field="adapter_config.transformers_cache_dir", directory=True)
            )
        self._model = self._auto_model.from_pretrained(checkpoint_path, **loading_kwargs)
        self._model.eval()
        device = str(self._config["device"])
        self._model.to(device=device, dtype=self._torch.bfloat16)
        self._model_label = label
        self.runtime_flags["model_construction"] = True
        return self._model

    def _noise_for(self, noise_bank: Path, spec: Mapping[str, Any]) -> Any:
        index = _strict_integer(spec["dataset_index"], field="window.adapter_input.dataset_index")
        loaded = np.load(noise_bank, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if "noise" in loaded.files:
                    bank = np.asarray(loaded["noise"])
                else:
                    keys = sorted(loaded.files)
                    _require(keys, "noise bank is empty")
                    bank = np.asarray(loaded[keys[0]])
            finally:
                loaded.close()
        else:
            bank = np.asarray(loaded)
        if tuple(bank.shape) == (1, 40, 132):
            noise = bank
        else:
            _require(
                bank.ndim == 4 and tuple(bank.shape[1:]) == (1, 40, 132),
                "noise bank must contain [N,1,40,132] entries",
            )
            _require(0 <= index < int(bank.shape[0]), "noise bank index is out of range")
            noise = bank[index]
        _require(np.issubdtype(noise.dtype, np.number), "noise bank is not numeric")
        _require(bool(np.isfinite(noise).all()), "noise bank contains nonfinite values")
        return self._torch.from_numpy(np.ascontiguousarray(noise.astype(np.float32, copy=False)))

    def _offline_step(
        self,
        action_head: Any,
        processed_backbone: Any,
        action_input: Any,
        state_features: Any,
        actions: Any,
        timestep: int,
    ) -> Any:
        """Run one source-mapped action-head update from the named profile."""

        torch = self._torch
        vl_embeds = processed_backbone.backbone_features
        device = vl_embeds.device
        batch_size = int(actions.shape[0])
        timesteps = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
        action_features = action_head.action_encoder(
            actions, timesteps, action_input.embodiment_id
        )
        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + action_head.position_embedding(pos_ids).unsqueeze(0)
        state_action_embeddings = torch.cat((state_features, action_features), dim=1)
        if action_head.config.use_alternate_vl_dit:
            model_output = action_head.model(
                hidden_states=state_action_embeddings,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps,
                image_mask=processed_backbone.image_mask,
                backbone_attention_mask=processed_backbone.backbone_attention_mask,
            )
        else:
            model_output = action_head.model(
                hidden_states=state_action_embeddings,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps,
            )
        prediction = action_head.action_decoder(model_output, action_input.embodiment_id)
        velocity = prediction[:, -actions.shape[1] :]
        _require(tuple(velocity.shape) == (1, 40, 132), "action-head output shape drift")
        _require(bool(torch.isfinite(velocity).all().item()), "action-head output is nonfinite")
        return velocity

    def _sample(self, model: Any, model_inputs: Mapping[str, Any], noise: Any, context: Any) -> tuple[np.ndarray, int]:
        torch = self._torch
        backbone_inputs, action_input = model.prepare_input(dict(model_inputs))
        with torch.inference_mode():
            backbone_output = model.backbone(backbone_inputs)
            processed = model.action_head.process_backbone_output(backbone_output)
            state_history = action_input.state
            expected_history = int(model.action_head.config.state_history_length)
            _require(int(state_history.shape[1]) == expected_history, "state history length drift")
            state_features = model.action_head.state_encoder(
                state_history.view(state_history.shape[0], 1, -1),
                action_input.embodiment_id,
            )
            current = noise.to(
                device=processed.backbone_features.device,
                dtype=next(model.parameters()).dtype,
            )
            timesteps = list(context.manifest["sampler"]["timesteps"])
            dt = float(context.manifest["sampler"]["dt"])
            calls = 0
            for timestep in timesteps:
                velocity = self._offline_step(
                    model.action_head,
                    processed,
                    action_input,
                    state_features,
                    current,
                    _strict_integer(timestep, field="manifest.sampler.timesteps"),
                )
                current = current + dt * velocity
                calls += 1
        raw = current.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        _require(tuple(raw.shape) == (1, 40, 132), "raw universal action shape drift")
        _require(bool(np.isfinite(raw).all()), "raw universal action is nonfinite")
        self.runtime_flags["model_forward"] = True
        return np.ascontiguousarray(raw), calls

    def infer(
        self,
        *,
        checkpoint: Mapping[str, Any],
        window: Mapping[str, Any],
        noise_bank: Path,
        context: Any,
    ) -> Any:
        _require(self._processor is not None, "adapter.prepare() was not called")
        spec = self._window_spec(window)
        model_inputs = self._prepare_model_inputs(window, spec)
        model = self._model_for(checkpoint)
        noise = self._noise_for(noise_bank, spec)
        raw, calls = self._sample(model, model_inputs, noise, context)
        from .h200_engine import AdapterInference

        return AdapterInference(
            raw=raw,
            sampler_calls=calls,
            observations=spec["observations"],
            state_v1=spec["state_v1"],
            expert_physical_action=spec["expert_physical_action"],
            model_input_keys=("body_rgb", "wrist_rgb", "state"),
            target_xyz_in_model_input=False,
            metadata=dict(SOURCE_EQUIVALENCE),
        )

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._torch is not None and bool(self.runtime_flags.get("model_forward", False)):
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()


def build_adapter(context: Any) -> GrootN1d7StructuredAdapter:
    """Factory registered by the exact allowlisted adapter ID."""

    return GrootN1d7StructuredAdapter(context)


__all__ = [
    "ADAPTER_ID",
    "ADAPTER_VERSION",
    "GrootN1d7AdapterError",
    "GrootN1d7StructuredAdapter",
    "SOURCE_EQUIVALENCE",
    "build_adapter",
]
