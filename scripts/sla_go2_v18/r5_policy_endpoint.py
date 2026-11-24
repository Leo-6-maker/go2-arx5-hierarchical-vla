"""Attach a committed Go2 adapter to the official SLA_Memory policy builder."""

import dataclasses
import math
from pathlib import Path
import sys
from threading import Lock
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict

from r5_checkpoint import binding_digest, payload_digest, restore
from r5_training_owner import checksum, pinned

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_hierarchical_v16"))
from demonstrated_grip import read, require
from feedback_gpu import verify_files
from openpi.training.optimizer import create_optimizer
from right_arm_recipe import TargetConfig, candidate, initialize_lora

OPTIMIZER = BASE / "reports/SLA_GO2_V18/P1_P2/OPTIMIZER_ABI.json"
_FACTORY_CAPTURE_LOCK = Lock()


def parameter_layout(params):
    return {
        "/".join(key): {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in flatten_dict(params).items()
    }


def frozen_digests(params):
    from audit_r5_training import digest

    flat = flatten_dict(params)
    return {"/".join(key): digest(value) for key, value in flat.items()}


def adapter_template(config, checkpoint):
    recipe = candidate(config, checkpoint)
    shape = jax.eval_shape(
        lambda key: nnx.state(recipe.model.create(key), recipe.trainable_filter),
        jax.random.key(1729),
    )
    weights = initialize_lora(shape.to_pure_dict(), recipe.weight_loader.seed)
    shape.replace_by_pure_dict(weights)
    tx = create_optimizer(config.optimizer, config.lr_schedule)
    return dict(completed_steps=np.int32(0), adapters=weights, adamw=tx.init(shape))


@dataclasses.dataclass(frozen=True)
class BoundInferenceConfig(TargetConfig):
    learned_adapters: dict = dataclasses.field(default_factory=dict, repr=False, compare=False)
    expected_frozen: dict = dataclasses.field(default_factory=dict, repr=False, compare=False)

    def load(self, params, *, remove_extra_params=True):
        layout = read(OPTIMIZER)["trainable"]
        require(parameter_layout(self.learned_adapters) == layout, "bad right-arm adapter layout")
        flat = flatten_dict(params)
        overlay = flatten_dict(self.learned_adapters)
        require(not flat.keys() & overlay.keys() and len(flat) == 51
                and all(p.dtype == jnp.bfloat16 for p in flat.values()),
                "LoRA overlay expects the original 51 BF16 leaves")
        # BaseModelConfig.load checks every parameter shape against the model
        # graph. Do not intersect the trees or discard mismatched parameters.
        if self.expected_frozen:
            require(frozen_digests(params) == self.expected_frozen, "frozen inference parent changed")
        full = unflatten_dict({**flat, **overlay})
        return super().load(full, remove_extra_params=False)


def official_policy(conf, *, load_adapters=None):
    """Use the same official factory, processor, history and EEF packer for both arms."""
    from XPolicyLab.policy.SLA_Memory.model import Model, _policy_config

    with _FACTORY_CAPTURE_LOCK:
        factory = _policy_config.create_trained_policy
        if load_adapters is None:
            return Model(conf)
        # Capture exactly the configuration that the pinned wrapper resolved.
        # Exit the patch before checkpoint I/O or any inference.
        with patch.object(
            _policy_config, "create_trained_policy",
            side_effect=lambda train_config, checkpoint: (train_config, checkpoint),
        ):
            wrapper = Model(conf)

    train_config, checkpoint_dir = wrapper._policy
    checkpoint = checkpoint_dir / "params"
    recipe = candidate(train_config, checkpoint)
    adapters, expected_frozen = load_adapters(train_config, checkpoint)
    model = BoundInferenceConfig(
        **dataclasses.asdict(recipe.model),
        learned_adapters=adapters,
        expected_frozen=expected_frozen,
    )
    wrapper._policy = factory(
        dataclasses.replace(train_config, model=model), checkpoint_dir
    )
    wrapper.model = wrapper._policy
    return wrapper


def endpoint_receipts(root, endpoint):
    trial = Path(endpoint["trial"]).resolve()
    step = endpoint["completed_step"]
    require(trial.parent == (root / "v18").resolve() and trial.name.startswith("r5-smoke-")
            and type(step) is int and step > 0, "R5 committed training checkpoint required")
    binding = pinned(endpoint["binding"])
    segment = pinned(endpoint["segment"])
    update = pinned(endpoint["update"])
    parent = pinned(endpoint["frozen_parent"])
    require(
        Path(endpoint["binding"]["path"]).resolve() == trial / "BINDING.json"
        and Path(endpoint["frozen_parent"]["path"]).resolve() == trial / "FROZEN_SHA256.json"
        and Path(endpoint["segment"]["path"]).resolve().parent == trial
        and Path(endpoint["update"]["path"]).resolve() == trial / "updates" / f"step-{step:05d}.json"
        and binding["scope"] == trial.name
        and binding["run_purpose"] == "GO2_POST_TRAINING_REAL_GPU_SMOKE"
        and binding["parent_manifest_sha256"] == checksum(root / "evidence/SOURCE_AND_WEIGHT_MANIFEST.actual.json")
        and segment["status"] in ("COMPLETED", "SAFE_STOP") and segment["unknown_step"] is None
        and segment["actual_updates"] > 0
        and segment["start_step"] < step <= segment["last_committed_step"]
        and segment["binding_sha256"] == update["binding_sha256"] == binding_digest(binding)
        and update["status"] == "COMMITTED" and update["completed_steps"] == step
        and update["metrics"]["grad_norm"] > 0
        and all(math.isfinite(v) for v in update["metrics"].values())
        and len(parent) == 51,
        "C endpoint requires a verified parent, closed segment and matching committed update",
    )
    return trial, step, binding, parent


def load_c(root, endpoint):
    """Build a production C policy. Sampling stays with the allocated caller."""
    trial, step, binding, parent = endpoint_receipts(root, endpoint)
    conf = verify_files(root)

    def adapters(config, checkpoint):
        restored = restore(
            trial / "checkpoints", completed=step,
            template=adapter_template(config, checkpoint),
            binding=binding, adapter_layout=read(OPTIMIZER)["trainable"],
        )
        require(payload_digest(restored) == endpoint["checkpoint_state_sha256"],
                "requested committed checkpoint content changed")
        return restored["adapters"], parent

    return official_policy(conf, load_adapters=adapters)
