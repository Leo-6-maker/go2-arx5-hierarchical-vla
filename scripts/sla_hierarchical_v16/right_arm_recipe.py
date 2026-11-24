"""Candidate Go2 target-only action-expert recipe for the pinned RoboDojo release."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, traverse_util
from openpi.models import model, pi0, pi0_config
from openpi.shared import nnx_utils
from openpi.training.weight_loaders import CheckpointWeightLoader


def flow_inputs(actions, noise, time):
    if actions.shape != noise.shape or actions.shape[-2:] != (50, 32):
        raise ValueError("expected matching official 50 x 32 actions and noise")
    active = (jnp.arange(32) >= 7) & (jnp.arange(32) < 14)
    # Mask targets before embedding: a change to an inactive label must not affect
    # active predictions through the model's noisy action input.
    target = jnp.where(active, actions, 0.0)
    return time * noise + (1 - time) * target, noise - target


def active_velocity_loss(predicted, target):
    if predicted.shape != target.shape or predicted.shape[-2:] != (50, 32):
        raise ValueError("expected matching velocity tensors")
    return jnp.mean(jnp.square(predicted[..., 7:14] - target[..., 7:14]), axis=-1)


class TargetPi0(pi0.Pi0):
    def compute_loss(self, rng, observation, actions, *, train=False):
        # Same model and random schedule as the pinned Pi0.compute_loss.
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        obs = model.preprocess_observation(preprocess_rng, observation, train=train)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        x_t, u_t = flow_inputs(actions, noise, time[..., None, None])
        prefix, pmask, par = self.embed_prefix(obs)
        suffix, smask, sar, cond = self.embed_suffix(obs, x_t, time)
        mask = jnp.concatenate([pmask, smask], axis=1)
        attention = pi0.make_attn_mask(mask, jnp.concatenate([par, sar], axis=0))
        (_, result), _ = self.PaliGemma.llm(
            [prefix, suffix],
            mask=attention,
            positions=jnp.cumsum(mask, axis=1) - 1,
            adarms_cond=[None, cond],
        )
        velocity = self.action_out_proj(result[:, -self.action_horizon :])
        return active_velocity_loss(velocity, u_t)


@dataclasses.dataclass(frozen=True)
class TargetConfig(pi0_config.Pi0Config):
    def create(self, rng):
        return TargetPi0(self, rngs=nnx.Rngs(rng))


@dataclasses.dataclass(frozen=True)
class ParentAndAdapters:
    params_path: str
    seed: int

    def load(self, shape_tree):
        flat = traverse_util.flatten_dict(shape_tree)
        rng = np.random.default_rng(self.seed)
        for key in sorted(flat):
            leaf = flat[key]
            if "lora" not in key[-1]:
                continue
            if key[-1].endswith("lora_b"):
                flat[key] = np.zeros(leaf.shape, dtype=leaf.dtype)
            elif key[-1].endswith("lora_a"):
                flat[key] = (
                    0.01 * rng.standard_normal(leaf.shape, dtype=np.float32)
                ).astype(leaf.dtype)
            else:
                raise ValueError(f"unknown adapter initializer {key}")
        return CheckpointWeightLoader(self.params_path).load(
            traverse_util.unflatten_dict(flat)
        )


@dataclasses.dataclass(frozen=True)
class Recipe:
    model: TargetConfig
    trainable_filter: object
    weight_loader: ParentAndAdapters


def candidate(base, checkpoint):
    if (
        base.name != "train_robodojo_real"
        or base.model.action_expert_variant != "gemma_300m"
        or not base.model.pi05
        or base.model.extra_frames_first
        or checkpoint.parent.name != "SLA_Memory_robodojo_real_ckpt"
        or checkpoint.name != "params"
    ):
        raise ValueError("unexpected parent release config")
    fields = dataclasses.asdict(base.model)
    fields["action_expert_variant"] = "gemma_300m_lora"
    return Recipe(
        TargetConfig(**fields),
        nnx.All(nnx.Param, nnx_utils.PathRegex(r"PaliGemma/llm/.*_1/.*lora.*")),
        ParentAndAdapters(str(checkpoint), seed=1729),
    )
