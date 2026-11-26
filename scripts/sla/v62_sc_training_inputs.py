"""Prepare the independent SC-VLA FM input at its recorded 15 Hz clock."""

from dataclasses import replace

import numpy as np

from v62_sc_source import HORIZON, NATIVE_HZ, SCHEMA
from v6_ville_pilot import need


ROLE = "sc_vla"
ROLE_KEY = 3
VALID_CHANNELS = 7
MODEL_CHANNELS = 32
VIEWS = {"base_0_rgb": True, "left_wrist_0_rgb": False, "right_wrist_0_rgb": True}


def prepare(reader, tokenizer, positions, indices, *, training=True):
    from go2arx5_vla.sla_adapter.batch_v2 import drop_state

    manifest = reader.manifest
    need(manifest["schema"] == SCHEMA and manifest["native_hz"] == NATIVE_HZ and manifest["horizon"] == HORIZON
         and manifest["target_command"] is False and manifest["camera_mask"] == VIEWS, "Expected the SC-VLA source codec")
    rows, token_ids, token_masks, masks = [], [], [], []
    for cursor, index in zip(positions, indices, strict=True):
        episode, _ = reader.rows[int(index)]
        need(not training or reader.episodes[episode]["split"] == "TRAIN", "SOURCE_DEV cannot enter an H30 training batch")
        item = reader[int(index)]
        state, action, support = item["state"], item["actions"], item["support"]
        need(state.shape == (MODEL_CHANNELS,) and action.shape == (HORIZON, MODEL_CHANNELS)
             and support.shape == (HORIZON,) and support.dtype == bool and support.any()
             and all(np.isfinite(x).all() and not np.any(x[..., VALID_CHANNELS:]) for x in (state, action)),
             "Invalid recorded SC-VLA state or H30 action")
        need(item["image_mask"] == VIEWS and set(item["image"]) == set(item["image_padding_mask"]) == set(VIEWS)
             and all(item["image"][view].shape == (224, 224, 3) and item["image"][view].dtype == np.uint8
                     and item["image_padding_mask"][view].shape == (224, 224) for view in VIEWS)
             and not np.any(item["image"]["left_wrist_0_rgb"])
             and not np.any(item["image_padding_mask"]["left_wrist_0_rgb"]), "Invalid recorded SC-VLA camera input")
        state_tokens = None if training and drop_state(int(cursor)) else state[:VALID_CHANNELS]
        ids, mask = tokenizer.delegate.tokenize(item["prompt"], state_tokens, use_discrete_state_format=True)
        rows.append(item)
        token_ids.append(ids)
        token_masks.append(mask)
        masks.append(support[:, None] & (np.arange(MODEL_CHANNELS) < VALID_CHANNELS))
    need(bool(rows), "An H30 batch requires at least one actual visit")
    observation = dict(state=np.stack([row["state"] for row in rows]),
                       tokenized_prompt=np.stack(token_ids), tokenized_prompt_mask=np.stack(token_masks))
    for field in ("image", "image_mask", "image_padding_mask"):
        observation[field] = {view: np.stack([row[field][view] for row in rows]) for view in VIEWS}
    return dict(observation=observation, actions=np.stack([row["actions"] for row in rows]),
                action_support=np.stack(masks))


def frozen_draws(size):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def generate(rng, cursor):
        role_rng = jax.random.fold_in(rng, ROLE_KEY)
        positions = jnp.asarray(cursor, dtype=jnp.uint32) + jnp.arange(size, dtype=jnp.uint32)
        keys = jax.vmap(lambda pos: jax.random.fold_in(role_rng, pos))(positions)
        parts = jax.vmap(lambda key: jax.random.split(key, 2))(keys)
        noise = jax.vmap(lambda key: jax.random.normal(key, (HORIZON, MODEL_CHANNELS), dtype=jnp.float32))(parts[:, 0])
        tau = jax.vmap(lambda key: jax.random.beta(key, 1.5, 1, dtype=jnp.float32))(parts[:, 1])
        return noise, tau * .999 + .001

    return generate


def abstract_model():
    from flax import nnx
    import jax
    from go2arx5_vla.sla_adapter.scale20k_contract import scale_graph_config
    from go2arx5_vla.sla_adapter.scale20k_model import INITIAL_SEED

    template = replace(scale_graph_config("target"), action_horizon=HORIZON)
    return nnx.eval_shape(template.create, jax.random.PRNGKey(INITIAL_SEED))


def flow_loss(graph):
    from flax import nnx
    import jax
    import jax.numpy as jnp
    from go2arx5_vla.sla_adapter.batch_v2 import error32

    def loss(active, frozen, observation, actions, noise, tau, support):
        actions, noise = (jnp.where(support, value, 0.) for value in (actions, noise))
        with jax.default_matmul_precision("highest"):
            model = nnx.merge(graph, active["shared"], active["private"], frozen)
            error = error32(model, observation, actions, noise, tau)
        return jnp.mean(jnp.sum(jnp.where(support, error, 0.), axis=(-2, -1)) / jnp.sum(support, axis=(-2, -1)))

    return loss
