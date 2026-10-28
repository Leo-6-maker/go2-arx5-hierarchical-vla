"""Prepared frozen N1.7 feature/velocity bridge; activation requires a new GO.

Velocity calculation adapted from Microsoft FlowDAgger groot_inversion.py,
3426f246510fd67a9e39c3c08bd5b02d586ad688; MIT notice in third_party/flowdagger.
Copyright (c) Microsoft Corporation.
No policy constructor or launcher is provided. E0 does not import this module.
"""
from dataclasses import dataclass

import torch

from .inversion import CountedVelocity


@dataclass(frozen=True)
class FlowContext:
    backbone_features: object
    state_features: object
    embodiment_id: object
    backbone_output: object
    observation_id: str


def require_frozen(module):
    if module.training or any(p.requires_grad for p in module.parameters()):
        raise ValueError('BASE_GR00T_MUST_ALREADY_BE_EVAL_AND_FULLY_FROZEN')


@torch.no_grad()
def context_from_observation(model, processor, payload):
    """Use the existing observation-only processor; no discarded policy draw."""
    from scripts.pi0292.live_inputs import prepare
    from scripts.pi0292.relay_protocol import validate_observation
    require_frozen(model)
    validate_observation(payload)
    batch = prepare(processor, payload)
    backbone_input, action_input = model.prepare_input(batch['inputs'])
    backbone_output = model.backbone(backbone_input)
    features = model.action_head._encode_features(backbone_output, action_input)
    return FlowContext(features.backbone_features, features.state_features,
        action_input.embodiment_id, backbone_output, payload['snapshot_uuid'])


def native_velocity(head, context, space, *, limit):
    require_frozen(head)
    if (head.action_horizon != space.horizon or head.action_dim != space.padded_dim
            or head.num_inference_timesteps != space.inference_steps):
        raise ValueError('EXACT_BOUND_MODEL_SPACE_AND_NATIVE_SAMPLER_REQUIRED')

    @torch.no_grad()
    def evaluate(actions, t_cont):
        require_frozen(head)
        batch = actions.shape[0]
        if actions.shape[1:] != (space.horizon, space.padded_dim):
            raise ValueError('FULL_PADDED_MODEL_SPACE_REQUIRED')
        timesteps = torch.full((batch,), int(t_cont*head.num_timestep_buckets), device=actions.device)
        action_features = head.action_encoder(actions, timesteps, context.embodiment_id)
        if head.config.add_pos_embed:
            ids = torch.arange(action_features.shape[1], dtype=torch.long, device=actions.device)
            action_features = action_features+head.position_embedding(ids).unsqueeze(0)
        arguments = dict(hidden_states=torch.cat((context.state_features, action_features), dim=1),
            encoder_hidden_states=context.backbone_features, timestep=timesteps)
        if head.config.use_alternate_vl_dit:
            arguments.update(image_mask=context.backbone_output.image_mask,
                backbone_attention_mask=context.backbone_output.backbone_attention_mask)
        output = head.model(**arguments)
        prediction = head.action_decoder(output, context.embodiment_id)
        return prediction[:, -head.action_horizon:]

    return CountedVelocity(evaluate, limit=limit)
