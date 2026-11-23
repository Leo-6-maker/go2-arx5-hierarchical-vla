"""Identify the actual observation processed by a policy worker and its absolute action result."""

import hashlib
import json
import numpy as np


def tensor_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(json.dumps([a.dtype.str, a.shape], separators=(",", ":")).encode() + b"\0")
    h.update(a.tobytes())
    return h.hexdigest()


def input_identity(obs):
    fields = {k: tensor_sha(a) for k, a in obs["images"].items()}
    fields.update(
        state=tensor_sha(obs["state"]),
        prompt=hashlib.sha256(obs["prompt"].encode()).hexdigest(),
    )
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(), fields


def frozen_predictor(model, obs, *, infer=None):
    """Pack one official result, optionally retaining its sampler output at the model boundary."""
    from XPolicyLab.utils.process_data import pack_robot_state

    result = (model._policy.infer if infer is None else infer)(obs)
    return np.stack(
        [
            pack_robot_state({"state": a}, "ee", model._robot_action_dim_info)
            for a in model._from_openpi_actions(result["actions"])
        ]
    )


def policy_response(query_id, obs, predictor):
    """predictor receives only the official observation, with no command ledger or teacher labels."""
    digest, _ = input_identity(obs)
    return query_id, digest, predictor(obs)
