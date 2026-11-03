"""Explicit imagined-state recursion; restored caches are not future model inputs."""
from copy import deepcopy
import numpy as np

ROLES = {
 'base_pose':'PREDICT', 'base_velocity':'PREDICT', 'joint_position':'PREDICT',
 'joint_velocity':'PREDICT', 'tcp':'PREDICT', 'tool_velocity':'PREDICT',
 'gripper_fraction':'PREDICT', 'gripper_speed':'PREDICT', 'object_pose':'PREDICT',
 'object_velocity':'PREDICT', 'contact':'PREDICT', 'native_progress':'PREDICT',
 'time_s':'ANALYTIC_UPDATE', 'installed_reference':'ANALYTIC_UPDATE',
 'reference_elapsed_s':'ANALYTIC_UPDATE', 'history':'ANALYTIC_UPDATE',
 'goal':'CONTEXT', 'geometry':'CONTEXT', 'mass_friction':'CONTEXT',
 'integration':'RESTORE_ONLY', 'solver_warmstart':'RESTORE_ONLY',
 'wbc_observation_stack':'RESTORE_ONLY', 'wbc_hidden':'RESTORE_ONLY',
 'mpc_policy_cache':'RESTORE_ONLY', 'rng':'RESTORE_ONLY', 'monitor_memory':'RESTORE_ONLY'}
ACTOR_FIELDS = ('tcp','gripper_fraction','gripper_speed','tool_velocity','object_pose',
                'object_velocity','goal','last_action','tracking_residual','remaining_s')


def actor_projection(current):
    if not set(ACTOR_FIELDS) <= current.keys():
        raise ValueError('Actor requires declared measured task fields, never padded hidden state')
    return {key:deepcopy(current[key]) for key in ACTOR_FIELDS}


def future_inputs(predicted, analytic, context):
    """No replay/controller-future argument exists after burn-in."""
    result = {}
    for values,role in ((predicted,'PREDICT'),(analytic,'ANALYTIC_UPDATE'),(context,'CONTEXT')):
        if any(ROLES.get(key)!=role for key in values):
            raise ValueError('Field role mismatch or restore-only future leak')
        result.update(deepcopy(values))
    return result


def shift_history(history, predicted_state, command):
    # Keeps torch tensors/autograd intact; no NumPy/detach conversion.
    if len(history)!=8:
        raise ValueError('Exactly8 causal burn-in/history steps required')
    return [*history[1:], (predicted_state,command)]


def analytic_step(state, reference, atomic_index):
    if type(atomic_index) is not int or atomic_index<0:
        raise ValueError('Explicit atomic clock required')
    return dict(time_s=state['time_s']+.05, installed_reference=reference,
                reference_elapsed_s=(atomic_index%8+1)*.05)
