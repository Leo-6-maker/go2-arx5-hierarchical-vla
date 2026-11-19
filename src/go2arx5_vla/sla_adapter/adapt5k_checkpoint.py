"""Seal ADAPT5K phase state separately from its immutable MIX parent."""
from pathlib import Path

from . import adapt5k_contract as contract
from . import scale20k_checkpoint as shared
from .checkpoint_v1 import sha
from .multitask_data_v6 import C20_STATISTICS
from .scale20k_contract import DOMAINS, bank_sha256
from scripts.pi0292.durable import read


SCHEMA = 'SLA_ADAPT5K_COMPLETE_PHASE_STATE_V1'
RECIPE = 'SLA_ADAPT5K_REAL_R200_TARGET_ANCHOR_V1'
ENDPOINTS = (0, 500, 2000, 5000)
TECHNICAL_CHECKPOINT = 25


def identity(binding, step):
    for name in ('parent_checkpoint_sha256','parent_training_spec_sha256',
                 'target_state_sha256','target_D1_sha256','optimizer_sha256','tokenizer_sha256',
                 'partition_sha256','parent_manifest_sha256','source_sha256'):
        if not shared.is_digest(binding[name]):
            raise ValueError(f'ADAPT5K_CHECKPOINT_MISSING_PINNED_IDENTITY:{name}')
    stages = ((0, contract.SWITCH_PHASE),(contract.SWITCH_PHASE,contract.UPDATES))
    start, stop = binding['stage_start'], binding['stage_stop']
    if (binding['plan'] != contract.PLAN or binding['recipe'] != RECIPE or (start, stop) not in stages
            or not start <= step <= stop or binding['frozen_C20_action_statistics_sha256'] != C20_STATISTICS
            or binding['learning_rates_sha256'] != bank_sha256(contract.phase_rates())
            or set(binding['stage_bank_sha256']) != set(DOMAINS)
            or not all(shared.is_digest(v) for v in binding['stage_bank_sha256'].values())
            or not shared.is_digest(binding['R200_membership_sha256'])
            or not shared.is_digest(binding['LIBERO_membership_sha256'])
            or not shared.is_digest(binding['stage_target_data_sha256'])):
        raise ValueError('ADAPT5K_PHASE_BINDING_REQUIRES_FROZEN_PARENT_AND_EXACT_STAGE_BANKS')
    return contract.progress(binding['parent_step'], step, start)


def save(path, *, phase_step, parameters, optimizer, rng, binding):
    import jax

    clocks = identity(binding, phase_step)
    tree = shared.training_tree(parameters,optimizer,rng)
    if len(jax.tree.leaves(tree)) != 140:
        raise ValueError('ADAPT5K_CHECKPOINT_REQUIRES_THE_ORIGINAL_140_STATE_ARRAYS')
    metadata = dict(schema=SCHEMA, binding=binding, phase_step=phase_step, progress=clocks,
                    training_generations=4*phase_step, raw_endpoint=True, physics=False)
    return shared.seal_arrays(path,tree,metadata)


def read_complete(pin, binding):
    root = Path(pin['path']).resolve(strict=True)
    marker = root/'COMPLETE.json'
    if sha(marker) != pin['sha256']:
        raise ValueError('ADAPT5K_COMPLETE_PHASE_MARKER_HASH_CHANGED')
    data = read(marker)
    step = data['phase_step']
    if (data['schema'] != SCHEMA or data['binding'] != binding or data['progress'] != identity(binding,step)
            or data['training_generations'] != 4*step or data['physics'] is not False
            or data['raw_endpoint'] is not True or len(data['arrays']) != 140):
        raise ValueError('ADAPT5K_COMPLETE_OPTIMIZER_RNG_AND_PHASE_CLOCK_REQUIRED')
    return root,data


def restore(pin, *, parameters, optimizer, rng, binding):
    root, data = read_complete(pin,binding)
    values = shared.restore_arrays(root,data,shared.training_tree(parameters,optimizer,rng))
    return values,data['progress']


def next_stage_binding(previous, proposed, phase_step):
    """Permit one data transition while all optimizer and representation identities stay frozen."""
    identity(previous,phase_step)
    identity(proposed,phase_step)
    changed_at_boundary = {'stage_start','stage_stop','stage_target_data_sha256','stage_bank_sha256'}
    if (phase_step != contract.SWITCH_PHASE or (previous['stage_start'], previous['stage_stop']) != (0,phase_step)
            or (proposed['stage_start'], proposed['stage_stop']) != (phase_step,contract.UPDATES)
            or {k:v for k,v in previous.items() if k not in changed_at_boundary}
            != {k:v for k,v in proposed.items() if k not in changed_at_boundary}):
        raise ValueError('ADAPT5K_ONLY_D2_DATA_AND_VISIT_BANKS_MAY_CHANGE_AT_THE_SINGLE_COLD_2K_BOUNDARY')
    return proposed
