"""Seal and restore all training state across H200 allocations, with role-specific clocks."""
import json
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter import scale20k_checkpoint as shared
from go2arx5_vla.sla_adapter.scale20k_contract import bank_sha256
from scripts.sla.hopper_inputs import BATCH, D2_SPEC, PARENT, PLAN, STEPS
from scripts.sla.hopper_parent_custody import digest

SCHEMA = 'HOPPER_ADAPT20K_COMPLETE_STATE_V1'
BANKS = 'c3c308dfc00517894735c0f7386dc2e79385e86717fa33f4e817b44fc0afadb0'
BOUNDARIES = (2000, 5000, 10000, 15000)
ENGINEERING = {f'E{i}' for i in range(1, 7)}


def binding(restored, *, role, source, fm_micro, aux_micro, role_start=0,
            target_origin=0, target_manifest=D2_SPEC):
    _, _, _, banks, _, _, proof = restored
    if (proof['plan'] != PLAN or proof['parent'] != PARENT or proof['banks_sha256'] != BANKS
            or proof['ancestor']['lineage_total_updates'] != 2500
            or proof['ancestor']['lineage_cursors'] != dict(target=144000, robodojo=88000, libero=88000)):
        raise ValueError('Wrong inherited state for the new 20K phase')
    if (len(source) != 40 or any(c not in '0123456789abcdef' for c in source)
            or fm_micro not in (1, 4, 8, 16) or aux_micro not in (1, 2, 4)):
        raise ValueError('Unrecognized backend or source')
    if role not in ENGINEERING | {'MAIN', 'CONTROL2000', 'CPU_ROUNDTRIP'}:
        raise ValueError('Unknown training or diagnostic identity')
    if ((role == 'CONTROL2000' and role_start not in BOUNDARIES)
            or (role != 'CONTROL2000' and role_start != 0)
            or target_origin not in (0, *BOUNDARIES) or not shared.is_digest(target_manifest)
            or (target_origin == 0) != (target_manifest == D2_SPEC)
            or (role in ENGINEERING | {'CPU_ROUNDTRIP'} and target_origin != 0)
            or (role == 'CONTROL2000' and target_origin >= role_start)):
        raise ValueError('Invalid role or target-data boundary')
    return dict(plan=PLAN, parent=PARENT, initial_banks=BANKS, source=source, role=role,
                role_start=role_start, target_origin=target_origin, target_manifest=target_manifest,
                target_bank_sha256=bank_sha256(banks['target']), rates_sha256=proof['rates_sha256'],
                frozen_model=proof['frozen_model_fingerprint'], rng_sha256=bank_sha256(banks['rng']),
                fm_micro=fm_micro, aux_micro=aux_micro)


def progress(bound, phase):
    role = bound['role']
    stop = STEPS
    if role in ENGINEERING:
        stop = 16
    elif role == 'CONTROL2000':
        stop = bound['role_start'] + 2000
    elif role == 'CPU_ROUNDTRIP':
        stop = 0
    if type(phase) is not int or not max(bound['role_start'], bound['target_origin']) <= phase <= stop:
        raise ValueError('Phase outside this role or data stage')
    prior = dict(target=144000, robodojo=88000, libero=88000)
    local_steps = phase - bound['role_start']
    return dict(phase_step=phase, lineage_updates=2500 + phase, role_updates=local_steps,
                role_visits={d: n * local_steps for d, n in BATCH.items()},
                lineage_cursors={d: prior[d] + n * phase for d, n in BATCH.items()},
                target_bank_cursor=(phase - bound['target_origin']) * BATCH['target'],
                role_training_full_generations=4 * local_steps)


def check_state(state, clocks, bound):
    import jax

    counts = {d: int(np.asarray(group[0].count)) for d, group in state['optimizer'].items()}
    if (set(counts) != {'shared', *BATCH} or set(counts.values()) != {clocks['lineage_updates']}
            or bank_sha256(np.asarray(state['rng'])) != bound['rng_sha256']
            or len(jax.tree.leaves(state)) != 140):
        raise ValueError('Stored Adam, RNG or parameter tree differs from its declared phase')


def save(path, state, *, phase, bound, predecessor):
    if not shared.is_digest(predecessor) or (phase == 0 and predecessor != PARENT):
        raise ValueError('A complete immutable predecessor is required')
    tree = shared.training_tree(state['parameters'], state['optimizer'], state['rng'])
    clocks = progress(bound, phase)
    check_state(tree, clocks, bound)
    return shared.seal_arrays(path, tree, dict(schema=SCHEMA, binding=bound, progress=clocks,
                                              predecessor=predecessor))


def resume(pin, template, *, phase, bound, predecessor=None):
    import jax

    root = Path(pin['path']).resolve(strict=True)
    if digest(root/'COMPLETE.json') != pin['sha256']:
        raise ValueError('Checkpoint seal differs from the registered input')
    marker = json.loads((root/'COMPLETE.json').read_bytes())
    clocks = progress(bound, phase)
    if (marker['schema'] != SCHEMA or marker['binding'] != bound or marker['progress'] != clocks
            or not shared.is_digest(marker['predecessor']) or len(marker['arrays']) != 140
            or (predecessor is not None and marker['predecessor'] != predecessor)):
        raise ValueError('Checkpoint role, configuration, predecessor or clocks changed')
    state = shared.restore_arrays(root, marker, template)
    jax.block_until_ready(state)
    check_state(state, clocks, bound)
    return state, clocks
