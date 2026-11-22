"""Reconcile the two closed Core DEV6 blocks and compare their original native scores."""
import argparse
import json
from pathlib import Path
import time

import mujoco

from go2arx5_vla.world_model.multitask_terminal_audit_v1 import replay
from scripts.pi0292.durable import read
from scripts.sla.core_hopper_carry_audit import last_stop, stopped_pairs
from scripts.sla.hopper_parent_custody import digest, write_once

ARMS = ('MAIN-phase00000', 'MAIN-phase02000')
KEYS = (
    'task_instruction_id', 'multitask_geometry', 'max_ticks', 'policy_queries',
    'mobile_wbc_boundary_v1', 'evaluation_split',
)
SCORE = (
    'ever_target_grasp', 'ever_target_lift20mm', 'maximum_lift_m', 'longest_grasp_lift_hold_s',
    'base_directional_motion_m', 'inside_selected_region', 'released',
    'stable_release_condition', 'missing_native_intervals', 'unsafe_states', 'task_success',
)
EXECUTORS = ('model_path', 'native_binary', 'controller_binary', 'solver_binary',
             'outer_library', 'observer_library', 'trace_library')


def checked_block(root, arm):
    folder = root/'blocks'/f'{arm}-DEV-00'
    spec = folder/'EXECUTION_SPEC.json'
    block = folder/'RESULT.private.json'
    proof = read(folder/'RECONCILED.json')
    qualification = read(root/'endpoints'/f'{arm}.json')
    cards = list(map(Path,read(spec)['cases']))
    if (proof['block']!=dict(path=str(block),sha256=digest(block))
            or qualification['gate']['status']!='ENDPOINT_CORE_CANONICAL_VERIFIED'
            or qualification['gate']['arm']!=arm or qualification['model_generations']!=48
            or proof['requests']!=len(cards) or len(cards)!=6 or len(proof['records'])!=len(cards)):
        raise ValueError(f'{arm} has no immutable original DEV6 closeout')
    for card,sha in zip(cards,proof['records'],strict=True):
        if (card.parent.parent!=folder/'cases'
                or digest(card.parent/'RESULT.private.json')!=sha):
            raise ValueError(f'{arm} result differs from the confirmed original roster')
    return proof,qualification,cards


def pair_identity(a,b):
    if (any(a[k]!=b[k] for k in KEYS)
            or any(a['frozen_files'][a[k]]!=b['frozen_files'][b[k]] for k in EXECUTORS)):
        raise PermissionError('Paired evaluation changed the experiment or a frozen executor')
    x,y = [dict(z['mobile_condition']) for z in (a,b)]
    x.pop('collection_episode')
    y.pop('collection_episode')
    if x!=y:
        raise PermissionError('Paired evaluation changed the original scene or condition seed')
    return dict(instruction=a['task_instruction_id'], reset=x['reset'], seed=x['seed'],
                layout=x['layout'],scene_sha256=a['frozen_files'][a['model_path']],
                original_tick_cap=a['max_ticks'])


def one_case(path,spec,arm):
    case = path.parent
    result_file = case/'RESULT.private.json'
    native = read(result_file)
    if (spec['arm']!=arm or spec['registered_root']!=str(case) or spec['evaluation_split']!='DEV'
            or native['condition']['collection_episode']!=spec['panel_entry_id']
            or native['physical_requests']!=1 or native['technical_failure'] or native['optimizer_updates']):
        raise PermissionError('Expected a single original student request with a valid terminal')
    output = case/'native-01'
    detail = native['detail'] or {}
    terminal_tick = native['actual_native_integrations']
    audit = replay(spec,output/'state.bin.journal',terminal_tick,detail.get('score'))
    score = audit['score']
    early = detail.get('terminal_reason')
    hits = None
    if native['unsafe']:
        if native['trace_audit']['integrations_after_first_STOP']!=0:
            raise ValueError('Original STOP cannot have later integrations')
        stop = native['physics_stops'][0] if native['physics_stops'] else native['mobile_safety_stop']
        early = stop['reason']
        if early=='ILLEGAL_CONTACT':
            state = last_stop(output/'physics.jsonl')
            model = mujoco.MjModel.from_binary_path(spec['model_path'])
            target = 'task_cube_geom' if '-red' in spec['task_instruction_id'] else 'task_blue_cube_geom'
            hits = stopped_pairs(model,state,target)
            if state['step']!=terminal_tick or not hits:
                raise ValueError('Original STOP lacks its matching contact evidence')
    elif audit['differs_from_cache']:
        raise ValueError('Original terminal replay disagrees with its saved score')
    if not early or score['missing_native_intervals']:
        raise ValueError('Missing original result reason or continuous native state')
    return dict(arm=arm,condition=spec['panel_entry_id'],result_sha256=digest(result_file),
                journal_sha256=digest(output/'state.bin.journal'),first_failure=early,
                physical_requests=1,ticks=terminal_tick,model_generations=native['model_generations'],
                confirmed_atoms=native['metrics']['task_atoms'],partial_ticks=audit['final_partial_atom_ticks'],
                contact_pairs=hits,post_STOP_ticks=0 if native['unsafe'] else None,
                rejection=detail.get('valid_policy_rejection'),score={k:score[k] for k in SCORE},
                scoring_events=score['events'])


def compare(root):
    if mujoco.__version__!='3.2.7' or mujoco.get_mjcb_control() is not None:
        raise RuntimeError('Need the original geometry-only scoring runtime')
    ready = [checked_block(root,arm) for arm in ARMS]
    start = time.monotonic()
    pairs = []
    for left,right in zip(ready[0][2],ready[1][2],strict=True):
        specs = [read(p) for p in (left,right)]
        bound = pair_identity(*specs)
        outcomes = [one_case(card,spec,arm) for card,spec,arm in zip((left,right),specs,ARMS,strict=True)]
        pairs.append(dict(**bound,outcomes=outcomes))
    for i,arm in enumerate(ARMS):
        proof = ready[i][0]
        rows = [v['outcomes'][i] for v in pairs]
        if (sum(r['ticks'] for r in rows)!=proof['ticks']
                or sum(r['model_generations'] for r in rows)!=proof['generations']):
            raise RuntimeError(f'Actual {arm} denominators differ from the original block')
    return dict(status='CLOSED_ORIGINAL_CORE_DEV6_PAIR_RECONCILED', pairs=pairs,
                checkpoints={arm:ready[i][1]['gate']['checkpoint'] for i,arm in enumerate(ARMS)},
                blocks={arm:ready[i][0]['block'] for i,arm in enumerate(ARMS)},
                new_native_requests=0,new_physics_integrations=0,new_model_generations=0,
                optimizer_updates=0,elapsed_s=time.monotonic()-start)


if __name__=='__main__':
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument('root',type=Path)
    arg.add_argument('output',type=Path)
    parsed = arg.parse_args()
    result = compare(parsed.root.resolve(strict=True))
    write_once(parsed.output,result)
    print(json.dumps(dict(status=result['status'],sha256=digest(parsed.output)),sort_keys=True))
