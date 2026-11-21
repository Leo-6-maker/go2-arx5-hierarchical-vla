"""New PR425 single-world population, using the original mobile runtime."""
import json
from pathlib import Path

from go2arx5_vla.world_model.storage import digest
from scripts.sla.core_hopper_expert import CELLS, OWNER
from scripts.sla.core_multienv_geometry import POPULATION_SHA, load_population

PLAN = '4fe2c8cf1b69c168b2e3079e061bb7c702089cd3'
SCOPE = 'SLA_CORE_MULTIENV_V1_EXPERT'
AUTHORITY = 'CURRENT_USER_SLA_CORE_MULTIENV_DATA_V1'


def compile_wave(logical, geometry, wave):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

    doc = load_population(Path(logical['path']))
    if logical['sha256'] != POPULATION_SHA or wave not in ('A','B'):
        raise ValueError('Wrong preregistered population or wave')
    report = read_pin(geometry)
    if (report['status'] != 'COLD_GEOMETRY_PASS' or report['population_sha256'] != POPULATION_SHA
            or report['native_requests'] or report['new_integrations']
            or report['conditions'] != 528 or len(report['scenes']) != 88):
        raise ValueError('Cold compilation did not reconcile the original population')
    outputs = {x['id']:x for x in report['scenes']}
    if len(outputs) != 88:
        raise ValueError('Duplicate compiled geometry')
    clusters = []
    for original in doc['scenes']:
        row = outputs[original['id']]
        if any(row[k] != v for k,v in original.items()):
            raise ValueError('Numeric scene did not retain its preregistered identity')
        if row['wave'] != wave or not row['compiled']:
            continue
        scene = row['compiled']['manifest']
        if digest(Path(scene['path'])) != scene['sha256']:
            raise ValueError('Compiled native scene changed')
        clusters.append(dict(id=row['id'],reset=row['reset_seed'],family=row['family'],split=row['split'],
            scene_seed=row['scene_seed'],scene=scene,scoring_geometry=row['compiled']['scoring_geometry']))
    bound = {x['id']:x for x in clusters}
    result=dict(plan=PLAN,wave=wave,logical=logical,geometry=geometry,
                old_TEST_reused=False,assignment_uses_outcomes=False,clusters=clusters,conditions={},unbound=[])
    for split in ('TRAIN','VAL'):
        rows=[]
        for card in doc['collection']:
            if card['split'] != split or card['wave'] != wave:
                continue
            if card['scene'] not in bound:
                result['unbound'].append(dict(id=card['id'],status='STATIC_UNBOUND'))
                continue
            scene=bound[card['scene']]
            rows.append(dict(id=card['id'],condition_id=card['id'],cluster=scene['id'],family=card['family'],
                split=split,instruction=card['instruction'],scene_family=scene['id'],reset=scene['reset'],
                scene_seed=scene['scene_seed'],native_max_ticks=card['max_native_ticks'],
                max_queries_or_generations=card['max_teacher_queries']))
        result['conditions'][split]=rows
    signature=[x['scene']['sha256'] for x in clusters]
    if len(set(signature)) != len(signature):
        raise ValueError('The compiled wave contains repeated scenes')
    return result


def order(pop):
    if (pop['plan']!=PLAN or pop['old_TEST_reused'] or pop['assignment_uses_outcomes']
            or pop['logical']['sha256']!=POPULATION_SHA or pop['wave'] not in ('A','B')):
        raise ValueError('Unexpected PR425 wave')
    counts=(96,24) if pop['wave']=='A' else (384,24)
    plans=[]
    for split,limit in zip(('TRAIN','VAL'),counts,strict=True):
        bank=pop['conditions'][split]
        unbound=sum(row['id'].startswith(f'MEV1-{split}-') for row in pop['unbound'])
        if len(bank)+unbound!=limit or len({r['id'] for r in bank})!=len(bank):
            raise ValueError('A bound or static-unbound condition is missing')
        # Keep the original family/instruction rotation and fixed scene order per cell.
        buckets={cell:[r for r in bank if (r['family'],r['instruction'])==cell] for cell in CELLS}
        for index in range(max(map(len,buckets.values()),default=0)):
            plans.extend(rows[index] for rows in buckets.values() if index<len(rows))
    return plans


def validate_member(case, authority):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

    ref=case['compiled_population']
    if (case['owner']!=OWNER or case['registry_scope']!=SCOPE or case['sim_only_role']!='EXPERT'
            or case['resident_server_spec'] is not None or case['new_model_generations_budget']
            or authority['source']!=AUTHORITY or authority['owner']!=OWNER
            or authority['source_sha']!=case['source_sha'] or authority['compiled_population']!=ref
            or authority['compiled_population_sha256']!=ref['sha256']
            or authority['frozen_producer']!=case['source_sha']
            or authority['deadline_epoch']>authority['goal_start']+7*86400
            or authority['wave']!='A'):
        raise PermissionError('Unexpected PR425 expert site or scope')
    pop=read_pin(ref)
    ordered=order(pop)
    if ([r['id'] for r in ordered]!=authority['case_ids'] or authority['native_request_cap']!=len(ordered)
            or authority['native_tick_cap']!=sum(r['native_max_ticks'] for r in ordered)):
        raise PermissionError('Wave differs from the fixed request and tick budget')
    row=next(r for r in ordered if r['id']==case['panel_entry_id'])
    cluster=next(r for r in pop['clusters'] if r['id']==row['cluster'])
    native=case['mobile_condition']
    if (row['split']!=case['collection_split'] or row['instruction']!=case['task_instruction_id']
            or row['native_max_ticks']!=case['max_ticks'] or row['max_queries_or_generations']!=case['policy_queries']
            or row['reset']!=native['reset'] or row['scene_seed']!=native['seed'] or row['family']!=native['family']
            or cluster['id']!=native['layout'] or cluster['scoring_geometry']!=case['multitask_geometry']
            or cluster['scene']!=dict(path=case['model_path'],sha256=case['frozen_files'][case['model_path']])
            or case['expert_training_eligible']!=(row['split']=='TRAIN')):
        raise PermissionError('Case changed its precompiled geometry, role or clock')


def failure_signature(case, record):
    """Group only the same evidenced stage and contact bodies at the first STOP."""
    if record['first_failure']!='ILLEGAL_CONTACT':
        return None
    import mujoco
    from scripts.sla.core_hopper_carry_audit import last_stop, stopped_pairs
    path=Path(case['registered_root'])/'native-01'
    try:
        stop=last_stop(path/'physics.jsonl')
        queries=json.loads((path/'LIVE_DATA/QUERIES.private.json').read_bytes())
        if stop['step']!=record['ticks'] or not queries or queries[-1]['native_step']>stop['step']:
            return None
        phase=queries[-1].get('phase')
        model=mujoco.MjModel.from_binary_path(case['model_path'])
        target='task_cube_geom' if '-red' in case['task_instruction_id'] else 'task_blue_cube_geom'
        pairs=stopped_pairs(model,stop,target)
    except (OSError, ValueError, KeyError, IndexError, StopIteration):
        return None
    if not phase or not pairs:
        return None
    body_pairs={tuple(sorted(p['bodies'])) for p in pairs}
    return 'ILLEGAL_CONTACT',phase,tuple(sorted(body_pairs))
