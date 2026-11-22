"""Bind a student case to its own accepted replay, geometry and bounded slice."""
import hashlib
import json
from pathlib import Path

from scripts.sla.core_hopper_expert import PLAN, POP

SCOPE = 'SLA_CORE_HOPPER_V1_STUDENT'
AUTHORITY = 'CURRENT_USER_SLA_CORE_HOPPER_PARALLEL_V1'
MAIN = (0,2000,5000,10000,20000)
PAIRED = (4000,7000,12000,17000)
ENDPOINTS = tuple(f'MAIN-phase{n:05d}' for n in MAIN+PAIRED) + tuple(f'CONTROL2000-phase{n:05d}' for n in PAIRED)


def read_pin(pin):
    data=Path(pin['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=pin['sha256']:
        raise ValueError('Student evidence changed: '+Path(pin['path']).name)
    return json.loads(data)


def accepted(spec):
    arm = spec['arm']
    site = read_pin(spec['receiver'])
    proof = read_pin(spec['student_proof'])
    gate = proof['gate']
    if arm == 'MAIN-phase03800':
        from scripts.sla.core_main3800_native_proof import check_binding
        check_binding(spec, proof, gate, site)
        return proof,gate,site
    if (arm not in ENDPOINTS or spec['scope'] != SCOPE or spec['plan_commit'] != PLAN
            or proof['plan'] != PLAN or proof['compiled_population'] != spec['compiled_population']
            or proof['core_profile'] != site['gpu'] or site['plan'] != PLAN or gate['exact'] != 16
            or proof['hardware_allowed'] or proof['physical_requests']):
        raise PermissionError('Missing endpoint qualification for the original student population')
    from scripts.sla.core_canonical_proof import VERIFIED, check_native_binding
    if gate['status'] == VERIFIED:
        check_native_binding(spec,proof,gate,site)
        return proof,gate,site
    if gate['status'] != 'ENDPOINT_RGB16_VERIFIED' or proof['model_generations']:
        raise PermissionError('Missing original strict endpoint qualification')
    root = Path(spec['receiver']['path']).parent/'endpoints'/arm
    package = spec['packages'][arm]
    replay = read_pin(gate['core_result'])
    receipt = read_pin(gate['receipt'])
    if (set(spec['packages']) != {arm} or package != dict(path=str(root/'payload'),sha256=gate['manifest'])
            or gate['core_result']['path'] != str(root/'RESULT.json') or gate['receipt']['path'] != str(root/'RECEIVED.json')
            or receipt['manifest'] != gate['manifest'] or receipt['checkpoint'] != gate['checkpoint']
            or replay['status'] != 'CORE_RGB16_PASS' or replay['exact'] != 16
            or replay['checkpoint'] != gate['checkpoint'] or replay['manifest'] != gate['manifest']
            or read_pin(gate['site_gpu'])['gpu'] != site['gpu']):
        raise PermissionError('Strict replay, native endpoint or GPU binding differs')
    return proof,gate,site


def validate_case(case):
    proof,gate,_ = accepted(case)
    authority = read_pin(case['sim_only_project_authorization'])
    split = case['evaluation_split']
    if (case['compiled_population']['sha256'] != POP or split not in proof['panels']
            or (split=='CONF' and case['arm']!='MAIN-phase20000') or case['expert_training_eligible']
            or case['ordinary_RGB_restore'] != gate['core_result'] or authority['student_proof'] != case['student_proof']
            or case['owner'] != authority['owner'] or authority['compiled_population_sha256'] != POP
            or authority['hardware_allowed'] or authority['native_request_cap'] != len(authority['case_ids'])
            or not 1 <= len(authority['case_ids']) <= 6 or case['panel_entry_id'] not in authority['case_ids']):
        raise PermissionError('Unregistered student case or bounded block')
    rows = proof['panels'][split]
    start = authority['panel_start']
    block = rows[start:start+len(authority['case_ids'])]
    if ([r['id'] for r in block] != authority['case_ids'] or sum(r['tick_cap'] for r in block) != authority['native_tick_cap']
            or sum(r['query_cap'] for r in block) != authority['generation_cap']):
        raise PermissionError('Student slice differs from its original order or budget')
    row, = (r for r in block if r['id']==case['panel_entry_id'])
    if case['arm'] == 'MAIN-phase03800':
        from scripts.sla.core_main3800_native_proof import check_case
        check_case(case, authority, proof)
    cond = case['mobile_condition']
    if (row['arm'] != case['arm'] or row['geometry'] != cond['layout'] or row['reset'] != cond['reset']
            or row['scene_seed'] != cond['seed'] or row['family'] != cond['family'] or row['instruction'] != case['task_instruction_id']
            or row['scene'] != dict(path=case['model_path'],sha256=case['frozen_files'][case['model_path']])
            or row['query_cap'] != case['policy_queries'] or row['tick_cap'] != case['max_ticks']):
        raise PermissionError('Student scene, clock, language or endpoint changed')
