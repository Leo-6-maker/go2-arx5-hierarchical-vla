"""Frozen V6 matched screen and conditional single-candidate panels."""
from collections import Counter
import json
from pathlib import Path
from statistics import median

import numpy as np

from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.multitask_contract_v6 import ARMS, NATIVE_METHOD, contract
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
from go2arx5_vla.world_model.multitask_scope_v6 import PLAN, SCOPE, read_pin, validate
from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS
from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.storage import digest
from .multitask_teacher_campaign_v6 import interface_records
from .sim_only_campaign_v51 import compact_case, fatal_control_violation, run as run_campaign

STAGE_SIZE = dict(SCREEN=24, CONFIRM=24, DEV=60, TEST=60)
PREDECESSOR = dict(CONFIRM='SCREEN',DEV='CONFIRM',TEST='DEV')


def read(path):
    return json.loads(Path(path).read_bytes())


def expected_conditions(population, stage, arms):
    rows = population['conditions'][stage]
    size = STAGE_SIZE[stage]
    if len(rows) != size or len({row['id'] for row in rows}) != size:
        raise ValueError('EXACT_NEW_M_FIXED_POPULATION_REQUIRED')
    if (Counter(row['instruction'] for row in rows) != Counter({task:size//6 for task in INSTRUCTIONS})
            or Counter(row['family'] for row in rows) != Counter({'F1':size//2,'F2':size//2})):
        raise ValueError('EXACT_FAMILY_AND_INSTRUCTION_DENOMINATOR_REQUIRED')
    picked = []
    for row in rows:
        order = row['arm_order'] if stage == 'SCREEN' else tuple(arms)
        if stage == 'SCREEN' and (len(order)!=len(ARMS) or set(order) != set(ARMS)):
            raise ValueError('ORIGINAL_SCREEN_ARM_INTERLEAVING_REQUIRED')
        picked.extend((row,arm) for arm in order if arm in arms)
    return picked


def endpoint_records(spec):
    records = {}
    if set(spec['endpoints']) != set(spec['packages']):
        raise ValueError('EVERY_FROZEN_G7_ARM_REQUIRES_THE_ACTUAL_ENDPOINT')
    for arm, pin in spec['endpoints'].items():
        actual = read_pin(pin)
        package = spec['packages'][arm]
        manifest = read_pin(dict(path=str(Path(package['path'])/'MANIFEST.private.json'),sha256=package['sha256']))
        contract(manifest)
        if (actual['arm'] != arm or actual['exit_code'] or actual['package'] != package
                or actual['new_updates'] != 1000 or actual['new_visits'] != 32000
                or actual['legacy_C20_visits'] != 6400 or actual['teacher_visits'] != 25600
                or actual['actual']['training_full_generations'] != 4000
                or not actual['full_restore']['bit_exact'] or actual['full_restore']['arrays'] != 87
                or manifest['bindings']['arm'] != arm
                or manifest['bindings']['compiled_population_sha256']!=spec['compiled_population']['sha256']):
            raise ValueError('ACTUAL_FULLY_SEALED_G7_ENDPOINT_REQUIRED')
        records[arm] = actual
    return records


def previous_panel(spec, stage):
    decision = read_pin(spec['predecessor_decision'])
    parent = read_pin(decision['campaign_result'])
    root = Path(decision['campaign_result']['path']).parent
    if (read(root/'control/LEASE.json')['status'] != 'CLOSED' or not decision['passed']
            or parent['exit_code'] or parent['unresolved_case_launches'] or decision['stage'] != PREDECESSOR[stage]
            or decision['population_sha256'] != spec['compiled_population']['sha256']
            or decision['selected_arm'] != spec['frozen_arm']
            or decision['selected_package'] != spec['packages'][spec['frozen_arm']]):
        raise ValueError('EXACT_ACTUAL_PREDECESSOR_PANEL_AND_SINGLE_FROZEN_ARM_REQUIRED')
    return decision


def student_arms(spec):
    stage = spec['evaluation_stage']
    arms = set(spec['packages'])
    if (stage not in STAGE_SIZE or spec['scope'] != SCOPE or spec['plan_commit'] != PLAN
            or not spec['physics'] or spec['new_optimizer_updates_budget'] or not arms.issubset(ARMS) or not arms):
        raise PermissionError('FIXED_FINITE_V6_MULTITASK_STUDENT_PANEL_ONLY')
    if stage == 'SCREEN' and arms != set(ARMS):
        source = read_pin(spec['source_no_transfer_result'])
        root = Path(spec['source_no_transfer_result']['path']).parent
        if (arms!={'M-BASE'} or read(root/'control/LEASE.json')['status']!='CLOSED'
                or (not source['exit_code'] and source['transfer_gate']['passed'])):
            raise ValueError('BASE_ONLY_SCREEN_REQUIRES_A_SEALED_SOURCE_NO_TRANSFER_OUTCOME')
    if stage != 'SCREEN':
        if arms != {spec['frozen_arm']}:
            raise ValueError('EXACTLY_ONE_FROZEN_POST_SCREEN_ARM_REQUIRED')
        previous_panel(spec,stage)
    return arms


def preflight(spec):
    stage = spec['evaluation_stage']
    arms = student_arms(spec)
    endpoints = endpoint_records(spec)
    population = read_pin(spec['compiled_population'])
    expected = expected_conditions(population,stage,arms)
    panel = read_pin(spec['registered_panel'])
    cases = spec['cases']
    if len(panel) != len(cases) or len(panel) != len(expected) or len(panel) != spec['new_physical_requests_budget']:
        raise ValueError('COMPLETE_PREREGISTERED_PANEL_WITHOUT_PREFIX_SELECTION')
    clusters = {row['id']:row for row in population['clusters']}
    requests = 0
    for path, row, (expected_row, arm) in zip(cases,panel,expected,strict=True):
        case = read(path)
        validate(case)
        cluster = clusters[expected_row['cluster']]
        condition = row['condition']
        if (row['id'] != f"{expected_row['id']}-{arm}" or row['arm'] != arm or row['role'] != 'STUDENT'
                or row['instruction'] != expected_row['instruction'] or row['scene'] != cluster['scene']
                or row['geometry'] != cluster['scoring_geometry'] or condition['layout'] != expected_row['cluster']
                or condition['reset'] != expected_row['reset'] or condition['seed'] != expected_row['noise_seed']
                or condition['family'] != expected_row['family'] or condition['method'] != NATIVE_METHOD
                or case['registered_panel'] != spec['registered_panel'] or case['panel_entry_id'] != row['id']
                or case['mobile_group'] != spec['registered_root'] or case['source_sha'] != spec['source_sha']
                or case['mobile_world_lock'] != spec['mobile_world_lock'] or case['ROS_DOMAIN_ID'] != spec['ROS_DOMAIN_ID']
                or case['resident_server_spec'] != spec['resident_server_spec'] or case['packages'] != spec['packages']):
            raise ValueError('M_CASE_CARD_MUST_MATCH_SEALED_SCENE_RESET_NOISE_INSTRUCTION_AND_OWNER')
        requests += case['policy_queries']
    worker = read_pin(spec['resident_server_spec'])
    if (worker['new_complete_generations_budget'] != requests or worker['unified_requests_budget'] != requests
            or worker['new_optimizer_updates_budget'] or requests > 34560):
        raise ValueError('M_MODEL_REQUEST_BUDGET_MUST_MATCH_ACTUAL_HOLD_PLACE_CAPS')
    interface = read_pin(spec['interface_predecessor'])
    return interface_records(interface,spec['mobile_world_lock']), endpoints


def student_row(spec,result):
    row = compact_case(spec,result)
    root = Path(spec['registered_root'])/'native-01'
    detail = result.get('detail') or {}
    score = detail.get('score')
    recorded = root/'MULTITASK_SCORE_PROGRESS.private.json'
    if score is None and recorded.exists():
        score = read(recorded)
    initial = None
    journal = root/'state.bin.journal'
    if journal.exists():
        for packet in packets(journal):
            if packet['step'] >= 5703:
                if packet['step'] == 5703:
                    initial = {key:packet[key].tolist() for key in ('qpos','qvel','ctrl')}
                break
    row.update(instruction=spec['task_instruction_id'],cluster=spec['mobile_condition']['layout'],
        score=score,initial_native_state=initial,actual_model_generations=result['model_generations'])
    return row


def arm_totals(rows, stage, diagnostic):
    size = STAGE_SIZE[stage]
    win = [row for row in rows if row['completed'] and row['task_and_mobile_success']
           and not row['hard_STOP'] and not row['technical_failure']]
    by_instruction = {task:sum(row['instruction']==task for row in win) for task in INSTRUCTIONS}
    by_family = {family:sum(row['family']==family for row in win) for family in ('F1','F2')}
    hard = sum(row['hard_STOP'] for row in rows)
    technical = sum(row['technical_failure'] for row in rows)
    goal = (18,3,9) if size==24 else (48,8,24)
    valid_offline = diagnostic['predictions']==96 and diagnostic['original_H40_and_G7_seam_pass']==96
    passed = (len(rows)==size and len(win)>=goal[0] and min(by_instruction.values())>=goal[1]
              and min(by_family.values())>=goal[2] and not hard and not technical and valid_offline)
    rejected = [row['valid_policy_rejection'] for row in rows if row['valid_policy_rejection']]
    scores = [row['score'] for row in rows if row.get('score') is not None]
    events = ('target_grasp','target_lift20mm','base80mm','correct_region','released','retreated50mm',
              'wrong_object_grasp','wrong_object_contact','wrong_destination_release')
    return dict(passed=passed,attempts=len(rows),task_successes=len(win),success_by_instruction=by_instruction,
        success_by_family=by_family,hard_STOP=hard,technical_failures=technical,
        G7_seam_rejections=sum(row['kind']=='G7_ACTUAL_PREVIOUS_COMMAND_SPEED_REJECTED' for row in rejected),
        H40_or_seam_rejections=len(rejected),offline_qualified=valid_offline,
        cases_with_score_evidence=len(scores),cases_with_observed_event={k:sum(k in score['events'] for score in scores) for k in events},
        success_median_task_ticks=median(row['task_ticks'] for row in win) if win else None,
        by_scene_cluster={cluster:dict(attempts=sum(r['cluster']==cluster for r in rows),
                                     successes=sum(r['cluster']==cluster for r in win))
                          for cluster in sorted({r['cluster'] for r in rows})})


def summarize(result, spec, endpoints):
    stage = spec['evaluation_stage']
    students = [row for row in result['records'] if row['role']=='STUDENT']
    totals = {arm:arm_totals([r for r in students if r['arm']==arm],stage,ep['TRAIN96']) for arm,ep in endpoints.items()}
    pairs = {}
    for row in students:
        pairs.setdefault((row['cluster'],row['instruction']),{})[row['arm']] = row
    paired = []
    for (cluster,instruction), rows in pairs.items():
        if set(rows)!=set(ARMS):
            continue
        a,b = (rows[arm]['initial_native_state'] for arm in ARMS)
        delta = None if a is None or b is None else {key:float(np.max(np.abs(np.asarray(a[key])-b[key]))) for key in a}
        paired.append(dict(cluster=cluster,instruction=instruction,actual_initial_difference_max_abs=delta,
            G7_M_SRC_minus_M_BASE_success=int(rows['M-SRC']['task_and_mobile_success'])-int(rows['M-BASE']['task_and_mobile_success'])))
    candidates = [arm for arm, row in totals.items() if row['passed']]
    generations = sum(row['actual_model_generations'] for row in students)
    worker = result['model_service']
    reconciled = (worker is not None and not worker['exit_code']
                  and worker['attempted_generations']==worker['completed_generations']==generations)
    complete = (not result['exit_code'] and not result['unresolved_case_launches']
                and len(students)==spec['new_physical_requests_budget']
                and reconciled
                and not any(fatal_control_violation(row) for row in students))
    selected = None
    if complete and candidates:
        def rank(arm):
            row = totals[arm]
            return (-row['task_successes'],-min(row['success_by_instruction'].values()),row['hard_STOP'],
                    row['H40_or_seam_rejections'],row['success_median_task_ticks'],ARMS.index(arm))
        selected = min(candidates,key=rank)
    return dict(stage=stage,passed=selected is not None,complete_panel=complete,selected_arm=selected,
        selected_package=None if selected is None else spec['packages'][selected],
        population_sha256=spec['compiled_population']['sha256'],arms=totals,paired_conditions=paired,
        new_native_requests=len(students),new_native_ticks=sum(r['native_ticks'] for r in students),
        attributed_case_generations=generations,generations_reconciled=reconciled,
        worker_result=worker,unresolved_case_launches=result['unresolved_case_launches'],
        original_WBC_frozen=True,hardware_requests=0,new_optimizer_updates=0)


def run(path,spec):
    root = Path(spec['registered_root'])
    try:
        interfaces,endpoints = preflight(spec)
    except (KeyError, FileNotFoundError, ValueError, PermissionError) as error:
        paths = [*spec['cases'], spec['resident_server_spec']['path']]
        for pinned in paths:
            file = Path(pinned)
            child_root = file.parent
            lease_path = child_root/'control/LEASE.json'
            if (not child_root.resolve().is_relative_to(root.resolve()) or child_root == root
                    or not file.is_file() or not lease_path.is_file()):
                continue
            child = read(file)
            lease = read(lease_path)
            if (child['registered_root']==str(child_root) and lease['spec_sha256']==digest(file)
                    and lease['status']=='ACTIVE' and not (child_root/'SUPERVISOR.private.json').exists()):
                close(child,0,not_started=True,reason='MULTITASK_STUDENT_PRECONDITION')
        write(root/'RESULT.private.json',dict(exit_code=1,status='M_STUDENT_PRECONDITION_NOT_DISPATCHED',
            failure=repr(error),source_sha=spec['source_sha'],records=[],new_verified_native_requests=0,
            new_native_ticks=0,new_model_generations=0,new_optimizer_updates=0))
        close(spec,1)
        return 1
    code = run_campaign(path,preflight=lambda _:interfaces,load_carried=lambda rows:rows,compact=student_row,
        bootstrap='v6_bootstrap.py',expected_count=len(spec['cases'])+2,
        result_schema='SLA_V6_MULTITASK_STUDENT_PANEL_ACTUAL',pilot_evaluate=lambda _:False,case_lead_s=1980)
    file = root/'RESULT.private.json'
    actual = summarize(read(file),spec,endpoints)
    actual['campaign_result'] = dict(path=str(file),sha256=digest(file))
    write(root/'PANEL_DECISION.private.json',actual)
    return code
