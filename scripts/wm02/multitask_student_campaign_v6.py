"""Frozen V6 matched screen and conditional single-candidate panels."""
from collections import Counter
import json
from pathlib import Path
from statistics import median

import numpy as np

from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.multitask_contract_v6 import ARMS, NATIVE_METHOD, contract
from go2arx5_vla.world_model import multisource_d4_native as d4
from go2arx5_vla.world_model import sla_followup_native_v1 as followup
from go2arx5_vla.world_model import scale20k_native as scale
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
from go2arx5_vla.world_model.multitask_scope_v6 import PLAN, SCOPE, read_pin, validate
from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS
from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.storage import digest
from .multitask_teacher_campaign_v6 import interface_records
from .multitask_panel_suffix import budgets, verified_prefix
from .sim_only_campaign_v51 import compact_case, fatal_control_violation, run as run_campaign

STAGE_SIZE = dict(SCREEN=24, CONFIRM=24, DEV=60, TEST=60, D4_DEV=24, J_SCREEN48=24, **dict.fromkeys(scale.CAPS,24))
PREDECESSOR = dict(CONFIRM='SCREEN',DEV='CONFIRM',TEST='DEV')


def read(path):
    return json.loads(Path(path).read_bytes())


def expected_conditions(population, stage, arms):
    if stage in scale.CAPS:
        return scale.conditions(population,stage,arms)
    if stage in followup.CAPS:
        return followup.conditions(population, stage, arms)
    if stage == d4.STAGE:
        if arms != set(d4.ARMS):
            raise ValueError('D4_REQUIRES_BOTH_SEALED_ARMS')
        return d4.conditions(population)
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
    if (set(spec['endpoints']) != set(spec['packages'])
            or (spec.get('scope') in (d4.SCOPE,scale.SCOPE) and set(spec['ordinary_RGB_restores']) != set(spec['packages']))):
        raise ValueError('EVERY_FROZEN_G7_ARM_REQUIRES_THE_ACTUAL_ENDPOINT')
    for arm, pin in spec['endpoints'].items():
        if spec['scope'] == scale.SCOPE:
            _, records[arm] = scale.endpoint(arm,spec['packages'][arm],pin,spec['ordinary_RGB_restores'][arm])
            continue
        if spec['scope'] == followup.SCOPE:
            _, records[arm] = followup.endpoint(
                arm, spec['packages'][arm], pin, spec['ordinary_RGB_restores'][arm],
                core_replay=spec['core_RGB_restores'].get(arm),
                source_restore=spec.get('source_RGB_restores', {}).get(arm))
            continue
        if spec.get('scope') == d4.SCOPE:
            _, records[arm] = d4.endpoint(arm, spec['packages'][arm], pin, spec['ordinary_RGB_restores'][arm])
            continue
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
    if spec['scope'] == scale.SCOPE:
        return scale.arms_for(spec)
    if spec['scope'] == followup.SCOPE:
        return set(followup.arms_for(spec))
    if spec['scope'] == d4.SCOPE:
        if (stage != d4.STAGE or spec['plan_commit'] != d4.PLAN or spec['physics'] is not True
                or spec['new_optimizer_updates_budget'] or spec['new_training_visits_budget']
                or arms != set(d4.ARMS) or spec['compiled_population']['sha256'] != d4.POPULATION_SHA):
            raise PermissionError('FIXED_D4_MATCHED_DEV_WITH_ZERO_TRAINING_REQUIRED')
        return arms
    if (stage not in STAGE_SIZE or stage == d4.STAGE or spec['scope'] != SCOPE or spec['plan_commit'] != PLAN
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
    staged = spec['scope'] in (d4.SCOPE,followup.SCOPE,scale.SCOPE)
    carried = []
    if staged:
        if spec['scope'] == scale.SCOPE:
            full_budget = scale.CAPS[stage]
        elif spec['scope'] == d4.SCOPE:
            full_budget = (d4.REQUEST_CAP,d4.TICK_CAP,d4.GENERATION_CAP)
        else:
            full_budget = followup.CAPS[stage]
        if budgets(expected) != full_budget:
            raise ValueError('FIXED_NATIVE_PANEL_CONDITION_BUDGET_CHANGED')
        carried = verified_prefix(spec,expected)
        expected = expected[len(carried):]
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
    if staged:
        actual_caps = len(cases),sum(read(path)['max_ticks'] for path in cases),requests
        if actual_caps != budgets(expected) or (
                spec.get('new_native_tick_budget',actual_caps[1]),spec['new_model_generations_budget']) != actual_caps[1:]:
            raise ValueError('ONLY_THE_FINITE_UNDISPATCHED_NATIVE_TICK_AND_GENERATION_CAPS_ARE_AVAILABLE')
    interface = read_pin(spec['interface_predecessor'])
    if spec['scope'] in (followup.SCOPE,scale.SCOPE):
        interfaces = interface_records(interface,spec['mobile_world_lock'],identity=(d4.E0_SCOPE,d4.E0_PLAN))
    else:
        interfaces = interface_records(interface,spec['mobile_world_lock'])
    return interfaces+carried, endpoints


def student_row(spec,result):
    row = compact_case(spec,result)
    root = Path(spec['registered_root'])/'native-01'
    detail = result.get('detail') or {}
    score = detail.get('score')
    recorded = root/'MULTITASK_SCORE_PROGRESS.private.json'
    if score is None and recorded.exists():
        score = read(recorded)
    replay = None
    if spec.get('scope') in (d4.SCOPE, followup.SCOPE, scale.SCOPE):
        last_tick = result['actual_native_integrations']
        if score is None or score['native_step'] != last_tick or score['missing_native_intervals']:
            from go2arx5_vla.world_model.multitask_terminal_audit_v1 import replay as audit_terminal

            try:
                replay = audit_terminal(spec, root/'state.bin.journal', last_tick, score)
            except (ValueError, OSError) as error:
                row.update(technical_failure=True, terminal_score_error=str(error))
            else:
                score = replay['score']
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
    rejection = detail.get('valid_policy_rejection') or {}
    if row['valid_policy_rejection']:
        row['kind'] = rejection.get('kind', row['first_failure'])
    if replay is not None:
        row['terminal_score_readback'] = replay
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
    valid_offline = passed = None
    if diagnostic is not None:
        original_passes = diagnostic.get('original_H40_and_G7_seam_pass',
                                         diagnostic.get('original_H40_and_gripper_speed_pass'))
        valid_offline = diagnostic['predictions']==96 and original_passes==96
        passed = (len(rows)==size and len(win)>=goal[0] and min(by_instruction.values())>=goal[1]
                  and min(by_family.values())>=goal[2] and not hard and not technical and valid_offline)
    rejected = [row['valid_policy_rejection'] for row in rows if row['valid_policy_rejection']]
    scores = [row['score'] for row in rows if row.get('score') is not None]
    events = ('target_grasp','target_lift20mm','base80mm','correct_region','released','retreated50mm',
              'wrong_object_grasp','wrong_object_contact','wrong_destination_release')
    return dict(passed=passed,attempts=len(rows),task_successes=len(win),success_by_instruction=by_instruction,
        success_by_family=by_family,hard_STOP=hard,technical_failures=technical,
        G7_seam_rejections=sum(row.get('kind')=='G7_ACTUAL_PREVIOUS_COMMAND_SPEED_REJECTED' for row in rejected),
        H40_or_seam_rejections=len(rejected),offline_qualified=valid_offline,
        cases_with_score_evidence=len(scores),cases_with_observed_event={k:sum(k in score['events'] for score in scores) for k in events},
        success_median_task_ticks=median(row['task_ticks'] for row in win) if win else None,
        by_scene_cluster={cluster:dict(attempts=sum(r['cluster']==cluster for r in rows),
                                     successes=sum(r['cluster']==cluster for r in win))
                          for cluster in sorted({r['cluster'] for r in rows})})


def summarize(result, spec, endpoints):
    stage = spec['evaluation_stage']
    is_d4 = stage == d4.STAGE
    is_followup = spec.get('scope') == followup.SCOPE
    is_scale = spec.get('scope') == scale.SCOPE
    arms = tuple(scale.arms_for(spec)) if is_scale else followup.arms_for(spec) if is_followup else d4.ARMS if is_d4 else ARMS
    students = [row for row in result['records'] if row['role']=='STUDENT']
    remaining = spec['new_physical_requests_budget']
    full = scale.CAPS[stage][0] if is_scale else followup.CAPS[stage][0] if is_followup else d4.REQUEST_CAP if is_d4 else remaining
    current = students[full-remaining:]
    if is_followup and stage != followup.JOINT:
        from go2arx5_vla.world_model.d5_grasp_decision import grasp_panel

        bank = read_pin(spec['compiled_population'])
        chosen = arms[-1] if stage == followup.CONFIRM else None
        decision = grasp_panel(students, bank['conditions'][stage], stage=stage, candidate=chosen)
        generation_count = sum(r['actual_model_generations'] for r in current)
        worker = result['model_service']
        reconciled = (worker is not None and not worker['exit_code']
                      and worker['attempted_generations'] == worker['completed_generations'] == generation_count)
        if result['exit_code'] or result['unresolved_case_launches'] or not reconciled:
            decision.update(complete_comparable=False, selected_arm=None, confirmation_goal_met=None)
        selected = decision['selected_arm']
        return dict(decision, population_sha256=spec['compiled_population']['sha256'],
                    selected_package_sha256=None if selected is None else spec['packages'][selected]['sha256'],
                    new_native_requests=sum(r['actual_native_requests'] for r in current),
                    total_panel_native_requests=sum(r['actual_native_requests'] for r in students),
                    new_native_ticks=sum(r['native_ticks'] for r in current),
                    total_panel_native_ticks=sum(r['native_ticks'] for r in students),
                    attributed_case_generations=generation_count, generations_reconciled=reconciled,
                    worker_result=worker, unresolved_case_launches=result['unresolved_case_launches'])
    totals = {arm:arm_totals([r for r in students if r['arm']==arm],stage,None if is_scale else ep['TRAIN96'])
              for arm,ep in endpoints.items()}
    pairs = {}
    for row in students:
        pairs.setdefault((row['cluster'],row['instruction']),{})[row['arm']] = row
    paired = []
    for (cluster,instruction), rows in pairs.items():
        if is_scale or set(rows)!=set(arms):
            continue
        a,b = (rows[arm]['initial_native_state'] for arm in arms)
        delta = None if a is None or b is None else {key:float(np.max(np.abs(np.asarray(a[key])-b[key]))) for key in a}
        key = 'J1_minus_J0_success' if is_followup else 'D4_B_minus_A_success' if is_d4 else 'G7_M_SRC_minus_M_BASE_success'
        paired.append(dict(cluster=cluster, instruction=instruction, actual_initial_difference_max_abs=delta,
            **{key: int(rows[arms[1]]['task_and_mobile_success'])-int(rows[arms[0]]['task_and_mobile_success'])}))
    candidates = [arm for arm, row in totals.items() if row['passed']]
    generations = sum(row['actual_model_generations'] for row in current)
    worker = result['model_service']
    reconciled = (worker is not None and not worker['exit_code']
                  and worker['attempted_generations']==worker['completed_generations']==generations)
    complete = (not result['exit_code'] and not result['unresolved_case_launches']
                and len(students)==full
                and reconciled
                and not any(fatal_control_violation(row) for row in students))
    selected = None
    if complete and candidates and not (is_d4 or is_followup or is_scale):
        def rank(arm):
            row = totals[arm]
            return (-row['task_successes'],-min(row['success_by_instruction'].values()),row['hard_STOP'],
                    row['H40_or_seam_rejections'],row['success_median_task_ticks'],arms.index(arm))
        selected = min(candidates,key=rank)
    count_requests = is_d4 or is_followup or is_scale
    return dict(stage=stage,passed=None if is_scale else selected is not None,complete_panel=complete,selected_arm=selected,
        selected_package=None if selected is None else spec['packages'][selected],
        population_sha256=spec['compiled_population']['sha256'],arms=totals,paired_conditions=paired,
        new_native_requests=(sum(r['actual_native_requests'] for r in current) if count_requests else len(students)),
        new_native_ticks=sum(r['native_ticks'] for r in current),
        total_panel_native_requests=(sum(r['actual_native_requests'] for r in students) if count_requests else len(students)),
        total_panel_native_ticks=sum(r['native_ticks'] for r in students),
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
    is_followup = spec['scope'] == followup.SCOPE
    matched_native = spec['scope'] in (d4.SCOPE,followup.SCOPE,scale.SCOPE)
    code = run_campaign(path,preflight=lambda _:interfaces,load_carried=lambda rows:rows,compact=student_row,
        bootstrap='v6_bootstrap.py',expected_count=len(spec['cases'])+len(interfaces),
        result_schema=('SLA_SCALE20K_NATIVE_PANEL_ACTUAL' if spec['scope'] == scale.SCOPE else
                       'SLA_D416_NATIVE_FOLLOWUP_ACTUAL' if is_followup else
                       'SLA_D4_MATCHED_DEV_ACTUAL' if spec['evaluation_stage'] == d4.STAGE
                       else 'SLA_V6_MULTITASK_STUDENT_PANEL_ACTUAL'),
        pilot_evaluate=lambda _:False,case_lead_s=1980,
        condition_group=len(spec['packages']) if matched_native else 1)
    file = root/'RESULT.private.json'
    actual = summarize(read(file),spec,endpoints)
    actual['campaign_result'] = dict(path=str(file),sha256=digest(file))
    write(root/'PANEL_DECISION.private.json',actual)
    return code
