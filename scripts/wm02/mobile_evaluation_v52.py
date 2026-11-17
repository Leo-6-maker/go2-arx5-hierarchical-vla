"""Finite CONFIRM/DEV/TEST successors of the actual V5.2 three-arm screen.

Preparation is a once-only registration action, not a background scheduler.
The old live source is never changed. Each stage rechecks its real predecessor
and uses the original native controller/worker lifecycle and safety contract.
"""
from pathlib import Path
import time

from scripts.pi0292.durable import write
from scripts.wm02.sim_only_campaign_v51 import (
    carried_rows, compact_case, fatal_control_violation, interface_gate, run as original,
)
from scripts.wm02.mobile_e1_followup_v52 import common, pin, site_binding
from go2arx5_vla.sla_adapter.mobile_contract_v52 import PLAN
from go2arx5_vla.sla_adapter.mobile_progression_v52 import (
    authorize_next, confirmation_decision, generalization_decision, screen_decision,
)
from go2arx5_vla.world_model.mobile_scope_v52 import read, validate
from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs


def verified_terminal(spec_pin, result_pin, expected):
    parent = read(spec_pin)
    result = read(result_pin)
    lease = read(pin(Path(parent['registered_root'])/'control/LEASE.json'))
    if (result['source_sha'] != parent['source_sha'] or
            result['attempted_case_launches'] != expected or
            result['new_verified_native_requests'] != expected or
            result['unresolved_case_launches'] != 0 or result['unstarted_cases'] != 0 or
            len(parent['cases']) != expected or
            Path(result_pin['path']).parent != Path(parent['registered_root']) or
            lease['status'] != 'CLOSED' or lease['spec_sha256'] != spec_pin['sha256']):
        raise ValueError('COMPLETE_CLOSED_ACTUAL_PREDECESSOR_REQUIRED')
    students = [r for r in result['records'] if r['role'] == 'STUDENT']
    saved = {r['id']: r for r in students}
    rows = []
    for path in parent['cases']:
        case = read(pin(path))
        case_lease = read(pin(Path(case['registered_root'])/'control/LEASE.json'))
        native = read(pin(Path(case['registered_root'])/'RESULT.private.json'))
        row = compact_case(case, native)
        if (case['source_sha'] != parent['source_sha'] or
                native['source_sha'] != case['source_sha'] or saved.get(row['id']) != row or
                case_lease['status'] != 'CLOSED' or case_lease['spec_sha256'] != pin(path)['sha256'] or
                type(row['native_ticks']) is not int or not 0 <= row['native_ticks'] <= 35703 or
                fatal_control_violation(row)):
            raise ValueError('SEALED_ACTUAL_CASE_IDENTITY_OR_CONTROL_VIOLATION')
        rows.append(row)
    if len(saved) != expected or len(students) != expected:
        raise ValueError('NO_MISSING_OR_DUPLICATE_PREDECESSOR_EPISODES')
    return parent, rows


def evidence(spec):
    """Derive the admission from real results; never trust a manual passed flag."""
    from scripts.wm02.mobile_offline_report_v52 import run as offline_report
    screen_spec, rows = verified_terminal(spec['E2_spec'], spec['E2_result'], 24)
    followup = read(spec['E2_followup_spec'])
    if (followup['source_sha'] != screen_spec['source_sha'] or
            spec['E2_spec']['path'] != str(Path(followup['registered_root'])/'E2/EXECUTION_SPEC.json')):
        raise ValueError('EXACT_ACTUAL_E1_E2_FOLLOWUP_REQUIRED')
    if (spec['plan_commit'] != PLAN or spec['scope'] != 'SLA_MOBILE_F2_V52' or
            any(spec[k] != followup[k] for k in ('allocation_deadline_epoch', 'campaign_deadline_epoch')) or
            spec['deadline_epoch'] > min(spec['allocation_deadline_epoch'], spec['campaign_deadline_epoch']) or
            not 0 < spec['deadline_epoch'] - time.time() <= 12*3600):
        raise PermissionError('UNCHANGED_ALLOCATION_CAMPAIGN_AND_FINITE_SERVICE_DEADLINES')
    offline = offline_report(followup)
    selected = screen_decision(rows, {a: d['promotion_offline_gates']
        for a, d in offline['by_arm'].items()})
    arm = spec['frozen_arm']
    prepared = read(spec['prepared_result'])
    split = read(prepared['split'])
    if prepared['split'] != followup['evaluation_split']:
        raise ValueError('ORIGINAL_PRETRAINING_SPLIT_PIN_REQUIRED')
    confirmed = development = None
    if spec['evaluation_stage'] in ('DEV', 'TEST'):
        confirmation_spec, confirmation_rows = verified_terminal(
            spec['confirmation_spec'], spec['confirmation_result'], 8)
        if (confirmation_spec['frozen_arm'] != arm or
                confirmation_spec['E2_spec'] != spec['E2_spec'] or
                confirmation_spec['E2_result'] != spec['E2_result']):
            raise ValueError('SAME_FROZEN_SCREEN_CANDIDATE_CONFIRMATION_REQUIRED')
        confirmed = confirmation_decision(confirmation_rows, read(prepared['confirmation']), arm)
    if spec['evaluation_stage'] == 'TEST':
        development_spec, development_rows = verified_terminal(spec['DEV_spec'], spec['DEV_result'], 32)
        if (development_spec['frozen_arm'] != arm or
                development_spec['confirmation_spec'] != spec['confirmation_spec'] or
                development_spec['confirmation_result'] != spec['confirmation_result'] or
                development_spec['prepared_result'] != spec['prepared_result']):
            raise ValueError('SAME_CANDIDATE_AND_SPLIT_PASSED_DEV_REQUIRED')
        development = generalization_decision(development_rows, split, 'DEV', arm)
    admission = authorize_next(spec['evaluation_stage'], arm, screen=selected,
        confirmation=confirmed, development=development)
    packages = read(screen_spec['resident_server_spec'])['packages']
    if packages != spec['packages']:
        raise ValueError('FROZEN_ACTUAL_SCREEN_MODEL_PACKAGES_REQUIRED')
    templates = {}
    for path in screen_spec['cases']:
        case = read(pin(path))
        if case['arm'] == arm:
            templates.setdefault(case['mobile_condition']['family'], (case, pin(path)))
    if set(templates) != {'F1', 'F2'}:
        raise ValueError('BOTH_FAMILY_TEMPLATES_OF_THE_FROZEN_CANDIDATE_REQUIRED')
    if spec['evaluation_stage'] == 'CONFIRM':
        panel = [dict(r, arm=arm) for r in read(prepared['confirmation'])]
    else:
        panel = [dict(r, id=r['episode_id'], layout=r['layout_id'], arm=arm,
            template_spec=templates[r['family']][1])
            for r in split['episodes'] if r['split'] == spec['evaluation_stage']]
    if len(panel) != admission['requests']:
        raise ValueError('EXACT_PREDECLARED_SUCCESSOR_PANEL_REQUIRED')
    return dict(admission=admission, screen=selected, confirmation=confirmed,
        development=development), panel, templates, screen_spec, split


def preflight(spec):
    proof, panel, _, _, _ = evidence(spec)
    count = proof['admission']['requests']
    if (spec['plan_commit'] != PLAN or spec['scope'] != 'SLA_MOBILE_F2_V52' or
            spec['new_physical_requests_budget'] != count or
            spec['new_optimizer_updates_budget'] != 0 or len(spec['cases']) != count or
            read(spec['registered_panel']) != panel):
        raise PermissionError('FINITE_UNIQUE_CANDIDATE_NATIVE_STAGE_ONLY')
    for path, entry in zip(spec['cases'], panel, strict=True):
        case = read(pin(path))
        validate(case)
        if (case['arm'] != spec['frozen_arm'] or case['evaluation_stage'] != spec['evaluation_stage'] or
                case['panel_entry_id'] != entry['id'] or
                case['python'] != read(entry['template_spec'])['python'] or
                case['mobile_world_lock'] != spec['mobile_world_lock'] or
                case['mobile_group'] != spec['registered_root'] or
                case['source_sha'] != spec['source_sha'] or case['registered_panel'] != spec['registered_panel']):
            raise ValueError('EXACT_FROZEN_CANDIDATE_NATIVE_INTERPRETER_AND_OWNER')
    prior = spec['prior_case_results']
    if len(prior) != 2 or not interface_gate(carried_rows(prior)):
        raise ValueError('REUSE_TWO_PASSED_INTERFACES_WITHOUT_NEW_REQUESTS')
    return prior


def prepare(spec):
    """Register one new stage after its actual gate; caller uses existing onceIO."""
    proof, panel, templates, screen_spec, _ = evidence(spec)
    stage = spec['evaluation_stage']
    count = proof['admission']['requests']
    root = Path(spec['registered_root'])
    dirs(root)
    (root/'cases').mkdir()
    write(root/'STAGE_ADMISSION_ACTUAL.private.json', proof)
    write(root/'REGISTERED_PANEL.private.json', panel)
    panel_pin = pin(root/'REGISTERED_PANEL.private.json')
    authority = root/'PROJECT_AUTHORIZATION.private.json'
    write(authority, dict(schema='USER_PROJECT_AUTHORIZED_ISOLATED_MUJOCO_V52',
        source='CURRENT_USER_EXPLICIT_PR404_V52_INSTRUCTION', source_sha=spec['source_sha'],
        plan_commit=PLAN, hardware_allowed=False, deadline_epoch=spec['deadline_epoch'],
        new_native_requests_max=count, full_H40_P10_TTL_STOP_unchanged=True))
    server_root = root/'worker'
    dirs(server_root)
    (server_root/'ipc').mkdir()
    server = dict(read(screen_spec['resident_server_spec']))
    server.update(common(spec, server_root, 'SLA_MOBILE_V52_'+stage+'_SINGLE_MODEL_POOL'))
    server.update(campaign_root=str(root), physics=False, new_physical_requests_budget=0,
        new_complete_generations_budget=count*120, new_model_generations_budget=count*120,
        unified_requests_budget=count*120, ipc_root=str(server_root/'ipc'), live_root=str(server_root),
        max_walltime_s=12*3600, max_gpu_hours=12)
    server['request_bindings_by_arm'] = {a: dict(b, source_sha=spec['source_sha'])
        for a, b in server['request_bindings_by_arm'].items()}
    for key in ('native_environment', 'gpu_environment'):
        server[key] = dict(server[key], TMPDIR=str(server_root/'tmp'),
            XDG_CACHE_HOME=str(server_root/'cache'), ROS_LOG_DIR=str(server_root/'logs/ros'))
    for name in ('tmp', 'cache', 'logs', 'data'):
        server[name] = str(server_root/name)
    server['site_binding'] = site_binding(server, server_root, root,
        permission='CURRENT_USER_EXPLICIT_PR404_V52_CONDITIONAL_'+stage+'_AFTER_ACTUAL_GATE')
    register(spec, [(server_root, server)])
    server_pin = pin(server_root/'EXECUTION_SPEC.json')
    entries = []
    for row in panel:
        folder = root/'cases'/row['id']
        dirs(folder)
        template, _ = templates[row['family']]
        case = dict(template)
        old_model = case['model_path']
        case.update(common(spec, folder, 'SLA_MOBILE_V52_'+stage+'_'+spec['frozen_arm']))
        case.update(python=read(row['template_spec'])['python'],
            evaluation_stage=stage, mobile_group=str(root), campaign_root=str(root),
            arm=spec['frozen_arm'], SHIFT7_package=spec['packages'][spec['frozen_arm']],
            registered_panel=panel_pin, panel_entry_id=row['id'], resident_server_spec=server_pin,
            sim_only_project_authorization=pin(authority), model_path=row['scene']['path'],
            language=row['language'], physics=True, new_physical_requests_budget=1,
            unattended_control=str(root/'control'), cpus=[10, 11])
        case['frozen_files'] = dict(case['frozen_files'])
        case['frozen_files'].pop(old_model, None)
        case['frozen_files'][case['model_path']] = row['scene']['sha256']
        case['mobile_condition'] = dict(case['mobile_condition'], layout=row['layout'],
            reset=row['reset'], seed=row['noise_seed'], collection_episode=row['id'],
            scene_lineage=stage+'-'+row['layout'])
        if 'initial_state_cluster' in row:
            case['initial_state_cluster'] = row['initial_state_cluster']
        # Keep the original native NumPy/MuJoCo/ROS profile; only the neural
        # worker and its controller run in sla-pt.
        case['native_environment'] = dict(case['native_environment'], TMPDIR=str(folder/'tmp'),
            XDG_CACHE_HOME=str(folder/'cache'), ROS_LOG_DIR=str(folder/'logs/ros'))
        for name in ('tmp', 'cache', 'logs', 'data'):
            case[name] = str(folder/name)
        entries.append((folder, case))
    parent = dict(screen_spec)
    parent.update(spec)
    parent.update(common(spec, root, 'SLA_MOBILE_V52_'+stage+'_CONTROLLER'))
    parent.update(physics=True, cpus=[10, 11, 12, 13],
        new_physical_requests_budget=count, registered_panel=panel_pin,
        new_complete_generations_budget=count*120, new_model_generations_budget=count*120,
        generation_budget_category='NATIVE_POLICY',
        resident_server_spec=server_pin, cases=[str(p/'EXECUTION_SPEC.json') for p, _ in entries])
    register(spec, [(root, parent), *entries])
    preflight(parent)
    return parent


def run(path):
    spec = read(pin(path))
    proof, panel, _, _, split = evidence(spec)
    count = proof['admission']['requests']
    stage, arm = spec['evaluation_stage'], spec['frozen_arm']
    def evaluate(rows):
        students = [r for r in rows if r['role'] == 'STUDENT']
        if len(students) != count:
            return False
        decision = (confirmation_decision(students, panel, arm) if stage == 'CONFIRM'
            else generalization_decision(students, split, stage, arm))
        write(Path(spec['registered_root'])/'STAGE_DECISION_ACTUAL.private.json', decision)
        return decision['passed']
    return original(path, preflight=preflight, bootstrap='mobile_v52_bootstrap.py',
        expected_count=count+2, result_schema='SLA_MOBILE_V52_'+stage+'_ACTUAL_TERMINAL',
        pilot_evaluate=evaluate)
