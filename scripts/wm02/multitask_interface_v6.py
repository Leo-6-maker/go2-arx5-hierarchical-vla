"""Run the two predeclared noncontact ABI interfaces on the existing owner."""
import json
from pathlib import Path
import traceback

from scripts.pi0292.durable import write
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
from go2arx5_vla.world_model.multitask_scope_v6 import PLAN, SCOPE, validate
from .sim_only_campaign_v51 import run as run_campaign, fatal_control_violation
from .source_test_campaign_v6 import wait_terminal


def preflight(spec):
    if (spec['plan_commit']!=PLAN or spec['scope']!=SCOPE or
            spec['new_physical_requests_budget']!=2 or spec['physics'] is not True or
            spec['new_optimizer_updates_budget'] or spec['new_model_generations_budget'] or
            len(spec['cases'])!=2 or spec.get('resident_server_spec') is not None):
        raise PermissionError('EXACT_TWO_NONCONTACT_MULTITASK_INTERFACES_REQUIRED')
    cases = [json.loads(Path(path).read_bytes()) for path in spec['cases']]
    if {case['mobile_condition']['family'] for case in cases}!={'F1','F2'}:
        raise ValueError('ONE_F1_AND_ONE_F2_INTERFACE_REQUIRED')
    for case in cases:
        validate(case)
        if (case['sim_only_role']!='INTERFACE' or case['mobile_group']!=spec['registered_root']
                or case['mobile_world_lock']!=spec['mobile_world_lock']
                or case['ROS_DOMAIN_ID']!=spec['ROS_DOMAIN_ID']
                or case['registered_panel']!=spec['registered_panel']):
            raise ValueError('EXACT_SERIAL_MULTITASK_OWNER_AND_PANEL_REQUIRED')
    return []


def run(path,spec):
    root=Path(spec['registered_root'])
    try:
        preflight(spec)
        terminal=wait_terminal(spec,spec['wait_target_followup'],'WAIT_EXISTING_T_NATIVE_OWNER_COLD_BOUNDARY')
        prior=Path(terminal['path']).parent
        for stage in ('SCREEN','CONFIRM'):
            result=prior/stage/'RESULT.private.json'
            if not result.exists():continue
            report=json.loads(result.read_bytes())
            if report['unresolved_case_launches'] or any(fatal_control_violation(row) for row in report['records']):
                raise ValueError('RECONCILE_PREVIOUS_NATIVE_OWNER_OR_TRACE_VIOLATION_FIRST')
    except BaseException:
        for case_path in spec['cases']:
            case=json.loads(Path(case_path).read_bytes())
            case_root=Path(case['registered_root'])
            lease=json.loads((case_root/'control/LEASE.json').read_bytes())
            if lease['status']=='ACTIVE' and not (case_root/'SUPERVISOR.private.json').exists():
                close(case,0,not_started=True,reason='MULTITASK_INTERFACE_PRECONDITION')
        result=dict(exit_code=1,status='MULTITASK_INTERFACE_NOT_DISPATCHED',
            source_sha=spec['source_sha'],failure=traceback.format_exc(),
            new_verified_native_requests=0,native_ticks=0)
        write(root/'RESULT.private.json',result)
        close(spec,1)
        return 1
    return run_campaign(path,preflight=preflight,bootstrap='v6_bootstrap.py',expected_count=2,
        result_schema='SLA_V6_MULTITASK_TWO_INTERFACES_TERMINAL',pilot_evaluate=lambda rows:False)
