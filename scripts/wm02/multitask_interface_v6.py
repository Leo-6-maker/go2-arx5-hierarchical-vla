"""Run the two predeclared noncontact ABI interfaces on the existing owner."""
import json
from pathlib import Path
import time
import traceback

from scripts.pi0292.durable import write
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close, unit_state
from go2arx5_vla.world_model.multitask_scope_v6 import PLAN, SCOPE, read_pin, validate
from go2arx5_vla.world_model import multisource_d4_native as d4
from .sim_only_campaign_v51 import run as run_campaign, fatal_control_violation


def wait_target_owner(spec):
    root=Path(spec['registered_root'])
    predecessor=spec['wait_target_followup']
    prior=Path(read_pin(predecessor)['registered_root'])
    while time.time()<spec['deadline_epoch']-1800 and not (root/'STOP.json').exists():
        lease=json.loads((prior/'control/LEASE.json').read_bytes())
        state=unit_state('wm02-'+prior.name)
        if lease['spec_sha256']!=predecessor['sha256']:
            raise ValueError('T_NATIVE_OWNER_IDENTITY_CHANGED')
        write(root/'PROGRESS.private.json',dict(phase='WAIT_EXISTING_T_NATIVE_OWNER_COLD_BOUNDARY',
            epoch=time.time(),waited_lease=lease['status'],waited_MainPID=state['pid'],
            new_native_requests=0,new_native_ticks=0))
        if (lease['status']=='CLOSED' and not state['pid'] and not state['cgroup_pids']
                and (prior/'RESULT.private.json').exists()):
            return prior
        time.sleep(30)
    raise TimeoutError('FINITE_MULTITASK_INTERFACE_DEPENDENCY_WINDOW')


def preflight(spec):
    identity = (spec['plan_commit'], spec['scope'])
    if (identity not in ((PLAN, SCOPE), (d4.E0_PLAN, d4.E0_SCOPE)) or
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


def e0_predecessor(spec):
    before = read_pin(spec['wait_D4_native'])
    if before['scope'] != d4.SCOPE or before['mobile_world_lock'] != spec['mobile_world_lock']:
        raise ValueError('E0_REQUIRES_THE_ACTUAL_EXISTING_D4_NATIVE_OWNER')
    root = Path(before['registered_root'])
    waiting = Path(spec['registered_root'])
    observed = json.loads((root/'control/LEASE.json').read_bytes())
    while time.time() < spec['deadline_epoch']-480 and not (waiting/'STOP.json').exists():
        state = unit_state('wm02-'+root.name)
        observed = json.loads((root/'control/LEASE.json').read_bytes())
        if observed['spec_sha256'] != spec['wait_D4_native']['sha256']:
            raise ValueError('D4_NATIVE_PREDECESSOR_IDENTITY_CHANGED')
        write(waiting/'PROGRESS.private.json', dict(phase='WAIT_D4_EXISTING_NATIVE_WORLD', epoch=time.time(),
              waited_lease=observed['status'], waited_MainPID=state['pid'], new_native_requests=0, new_native_ticks=0))
        if observed['status'] == 'CLOSED' and not state['pid'] and not state['cgroup_pids']:
            result = json.loads((root/'RESULT.private.json').read_bytes())
            if (result['unresolved_case_launches']
                    or any(fatal_control_violation(row) for row in result['records'])):
                raise ValueError('RECONCILE_D4_PREDECESSOR_TRACE_OR_OWNER_BEFORE_E0')
            return
        if not state['pid'] and not state['cgroup_pids'] and observed['status'] == 'ACTIVE':
            raise ValueError('D4_PREDECESSOR_OWNER_UNKNOWN_NO_AUTOMATIC_E0_DISPATCH')
        time.sleep(30)
    raise TimeoutError('FINITE_D4_E0_WAIT_FOR_THE_SINGLE_NATIVE_WORLD')


def run(path,spec):
    root=Path(spec['registered_root'])
    try:
        preflight(spec)
        if spec['scope'] == d4.E0_SCOPE:
            e0_predecessor(spec)
        else:
            prior=wait_target_owner(spec)
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
