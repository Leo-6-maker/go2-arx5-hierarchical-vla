"""Prepare or launch the single-world first-six Go2 PLACE data collection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from go2arx5_vla.world_model import multisource_d4_native as d4
from go2arx5_vla.world_model import target_adaptation_teacher as adaptation
from go2arx5_vla.world_model.mfree_recovery_v2 import launch_service, register
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import OwnerLock, write

from .multitask_stage_cards_v6 import cold_native_owner, pin, prepare_cards
from .multitask_teacher_campaign_v6 import interface_records, preflight


def prepare(config):
    root = Path(config['root'])
    source = Path(config['source_root']).resolve(strict=True)
    authority = read_pin(config['authority'])
    scope = config.get('scope',adaptation.SCOPE)
    plan, authorized_by, _ = adaptation.SCOPES[scope]
    interface = cold_native_owner(config['interface_predecessor'])
    interface_records(interface,interface['mobile_world_lock'],identity=(d4.E0_SCOPE,d4.E0_PLAN))
    if (root.exists() or root.parent.stat().st_uid != os.getuid()
            or os.getuid() != interface['uid'] or os.uname().nodename != interface['host']
            or source != Path(__file__).resolve().parents[2]
            or authority['source'] != authorized_by or authority['plan_commit'] != plan
            or authority['source_sha'] != config['source_sha'] or authority['hardware_allowed'] is not False
            or authority['native_request_cap'] != len(adaptation.batch_cells(scope))
            or authority['native_tick_cap'] != adaptation.tick_cap(scope)
            or authority['compiled_population_sha256'] != config['population']['sha256']):
        raise PermissionError('EXACT_FINITE_SITE_AND_TARGET_TRAIN_AUTHORITY_REQUIRED')
    source_files = {str(path.relative_to(source)):digest(path)
                    for folder in ('src','scripts') for path in sorted((source/folder).rglob('*.py'))}
    if any(source_files[key] != expected for key,expected in config['source_members'].items()):
        raise ValueError('The staged adaptation source changed')
    source_seal = hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest()
    env = dict(interface['native_environment'],PYTHONPATH=':'.join((
        str(source),str(source/'src'),interface['native_environment']['PYTHONPATH'])),
        ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',CUDA_VISIBLE_DEVICES='')
    spec = dict(interface,scope=scope,registry_scope=scope,
        role='SLA_SCALE_TARGET_ADAPTATION_EXPERT_PANEL',plan_commit=plan,goal_plan_sha=plan,
        source_sha=config['source_sha'],source_root=str(source),source_seal=source_seal,source_files=source_files,
        native_environment=env,registered_root=str(root),campaign_root=str(root),physics=True,
        new_optimizer_updates_budget=0,new_training_visits_budget=0,new_model_generations_budget=0,
        deadline_epoch=authority['deadline_epoch'],campaign_deadline_epoch=authority['deadline_epoch'],
        allocation_deadline_epoch=authority['allocation_deadline_epoch'],automatic_retry=False,
        frontend_dependency=False,sim_only_project_authorization=config['authority'],
        interface_predecessor=config['interface_predecessor'],compiled_population=config['population'],
        previous_teacher_results=[])
    parent,entries = prepare_cards(spec,teacher_stop=len(adaptation.batch_cells(scope)))
    return parent,entries


def run(config, *, launch):
    if launch:
        interface = read_pin(config['interface_predecessor'])
        previous = cold_native_owner(config['previous_world_owner'])
        terminal = read_pin(config['previous_world_result'])
        if (previous['mobile_world_lock'] != interface['mobile_world_lock']
                or terminal['unresolved_case_launches'] or terminal['failure']
                or terminal['source_sha'] != previous['source_sha']
                or config['previous_world_result']['path'] != str(Path(previous['registered_root'])/'RESULT.private.json')):
            raise ValueError('The prior native world owner must be reconciled before launch')
        scope = config.get('scope',adaptation.SCOPE)
        if scope in adaptation.PREVIOUS_COLLECTION:
            prior_scope = adaptation.PREVIOUS_COLLECTION[scope]
            authority = read_pin(config['authority'])
            first = [row for row in terminal['records'] if row['role'] == 'EXPERT']
            new_population = read_pin(config['population'])
            prior_population = read_pin(previous['compiled_population'])
            new_cases = adaptation.conditions(new_population,scope)
            old_cases = adaptation.conditions(prior_population,prior_scope)
            if (previous['scope'] != prior_scope or previous['plan_commit'] != adaptation.SCOPES[prior_scope][0]
                    or authority['previous_result_sha256'] != config['previous_world_result']['sha256']
                    or terminal['new_verified_native_requests'] != len(old_cases)
                    or [row['id'] for row in first] != [row['id'] for row in old_cases]
                    or any(row['technical_failure'] or row['actual_native_requests'] != 1
                           or (row['hard_STOP'] and row['integrations_after_STOP'] != 0) for row in first)
                    or {row['reset'] for row in new_cases} & {row['reset'] for row in old_cases}
                    or {row['scene']['sha256'] for row in new_population['clusters']}
                        & {row['scene']['sha256'] for row in prior_population['clusters']}):
                raise ValueError('A_NEW_TARGET_TRAIN_BATCH_REQUIRES_COMPLETE_COLD_PRIOR_RECEIPTS')
            if scope == adaptation.RELEASE_SCOPE:
                audit = read_pin(authority['static_release_audit'])
                row = next(result for case,result in zip(old_cases,first,strict=True)
                           if (case['family'],case['instruction']) == adaptation.RELEASE_CELL)
                audited = audit['rows']['F2-05-PLACE-blue-left']['audit']['offsets']
                chosen = next(item for item in audited if item['offset_mm']==[0,8])
                if (not row['hard_STOP'] or row['task_success'] or audit['new_requests'] or audit['native_integrations']
                        or not chosen['within_zone'] or not chosen['verdict']['passed']
                        or authority['release_offset_xy_m'] != list(adaptation.RELEASE_OFFSET_XY_M)):
                    raise ValueError('RELEASE_OFFSET_REQUIRES_THE_ARCHIVED_OPEN_FAILURE_AND_FIXED_GEOMETRY_EVIDENCE')
        linger = subprocess.check_output(['loginctl','show-user',str(os.getuid()),'-p','Linger'],text=True,timeout=20)
        if linger.strip() != 'Linger=yes':
            raise PermissionError('Core user services require actual linger')
        lock = OwnerLock(interface['mobile_world_lock'])
        lock.close()
    parent,entries = prepare(config)
    root = Path(parent['registered_root'])
    preflight(parent,prepared_cases=(case for _,case in entries))
    if launch:
        if time.time() + parent['case_lead_s'] >= parent['deadline_epoch']:
            raise TimeoutError('First-six launch missed its finite native case start window')
        register(parent,[(root,parent),*entries])
    else:
        for folder,case in entries:
            write(folder/'EXECUTION_SPEC.json',case)
        write(root/'EXECUTION_SPEC.json',parent)
    spec = pin(root/'EXECUTION_SPEC.json')
    result = dict(spec=spec,source_sha=parent['source_sha'],world=parent['native_backend'],
        cases=len(entries),planned_tick_cap=sum(case['max_ticks'] for _,case in entries),
        model_generations=0,optimizer_updates=0,native_requests=0,registered=launch)
    write(root/'PREFLIGHT.private.json',result)
    if launch:
        result['launch'] = launch_service(parent,'wm02-'+root.name,'--multitask-teacher',
                                          bootstrap='v6_bootstrap.py')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    parser.add_argument('--launch',action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(run(read_pin(dict(path=args.config,sha256=args.sha256)),launch=args.launch),allow_nan=False))


if __name__ == '__main__':
    main()
