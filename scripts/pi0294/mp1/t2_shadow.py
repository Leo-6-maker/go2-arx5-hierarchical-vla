"""Exact saved-history attribution, with no new live rollout or neural query."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.control_read import read
from scripts.pi0294.e1.collision import reconstruct_case
from scripts.pi0294.native_dataset import sha
from scripts.pi0291_arx5_data.config import PARAMETERS as P
from go2arx5_vla.trajectory.so3 import geodesic_angle
from .science import Ledger,validate_runtime
from . import t2_config as config


class DiagnosticLedger(Ledger):
    def __init__(self,root):
        super().__init__(root,'core');self.deadline=time.monotonic()+1800

    def guard(self,*,closeout=False):
        super().guard(closeout=closeout)
        if time.monotonic()>=self.deadline:raise TimeoutError('SAVED_T2_DIAGNOSTIC_1800S_BOUNDARY')


def qualify_inputs(root,request):
    root=Path(root);old=Path(read(root/'E1_INPUTS.private.json')['E0_root'])
    if request['attempts']!=list(config.ATTEMPTS) or sha(old/'RUN_MANIFEST.json')!=request['old_manifest_sha256']:
        raise PermissionError('ONLY_TWO_FIXED_SAVED_M1_T2_INPUTS')
    b=read(old/'OUTPUT_BINDING.private.json')
    if read(b['lease'])['status']!='CLOSED' or (old/'SOURCE_PHYSICS_STOP.json').exists():
        raise PermissionError('OLD_SOURCE_SCOPE_MUST_BE_CLOSED_AND_NOT_SAFETY_STOPPED')
    for identifier in config.ATTEMPTS:
        folder=old/'episodes'/identifier;a=read(folder/'E1_ADJUDICATION.json');result=read(folder/'RESULT.json')
        if a['status']!='VALID_TASK_FAIL' or not a['pipeline_trustworthy'] or a['live_native_failures'] or a['model']!='M1-1024-CANONICAL':
            raise PermissionError('EXACT_VALID_S1_NEGATIVE_NOT_ENGINEERING_OR_NATIVE_FAILURE')
        if result['physics_steps']!=a['main_integrations'] or a['result_sha256']!=sha(folder/'RESULT.json'):
            raise PermissionError('EXACT_ORIGINAL_PHYSICAL_PROGRESS')
        for name,digest in request['files'][identifier].items():
            if Path(name).is_absolute() or '..' in Path(name).parts or sha(folder/name)!=digest:
                raise PermissionError('IMMUTABLE_ORIGINAL_T2_INPUT_CHANGED')
    return old


def summarize(result):
    value={k:v for k,v in result.items() if k not in ['states','same_state_comparison']}
    comparison=result['same_state_comparison'];anchor=np.asarray(comparison['source_actual_TCP']);model=np.asarray(comparison['model_H40_poses']);expert=np.asarray(comparison['expert_H40_poses'])
    value['same_state_comparison']={k:v for k,v in comparison.items() if k not in ['model_H40_poses','model_H40_gripper','expert_H40_poses','expert_H40_gripper']}
    value['signed_prefix_XYZ_model_minus_expert_m']=(model[:8,:3,3]-expert[:8,:3,3]).tolist()
    value['prefix8_minimum_model_TCP_z_above_table_m']=float(model[:8,2,3].min()-P['table_top'])
    value['first_command_model_minus_anchor_XYZ_m']=(model[0,:3,3]-anchor[:3,3]).tolist()
    for check in value['collision_checks']:
        for pair in check['illegal_pairs']:
            pair['all_original_rules_triggered']=([pair['original_rule']] if pair['original_rule']=='ARM_TABLE_OR_FLOOR' else [])+(['PENETRATION_STOP'] if pair['signed_distance_m'] < -P['penetration_stop_m'] else [])
        requested=np.asarray(check['requested_TCP_at_end']);actual=np.asarray(check['actual_TCP_at_end'])
        check['signed_requested_minus_achieved_XYZ_m']=(requested[:3,3]-actual[:3,3]).tolist()
        check['requested_achieved_SO3_rad']=float(geodesic_angle(requested[:3,:3],actual[:3,:3]))
        contacts=[p['signed_distance_m'] for p in check['illegal_pairs'] if 'ARM_TABLE_OR_FLOOR' in p['all_original_rules_triggered']]
        check['minimum_reported_illegal_arm_surface_contact_distance_m']=min(contacts) if contacts else None
        check['all_geometry_minimum_distance_not_measured']=True
    return value


def run(root):
    root=Path(root);manifest=validate_runtime(root,'4080-simulation');stage=read(root/'MP1_STAGE_BINDING.json')
    if not config.is_diagnostic(stage):raise PermissionError('ONLY_FIXED_SAVED_T2_DIAGNOSTIC_STAGE')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('NO_CUDA_IN_SAVED_HISTORY_DIAGNOSTIC')
    request=read(root/'plans/T2_SHADOW_INPUTS.private.json');old=qualify_inputs(root,request)
    owner=OwnerLock(root/'T2_SHADOW_OWNER.lock');ledger=DiagnosticLedger(root);started=time.process_time();rows=[]
    output=root/'evidence/M1_T2_FIRST_REJECTION_ATTRIBUTION.private.json';assert not output.exists()
    try:
        for identifier in config.ATTEMPTS:
            ledger.guard()
            if (root/'SOURCE_PHYSICS_STOP.json').exists():raise PermissionError('DIAGNOSTIC_SOURCE_STOP')
            case=request['cases'][identifier]
            if case['attempt_id']!=identifier or case['card']['task']!='T2' or case['card']['split']!='TRAIN':
                raise PermissionError('ONLY_ORIGINAL_TRAIN_T2_CASES')
            result,snapshots=reconstruct_case(root,case,ledger,save=False,diagnose=True)
            try:rows.append(summarize(result))
            finally:
                for engine,_ in snapshots.values():engine.close()
            write(output,dict(status='ACTUAL_EXACT_HISTORY_T2_SHADOW_ATTRIBUTION',source_sha=manifest['code']['vla_commit'],rows=rows,completed=len(rows),planned=2,
                new_live_integrations=0,new_model_queries=0,new_optimizer_updates=0,new_autonomous_policy_evaluations=0,
                original_task_results_unchanged=True,teacher_is_same_state_diagnostic_not_qualified_recovery=True,
                selected='NONE',deployment='NONE'))
        return read(output)
    except BaseException as exc:
        ledger.stop_physics('SAVED_T2_DIAGNOSTIC_INTEGRITY_OR_ENGINEERING_STOP',dict(error_type=type(exc).__name__,error=str(exc),completed=len(rows)))
        write(root/'evidence/T2_SHADOW_FAILURE.private.json',dict(error_type=type(exc).__name__,error=str(exc),completed=len(rows)));raise
    finally:ledger.cpu(time.process_time()-started);owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:3]));resource.setrlimit(resource.RLIMIT_CPU,(3420,3420))
    result=run(a.root);print(json.dumps(dict(status=result['status'],completed=result['completed'])),flush=True)
