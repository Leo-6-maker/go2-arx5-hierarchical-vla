"""Read-only M1 T2 object/action diagnostics; no simulator or model is loaded."""
import argparse
import itertools
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from scripts.pi0292.durable import read,write
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.score import score
from scripts.pi0291_arx5_data.config import PARAMETERS as P
from .scope import validate,charge_cpu

ATTEMPTS=tuple('PI0294-E1-M1-1024-CANONICAL-'+f+'-000-T2' for f in ['F0','F1'])


def progress(arrays,card):
    times=np.asarray(arrays['time_s']);obj=np.asarray(arrays['object_pose']);tcp=np.asarray(arrays['tcp_world'])
    xyz=obj[:,:3,3];corners=np.einsum('nij,kj->nki',obj[:,:3,:3],np.asarray(list(itertools.product((-.02,.02),repeat=3))))+xyz[:,None,:]
    zone=(np.abs(corners[:,:,:2]-np.asarray(card['goal_xy']))<=.07).all(axis=(1,2))
    reached=np.flatnonzero(zone);contact=np.flatnonzero(arrays['robot_object_contact'])
    final=xyz[-1];distance=np.linalg.norm(xyz[:,:2]-np.asarray(card['goal_xy']),axis=1)
    return dict(saved_first_time_s=float(times[0]),saved_last_time_s=float(times[-1]),samples=len(times),
        initial_object_xyz_m=xyz[0].tolist(),last_saved_object_xyz_m=final.tolist(),goal_xy_m=card['goal_xy'],
        maximum_horizontal_displacement_m=float(np.linalg.norm(xyz[:,:2]-xyz[0,:2],axis=1).max()),
        minimum_center_goal_XY_distance_m=float(distance.min()),last_center_goal_XY_distance_m=float(distance[-1]),
        goal_zone_all_corners_ever=bool(zone.any()),first_goal_zone_all_corners_s=float(times[reached[0]]) if len(reached) else None,
        last_goal_zone_all_corners=bool(zone[-1]),first_saved_robot_contact_s=float(times[contact[0]]) if len(contact) else None,
        last_saved_robot_contact=bool(arrays['robot_object_contact'][-1]),last_saved_TCP_object_distance_m=float(np.linalg.norm(tcp[-1,:3,3]-final)),
        last_saved_table_height_predicate=bool(abs(final[2]-(P['table_top']+.02))<=.005),
        last_saved_linear_speed_m_s=float(np.linalg.norm(arrays['object_linear_velocity'][-1])),
        last_saved_angular_speed_rad_s=float(np.linalg.norm(arrays['object_angular_velocity'][-1])),
        these_are_saved_frame_diagnostics_not_complete_task_success=True)


def run(root):
    root=Path(root);m=validate(root,'4080-cpu-data');started=time.process_time();output=root/'evidence/M1_T2_SAVED_ANALYSIS_01'
    binding=root/'plans/M1_T2_SAVED_ANALYSIS_BINDING.private.json'
    if m['execution'].get('t2_saved_binding_sha256')!=sha(binding):
        raise PermissionError('IMMUTABLE_SAVED_T2_REQUEST_BINDING')
    output.mkdir(mode=0o700);request=read(binding)
    if request['attempts']!=list(ATTEMPTS) or request['maximum_CPU_s']!=600 or request['new_physics'] or request['new_model_queries']:
        raise PermissionError('ONLY_TWO_FIXED_SAVED_T2_NEGATIVES_NO_NEW_PHYSICS_OR_NEURAL')
    old=Path(request['old_root']);old_manifest=old/'RUN_MANIFEST.json'
    if sha(old_manifest)!=request['old_manifest_sha256'] or read(read(old/'OUTPUT_BINDING.private.json')['lease'])['status']!='CLOSED':
        raise PermissionError('OLD_E1_CLOSED_IMMUTABLE_INPUT_ONLY')
    rows=[]
    try:
        for identifier in ATTEMPTS:
            folder=old/'episodes'/identifier
            for name,digest in request['files'][identifier].items():
                if sha(folder/name)!=digest:raise PermissionError('FIXED_SAVED_T2_INPUT_CHANGED')
            card=read(folder/'CARD.json');adjudication=read(folder/'E1_ADJUDICATION.json');result=read(folder/'RESULT.json')
            if card['split']!='TRAIN' or card['task']!='T2' or adjudication['status']!='VALID_TASK_FAIL' or adjudication['live_native_failures']:
                raise PermissionError('ORIGINAL_VALID_TRAIN_T2_S1_NEGATIVES_ONLY')
            with np.load(folder/'physics.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            recalculated=score(arrays,card,complete=len(arrays['time_s'])==641)
            recalculated.update(physics_sha256=sha(folder/'physics.npz'),raw_sample_count=len(arrays['time_s']))
            if recalculated!=read(folder/'SCORE.json'):raise ValueError('SAVED_ORIGINAL_SCORE_RECALCULATION_MISMATCH')
            commands=sorted((folder/'relay/commands').glob('*.json'));last=commands[-1];bundle=read(last)
            action=unseal(bundle['action']);raw=np.asarray(action['payload']['trajectory_eef10']);observation=unseal(read(folder/'relay/observations'/last.name));state=np.asarray(observation['payload']['state_eef10'])
            sequence=int(last.stem)
            if (folder/'relay/acks'/last.name).exists() or action['sequence']!=sequence or observation['sequence']!=sequence or sequence*200!=result['physics_steps']:
                raise PermissionError('EXACT_REJECTED_UNEXECUTED_PREFIX_ANCHOR')
            forecast=read(folder/'S1_PREFLIGHTS.json')[-1]
            rows.append(dict(attempt_id=identifier,original_status=adjudication['status'],original_duration_s=adjudication['duration_s'],
                original_result_sha256=sha(folder/'RESULT.json'),input_hashes=request['files'][identifier],
                progress=progress(arrays,card),original_score_exact=True,
                rejected_anchor=dict(sequence=sequence,tick=observation['tick'],state_EEF10=state.tolist(),
                    first_command_minus_anchor_XYZ_m=(raw[0,:3]-state[:3]).tolist(),
                    prefix8_minimum_commanded_TCP_z_m=float(raw[:8,2].min()),
                    prefix8_minimum_commanded_TCP_z_above_table_m=float(raw[:8,2].min()-P['table_top']),
                    raw_H40=raw.tolist(),command_sha256=sha(last),original_S1=forecast,rejected_prefix_executed=False),
                specific_first_illegal_geom='NOT_SAVED_REQUIRES_SEPARATE_EXACT_HISTORY_SHADOW_DIAGNOSTIC',
                model_intent_inferred_from_AUDIT_ONLY_teacher_stage=False,unique_cause_proved=False))
        value=dict(status='PASS_READONLY_SAVED_T2_PROGRESS_AND_ACTION_AUDIT_NOT_GEOM_ATTRIBUTION',run_uuid=root.name,source_sha=m['code']['vla_commit'],
            rows=rows,new_physics_integrations=0,new_model_queries=0,new_training_updates=0,original_files_modified=False,
            original_valid_negative_replayed=False,selected='NONE',deployment='NONE')
        write(output/'RESULT.json',value);return value
    finally:charge_cpu(root,time.process_time()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    value=run(a.root);print(json.dumps(dict(status=value['status'],rows=len(value['rows']))),flush=True)
