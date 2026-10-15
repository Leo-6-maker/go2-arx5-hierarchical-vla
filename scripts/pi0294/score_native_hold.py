"""Independent CPU reconstruction from saved native arrays; no replay or correction."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.pi0292.durable import read,write
from .r4a import validate_scope,cpu_stage
from go2arx5_vla.trajectory.so3 import quat_to_matrix


def errors(positions,rotations,target):
    xyz=np.linalg.norm(np.asarray(positions)-target[:3],axis=1)
    r=quat_to_matrix(target[3:]);mat=np.asarray(rotations).reshape(-1,3,3)
    angle=np.arccos(np.clip((np.einsum('nij,ij->n',mat,r)-1)/2,-1,1))
    return dict(XYZ_RMS_m=float(np.sqrt(np.mean(xyz**2))),XYZ_max_m=float(xyz.max()),
        SO3_RMS_rad=float(np.sqrt(np.mean(angle**2))),SO3_max_rad=float(angle.max()))


def main(root):
    root=Path(root);validate_scope(root);p=root/'fullbody/native-hold-01'
    with cpu_stage(root,'NATIVE_HOLD_CPU_SCORE_01',cores=1,cpu_s_cap=120):
        result=read(p/'RESULT.json');task=read(p/'TASK_START.json');target=np.asarray(task['initial_tcp'],dtype=float)
        physics=[json.loads(x) for x in (p/'physics.jsonl').read_text().splitlines()]
        model=next(x for x in physics if x['kind']=='model');first=next(x for x in physics if x['kind']=='first_step_input')
        initial=read(p/'initialization.jsonl')
        assert first['qpos']==initial['qpos'] and first['data_pointer']==initial['data_pointer'] and first['time']==0
        samples=[x for x in physics if x['kind']=='sample']
        ros=[json.loads(x) for x in (p/'ros.jsonl').read_text().splitlines()]
        ee=[x for x in ros if x['kind']=='ee' and x['monotonic_s']>=task['monotonic_s']]
        physical_clock=np.interp([x['monotonic_s'] for x in ee],
            [x['monotonic_s'] for x in samples],[x['sim_time_s'] for x in samples])
        delta=np.array([x['msg']['time'] for x in ee])-physical_clock
        offset=float(np.median(delta));residual=float(np.max(np.abs(delta-offset)))
        selected=[x for x in samples if 0<=x['sim_time_s']+offset-task['mpc_time_s']<=10.075]
        physical=errors([x['tcp_xyz'] for x in selected],[x['tcp_R'] for x in selected],target)
        feedback=errors([x['msg']['state']['value'][:3] for x in ee],
            [quat_to_matrix(x['msg']['state']['value'][3:7]) for x in ee],target)
        trace=[json.loads(x) for x in (p/'planner.jsonl').read_text().splitlines()]
        installed=[x for x in trace if x['kind']=='installed' and x.get('ref_id') is not None]
        accepted=[x for x in trace if x['kind']=='accept'];rejects=[x for x in trace if x['kind']=='reject']
        installed_errors=errors([v[:3] for x in installed for v in x['poses']],
            [quat_to_matrix(v[3:7]) for x in installed for v in x['poses']],target) if installed else None
        anchors=all(x['bound_start']==next(a['source_mpc_s'] for a in accepted if a['ref_id']==x['ref_id']) for x in installed)
        pins={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir() if x.is_file()}
        safety=not result['native_stops'] and not any(x['illegal_contact'] or x['joint_stop'] or any(x['native_warnings']) for x in samples)
        duration=max(x['msg']['time'] for x in ee)-task['mpc_time_s']
        complete=duration>=10 and (p/'PAUSE_CHECK.json').exists() and (p/'ISSUED.json').exists()
        ref_ok=bool(installed) and not rejects and anchors and installed_errors['XYZ_max_m']<=1e-8 and installed_errors['SO3_max_rad']<=1e-6
        integrity=complete and safety and ref_ok and residual<=.025
        quality=integrity and all(v['XYZ_RMS_m']<=.015 and v['SO3_RMS_rad']<=.05 for v in (physical,feedback))
        q=np.asarray([s['qpos'][:3] for s in samples]);ctrl=np.asarray([s['ctrl'][:12] for s in samples])
        pause=read(p/'PAUSE_CHECK.json') if (p/'PAUSE_CHECK.json').exists() else None
        pause_physics=bool(pause and pause['before']['sim_time_s']==pause['after']['sim_time_s'])
        late_message_only=bool(pause and pause['ee_time_after']<=pause['after']['sim_time_s'])
        receipt=dict(status='NATIVE_HOLD_PASS_ONLY' if quality else 'VALID_NATIVE_HOLD_QUALITY_FAIL' if integrity else 'NATIVE_INTEGRITY_OR_INCOMPLETE_BLOCKED',
            complete_10s=complete,original_native_safety=safety,reference_integrity=ref_ok,source_anchor_unchanged=anchors,
            actual_task_duration_s=duration,
            actual_physics=physical,actual_native_FK=feedback,installed_reference_error=installed_errors,
            source_clock=dict(offset_s=offset,max_residual_s=residual,gate_s=.025),
            accepted_packets=len(accepted),rejected_packets=len(rejects),
            actual_free_root=model['joints'][0]['type']==0,actual_qpos_dimension=model['nq'],
            root_xyz_span_m=np.ptp(q,axis=0).tolist(),all_12_leg_channels_actuated=bool(np.all(np.max(np.abs(ctrl),axis=0)>0)),
            observed_physics_lower_bound=result['physics_lower_bound'],original_RESULT_sha256=pins['RESULT.json'],
            original_array_hashes=pins,pause_actual_physics_clock_stable=pause_physics,
            pause_late_EE_message_bounded_by_frozen_physics_time=late_message_only,
            original_pause_comparison_error_retained=result.get('error'),
            resume_history_qualified=False,gripper_camera_task_plant_qualified=False,teacher_qualified=False,
            new_neural_queries=0,new_physics=0,selected='NONE',deployment='NONE')
        write(p/'CPU_VERDICT.json',receipt)
        if integrity and not quality:write(root/'fullbody/NATIVE_VALID_QUALITY_FAIL.json',receipt)
        print(json.dumps(receipt));return receipt


if __name__=='__main__':
    import sys
    main(sys.argv[1])
