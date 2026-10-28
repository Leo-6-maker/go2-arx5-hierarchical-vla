"""Shared non-corrective action contract for offline probes and paused relay."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import numpy as np

FRAME='PI0291_MUJOCO_WORLD'
TCP='ARX5_NATIVE_SITE_x5_ee'
STAGES=('close','lift','carry','release','hold')


def uniform_train_ids(identifiers):
    identifiers=sorted(identifiers,key=lambda s:int(s[1:]))
    if len(identifiers)<16:raise ValueError('AT_LEAST_16_TRAIN_REQUIRED')
    return [identifiers[j*(len(identifiers)-1)//15] for j in range(16)]


def stage_anchors(samples):
    anchors={0:['start']};missing=[]
    for stage in STAGES:
        indices=[i for i,s in enumerate(samples) if i<=600 and s['privileged']['teacher_stage']==stage]
        if not indices:missing.append(stage);continue
        anchors.setdefault(indices[0],[]).append(stage)
    return [dict(anchor=i,stages=labels) for i,labels in sorted(anchors.items())],missing


def noise_seed(bank,identifier,anchor):
    if bank not in (29201,29202):raise ValueError('FROZEN_TWO_NOISE_BANKS_REQUIRED')
    key=f'PI0292/OFFLINE/{bank}/{identifier}/{anchor}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')%(2**63)


def matrix10(value):
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    value=np.asarray(value,dtype=np.float64)
    if value.shape[-1]!=10 or not np.isfinite(value).all():raise ValueError('NONFINITE_OR_INVALID_EEF10')
    matrix=np.broadcast_to(np.eye(4),value.shape[:-1]+(4,4)).copy()
    matrix[...,:3,:3]=rot6d_rows_to_matrix(value[...,3:9])
    matrix[...,:3,3]=value[...,:3]
    return matrix


def gate_configuration():
    from go2arx5_vla.safety.trajectory_gate import TrajectorySafetyGateConfig
    return dict(parameters=asdict(TrajectorySafetyGateConfig()),source_frame=FRAME,tcp_semantic_id=TCP,
        orientation='N1.7 native rows -> matrix -> canonical xyzw; no column-rot6d shortcut',
        clock='Paused simulation time: obs tick/500; wall timeout independently <=60s',
        gripper='0 closed, 1 open; commands must be within [0,1]; no extra clipping',
        corrective_action='NONE',real_robot_qualification=False)


def gate_action(action,state,observation_time,identifier):
    from go2arx5_vla.contracts.trajectory import TrajectoryChunkV1
    from go2arx5_vla.safety.trajectory_gate import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat
    action=np.asarray(action,dtype=np.float64);state=np.asarray(state,dtype=np.float64)
    if action.shape!=(40,10) or state.shape!=(10,):raise ValueError('FULL_H40_AND_ACTUAL_STATE_REQUIRED')
    try:
        if not np.isfinite(action).all() or not np.isfinite(state).all():raise ValueError('NON_FINITE')
        if np.any(action[:,9]<0) or np.any(action[:,9]>1):raise ValueError('GRIPPER_COMMAND_RANGE')
        matrices=matrix10(action);anchor=matrix10(state)
        times=float(observation_time)+np.arange(40)/20
        chunk=TrajectoryChunkV1(chunk_id=identifier,source='PI0292_NATIVE_DECODE',
            observation_timestamp=float(observation_time),point_timestamps=times,
            eef_position=action[:,:3],eef_orientation=matrix_to_quat(matrices[:,:3,:3]),
            source_frame=FRAME,tcp_semantic_id=TCP,validity_start=times[0],validity_end=times[-1],gripper=action[:,9])
        return TrajectorySafetyGateV1().evaluate(chunk,now=float(observation_time),
            achieved_position=anchor[:3,3],achieved_orientation=matrix_to_quat(anchor[:3,:3])).as_dict()
    except ValueError as error:
        return dict(decision='REJECT',issues=[dict(code=str(error),severity='REJECT',provenance='CONTROLLER_CONTRACT')],
            metrics={},provenance=dict(corrective_action='NONE'))


def expert_decode(processor,state,commands):
    from gr00t.data.embodiment_tags import EmbodimentTag
    state_dict={'eef_9d':state[None,:9],'gripper_position':state[None,9:]}
    values=processor.state_action_processor.apply_action(
        {'eef_9d':commands[:,:9],'gripper_position':commands[:,9:]},'new_embodiment',state=state_dict)
    normalized=np.concatenate([values[k] for k in ('eef_9d','gripper_position')],axis=-1).astype(np.float32)
    padded=np.pad(normalized,((0,0),(0,122)))[None]
    decoded=processor.decode_action(padded,EmbodimentTag.NEW_EMBODIMENT,{k:v[None] for k,v in state_dict.items()})
    physical=np.concatenate([decoded[k] for k in ('eef_9d','gripper_position')],axis=-1)[0]
    return physical,dict(normalized_outside_unit_components=int(np.count_nonzero(np.abs(normalized)>1)),
        native_inverse_clip_declared=True,extra_postprocessing=False)


def error_metrics(prediction,expert,state):
    from go2arx5_vla.trajectory.so3 import geodesic_angle
    prediction=np.asarray(prediction);expert=np.asarray(expert);state=np.asarray(state)
    xyz=np.linalg.norm(prediction[:,:3]-expert[:,:3],axis=-1)
    grip=np.abs(prediction[:,9]-expert[:,9])
    try:
        rotations=matrix10(prediction)[:,:3,:3];reference=matrix10(expert)[:,:3,:3]
        angles=geodesic_angle(rotations,reference)
        angular=np.asarray(geodesic_angle(rotations[:-1],rotations[1:]))/.05
    except ValueError:angles=np.full(40,np.nan);angular=np.full(39,np.nan)
    def summary(values):
        return dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),mean=float(np.mean(values))) if np.isfinite(values).all() else None
    metrics={name:dict(XYZ_euclidean_m=summary(xyz[:n]),SO3_rad=summary(angles[:n]),gripper_MAE=float(np.mean(grip[:n])) if np.isfinite(grip[:n]).all() else None)
        for name,n in (('H40',40),('prefix8',8))}
    metrics.update(first_position_jump_m=float(np.linalg.norm(prediction[0,:3]-state[:3])) if np.isfinite(prediction[0,:3]).all() else None,
        linear_speed_mps=summary(np.linalg.norm(np.diff(prediction[:,:3],axis=0),axis=-1)/.05),
        angular_speed_radps=summary(angular))
    metrics['pointwise']={name:[float(v) if np.isfinite(v) else None for v in values]
        for name,values in (('XYZ_euclidean_m',xyz),('SO3_rad',angles),('gripper_absolute',grip))}
    for label,less in (('close',True),('open',False)):
        def transition(values):
            sequence=np.r_[state[9],values[:,9]]
            hit=(sequence[1:]<.5)&(sequence[:-1]>=.5) if less else (sequence[1:]>=.5)&(sequence[:-1]<.5)
            return [int(i) for i in np.flatnonzero(hit)]
        expected,actual=transition(expert),transition(prediction)
        metrics[label+'_events']=dict(expert_indices=expected,predicted_indices=actual,
            first_timing_error_s=(actual[0]-expected[0])*.05 if expected and actual else None,
            missing=bool(expected and not actual),spurious=bool(actual and not expected),threshold=.5)
    return metrics


def summarize_records(records):
    """Every finite Gate-REJECT row remains in the same error denominator."""
    from collections import Counter
    result={}
    for split in ('TRAIN','DEV'):
        rows=[r for r in records if r['split']==split]
        episodes={}
        for identifier in sorted({r['id'] for r in rows}):
            own=[r for r in rows if r['id']==identifier];views={}
            for name,length in (('H40',40),('prefix8',8)):
                views[name]={}
                for metric in ('XYZ_euclidean_m','SO3_rad','gripper_absolute'):
                    values=[v for row in own for v in row.get('errors',{}).get('pointwise',{}).get(metric,[None]*40)[:length]]
                    finite=[v for v in values if v is not None]
                    views[name][metric]=dict(expected_points=len(values),finite_points=len(finite),
                        median=float(np.median(finite)) if finite else None,p95=float(np.percentile(finite,95)) if finite else None,
                        mean=float(np.mean(finite)) if finite else None,complete=len(finite)==len(values))
            episodes[identifier]=views
        macro={}
        for view in ('H40','prefix8'):
            macro[view]={}
            for metric in ('XYZ_euclidean_m','SO3_rad','gripper_absolute'):
                values=[r[view][metric] for r in episodes.values()]
                complete=bool(values) and all(v['complete'] for v in values)
                macro[view][metric]=dict(episodes=len(values),complete=complete,
                    **{key:float(np.mean([v[key] for v in values])) if complete else None for key in ('mean','median','p95')})
        result[split]=dict(queries=len(rows),episodes=episodes,episode_macro=macro,
            decisions=dict(Counter(r['gate']['decision'] for r in rows)),
            reasons=dict(Counter(issue['code'] for r in rows for issue in r['gate']['issues'])),
            rejected_rows_excluded=False,macro_rule='Mean of within-episode pooled-window/noise statistics; incomplete dimensions remain null')
    return result
