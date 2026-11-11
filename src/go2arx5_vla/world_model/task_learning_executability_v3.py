"""Saved H40 kinematics at the original 50ms clock; no model or execution.

All joins are counterfactuals at recorded Teacher states. Actual preceding
references are consumed commands, never the measured physical gripper opening.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import fusion_trajectory_adapter as limits
from .task_learning_local_analysis_v1 import checked, geometry
from .storage import digest
from ..trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix, rotation_matrix_to_rot6d_rows

DT = .05
SCHEMA = 'TASK_LEARNING_H40_EXECUTABILITY_ATTRIBUTION_V3'


def angular_velocity(rotations):
    relative=rotations[1:]@np.swapaxes(rotations[:-1],-1,-2)
    q=matrix_to_quat(relative);q=np.where(q[:,3:]<0,-q,q)
    sine=np.linalg.norm(q[:,:3],axis=1,keepdims=True)
    return q[:,:3]*(2*np.arctan2(sine,q[:,3:]))/np.where(sine>0,sine,1.)/DT


def distribution(values):
    v=np.asarray(values,float)
    return dict(count=int(v.size),p50=float(np.quantile(v,.5)),p95=float(np.quantile(v,.95)),maximum=float(v.max()))


def sequence(values,limit,*,boundary=False,unit):
    v=np.asarray(values,float);norm=np.abs(v)if v.ndim==1 else np.linalg.norm(v,axis=1)
    over=np.flatnonzero(norm>limit*(1+1e-10)).tolist()if limit is not None else []
    parts=({'first':[0],'interior':list(range(1,len(norm)-1)),'last':[len(norm)-1]}if boundary
           else {'first_interval':[0],'interior':list(range(1,len(norm)-1)),'last_interval':[len(norm)-1]})
    peak=int(np.argmax(norm))
    return dict(unit=unit,dt_s=DT,vectors=v.tolist(),magnitudes=norm.tolist(),limit=limit,
        peak_index=peak,peak_part=next(k for k,index in parts.items()if peak in index),
        peak_value=float(norm[peak]),exceedance_indices=over,exceedance_count=len(over),
        exceedances_by_part={k:sum(i in over for i in index)for k,index in parts.items()},
        distribution=distribution(norm))


def path_series(raw,*,grip_is_mechanical_fraction=False):
    raw=np.asarray(raw,float)
    if raw.shape!=(40,10)or not np.isfinite(raw).all():raise ValueError('Full finite H40 required')
    rotation=rot6d_rows_to_matrix(raw[:,3:9])
    width=(limits.MECHANICAL_TRAVEL_M*raw[:,9]if grip_is_mechanical_fraction else
        limits.WORKING_CLOSED_M+(limits.WORKING_OPEN_M-limits.WORKING_CLOSED_M)*raw[:,9])
    linear=np.diff(raw[:,:3],axis=0)/DT;angular=angular_velocity(rotation);finger=np.diff(width)/DT
    return dict(linear_velocity=sequence(linear,limits.MAX_TRANSLATION_SPEED_M_S,unit='m/s'),
        linear_acceleration=sequence(np.diff(np.vstack((np.zeros(3),linear,np.zeros(3))),axis=0)/DT,
            limits.MAX_TRANSLATION_ACCELERATION_M_S2,boundary=True,unit='m/s^2'),
        angular_velocity=sequence(angular,limits.MAX_ANGULAR_SPEED_RAD_S,unit='rad/s'),
        angular_acceleration=sequence(np.diff(np.vstack((np.zeros(3),angular,np.zeros(3))),axis=0)/DT,
            limits.MAX_ANGULAR_ACCELERATION_RAD_S2,boundary=True,unit='rad/s^2'),
        finger_velocity=sequence(finger,limits.MAX_FINGER_SPEED_M_S,unit='m/s'),
        finger_acceleration=sequence(np.diff(np.r_[0.,finger,0.])/DT,None,boundary=True,unit='m/s^2'),
        working_gripper_width_m=width.tolist(),nominal_gripper=raw[:,9].tolist(),
        zero_velocity_padded_at_both_ends=True,first_state_to_point_gap_included=False,
        finger_acceleration_has_no_original_gate=True)


def reference_history(episode,anchor,history):
    from scripts.pi0292.eval_contract import matrix10
    from .task_learning_grasp_v1 import B
    if episode['id']not in history:
        commands=checked(episode['proof']['confirmed_commands'])['records']
        native=checked(episode['proof']['native'])['reference_qualification']
        prior=[r for r in native['windows']if r.get('native_end_step')in(B-25,B)]
        if len(prior)!=2:raise ValueError('Two actually consumed preaction references required')
        history[episode['id']]=(commands,prior)
    commands,prior=history[episode['id']]
    if anchor:
        pair=commands[40*anchor-2:40*anchor]
        if len(pair)!=2 or not all(x['solver_gripper_confirmed']for x in pair):
            raise ValueError('Two actually consumed preceding commands required')
        rows=np.asarray([x['installed_consumed_EEF10']for x in pair],float)
    else:rows=np.asarray([x['actual_installed_consumed_EEF10_BASE7'][:10]for x in prior],float)
    return matrix10(rows),rows[:,9]*limits.MECHANICAL_TRAVEL_M


def reference_join(raw,state,previous,previous_width,frame):
    from scripts.pi0292.eval_contract import matrix10
    inverse=np.linalg.inv(frame);actual=inverse@matrix10(state);desired=inverse@matrix10(raw)
    installed=desired.copy();installed[:,:3,3]=previous[-1,:3,3]+desired[:,:3,3]-actual[:3,3]
    installed[:,:3,:3]=previous[-1,:3,:3]@actual[:3,:3].T@desired[:,:3,:3]
    width=limits.WORKING_CLOSED_M+(limits.WORKING_OPEN_M-limits.WORKING_CLOSED_M)*raw[:,9]
    poses=np.concatenate((previous,installed[:2]));positions=poses[:,:3,3]
    velocity=np.diff(positions,axis=0)/DT;angular=angular_velocity(poses[:,:3,:3])
    finger=np.diff(np.r_[previous_width,width[:2]])/DT
    installed_rows=np.column_stack((installed[:,:3,3],rotation_matrix_to_rot6d_rows(installed[:,:3,:3]),
                                   width/limits.MECHANICAL_TRAVEL_M))
    return dict(previous_consumed_velocity_m_s=velocity[0].tolist(),first_reference_join_velocity_m_s=velocity[1].tolist(),
        first_reference_join_acceleration_m_s2=((velocity[1]-velocity[0])/DT).tolist(),
        join_to_first_internal_acceleration_m_s2=((velocity[2]-velocity[1])/DT).tolist(),
        previous_consumed_angular_velocity_rad_s=angular[0].tolist(),
        first_reference_join_angular_velocity_rad_s=angular[1].tolist(),
        first_reference_join_angular_acceleration_rad_s2=((angular[1]-angular[0])/DT).tolist(),
        previous_consumed_finger_velocity_m_s=float(finger[0]),first_reference_finger_velocity_m_s=float(finger[1]),
        command_to_command_gripper_gap_m=float(width[0]-previous_width[-1]),
        actual_measured_gripper_used_as_previous_command=False,
        initial_reference_to_first_point_gap_m=float(np.linalg.norm(positions[2]-positions[1])),
        native_reference_join_speed_limit_m_s=.06,native_reference_join_angular_limit_rad_s=.3,
        native_first_join_rate_pass=bool(np.linalg.norm(velocity[1])<=.06+1e-10 and np.linalg.norm(angular[1])<=.3+1e-10),
        installed_series=path_series(installed_rows,grip_is_mechanical_fraction=True),
        student_proposal_executed=False,contact_certificate_evaluated=False)


def summarize(rows):
    phases={}
    for phase in sorted({r['phase']for r in rows}):
        values=[r for r in rows if r['phase']==phase]
        phases[phase]=dict(windows=len(values),envelope_pass=sum(r['envelope']['original_envelope_pass']for r in values),
            failures=dict(Counter(k for r in values for k in r['envelope']['failed_components'])),
            sequence_distributions={key:distribution([v for r in values for v in r['desired'][key]['magnitudes']])
                for key in ('linear_velocity','linear_acceleration','angular_velocity','angular_acceleration','finger_velocity')},
            acceleration_windows_exceeding_by_part={part:sum(r['desired']['linear_acceleration']['exceedances_by_part'][part]>0 for r in values)
                for part in ('first','interior','last')})
    return dict(requested=40,audited=len(rows),envelope_pass=sum(r['envelope']['original_envelope_pass']for r in rows),
        acceleration_windows_exceeding_by_part={p:sum(r['desired']['linear_acceleration']['exceedances_by_part'][p]>0 for r in rows)for p in('first','interior','last')},
        acceleration_peak_location_windows=dict(Counter(r['desired']['linear_acceleration']['peak_part']for r in rows)),
        phases=phases)


def run(spec,output,guard):
    from scripts.pi0292.durable import write
    manifest=checked(spec['dataset']);episodes={r['id']:r for r in manifest['episodes']};frame=np.asarray(manifest['T_policy_from_target'],float)
    output=Path(output);output.mkdir();history={};all_rows={};all_summaries={};experts=[]
    for name,candidate in spec['candidates'].items():
        result=checked(candidate['result']);records=result['records'];rows=[];predictions={}
        if len(records)!=40:raise ValueError('Fixed original forty required')
        for record in records:
            guard();file=Path(candidate['samples'])/record['sample_file']
            if digest(file)!=record['sha256']:raise ValueError('Changed sealed prediction')
            with np.load(file,allow_pickle=False)as z:raw,expert,state=[z[k].copy()for k in('physical_H40','expert_policy_H40','actual_state10')]
            previous,previous_width=reference_history(episodes[record['episode']],record['anchor'],history)
            series=path_series(raw);envelope=geometry(raw,state)
            if not np.isclose(series['linear_acceleration']['peak_value'],envelope['metrics']['maximum_discrete_translation_acceleration_m_s2'],rtol=0,atol=1e-10):
                raise ValueError('Original envelope arithmetic mismatch')
            d=raw[-1,:3]-raw[0,:3];e=expert[-1,:3]-expert[0,:3];den=float(np.linalg.norm(d)*np.linalg.norm(e))
            row=dict(episode=record['episode'],anchor=record['anchor'],seed=record['seed'],phase=record['phase_audit_only'],
                prediction_sha256=record['sha256'],desired=series,envelope=envelope,
                first_point_actual_state_mismatch=dict(position_m=float(np.linalg.norm(raw[0,:3]-state[:3])),
                    nominal_gripper=float(raw[0,9]-state[9]),not_a_command_discontinuity_certificate=True),
                actual_reference_join=reference_join(raw,state,previous,previous_width,frame),
                motion=dict(predicted_displacement_m=d.tolist(),expert_displacement_m=e.tolist(),direction_cosine=float(np.dot(d,e)/den)if den>1e-14 else None),
                close_events=record['errors']['close_events'],open_events=record['errors']['open_events'],counterfactual_adjacent_prediction_join=None)
            rows.append(row);predictions[record['episode'],record['anchor']]=raw
            if len(all_rows)==0:experts.append(dict(episode=record['episode'],anchor=record['anchor'],phase=record['phase_audit_only'],
                desired=path_series(expert),envelope=geometry(expert,state)))
        for row in rows:
            previous=predictions.get((row['episode'],row['anchor']-1))
            if previous is not None:
                current=predictions[row['episode'],row['anchor']];join=(current[0,:3]-previous[-1,:3])/DT
                row['counterfactual_adjacent_prediction_join']=dict(previous_anchor=row['anchor']-1,
                    join_velocity_m_s=join.tolist(),join_acceleration_m_s2=((join-(previous[-1,:3]-previous[-2,:3])/DT)/DT).tolist(),
                    nominal_gripper_gap=float(current[0,9]-previous[-1,9]),both_predicted_at_Teacher_states_not_executed=True)
        all_rows[name]=rows;all_summaries[name]=summarize(rows)
    result=dict(schema=SCHEMA,model_loads=0,model_queries=0,optimizer_updates=0,physics_integrations=0,
        action_period_s=DT,native_dt_s=.002,source_sha=spec['source_sha'],dataset=spec['dataset'],
        candidate_summaries=all_summaries,expert_summary=summarize(experts),candidates=all_rows,experts=experts,
        raw_gripper_unit='NOMINAL_0_CLOSED_1_OPEN',working_closed_m=limits.WORKING_CLOSED_M,working_open_m=limits.WORKING_OPEN_M,
        working_mapping_applied_once=True,original_limits_unchanged=True,policy_execution_authorized=False)
    write(output/'RESULT.private.json',result)
    if spec.get('feature_cache'):
        cache=checked(spec['feature_cache']);states=[];expert_actions=[];ids=[];anchors=[];packets={};pins={}
        for index,row in enumerate(cache['rows']):
            guard();episode=episodes[row['episode']]
            if episode['split']!='TRAIN' or row['index']!=index:raise ValueError('TRAIN-only cache order')
            if episode['id']not in packets:
                path=Path(spec['dataset']['path']).parent/episode['file']
                if digest(path)!=episode['sha256']:raise ValueError('Changed original numeric episode')
                with np.load(path,allow_pickle=False)as z:packets[episode['id']]={k:z[k].copy()for k in
                    ('actual_state10','committed_absolute10','native_command_anchor','valid_window','label_representation')}
                pins[episode['id']]=episode['sha256']
            packet=packets[episode['id']];anchor=row['anchor'];atom=int(packet['native_command_anchor'][anchor])
            if not packet['valid_window'][anchor]or packet['label_representation'].item()!='ABSOLUTE_COMMITTED_EEF10':
                raise ValueError('Only valid actually committed H40 supervision')
            states.append(packet['actual_state10'][anchor]);expert_actions.append(packet['committed_absolute10'][atom:atom+40]);ids.append(episode['id']);anchors.append(anchor)
        if len(states)!=1450 or len(set(zip(ids,anchors)))!=1450:raise ValueError('Unchanged1450 unique TRAIN windows')
        file=output/'TRAIN1450_PHYSICAL_LOSS_INPUTS.private.npz'
        np.savez(file,actual_state10=np.asarray(states),expert_policy_H40=np.asarray(expert_actions),episode=np.asarray(ids),anchor=np.asarray(anchors,dtype=np.int64))
        write(output/'TRAIN1450_PHYSICAL_LOSS_INPUTS_MANIFEST.private.json',dict(
            schema='TRAIN_ONLY_PHYSICAL_AUXILIARY_INPUTS_V3',dataset=spec['dataset'],feature_cache=spec['feature_cache'],
            rows=1450,numeric_file_sha256=digest(file),episode_numeric_pins=pins,
            expert_actions='ORIGINAL_ACTUALLY_CONSUMED_H40',stats_changed=False,DEV_or_TEST_included=False,
            source_sha=spec['source_sha'],GR00T_calls=0,physics_integrations=0))
    return result
