"""Original-action history replay and passive first-violation measurement only."""
import hashlib
from pathlib import Path

import numpy as np

from scripts.pi0294.fd1 import read, put, update, sha, seal_or_verify


def pad_clearance(engine):
    from scripts.pi0291_arx5_data.config import PARAMETERS as P
    rows=[]
    table=engine.geoms['table']
    table_center=engine.data.geom_xpos[table]
    half=engine.model.geom_size[table]
    for side in ['left','right']:
        g=engine.geoms[side+'_pad']
        center=engine.data.geom_xpos[g]
        axes=engine.data.geom_xmat[g].reshape(3,3)
        size=engine.model.geom_size[g]
        extent=np.abs(axes) @ size
        overlap=bool(np.all(center[:2]+extent[:2]>=table_center[:2]-half[:2])
                     and np.all(center[:2]-extent[:2]<=table_center[:2]+half[:2]))
        rows.append(dict(side=side,vertical_box_bottom_minus_table_top_m=float(center[2]-extent[2]-P['table_top']),
                         projected_box_overlaps_table=overlap))
    return rows


def geometry_category(check):
    if check['predicted_native_safety']==['TCP_VELOCITY']:
        return 'TCP_velocity'
    if check['predicted_native_safety']==['JOINT_LIMIT']:
        return 'joint_limit'
    pairs=check['illegal_pairs']
    if not pairs:
        return 'UNPROVEN'
    names=[set(x['geom_names']) for x in pairs]
    if any('table' in x and bool({'left_pad','right_pad'} & x) for x in names):
        return 'table-pad'
    if any('table' in x and p['original_rule']=='ARM_TABLE_OR_FLOOR' for x,p in zip(names,pairs)):
        return 'arm-table-other'
    if any('floor' in x for x in names):
        return 'table-floor-other'
    return 'penetration-other'


def request_geometry(engine, poses, opening, state):
    from scripts.pi0291_arx5_data.config import PARAMETERS as P
    from scripts.pi0294.scene import push_geometry
    from go2arx5_vla.trajectory.so3 import geodesic_angle
    basis=push_geometry(engine,[1.,0.])
    outputs=[]
    for index,(pose,g) in enumerate(zip(poses,opening)):
        for pad in basis['pads']:
            local=np.asarray(pad['center_tcp']).copy()
            local[1]+=(1 if pad['side']=='left' else -1)*float(g)*P['finger_travel']
            axes=np.asarray(pad['axes_tcp'])
            size=np.asarray(pad['half_size'])
            def clearance(xyz,rotation,local_point):
                return float((xyz+rotation@local_point)[2]-np.abs((rotation@axes)[2])@size-P['table_top'])
            now=state['tcp']
            actual_local=np.asarray(pad['center_tcp']).copy()
            actual_local[1]+=(1 if pad['side']=='left' else -1)*engine.data.qpos[engine.finger_q[0 if pad['side']=='left' else 1]]
            outputs.append(dict(prefix_point=index,side=pad['side'],requested_static_box_clearance_m=clearance(pose[:3,3],pose[:3,:3],local),
                actual_pose_actual_opening_static_clearance_m=clearance(now[:3,3],now[:3,:3],actual_local),
                requested_XYZ_only_static_clearance_m=clearance(pose[:3,3],now[:3,:3],actual_local),
                requested_SO3_only_static_clearance_m=clearance(now[:3,3],pose[:3,:3],actual_local),
                requested_gripper_only_static_clearance_m=clearance(now[:3,3],now[:3,:3],local)))
    return dict(actual_TCP=state['tcp'].tolist(),actual_gripper_fraction=float(state['gripper_fraction']),
        original_prefix8_requested_TCP=poses.tolist(),original_prefix8_requested_opening=opening.tolist(),
        request_minus_current_XYZ_m=(poses[:,:3,3]-state['tcp'][:3,3]).tolist(),
        request_minus_current_SO3_rad=np.asarray(geodesic_angle(poses[:,:3,:3],state['tcp'][:3,:3])).tolist(),
        passive_static_geometry_not_channel_replacement_rollout=True,static_pad_clearances=outputs,
        static_box_projections_are_not_dynamic_clearance_or_expert_error=True)


def replay_one(meter,row):
    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0292.relay_protocol import unseal
    from scripts.pi0294.e1.initial_history import qualify_recorded_initial
    from scripts.pi0294.e1.collision import collision_pairs
    from scripts.pi0294.scene import make_engine
    from scripts.pi0294.s1_preflight import clone, data_state, observed_state, fingerprint
    from scripts.pi0294.teacher import make_teacher
    from scripts.pi0294.episode import _robot_contact
    from go2arx5_vla.task_data.codec import pose10

    identifier=row['attempt_id'];folder=meter.source/'episodes'/identifier
    if any(sha(folder/name)!=digest for name,digest in row['original_files_sha256'].items()):
        raise ValueError('FD1_SEALED_ATLAS_INPUT_CHANGED_BEFORE_RECONSTRUCTION')
    if identifier in meter.ledger['case_reconstructions']:
        raise PermissionError('FD1_ONCE_PER_ORIGINAL_CASE_NO_RETRY')
    meter.ledger['case_reconstructions'].append(identifier)
    update(meter.path,meter.ledger)
    card=read(folder/'CARD.json')
    forecasts=read(folder/'S1_PREFLIGHTS.json')
    with np.load(folder/'highrate.npz',allow_pickle=False) as z:
        high=z['rows'].copy()
    last=len(high)//200
    if len(high)%200 or forecasts[-1]['decision']!='REJECT' or len(forecasts)!=last+1:
        raise ValueError('FD1_ONLY_ORIGINAL_SINGLE_LAST_REJECTION')
    meter.start();engine=make_engine(card,meter.inputs['asset_root'])
    maximum=0.;observation_error=0.;min_live_clearance=float('inf');prefixes=0
    try:
        initial=engine.snapshot();saved=read(folder/'INITIAL_STATE.json')
        if (hashlib.sha256(engine.xml.encode()).hexdigest()!=saved['model_xml_sha256']
                or not np.array_equal(initial['qpos'],saved['qpos']) or not np.array_equal(initial['qvel'],saved['qvel'])):
            raise ValueError('FD1_ORIGINAL_INITIAL_MODEL_AND_FULL_NATIVE_HISTORY')
        teacher=make_teacher(card,initial['object_pose'],engine)
        for sequence in range(last+1):
            meter.guard()
            state=engine.snapshot();state['robot_object_contact']=_robot_contact(engine,state)
            if sequence==0:
                qualify_recorded_initial(engine,folder,meter.guard)
            teacher.observe(state)
            obs=unseal(read(folder/'relay/observations'/f'{sequence:03d}.json'))
            error=float(np.max(np.abs(pose10(state['tcp'],state['gripper_fraction'])-obs['payload']['state_eef10'])))
            observation_error=max(observation_error,error)
            if error>1e-10 or obs['tick']!=engine.steps or obs['sequence']!=sequence:
                raise ValueError('FD1_ORIGINAL_FRESH_OBSERVATION_PARITY')
            bundle=read(folder/'relay/commands'/f'{sequence:03d}.json')
            action=np.asarray(unseal(bundle['action'])['payload']['trajectory_eef10'],dtype=np.float64)
            poses=matrix10(action);opening=action[:,9]
            digest=hashlib.sha256(poses[:8].tobytes()+opening[:8].tobytes()).hexdigest()
            original=forecasts[sequence]
            if digest!=original['action_sha256'] or original['sequence']!=sequence*8:
                raise ValueError('FD1_ORIGINAL_ACTION_ARRAY_AND_S1_IDENTITY')
            if sequence==last:
                before=fingerprint(engine)
                channels=request_geometry(engine,poses[:8],opening[:8],state)
                if fingerprint(engine)!=before:
                    raise ValueError('FD1_PASSIVE_MEASUREMENT_CHANGED_ORIGINAL_STATE')
                meter.start();shadow=clone(engine)
                try:
                    full_copy_error=float(np.max(np.abs(observed_state(shadow)-observed_state(engine))))
                    if full_copy_error>1e-10:
                        raise ValueError('FD1_FULL_NATIVE_IK_CONTROL_COPY_PARITY')
                    error=None
                    with meter.steps('shadow_integrations',200,'ORIGINAL-REJECT-'+identifier) as counter:
                        for tick in range(200):
                            meter.guard()
                            if tick and tick%25==0:shadow.snapshot()
                            previous=shadow.steps
                            try:
                                shadow.step(poses[tick//25],float(opening[tick//25]))
                            except Exception as exc:
                                error=type(exc).__name__+': '+str(exc)
                            finally:
                                counter['actual']+=shadow.steps-previous
                            state=shadow.snapshot()
                            if error:break
                    check=dict(decision='REJECT' if error or shadow.safety else 'PASS',
                        predicted_native_safety=list(shadow.safety),shadow_integrations=counter['actual'],
                        first_predicted_violation_offset_tick=counter['actual'] if shadow.safety else None,
                        illegal_pairs=collision_pairs(shadow),actual_TCP_at_violation=state['tcp'].tolist(),
                        actual_object_pose_at_violation=state['object_pose'].tolist(),
                        actual_gripper_fraction_at_violation=float(state['gripper_fraction']),
                        arm_joint_q_at_violation=state['arm_q'].tolist(),
                        arm_joint_margin_lower_m_or_rad=(shadow.data.qpos[shadow.arm_q]-shadow.arm_bounds[:,0]+.006).tolist(),
                        arm_joint_margin_upper_m_or_rad=(shadow.arm_bounds[:,1]+.006-shadow.data.qpos[shadow.arm_q]).tolist(),
                        TCP_linear_speed_m_s=float(np.linalg.norm(state['tool_linear_velocity'])),
                        pad_clearance_at_violation=pad_clearance(shadow),
                        raw_error=error,diagnostic_shadow_not_original_live_collision=True)
                    for name in ['decision','predicted_native_safety','shadow_integrations','first_predicted_violation_offset_tick']:
                        if check[name]!=original[name]:
                            raise ValueError('FD1_ORIGINAL_REJECTION_CATEGORY_OFFSET_PARITY:'+name)
                    if fingerprint(engine)!=before:
                        raise ValueError('FD1_SHADOW_CHANGED_HISTORY_ENGINE')
                    return dict(attempt_id=identifier,task=card['task'],status='QUALIFIED_ACTUAL_CAUSE_OF_ORIGINAL_PREDICTED_REJECTION',
                        original_source_sha=row['original_source_sha'],source_input_hashes=row['original_files_sha256'],
                        model_xml_sha256=saved['model_xml_sha256'],original_decision_time_s=engine.steps*.002,
                        original_forecast=original,reconstructed_forecast=check,geometry_category=geometry_category(check),
                        original_highrate_ticks_reproduced=len(high),maximum_original_highrate_error=maximum,
                        maximum_original_EEF_observation_error=observation_error,full_native_IK_cache_copy_error=full_copy_error,
                        original_highrate_fields='TIME_QPOS_QVEL_CTRL_TOOL_EXTERNAL_WRENCH_SAFETY',
                        original_unrecorded_IK_python_RNG_bitwise_identity_claim=False,
                        original_deterministic_complete_history_and_static_IK_reconstructed=True,
                        minimum_actual_history_pad_vertical_clearance_m=min_live_clearance,
                        request_geometry=channels,original_rejected_prefix_live_integrations=0,
                        original_results_changed=False,new_model_queries=0)
                finally:
                    shadow.close()
            ack_path=folder/'relay/acks'/f'{sequence:03d}.json'
            if original['decision']!='PASS' or not ack_path.is_file():
                raise ValueError('FD1_ONLY_SAVED_ACKED_HISTORY')
            ack=unseal(read(ack_path))
            if (ack['payload']['tick_after']!=(sequence+1)*200 or ack['payload']['integrations']!=200
                    or ack['payload']['command_sha256']!=bundle['command']['sha256']
                    or sha(folder/'relay'/f'physics-{sequence:03d}.npz')!=ack['payload']['physics_receipt_sha256']):
                raise ValueError('FD1_ORIGINAL_ACK_NATIVE_PROGRESS')
            with meter.steps('shadow_integrations',200,'HISTORY-'+identifier+'-'+str(sequence)) as counter:
                for tick in range(200):
                    meter.guard()
                    if tick and tick%25==0:
                        s=engine.snapshot();s['robot_object_contact']=_robot_contact(engine,s);teacher.observe(s)
                    previous=engine.steps
                    try:
                        engine.step(poses[tick//25],float(opening[tick//25]))
                    finally:
                        counter['actual']+=engine.steps-previous
                    state=engine.snapshot()
                    actual=np.r_[engine.data.time,engine.data.qpos,engine.data.qvel,engine.data.ctrl,
                        engine.data.xfrc_applied[engine.ids['tool']],float(not state['safety_valid'])]
                    error=float(np.max(np.abs(actual-high[engine.steps-1])));maximum=max(maximum,error)
                    if error>1e-10 or engine.safety:
                        raise ValueError('FD1_RESTORE_UNQUALIFIED_ORIGINAL_NATIVE_HISTORY')
                    min_live_clearance=min(min_live_clearance,min(x['vertical_box_bottom_minus_table_top_m'] for x in pad_clearance(engine)))
            prefixes+=1
        raise ValueError('FD1_ORIGINAL_LAST_REJECTION_NOT_REACHED')
    finally:
        engine.close()


def run_shadow(meter):
    atlas=read(meter.root/'evidence/ATLAS_RESULT.private.json')
    if atlas['status']!='CPU_LOG_AUDIT_PASS_NO_NEW_PHYSICS' or atlas['first_geometry_not_in_archive']!=23:
        raise PermissionError('FD1_PROVEN_ARCHIVE_GEOMETRY_GAP_REQUIRED')
    rows=sorted([r for r in atlas['actual_rows'] if r['rejection']],key=lambda r:r['attempt_id'])
    if len(rows)!=23 or atlas['planned_all23_reconstruction_tick_upper']>500000:
        raise PermissionError('FD1_PREBOUND_ALL23_WITHIN_TICK_AND_START_CAP')
    seal_or_verify(meter.root/'plans/SHADOW_EXACT_CASES.json',dict(selection='ALL23_ORIGINAL_REJECTIONS_NO_OUTCOME_SELECTION',
        ids=[r['attempt_id'] for r in rows],ticks_upper=atlas['planned_all23_reconstruction_tick_upper'],starts_upper=46,
        successful_controls='NOT_RUN_NO_CLAIM_EXPERT_DYNAMIC_FEASIBILITY',original_actions_only=True))
    result=[]
    if meter.operation=='shadow-remaining-01':
        previous=read(meter.root/'evidence/SHADOW-RECOVERY-02_RESULT.private.json')
        result=[r for r in previous['cases'] if r['attempt_id'] in meter.ledger['case_reconstructions']]
        if len(result)!=2 or sum(r['status']=='RESTORE_UNQUALIFIED' for r in result)!=1:
            raise PermissionError('FD1_RETAIN_EXACT_PRIOR_TWO_ATTEMPTS_NO_RETRY')
        if next(r for r in result if r['status']=='RESTORE_UNQUALIFIED')['error']!='ValueError: FD1_PASSIVE_MEASUREMENT_CHANGED_ORIGINAL_STATE':
            raise PermissionError('FD1_ONLY_NEW_PASSIVE_READER_REPAIR_BOUNDARY')
    for row in [r for r in rows if r['attempt_id'] not in meter.ledger['case_reconstructions']]:
        meter.guard()
        try:
            value=replay_one(meter,row)
        except (ValueError,PermissionError,RuntimeError) as exc:
            value=dict(attempt_id=row['attempt_id'],task=row['adjudication']['task'],status='RESTORE_UNQUALIFIED',
                geometry_category='UNPROVEN',error=type(exc).__name__+': '+str(exc))
            result.append(value)
            update(meter.root/'evidence/SHADOW_PROGRESS.private.json',dict(cases=result,shadow_ticks=meter.ticks,engine_starts=meter.starts))
            # Identity/parity failure closes this lane; never retry or weaken it.
            break
        result.append(value)
        update(meter.root/'evidence/SHADOW_PROGRESS.private.json',dict(cases=result,shadow_ticks=meter.ticks,engine_starts=meter.starts))
    done={r['attempt_id'] for r in result}
    result.extend(dict(attempt_id=r['attempt_id'],task=r['adjudication']['task'],status='NOT_RUN_AFTER_PARITY_STOP',geometry_category='UNPROVEN')
                  for r in rows if r['attempt_id'] not in done)
    return dict(status='NATIVE_SHADOW_PASS_ALL23_ORIGINAL_REJECTIONS' if all(r['status'].startswith('QUALIFIED_') for r in result)
                else 'PARTIAL_SHADOW_QUALIFICATION_GAP_PRESERVED',cases=result,
        original_rejections=23,unique_case_reconstructions=len(done),actual_new_live_episodes=0,new_model_queries=0,
        expert_channel_replacement_or_new_teacher=0,training_labels=0,unrecorded_original_hidden_state_identity_claimed=False)
