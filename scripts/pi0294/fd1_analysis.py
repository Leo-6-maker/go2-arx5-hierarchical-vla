"""Pure saved-evidence joins for FD1; no simulator, teacher or neural imports."""
from collections import Counter, defaultdict
import math


def range_of(values):
    values=list(values)
    return dict(min=min(values),max=max(values)) if values else None


def layout(card):
    source,goal=card['source_xy'],card['goal_xy']
    delta=[goal[i]-source[i] for i in range(2)]
    return dict(source_xy_m=source,goal_xy_m=goal,direction_rad=math.atan2(delta[1],delta[0]),
                distance_m=math.hypot(*delta),cube_yaw_rad=card['cube_yaw'],family=card['family'],visual=card['visual'])


def compare_banks(left,right,noise):
    a={x['sequence']:x for x in left['commands']}
    b={x['sequence']:x for x in right['commands']}
    shared=sorted(set(a)&set(b))
    if not shared or shared[0]!=0:
        raise ValueError('FD1_ORIGINAL_INITIAL_PAIR_OBSERVATIONS_REQUIRED')
    first_action=next((i for i in shared if a[i]['trajectory_sha256']!=b[i]['trajectory_sha256']),None)
    first_camera=next((i for i in shared if a[i]['camera_hashes']!=b[i]['camera_hashes']),None)
    first_state=next((i for i in shared if a[i]['input_state_eef10']!=b[i]['input_state_eef10']),None)
    draws=[]
    for row in [left,right]:
        first=row['commands'][0]
        original=noise[(row['attempt_id'],0)]
        if original['action_sha256']!=first['action_sha256']:
            raise ValueError('FD1_ORIGINAL_ACTION_AND_NOISE_RECEIPT_JOIN')
        draws.append({k:original[k] for k in ['attempt_id','seed','noise_sha256','namespace','action_sha256','arrays_sha256']})
    return dict(attempt_ids=[left['attempt_id'],right['attempt_id']],scene_group=left['adjudication']['scene_group'],task=left['adjudication']['task'],
        initial_state_eef10_identical=a[0]['input_state_eef10']==b[0]['input_state_eef10'],
        initial_two_camera_pixels_hashes_identical=a[0]['camera_hashes']==b[0]['camera_hashes'],
        first_recorded_H40_action_divergence_sequence=first_action,first_recorded_H40_action_divergence_s=None if first_action is None else first_action*.4,
        first_recorded_EEF_input_state_divergence_sequence=first_state,first_recorded_EEF_input_state_divergence_s=None if first_state is None else first_state*.4,
        first_recorded_camera_divergence_sequence=first_camera,first_recorded_camera_divergence_s=None if first_camera is None else first_camera*.4,
        divergence_sampling_interval_s=.4,observation_packet_UUIDs_not_used_as_pixel_difference=True,original_initial_draws=draws,
        outcomes=[r['adjudication']['status'] for r in [left,right]],
        final_goal_errors_m=[r['actual_events']['final_goal_center_error_m'] for r in [left,right]],
        minimum_goal_errors_m=[r['actual_events']['minimum_goal_center_error_20Hz_m'] for r in [left,right]],
        terminal_hold=[r['unchanged_original_score']['hold_subpredicates'] for r in [left,right]],
        noise_is_unique_causal_mechanism_proven=False)


def geometry_row(row):
    if not row['status'].startswith('QUALIFIED_'):
        return dict(row)
    check=row['reconstructed_forecast'];original=row['original_forecast']
    offset=check['first_predicted_violation_offset_tick']
    index=(offset-1)//25
    channels=row['request_geometry']
    passive=[x for x in channels['static_pad_clearances'] if x['prefix_point']==index]
    return dict(attempt_id=row['attempt_id'],task=row['task'],status=row['status'],geometry_category=row['geometry_category'],
        original_source_sha=row['original_source_sha'],original_files_sha256=row['source_input_hashes'],model_xml_sha256=row['model_xml_sha256'],
        original_action_sha256=original['action_sha256'],original_decision_time_s=row['original_decision_time_s'],
        first_predicted_violation_offset_tick=offset,first_predicted_violation_absolute_s=row['original_decision_time_s']+offset*.002,
        original_native_rule_categories=original['predicted_native_safety'],first_illegal_pairs=check['illegal_pairs'],
        TCP_linear_speed_m_s=check['TCP_linear_speed_m_s'],arm_joint_q_at_violation=check['arm_joint_q_at_violation'],
        arm_joint_margin_lower_rad=check['arm_joint_margin_lower_m_or_rad'],arm_joint_margin_upper_rad=check['arm_joint_margin_upper_m_or_rad'],
        pad_clearance_at_first_violation=check['pad_clearance_at_violation'],
        minimum_actual_history_pad_vertical_clearance_m=row['minimum_actual_history_pad_vertical_clearance_m'],
        request_point_at_first_violation=index,
        request_XYZ_m=[r[3] for r in channels['original_prefix8_requested_TCP'][index][:3]],
        request_rotation=[r[:3] for r in channels['original_prefix8_requested_TCP'][index][:3]],
        request_gripper_fraction=channels['original_prefix8_requested_opening'][index],
        request_minus_predecision_XYZ_m=channels['request_minus_current_XYZ_m'][index],
        request_minus_predecision_SO3_rad=channels['request_minus_current_SO3_rad'][index],
        predecision_gripper_fraction=channels['actual_gripper_fraction'],
        passive_channel_static_clearance_at_trigger=passive,
        passive_channel_geometry_is_not_an_expert_substitution_or_causal_dynamic_trial=True,
        maximum_original_highrate_error=row['maximum_original_highrate_error'],
        maximum_original_EEF_observation_error=row['maximum_original_EEF_observation_error'],
        full_native_IK_cache_copy_error=row['full_native_IK_cache_copy_error'],
        original_unrecorded_hidden_state_bitwise_identity_claim=False,
        original_rejected_prefix_live_integrations=0,predicted_not_actual_live_collision=True)


def derive(atlas,shadow,noise_receipts,train):
    rows=atlas['actual_rows']
    if len(rows)!=42 or len({r['attempt_id'] for r in rows})!=42:
        raise ValueError('FD1_ORIGINAL42_UNIQUE_ATTEMPTS')
    qualified=[r for r in rows if r['adjudication']['status']!='ENGINEERING_INCOMPLETE']
    if len(qualified)!=40:
        raise ValueError('FD1_ORIGINAL40_FIXED_CELL_DENOMINATOR')
    noise={(r['attempt_id'],r['sequence']):r for r in noise_receipts['actual_rows']}
    count=0
    for r in rows:
        for c in r['commands']:
            n=noise[(r['attempt_id'],c['sequence'])]
            if n['action_sha256']!=c['action_sha256']:
                raise ValueError('FD1_ALL_ORIGINAL_COMMAND_NOISE_ARRAY_JOIN')
            count+=1
    if count!=2319 or count!=len(noise):
        raise ValueError('FD1_ALL2319_ORIGINAL_GENERATIONS_PRESERVED')
    groups=defaultdict(list)
    for r in qualified:
        groups[(r['adjudication']['task'],r['adjudication']['scene_group'])].append(r)
    pairs=[]
    for key,values in sorted(groups.items()):
        values=sorted(values,key=lambda r:r['card']['confirmation_noise_bank'])
        if len(values)!=2 or [r['card']['confirmation_noise_bank'] for r in values]!=[29201,29202]:
            raise ValueError('FD1_EXACT_TWO_ORIGINAL_BANKS_EACH_SCENE')
        pair=compare_banks(*values,noise)
        pair['successes']=sum(r['adjudication']['status']=='SUCCESS' for r in values)
        pair['discordant']=pair['successes']==1
        pairs.append(pair)
    public=[]
    for r in rows:
        public.append({k:r[k] for k in ['attempt_id','adjudication','card','original_source_sha','original_files_sha256',
            'all_raw_manifest_members_verified','model_manifest_sha256','plan_sha256','actual_final_time_s','original_teacher_final_label',
            'original_teacher_labels','teacher_approach_vs_actual_event_mismatch','actual_events','rejection','unchanged_original_score','scene_xml_sha256','video']})
    geometry=[geometry_row(r) for r in shadow['cases']]
    rejects=[r for r in qualified if r['rejection']]
    terminal=[r for r in qualified if r['adjudication']['status']=='VALID_TASK_FAIL' and not r['rejection']]
    statistics={}
    for task in ['T1','T2']:
        t=[r for r in qualified if r['adjudication']['task']==task]
        veto=[r for r in t if r['rejection']]
        statistics[task]=dict(original_conditions=len(t),independent_scene_groups=10,
            success=sum(r['adjudication']['status']=='SUCCESS' for r in t),S1_reject=len(veto),
            safe_full_goal_fail=sum(r['adjudication']['status']=='VALID_TASK_FAIL' and not r['rejection'] for r in t),
            paired_success_counts=dict(Counter(str(p['successes'])+'_OF_2' for p in pairs if p['task']==task)),
            veto_times_s=range_of(r['actual_final_time_s'] for r in veto),
            physical_event_phases=dict(Counter(r['actual_events']['actual_phase'] for r in veto)),
            original_teacher_approach_event_mismatches=sum(r['teacher_approach_vs_actual_event_mismatch'] for r in veto),
            veto_with_prior_contact=sum(r['actual_events']['first_robot_object_contact_s'] is not None for r in veto),
            veto_with_at_least10cm_actual_motion=sum(r['actual_events']['highrate_maximum_displacement_m']>=.1 for r in veto))
    train_layouts=[]
    for r in train['rows']:
        g=r['geometry']
        train_layouts.append(dict(id=r['id'],family=g['family'],source_xy=g['source_xy_m'],goal_xy=g['goal_xy_m'],cube_yaw=g['cube_yaw_rad'],
            direction_rad=math.atan2(g['goal_xy_m'][1]-g['source_xy_m'][1],g['goal_xy_m'][0]-g['source_xy_m'][0]),distance_m=g['source_goal_distance_m']))
    confirmation_layouts=[dict(scene_group=r['adjudication']['scene_group'],**layout(r['card']))
                          for r in qualified if r['adjudication']['task']=='T2' and r['card']['confirmation_noise_bank']==29201]
    layout_ranges={}
    for label,population in [('TRAIN_T2_30',train_layouts),('CONFIRM_T2_10',confirmation_layouts)]:
        layout_ranges[label]={field:range_of(x[field] for x in population) for field in ['direction_rad','distance_m','cube_yaw_rad' if label=='CONFIRM_T2_10' else 'cube_yaw']}
    return dict(status='CPU_ANALYSIS_OF_ORIGINAL_EVIDENCE_NOT_NEW_EVALUATION',attempts=public,
        stats=statistics,pairs=pairs,discordant_pairs=[p for p in pairs if p['discordant']],
        geometry=geometry,geometry_category_counts=dict(Counter(r['geometry_category'] for r in geometry)),
        qualified_geometry_cases=sum(r['status'].startswith('QUALIFIED_') for r in geometry),
        collision_geometry_cases=sum(bool(r.get('first_illegal_pairs')) for r in geometry),
        terminal_failures=[dict(attempt_id=r['attempt_id'],actual_events=r['actual_events'],original_score=r['unchanged_original_score']) for r in terminal],
        original_noise_array_joins=count,original_attempts=42,qualified_conditions=40,
        layout_ranges=layout_ranges,TRAIN_T2_layouts=train_layouts,CONFIRM_T2_layouts=confirmation_layouts,
        TRAIN_stage_anchor_counts=train['anchor_stage_counts'],TRAIN_all30_have_every_original_teacher_stage=True,
        layout_and_visuals_are_confounded_not_causal=True,
        actual_phase_endpoint_zone_TCP_and_gripper_resolution_s=.05,
        actual_contact_displacement_lift_resolution_s=.002,
        twenty_Hz_endpoint_can_precede_rejection_decision_by_up_to_s=.05,
        diagnostic_thresholds_do_not_change_original_scores=True,new_model_queries=0,new_training_labels=0)
