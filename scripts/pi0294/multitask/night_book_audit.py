"""Independent complete-episode audit of actual records, without simulation.

This audits the new source port, never a neural policy or target-WBC result.
It does not create a TRAIN manifest or refit any statistics.
"""
import hashlib,json,struct
from pathlib import Path,PurePosixPath
import numpy as np
from .libero_book_assets import native_site_contains
from scripts.pi0291_arx5_data.config import PARAMETERS as P
from .object_relations import pose


def audit(folder,*,task_kind='BOOK',execution_prefix=8):
    if task_kind not in('BOOK','ROBOCASA_CABINET','T1','T2'):raise ValueError('EXPLICIT_FIXED_SOURCE_RAW_AUDIT_KIND')
    if type(execution_prefix)is not int or execution_prefix not in(4,8)or(execution_prefix==4 and task_kind!='BOOK'):
        raise ValueError('EXPLICIT_BOOK_FEEDBACK_DIAGNOSTIC_PREFIX')
    folder=Path(folder);terminal=json.loads((folder/'TERMINAL.json').read_bytes())
    sha=lambda b:hashlib.sha256(b).hexdigest()
    proof={'schema':'NIGHT_SOURCE_BOOK_INDEPENDENT_RAW_AUDIT_V1'if task_kind=='BOOK'else'SHARED_CUBE_INDEPENDENT_RAW_AUDIT_V1'if task_kind in('T1','T2')else'NIGHT_SOURCE_CABINET_INDEPENDENT_RAW_AUDIT_V1','raw_sha256':sha((folder/'ACTUAL_RECORDS.jsonl').read_bytes()),
        'terminal_sha256':sha((folder/'TERMINAL.json').read_bytes()),'new_simulator_or_model_calls':0,
        'task_success':False,'qualified_source_pilot':False,'training_admitted':False}
    if execution_prefix==4:
        proof.update(schema='BOOK_STUDENT_PREFIX4_INDEPENDENT_RAW_AUDIT_V1',execution_prefix=4,formal_original_DEV=False)
    rows=[json.loads(s)for s in(folder/'ACTUAL_RECORDS.jsonl').read_text().splitlines()]
    qualification=[r['value']for r in rows if r['kind']=='qualification']
    if len(qualification)!=1:
        return dict(proof,status='INCOMPLETE_BEFORE_NATIVE_QUALIFICATION',error=terminal['error'])
    q=qualification[0];gb=q['geom_bindings'];robots=set(q['robot_geom_ids']);arms=set(q['arm_geom_ids']);obstacles=set(q['obstacle_geom_ids'])
    cabinet=None
    if task_kind=='ROBOCASA_CABINET':
        from .cabinet_raw_checks import CabinetChecks
        cabinet=CabinetChecks(folder,q)
    warm=[r['value']for r in rows if r['kind']=='warmup_tick']
    ticks=[r['value']for r in rows if r['kind']=='native_tick']
    obs=[r['value']for r in rows if r['kind']=='observation']
    submitted=[r['value']for r in rows if r['kind']=='submitted_interval']
    completed=[r['value']for r in rows if r['kind']=='completed_interval']
    plan_rows=[r['value']for r in rows if r['kind']=='planned_H40']
    plans={r['ref_id']:r for r in plan_rows}
    problems=[];safe=True
    def native_safe(s):
        if not all(np.isfinite(np.asarray(s[k])).all()for k in('qpos','qvel','tcp','arm_q','arm_qd')):return False
        if not s['safety_valid'] or s['native_abort']:return False
        bounds=np.asarray(q['original_arm_bounds']);arm=np.asarray(s['arm_q'])
        if np.any(arm<bounds[:,0]-.006)or np.any(arm>bounds[:,1]+.006):return False
        if np.any(np.abs(s['arm_qd'])>P['arm_speed_stop_rad_s']):return False
        if np.any(np.abs(np.asarray(s['qvel'])[q['original_finger_v_indices']])>P['finger_speed_stop_m_s']):return False
        f=np.asarray(s['finger_q'])
        if np.any(f<-.001)or np.any(f>P['finger_travel']+.001):return False
        if np.linalg.norm(s['tool_linear_velocity'])>P['tool_speed_stop_m_s']:return False
        for c in s['contacts']:
            pair={c['geom0'],c['geom1']}
            if c['distance_m'] < -P['penetration_stop_m']:return False
            if pair&robots and pair&obstacles:return False
            if pair&arms and pair&{gb['table'],gb['floor']}:return False
        return True
    # A driver's failing tick may already have integrated. Its actual state
    # must participate in safety even though it never completed an interval.
    all_native=[r['value']for r in rows if r['kind']in('warmup_tick','native_tick','native_tick_failure')]
    for i,row in enumerate(all_native,1):
        s=row['actual_state']
        if row['native_step']!=i or abs(s['time_s']-i*.002)>1e-8:problems.append('NATIVE_CLOCK_OR_STEP_SEQUENCE');break
        if cabinet is not None:cabinet.validate(s)
        if not native_safe(s):safe=False
    terminal_states=[r['actual_state']for r in rows if r['kind']=='terminal_failure'and 'actual_state'in r]
    for s in terminal_states:
        if not native_safe(s):safe=False
    # A mid-interval failure cannot be relabelled as a complete episode.
    complete=(terminal['status']=='RAW_SOURCE_COMPLETE_UNAUDITED' and len(warm)==2500 and len(ticks)==16000
        and len(obs)==641 and len(submitted)==len(completed)==640 and len(plans)==640//execution_prefix
        and terminal['actual_integrations']==18500 and terminal['actual_RGB_pairs']==terminal['recorded_RGB_pairs']==642)
    if not complete:problems.append('FULL_5S_WARMUP_AND_32S_TASK_NOT_COMPLETE')
    expected=['qualification']+['warmup_tick']*2500+['warmup_complete']
    for i in range(640):
        expected+=['observation']+(['planned_H40']if i%execution_prefix==0 else[])+['submitted_interval']+['native_tick']*25+['completed_interval']
    expected+=['observation']
    if [r['kind']for r in rows]!=expected:problems.append('COMMAND_MUST_PRECEDE_ITS_NATIVE_TICKS_AND_COMPLETION')
    if q['steps']!=0 or q['simulation_time']!=0 or q['new_constructor']!=1:
        problems.append('QUALIFICATION_BEFORE_ANY_INTEGRATION')
    if len(plan_rows)!=640//execution_prefix:problems.append(f'ONE_ACTUAL_H40_PER_PREFIX{execution_prefix}_BOUNDARY')
    for k,p in enumerate(plan_rows):
        action=np.asarray(p['rows10']);index=k*execution_prefix
        expected_id=sha(json.dumps(dict(index=p['observation_index'],source_time_s=p['source_time_s'],
            rows10=action.tolist()),sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        if (p['ref_id']!=expected_id or p['observation_index']!=index or action.shape!=(40,10)
                or not np.isfinite(action).all()or abs(p['source_time_s']-5-index*.05)>1e-8
                or p['committed_prefix_count']!=0 or p['role']!='UNEXECUTED_REFERENCE_EVIDENCE_NOT_TRAINING_LABEL'):
            problems.append('ORIGINAL_H40_HASH_CLOCK_AND_UNEXECUTED_SUFFIX');break
    for i,o in enumerate(obs):
        if o['index']!=i or o['native_step']!=2500+i*25 or abs(o['time_s']-5-i*.05)>1e-8:
            problems.append('OBSERVATION_CLOCK');break
        for name in('body_rgb','wrist_rgb'):
            pin=o['policy_obs'][name];relative=PurePosixPath(pin['file'])
            if relative.is_absolute()or len(relative.parts)!=1 or '..'in relative.parts:
                raise ValueError('OWN_RAW_IMAGE_BASENAME_ONLY')
            file=folder/pin['file']
            if file.is_symlink():raise ValueError('NO_IMAGE_SYMLINK')
            b=file.read_bytes()
            if len(b)!=pin['bytes']or sha(b)!=pin['sha256']:problems.append('ACTUAL_RGB_BYTES')
            if (len(b)<33 or b[:8]!=b'\x89PNG\r\n\x1a\n'or b[12:16]!=b'IHDR'
                    or struct.unpack('>IIBBBBB',b[16:29])!=(256,256,8,2,0,0,0)):
                problems.append('ACTUAL_256_RGB_PNG_NOT_A_PLACEHOLDER')
            if abs(o['calibration'][name]['timestamp_s']-o['time_s'])>1e-8:problems.append('CAMERA_CLOCK')
            pose(o['calibration'][name]['T_world_optical']);intrinsics=np.asarray(o['calibration'][name]['K'])
            if intrinsics.shape!=(3,3)or not np.isfinite(intrinsics).all()or intrinsics[0,0]<=0 or intrinsics[1,1]<=0:
                problems.append('ACTUAL_CAMERA_INTRINSICS')
    for i,c in enumerate(completed):
        if i>=len(submitted):problems.append('COMPLETED_WITHOUT_SUBMITTED_INTERVAL');break
        s=submitted[i];p=plans.get(c['ref_id'])
        if (p is None or p['observation_index']!=i-i%execution_prefix or c['ref_id']!=s['ref_id'] or c['observation_index']!=i
                or s['observation_index']!=i or c['prefix_index']!=s['prefix_index']or c['prefix_index']!=i%execution_prefix
                or c['start_native_step']!=2500+i*25 or c['end_native_step']!=2525+i*25
                or s['start_native_step']!=c['start_native_step']or s['expected_end_native_step']!=c['end_native_step']
                or abs(c['source_time_s']-5-i*.05)>1e-8 or c['source_time_s']!=s['source_time_s']
                or s['completed']is not False or c['completed']is not True
                or s['training_admitted']is not False or c['training_admitted']is not False
                or not np.array_equal(c['action_EEF10'],s['action_EEF10'])
                or not np.array_equal(c['action_EEF10'],p['rows10'][i%execution_prefix])):
            problems.append('ONLY_ACTUALLY_COMPLETED_SUBMITTED_PREFIX_CAN_LABEL');break
    def goal_and_extra(s):
        if cabinet is not None:return cabinet.goal_and_extra(s)
        goal=native_site_contains(site_position=s['native_goal_site_position'],site_rotation=s['native_goal_site_rotation'],
            half_size=s['native_goal_site_half_size'],object_body_origin=s['native_book_body_origin'])
        support=any(gb['cube_geom']in(c['geom0'],c['geom1'])and bool({c['geom0'],c['geom1']}&obstacles)
            and c['normal_force_N']>=P['minimum_contact_force_N'] for c in s['contacts'])
        robot_contact=any(gb['cube_geom']in(c['geom0'],c['geom1'])and bool({c['geom0'],c['geom1']}&robots)
            for c in s['contacts'])
        return bool(goal and support and not robot_contact and s['gripper_fraction']>=.85
            and np.linalg.norm(np.asarray(s['tcp'])[:3,3]-np.asarray(s['object_pose'])[:3,3])>=.08
            and np.linalg.norm(s['object_linear_velocity'])<=.02 and np.linalg.norm(s['object_angular_velocity'])<=.1)
    final_native_goal=None;hold=False
    cube_score=None
    if task_kind in('T1','T2'):
        from .cube_raw_score import score
        card=json.loads((folder/'CARD.json').read_bytes())
        if card['task']!=task_kind:raise PermissionError('CUBE_SCORER_EXACT_RECORDED_TASK')
        cube_score=score(rows,card,complete=complete)
        hold=cube_score['persistent_success']
    elif ticks:
        s=ticks[-1]['actual_state']
        final_native_goal=(cabinet.native(s)['native_is_open']if cabinet is not None else native_site_contains(site_position=s['native_goal_site_position'],
            site_rotation=s['native_goal_site_rotation'],half_size=s['native_goal_site_half_size'],object_body_origin=s['native_book_body_origin']))
        hold=len(ticks)>=1001 and all(goal_and_extra(r['actual_state'])for r in ticks[-1001:])
    if cabinet is not None:
        if not obs or not cabinet.native(obs[0]['actual_state'])['native_is_closed']:problems.append('TASK_MUST_BEGIN_WITH_NATIVE_CLOSED_DOOR')
        if not any(cabinet.grasp(r['actual_state'])for r in ticks):problems.append('NO_ACTUAL_ORIGINAL_HANDLE_GRASP')
    success=bool(complete and safe and not problems and hold)
    return dict(proof,status='COMPLETE_SOURCE_PILOT_QUALIFIED'if success else'INDEPENDENT_SOURCE_FAILURE_OR_INCOMPLETE',
        **({'original_cube_task_score':cube_score}if cube_score is not None else{('native_LIBERO_goal_final'if cabinet is None else'native_ROBOCASA_goal_final'):final_native_goal}),additional_support_release_withdraw_hold_pass=hold,
        complete_episode=complete,original_native_safety_pass=safe,integrity_problems=sorted(set(problems)),
        actual_integrations=terminal['actual_integrations'],recorded_native_ticks=len(all_native),
        recorded_failed_native_ticks=sum(r['kind']=='native_tick_failure'for r in rows),
        terminal_safety_states_checked=len(terminal_states),
        actual_observations=len(obs),completed_intervals=len(completed),
        task_success=success,qualified_source_pilot=success and execution_prefix==8,terminal_error=terminal['error'])
