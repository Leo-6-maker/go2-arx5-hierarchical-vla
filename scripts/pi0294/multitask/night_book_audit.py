"""Independent complete-episode audit of actual records, without simulation.

This audits the new source port, never a neural policy or target-WBC result.
It does not create a TRAIN manifest or refit any statistics.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from .libero_book_assets import native_site_contains
from scripts.pi0291_arx5_data.config import PARAMETERS as P


def audit(folder):
    folder=Path(folder);terminal=json.loads((folder/'TERMINAL.json').read_bytes())
    sha=lambda b:hashlib.sha256(b).hexdigest()
    proof={'schema':'NIGHT_SOURCE_BOOK_INDEPENDENT_RAW_AUDIT_V1','raw_sha256':sha((folder/'ACTUAL_RECORDS.jsonl').read_bytes()),
        'terminal_sha256':sha((folder/'TERMINAL.json').read_bytes()),'new_simulator_or_model_calls':0,
        'task_success':False,'qualified_source_pilot':False,'training_admitted':False}
    rows=[json.loads(s)for s in(folder/'ACTUAL_RECORDS.jsonl').read_text().splitlines()]
    qualification=[r['value']for r in rows if r['kind']=='qualification']
    if len(qualification)!=1:
        return dict(proof,status='INCOMPLETE_BEFORE_NATIVE_QUALIFICATION',error=terminal['error'])
    q=qualification[0];gb=q['geom_bindings'];robots=set(q['robot_geom_ids']);arms=set(q['arm_geom_ids']);obstacles=set(q['obstacle_geom_ids'])
    warm=[r['value']for r in rows if r['kind']=='warmup_tick']
    ticks=[r['value']for r in rows if r['kind']=='native_tick']
    obs=[r['value']for r in rows if r['kind']=='observation']
    submitted=[r['value']for r in rows if r['kind']=='submitted_interval']
    completed=[r['value']for r in rows if r['kind']=='completed_interval']
    plans={r['value']['ref_id']:r['value']for r in rows if r['kind']=='planned_H40'}
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
    all_native=warm+ticks
    for i,row in enumerate(all_native,1):
        s=row['actual_state']
        if row['native_step']!=i or abs(s['time_s']-i*.002)>1e-8:problems.append('NATIVE_CLOCK_OR_STEP_SEQUENCE');break
        if not native_safe(s):safe=False
    # A mid-interval failure cannot be relabelled as a complete episode.
    complete=(terminal['status']=='RAW_SOURCE_COMPLETE_UNAUDITED' and len(warm)==2500 and len(ticks)==16000
        and len(obs)==641 and len(submitted)==len(completed)==640 and len(plans)==80)
    if not complete:problems.append('FULL_5S_WARMUP_AND_32S_TASK_NOT_COMPLETE')
    for i,o in enumerate(obs):
        if o['index']!=i or o['native_step']!=2500+i*25 or abs(o['time_s']-5-i*.05)>1e-8:
            problems.append('OBSERVATION_CLOCK');break
        for name in('body_rgb','wrist_rgb'):
            pin=o['policy_obs'][name];b=(folder/pin['file']).read_bytes()
            if len(b)!=pin['bytes']or sha(b)!=pin['sha256']:problems.append('ACTUAL_RGB_BYTES')
            if abs(o['calibration'][name]['timestamp_s']-o['time_s'])>1e-8:problems.append('CAMERA_CLOCK')
    for i,c in enumerate(completed):
        s=submitted[i];p=plans.get(c['ref_id'])
        if (p is None or c['ref_id']!=s['ref_id'] or c['observation_index']!=i
                or c['start_native_step']!=2500+i*25 or c['end_native_step']!=2525+i*25
                or not np.array_equal(c['action_EEF10'],s['action_EEF10'])
                or not np.array_equal(c['action_EEF10'],p['rows10'][i%8])):
            problems.append('ONLY_ACTUALLY_COMPLETED_SUBMITTED_PREFIX_CAN_LABEL');break
    def goal_and_extra(s):
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
    if ticks:
        s=ticks[-1]['actual_state'];final_native_goal=native_site_contains(site_position=s['native_goal_site_position'],
            site_rotation=s['native_goal_site_rotation'],half_size=s['native_goal_site_half_size'],object_body_origin=s['native_book_body_origin'])
        hold=len(ticks)>=1001 and all(goal_and_extra(r['actual_state'])for r in ticks[-1001:])
    success=bool(complete and safe and not problems and hold)
    return dict(proof,status='COMPLETE_SOURCE_PILOT_QUALIFIED'if success else'INDEPENDENT_SOURCE_FAILURE_OR_INCOMPLETE',
        native_LIBERO_goal_final=final_native_goal,additional_support_release_withdraw_hold_pass=hold,
        complete_episode=complete,original_native_safety_pass=safe,integrity_problems=sorted(set(problems)),
        actual_integrations=terminal['actual_integrations'],recorded_native_ticks=len(all_native),
        actual_observations=len(obs),completed_intervals=len(completed),
        task_success=success,qualified_source_pilot=success,terminal_error=terminal['error'])
