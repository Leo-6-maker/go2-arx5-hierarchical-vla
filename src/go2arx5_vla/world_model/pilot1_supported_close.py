"""Single-process supported pinch, hold and release in a separately qualified scene.

Robot, WBC, force thresholds and native grasp definition are unchanged. All
contact/hold durations use every original native tick, never geometry labels.
"""
from pathlib import Path
import json,struct
import numpy as np
from .storage import digest

PHASE='collect-target-pilot1-supported-close'
RULE='FIXED_EEF_HOLD40_NATIVE_GRIPPER_CLOSE_HOLD_RELEASE_V1'
STAGES=((0,'OPEN_STEADY'),(64,'CLOSE'),(96,'SUPPORTED_HOLD'),(120,'RELEASE'),(152,'OPEN_HOLD'))


def stage_for(atom):return next(name for start,name in reversed(STAGES)if atom>=start)


def validate(spec,previous):
    path=Path(spec['episode_manifest']);e=json.loads(path.read_text())
    expected=dict(schema='WM02_PILOT1_SUPPORTED_CLOSE_V1',phase=PHASE,reference_rule=RULE,
                  purpose='SUPPORTED_CLOSE_COMMISSION',split='TRAIN',task_atoms=168,prehold_atoms=40,
                  screen_atom=48,close_logical_opening=.4,required_supported_grasp_native_ticks=500,
                  required_released_native_ticks=100,minimum_open_finger_q_m=.040,contact_force_threshold_N=.05,
                  EEF_offset_xyz_m=[0.,0.,0.],stages=[list(s)for s in STAGES])
    if (digest(path)!=spec['episode_manifest_sha256']or any(e.get(k)!=v for k,v in expected.items())
            or e.get('run')!=spec['registered_root']or not spec.get('pilot1_binding')or spec.get('parallel_batch_spec')):
        raise PermissionError('EXACT_FRESH_SINGLE_SUPPORTED_CLOSE_PROTOCOL_REQUIRED')
    assets=json.loads(Path(spec['scene_allowlist']).read_text());scenes=[s for s in assets['scenes']if s['scene_id']==e['scene_id']]
    if (digest(spec['scene_allowlist'])!=spec['scene_allowlist_sha256']or len(scenes)!=1
            or scenes[0]['parent']!=e['parent']or scenes[0]['split']!='TRAIN'
            or scenes[0]['model_path']!=spec['model_path']or scenes[0]['model_sha256']!=digest(spec['model_path'])
            or assets['native_binary_sha256']!=digest(spec['native_binary'])):
        raise PermissionError('EXACT_NEW_SUPPORTED_CLOSE_SCENE_REQUIRED')
    p=Path(spec['scene_qualification']);native=json.loads(p.read_text());prefix=json.loads((p.parent/'STEADY_PREFIX_RESULT.private.json').read_text())
    if (digest(p)!=spec['scene_qualification_sha256']or native.get('status')!='NATIVE_STAGE_PASSED'
            or not native.get('shutdown_clean')or native.get('physics_stops')or not prefix.get('qualified')
            or native.get('scene_sha256')!=digest(spec['model_path'])
            or native.get('native_binary_sha256')!=digest(spec['native_binary'])
            or not previous or previous.get('status')!='NATIVE_STAGE_PASSED'or not previous.get('shutdown_clean')
            or previous.get('native_binary_sha256')!=digest(spec['native_binary'])):
        raise PermissionError('REAL_SAFE_NEW_SCENE_PREFIX_AND_FROZEN_C08_REQUIRED')
    return 6703


class GraspReader:
    def __init__(self,path,actual):
        import mujoco
        self.path=Path(path);self.actual=actual;self.offset=0;self.last_step=0
        self.table=mujoco.mj_name2id(actual.model,mujoco.mjtObj.mjOBJ_GEOM,'task_table')
        if self.table<0:raise ValueError('ACTUAL_TABLE_ID_REQUIRED')

    def interval(self,start,end):
        import mujoco
        from scripts.pi0294.task_state import parse_state
        from .native_replay import forbid_integration
        events=[]
        with self.path.open('rb')as f,forbid_integration(mujoco):
            f.seek(self.offset)
            while self.last_step<end:
                header=f.read(4)
                if len(header)!=4:raise ValueError('INCOMPLETE_PAUSED_NATIVE_GRASP_JOURNAL')
                size,=struct.unpack('<I',header)
                if not 40<=size<=2*1024**2:raise ValueError('NATIVE_GRASP_PACKET_SIZE')
                raw=f.read(size)
                if len(raw)!=size:raise ValueError('INCOMPLETE_NATIVE_GRASP_PACKET')
                row=parse_state(raw)
                if row['step']!=self.last_step+1 or abs(row['time_s']-row['step']*.002)>1e-8:raise ValueError('EVERY_ORIGINAL_GRASP_TICK_REQUIRED')
                self.last_step=row['step'];self.offset=f.tell()
                if row['step']<=start:continue
                if any(not np.isfinite(np.r_[c['wrench'],c['normal'],c['distance_m']]).all()for c in row['contacts']):
                    raise ValueError('FINITE_ORIGINAL_CONTACT_WRENCH_AND_NORMAL_REQUIRED')
                state=self.actual.update(row);force=[]
                for pad in self.actual.pad:
                    force.append(max([0.]+[float(c['wrench'][0])for c in row['contacts']if{c['geom0'],c['geom1']}=={pad,self.actual.cube_geom}]))
                if not np.isfinite(force).all():raise ValueError('FINITE_ACTUAL_PAD_FORCES_REQUIRED')
                support=any({c['geom0'],c['geom1']}=={self.table,self.actual.cube_geom}and c['wrench'][0]>=.05 for c in row['contacts'])
                if not state['safety_valid']:raise RuntimeError('ORIGINAL_NATIVE_CONTACT_CLASSIFICATION_FAILED')
                events.append(dict(native_step=row['step'],contact_time_s=row['contact_time_s'],left_normal_force_N=force[0],right_normal_force_N=force[1],
                                   max_pad_cube_normal_force_N=max(force),contact_positive=max(force)>=.05,
                                   native_grasp=state['native_grasp'],opposing_contact=state['opposing_contact'],relative_motion_valid=state['relative_motion_valid'],
                                   table_support=bool(support),finger_q=state['finger_q'].tolist()))
        if [e['native_step']for e in events]!=list(range(start+1,end+1)):raise ValueError('COMPLETE25_NATIVE_GRASP_TICKS_REQUIRED')
        return events


def score(events):
    if not events or [e['native_step']for e in events]!=list(range(2504,6704)):raise ValueError('COMPLETE_4200_TICK_SUPPORTED_CLOSE_EPISODE_REQUIRED')
    longest=run=0;released=0
    for e in events:
        in_hold=4903<e['native_step']<=5503 and e['stage']=='SUPPORTED_HOLD'
        run=run+1 if in_hold and e['native_grasp']and e['table_support']else 0;longest=max(longest,run)
        ready=e['native_step']>5503 and not e['contact_positive']and min(e['finger_q'])>=.040 and e['table_support']
        released=released+1 if ready else 0
    return dict(qualified_supported_grasp_hold=longest>=500,longest_continuous_supported_grasp_native_ticks=longest,
                longest_continuous_supported_grasp_s=longest*.002,terminal_released_native_ticks=released,terminal_released_s=released*.002,
                qualified_release=released>=100,qualified_supported_close_hold_release=longest>=500 and released>=100,
                full_manipulation_success=False,original_native_grasp_predicate=True)


def run(a):
    from scripts.pi0292.durable import write
    from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
    from go2arx5_vla.task_data.codec import pose10
    from .next1_episode import assess
    from .next2_contact import advance_atom,gripper_memory
    e=json.loads(Path(a.spec['episode_manifest']).read_text());a.prepare_dynamic();initial=a.measured_task_state();a.next1_samples=[initial]
    pilot2=a.spec['phase']=='collect-target-pilot2-supported'
    if pilot2:
        from .pilot2_supported import stage_for as select_stage,score as score_goal
    else:
        select_stage=lambda atom,episode:stage_for(atom)
        score_goal=lambda events,episode:score(events)
    policy_episode=e if e.get('schema')=='WM02_PILOT2_SUPPORTED_POLICY_EPISODE_V1' else None
    if policy_episode:
        from .pilot2_supported_policy_runtime import effective
        e=effective(policy_episode)
    reader=GraspReader(a.output/'state.bin.journal',a.actual_state);installed_step=2503;windows=[];samples=[];events=[];early_release=False;first_blocker=None;previous_stage=None
    def install(stage):
        opening=e['close_logical_opening']if stage in('CLOSE','SUPPORTED_HOLD')else 1.
        a.pilot1_supported_stage=stage
        action=np.repeat(pose10(initial['tcp'],float(teacher_physical_fraction(opening)))[None],40,axis=0)
        a.install_reference(action)
    for atom in range(e['task_atoms']):
        if policy_episode and atom==policy_episode['policy_decision_atom']:
            from .pilot2_supported_policy_runtime import choose
            e=choose(a,policy_episode,samples)
        step=a.observe()['step'];stage=select_stage(atom,e)
        if atom==48 and not assess(a,e)['qualified']:raise RuntimeError('UNSTABLE_SUPPORTED_CLOSE_PREFIX')
        if stage=='SUPPORTED_HOLD'and not early_release:
            current=a.measured_task_state();packet=a.observe();supported=any({c['geom0'],c['geom1']}=={reader.table,a.actual_state.cube_geom}and c['wrench'][0]>=.05 for c in packet['contacts'])
            if not current['native_grasp']or not supported:early_release=True;first_blocker='NO_CURRENT_SUPPORTED_NATIVE_GRASP_FOR_HOLD'
        if early_release and stage in('CLOSE','SUPPORTED_HOLD'):stage='EARLY_RELEASE'
        if stage!=previous_stage or atom==40 or step-installed_step>=175:
            a.command_event(dict(phase='supported-close-before-install',native_step=step,stage=stage,future_response_observed=False))
            install(stage);installed_step=step;previous_stage=stage
        w=advance_atom(a,installed_step);windows.append(w);actual_events=reader.interval(step,step+25);events.extend(actual_events)
        for event in actual_events:event['stage']=stage
        state=a.measured_task_state();a.next1_samples.append(state);memory=gripper_memory(a)
        if memory['values']['steps']!=state['native_step']or memory['values']['sequence']!=a.active_reference['sequence']:raise RuntimeError('GRIPPER_MEMORY_NOT_AT_EXACT_SUPPORTED_PAUSE')
        row=dict(native_start_step=step,native_end_step=step+25,stage=stage,TCP_m=state['tcp'][:3,3].tolist(),object_pose=state['object_pose'].tolist(),qpos=state['qpos'].tolist(),qvel=state['qvel'].tolist(),ctrl=state['ctrl'].tolist(),finger_q=state['finger_q'].tolist(),
                 native_grasp=state['native_grasp'],contact_any=any(v['contact_positive']for v in actual_events),native_contact_events=actual_events,actual_gripper_internal_memory=memory,
                 executed_reference_index=(step-installed_step)//25,executed_reference_EEF10=a.active_reference['action_EEF10'][(step-installed_step)//25],reference_sha256=w['reference_sha256'])
        samples.append(row);a.command_event(dict(phase='supported-close-actual-atom',**row))
    result=score_goal(events,e);result.update(schema='WM02_PILOT2_SUPPORTED_GOAL_SCORE_V1'if pilot2 else'WM02_PILOT1_SUPPORTED_CLOSE_SCORE_V1',source_sha=a.spec['source_sha'],same_native_process=True,physics_resets=0,state_teleports=0,actual_task_atoms=len(samples),
                                     first_blocker=first_blocker,contact_positive_native_ticks=sum(v['contact_positive']for v in events),bilateral_native_grasp_ticks=sum(v['native_grasp']for v in events),samples=samples)
    if policy_episode:
        result.update(original_episode_schema=policy_episode['schema'],effective_episode=e,
            policy_decision_sha256=digest(a.output/'SUPPORTED_POLICY_DECISION.private.json'))
    write(a.output/'SUPPORTED_CLOSE_RESULT.private.json',result);a.capture_loaded_identity();cancel=a.cancel_reference();a.seal_trace(a.spec['max_ticks'])
    value=dict(schema='WM02_TARGET_DYNAMIC_DIAGNOSTIC_V1',phase=a.spec['phase'],actual_collection_start_step=2503,task_native_ticks=25*e['task_atoms'],complete_task_atoms=e['task_atoms'],windows=windows,operator_cancel=cancel,recorded_gripper_memory=True,full_task_success=False)
    write(a.output/'DYNAMIC_DIAGNOSTIC.json',value);a.reference_qualification=value
    if not all(w['solver_policies_with_exact_installed_target']for w in windows):raise RuntimeError('SOLVER_TARGET_NOT_CONFIRMED_FOR_EVERY_SUPPORTED_ATOM')
    return value
