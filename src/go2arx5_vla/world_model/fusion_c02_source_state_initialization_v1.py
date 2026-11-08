"""Distinct C02 fixed-goal Teacher initialization using the original native loop.

A real robot pose must enter the frozen source-state range. No source/target
observation is replaced, no learned model is evaluated, and no old task passes.
"""
from collections import deque
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .storage import digest,write_json

PHASE='collect-target-fusion-c02-source-state-initialization-v1'
KIND='FUSION_C02_SOURCE_STATE_INITIALIZATION_V1'
SCENE='f27d7026ef8f182a169d7c5b4a7ed066de414196ce59d0dd97a605819050fdbf'
STATIC='817e44d23f34dfa7c49591dfc2fbb7007bb1443d4e70ec527c423eab942eb033'
SOURCE='71558ff5bcbc03e83eef9fbe79c868e75a2fece5c712f708682c8f23a090a594'
FRAME='fc347a59fe6b82f758d198f2db93955ba2697a55e8ec24e490faeda53c2761b1'
STATS='9160be2f9acc2bdb2a2f120c439b4a3bfa537dda1c7c90e6a9524eb7ef16bdbf'
INIT,DECISION,E=2503,3703,29503
MAX_MOVEMENT_ATOMS=1024
SOURCE_MEMBERS=('src/go2arx5_vla/world_model/fusion_c02_source_state_initialization_v1.py',
 'src/go2arx5_vla/world_model/target_sequential.py','src/go2arx5_vla/world_model/auto1_native.py',
 'src/go2arx5_vla/world_model/fusion_trajectory_qualification_native.py',
 'src/go2arx5_vla/world_model/fusion_trajectory_adapter.py',
 'src/go2arx5_vla/world_model/fusion_task_phase_initialization.py','scripts/pi0294/task_state.py')


def need(ok,reason):
    if not ok:raise ValueError('C02_SOURCE_INITIALIZATION_'+reason)


def episode(run):
    return dict(schema=KIND+'_EPISODE',phase=PHASE,run=str(run),scene_id='FUSION_SOURCE_STATE_C02_V1',
        parent='FUSION_SOURCE_STATE_C02_PRIVILEGED_STATIC_DESIGN',split='DIAGNOSTIC',
        purpose='ACTUAL_SOURCE_STATE_RANGE_INITIALIZATION_NOT_GR00T_OR_WM_TRANSFER',
        planned_end_atom=MAX_MOVEMENT_ATOMS,tail_hold_atoms=8,prehold_atoms=48,
        target_source_xyz=[.46,-.14,.58],orientation='ORIGINAL71558_SOURCE_ROTATION',
        geometry_scale=1.,ramp_atoms=1024,reference_timing='FIXED_51_2S_GLOBAL_QUINTIC_NOT_GR00T_RETIME',minimum_course_clearance_m=.07,
        query_pins=dict(source_observation_sha256=SOURCE,scene_sha256=SCENE,frame_map_sha256=FRAME),
        old_task_scores_unchanged=True,GR00T_queries=0,WM_updates=0,Actor_updates=0,
        native_dynamic_qualification=False)


def read(path,pin):
    p=Path(path);need(p.is_absolute()and p.resolve()==p and not p.is_symlink()and digest(p)==pin,'PINNED_CANONICAL_INPUT')
    return json.loads(p.read_bytes())


def config(spec):
    return configured_inputs(spec, PHASE, KIND, E, episode, SOURCE_MEMBERS, 900)


def configured_inputs(spec, phase, kind, end, episode_factory, members, wall_bound):
    from scripts.pi0292.relay_protocol import unseal
    from .fusion_observation import transform_hash
    need(spec.get('phase')==phase and spec.get('kind')==kind and spec.get('profile')=='wm02-auto1-target'
         and spec.get('scope')=='WM02_PILOT1_SIM_ONLY'and spec.get('max_ticks')==end
         and spec.get('physics_dt_s')==.002 and spec.get('WBC_changes')is False
         and type(spec.get('wall_limit_s'))in(int,float)and 0<spec['wall_limit_s']<=wall_bound
         and all(spec.get(k)is None for k in('paused_query_gpu_spec','program_response_forecast','program_reference_selection','program_reference_intervention','program_response_counterfactual')),
         'EXACT_FINITE_C02_TEACHER_SCOPE_NO_LEARNED_SELECTOR_OR_QUERY')
    doc=read(spec['episode_manifest'],spec['episode_manifest_sha256']);need(doc==episode_factory(spec['registered_root']),'EXACT_PREDECLARED_GOAL_AND_ENDPOINT')
    need(digest(spec['model_path'])==SCENE and spec['course_static_result_sha256']==STATIC,'COMPILED_C02_SCENE')
    proof=read(spec['course_static_result'],STATIC)
    need(proof['status']=='COURSE_STATIC_PASSED_NATIVE_STILL_REQUIRED'and proof['scene_sha256']==SCENE
         and proof['robot_contract']['controller_or_safety_changes']is False and proof['new_physics_integrations']==0,'STATIC_PROOF_IS_NOT_DYNAMIC_QUALIFICATION')
    source=unseal(read(spec['task_phase_source_observation'],SOURCE))['payload']
    frame=np.asarray(spec['task_phase_source_from_target'],float)
    need(transform_hash(frame)==spec['task_phase_frame_map_sha256']==FRAME,'UNCHANGED_VERIFIED_FRAME')
    statistics=read(spec['course_state_statistics'],STATS)['new_embodiment']['state']
    files=spec['source_files'];root=Path(spec['source_root'])
    need(all(name in files and digest(root/name)==files[name]for name in members),'SEALED_NEW_AND_ORIGINAL_EXECUTION_CODE')
    return doc,dict(raw=None,source_payload=source,frame=frame,statistics=statistics,
        provenance=dict(role='PRIVILEGED_FIXED_SOURCE_STATE_GOAL_NOT_GR00T',source_observation_sha256=SOURCE,
            frame_map_sha256=FRAME,source_statistics_sha256=STATS,future_achieved_state_labels=False))


def episode_limit(spec,previous):
    config(spec)
    path=Path(spec['previous_stage']);actual=read(path,spec['previous_stage_sha256'])
    prior=read(path.parent.parent/'EXECUTION_SPEC.json',json.loads((path.parent.parent/'control/LEASE.json').read_bytes())['spec_sha256'])
    lease=json.loads((path.parent.parent/'control/LEASE.json').read_bytes());worker=json.loads((path.parent.parent/'WORKER_EXIT.private.json').read_bytes())
    need(actual==previous and lease['status']=='CLOSED'and worker['exit_code']==lease['worker_exit_code']==0
         and actual['status']=='NATIVE_STAGE_PASSED'and actual['phase']==prior['phase']=='g2-reference-qualification'
         and actual['scene_sha256']==SCENE and actual['actual_target_integrations']==2503 and actual['shutdown_clean']is True
         and actual['physics_stops']==[] and actual['original_joint_or_warning_samples']==0
         and actual['trace_audit']['status']=='RECORDED_TRACE_CHECKS_PASS'
         and actual['reference_qualification']['status']=='ORIGINAL_FEEDBACK_INSTALL_CANCEL_HOLD_AT_PAUSE_PASSED',
         'ACTUAL_NEW_C02_REFERENCE_QUALIFICATION_CLOSED_BEFORE_NEW_MOTION')
    return E


@dataclass(frozen=True,init=False,eq=False)
class FixedGoalProgram:
    """Original global quintic geometry; this is a Teacher, not a VLA retimer."""
    def __init__(self,initial,decision,goal,base):
        from .fusion_task_phase_initialization import _interpolate,_immutable,OPEN
        from ..task_data.codec import pose10
        from ..trajectory.so3 import geodesic_angle
        base=np.asarray(base,float)
        need(base.shape==(7,)and np.isfinite(base).all()and abs(np.linalg.norm(base[3:])-1)<=1e-6,'MEASURED_FINITE_UNIT_BASE7')
        n=MAX_MOVEMENT_ATOMS;u=np.arange(n+1,dtype=float)/n;w=10*u**3-15*u**4+6*u**5
        rates={}
        for name,first in [('reference',initial),('desired',decision)]:
            distance=float(np.linalg.norm(goal[:3,3]-first[:3,3]));angle=float(geodesic_angle(first[:3,:3],goal[:3,:3]));duration=n*.05
            r=dict(peak_translation_speed_m_s=1.875*distance/duration,peak_translation_acceleration_m_s2=(10*np.sqrt(3)/3)*distance/duration**2,peak_angular_speed_rad_s=1.875*angle/duration,peak_angular_acceleration_rad_s2=(10*np.sqrt(3)/3)*angle/duration**2)
            need(r['peak_translation_speed_m_s']<=.05 and r['peak_translation_acceleration_m_s2']<=.25 and r['peak_angular_speed_rad_s']<=.25 and r['peak_angular_acceleration_rad_s2']<=1.,'ORIGINAL_ANALYTICAL_RATE_BOUNDS_NO_RETIME_OR_CLIP');rates[name]=r
        desired=np.stack([pose10(p,OPEN)for p in _interpolate(decision,goal,w)])
        reference=np.stack([pose10(p,OPEN)for p in _interpolate(initial,goal,w)])
        for name,value in dict(desired_path=_immutable(desired),reference_path=_immutable(reference),base_pose_hold=_immutable(np.asarray(base,float)),initial_reference_TCP_target=_immutable(initial),decision_TCP_target=_immutable(decision),knot_indices=tuple(int(i)for i in np.rint(np.linspace(0,n,40))),end_atom=n).items():object.__setattr__(self,name,value)
        object.__setattr__(self,'_evidence',dict(schema=KIND+'_FIXED_CLOCK_TEACHER',generated_calibration_waypoints=40,raw_VLA_knots=0,rest_to_rest_time_law='10u3-15u4+6u5',ramp_atoms=n,atom_dt_s=.05,native_dt_s=.002,clock=dict(atom_dt_s=.05,native_dt_s=.002,native_ticks_per_atom=25,end_atom=n,knot_indices=list(self.knot_indices)),reference_path_sha256=hashlib.sha256(reference.astype('<f8').tobytes()).hexdigest(),desired_path_sha256=hashlib.sha256(desired.astype('<f8').tobytes()).hexdigest(),analytical_rates=rates,native_execution_authorized=False))
    def preview(self,elapsed=0):
        need(type(elapsed)is int and elapsed>=0,'INTEGER_PREVIEW_CLOCK');return self.reference_path[np.minimum(elapsed+np.arange(40),self.end_atom)].copy()
    def desired_at(self,elapsed):
        need(type(elapsed)is int and elapsed>=0,'INTEGER_DESIRED_CLOCK');return self.desired_path[min(elapsed,self.end_atom)].copy()
    def evidence(self):return copy.deepcopy(self._evidence)


def build(inputs,doc,initial_tcp,decision_tcp,base,cube):
    from ..trajectory.so3 import rot6d_rows_to_matrix
    need(doc==episode(doc['run']),'EXACT_EPISODE')
    goal=np.eye(4);goal[:3,:3]=rot6d_rows_to_matrix(np.asarray(inputs['source_payload']['state_eef10'])[3:9]);goal[:3,3]=doc['target_source_xyz'];goal=np.linalg.inv(inputs['frame'])@goal
    program=FixedGoalProgram(initial_tcp,decision_tcp,goal,base)
    clearance=min(np.linalg.norm(program.desired_path[:,:3]-cube[:3,3],axis=1).min(),np.linalg.norm(program.reference_path[:,:3]-cube[:3,3],axis=1).min())
    need(float(clearance)>=.07,'WHOLE_NEW_COURSE_CLEARANCE_BEFORE_INSTALL')
    inputs['raw']=program.desired_path[list(program.knot_indices)].copy()
    return program,dict(role='EXPLICIT_TEACHER_INITIALIZATION_WORKING_REFERENCE_NOT_GR00T_PHYSICAL_ACTION',raw_H40=inputs['raw'],
        goal_TCP_target=goal,source_goal_xyz=doc['target_source_xyz'],minimum_full_path_clearance_m=float(clearance),
        original_native_certificates_still_required=True,native_execution_authorized=False)


def validate_install(adapter,action):
    from .fusion_trajectory_qualification_contract import hold_preview
    step=adapter.observe()['step'];program=getattr(adapter,'trajectory_program',None)
    need(adapter.spec['phase']==PHASE and INIT<=step<E and (step-INIT)%25==0,'ORIGINAL_FINITE_ATOMIC_CLOCK')
    if step>=DECISION:need(type(program)is FixedGoalProgram and program.end_atom==1024 and np.array_equal(program.base_pose_hold,adapter.trajectory_initial_base),'EXACT_FIXED_GOAL_PROGRAM_AND_BASE')
    expected=hold_preview(adapter.trajectory_initial_tcp)if step<DECISION else program.preview((step-DECISION)//25)
    need(np.asarray(action).shape==(40,10)and np.array_equal(action,expected),'FULL40_PERSISTENT_PREVIEW_NO_SUBSTITUTION')
    return dict(native_step=step,points_checked=40,original_certify_chunk_required=True,physics_advance_authorized=False)


def qualification(samples,inputs,*,end_step=E):
    from ..task_data.codec import pose10
    from .fusion_short_skill import score_contract
    tail=list(samples);need(len(tail)==9 and [s['native_step']for s in tail]==list(range(end_step-200,end_step+1,25)),'ACTUAL_FINAL_HOLD_CLOCK')
    last=tail[-1];mapped=inputs['frame']@np.asarray(last['tcp']);vector=pose10(mapped,float(np.asarray(last['finger_q']).mean()/.044))
    st=inputs['statistics'];low=np.r_[st['eef_9d']['min'],st['gripper_position']['min']];high=np.r_[st['eef_9d']['max'],st['gripper_position']['max']]
    outside=np.flatnonzero((vector<low)|(vector>high));points=np.asarray([s['tcp'][:3,3]for s in tail]);base=np.asarray([s['qpos'][:7]for s in tail]);bounds=score_contract()
    metrics=dict(outside_training_range_components=outside.tolist(),actual_source_state_eef10=vector.tolist(),
        actual_cube_distance_m=float(np.linalg.norm(np.asarray(last['tcp'])[:3,3]-np.asarray(last['object_pose'])[:3,3])),
        hold_TCP_diameter_m=float(np.linalg.norm(points[:,None]-points[None],axis=2).max()),hold_TCP_max_speed_m_s=float((np.linalg.norm(np.diff(points,axis=0),axis=1)/.05).max()),
        hold_BASE_diameter_m=float(np.linalg.norm(base[:,None,:3]-base[None,:,:3],axis=2).max()),
        hold_BASE_rotation_rad=float((2*np.arccos(np.clip(abs(base[:,3:]@base[0,3:]),0.,1.))).max()))
    predicates=dict(actual_source_state_inside_training_range=not len(outside),actual_70_to_90mm_course_distance=.07<=metrics['actual_cube_distance_m']<=.09,
        TCP_hold=metrics['hold_TCP_diameter_m']<=bounds['final_hold_TCP_diameter_m']and metrics['hold_TCP_max_speed_m_s']<=bounds['final_hold_TCP_max_speed_m_s'],
        BASE_hold=metrics['hold_BASE_diameter_m']<=bounds['maximum_base_displacement_m']and metrics['hold_BASE_rotation_rad']<=bounds['maximum_base_rotation_rad'])
    return dict(schema=KIND+'_ACTUAL_QUALIFICATION',qualified=all(predicates.values()),predicates=predicates,metrics=metrics,
        only_initialization_not_task_transfer=True,old_SHORT_or_T1_score_reclassified=False,source_inputs_replaced=False)


def run(adapter):
    from .fusion_trajectory_qualification_native import run_program
    from .fusion_task_phase_live_skill_native_v8 import _noncontact
    doc,inputs=config(adapter.spec);tail=deque(maxlen=9);cube=None
    def observe(sample):
        nonlocal cube
        if cube is None:cube=np.asarray(sample['object_pose']).copy()
        _noncontact(sample,cube);tail.append(sample)
    def factory(value,document,initial,decision,base):
        program,receipt=build(value,document,initial,decision,base,cube)
        write_json(adapter.output/'C02_FIXED_GOAL_REFERENCE.private.json',receipt);return program
    def finish(report):
        q=qualification(tail,inputs);write_json(adapter.output/'C02_MEASURED_QUALIFICATION.private.json',q)
        report.update(C02_initialization_qualification=q,GR00T_raw_H40_consumed=False,Teacher_initialization_all40_consumed=report['all_40_knots_consumed'],
            VLA_queries=0,actual_VLA_inferences=0,actual_VLA_commands=0,source_task_phase_equivalence=False,
            original_T1_and_SHORT_scores_unchanged=True,expert_training_eligible=False)
    report=run_program(adapter,doc,inputs,program_factory=factory,sample_observer=observe,report_finalizer=finish,
        report_context=dict(experiment_role='C02_FIXED_SOURCE_STATE_TEACHER_INITIALIZATION',desired_outcome_provenance='PRIVILEGED_INITIALIZATION_NOT_GR00T'))
    need(report['C02_initialization_qualification']['qualified'],'MEASURED_INITIALIZATION_GATE_FAILED_KEEP_ALL_EVIDENCE')
    return report
