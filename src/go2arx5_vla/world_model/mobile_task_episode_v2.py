"""Separate BASE/observation intervention; original V1 teacher is retained.

This version is not dispatched by the four continuous-expert timing trials.
No F4/F5/F6 trial is automatically authorized by its implementation.
"""
import numpy as np
from .mobile_l1_episode_v1 import predict
from .mobile_wbc_reference_v1 import START,m0_reference
from .task_learning_grasp_v1 import Teacher,FiftyMsProgram
from .mobile_observation_v2 import ObservedGrasp,AfterGraspRoute,VERSION,MODE
from .mobile_expert_episode_v2 import funnel
from .storage import write_json
from ..trajectory.so3 import matrix_to_quat,quat_slerp,quat_to_matrix,rotation_matrix_to_rot6d_rows


class TransportTeacher:
    def __init__(self,initial,family,*,held=False):
        self.initial=initial;self.family=family;self.pick=Teacher(initial)
        self.stage='CARRY'if held else 'PICK';self.grip=0. if held else 1.
        self.carry=None;self.offset=None;self.release_tcp=None

    def decide(self,sample,reference,base,carry_allowed):
        if self.stage=='PICK':
            if self.family=='F2'or not carry_allowed:return self.pick.decide(sample,reference,base)
            self.stage='CARRY';self.grip=0.
        actual=sample['tcp'];destination=self.initial['object_pose'][:3,3]+[.1,0,0]
        if self.carry is None:
            self.carry=actual[:3,3]+[.1,0,0];self.offset=actual[:3,3]-sample['object_pose'][:3,3]
        goal=self.carry.copy();opening=0.
        if self.family=='F6':
            if self.stage=='CARRY'and sample['object_pose'][0,3]-self.initial['object_pose'][0,3]>=.08:self.stage='LOWER'
            if self.stage=='LOWER':
                goal=destination+self.offset
                if (np.linalg.norm(sample['object_pose'][:2,3]-destination[:2])<=.025
                        and abs(sample['object_pose'][2,3]-destination[2])<=.003):self.stage='OPEN'
            if self.stage=='OPEN':
                goal=destination+self.offset;opening=1.
                if (sample['gripper_fraction']>=.9 and not sample['native_grasp']
                        and not sample.get('robot_object_contact',False)):
                    self.release_tcp=actual[:3,3].copy();self.stage='WITHDRAW'
            if self.stage=='WITHDRAW':goal=self.release_tcp+[0,0,.05];opening=1.
        delta=goal-actual[:3,3];distance=float(np.linalg.norm(delta));step=delta*min(1.,.015/distance)if distance else delta
        u=np.linspace(0,1,40);w=u**3*(10+u*(-15+6*u));raw=np.empty((40,10))
        raw[:,:3]=actual[:3,3]+w[:,None]*step
        rotation=quat_to_matrix(quat_slerp(matrix_to_quat(actual[:3,:3]),matrix_to_quat(self.initial['tcp'][:3,:3]),w[:,None]))
        raw[:,3:9]=rotation_matrix_to_rot6d_rows(rotation);raw[:,9]=self.grip+w*(opening-self.grip);self.grip=opening
        return FiftyMsProgram(raw,actual,reference,base),dict(stage=self.stage,privileged_goal=goal.tolist(),
            submitted_policy_H40=raw.tolist(),no_future_achieved_state_labels=True,student_metadata=False)


def episode(adapter):
    from scripts.wm02.mobile_boundary_runner_v1 import arm,install,consume,measured_mount
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from .fusion_trajectory_adapter import WORKING_OPEN_M,MECHANICAL_TRAVEL_M
    from .native_body_camera_capture_v1 import BodyBaseCameraSession
    from .sla_dense_policy_v5 import compile_proposal
    condition=adapter.spec['mobile_condition'];family=condition['family'];method=condition['method']
    if(adapter.spec.get('mobile_experiment_version')!=VERSION or family not in('F4','F5','F6')
            or condition['mode']not in('M_FREE',MODE)):
        raise PermissionError('SEPARATE_OBSERVED_AFTER_GRASP_V2_CONDITION_REQUIRED')
    adapter.prepare_dynamic();mount=arm(adapter);sample=adapter.measured_task_state()
    hold=np.repeat(pose10(sample['tcp'],WORKING_OPEN_M/MECHANICAL_TRAVEL_M)[None],40,axis=0)
    for i in range(128):
        if i%7==0:install(adapter,hold,np.repeat(mount[None],41,axis=0))
        sample,_=consume(adapter,hold,sequence=i)
    if adapter.observe()['step']!=START:raise RuntimeError('MOBILE_TASK_EXACT_INITIALIZATION')
    rest=sample;reference=adapter.reference_history[-1]['pose'];sequence=128
    observed=ObservedGrasp(rest['object_pose'][2,3]);route=AfterGraspRoute(condition['axis'])
    prefix_atoms=0;initialization_failed=False
    if family=='F4':
        adapter.mobile_segment='EXPERT_PREGRASP_PREFIX';prefix=Teacher(rest);since=None
        for query in range(60):
            sample=adapter.measured_task_state();program,evidence=prefix.decide(sample,reference,adapter.base_pose_hold)
            adapter.command_event(dict(phase='F4-real-expert-prefix',query=query,**evidence))
            adapter.mobile_program=program;adapter.mobile_program_start_step=adapter.observe()['step']
            adapter.mobile_contact_stage={'CLOSE':'close','LIFT':'lift','HOLD':'hold'}.get(prefix.stage,'approach')
            for atom in range(10):
                action=program.preview(atom)
                if atom in(0,7):install(adapter,action,np.repeat(mount[None],41,axis=0))
                sample,_=consume(adapter,action,sequence=sequence);sequence+=1;prefix_atoms+=1
                observed.observe(sample)
                funnel(adapter,sample,prefix.stage,query,evidence['privileged_goal'])
                reference=adapter.reference_history[-1]['pose']
                if sample['native_grasp']and sample['object_pose'][2,3]-rest['object_pose'][2,3]>=.02:
                    if since is None:since=sample['time_s']
                else:since=None
            if since is not None and sample['time_s']-since>=1.:break
        initialization_failed=since is None or sample['time_s']-since<1.
    initial=sample;initial_mount=measured_mount(adapter);task_start=adapter.observe()['step']
    adapter.mobile_mode='M_FREE';adapter.mobile_segment='MOBILE_TASK'
    if family=='F4'and not initialization_failed:route.trigger(sample,initial_mount,observed.qualified)
    body=adapter.actual_state.model.body('x5_base_link').id;d=adapter.actual_state.data
    transform=np.eye(4);transform[:3,:3]=d.xmat[body].reshape(3,3);transform[:3,3]=d.xpos[body]
    frame=np.asarray(adapter.spec['policy_from_world'])@transform
    write_json(adapter.output/'SLA_TASK_FRAME.private.json',dict(T_policy_from_mount=frame.tolist(),native_step=task_start,
        fixed_for_episode=True,expert_prefix_atoms=prefix_atoms,student_autonomous_grasp_claimed=family!='F4'))
    teacher=TransportTeacher(initial if family=='F4'else rest,family,held=family=='F4')if method=='EXPERT'else None
    session=BodyBaseCameraSession(adapter.spec['model_path'])if teacher is None and not initialization_failed else None
    queries=0;samples=[];rejection=None;success=False;grasp_since=None;success_since=None;release_since=None;release_tcp=None
    route_start=route.start;route_trigger=route.evidence();transported=False
    destination=rest['object_pose'][:3,3]+[.1,0,0]
    try:
        for query in range(0 if initialization_failed else adapter.spec['policy_queries']):
            sample=adapter.measured_task_state();carry_allowed=route_start is not None
            if teacher is not None:
                program,evidence=teacher.decide(sample,reference,adapter.base_pose_hold,carry_allowed)
                adapter.mobile_contact_stage={'CARRY':'lift','LOWER':'release','OPEN':'release','WITHDRAW':'detach'}.get(teacher.stage,
                    {'CLOSE':'close','LIFT':'lift','HOLD':'hold'}.get(teacher.pick.stage,'approach'))
                adapter.command_event(dict(phase='mobile-task-expert',query=query,**evidence))
            else:
                response,state=predict(adapter,sample,session,frame,query);queries+=1
                try:program,evidence=compile_proposal(response['action'],response['action_sha256'],source_from_target=adapter.spec['policy_from_world'],
                    query_state10=state,actual_TCP=sample['tcp'],consumed_reference_TCP=reference,base_hold=adapter.base_pose_hold)
                except ValueError as error:rejection=dict(query=query,reason=str(error),full_H40_checked=True,installed=False);break
                adapter.command_event(dict(phase='mobile-task-student',query=query,**evidence))
            adapter.mobile_program=program;adapter.mobile_program_start_step=adapter.observe()['step']
            for atom in range(10):
                action=program.preview(atom)
                mode,base=route.preview(adapter.observe()['step'])if condition['mode']==MODE else('M_FREE',None)
                changed=mode!=adapter.mobile_mode;adapter.mobile_mode=mode
                if atom in(0,7)or changed:install(adapter,action,base)
                sample,_=consume(adapter,action,sequence=sequence);sequence+=1;samples.append(sample);reference=adapter.reference_history[-1]['pose']
                observed.observe(sample)
                funnel(adapter,sample,teacher.pick.stage if teacher is not None and teacher.stage=='PICK'else teacher.stage if teacher else 'STUDENT_UNOBSERVED_INTENT',query,evidence.get('privileged_goal'))
                lift=float(sample['object_pose'][2,3]-rest['object_pose'][2,3]);grasp=sample['native_grasp']and lift>=.02
                if grasp:
                    if grasp_since is None:grasp_since=sample['time_s']
                else:grasp_since=None
                if route.trigger(sample,measured_mount(adapter),observed.qualified):
                    route_start=route.start;route_trigger=route.evidence()
                progress=float(sample['object_pose'][0,3]-initial['object_pose'][0,3]);transported|=progress>=.08
                qualified=grasp and(family=='F2'or progress>=.08)
                if qualified:
                    if success_since is None:success_since=sample['time_s']
                    if family!='F6'and sample['time_s']-success_since>=1.:success=True
                else:success_since=None
                if family=='F6':
                    placed=(observed.ever_grasp and transported and not sample['native_grasp']and sample['gripper_fraction']>=.8
                        and not sample['robot_object_contact']and np.linalg.norm(sample['object_pose'][:2,3]-destination[:2])<=.025
                        and abs(sample['object_pose'][2,3]-destination[2])<=.005
                        and np.linalg.norm(sample['object_linear_velocity'])<=.02)
                    if placed:
                        if release_tcp is None:release_tcp=sample['tcp'][:3,3].copy()
                        if release_since is None:release_since=sample['time_s']
                        if sample['time_s']-release_since>=1. and sample['tcp'][2,3]-release_tcp[2]>=.04:success=True
                    else:release_since=None
            if success:break
    finally:
        if session is not None:session.close()
    cancel=adapter.cancel_reference()
    return dict(schema='MOBILE_OBSERVED_AFTER_GRASP_ENDPOINT_V2',condition=condition,task_success=success,
        initialization_failed=initialization_failed,expert_prefix_atoms=prefix_atoms,student_autonomous_grasp_claimed=family!='F4',
        initialization_qualified=not initialization_failed,student_segment_started=method!='EXPERT'and not initialization_failed and bool(samples),
        actual_SLA_queries=queries,consumed_task_atoms=len(samples),terminal_native_step=adapter.observe()['step'],
        valid_policy_rejection=rejection,base_route_trigger=route_trigger,task_start_native_step=task_start,
        ever_grasp=observed.ever_grasp,object_transport_80mm=transported,maximum_lift_m=max((float(s['object_pose'][2,3]-rest['object_pose'][2,3])for s in samples),default=0.),
        cancel=cancel,initial_mount=initial_mount.tolist(),fixed_frame=frame.tolist(),expert_takeover=False,
        placement_ROI_half_width_m=.025 if family=='F6'else None,placement_z_tolerance_m=.005 if family=='F6'else None,
        placement_speed_limit_m_s=.02 if family=='F6'else None,placement_stability_s=1. if family=='F6'else None)
