"""At most four separately registered continuous-expert F1/F2 conditions."""
import json
import numpy as np

from .mobile_continuous_teacher_v2 import ContinuousTeacher, VERSION
from .mobile_wbc_reference_v1 import START, m0_reference


def funnel(adapter, sample, stage, query, goal, *, terminal_reason=None):
    value=dict(schema='MOBILE_TASK_FUNNEL_OBSERVED_V2',native_step=sample['native_step'],
        query=query,stage='PLACE'if stage in('LOWER','OPEN','WITHDRAW')else stage,
        expert_substage=stage,actual_TCP=sample['tcp'].tolist(),
        actual_object=sample['object_pose'].tolist(),actual_gripper_fraction=sample['gripper_fraction'],
        left_pad_contact=bool(sample['left_object_contact']),right_pad_contact=bool(sample['right_object_contact']),
        native_grasp=bool(sample['native_grasp']),terminal_reason=terminal_reason,
        privileged_expert_goal=goal,student_input=False)
    if goal is not None:value['actual_goal_error_m']=float(np.linalg.norm(sample['tcp'][:3,3]-goal))
    with(adapter.output/'TASK_FUNNEL.private.jsonl').open('a')as f:f.write(json.dumps(value)+'\n')


def episode(adapter):
    from scripts.wm02.mobile_boundary_runner_v1 import arm,install,consume,measured_mount
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from .fusion_trajectory_adapter import WORKING_OPEN_M,MECHANICAL_TRAVEL_M
    condition=adapter.spec['mobile_condition']
    if (adapter.spec.get('mobile_experiment_version')!=VERSION or condition['method']!='EXPERT'
            or condition['family']not in('F1','F2')or condition['mode']not in('M_FREE','M_ASSIST')):
        raise PermissionError('ONLY_FOUR_NEW_EXPERT_TIMING_CONDITIONS')
    adapter.prepare_dynamic();mount=arm(adapter);sample=adapter.measured_task_state()
    hold=np.repeat(pose10(sample['tcp'],WORKING_OPEN_M/MECHANICAL_TRAVEL_M)[None],40,axis=0)
    for i in range(128):
        if i%7==0:install(adapter,hold,np.repeat(mount[None],41,axis=0))
        sample,_=consume(adapter,hold,sequence=i)
    if adapter.observe()['step']!=START:raise RuntimeError('V2_EXACT_UNCHANGED_POST_HOLD_CLOCK')
    initial=sample;initial_mount=measured_mount(adapter);reference=adapter.reference_history[-1]['pose']
    adapter.mobile_mode=condition['mode'];adapter.mobile_segment='MOBILE_TASK'
    teacher=ContinuousTeacher(initial);samples=[];grasp_since=None;success=False;ever_grasp=False;rejection=None;queries=0
    for query in range(adapter.spec['policy_queries']):
        sample=adapter.measured_task_state()
        try:program,evidence=teacher.decide(sample,reference,adapter.base_pose_hold)
        except ValueError as error:
            if 'ORIGINAL_FULL_PREVIEW_RATE_AND_ACCELERATION_LIMITS'not in str(error):raise
            rejection=dict(query=query,reason=str(error),full_H40_checked=True,installed=False);break
        queries+=1
        adapter.mobile_contact_stage={'CLOSE':'close','LIFT':'lift','HOLD':'hold'}.get(teacher.stage,'approach')
        adapter.command_event(dict(phase='mobile-continuous-expert-query',query=query,native_step=sample['native_step'],**evidence))
        adapter.mobile_program=program;adapter.mobile_program_start_step=adapter.observe()['step']
        for atom in range(10):
            action=program.preview(atom)
            # The original F1/F2 BASE protocol is deliberately retained here.
            # AFTER_GRASP is a distinct, unlaunched future comparison.
            _,base=m0_reference(initial['tcp'],initial_mount,condition['axis'],'WORLD_HOLD',query*10+atom)
            if atom in(0,7):install(adapter,action,base if adapter.mobile_mode=='M_ASSIST'else None)
            sample,_=consume(adapter,action,sequence=128+query*10+atom);samples.append(sample)
            funnel(adapter,sample,teacher.stage,query,evidence['privileged_goal'])
            ever_grasp|=bool(sample['native_grasp'])
            lift=float(sample['object_pose'][2,3]-initial['object_pose'][2,3])
            if sample['native_grasp']and lift>=.020:
                if grasp_since is None:grasp_since=sample['time_s']
                if sample['time_s']-grasp_since>=1.:success=True
            else:grasp_since=None
        reference=teacher.commit(program,adapter.observe()['step'])
        if success:break
    reason='TASK_SUCCESS'if success else 'ORIGINAL_H40_EXPERT_PROPOSAL_REJECTED'if rejection else 'TASK_BUDGET_EXHAUSTED_IN_'+teacher.stage
    funnel(adapter,sample,teacher.stage,query,None,terminal_reason=reason)
    cancel=adapter.cancel_reference()
    return dict(schema='MOBILE_CONTINUOUS_EXPERT_ACTUAL_ENDPOINT_V2',condition=condition,teacher_version=VERSION,
        task_success=success,valid_policy_rejection=None,valid_expert_proposal_rejection=rejection,
        expert_queries=queries,actual_SLA_queries=0,consumed_task_atoms=len(samples),
        initialization_qualified=True,student_segment_started=False,expert_segment_started=bool(samples),
        ever_grasp=ever_grasp,grasp_atoms=sum(s['native_grasp']for s in samples),
        maximum_lift_m=max((float(s['object_pose'][2,3]-initial['object_pose'][2,3])for s in samples),default=0.),
        terminal_stage=teacher.stage,terminal_reason=reason,terminal_native_step=adapter.observe()['step'],
        initial_mount=initial_mount.tolist(),initial_TCP=initial['tcp'].tolist(),cancel=cancel,
        BASE_semantics='ORIGINAL_V1_F1_F2_UNCHANGED',stage_tolerance_m=.004,
        task_predicate='ACTUAL_NATIVE_GRASP_AND_LIFT_20MM_CONTINUOUS_ONE_SECOND',expert_takeover=False)
