"""Versioned finite expert qualification, original native H40/P10 and scoring."""
import numpy as np

from .mobile_grasp_budget_v3 import BudgetTeacher,StageClock,VERSION,MAX_QUERIES,grasp_raise
from .mobile_expert_episode_v2 import funnel
from .mobile_wbc_reference_v1 import START,m0_reference


def episode(adapter):
    from scripts.wm02.mobile_boundary_runner_v1 import arm,install,consume,measured_mount
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from .fusion_trajectory_adapter import WORKING_OPEN_M,MECHANICAL_TRAVEL_M
    condition=adapter.spec['mobile_condition']
    if(adapter.spec.get('mobile_experiment_version')not in(VERSION,'MFREE_SHIFT7_SIM_ONLY_V51') or condition['method']!='EXPERT'
            or condition['family']not in('F1','F2')or condition['mode']not in('M_FREE','M_ASSIST')
            or adapter.spec['policy_queries']!=MAX_QUERIES
            or adapter.spec['grasp_goal_raise_m']!=grasp_raise(condition)):
        raise PermissionError('EXPLICIT_NEW_V3_BUDGET_AND_CLEARANCE_CONDITION')
    adapter.prepare_dynamic();mount=arm(adapter);sample=adapter.measured_task_state()
    hold=np.repeat(pose10(sample['tcp'],WORKING_OPEN_M/MECHANICAL_TRAVEL_M)[None],40,axis=0)
    for i in range(128):
        if i%7==0:install(adapter,hold,np.repeat(mount[None],41,axis=0))
        sample,_=consume(adapter,hold,sequence=i)
    if adapter.observe()['step']!=START:raise RuntimeError('V3_UNCHANGED_POST_HOLD_CLOCK')
    initial=sample;initial_mount=measured_mount(adapter);reference=adapter.reference_history[-1]['pose']
    adapter.mobile_mode=condition['mode'];adapter.mobile_segment='MOBILE_TASK'
    teacher=BudgetTeacher(initial,condition);clock=StageClock(START);samples=[]
    grasp_since=None;success=False;ever_grasp=False;rejection=None;queries=0;reason=None
    for query in range(MAX_QUERIES):
        sample=adapter.measured_task_state()
        # Actual predicates may advance a phase; a time limit never does.
        teacher.goal(sample);budget=clock.observe(teacher.stage,sample['native_step'])
        adapter.command_event(dict(phase='mobile-v3-stage-budget',query=query,**budget))
        if budget['exhausted']:
            reason='STAGE_BUDGET_EXHAUSTED_IN_'+teacher.stage;break
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
            # BASE semantics stay exactly V1/V2, independent of this repair.
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
    reason=('TASK_SUCCESS'if success else 'ORIGINAL_H40_EXPERT_PROPOSAL_REJECTED'if rejection
            else reason or 'GLOBAL_TASK_BUDGET_EXHAUSTED_IN_'+teacher.stage)
    funnel(adapter,sample,teacher.stage,query,None,terminal_reason=reason)
    cancel=adapter.cancel_reference()
    return dict(schema='MOBILE_GRASP_BUDGET_V3_ACTUAL_ENDPOINT',condition=condition,teacher_version=VERSION,
        task_success=success,complete_expert_qualification=success,expert_queries=queries,
        actual_SLA_queries=0,consumed_task_atoms=len(samples),valid_policy_rejection=None,
        valid_expert_proposal_rejection=rejection,initialization_qualified=True,student_segment_started=False,
        expert_segment_started=bool(samples),ever_grasp=ever_grasp,grasp_atoms=sum(s['native_grasp']for s in samples),
        maximum_lift_m=max((float(s['object_pose'][2,3]-initial['object_pose'][2,3])for s in samples),default=0.),
        terminal_stage=teacher.stage,terminal_reason=reason,terminal_native_step=adapter.observe()['step'],
        phase_transitions=clock.transitions,stage_budget_s=adapter.spec['stage_budget_s'],
        global_task_budget_s=60.,grasp_goal_raise_m=teacher.raise_m,
        initial_mount=initial_mount.tolist(),initial_TCP=initial['tcp'].tolist(),cancel=cancel,
        BASE_semantics='ORIGINAL_V1_F1_F2_UNCHANGED',stage_tolerance_m=.004,
        task_predicate='ACTUAL_NATIVE_GRASP_AND_LIFT_20MM_CONTINUOUS_ONE_SECOND',
        expert_takeover=False)
