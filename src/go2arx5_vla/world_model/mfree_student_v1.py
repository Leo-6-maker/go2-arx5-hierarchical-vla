"""New60s raw student identity, shared accepted0.5s boundary, no teacher."""
import numpy as np
from .mfree_expert_capture_v1 import initialize
from .mfree_capture_v1 import LiveCamera
from .mobile_l1_episode_v1 import predict
from .mobile_boundary_clock_v1 import boundary
from .mobile_wbc_reference_v1 import START
from .mobile_expert_episode_v2 import funnel

VERSION='MFREE_STUDENT_60S_V1'


def episode(adapter):
    from scripts.wm02.mobile_boundary_runner_v1 import install, consume
    from .sla_dense_policy_v5 import compile_proposal
    condition=adapter.spec['mobile_condition']
    if (adapter.spec['mobile_experiment_version']!=VERSION or condition['mode']!='M_FREE'
            or condition['family']not in ('F1','F2') or condition['method']=='EXPERT'
            or adapter.spec['policy_queries']!=120 or adapter.spec.get('student_goal_offset_m')!=0.):
        raise PermissionError('NEW_60S_RAW_MFREE_STUDENT_ONLY_IDENTITY')
    initial,initial_mount,reference,_,frame=initialize(adapter)
    samples=[];queries=0;rejection=None;grasp_since=None;success=False;ever=False;longest=0.;at30=None
    session=LiveCamera(adapter.spec['model_path'])
    try:
        for query in range(120):
            sample=adapter.measured_task_state()
            response,state=predict(adapter,sample,session,frame,query);queries+=1
            try:
                program,evidence=compile_proposal(response['action'],response['action_sha256'],
                    source_from_target=adapter.spec['policy_from_world'],query_state10=state,
                    actual_TCP=sample['tcp'],consumed_reference_TCP=reference,base_hold=adapter.base_pose_hold)
            except ValueError as error:
                rejection=dict(query=query,reason=str(error),full_H40_checked=True,installed=False);break
            adapter.command_event(dict(phase='mobile-student-query',query=query,**evidence))
            adapter.mobile_program=program;adapter.mobile_program_start_step=sample['native_step']
            for atom in range(10):
                action=program.preview(atom)
                if atom in (0,7):install(adapter,action,None)
                sample,_=consume(adapter,action,sequence=128+query*10+atom);samples.append(sample)
                reference,witness=boundary(adapter);adapter.reference_history[-1]['pose']=reference.copy()
                ever|=bool(sample['native_grasp']);lift=float(sample['object_pose'][2,3]-initial['object_pose'][2,3])
                funnel(adapter,sample,'STUDENT_NO_EXPERT_STAGE',query,None)
                if sample['native_grasp']and lift>=.020:
                    if grasp_since is None:grasp_since=sample['time_s']
                    longest=max(longest,sample['time_s']-grasp_since)
                    if longest>=1.:success=True
                else:grasp_since=None
                if sample['native_step']==START+15000:
                    at30=dict(task_success=success,ever_grasp=ever,lift_m=lift,longest_hold_s=longest,native_step=sample['native_step'])
            if success:break
    finally:session.close()
    reason='TASK_SUCCESS'if success else'ORIGINAL_RAW_FULL_H40_POLICY_REJECTED'if rejection else'60S_POLICY_BUDGET_EXHAUSTED'
    if at30 is None:at30=dict(terminated_before_30s=True,elapsed_task_s=(adapter.observe()['step']-START)*.002,
        task_success=success,ever_grasp=ever,terminal_reason=reason)
    funnel(adapter,sample,'STUDENT_NO_EXPERT_STAGE',query,None,terminal_reason=reason)
    cancel=adapter.cancel_reference()
    return dict(schema='MFREE_RAW_STUDENT_60S_ACTUAL_V1',condition=condition,task_success=success,
        valid_policy_rejection=rejection,actual_SLA_queries=queries,expert_queries=0,consumed_task_atoms=len(samples),
        terminal_native_step=adapter.observe()['step'],terminal_reason=reason,ever_grasp=ever,
        grasp_atoms=sum(s['native_grasp']for s in samples),longest_grasp_lift_hold_s=longest,
        maximum_lift_m=max((float(s['object_pose'][2,3]-initial['object_pose'][2,3])for s in samples),default=0.),
        at_30_seconds=at30,task_budget_s=60.,cancel=cancel,initial_mount=initial_mount.tolist(),
        initial_TCP=initial['tcp'].tolist(),fixed_frame=frame.tolist(),expert_takeover=False,
        privileged_stage_used=False,student_goal_offset_m=0.,BASE_reference_present=False,
        task_predicate='ACTUAL_NATIVE_GRASP_AND_LIFT_20MM_CONTINUOUS_ONE_SECOND')
