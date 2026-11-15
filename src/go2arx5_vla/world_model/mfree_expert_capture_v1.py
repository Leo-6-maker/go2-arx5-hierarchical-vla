"""C20 production, V3 expert with live RGB and an accepted-clock boundary."""
import numpy as np
from .mobile_grasp_budget_v3 import BudgetTeacher, StageClock, MAX_QUERIES, grasp_raise
from .mobile_expert_episode_v2 import funnel
from .mobile_wbc_reference_v1 import START
from .mobile_boundary_clock_v1 import boundary
from .mfree_capture_v1 import LiveWriter

VERSION='MFREE_C20_FINAL_PLAN_V1'


def initialize(adapter):
    from scripts.wm02.mobile_boundary_runner_v1 import arm, install, consume, measured_mount
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from .fusion_trajectory_adapter import WORKING_OPEN_M, MECHANICAL_TRAVEL_M
    adapter.prepare_dynamic(); mount=arm(adapter); sample=adapter.measured_task_state()
    hold=np.repeat(pose10(sample['tcp'],WORKING_OPEN_M/MECHANICAL_TRAVEL_M)[None],40,axis=0)
    for i in range(128):
        if i%7==0: install(adapter,hold,np.repeat(mount[None],41,axis=0))
        sample,_=consume(adapter,hold,sequence=i)
    if adapter.observe()['step']!=START: raise RuntimeError('UNCHANGED_POST_HOLD_START_CLOCK')
    reference, witness=boundary(adapter)
    initial_mount=measured_mount(adapter)
    adapter.mobile_mode='M_FREE'; adapter.mobile_segment='MOBILE_TASK'
    body=adapter.actual_state.model.body('x5_base_link').id; data=adapter.actual_state.data
    transform=np.eye(4); transform[:3,:3]=data.xmat[body].reshape(3,3); transform[:3,3]=data.xpos[body]
    frame=np.asarray(adapter.spec['policy_from_world'])@transform
    return sample, initial_mount, reference, witness, frame


def episode(adapter):
    from scripts.wm02.mobile_boundary_runner_v1 import install, consume
    condition=adapter.spec['mobile_condition']
    if (adapter.spec['mobile_experiment_version']!=VERSION or condition['method']!='EXPERT'
            or condition['mode']!='M_FREE' or condition['family'] not in ('F1','F2')
            or adapter.spec['grasp_goal_raise_m']!=grasp_raise(condition)):
        raise PermissionError('EXPLICIT_MFREE_ONLY_NEW_COLLECTION_IDENTITY')
    initial, initial_mount, reference, witness, frame=initialize(adapter)
    teacher=BudgetTeacher(initial,condition); clock=StageClock(START); writer=LiveWriter(adapter,frame)
    samples=[]; grasp_since=None; success=False; ever_grasp=False; rejection=None; reason=None; queries=0
    capture=None; end30=None
    try:
        for query in range(MAX_QUERIES):
            sample=adapter.measured_task_state(); teacher.goal(sample)
            budget=clock.observe(teacher.stage,sample['native_step'])
            adapter.command_event(dict(phase='mfree-production-stage-budget',query=query,**budget))
            if budget['exhausted']: reason='STAGE_BUDGET_EXHAUSTED_IN_'+teacher.stage; break
            try: program,evidence=teacher.decide(sample,reference,adapter.base_pose_hold)
            except ValueError as error:
                if 'ORIGINAL_FULL_PREVIEW_RATE_AND_ACCELERATION_LIMITS' not in str(error): raise
                rejection=dict(query=query,reason=str(error),full_H40_checked=True,installed=False); break
            row=writer.capture(sample,program,evidence,witness,teacher); queries+=1
            adapter.mobile_contact_stage={'CLOSE':'close','LIFT':'lift','HOLD':'hold'}.get(teacher.stage,'approach')
            adapter.command_event(dict(phase='mobile-continuous-expert-query',query=query,native_step=sample['native_step'],**evidence))
            adapter.mobile_program=program; adapter.mobile_program_start_step=sample['native_step']
            for atom in range(10):
                action=program.preview(atom)
                if atom in (0,7): install(adapter,action,None)
                interval=dict(atom=atom, start_native_step=adapter.observe()['step'],
                    installed_reference_sha256=adapter.active_reference['wire_sha256'], confirmed_ticks=0)
                row['intervals'].append(interval)
                sample,_=consume(adapter,action,sequence=128+query*10+atom); samples.append(sample)
                reference,witness=boundary(adapter)
                adapter.reference_history[-1]['pose']=reference.copy()
                interval.update(end_native_step=sample['native_step'],confirmed_ticks=25,boundary=witness)
                row['consumed_mask'][atom]=True
                funnel(adapter,sample,teacher.stage,query,evidence['privileged_goal'])
                ever_grasp|=bool(sample['native_grasp'])
                lift=float(sample['object_pose'][2,3]-initial['object_pose'][2,3])
                if sample['native_grasp'] and lift>=.020:
                    if grasp_since is None: grasp_since=sample['time_s']
                    if sample['time_s']-grasp_since>=1.: success=True
                else: grasp_since=None
                if sample['native_step']==START+15000:
                    end30=dict(task_success=success,ever_grasp=ever_grasp,lift_m=lift,native_step=sample['native_step'])
            teacher.commit(program,adapter.observe()['step'])
            row['next_query_boundary']=witness
            if success: break
    finally:
        capture=writer.close()
    reason=('TASK_SUCCESS' if success else 'ORIGINAL_H40_EXPERT_PROPOSAL_REJECTED' if rejection else reason or 'GLOBAL_TASK_BUDGET_EXHAUSTED_IN_'+teacher.stage)
    funnel(adapter,sample,teacher.stage,query,None,terminal_reason=reason)
    cancel=adapter.cancel_reference()
    return dict(schema='MFREE_C20_EXPERT_CAPTURE_ACTUAL_V1',condition=condition,task_success=success,
        complete_expert_qualification=success,expert_queries=queries,actual_SLA_queries=0,
        consumed_task_atoms=len(samples),valid_policy_rejection=None,valid_expert_proposal_rejection=rejection,
        initialization_qualified=True,student_segment_started=False,expert_segment_started=bool(samples),
        ever_grasp=ever_grasp,grasp_atoms=sum(s['native_grasp'] for s in samples),
        maximum_lift_m=max((float(s['object_pose'][2,3]-initial['object_pose'][2,3]) for s in samples),default=0.),
        terminal_stage=teacher.stage,terminal_reason=reason,terminal_native_step=adapter.observe()['step'],
        phase_transitions=clock.transitions,stage_budget_s=adapter.spec['stage_budget_s'],global_task_budget_s=60.,
        grasp_goal_raise_m=teacher.raise_m,initial_mount=initial_mount.tolist(),initial_TCP=initial['tcp'].tolist(),
        task_frame=frame.tolist(),capture=capture,cancel=cancel,at_30_seconds=end30,
        task_predicate='ACTUAL_NATIVE_GRASP_AND_LIFT_20MM_CONTINUOUS_ONE_SECOND',expert_takeover=False,
        future_suffix_is_executed=False)
