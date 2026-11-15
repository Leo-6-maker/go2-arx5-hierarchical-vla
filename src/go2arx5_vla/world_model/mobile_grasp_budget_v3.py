"""Independent native expert qualification: stage budgets, explicit geometry arm.

No WBC/safety change. Only F2/M_FREE uses the separately identified +8mm
high-level grasp target, based on saved base-motion/table-contact evidence.
"""
import numpy as np

from .mobile_continuous_teacher_v2 import ContinuousTeacher

VERSION='MOBILE_GRASP_BUDGET_V3'
STAGE_SECONDS=dict(PREGRASP=20.,DESCEND=20.,CLOSE=6.,LIFT=10.,HOLD=3.)
MAX_TASK_SECONDS=60.
MAX_QUERIES=120


def grasp_raise(condition):
    return .008 if condition['family']=='F2'and condition['mode']=='M_FREE'else 0.


class BudgetTeacher(ContinuousTeacher):
    version=VERSION

    def __init__(self,initial,condition):
        super().__init__(initial)
        self.raise_m=grasp_raise(condition)

    def grasp_goal(self):
        return super().grasp_goal()+np.array([0.,0.,self.raise_m])

    def decide(self,sample,reference,base):
        program,evidence=super().decide(sample,reference,base)
        evidence.update(teacher_version=VERSION,grasp_goal_raise_m=self.raise_m,
            experimental_arm='BUDGET_AND_F2_FREE_PAD_CLEARANCE'if self.raise_m else'BUDGET_ONLY',
            original_WBC_and_safety=True)
        return program,evidence


class StageClock:
    def __init__(self,start):
        self.start=start;self.stage=None;self.stage_start=None;self.transitions=[]

    def observe(self,stage,native_step):
        if stage not in STAGE_SECONDS or native_step<self.start or(native_step-self.start)%250:
            raise ValueError('V3_ACTUAL_P10_STAGE_CLOCK')
        if stage!=self.stage:
            self.transitions.append(dict(stage=stage,native_step=native_step,
                task_elapsed_s=(native_step-self.start)*.002))
            self.stage=stage;self.stage_start=native_step
        elapsed=(native_step-self.stage_start)*.002
        return dict(stage=stage,stage_elapsed_s=elapsed,stage_limit_s=STAGE_SECONDS[stage],
            task_elapsed_s=(native_step-self.start)*.002,
            exhausted=elapsed>=STAGE_SECONDS[stage],forced_transition=False)


def qualification_record(result):
    detail=result.get('detail')or{}
    passed=bool(result['completed']and not result['unsafe']and not result['technical_failure']
        and result['shutdown_clean']and detail.get('task_success')
        and(result.get('trace_audit')or{}).get('status')=='RECORDED_TRACE_CHECKS_PASS'
        and result['actual_native_integrations']==detail.get('terminal_native_step'))
    return dict(condition=result['condition'],qualified=passed,
        native_integrations=result['actual_native_integrations'],
        reason='ORIGINAL_COMPLETE_GRASP_LIFT_HOLD_AND_CLEAN_NATIVE_TRACE'if passed else
            'ORIGINAL_NATIVE_SAFETY_STOP'if result['unsafe']else
            'TECHNICAL_FAILURE'if result['technical_failure']else detail.get('terminal_reason','UNQUALIFIED'),
        new_training_updates=0,new_training_visits=0)
