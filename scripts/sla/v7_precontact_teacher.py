"""Teacher-only intervention after a genuine complete P10, before a hard STOP."""
import numpy as np

PRECONTACT_MM=20.
STALL_TICKS=500
STALL_PROGRESS_M=.002


class EarlyWarning:
    def __init__(self, stage, model, color):
        if stage not in ('PREGRASP','LIFT') or color not in ('red','blue'):
            raise ValueError('Declare one early stage and original target')
        self.stage,self.model,self.color=stage,model,color
        self.history=[]

    def observe(self, sample, scored, consumed):
        from scripts.sla.v64_recovery_carry import exact_boundary
        from go2arx5_vla.world_model.multitask_score_v6 import BASE_MOTION
        import mujoco
        from go2arx5_vla.world_model.native_replay import forbid_integration

        step=sample['native_step']
        exact_boundary(consumed,step)
        if (step!=scored['native_step'] or abs(sample['time_s']-.002*step)>1e-8
                or not sample['safety_valid'] or not sample['gripper_state_valid']
                or sample.get('native_abort') or sample.get('illegal_contact')
                or any(scored[k] for k in ('unsafe_states','missing_native_intervals','wrong_object_contact_ever','wrong_object_grasp_ever'))):
            raise ValueError('Intervention requires a safe current full-P10 boundary; never revive STOP')
        if scored['target_grasp']!=sample['native_grasp']:
            raise ValueError('Original scorer and measured grasp disagree')
        if (scored['base_directional_motion_m']<BASE_MOTION or scored['ever_target_lift20mm']
                or (scored['ever_target_grasp'] and not scored['target_grasp'])):
            self.history=[]
            return None
        if self.stage=='PREGRASP':
            if scored['ever_target_grasp'] or sample['native_grasp']:
                return None
            model=self.model;data=mujoco.MjData(model)
            wrist=[i for i in range(model.ngeom) if model.body(int(model.geom_bodyid[i])).name=='x5_link6']
            body='task_cube' if self.color=='red' else 'task_blue_cube'
            target=[i for i in range(model.ngeom) if model.body(int(model.geom_bodyid[i])).name==body]
            if not wrist or not target:
                raise ValueError('Original wrist and target geometry are required')
            with forbid_integration(mujoco) as effects:
                data.qpos[:]=sample['qpos'];mujoco.mj_fwdPosition(model,data)
                gap=1000*min(mujoco.mj_geomDistance(model,data,a,b,.25,None) for a in wrist for b in target)
            if effects:
                raise ValueError('Teacher warning cannot integrate physics')
            if not 0.<gap<PRECONTACT_MM:
                return None
            return dict(trigger='PRECONTACT_20MM',native_step=step,native_time_s=sample['time_s'],
                        wrist_target_mm=float(gap),threshold_mm=PRECONTACT_MM,teacher_privileged_geometry=True)
        if not sample['native_grasp']:
            self.history=[]
            return None
        height=float(np.asarray(sample['objects'][self.color]['pose'])[2,3])
        if self.history and step-self.history[-1][0]!=250:
            self.history=[]
        self.history.append((step,height))
        self.history=[row for row in self.history if step-row[0]<=STALL_TICKS]
        if (len(self.history)<3 or self.history[-1][0]-self.history[0][0]!=STALL_TICKS
                or max(row[1] for row in self.history)-self.history[0][1]>=STALL_PROGRESS_M):
            return None
        return dict(trigger='LIFT_STALL_1S',native_step=step,native_time_s=sample['time_s'],
            observed_ticks=STALL_TICKS,maximum_upward_progress_m=max(row[1] for row in self.history)-self.history[0][1],
            minimum_progress_m=STALL_PROGRESS_M,teacher_privileged_object_height=True)


def prepare_warning(initial,sample,scored,spec,consumed,previous,base,event):
    from scripts.sla.v64_recovery_carry import accepted_pose,checked_plan
    from scripts.sla.v64_recovery_early_teacher import EarlyTeacher

    if event['trigger'] not in ('PRECONTACT_20MM','LIFT_STALL_1S') or event['native_step']!=sample['native_step']:
        raise ValueError('Use the actual fixed early warning event')
    if not sample['safety_valid'] or sample.get('native_abort') or scored['unsafe_states']:
        raise ValueError('No Teacher preparation after hard STOP')
    reference=accepted_pose(sample,consumed,previous)
    teacher=EarlyTeacher(initial,sample,spec['mobile_condition'],spec['multitask_geometry'],
                         spec['task_instruction_id'],previous['width_m'])
    return checked_plan(teacher,sample,reference,previous,base)
