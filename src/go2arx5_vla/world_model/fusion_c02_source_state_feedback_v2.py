"""Finite measured-pose initialization repair; no GR00T or learned correction.

The failed fixed-goal run remains failed. Sixteen predeclared ordinary feedback
slots correct its evidenced steady tracking error in a fresh independent run.
"""
from collections import deque
import copy
import hashlib
import json
import numpy as np
from . import fusion_c02_source_state_initialization_v1 as fixed
from .fusion_task_phase_pose_feedback_v6 import _bounded, _log_so3, _exp_so3
from .storage import digest, write_json

PHASE='collect-target-fusion-c02-source-state-feedback-v2'
KIND='FUSION_C02_SOURCE_STATE_FEEDBACK_V2'
DECISION=fixed.DECISION
MOVE, SETTLE, COUNT=28,8,16
FIRST_ATOM=fixed.MAX_MOVEMENT_ATOMS+8
END_ATOM=FIRST_ATOM+COUNT*(MOVE+SETTLE)+24
E=DECISION+25*(END_ATOM+8)
STEPS=tuple(DECISION+25*(FIRST_ATOM+i*(MOVE+SETTLE))for i in range(COUNT))
FAILED_RESULT='2a5c05f78145ab3ce551a4d7695f32540778d207b1ac666565ee2e69da218a55'
need=fixed.need
MEMBERS=fixed.SOURCE_MEMBERS+('src/go2arx5_vla/world_model/fusion_c02_source_state_feedback_v2.py',
 'src/go2arx5_vla/world_model/fusion_task_phase_pose_feedback_v6.py')


def episode(run):
    return dict(fixed.episode(run),schema=KIND+'_EPISODE',phase=PHASE,
        planned_end_atom=END_ATOM,reference_timing='SAME_51_2S_TEACHER_THEN_FINITE_MEASURED_POSE_FEEDBACK',
        feedback_steps=list(STEPS),feedback_gain=.5,feedback_deadband_m=.001,
        feedback_step_cap_m=.008,feedback_cumulative_cap_m=.128,
        feedback_angular_gain=.5,feedback_angular_deadband_rad=.005,
        feedback_angular_step_cap_rad=.01,feedback_angular_cumulative_cap_rad=.16,
        feedback_move_atoms=MOVE,feedback_settle_atoms=SETTLE,feedback_final_hold_atoms=24,
        old_fixed_goal_failed_result_sha256=FAILED_RESULT,independent_restart_not_exact_fork=True)


def config(spec):
    return fixed.configured_inputs(spec,PHASE,KIND,E,episode,MEMBERS,1200)


def episode_limit(spec,previous):
    config(spec);p=fixed.read(spec['previous_stage'],FAILED_RESULT)
    from pathlib import Path
    root=Path(spec['previous_stage']).parent.parent
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    q=p.get('reference_qualification')or{}
    need(previous==p and spec['previous_stage_sha256']==FAILED_RESULT and lease['status']=='CLOSED'
        and worker['exit_code']==lease['worker_exit_code']==1 and p['phase']==fixed.PHASE
        and p['status']=='NATIVE_STAGE_FAILED_PRESERVED'and p['actual_target_integrations']==fixed.E
        and p['shutdown_clean']and p['physics_stops']==[]and p['original_joint_or_warning_samples']==0
        and p['trace_audit']['status']=='RECORDED_TRACE_CHECKS_PASS'and q['all_40_knots_consumed']
        and q['C02_initialization_qualification']['qualified']is False,
        'SEALED_ACTUAL_FIXED_GOAL_TRACKING_FAILURE_BEFORE_NEW_FEEDBACK')
    return E


class FeedbackProgram(fixed.FixedGoalProgram):
    def __init__(self,initial,decision,goal,base,cube):
        super().__init__(initial,decision,goal,base)
        from .fusion_task_phase_initialization import _immutable
        for name in('reference_path','desired_path'):
            old=getattr(self,name);object.__setattr__(self,name,_immutable(np.r_[old,np.repeat(old[-1:],END_ATOM-fixed.MAX_MOVEMENT_ATOMS,axis=0)]))
        object.__setattr__(self,'end_atom',END_ATOM)
        object.__setattr__(self,'goal',_immutable(np.asarray(goal)))
        object.__setattr__(self,'cube',_immutable(np.asarray(cube)))
        object.__setattr__(self,'decisions',[])
        object.__setattr__(self,'sum_m',0.);object.__setattr__(self,'sum_rad',0.)
        self._evidence.update(schema=KIND+'_PROGRAM',feedback_steps=list(STEPS),finite_feedback_count=COUNT,
            original_fixed_teacher_end_atom=fixed.MAX_MOVEMENT_ATOMS,final_end_atom=END_ATOM,
            ordinary_feedback_not_WM_or_VLA_gain=True)

    def prepare_feedback(self,step,actual):
        from scripts.pi0294.dec1.target_observation import rigid
        from scripts.pi0292.eval_contract import matrix10
        from ..task_data.codec import pose10
        from .fusion_task_phase_initialization import _immutable
        actual=rigid(actual);i=len(self.decisions)
        need(i<COUNT and type(step)is int and step==STEPS[i],'EXACT_NEXT_FINITE_C02_FEEDBACK_SLOT')
        atom=(step-DECISION)//25;start=self.reference_path[atom].copy()
        need(np.array_equal(self.reference_path[atom:],np.repeat(start[None],len(self.reference_path)-atom,axis=0)),
            'PREVIOUS_COMMITTED_REFERENCE_HOLD_NO_REANCHOR')
        xyz=self.goal[:3,3]-actual[:3,3];rot=_log_so3(self.goal[:3,:3]@actual[:3,:3].T)
        _,_,delta,length=_bounded(xyz,.5,.001,.008)
        _,_,rotation,angle=_bounded(rot,.5,.005,.01)
        need(self.sum_m+length<=.128+1e-12 and self.sum_rad+angle<=.16+1e-12,'FINITE_REFERENCE_PATH_LENGTHS')
        duration=MOVE*.05
        need(1.875*length/duration<=.05 and 10*np.sqrt(3)/3*length/duration**2<=.25
            and 1.875*angle/duration<=.25 and 10*np.sqrt(3)/3*angle/duration**2<=1.,'ORIGINAL_REFERENCE_RATE_DOMAIN')
        path=self.reference_path.copy();u=np.arange(MOVE+1)/MOVE;weight=10*u**3-15*u**4+6*u**5
        path[atom:atom+MOVE+1,:3]=start[:3]+weight[:,None]*delta
        first=matrix10(start)
        for j,w in enumerate(weight[1:],1):
            pose=first.copy();pose[:3,:3]=_exp_so3(w*rotation)@first[:3,:3]
            path[atom+j,3:9]=pose10(pose,float(start[9]))[3:9]
        path[atom+MOVE:]=path[atom+MOVE]
        need(np.array_equal(path[:atom+1],self.reference_path[:atom+1])and np.array_equal(path[:,9],self.reference_path[:,9]),'PAST_START_AND_GRIPPER_IMMUTABLE')
        clearance=float(np.linalg.norm(path[atom:,:3]-self.cube[:3,3],axis=1).min())
        need(clearance>=.07,'FULL_REMAINING_REFERENCE_COURSE_CLEARANCE')
        return _immutable(path),dict(index=i,native_step=step,actual_TCP=actual.tolist(),fixed_desired_goal_TCP=self.goal.tolist(),
            xyz_error_m=xyz.tolist(),angular_error_world_rad=rot.tolist(),applied_delta_world_m=delta.tolist(),
            applied_delta_norm_m=length,applied_rotation_world_rad=rotation.tolist(),applied_rotation_norm_rad=angle,
            remaining_reference_clearance_m=clearance,reference_path_sha256=hashlib.sha256(path.astype('<f8').tobytes()).hexdigest(),
            actual_pose_is_reference_anchor=False,expert_label=False,WM_queries=0)

    def commit(self,path,proof,pin):
        need(proof['index']==len(self.decisions)and len(pin)==64,'SAVED_DECISION_BEFORE_PROGRAM_COMMIT')
        object.__setattr__(self,'reference_path',path)
        object.__setattr__(self,'sum_m',self.sum_m+proof['applied_delta_norm_m'])
        object.__setattr__(self,'sum_rad',self.sum_rad+proof['applied_rotation_norm_rad'])
        self.decisions.append(dict(proof,decision_sha256=pin))

    def evidence(self):
        return dict(super().evidence(),feedback_decisions=copy.deepcopy(self.decisions),
            final_reference_path_sha256=hashlib.sha256(self.reference_path.astype('<f8').tobytes()).hexdigest())


def validate_install(adapter,action):
    from .fusion_trajectory_qualification_contract import hold_preview
    step=adapter.observe()['step'];program=getattr(adapter,'trajectory_program',None)
    need(adapter.spec['phase']==PHASE and fixed.INIT<=step<E and(step-fixed.INIT)%25==0,'FINITE_V2_ATOM_CLOCK')
    if step>=DECISION:
        need(type(program)is FeedbackProgram and np.array_equal(program.base_pose_hold,adapter.trajectory_initial_base)
            and len(program.decisions)==sum(s<=step for s in STEPS),'MEASURED_BASE_AND_ALL_DUE_FEEDBACK_DECISIONS')
    expected=hold_preview(adapter.trajectory_initial_tcp)if step<DECISION else program.preview((step-DECISION)//25)
    need(np.array_equal(action,expected),'COMPLETE_ACTUAL_V2_FULL40_PREVIEW')
    return dict(native_step=step,points_checked=40,original_certify_chunk_required=True,physics_advance_authorized=False)


def run(adapter):
    from .fusion_trajectory_qualification_native import run_program
    from .fusion_task_phase_live_skill_native_v8 import _noncontact
    from .fusion_reference_qualification_native import require_original_install_ack
    from scripts.pi0294.task_state import parse_state
    from .fusion_task_phase_initialization_native import _plain
    doc,inputs=config(adapter.spec);tail=deque(maxlen=9);cube=None
    def observe(sample):
        nonlocal cube
        if cube is None:cube=np.asarray(sample['object_pose']).copy()
        _noncontact(sample,cube);tail.append(sample)
    def factory(value,document,initial,decision,base):
        _,receipt=fixed.build(value,fixed.episode(document['run']),initial,decision,base,cube)
        write_json(adapter.output/'C02_FIXED_GOAL_REFERENCE.private.json',receipt)
        return FeedbackProgram(initial,decision,np.asarray(receipt['goal_TCP_target']),base,cube)
    def feedback(current,before,last,window):
        step=before['step']
        if step not in STEPS:return False
        need(current is adapter and window is not None and window['native_end_step']==step
            and window['native_start_step']==step-25 and window['actual_ticks']==25
            and window['solver_policies_with_exact_installed_target']and window['gripper']['command_consumed']
            and window['reference_sha256']==adapter.active_reference['wire_sha256'],'ACTUAL_COMPLETE_CONSUMPTION_BEFORE_FEEDBACK')
        raw=(adapter.output/'state.bin').read_bytes();packet=parse_state(raw);measured=adapter.measured_task_state()
        need(packet['step']==measured['native_step']==last['native_step']==step and all(np.array_equal(packet[k],measured[k])and np.array_equal(measured[k],last[k])for k in('qpos','qvel','ctrl'))and np.array_equal(measured['tcp'],last['tcp']),'FRESH_ORIGINAL_ACTUALSTATE_AND_PACKET')
        program=adapter.trajectory_program;path,proof=program.prepare_feedback(step,measured['tcp'])
        dest=adapter.output/f'C02_FEEDBACK_{len(program.decisions)+1:02d}.private.json'
        need(not dest.exists(),'FRESH_FINITE_DECISION_NO_OVERWRITE')
        receipt=dict(schema=KIND+'_SAVED_CAUSAL_DECISION',decision=proof,actual_measurement=_plain(measured),
            packet_sha256=hashlib.sha256(raw).hexdigest(),previous_complete_window=window,
            previous_original_ACK=require_original_install_ack(adapter.active_reference),reference_path=path.tolist(),
            saved_before_commit_and_install=True,source_sha=adapter.spec['source_sha'])
        write_json(dest,receipt)
        need(json.loads(dest.read_bytes())==receipt and(adapter.output/'state.bin').read_bytes()==raw and adapter.observe()['step']==step,'DURABLE_READBACK_AND_PAUSED_PACKET')
        program.commit(path,proof,digest(dest));adapter.command_event(dict(phase='c02-feedback-saved-before-install',native_step=step,decision_sha256=digest(dest)))
        return True
    def finish(report):
        need(len(adapter.trajectory_program.decisions)==COUNT,'ALL_FINITE_C02_FEEDBACK_SLOTS')
        q=fixed.qualification(tail,inputs,end_step=E);q.update(schema=KIND+'_ACTUAL_QUALIFICATION',original_fixed_goal_thresholds_unchanged=True);write_json(adapter.output/'C02_MEASURED_QUALIFICATION.private.json',q)
        report.update(C02_initialization_qualification=q,ordinary_feedback_not_WM_or_GR00T_gain=True,
            actual_initialization_feedback_decisions=COUNT,GR00T_raw_H40_consumed=False,Teacher_initialization_all40_consumed=report['all_40_knots_consumed'],expert_training_eligible=False)
    result=run_program(adapter,doc,inputs,program_factory=factory,sample_observer=observe,before_preview=feedback,
        report_finalizer=finish,report_context=dict(experiment_role='C02_SOURCE_STATE_FEEDBACK_COMMISSIONING_V2',desired_outcome_provenance='PRIVILEGED_INITIALIZATION_NOT_GR00T'))
    need(result['C02_initialization_qualification']['qualified'],'V2_MEASURED_INITIALIZATION_FAILED_KEEP_ALL_EVIDENCE')
    return result
