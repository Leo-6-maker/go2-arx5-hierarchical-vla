"""One registered partial-reference cycle through the original native chain.

This is a distinct protocol. It never weakens the older full-H40 consumer.
Every issued preview remains full length; cancelled futures remain evidence.
"""
import copy
from pathlib import Path
import numpy as np
from .fusion_trajectory_adapter import TrajectoryAdaptation
from . import fusion_live_skill_segment as history
from . import fusion_trajectory_qualification_native as native
from .fusion_reference_qualification_native import require_original_install_ack
from .target_reference import require_install_clock,require_live_window
from .storage import write_json,jsonable
from .next2_episode import state_vector

PHASE='collect-target-stable-task-closed-loop-v1'
KIND='STABLE_TASK_CLOSED_LOOP_V1'
B=34303
PREFIX_ATOMS,POLICY_QUERIES,FINAL_HOLD_ATOMS=120,3,8
E=B+25*(PREFIX_ATOMS*POLICY_QUERIES+FINAL_HOLD_ATOMS)


def need(ok,why):
    if not ok:raise ValueError('STABLE_TASK_PREFIX_'+why)


def protocol():
    return dict(schema='WM02_STABLE_TASK_PAUSED_PREFIX_PROTOCOL_V1',
        prefix_atoms=PREFIX_ATOMS,policy_queries=POLICY_QUERIES,final_hold_atoms=FINAL_HOLD_ATOMS,
        assistance='NATIVE_QUOTA_PAUSED_REAL_OBSERVATION',
        cancelled_suffix='SUPERSEDED_ONLY_AFTER_ORIGINAL_NEW_REFERENCE_ACK',
        full_controller_preview_retained=True,observation_timestamp_rebased=False,
        original_per_install_MPC_clock=True,fresh_query_per_cycle=True,
        teacher_fallback=False,physics_dt_s=.002,WBC_changes=False)


def declared(spec):
    if spec.get('kind')=='TARGET_GR00T_PREFIX_LOOP_V1':
        from .target_gr00t_prefix_v1 import declared as target_declared
        return target_declared(spec)
    need(spec.get('phase')==PHASE and spec.get('kind')==KIND
         and spec.get('stable_task_closed_loop')==protocol()
         and spec.get('live_skill_start_step')==B and spec.get('max_ticks')==E
         and spec.get('physics_dt_s')==.002 and spec.get('WBC_changes')is False,
         'EXACT_NEW_TYPED_FINITE_PAUSED_PROTOCOL')


def _configuration(spec):
    declared(spec)
    if spec.get('kind')=='TARGET_GR00T_PREFIX_LOOP_V1':
        return spec['target_gr00t_prefix_protocol']
    return protocol()


def observation_steps(spec):
    c=_configuration(spec)
    return tuple(B+25*c['prefix_atoms']*i for i in range(c['policy_queries']))


def cycle(spec,index):
    c=_configuration(spec);queries=c['policy_queries']
    need(type(index)is int and 0<=index<=queries,'FINITE_CYCLE_INDEX')
    start=B+25*c['prefix_atoms']*index
    atoms=c['prefix_atoms']if index<queries else c['final_hold_atoms']
    return dict(index=index,kind='POLICY_PREFIX'if index<queries else'DETERMINISTIC_TERMINATION_HOLD',
        start_step=start,end_step=start+25*atoms,atoms=atoms)


def run_prefix(adapter,program,*,index,past_samples,past_windows,sample_observer):
    from scripts.pi0292.eval_contract import matrix10
    c=cycle(adapter.spec,index);start,end,atoms=c['start_step'],c['end_step'],c['atoms']
    samples,windows=list(past_samples),list(past_windows)
    need(type(program)is TrajectoryAdaptation and program.geometry_scale==1.
         and program.time_stretch_factor==2 and np.array_equal(program.base_pose_hold,adapter.base_pose_hold)
         and np.array_equal(adapter.base_pose_hold,adapter.trajectory_initial_base),
         'ORIGINAL_FULL_REFERENCE_PROGRAM_AND_FIXED_BASE')
    need(adapter.trace_sealed is False and not(adapter.output/'trace_seal_request').exists()
         and not(adapter.output/'init.trace.summary').exists()and adapter.latest['mode'].data=='WBC'
         and adapter.observe()['step']==samples[-1]['native_step']==start
         and len(windows)==(start-2503)//25 and len(adapter.reference_history)==len(windows),
         'SAME_EXISTING_UNSEALED_WORLD_AND_COMPLETE_GLOBAL_HISTORY')
    history._history(samples,windows)
    need(np.array_equal(program.decision_TCP_target,samples[-1]['tcp'])
         and np.array_equal(program.initial_reference_TCP_target,adapter.reference_history[-1]['pose']),
         'CURRENT_REAL_OBSERVATION_AND_LAST_CONSUMED_REFERENCE')
    sample_observer(samples[-1])
    folder=adapter.output/'STABLE_TASK_CYCLES'/f'{index:02d}'
    folder.mkdir(parents=True,exist_ok=False)
    rows,installs,acks=[],[],[]
    report=dict(schema='WM02_STABLE_TASK_PARTIAL_REFERENCE_ACTUAL_V1',cycle=c,
        status='STARTED_NOT_NATIVE_QUALIFIED',raw_H40=program.decoded_H40,
        full_reference_path=program.reference_path,adaptation=program.evidence(),
        installs=installs,original_install_ACKs=acks,windows=rows,
        teacher_fallbacks=0,original_low_level_state_reset=False,full_MPC_hidden_state_restored=False)
    write_json(folder/'INTENT.private.json',report)
    adapter.stable_task_active_cycle=dict(c,program=program)
    adapter.live_skill_program=program
    attempt=None
    def install(elapsed):
        previous=copy.deepcopy(adapter.active_reference)
        action=program.preview(elapsed)
        record=dict(native_step=start+25*elapsed,elapsed_atom=elapsed,
            full_proposed_H40=action.tolist(),previous_reference_sha256=previous.get('wire_sha256'),
            previous_full_controller_preview=previous.get('action_EEF10'),stage='BEFORE_EFFECT_AND_ACK')
        installs.append(record)
        reference=adapter.install_reference(action)
        ack=require_original_install_ack(reference);packet=adapter.observe()
        require_install_clock(reference['source_mpc_s'],packet['time_s'])
        need(packet['step']==start+25*elapsed and np.array_equal(reference['action_EEF10'],action)
             and np.array_equal(adapter.active_reference['action_EEF10'],action)
             and adapter.active_reference['wire_sha256']==reference['wire_sha256'],
             'EXACT_NEW_FULL_PREVIEW_ACK_AT_UNCHANGED_NATIVE_CLOCK')
        record.update(stage='ORIGINAL_INSTALL_ACKED_BEFORE_EFFECT',wire_sha256=reference['wire_sha256'],
            source_mpc_s=reference['source_mpc_s'],receiver_events=reference['receiver_events'],
            supersedes_previous_from_native_step=packet['step'])
        acks.append(dict(ack,native_step=packet['step'],elapsed_atom=elapsed))
        return packet['step']
    try:
        installed=install(0)
        for elapsed in range(atoms):
            step=start+25*elapsed;before=adapter.observe()
            need(before['step']==step and abs(before['time_s']-.002*step)<=1e-8
                 and not adapter.trace_sealed,'EXACT_NATIVE_ATOM_CLOCK')
            h=history._history(samples,windows)
            if step-installed>=175:installed=install(elapsed)
            clock=require_live_window(adapter.active_reference['source_mpc_s'],before['time_s'],25)
            position=(step-installed)//25
            actual_row=np.asarray(adapter.active_reference['action_EEF10'])[position]
            need(np.array_equal(actual_row,program.preview(elapsed)[0]),'EXACT_INSTALLED_PREFIX_ROW')
            attempt=dict(trajectory_elapsed_atom=elapsed,native_start_step=step,
                actual_history_before_atom=h,actual_installed_consumed_EEF10_BASE7=np.r_[actual_row,adapter.base_pose_hold].tolist(),
                full_final_installed_H40=adapter.active_reference['action_EEF10'],
                original_reference_sha256=adapter.active_reference['wire_sha256'],clock=clock)
            submitted=native.submit_gripper(adapter,len(windows),actual_row[9])
            need(submitted['native_start_step']==step and submitted['sequence']==len(windows),
                 'ORIGINAL_CONTIGUOUS_GRIPPER_SUBMISSION')
            attempt.update(gripper_submitted=submitted,stage='SUBMITTED_BEFORE_EFFECT')
            adapter.command_event(dict(phase='live-skill-before-atom',protocol=adapter.spec['phase'],**attempt))
            event_start=len(adapter.ros_evidence)
            after=adapter.advance_to(step+25);adapter.spin(.15)
            actual=adapter.measured_task_state();samples.append(actual)
            attempt.update(actual_TCP=np.asarray(actual['tcp']).tolist(),
                actual_state_vector=np.asarray(state_vector(actual)).tolist(),
                actual_native_step=actual['native_step'],actual_packet=dict(step=after['step'],time_s=after['time_s']),
                stage='ACTUAL_EFFECT_OBSERVED_NOT_YET_CONFIRMED')
            history._sample(actual,step+25);sample_observer(actual)
            need(after['step']==step+25 and abs(after['time_s']-actual['time_s'])<=1e-8
                 and adapter.latest['mode'].data=='WBC'
                 and not any(v.get('reason')=='PREFIX_TTL_EXPIRED'for v in adapter.reference_events()),
                 'ORIGINAL_SAFE_WBC_TTL_AND_PAUSE')
            grip=native.confirm_gripper(adapter,submitted,actual)
            consumed=native.consumed_window(adapter,event_start,before,after,clock)
            need(consumed['native_start_step']==step and consumed['native_end_step']==step+25
                 and consumed['actual_ticks']==25 and consumed['reference_sha256']==adapter.active_reference['wire_sha256']
                 and bool(consumed['solver_policies_with_exact_installed_target']),
                 'REAL_ORIGINAL_SOLVER_CONSUMPTION')
            attempt.update(consumed,gripper=grip,actual_TCP=actual['tcp'].tolist(),
                actual_state_vector=state_vector(actual).tolist(),stage='ACTUAL_CONSUMPTION_CONFIRMED')
            rows.append(attempt);windows.append(attempt)
            adapter.reference_history.append(dict(end_native_step=step+25,pose=matrix10(actual_row)))
            adapter.command_event(dict(phase='live-skill-controller-consumed',protocol=adapter.spec['phase'],**attempt))
            attempt=None
        mask=[int(k)<atoms for k in program.knot_indices]
        report.update(status='REGISTERED_PREFIX_CONSUMED_REQUIRES_OUTER_TRACE_AND_REPLACEMENT',
            actual_native_integrations=25*len(rows),complete_suffix_atoms=len(rows),solver_target_confirmed_atoms=len(rows),gripper_confirmed_atoms=len(rows),
            raw_knot_consumed=mask,all_40_knots_consumed=bool(all(mask)),
            unexecuted_raw_knot_indices=[i for i,v in enumerate(mask)if not v],
            unexecuted_reference_path=program.reference_path[atoms:],
            last_controller_visible_H40=copy.deepcopy(adapter.active_reference['action_EEF10']),
            final_reference_sha256=adapter.active_reference['wire_sha256'],
            cancellation_status='PENDING_NEXT_ORIGINAL_INSTALL_OR_OUTER_FINAL_CANCEL',trace_sealed_by_segment=False)
        write_json(folder/'RESULT.private.json',report)
        return report,samples,windows
    except Exception as error:
        try:actual_step=adapter.observe()['step']
        except Exception:actual_step=None
        report.update(status='FAILED_PREFIX_RETAINED_NO_RETRY',failure=type(error).__name__+': '+str(error),
            complete_suffix_atoms=len(rows),last_observed_native_step=actual_step,
            actual_native_integrations=(actual_step-start if type(actual_step)is int else None),failure_attempt=attempt)
        write_json(folder/'RESULT.private.json',history._failed_evidence(report))
        raise
