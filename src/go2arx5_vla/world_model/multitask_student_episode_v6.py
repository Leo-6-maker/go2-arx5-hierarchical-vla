"""Shared HOLD/PLACE policy on the owned clock, original install and P10."""
from pathlib import Path

from .multisource_d4_native import SCOPE as D4_SCOPE
from .sla_followup_native_v1 import SCOPE as D416_SCOPE
from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
from .mfree_expert_capture_v1 import initialize
from .mobile_boundary_clock_v1 import boundary
from .multitask_camera_v6 import TwoObjectCameraSession
from .multitask_score_v6 import Score
from .multitask_state_v6 import TwoObjectState, score_through
from .shift7_native_input_v5 import capture, verify_query_return
from .sim_only_mobile_v51 import request_policy
from .sla_dense_policy_v5 import compile_proposal
from .task_learning_student_v1 import StudentProposalRejected


def compile_g7(adapter, prepared, sample, reference, response):
    """Apply the unchanged compiler and the recorded previous-command speed check."""
    state = prepared['observation'].values_for(prepared['metadata']['input'])['state_eef10']
    action = response['action']
    program, evidence = compile_proposal(action['path'],action['sha256'],
        source_from_target=adapter.spec['policy_from_world'],query_state10=state,
        actual_TCP=sample['tcp'],consumed_reference_TCP=reference,base_hold=adapter.base_pose_hold)
    speed = label_margin_report(program.decoded_H40,prepared['previous_reference'])
    return program, evidence, speed


def consume_prefix(adapter, prepared, program, query, action, speed):
    from scripts.wm02.mobile_boundary_runner_v1 import install, consume

    meta = prepared['metadata']
    folder = Path(meta['input']['path']).parent
    start = meta['native_step']
    adapter.mobile_program = program
    adapter.mobile_program_start_step = start
    intervals, acks = [], []
    reference = None
    is_d4 = adapter.spec.get('scope') in (D4_SCOPE, D416_SCOPE)
    try:
        for atom in range(meta['available_prefix_atoms']):
            adapter.guard()
            preview = program.preview(atom)
            if atom in (0,7):
                if atom == 0:
                    verify_query_return(adapter,prepared)
                install(adapter,preview,None)
                acks.append(dict(atom=atom,wire_sha256=adapter.active_reference['wire_sha256'],
                                 source_native_s=adapter.active_reference['source_mpc_s']))
            sample,_ = consume(adapter,preview,sequence=128+query*10+atom)
            interval = dict(atom=atom,native_start_step=start+25*atom,native_end_step=sample['native_step'],
                            wire_sha256=adapter.active_reference['wire_sha256'])
            intervals.append(interval)
            reference, witness = boundary(adapter)
            adapter.reference_history[-1]['pose'] = reference.copy()
            interval['accepted_boundary'] = witness
            if is_d4:
                emitted = adapter.close_previous_reference
                interval.update(emitted_gripper_width_m=emitted['width_m'],
                                emitted_gripper_source_native_s=emitted['source_native_s'],
                                measured_gripper_width_m=float(sample['finger_q'].mean()))
    finally:
        try:
            terminal = adapter.observe()['step']
        except Exception:
            terminal = None
        remaining = None if terminal is None else terminal-start-25*len(intervals)
        record = dict(query=query,action=action,actual_model_generations=1,
            full_H40_compiled=True,original_gripper_speed=speed,install_ACKs=acks,confirmed_intervals=intervals,
            confirmed_atoms=len(intervals),full_P10=len(intervals)==10,requested_atoms=meta['available_prefix_atoms'],
            native_start_step=start,terminal_native_step=terminal,unconfirmed_native_ticks=remaining,
            partial_native_ticks=None if remaining is None else remaining%25,mock_consumption=False,
            controller_selected_reference='UNKNOWN')
        if is_d4:
            record['previous_actual_gripper_command'] = prepared['previous_reference']
        write(folder/'QUERY_ACTUAL.private.json', record)
    return reference, len(intervals)


def episode(adapter):
    spec = adapter.spec
    instruction = adapter.task.instruction
    initial, mount, reference, _, frame = initialize(adapter)
    adapter.shift7_fixed_task_frame = frame.copy()
    score = Score(instruction,initial,spec['multitask_geometry'])
    reader = TwoObjectState(adapter.actual_state.model,adapter.task.color)
    camera = TwoObjectCameraSession(spec['model_path'])
    results = adapter.output/'MULTITASK_SCORE_PROGRESS.private.json'
    queries = atoms = 0
    rejection = None
    try:
        with (adapter.output/'state.bin.journal').open('rb') as stream:
            result = score_through(stream,reader,score,initial['native_step'])
            write(results,result)
            for query in range(spec['policy_queries']):
                sample = adapter.measured_task_state()
                prepared = capture(adapter,sample,camera,frame,reference,query)
                response = request_policy(adapter,prepared,query)
                queries += 1
                speed = None
                try:
                    program,evidence,speed = compile_g7(adapter,prepared,sample,reference,response)
                except StudentProposalRejected as error:
                    rejection = dict(kind='ORIGINAL_FULL_H40_POLICY_REJECTED',reason=str(error),
                                     full_H40_compiled=False)
                else:
                    if speed['original_execution_speed_exceeded']:
                        rejection = dict(kind='G7_ACTUAL_PREVIOUS_COMMAND_SPEED_REJECTED',
                                         original_gripper_speed=speed,full_H40_compiled=True)
                if rejection is not None:
                    rejection.update(query=query,installed=False,full_H40_checked=True)
                    write(Path(prepared['metadata']['input']['path']).parent/'QUERY_ACTUAL.private.json',
                          dict(query=query,action=response['action'],actual_model_generations=1,
                               install_ACKs=[],confirmed_atoms=0,native_start_step=sample['native_step'],
                               full_H40_compiled=rejection['full_H40_compiled'],rejection=rejection))
                    break
                verify_query_return(adapter,prepared)
                adapter.command_event(dict(phase='v6-g7-original-h40',query=query,
                    original_gripper_speed=speed,**evidence))
                reference, consumed = consume_prefix(adapter,prepared,program,query,response['action'],speed)
                atoms += consumed
                result = score_through(stream,reader,score,adapter.observe()['step'])
                write(results,result)
                if result['task_success']:
                    break
    finally:
        camera.close()
    succeeded = result['task_success']
    reason = 'TASK_SUCCESS' if succeeded else rejection['kind'] if rejection else 'FIXED_TASK_BUDGET_EXHAUSTED'
    cancel = adapter.cancel_reference()
    return dict(schema='SLA_V6_MULTITASK_STUDENT_ACTUAL',arm=spec['arm'],task_instruction=instruction,
        task_success=succeeded,score=result,actual_SLA_queries=queries,expert_queries=0,consumed_task_atoms=atoms,
        ever_grasp=result['ever_target_grasp'],maximum_lift_m=result['maximum_lift_m'],
        longest_grasp_lift_hold_s=result['longest_grasp_lift_hold_s'],valid_policy_rejection=rejection,
        terminal_native_step=adapter.observe()['step'],terminal_reason=reason,initial_mount=mount.tolist(),
        initial_TCP=initial['tcp'].tolist(),fixed_frame=frame.tolist(),cancel=cancel,
        teacher_phase_input=False,teacher_takeover=False,BASE_reference_present=False,
        strict_wall_realtime_qualified=False,controller_selected_reference='UNKNOWN')
