"""Collect full native Teacher plans with a separate measured consumption mask."""
import json

from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
from .mfree_capture_v1 import LiveWriter
from .mfree_expert_capture_v1 import initialize
from .mobile_boundary_clock_v1 import boundary
from .multitask_camera_v6 import TwoObjectCameraSession
from .multitask_score_v6 import Score
from .multitask_state_v6 import TwoObjectState, score_through
from .multitask_teacher_v6 import MultitaskTeacher


def confirmed_context(adapter, step, query_time):
    """Use the final actually acknowledged atom already recorded by consume."""
    log = adapter.output/'MOBILE_SAMPLES.private.jsonl'
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-16384))
        row = json.loads(stream.read().splitlines()[-1])
    if row['native_end_step'] != step or row['native_start_step'] != step-25 or row['ticks'] != 25:
        raise ValueError('ACTUAL_PREVIOUS_MULTITASK_ATOM_REQUIRED')
    return dict(width_m=.044*row['installed_EEF10'][9],
                source_native_s=row['gripper_command']['source_native_s'],
                query_native_s=query_time)


def episode(adapter):
    from scripts.wm02.mobile_boundary_runner_v1 import install, consume

    condition = adapter.spec['mobile_condition']
    instruction = adapter.task.instruction
    geometry = adapter.spec['multitask_geometry']
    initial, _, reference, witness, frame = initialize(adapter)
    score = Score(instruction,initial,geometry)
    reader = TwoObjectState(adapter.actual_state.model,adapter.task.color)
    teacher = MultitaskTeacher(initial,condition,geometry,instruction)
    writer = LiveWriter(adapter,frame,camera_factory=TwoObjectCameraSession)
    scored = None
    query_count = 0
    consumed = 0
    margin_exclusions = 0
    first_margin = None
    rejection = None
    stages = []
    try:
        with (adapter.output/'state.bin.journal').open('rb') as stream:
            scored = score_through(stream,reader,score,initial['native_step'])
            for query in range(adapter.spec['policy_queries']):
                sample = adapter.measured_task_state()
                try:
                    program, evidence = teacher.decide(sample,reference,adapter.base_pose_hold)
                except ValueError as error:
                    if 'ORIGINAL_FULL_PREVIEW_RATE_AND_ACCELERATION_LIMITS' not in str(error):
                        raise
                    rejection = dict(query=query,reason=str(error),full_H40_checked=True,installed=False)
                    break
                if not stages or stages[-1]['stage'] != teacher.stage:
                    stages.append(dict(stage=teacher.stage,native_step=sample['native_step']))
                previous = confirmed_context(adapter,sample['native_step'],sample['time_s'])
                margin = label_margin_report(program.decoded_H40,previous)
                if margin['training_margin_exceeded']:
                    margin_exclusions+=1
                    if first_margin is None:
                        first_margin = dict(query=query,native_step=sample['native_step'],**margin)
                row = writer.capture(sample,program,evidence,witness,teacher)
                row.update(instruction=instruction,previous_confirmed_command=previous,
                           original_label_G7_margin=margin)
                if teacher.stage == 'CARRY':
                    row['teacher_carry_clearance_audit_only'] = dict(
                        swept_footprint_overlaps=evidence['teacher_carry_footprint_overlap'],
                        waiting_for_measured_clearance=evidence['teacher_carry_waiting_for_measured_clearance'],
                    )
                query_count += 1
                adapter.mobile_contact_stage = (
                    'approach' if teacher.stage in ('PREGRASP','DESCEND') else teacher.stage.lower()
                )
                adapter.mobile_program=program
                adapter.mobile_program_start_step=sample['native_step']
                for atom in range(10):
                    action=program.preview(atom)
                    if atom in (0,7):
                        install(adapter,action,None)
                    interval=dict(atom=atom,start_native_step=adapter.observe()['step'],
                                  installed_reference_sha256=adapter.active_reference['wire_sha256'],
                                  confirmed_ticks=0)
                    row['intervals'].append(interval)
                    sample,_=consume(adapter,action,sequence=128+query*10+atom)
                    reference,witness=boundary(adapter)
                    adapter.reference_history[-1]['pose']=reference.copy()
                    interval.update(end_native_step=sample['native_step'],confirmed_ticks=25,boundary=witness)
                    row['consumed_mask'][atom]=True
                    consumed+=1
                teacher.commit(program,sample['native_step'])
                row['next_query_boundary']=witness
                scored=score_through(stream,reader,score,sample['native_step'])
                write(adapter.output/'MULTITASK_SCORE_PROGRESS.private.json',scored)
                if scored['task_success']:
                    break
    finally:
        capture=writer.close()
    success = scored is not None and scored['task_success']
    terminal_reason = ('TASK_SUCCESS' if success else 'ORIGINAL_H40_EXPERT_PROPOSAL_REJECTED'
                       if rejection else 'TASK_BUDGET_EXHAUSTED_IN_'+teacher.stage)
    return dict(schema='MFREE_MULTITASK_TEACHER_ACTUAL_V6',task_success=success,
        score=scored,task_instruction=instruction,ever_grasp=scored['ever_target_grasp'],
        maximum_lift_m=scored['maximum_lift_m'],longest_grasp_lift_hold_s=scored['longest_grasp_lift_hold_s'],
        terminal_reason=terminal_reason,valid_expert_proposal_rejection=rejection,
        terminal_native_step=adapter.observe()['step'],actual_SLA_queries=0,
        expert_queries=query_count,consumed_task_atoms=consumed,stages=stages,
        label_margin_windows=query_count,label_margin_exclusions=margin_exclusions,
        first_label_margin_exclusion=first_margin,capture=capture,cancel=adapter.cancel_reference())
