"""Frozen-label oracle diagnostic in the unchanged native single-world loop.

Reconstruct control history by real original startup and expert-prefix replay.
Never reset qpos/qvel, reanchor a label, or count the oracle as a policy.
Heavy imports stay in the actual execution functions.
"""
import json
from pathlib import Path
from functools import lru_cache

VERSION = 'SLA_FP32_LABEL_NATIVE_RP8_V6_1'
KEY = 'sla_label_replay_v61'
SELECTION = 'experiments/sla_go2_posttrain/s16_rp8_reference_v61.json'
STATE_ATOL = 1e-9  # Existing captured-state/frame binding; no widened gate.


def configuration(spec):
    value = spec.get(KEY)
    if value is None:
        return None
    from .storage import digest
    from .stable_task_scene_v1 import pinned
    from ..sla_adapter.action_contract_v61 import DATA_SHA, STATS_SHA
    selection = json.loads((Path(__file__).resolve().parents[3] / SELECTION).read_bytes())
    row = selection['RP8'][value['condition']]
    if (value['version'] != VERSION or value['condition'] not in range(8)
            or value['selection'] != row or value['state_atol'] != STATE_ATOL
            or value['data_manifest_sha256'] != DATA_SHA or value['statistics_sha256'] != STATS_SHA
            or spec['task_learning_role'] != 'CORELAB_EXPERT_TRAIN4_SEMANTICS_NOT_BC_V3'
            or spec.get('task_learning_teacher_mode') != 'NORMAL_TEACHER_V7'
            or spec.get('target_gr00t_policy') or spec.get('task_learning_control')
            or spec.get('sla_pick_place_v5') or spec.get('sla_five_object_hold_v5')):
        raise ValueError('RP8_FIXED_TRAIN_ORACLE_NOT_STUDENT_OR_PRODUCTION')
    original = pinned(value['original_spec'])
    if (original['registered_root'] != str(Path(value['original_spec']['path']).parent)
            or original['model_path'] != spec['model_path']
            or original['task_phase_source_from_target'] != spec['task_phase_source_from_target']
            or original['task_learning_scene_reference'] != spec['task_learning_scene_reference']
            or original['task_learning_teacher_mode'] != spec['task_learning_teacher_mode']
            or original['task_learning_start_readiness'] != spec['task_learning_start_readiness']
            or original['task_command'] != spec['task_command']
            or spec['WBC_changes'] is not False or spec['physics_dt_s'] != .002
            or value['action']['sha256'] != digest(value['action']['path'])
            or value['anchor_record']['sha256'] != digest(value['anchor_record']['path'])):
        raise ValueError('RP8_ORIGINAL_SCENE_FRAME_COMMAND_AND_IMMUTABLE_PACKET')
    return value


def endpoint(spec):
    return configuration(spec)['selection']['native_step'] + 250


def protocol(spec):
    from . import task_learning_grasp_v1 as task
    value = configuration(spec)
    return dict(task.protocol(task.CONSTRUCTION_ROLE), schema=VERSION,
        prefix_atoms=10, policy_queries=value['selection']['anchor'] + 1,
        final_hold_atoms=0, experiment='LABEL_ORACLE_NOT_AUTONOMOUS_STUDENT',
        expert_prefix_replay=True, reference_TTL_s=.4, reference_renewal_max_s=.35,
        full_controller_preview_retained=True)


@lru_cache(maxsize=8)
def _archived_scene(path, pin):
    """Check archived scene custody once at the immutable transfer boundary."""
    from .stable_task_scene_v1 import pinned
    from .storage import digest
    from .corelab_expert_semantics_v3 import native_scene_admission as original_admission
    from .task_learning_scene_v1 import MEMBER
    original = pinned(dict(path=path, sha256=pin))
    lease = json.loads((Path(original['registered_root'])/'control/LEASE.json').read_bytes())
    builder = original['source_files'][MEMBER]
    if (lease.get('status') != 'CLOSED' or lease.get('worker_exit_code') != 0
            or lease.get('spec_sha256') != pin or lease.get('source_sha') != original['source_sha']
            or digest(Path(original['source_root'])/MEMBER) != builder):
        raise ValueError('RP8_ARCHIVED_CLOSED_PRODUCER_AND_ACTUAL_BUILDER_CUSTODY')
    return original_admission(original), builder


def native_scene_admission(spec):
    from .stable_task_scene_v1 import pinned
    from .task_learning_scene_v1 import validate_static_provenance
    value = configuration(spec)
    original = pinned(value['original_spec'])
    proof, builder = _archived_scene(value['original_spec']['path'], value['original_spec']['sha256'])
    validate_static_provenance(spec, proof, expected_role=original['task_learning_role'],
                               verified_builder_sha256=builder)
    panel = pinned(dict(path=spec['stable_task_panel'], sha256=spec['stable_task_panel_sha256']))
    matches = [r for r in panel['assignments'] if r['native_run'] == Path(spec['registered_root']).name]
    if (panel['schema'] != VERSION + '_PANEL' or panel['source_sha'] != spec['source_sha']
            or len(matches) != 1 or matches[0]['configuration'] != value
            or proof['layout'] != value['selection']['layout']):
        raise ValueError('RP8_NEW_UNIQUE_REGISTERED_CONDITION_AND_ORIGINAL_TRAIN_PROVENANCE')
    return proof


def anchor_comparison(actual, reference, history, base, record):
    """The same clock/state/reference must exist before compiling the oracle."""
    import numpy as np
    errors = {}
    for name, current in dict(tcp=actual['tcp'], qpos=actual['qpos'], qvel=actual['qvel'],
            reference=reference, history=history['actual_state_history'],
            past_references=history['past_consumed_EEF10_BASE7'], base=base).items():
        current, old = np.asarray(current), np.asarray(record[name])
        if current.shape != old.shape or not np.isfinite(current).all():
            raise ValueError('RP8_ANCHOR_SHAPE_' + name)
        errors[name] = float(np.max(np.abs(current - old)))
    return dict(passes=actual['native_step'] == record['native_step']
                and all(e <= STATE_ATOL for e in errors.values()),
                absolute_component_atol=STATE_ATOL, errors=errors,
                actual_native_step=actual['native_step'], recorded_native_step=record['native_step'],
                hidden_state_restored=False, actual_original_prefix_replayed=True,
                state_reset=False, label_reanchored=False)


def continue_from_start(adapter, report, samples, windows, reset_cycle, teacher, observe):
    import copy
    import numpy as np
    from . import task_learning_grasp_v1 as task, stable_task_prefix_v1 as consumer
    from .storage import digest, jsonable, write_json
    from .stable_task_closed_loop_v1 import _link_replacement
    from .eef_stream_v2 import expert_program
    from .sla_dense_policy_v5 import compile_proposal
    from ..sla_adapter.action_contract_v61 import final_plan, consumed_mask, STATS_SHA
    value = configuration(adapter.spec)
    anchor = value['selection']['anchor']
    record = json.loads(Path(value['anchor_record']['path']).read_bytes())
    from .stable_task_data_v1 import capture_input
    cycles, decisions, masks, captures = [reset_cycle], [], [], []
    previous_grip = 1.
    for index, step in enumerate(consumer.observation_steps(adapter.spec)):
        task.need(step == samples[-1]['native_step'] == adapter.observe()['step'], 'RP8_REAL_PREFIX_CLOCK')
        observe(samples[-1])
        captures.append(capture_input(adapter, language=task.LANGUAGE, task_score=task.score_contract()))
        actual = samples[-1]
        reference = adapter.reference_history[-1]['pose']
        if index < anchor:
            program, proposal = teacher.decide(actual, reference, adapter.base_pose_hold)
            original_proposal = copy.deepcopy(proposal)
            program = expert_program(program, previous_grip)
            role = 'ACTUAL_ORIGINAL_EXPERT_PREFIX_NOT_ORACLE'
            adapter.task_learning_stage = teacher.stage
        else:
            from .fusion_live_skill_segment import _history
            comparison = anchor_comparison(actual, reference, _history(samples,windows),
                                            adapter.base_pose_hold, record)
            write_json(adapter.output / 'RP8_STATE_RECONSTRUCTION.private.json', comparison)
            task.need(comparison['passes'], 'RP8_STATE_RECONSTRUCTION_NOT_ESTABLISHED:' + repr(comparison))
            program, proposal = compile_proposal(value['action']['path'], value['action']['sha256'],
                source_from_target=adapter.spec['task_phase_source_from_target'],
                query_state10=record['state_eef10'], actual_TCP=actual['tcp'],
                consumed_reference_TCP=reference, base_hold=adapter.base_pose_hold)
            original_proposal = None
            role = 'FROZEN_FP32_LABEL_THROUGH_RAW_STUDENT_COMPILER'
            adapter.task_learning_stage = value['selection']['stage']
        # This is the post-planner program; the old proposal is retained apart.
        query_id = adapter.spec['registered_root'] + ':' + str(index)
        def plan(valid):
            return dict(final_plan(program, query_id=query_id, native_step=step,
                planner_sha256=digest(Path(__file__).with_name('eef_stream_v2.py')),
                adapter_sha256=digest(Path(__file__).with_name('sla_dense_policy_v5.py')),
                normalizer_sha256=STATS_SHA, full_H40_valid=valid),
                reference_before_query=np.asarray(reference).tolist(), role=role,
                original_preplanner_proposal=original_proposal)
        path = adapter.output / ('RP8_FINAL_PLAN_' + str(index).zfill(3) + '.private.json')
        # Full geometric construction has passed; native certification is still
        # pending. A failed install must not admit all40 as legal supervision.
        pending = dict(plan(False), status='PENDING_ORIGINAL_NATIVE_FULL_H40_CERTIFICATE')
        write_json(path.with_suffix('.INTENT.private.json'), pending)
        current, samples, windows = consumer.run_prefix(adapter, program, index=index,
            past_samples=samples, past_windows=windows, sample_observer=observe)
        _link_replacement(adapter, cycles[-1], current)
        cycles.append(current)
        sidecar = plan(True)
        events = [dict(kind='NATIVE_ATOM', k=k, origin_query_id=query_id,
            command_id=sidecar['command_ids'][k], native_start_step=w['native_start_step'],
            native_end_step=w['native_end_step'], actual_ticks=w['actual_ticks'],
            solver_policies_with_exact_installed_target=bool(w['solver_policies_with_exact_installed_target']),
            gripper_command_consumed=w['gripper']['command_consumed']) for k,w in enumerate(current['windows'])]
        sidecar['consumed_mask'] = consumed_mask(sidecar, events)
        sidecar['consumption_events'] = events
        write_json(path, sidecar)
        masks.append(dict(plan_sha256=sidecar['plan_sha256'], supervision=sum(sidecar['supervision_mask']),
                          consumed=sum(sidecar['consumed_mask']), role=role))
        decisions.append(dict(native_step=step, role=role, path=str(path), sha256=digest(path)))
        previous_grip = program.decoded_H40[9,9]
    if hasattr(adapter, '_body_memory_capture'):
        adapter._body_memory_capture.finish(captures, observed_task_success=False, confirmed=[])
    result = dict(copy.deepcopy(report), schema=VERSION + '_NATIVE_ACTUAL',
        actual_spec_phase=task.PHASE, native_endpoint_step=endpoint(adapter.spec),
        complete_task_atoms=len(windows), solver_target_confirmed_atoms=len(windows),
        gripper_confirmed_atoms=len(windows), generated_cycles=cycles, windows=windows,
        RP8_condition=value['selection'], state_reconstruction=comparison, oracle_P10_consumed=True,
        full_H40_checked=True, final_plan_masks=masks, decisions=decisions,
        expert_prefix_queries=anchor, oracle_queries=1, SLA_model_queries=0, model_updates=0,
        oracle_is_student=False, autonomous_student_success=False, task_condition_success=False,
        expert_training_eligible=False, full_task_success=False, target_task_success=False)
    write_json(adapter.output / 'RP8_ACTUAL.private.json', result)
    return jsonable(result)


def complete_consumption(spec, result):
    from . import stable_task_prefix_v1 as consumer
    value = configuration(spec)
    cycles = result.get('generated_cycles', [])
    queries = value['selection']['anchor'] + 1
    if (result.get('schema') != VERSION + '_NATIVE_ACTUAL' or len(cycles) != queries + 1
            or result.get('oracle_P10_consumed') is not True
            or result.get('state_reconstruction', {}).get('passes') is not True
            or result.get('autonomous_student_success') is not False
            or result.get('SLA_model_queries') != 0 or result.get('expert_training_eligible') is not False):
        return False
    for position, row in enumerate(cycles):
        cycle = consumer.cycle(spec, position - 1)
        rows = row.get('windows', [])
        offset = (cycle['start_step'] - 2503) // 25
        if (row['cycle'] != cycle or not (len(rows) == row.get('complete_suffix_atoms') == cycle['atoms'])
                or row.get('solver_target_confirmed_atoms') != cycle['atoms']
                or row.get('gripper_confirmed_atoms') != cycle['atoms']
                or rows != result['windows'][offset:offset+cycle['atoms']]):
            return False
        for k, w in enumerate(rows):
            if (w['native_start_step'] != cycle['start_step'] + 25*k
                    or w['native_end_step'] != cycle['start_step'] + 25*(k+1)
                    or w['actual_ticks'] != 25 or not w.get('solver_policies_with_exact_installed_target')
                    or w['gripper']['command_consumed'] is not True):
                return False
        if position < queries and row.get('cancellation_status') != 'SUPERSEDED_BY_ORIGINAL_NEW_REFERENCE_ACK':
            return False
    return all(m['supervision'] == 40 and m['consumed'] == 10 for m in result['final_plan_masks'])
