"""Pure readback of an explicitly pinned CLOSED full-posture/source-rig query.

This is a historical evidence reader, not a live scope/pause validator. It never
checks or substitutes today's clock, loads weights, decodes an action, renders,
constructs FK, registers work, or grants execution. The V1 archive is unchanged.
"""
import io
from pathlib import Path

import numpy as np

from scripts.pi0292.relay_protocol import digest as document_hash, unseal, uuid_string
from scripts.pi0294.task_clock import require_pause
from scripts.pi0294.task_state import parse_state
from .fusion_observation import transform_hash
from .fusion_paused_observation import paused_inference_payload
from .fusion_paused_query_archive import (
    CHECKPOINT, INVENTORY, PIN_KEYS as V1_PIN_KEYS, _bytes, _hash, _read, _same_number)
from . import fusion_source_posture_paused_query_v3 as posture
from . import fusion_source_posture_qualification_contract as ancestor

SCHEMA = 'WM02_FUSION_SOURCE_POSTURE_QUERY_ARCHIVE_V3'
ENDPOINT, END_ATOM, ATOMS = 16403, 500, 556
PIN_KEYS = V1_PIN_KEYS | {'episode_sha256', 'physical_report_sha256',
                         'query_report_sha256', 'source_camera_identity_sha256', 'noise_sha256'}
# Only directly relevant immutable code is read; no full-history/weight scan.
SOURCE_MEMBERS = (
    'src/go2arx5_vla/world_model/fusion_source_posture_paused_query_v3.py',
    'src/go2arx5_vla/world_model/fusion_source_posture_qualification_native.py',
    'src/go2arx5_vla/world_model/fusion_trajectory_qualification_native.py',
    'src/go2arx5_vla/world_model/fusion_source_camera_capture.py',
    'src/go2arx5_vla/world_model/fusion_source_camera_rig.py',
    'src/go2arx5_vla/world_model/fusion_paused_inference.py',
    'scripts/wm02/infer_fusion_paused_vla.py', 'scripts/pi0294/resident.py')


def _need(ok, reason):
    if not ok:
        raise ValueError('SOURCE_POSTURE_QUERY_ARCHIVE_V3_' + reason)


def _closed(spec, pin):
    """Read saved terminal records, without current registry/site admission."""
    root = Path(spec['registered_root'])
    _need(Path(spec['lease']) == root/'control/LEASE.json', 'ORIGINAL_CLOSED_LEASE_PATH')
    lease = _read(spec['lease'])
    worker = _read(root/'WORKER_EXIT.private.json')
    _need(lease.get('status') == 'CLOSED' and lease.get('spec_sha256') == pin
          and lease.get('run_uuid') == root.name
          and type(lease.get('worker_exit_code')) is int and lease['worker_exit_code'] == 0
          and type(worker.get('exit_code')) is int and worker['exit_code'] == 0
          and worker.get('failure') is None, 'ACTUAL_BOTH_WORKERS_CLOSED_ZERO')
    return dict(lease=lease, worker=worker, spec_sha256=pin)


def _physical(report, episode):
    _need(report.get('schema') == posture.REPORT_SCHEMA and report.get('episode') == episode
          and report.get('actual_spec_phase') == posture.PHASE
          and report.get('physical_recipe_phase') == ancestor.PHASE
          and report.get('query_stage') == 'NOT_YET_CALLED'
          and report.get('native_endpoint_step') == ENDPOINT
          and report.get('complete_task_atoms') == report.get('solver_target_confirmed_atoms')
              == report.get('gripper_confirmed_atoms') == ATOMS
          and report.get('all_40_knots_consumed') is True
          and report.get('raw_knot_consumed') == [True]*40
          and report.get('actual_VLA_commands') == report.get('VLA_commands') == report.get('VLA_inferences') == 0
          and report.get('WBC_changes') is False and report.get('physics_dt_s') == .002
          and report.get('full_task_success') is False, 'FULL_CALIBRATION_BEFORE_QUERY_NOT_VLA_EXECUTION')
    cancel = report.get('operator_cancel') or {}
    _need(cancel.get('kind') == 'operator_cancel' and cancel.get('mode') == 'HOLD'
          and cancel.get('native_step') == ENDPOINT, 'ORIGINAL_ENDPOINT_CANCEL_HOLD')
    adaptation = report.get('adaptation') or {}; clock = adaptation.get('clock') or {}
    knots = clock.get('knot_indices')
    _need(adaptation.get('raw_knots_retained') == adaptation.get('adapted_knots_retained') == 40
          and clock.get('end_atom') == END_ATOM and clock.get('time_stretch_factor') == 2
          and clock.get('atom_dt_s') == .05 and clock.get('native_dt_s') == .002
          and clock.get('native_ticks_per_atom') == 25
          and isinstance(knots, list) and len(knots) == 40
          and all(type(k) is int and 0 <= k <= END_ATOM for k in knots)
          and all(a < b for a,b in zip(knots,knots[1:])) and knots[-1] == END_ATOM,
          'FULL_FORTY_KNOT_TIME2_CLOCK')
    windows = report.get('windows') or []; elapsed = set()
    _need(len(windows) == ATOMS, 'EVERY_PHYSICAL_ATOM_RETAINED')
    for i,row in enumerate(windows):
        grip = row.get('gripper') or {}; sent = grip.get('submitted') or {}
        memory = grip.get('consumed_wrapper') or {}; values = memory.get('values') or {}
        _need(row.get('native_start_step') == 2503+25*i and row.get('native_end_step') == 2503+25*(i+1)
              and row.get('actual_ticks') == 25 and bool(row.get('solver_policies_with_exact_installed_target'))
              and grip.get('command_consumed') is True
              and type(sent.get('sequence')) is int and sent['sequence'] == values.get('sequence') == i
              and memory.get('status') == 'ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY'
              and values.get('steps') == row['native_end_step']
              and _same_number(values.get('opening'),sent.get('opening'))
              and _same_number(values.get('expiry'),sent.get('expiry')),
              'ORIGINAL_POSE_AND_GRIPPER_CONSUMPTION:'+str(i))
        expected = i-48 if i >= 48 else None
        _need(row.get('trajectory_elapsed_atom') == expected, 'ACTUAL_CONTIGUOUS_PROGRAM_CLOCK')
        if expected is not None: elapsed.add(expected)
    _need(set(knots) <= elapsed, 'ALL_FORTY_KNOTS_ACTUALLY_CONSUMED')


def _native(native, report):
    trace = native.get('trace_audit') or {}; exits = native.get('owned_process_exits') or []
    _need(native.get('status') == 'NATIVE_STAGE_PASSED' and native.get('phase') == posture.PHASE
          and native.get('actual_target_integrations') == ENDPOINT and native.get('reached_exact_pause') is True
          and native.get('shutdown_clean') is True and native.get('failure') is None
          and native.get('physics_stops') == [] and native.get('original_joint_or_warning_samples') == 0
          and native.get('native_outer_guard_rejection_count') == 0
          and native.get('native_outer_guard_rejections') == {} and native.get('trace_audit_error') is None
          and native.get('reference_qualification') == report, 'GENUINE_NATIVE_PASS_NOT_A_SAVED_GPU_DRAW_ALONE')
    _need(len(exits) == 4 and {e.get('name') for e in exits} == {'sim','planner','controller','solver'}
          and all(type(e.get('exit_code')) is int and e['exit_code'] == 0 for e in exits),
          'ALL_FOUR_ORIGINAL_CHILDREN_ZERO')
    _need(trace.get('status') == 'RECORDED_TRACE_CHECKS_PASS' and trace.get('errors') == []
          and trace.get('missing') == [] and (trace.get('writer') or {}).get('dropped') == 0
          and trace.get('native_step_records') == ENDPOINT, 'COMPLETE_NATIVE_TRACE_ZERO_DROPS')


def read_archive(native_spec_path, gpu_spec_path, *, pins):
    """Read the declared full1 V3 archive; expired deadlines remain metadata.

    Exact ``PIN_KEYS`` extends V1's fourteen identity fields with the episode,
    physical report, query report, source-camera audit and recorded noise SHAs.
    All pins are caller-supplied; no latest/best search or live grant is used.
    """
    _need(isinstance(pins,dict) and set(pins) == PIN_KEYS, 'COMPLETE_EXTERNAL_PIN_SCHEMA')
    _need(all(_hash(pins[k]) for k in PIN_KEYS-{'source_sha','query_uuid','seed'})
          and _hash(pins['source_sha'],40) and type(pins['seed']) is int and 0 <= pins['seed'] < 2**63,
          'EXACT_PIN_TYPES')
    uuid_string(pins['query_uuid'])
    _need(pins['checkpoint_manifest_sha256'] == CHECKPOINT
          and pins['checkpoint_file_inventory_sha256'] == INVENTORY
          and pins['source_camera_identity_sha256'] == posture.SOURCE_CAMERA_IDENTITY,
          'FROZEN_MODEL_PROCESSOR_AND_SOURCE_RIG_IDENTITY')
    npth,gpth = Path(native_spec_path),Path(gpu_spec_path)
    n,g = _read(npth,pins['native_spec_sha256']),_read(gpth,pins['gpu_spec_sha256'])
    nr,gr = Path(n['registered_root']),Path(g['registered_root'])
    _need(npth == nr/'EXECUTION_SPEC.json' and gpth == gr/'EXECUTION_SPEC.json' and nr != gr
          and all(p.is_absolute() and not p.is_symlink() and p.resolve() == p for p in (nr,gr))
          and not nr.is_relative_to(gr) and not gr.is_relative_to(nr), 'DISTINCT_CANONICAL_ARCHIVE_ROOTS')
    _need(n['source_sha'] == g['source_sha'] == pins['source_sha'] and n['source_root'] == g['source_root']
          and all(n.get(k) == g.get(k) and n.get(k) is not None for k in
                  ('owner','host','uid','campaign_root','pilot1_binding','pilot1_binding_sha256')),
          'COMMON_ORIGINAL_OWNER_AND_IMMUTABLE_SOURCE')
    _need(n.get('phase') == posture.PHASE and n.get('physics') is True and n.get('max_ticks') == ENDPOINT
          and n.get('physics_dt_s') == .002 and n.get('WBC_changes') is False
          and g.get('physics') is False and g.get('kind') == 'FROZEN_VLA_LIVE_PAUSED_QUERY'
          and g.get('profile') == 'wm02-fusion-vla-shadow' and g.get('device') == 'cuda'
          and g.get('execution_site') == 'CORE_FUSION_NEURAL'
          and g.get('inference_mode') == 'ACTUAL_LIVE_PAUSED_VLA_INFERENCE'
          and g.get('numeric_mode') == 'ORIGINAL_FP32_WEIGHTS_BF16_AUTOCAST'
          and g.get('observation_mode') == 'ACTUAL_OWNED_TARGET_LIVE_PAUSED_CAPTURE'
          and g.get('snapshot_binding_mode') == posture.PAIR_MODE
          and g.get('max_queries') == 1 and g.get('seed') == pins['seed']
          and g.get('query_uuid') == pins['query_uuid']
          and g.get('GR00T_updates') == g.get('WM_updates') == g.get('Actor_updates') == 0
          and g.get('checkpoint_manifest_sha256') == CHECKPOINT
          and document_hash(g['checkpoint_files']) == INVENTORY, 'EXACT_RECORDED_V3_ONE_QUERY_RECIPE')
    inventory = g['checkpoint_files']
    _need(all(not Path(k).is_absolute() and '..' not in Path(k).parts and _hash(v) for k,v in inventory.items())
          and any(k.startswith('processor/') for k in inventory), 'SAVED_PROCESSOR_IN_FULL_PINNED_INVENTORY')
    files = dict(n['source_files'])
    _need(all(k not in files or files[k] == v for k,v in g['source_files'].items()), 'SAME_CAPSULE_MEMBER_IDENTITIES')
    files.update(g['source_files'])
    for name in SOURCE_MEMBERS: _bytes(Path(n['source_root'])/name,files[name])
    closed = dict(native=_closed(n,pins['native_spec_sha256']),gpu=_closed(g,pins['gpu_spec_sha256']))
    episode = _read(n['episode_manifest'],pins['episode_sha256'])
    posture.episode_contract(episode)
    _need(n.get('episode_manifest_sha256') == pins['episode_sha256'] and episode['run'] == str(nr)
          and episode['planned_end_atom'] == END_ATOM and posture.endpoint(episode) == ENDPOINT
          and episode['capture_camera_recipe'] == n.get('posture_query_camera_recipe') == posture.SOURCE_CAMERA_RECIPE
          and episode['source_camera_identity_sha256'] == n.get('source_camera_identity_sha256')
              == pins['source_camera_identity_sha256'], 'FULL1_SOURCE_POSTURE_SOURCE_RIG_EPISODE')
    pair_path = nr/'QUERY_PAIR.private.json'
    _need(n.get('paused_query_pair') == g.get('paused_query_pair') == str(pair_path)
          and n.get('paused_query_pair_sha256') == g.get('paused_query_pair_sha256') == pins['pair_sha256'],
          'EXACT_REGISTERED_PAIR_BYTES')
    pair = _read(pair_path,pins['pair_sha256'])
    from .fusion_paused_query_bridge import KEYS
    _need(set(pair) == KEYS | posture.PAIR_EXTRA_KEYS and pair.get('schema') == posture.PAIR_SCHEMA
          and pair.get('native_root') == str(nr) and pair.get('gpu_root') == str(gr)
          and pair.get('native_run_id') == 'native-01' and pair.get('native_boundary_step') == ENDPOINT
          and type(pair.get('capture_index')) is int and pair['capture_index'] == 0
          and all(pair.get(k) == pins[k] for k in ('query_uuid','seed','source_sha','task_sha256','frame_map_sha256','checkpoint_manifest_sha256'))
          and pair.get('producer_phase') == posture.PHASE and pair.get('physical_recipe_phase') == ancestor.PHASE
          and pair.get('source_posture_episode_sha256') == pins['episode_sha256']
          and pair.get('posture_source_observation_sha256') == ancestor.POSTURE_OBSERVATION
          and pair.get('geometry_scale') == 1. and pair.get('time_stretch_factor') == 2
          and pair.get('max_queries') == 1 and pair.get('native_integrations_by_bridge') == 0
          and pair.get('returned_VLA_action_installations') == 0
          and pair.get('camera_recipe') == posture.SOURCE_CAMERA_RECIPE
          and pair.get('source_camera_identity_sha256') == pins['source_camera_identity_sha256'], 'EXACT_PAIR_V2_NOT_V1_OR_OLD_GEOMETRY')
    _need(g.get('producer_native_spec') == str(npth) and g.get('producer_native_spec_sha256') == pins['native_spec_sha256']
          and n.get('paused_query_gpu_spec') == str(gpth), 'ACYCLIC_NATIVE_GPU_SPEC_BACKREFERENCES')
    path = nr/'native-01/PAUSED_OBSERVATIONS'/pins['query_uuid']/'PAUSED_OBSERVATION.private.json'
    _need(g.get('recorded_observation') == str(path), 'EXACT_CAPTURE_PATH')
    record = _read(path,pins['observation_sha256']); body = unseal(record)
    payload = paused_inference_payload(record); shadow = unseal(body['original_shadow_ABI'])
    posture.validate_capture_camera(pair,shadow)
    provenance = shadow['provenance']
    _need(provenance.get('processor') == 'UNCHANGED_FROZEN_SOURCE_PROCESSOR'
          and provenance.get('embodiment') == 'new_embodiment' and provenance.get('embodiment_id') == 10,
          'UNCHANGED_SOURCE_PROCESSOR_CATEGORY10_NOT_NEW_EMBODIMENT')
    index = body['capture']['index']
    names = {'STATE.private.bin','BOUND_SHADOW_ABI.private.json',f'images/{index:06d}-calibration.json',
             *(f'images/{index:06d}-{k}.png' for k in ('body_rgb','wrist_rgb')),
             *(f'native-camera/images/{index:06d}-{k}' for k in ('body_rgb.png','wrist_rgb.png','calibration.json'))}
    _need(set(body['files']) == names and not (path.parent/'CAPTURE_FAILED.private.json').exists(), 'EXACT_EIGHT_SOURCE_RIG_CAPTURE_MEMBERS')
    retained = {}
    for name,pin in body['files'].items():
        member = path.parent/name
        _need(member.resolve().is_relative_to(path.parent.resolve()), 'CAPTURE_MEMBER_INSIDE_ROOT')
        retained[name] = _bytes(member,pin)
    import json
    _need(json.loads(retained['BOUND_SHADOW_ABI.private.json']) == body['original_shadow_ABI'], 'EXACT_NESTED_SHADOW_BYTES')
    packet = parse_state(retained['STATE.private.bin'])
    _need(packet['step'] == body['native_step'] == ENDPOINT and abs(packet['time_s']-ENDPOINT*.002) <= 1e-8
          and packet['time_s'] == body['native_time_s'] and index == 0
          and body['query_uuid'] == pins['query_uuid'] and body['task_sha256'] == pins['task_sha256']
          and n.get('paused_target_task_sha256') == g.get('paused_task_sha256') == pins['task_sha256']
          and body['world']['execution_spec_sha256'] == pins['native_spec_sha256']
          and body['world']['registered_root'] == str(nr) and body['world']['native_output'] == str(nr/'native-01')
          and body['world']['scene_sha256'] == pair['scene_sha256']
          and shadow['source_policy_view']['transform_sha256'] == n.get('paused_frame_map_sha256') == pins['frame_map_sha256']
          and transform_hash(n['paused_source_from_target']) == pins['frame_map_sha256'], 'ABSOLUTE_16403_PACKET_TASK_FRAME_AND_PRODUCER')
    _bytes(nr/'native-01/state.bin',body['files']['STATE.private.bin'])
    _bytes(n['model_path'],pair['scene_sha256'])
    _bytes(nr/'native-01/quota.binding',body['world']['quota_binding_sha256'])
    _bytes(nr/'native-01/step_quota',body['step_quota_sha256'])
    _need(_read(path.parent/'PAIR_SNAPSHOT_READY.private.json') == dict(schema=posture.PAIR_SCHEMA,
        pair_sha256=pins['pair_sha256'],producer_native_spec_sha256=pins['native_spec_sha256'],
        observation_sha256=pins['observation_sha256'],query_uuid=pins['query_uuid']), 'ORIGINAL_READY_MARKER')
    result = _read(gr/'fit-01/RESULT.private.json',pins['gpu_result_sha256'])
    expected_binding = dict(query_uuid=body['query_uuid'],snapshot_uuid=body['snapshot_uuid'],
        observation_sha256=pins['observation_sha256'],sealed_body_sha256=record['sha256'],task_sha256=body['task_sha256'],
        task=body['task'],world=body['world'],native_step=body['native_step'],native_time_s=body['native_time_s'],
        files=body['files'],renderer=body['renderer'],input_mode=body['input_mode'],mapped_payload_sha256=document_hash(payload),
        target_camera_provenance=provenance,source_policy_view={k:v for k,v in shadow['source_policy_view'].items() if k != 'payload'},
        snapshot_binding_mode=posture.PAIR_MODE,pause_before=body['pause_before'],pause_after_capture=body['pause_after'],
        native_pause_after_query='NOT_CHECKED_BY_GPU_PROCESS',native_caller_must_verify_still_paused=True,
        native_execution_authorized=False,target_embodiment_qualified=False,target_task_qualified=False,
        online_closed_loop=False,source_outcomes_supplied=False,producer_phase=posture.PHASE,physical_recipe_phase=ancestor.PHASE,
        capture_condition='FULL_SOURCE_POSTURE_AFTER_CANCEL_HOLD_AND_TRACE_SEAL',capture_camera_recipe=posture.SOURCE_CAMERA_RECIPE,
        source_camera_identity_sha256=pins['source_camera_identity_sha256'],returned_VLA_action_installations=0,camera_equivalence=False)
    _need(result.get('observation_binding') == expected_binding
          and all(result.get(k) == expected_binding[k] for k in ('producer_phase','physical_recipe_phase','capture_condition',
                    'capture_camera_recipe','source_camera_identity_sha256','returned_VLA_action_installations','camera_equivalence')),
          'EXACT_FULL_GPU_INPUT_BINDING')
    _need(result.get('schema') == 'WM02_FUSION_LIVE_PAUSED_INFERENCE_RESULT_V1'
          and result.get('status') == 'REAL_FROZEN_VLA_LIVE_PAUSED_QUERY_DONE'
          and result.get('execution_spec_sha256') == pins['gpu_spec_sha256']
          and all(result.get(k) == pins[k] for k in ('source_sha','query_uuid','seed','checkpoint_manifest_sha256',
                    'checkpoint_file_inventory_sha256','pair_sha256','action_sha256'))
          and result.get('recorded_observation_sha256') == pins['observation_sha256']
          and result.get('producer_native_spec_sha256') == pins['native_spec_sha256']
          and result.get('actual_model_loads') == result.get('actual_VLA_inferences') == 1
          and result.get('native_integrations') == result.get('GR00T_updates') == result.get('WM_updates') == result.get('Actor_updates') == 0
          and result.get('inference_mode') == g['inference_mode'] and result.get('numeric_mode') == g['numeric_mode']
          and result.get('source_frame') == payload['frame'] and result.get('source_TCP') == payload['tcp']
          and all(result.get(k) is False for k in ('native_execution_authorized','online_closed_loop',
                    'target_embodiment_qualified','target_task_qualified','reference_TTL_rebased','returned_action_is_live_reference'))
          and result.get('simulated_time_advanced_s') == 0, 'ONE_FROZEN_DRAW_WITH_NO_EXECUTION_OR_TTL_GRANT')
    precision = result.get('precision') or {}; noise = result.get('noise') or {}
    _need(precision.get('mode') == 'ORIGINAL_E0_BF16_AUTOCAST_FP32_MATMUL_HIGHEST'
          and precision.get('cuda_matmul_allow_tf32') is False and precision.get('bf16_autocast') is True
          and type(precision.get('training_flag_on_entry')) is bool
          and noise == dict(shape=[1,40,132],dtype='torch.float32',sha256=pins['noise_sha256']), 'ORIGINAL_PRECISION_AND_COMPLETE_RECORDED_NOISE_IDENTITY')
    with np.load(io.BytesIO(_bytes(gr/'fit-01/REAL_VLA_ACTION.npz',pins['action_sha256'])),allow_pickle=False) as arrays:
        _need(set(arrays.files) == {'physical_H40','normalized'}, 'ONLY_ORIGINAL_TWO_RAW_ACTION_ARRAYS')
        raw,normalized = arrays['physical_H40'],arrays['normalized']
        _need(raw.shape == (40,10) and normalized.shape == (1,40,132)
              and raw.dtype.kind == 'f' and normalized.dtype == np.dtype('float32')
              and np.isfinite(raw).all() and np.isfinite(normalized).all()
              and result.get('physical_H40_shape') == [40,10] and result.get('normalized_shape') == [1,40,132]
              and result.get('normalized_outside_unit_components') == int(np.count_nonzero(np.abs(normalized[...,:10]) > 1)),
              'COMPLETE_FINITE_UNCLIPPED_ACTION_AND_NORMALIZED_ARRAYS')
        raw = np.frombuffer(raw.tobytes(),dtype=raw.dtype).reshape(raw.shape)
        normalized = np.frombuffer(normalized.tobytes(),dtype=normalized.dtype).reshape(normalized.shape)
    physical_path = nr/'native-01/TRAJECTORY_QUALIFICATION.private.json'
    physical = _read(physical_path,pins['physical_report_sha256']); _physical(physical,episode)
    report = _read(nr/'native-01/SOURCE_POSTURE_PAUSED_QUERY.private.json',pins['query_report_sha256'])
    expected_report = dict(physical,query_stage='ONE_FROZEN_GPU_QUERY_RETURNED_AND_PAUSE_REVALIDATED',
        physical_report_path=str(physical_path),physical_report_sha256=pins['physical_report_sha256'],paused_query=report.get('paused_query'),
        actual_VLA_inferences=1,VLA_inferences=1,VLA_queries=1,VLA_action_installations=0,native_process_VLA_inferences=0,
        native_process_Processor_decodes=0,GPU_action_decode_contract='ONE_ORIGINAL_PROCESSOR_DECODE_INSIDE_RESIDENT_GENERATE',
        additional_action_decodes=0,query_wall_s=report.get('query_wall_s'),query_CPU_s=report.get('query_CPU_s'),
        camera_equivalence=False,full_policy_input_distribution_qualified=False,returned_action_is_live_reference=False,
        online_VLA=False,online_closed_loop=False,native_integrations_by_query=0,native_stage_RESULT_json_required=True)
    for k in ('original_Processor_decodes','Processor_decodes'): expected_report.pop(k,None)
    _need(report == expected_report and all(type(report.get(k)) in (int,float) and np.isfinite(report[k]) and report[k] >= 0
              for k in ('query_wall_s','query_CPU_s')), 'SAVED_PHYSICAL_TO_COMBINED_REPORT_UNCHANGED')
    native = _read(nr/'native-01/RESULT.json',pins['native_result_sha256']); _native(native,report)
    _need(native.get('source_sha') == pins['source_sha'] and native.get('scene_sha256') == pair['scene_sha256'], 'NATIVE_RESULT_SOURCE_AND_SCENE')
    query = report['paused_query']; checked = query.get('query_result') or {}; pause = checked.get('original_pause_revalidation') or {}
    _need(query.get('schema') == posture.PAIR_SCHEMA and query.get('query_uuid') == pins['query_uuid']
          and query.get('pair_sha256') == pins['pair_sha256'] and query.get('gpu_spec_sha256') == pins['gpu_spec_sha256']
          and query.get('capture_sha256') == pins['observation_sha256'] and query.get('native_integrations_by_bridge') == 0
          and query.get('native_execution_authorized') is False and query.get('online_action_execution') is False
          and checked.get('schema') == result['schema'] and checked.get('query_uuid') == pins['query_uuid']
          and checked.get('result_sha256') == pins['gpu_result_sha256'] and checked.get('action_sha256') == pins['action_sha256']
          and checked.get('model_result_checked') is True and checked.get('native_execution_authorized') is False
          and checked.get('original_certificate_and_ACK_still_required') is True,
          'ACTUAL_NATIVE_RECORDED_GPU_RESULT_ACTION_CAPTURE_RETURN')
    _need(pause.get('schema') == body['schema'] and pause.get('query_uuid') == pins['query_uuid']
          and pause.get('native_step') == ENDPOINT and pause.get('native_time_s') == body['native_time_s']
          and pause.get('unchanged_measured_packet_sha256') == body['files']['STATE.private.bin']
          and pause.get('native_execution_authorized') is False and pause.get('model_result_checked') is False,
          'SAME_ACTUAL_PACKET_AFTER_QUERY')
    require_pause(body['pause_after'],pause['pause_after_query'],ENDPOINT)
    return dict(schema=SCHEMA,status='PAIR_COMPLETED_AND_PAUSE_REVALIDATED',physical_H40=raw,normalized=normalized,
        payload=payload,observation=record,GPU_result=result,native_result=native,physical_report=physical,query_report=report,
        paired_return=dict(established=True,unresolved=[],recorded_proof=query),closed_workers=closed,external_pins=dict(pins),
        source_manifest_sha256=document_hash(files),checkpoint_inventory=inventory,noise_identity=noise,
        noise_bytes_retained=False,noise_recomputed=False,checkpoint_bytes_reread=False,
        reuse_original_frozen_inference_verified_inventory=True,original_deadline_epochs=dict(native=n['deadline_epoch'],gpu=g['deadline_epoch'],pair=pair['deadline_epoch']),
        historical_read=True,current_deadline_validation_performed=False,current_native_pause_checked=False,
        native_execution_authorized=False,target_action_qualified=False,target_task_success=False,
        online_closed_loop_established=False,new_queries=0,new_integrations=0,processor_decodes=0)
