"""Read an externally pinned, closed paused-query archive without a live grant.

No wall-clock/deadline substitution, live reader, model, renderer, FK, controller
or physics API is used. A failed native pair can retain a valid saved-input draw.
"""
import json
import hashlib
import io
from pathlib import Path
import re

import numpy as np

from scripts.pi0292.relay_protocol import digest as document_hash, unseal, uuid_string
from scripts.pi0294.task_clock import require_pause
from scripts.pi0294.task_state import parse_state
from .fusion_observation import transform_hash
from .fusion_paused_observation import paused_inference_payload
from .storage import digest

SCHEMA = 'WM02_FUSION_PAUSED_QUERY_ARCHIVE_V1'
CHECKPOINT = '94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38'
INVENTORY = 'f4c9486006de1a58721a1fc723361264e406ac4910ba46f14a31233c7a28fe23'
PIN_KEYS = {'native_spec_sha256','gpu_spec_sha256','gpu_result_sha256','action_sha256',
            'observation_sha256','pair_sha256','native_result_sha256','source_sha',
            'checkpoint_manifest_sha256','checkpoint_file_inventory_sha256',
            'task_sha256','frame_map_sha256','query_uuid','seed'}


def _need(ok, why):
    if not ok:
        raise ValueError('PAUSED_QUERY_ARCHIVE_'+why)


def _hash(value, length=64):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{'+str(length)+'}',value) is not None


def _same_number(a,b):
    return (type(a) in (int,float) and type(b) in (int,float)
            and np.isfinite([a,b]).all() and abs(a-b)<=1e-12)


def _bytes(path, pin=None):
    path = Path(path)
    _need(not path.is_symlink(), 'ACTUAL_REGULAR_FILE:'+path.name)
    raw=path.read_bytes()
    _need(pin is None or hashlib.sha256(raw).hexdigest()==pin, 'ACTUAL_FILE_BYTES:'+path.name)
    return raw


def _read(path, pin=None):
    return json.loads(_bytes(path,pin))


def _closed(spec, path, pin, registry):
    root = Path(spec['registered_root'])
    lease_path = root/'control/LEASE.json'
    worker_path = root/'WORKER_EXIT.private.json'
    _need(Path(spec['lease'])==lease_path, 'ORIGINAL_LEASE_PATH')
    lease, worker = _read(lease_path), _read(worker_path)
    _need(lease.get('status')=='CLOSED' and lease.get('spec_sha256')==pin
          and lease.get('run_uuid')==root.name
          and type(worker.get('exit_code')) is int
          and lease.get('worker_exit_code')==worker['exit_code'], 'CLOSED_EXACT_SPEC_AND_WORKER')
    _need(any(e.get('path')==str(root) and e.get('role')==spec['role']
              and e.get('source_sha')==spec['source_sha'] and e.get('owner')==spec['owner']
              and e.get('local_lease')==str(lease_path) for e in registry['entries']),
          'ORIGINAL_REGISTRY_MEMBER')
    return dict(lease=lease,worker=worker,spec_sha256=digest(path),
                lease_sha256=digest(lease_path),worker_sha256=digest(worker_path))


def _paired_return(native, native_closed, gpu_closed, body, pins):
    """Check recorded actual proof; never synthesize a current pause proof."""
    issues=[]
    def check(ok, reason):
        if not ok: issues.append(reason)
    report=native.get('reference_qualification') or {}
    query=report.get('paused_query') or {}
    result=query.get('query_result') or {}
    pause=result.get('original_pause_revalidation') or {}
    trace=native.get('trace_audit') or {}
    check(native_closed['worker']['exit_code']==gpu_closed['worker']['exit_code']==0
          and native_closed['worker'].get('failure') is None and gpu_closed['worker'].get('failure') is None,
          'BOTH_ORIGINAL_WORKERS_MUST_CLOSE_ZERO')
    check(native.get('status')=='NATIVE_STAGE_PASSED' and native.get('reached_exact_pause') is True
          and native.get('shutdown_clean') is True and native.get('failure') is None
          and native.get('actual_target_integrations')==3703
          and native.get('physics_stops')==[] and native.get('original_joint_or_warning_samples')==0
          and native.get('native_outer_guard_rejection_count')==0
          and native.get('native_outer_guard_rejections')=={} and native.get('trace_audit_error') is None,
          'GENUINE_NATIVE_PASS_AND_ORIGINAL_SAFETY')
    exits=native.get('owned_process_exits') or []
    check({x.get('name') for x in exits}=={'sim','planner','controller','solver'}
          and all(x.get('exit_code')==0 for x in exits), 'ALL_FOUR_ORIGINAL_CHILDREN_ZERO')
    check(trace.get('status')=='RECORDED_TRACE_CHECKS_PASS' and trace.get('errors')==[]
          and trace.get('missing')==[] and (trace.get('writer') or {}).get('dropped')==0
          and trace.get('native_step_records')==3703, 'COMPLETE_TRACE_ZERO_DROPS')
    check(report.get('native_endpoint_step')==3703
          and report.get('complete_task_atoms')==report.get('solver_target_confirmed_atoms')
              ==report.get('gripper_confirmed_atoms')==48
          and report.get('VLA_action_installations')==report.get('WM_corrections')==0
          and report.get('task_executed') is False and report.get('full_task_success') is False,
          'HOLD_ONLY_COMPLETE_CAPTURE_QUERY_REPORT')
    windows=report.get('windows') or []
    check(len(windows)==48,'EVERY_ORIGINAL_PREFIX_WINDOW_RETAINED')
    for i,row in enumerate(windows):
        grip=row.get('gripper') or {};sent=grip.get('submitted') or {}
        memory=grip.get('consumed_wrapper') or {};values=memory.get('values') or {}
        check(row.get('native_start_step')==2503+25*i and row.get('native_end_step')==2503+25*(i+1)
              and row.get('actual_ticks')==25 and bool(row.get('solver_policies_with_exact_installed_target'))
              and grip.get('command_consumed') is True and sent.get('sequence')==values.get('sequence')==i
              and memory.get('status')=='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY'
              and values.get('steps')==row.get('native_end_step')
              and _same_number(values.get('opening'),sent.get('opening'))
              and _same_number(values.get('expiry'),sent.get('expiry')),
              'ACTUAL_POSE_AND_GRIPPER_CONSUMPTION:'+str(i))
    check(query.get('schema')=='WM02_FUSION_PAUSED_QUERY_PAIR_V1'
          and query.get('query_uuid')==pins['query_uuid'] and query.get('pair_sha256')==pins['pair_sha256']
          and query.get('gpu_spec_sha256')==pins['gpu_spec_sha256']
          and query.get('capture_sha256')==pins['observation_sha256']
          and query.get('native_integrations_by_bridge')==0
          and query.get('native_execution_authorized') is False and query.get('online_action_execution') is False,
          'RECORDED_BRIDGE_EXACT_PAIR_SPEC_AND_CAPTURE')
    check(result.get('schema')=='WM02_FUSION_LIVE_PAUSED_INFERENCE_RESULT_V1'
          and result.get('query_uuid')==pins['query_uuid']
          and result.get('result_sha256')==pins['gpu_result_sha256']
          and result.get('action_sha256')==pins['action_sha256']
          and result.get('model_result_checked') is True and result.get('native_execution_authorized') is False
          and result.get('original_certificate_and_ACK_still_required') is True,
          'RECORDED_NATIVE_QUERY_RESULT_AND_ACTION_READBACK')
    check(pause.get('schema')==body['schema'] and pause.get('query_uuid')==pins['query_uuid']
          and pause.get('native_step')==body['native_step'] and pause.get('native_time_s')==body['native_time_s']
          and pause.get('unchanged_measured_packet_sha256')==body['files']['STATE.private.bin']
          and pause.get('native_execution_authorized') is False and pause.get('model_result_checked') is False,
          'RECORDED_POSTQUERY_UNCHANGED_ACTUAL_NATIVE_PACKET')
    try:
        require_pause(body['pause_after'],pause['pause_after_query'],body['native_step'])
    except (KeyError,TypeError,ValueError):
        issues.append('RECORDED_FRESH_POSTQUERY_ACK_AND_UNCHANGED_PHYSICS_FSM_LEG_HISTORY')
    return dict(established=not issues,unresolved=issues,recorded_proof=query or None)


def read_archive(native_spec_path, gpu_spec_path, *, pins):
    """Read only a fully closed archive pinned by an independent caller.

    ``pins`` has exactly ``PIN_KEYS``. File pins are SHA256; ``source_sha`` is
    Git's 40-hex identity; query UUID and seed are predeclared, never selected.
    Expired deadlines are retained metadata and confer no live permission.
    """
    _need(isinstance(pins,dict) and set(pins)==PIN_KEYS,'COMPLETE_EXTERNAL_PIN_SCHEMA')
    _need(all(_hash(pins[k]) for k in PIN_KEYS-{'source_sha','query_uuid','seed'})
          and _hash(pins['source_sha'],40) and type(pins['seed']) is int and 0<=pins['seed']<2**63,
          'EXACT_EXTERNAL_HASH_AND_SEED_TYPES')
    uuid_string(pins['query_uuid'])
    _need(pins['checkpoint_manifest_sha256']==CHECKPOINT
          and pins['checkpoint_file_inventory_sha256']==INVENTORY,'FROZEN_CORR2048_IDENTITY')
    native_spec_path,gpu_spec_path=Path(native_spec_path),Path(gpu_spec_path)
    n=_read(native_spec_path,pins['native_spec_sha256']);g=_read(gpu_spec_path,pins['gpu_spec_sha256'])
    nr,gr=Path(n['registered_root']),Path(g['registered_root'])
    _need(native_spec_path==nr/'EXECUTION_SPEC.json' and gpu_spec_path==gr/'EXECUTION_SPEC.json'
          and nr!=gr and all(p.is_absolute() and p.resolve()==p and not p.is_symlink() for p in(nr,gr))
          and not nr.is_relative_to(gr) and not gr.is_relative_to(nr),'EXACT_DISTINCT_CANONICAL_ROOTS')
    _need(n['source_sha']==g['source_sha']==pins['source_sha'] and n['source_root']==g['source_root']
          and all(n.get(k)==g.get(k) and n.get(k) is not None for k in
                  ('owner','host','uid','campaign_root','registry','pilot1_binding','pilot1_binding_sha256')),
          'COMMON_FROZEN_SOURCE_AND_ORIGINAL_OWNER')
    _need(n.get('physics') is True and n.get('phase')=='collect-target-fusion-paused-query-v2'
          and n.get('max_ticks')==3703 and g.get('physics') is False
          and g.get('kind')=='FROZEN_VLA_LIVE_PAUSED_QUERY'
          and g.get('profile')=='wm02-fusion-vla-shadow' and g.get('device')=='cuda'
          and g.get('execution_site')=='CORE_FUSION_NEURAL'
          and g.get('inference_mode')=='ACTUAL_LIVE_PAUSED_VLA_INFERENCE'
          and g.get('numeric_mode')=='ORIGINAL_FP32_WEIGHTS_BF16_AUTOCAST'
          and g.get('GR00T_updates')==g.get('WM_updates')==g.get('Actor_updates')==0
          and g.get('observation_mode')=='ACTUAL_OWNED_TARGET_LIVE_PAUSED_CAPTURE'
          and g.get('snapshot_binding_mode')=='PRE_REGISTERED_SINGLE_SNAPSHOT_V1'
          and g.get('max_queries')==1 and g.get('seed')==pins['seed'] and g.get('query_uuid')==pins['query_uuid']
          and g.get('checkpoint_manifest_sha256')==CHECKPOINT
          and document_hash(g['checkpoint_files'])==INVENTORY,'ORIGINAL_ONE_QUERY_CAPTURE_ONLY_CONTRACT')
    source_files=dict(n['source_files'])
    _need(all(name not in source_files or source_files[name]==pin for name,pin in g['source_files'].items()),
          'SAME_CAPSULE_MEMBER_PINS')
    source_files.update(g['source_files'])
    for name,pin in source_files.items():
        _need(not Path(name).is_absolute() and '..' not in Path(name).parts and _hash(pin)
              and digest(Path(g['source_root'])/name)==pin,'FROZEN_SOURCE_MEMBER:'+name)
    registry=_read(g['registry'])
    nc=_closed(n,native_spec_path,pins['native_spec_sha256'],registry)
    gc=_closed(g,gpu_spec_path,pins['gpu_spec_sha256'],registry)
    _need(gc['worker']['exit_code']==0 and gc['worker'].get('failure') is None,'SUCCESSFUL_SAVED_INPUT_GPU_QUERY')
    pair_path=Path(g['paused_query_pair']);pair=_read(pair_path,pins['pair_sha256'])
    _need(n.get('paused_query_pair')==str(pair_path)
          and n.get('paused_query_pair_sha256')==g.get('paused_query_pair_sha256')==pins['pair_sha256']
          and pair.get('schema')=='WM02_FUSION_PAUSED_QUERY_PAIR_V1'
          and pair.get('native_root')==str(nr) and pair.get('gpu_root')==str(gr)
          and pair.get('native_run_id')=='native-01' and pair.get('native_boundary_step')==3703
          and pair.get('query_uuid')==pins['query_uuid'] and pair.get('seed')==pins['seed']
          and pair.get('source_sha')==pins['source_sha'] and pair.get('task_sha256')==pins['task_sha256']
          and pair.get('frame_map_sha256')==pins['frame_map_sha256']
          and pair.get('checkpoint_manifest_sha256')==CHECKPOINT
          and pair.get('max_queries')==1 and pair.get('native_integrations_by_bridge')==0,
          'EXACT_PAIR_AND_PREDECLARED_QUERY_IDENTITY')
    _need(g.get('producer_native_spec')==str(native_spec_path)
          and g.get('producer_native_spec_sha256')==pins['native_spec_sha256']
          and n.get('paused_query_gpu_spec')==str(gpu_spec_path),'EXACT_NATIVE_PRODUCER_BACKREFERENCE')
    path=nr/'native-01/PAUSED_OBSERVATIONS'/pins['query_uuid']/'PAUSED_OBSERVATION.private.json'
    _need(g.get('recorded_observation')==str(path),'EXACT_ARCHIVED_SNAPSHOT_PATH')
    record=_read(path,pins['observation_sha256']);body=unseal(record)
    payload=paused_inference_payload(record);shadow=unseal(body['original_shadow_ABI'])
    index=body['capture']['index']
    names={'STATE.private.bin','BOUND_SHADOW_ABI.private.json',f'images/{index:06d}-calibration.json',
           *(f'images/{index:06d}-{camera}.png' for camera in('body_rgb','wrist_rgb'))}
    _need(set(body['files'])==names,'EXACT_FIVE_ARCHIVED_CAPTURE_MEMBERS')
    for name,pin in body['files'].items():
        member=path.parent/name
        _need(not member.is_symlink() and member.resolve().is_relative_to(path.parent.resolve())
              and digest(member)==pin,'ACTUAL_CAPTURE_MEMBER:'+name)
    _need(_read(path.parent/'BOUND_SHADOW_ABI.private.json',body['files']['BOUND_SHADOW_ABI.private.json'])==body['original_shadow_ABI'],
          'ACTUAL_NESTED_SIDECAR_BYTES')
    packet=parse_state(_bytes(path.parent/'STATE.private.bin',body['files']['STATE.private.bin']))
    _need(packet['step']==body['native_step']==3703 and abs(packet['time_s']-3703*.002)<=1e-8
          and body['native_time_s']==packet['time_s'] and pair.get('capture_index')==index
          and body['query_uuid']==pins['query_uuid'] and body['task_sha256']==pins['task_sha256']
          and n.get('paused_target_task_sha256')==g.get('paused_task_sha256')==pins['task_sha256']
          and body['world']['execution_spec_sha256']==pins['native_spec_sha256']
          and body['world']['registered_root']==str(nr) and body['world']['native_output']==str(nr/'native-01')
          and body['world']['scene_sha256']==pair['scene_sha256']==digest(n['model_path'])
          and shadow['source_policy_view']['transform_sha256']==pins['frame_map_sha256']
          and n.get('paused_frame_map_sha256')==pins['frame_map_sha256']
          and transform_hash(n['paused_source_from_target'])==pins['frame_map_sha256'],
          'ORIGINAL_TASK_FRAME_SCENE_PRODUCER_AND_ABSOLUTE_CLOCK')
    _need(digest(nr/'native-01/state.bin')==body['files']['STATE.private.bin'],
          'FINAL_ARCHIVED_NATIVE_PACKET_EQUALS_CAPTURE')
    ready=_read(path.parent/'PAIR_SNAPSHOT_READY.private.json')
    _need(ready==dict(schema=pair['schema'],pair_sha256=pins['pair_sha256'],
        producer_native_spec_sha256=pins['native_spec_sha256'],observation_sha256=pins['observation_sha256'],
        query_uuid=pins['query_uuid']),'ORIGINAL_CAPTURE_COMPLETION_RECORD')
    result_path=gr/'fit-01/RESULT.private.json';result=_read(result_path,pins['gpu_result_sha256'])
    _need(result.get('schema')=='WM02_FUSION_LIVE_PAUSED_INFERENCE_RESULT_V1'
          and result.get('status')=='REAL_FROZEN_VLA_LIVE_PAUSED_QUERY_DONE'
          and result.get('execution_spec_sha256')==pins['gpu_spec_sha256']
          and result.get('source_sha')==pins['source_sha'] and result.get('query_uuid')==pins['query_uuid']
          and result.get('seed')==pins['seed'] and result.get('checkpoint_manifest_sha256')==CHECKPOINT
          and result.get('checkpoint_file_inventory_sha256')==INVENTORY
          and result.get('recorded_observation_sha256')==pins['observation_sha256']
          and result.get('pair_sha256')==pins['pair_sha256']
          and result.get('producer_native_spec_sha256')==pins['native_spec_sha256']
          and result.get('action_sha256')==pins['action_sha256']
          and result.get('actual_model_loads')==result.get('actual_VLA_inferences')==1
          and result.get('native_integrations')==result.get('GR00T_updates')==result.get('WM_updates')==result.get('Actor_updates')==0
          and result.get('inference_mode')==g['inference_mode'] and result.get('numeric_mode')==g['numeric_mode']
          and result.get('native_execution_authorized') is False and result.get('online_closed_loop') is False,
          'ONE_ACTUAL_FROZEN_QUERY_AND_COMPLETE_EXTERNAL_RESULT_BINDING')
    expected_binding=dict(query_uuid=body['query_uuid'],snapshot_uuid=body['snapshot_uuid'],
        observation_sha256=pins['observation_sha256'],sealed_body_sha256=record['sha256'],
        task_sha256=body['task_sha256'],task=body['task'],world=body['world'],
        native_step=body['native_step'],native_time_s=body['native_time_s'],files=body['files'],
        renderer=body['renderer'],input_mode=body['input_mode'],mapped_payload_sha256=document_hash(payload),
        target_camera_provenance=shadow['provenance'],
        source_policy_view={k:v for k,v in shadow['source_policy_view'].items() if k!='payload'},
        snapshot_binding_mode='PRE_REGISTERED_SINGLE_SNAPSHOT_V1',pause_before=body['pause_before'],
        pause_after_capture=body['pause_after'],native_pause_after_query='NOT_CHECKED_BY_GPU_PROCESS',
        native_caller_must_verify_still_paused=True,native_execution_authorized=False,
        target_embodiment_qualified=False,target_task_qualified=False,online_closed_loop=False,source_outcomes_supplied=False)
    _need(result.get('observation_binding')==expected_binding,'RECORDED_GPU_FULL_INPUT_BINDING')
    action_path=gr/'fit-01/REAL_VLA_ACTION.npz'
    action_bytes=_bytes(action_path,pins['action_sha256'])
    with np.load(io.BytesIO(action_bytes),allow_pickle=False) as arrays:
        raw=arrays['physical_H40'];normalized=arrays['normalized']
        _need(raw.shape==(40,10) and normalized.shape==(1,40,132)
              and raw.dtype.kind==normalized.dtype.kind=='f'
              and np.isfinite(raw).all() and np.isfinite(normalized).all(),'FINITE_COMPLETE_UNMODIFIED_RAW_ARRAYS')
        raw=np.frombuffer(raw.tobytes(),dtype=raw.dtype).reshape(raw.shape)
        normalized=np.frombuffer(normalized.tobytes(),dtype=normalized.dtype).reshape(normalized.shape)
    native=_read(nr/'native-01/RESULT.json',pins['native_result_sha256'])
    _need(native.get('source_sha')==pins['source_sha'] and native.get('phase')==n['phase']
          and native.get('scene_sha256')==body['world']['scene_sha256'],'ORIGINAL_NATIVE_RESULT_IDENTITY')
    paired=_paired_return(native,nc,gc,body,pins)
    return dict(schema=SCHEMA,status='PAIR_COMPLETED_AND_PAUSE_REVALIDATED' if paired['established']
                else 'SAVED_SNAPSHOT_QUERY_ARCHIVED_NATIVE_PAIR_NOT_COMPLETED',
        physical_H40=raw,normalized=normalized,payload=payload,observation=record,
        GPU_result=result,native_result=native,paired_return=paired,
        closed_workers=dict(native=nc,gpu=gc),external_pins=dict(pins),
        source_manifest_sha256=document_hash(source_files),checkpoint_inventory=g['checkpoint_files'],
        checkpoint_bytes_reread=False,reuse_original_frozen_inference_verified_inventory=True,
        original_deadline_epochs=dict(native=n['deadline_epoch'],gpu=g['deadline_epoch'],pair=pair['deadline_epoch']),
        historical_read=True,current_deadline_validation_performed=False,current_native_pause_checked=False,
        native_execution_authorized=False,target_action_qualified=False,target_task_success=False,
        online_closed_loop_established=False,new_queries=0,new_integrations=0,processor_decodes=0)
