"""Archive seams use synthetic actual files, never a model or native trial."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
import pytest

from scripts.pi0292.relay_protocol import unseal
from scripts.wm02 import infer_fusion_paused_vla as cli
from go2arx5_vla.world_model import fusion_paused_observation as observation
from go2arx5_vla.world_model import fusion_paused_inference as live
from go2arx5_vla.world_model import fusion_paused_query_archive as archive
from go2arx5_vla.world_model.storage import digest,write_json
from tests.world_model.test_fusion_paused_query_bridge import setup_pair,close_gpu
from tests.world_model.test_fusion_paused_inference import install_fake_draw


def overwrite(path,value):Path(path).write_text(json.dumps(value))


def make_archive(tmp_path,monkeypatch,*,paired=True):
    g,adapter,pair,events,_=setup_pair(tmp_path,monkeypatch)
    monkeypatch.setattr(archive,'INVENTORY',live.CHECKPOINT_INVENTORY)
    n=adapter.spec;nr,gr=Path(n['registered_root']),Path(g['registered_root'])
    n['phase']='collect-target-fusion-paused-query-v2'
    n['source_files']=copy.deepcopy(g['source_files'])
    n['paused_source_from_target']=adapter.fixture_binding.T_source_from_target.tolist()
    overwrite(nr/'EXECUTION_SPEC.json',n)
    g['producer_native_spec_sha256']=digest(nr/'EXECUTION_SPEC.json')
    overwrite(gr/'EXECUTION_SPEC.json',g)
    adapter.fixture_binding=replace(adapter.fixture_binding,target_runtime_sha256=digest(nr/'EXECUTION_SPEC.json'))
    for spec,root in ((n,nr),(g,gr)):
        lease=json.loads(Path(spec['lease']).read_text());lease['spec_sha256']=digest(root/'EXECUTION_SPEC.json')
        overwrite(spec['lease'],lease)
    record=observation.capture_paused_observation(adapter,binding=adapter.fixture_binding,
        task=adapter.fixture_task,task_sha256=g['paused_task_sha256'],query_uuid=g['query_uuid'])
    path=Path(g['recorded_observation']);body=unseal(record)
    write_json(path.parent/'PAIR_SNAPSHOT_READY.private.json',dict(schema=pair['schema'],
        pair_sha256=g['paused_query_pair_sha256'],producer_native_spec_sha256=g['producer_native_spec_sha256'],
        observation_sha256=digest(path),query_uuid=g['query_uuid']))
    events_draw=install_fake_draw(monkeypatch)
    output=gr/'fit-01';output.mkdir()
    result=cli.infer(g,output)
    query_return=live.verify_paused_query_return(adapter,g,output/'RESULT.private.json')
    close_gpu(g)
    query=dict(schema=pair['schema'],pair_sha256=g['paused_query_pair_sha256'],query_uuid=g['query_uuid'],
        gpu_spec_sha256=digest(gr/'EXECUTION_SPEC.json'),capture_sha256=digest(path),query_result=query_return,
        native_integrations_by_bridge=0,native_execution_authorized=False,online_action_execution=False)
    rows=[]
    for i in range(48):
        sent=dict(sequence=i,opening=.95,expiry=2503*.002+i*.05+.4)
        values=dict(sequence=i,opening=sent['opening'],expiry=sent['expiry'],steps=2503+25*(i+1))
        rows.append(dict(native_start_step=2503+25*i,native_end_step=2503+25*(i+1),actual_ticks=25,
            solver_policies_with_exact_installed_target=[dict(trajectory_sha256='a'*64)],
            gripper=dict(submitted=sent,consumed_wrapper=dict(status='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY',values=values),
                         command_consumed=True)))
    report=dict(native_endpoint_step=3703,complete_task_atoms=48,solver_target_confirmed_atoms=48,
        gripper_confirmed_atoms=48,VLA_action_installations=0,WM_corrections=0,task_executed=False,
        full_task_success=False,windows=rows,paused_query=query)
    native=dict(status='NATIVE_STAGE_PASSED' if paired else 'NATIVE_STAGE_FAILED_PRESERVED',
        source_sha=n['source_sha'],phase='collect-target-fusion-paused-query-v2',scene_sha256=body['world']['scene_sha256'],
        actual_target_integrations=3703,shutdown_clean=True,reached_exact_pause=paired,
        failure=None if paired else 'FileNotFoundError: actual pause_ack read failure retained',
        physics_stops=[],original_joint_or_warning_samples=0,native_outer_guard_rejection_count=0,
        native_outer_guard_rejections={},trace_audit_error=None,
        owned_process_exits=[dict(name=name,exit_code=0)for name in('sim','planner','controller','solver')],
        trace_audit=dict(status='RECORDED_TRACE_CHECKS_PASS',errors=[],missing=[],writer=dict(dropped=0),native_step_records=3703),
        reference_qualification=report if paired else None)
    write_json(nr/'native-01/RESULT.json',native)
    nlease=json.loads(Path(n['lease']).read_text());nlease.update(status='CLOSED',worker_exit_code=0 if paired else 1)
    overwrite(n['lease'],nlease)
    write_json(nr/'WORKER_EXIT.private.json',dict(exit_code=0 if paired else 1,failure=None))
    pins=dict(native_spec_sha256=digest(nr/'EXECUTION_SPEC.json'),gpu_spec_sha256=digest(gr/'EXECUTION_SPEC.json'),
        gpu_result_sha256=digest(output/'RESULT.private.json'),action_sha256=result['action_sha256'],
        observation_sha256=digest(path),pair_sha256=digest(g['paused_query_pair']),native_result_sha256=digest(nr/'native-01/RESULT.json'),
        source_sha=g['source_sha'],checkpoint_manifest_sha256=archive.CHECKPOINT,
        checkpoint_file_inventory_sha256=archive.INVENTORY,task_sha256=g['paused_task_sha256'],
        frame_map_sha256=pair['frame_map_sha256'],query_uuid=g['query_uuid'],seed=g['seed'])
    # Real generating/renderer functions are replaced only in fixture setup;
    # archive reading must never reach the live reader or current clock logic.
    def forbidden(*args,**kw):raise AssertionError('ARCHIVE_MUST_NOT_CALL_LIVE_READER_OR_RUNTIME')
    monkeypatch.setattr(live,'read_payload',forbidden)
    monkeypatch.setattr(observation,'verify_still_paused',forbidden)
    return nr/'EXECUTION_SPEC.json',gr/'EXECUTION_SPEC.json',pins,native,events_draw


def test_true_closed_pair_requires_actual_nested_return_and_keeps_complete_raw_arrays(tmp_path,monkeypatch):
    n,g,pins,_,draw=make_archive(tmp_path,monkeypatch)
    value=archive.read_archive(n,g,pins=pins)
    assert value['status']=='PAIR_COMPLETED_AND_PAUSE_REVALIDATED'
    assert value['paired_return']['established'] and not value['paired_return']['unresolved']
    assert value['physical_H40'].shape==(40,10) and value['normalized'].shape==(1,40,132)
    assert value['physical_H40'][-1,-1]==.399 and np.all(value['normalized']==1.25)
    assert value['new_queries']==value['new_integrations']==value['processor_decodes']==0
    assert not value['native_execution_authorized'] and not value['target_action_qualified']
    assert not value['current_deadline_validation_performed'] and not value['current_native_pause_checked']
    assert not value['checkpoint_bytes_reread'] and draw.queries==1
    with pytest.raises(ValueError):value['physical_H40'].setflags(write=True)


def test_original_native_failure_retains_saved_query_without_fabricating_return(tmp_path,monkeypatch):
    n,g,pins,native,_=make_archive(tmp_path,monkeypatch,paired=False)
    value=archive.read_archive(n,g,pins=pins)
    assert value['status']=='SAVED_SNAPSHOT_QUERY_ARCHIVED_NATIVE_PAIR_NOT_COMPLETED'
    assert not value['paired_return']['established'] and value['paired_return']['recorded_proof'] is None
    assert value['native_result']['failure']==native['failure']
    assert value['closed_workers']['native']['worker']['exit_code']==1
    assert value['physical_H40'].shape==(40,10)


def test_closed_archive_remains_readable_long_after_original_deadline_without_patching_live_validation(tmp_path,monkeypatch):
    n,g,pins,_,_=make_archive(tmp_path,monkeypatch)
    deadline=json.loads(g.read_text())['deadline_epoch']
    monkeypatch.setattr(time,'time',lambda:deadline+10**9)
    value=archive.read_archive(n,g,pins=pins)
    assert value['status']=='PAIR_COMPLETED_AND_PAUSE_REVALIDATED'
    assert value['original_deadline_epochs']['gpu']<time.time()
    assert not value['current_deadline_validation_performed'] and not value['native_execution_authorized']


def test_external_json_pin_and_parse_use_the_same_single_byte_read(tmp_path,monkeypatch):
    path=tmp_path/'pin.json';path.write_text('{"original":true}')
    pin=digest(path);original=Path.read_bytes;calls=[]
    def once(p):
        raw=original(p);calls.append(p)
        p.write_text('{"changed_after_read":true}')
        return raw
    monkeypatch.setattr(Path,'read_bytes',once)
    value=archive._read(path,pin)
    assert value==dict(original=True) and calls==[path]


@pytest.mark.parametrize('fault',['gpu_result','action','capture_image','calibration','raw_packet','nested_sidecar',
    'pair','spec','source','open_gpu','open_native','worker','registry','missing_pin','wrong_seed','wrong_frame','wrong_model'])
def test_archive_rejects_tampered_artifacts_missing_external_pins_or_unclosed_roots(tmp_path,monkeypatch,fault):
    n,g,pins,_,_=make_archive(tmp_path,monkeypatch)
    ns,gs=json.loads(n.read_text()),json.loads(g.read_text())
    capture=Path(gs['recorded_observation']).parent
    members={'gpu_result':g.parent/'fit-01/RESULT.private.json','action':g.parent/'fit-01/REAL_VLA_ACTION.npz',
        'capture_image':capture/'images/000000-body_rgb.png','calibration':capture/'images/000000-calibration.json',
        'raw_packet':capture/'STATE.private.bin','nested_sidecar':capture/'BOUND_SHADOW_ABI.private.json',
        'pair':Path(gs['paused_query_pair']),'spec':n}
    if fault in members:members[fault].write_bytes(b'CHANGED')
    elif fault=='source':
        member=next(iter(gs['source_files']));(Path(gs['source_root'])/member).write_text('CHANGED')
    elif fault in('open_gpu','open_native'):
        path=Path(gs['lease']if fault=='open_gpu'else ns['lease']);lease=json.loads(path.read_text())
        lease['status']='ACTIVE';overwrite(path,lease)
    elif fault=='worker':overwrite(g.parent/'WORKER_EXIT.private.json',dict(exit_code=1,failure=None))
    elif fault=='registry':overwrite(gs['registry'],dict(entries=[]))
    elif fault=='missing_pin':pins.pop('gpu_result_sha256')
    elif fault=='wrong_seed':pins['seed']+=1
    elif fault=='wrong_frame':pins['frame_map_sha256']='e'*64
    elif fault=='wrong_model':pins['checkpoint_manifest_sha256']='e'*64
    with pytest.raises((ValueError,KeyError)):archive.read_archive(n,g,pins=pins)


@pytest.mark.parametrize('fault',['native_fail','trace_drop','trace_count','child_exit','missing_proof','foreign_result',
    'foreign_action','foreign_capture','foreign_spec','changed_packet_proof','no_fresh_ack','clock_advanced',
    'missing_window','wrong_gripper_sequence','missing_gripper_opening','missing_solver_match'])
def test_pinned_native_claims_cannot_substitute_for_complete_recorded_pause_return(tmp_path,monkeypatch,fault):
    n,g,pins,native,_=make_archive(tmp_path,monkeypatch)
    report=native['reference_qualification'];bridge=report['paused_query']
    checked=bridge['query_result'];pause=checked['original_pause_revalidation']
    if fault=='native_fail':native['status']='NATIVE_STAGE_FAILED_PRESERVED'
    elif fault=='trace_drop':native['trace_audit']['writer']['dropped']=1
    elif fault=='trace_count':native['trace_audit']['native_step_records']-=1
    elif fault=='child_exit':native['owned_process_exits'][0]['exit_code']=1
    elif fault=='missing_proof':report.pop('paused_query')
    elif fault=='foreign_result':checked['result_sha256']='e'*64
    elif fault=='foreign_action':checked['action_sha256']='e'*64
    elif fault=='foreign_capture':bridge['capture_sha256']='e'*64
    elif fault=='foreign_spec':bridge['gpu_spec_sha256']='e'*64
    elif fault=='changed_packet_proof':pause['unchanged_measured_packet_sha256']='e'*64
    elif fault=='no_fresh_ack':
        body=unseal(json.loads(Path(json.loads(g.read_text())['recorded_observation']).read_text()))
        pause['pause_after_query']['ack_sequence']=body['pause_after']['ack_sequence']
    elif fault=='clock_advanced':pause['pause_after_query']['leg_history_steps']+=1
    elif fault=='missing_window':report['windows'].pop()
    elif fault=='wrong_gripper_sequence':report['windows'][7]['gripper']['consumed_wrapper']['values']['sequence']-=1
    elif fault=='missing_gripper_opening':
        report['windows'][7]['gripper']['consumed_wrapper']['values'].pop('opening')
        report['windows'][7]['gripper']['submitted'].pop('opening')
    elif fault=='missing_solver_match':report['windows'][7]['solver_policies_with_exact_installed_target']=[]
    path=n.parent/'native-01/RESULT.json';overwrite(path,native);pins['native_result_sha256']=digest(path)
    value=archive.read_archive(n,g,pins=pins)
    assert not value['paired_return']['established'] and value['paired_return']['unresolved']
    assert value['status']=='SAVED_SNAPSHOT_QUERY_ARCHIVED_NATIVE_PAIR_NOT_COMPLETED'
    assert not value['native_execution_authorized']


def test_historical_reader_imports_no_model_renderer_ros_or_simulator():
    code="import sys;from go2arx5_vla.world_model import fusion_paused_query_archive;assert not any(x in sys.modules for x in ['torch','gr00t','mujoco','rclpy','transformers'])"
    reply=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
    assert reply.returncode==0,reply.stderr
