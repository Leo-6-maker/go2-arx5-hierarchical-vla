"""Hermetic retained-file ABI tests; no native episode, GPU, or real model.

Tiny CPU FK is used only to create a valid camera fixture. Every integration is
forbidden. The reader itself is separately checked to import no simulator/FK.
"""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
import pytest

from scripts.pi0292.relay_protocol import digest as document_hash, sealed, unseal
from scripts.pi0294.task_state import ActualState
from scripts.wm02 import infer_fusion_paused_vla as infer
from go2arx5_vla.world_model import fusion_paused_query_bridge as bridge
from go2arx5_vla.world_model import fusion_paused_inference as inference
from go2arx5_vla.world_model import fusion_paused_observation as observation
from go2arx5_vla.world_model import fusion_source_posture_paused_query_v3 as v3
from go2arx5_vla.world_model import fusion_source_posture_qualification_contract as ancestor
from go2arx5_vla.world_model import fusion_source_posture_query_archive_v3 as archive
from go2arx5_vla.world_model.storage import digest, write_json
from tests.world_model.test_fusion_paused_query_bridge import setup_pair, close_gpu
from tests.world_model.test_fusion_paused_inference import install_fake_draw
from tests.world_model.test_fusion_source_camera_capture import adapter_fixture, recipe, forbid_step

PROJECT = Path(__file__).resolve().parents[2]


def put(path,value):
    Path(path).write_text(json.dumps(value,allow_nan=False))
    return digest(path)


def make_archive(tmp_path,monkeypatch):
    original_update = ActualState.update
    g,old,pair,_,_ = setup_pair(tmp_path/'base',monkeypatch)
    adapter,binding,task,_ = adapter_fixture(tmp_path/'camera',monkeypatch)
    monkeypatch.setattr(ActualState,'update',original_update)
    camera = recipe(monkeypatch); forbid_step(monkeypatch)
    audit_pin = hashlib.sha256(camera.source_identity_bytes).hexdigest()
    monkeypatch.setattr(v3,'SOURCE_CAMERA_IDENTITY',audit_pin)
    monkeypatch.setattr(archive,'INVENTORY',inference.CHECKPOINT_INVENTORY)
    n = dict(old.spec,**adapter.spec)
    nr,gr = Path(n['registered_root']),Path(g['registered_root'])
    n.update(phase=v3.PHASE,physics=True,physics_dt_s=.002,WBC_changes=False,max_ticks=16403,
        source_files=copy.deepcopy(g['source_files']),lease=str(nr/'control/LEASE.json'),
        posture_query_camera_recipe=v3.SOURCE_CAMERA_RECIPE,source_camera_identity_sha256=audit_pin,
        paused_source_from_target=binding.T_source_from_target.tolist(),paused_frame_map_sha256=binding.expected_transform_sha256,
        paused_target_task_sha256=document_hash(task))
    for name in archive.SOURCE_MEMBERS:
        target=Path(n['source_root'])/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((PROJECT/name).read_bytes());n['source_files'][name]=digest(target)
    episode=dict(v3._fixed_episode(v3.SOURCE_CAMERA_RECIPE),run=str(nr),geometry_scale=1.,planned_end_atom=500)
    ep=nr/'EPISODE_MANIFEST.private.json'
    n.update(episode_manifest=str(ep),episode_manifest_sha256=put(ep,episode))
    pair_path=nr/'QUERY_PAIR.private.json'
    pair.update(schema=v3.PAIR_SCHEMA,native_root=str(nr),native_boundary_step=16403,
        task_sha256=n['paused_target_task_sha256'],scene_sha256=digest(n['model_path']),
        frame_map_sha256=n['paused_frame_map_sha256'],producer_phase=v3.PHASE,physical_recipe_phase=ancestor.PHASE,
        source_posture_episode_sha256=n['episode_manifest_sha256'],posture_source_observation_sha256=ancestor.POSTURE_OBSERVATION,
        geometry_scale=1.,time_stretch_factor=2,returned_VLA_action_installations=0,
        camera_recipe=v3.SOURCE_CAMERA_RECIPE,source_camera_identity_sha256=audit_pin)
    n.update(paused_query_pair=str(pair_path),paused_query_pair_sha256=put(pair_path,pair),
        paused_query_gpu_spec=str(gr/'EXECUTION_SPEC.json'))
    g.update(source_root=n['source_root'],source_files=copy.deepcopy(n['source_files']),
        snapshot_binding_mode=v3.PAIR_MODE,paused_query_pair=str(pair_path),paused_query_pair_sha256=digest(pair_path),
        producer_native_spec=str(nr/'EXECUTION_SPEC.json'),paused_task_sha256=n['paused_target_task_sha256'],
        recorded_observation=str(nr/'native-01/PAUSED_OBSERVATIONS'/g['query_uuid']/'PAUSED_OBSERVATION.private.json'))
    g['producer_native_spec_sha256']=put(nr/'EXECUTION_SPEC.json',n);put(gr/'EXECUTION_SPEC.json',g)
    # The genuine live recipe is covered by its own tests. This fixture only
    # replaces that boundary compiler, never the retained-record reader.
    monkeypatch.setattr(v3,'recipe_config',lambda spec:(episode,{},task))
    for s,r in ((n,nr),(g,gr)):
        (r/'control').mkdir(exist_ok=True)
        put(s['lease'],dict(status='ACTIVE',spec_sha256=digest(r/'EXECUTION_SPEC.json'),run_uuid=r.name))
    put(g['registry'],dict(entries=[dict(path=s['registered_root'],role=s['role'],source_sha=s['source_sha'],
        owner=s['owner'],local_lease=s['lease'])for s in(n,g)]))
    adapter.spec=n
    binding=replace(binding,target_runtime_sha256=digest(nr/'EXECUTION_SPEC.json'))
    wire=bytearray((adapter.output/'state.bin').read_bytes());struct.pack_into('<Qd',wire,8,16403,16403*.002)
    (adapter.output/'state.bin').write_bytes(wire)
    packet=adapter.observe();packet.update(step=16403,time_s=16403*.002)
    ack=json.loads((adapter.output/'pause_ack.json').read_text());ack.update(step=16403,time_s=16403*.002)
    put(adapter.output/'pause_ack.json',ack);(adapter.output/'step_quota').write_text('16403\n')
    quota=(adapter.output/'quota.binding').read_text().splitlines();quota[4]='16403'
    (adapter.output/'quota.binding').write_text('\n'.join(quota)+'\n')
    def spin(seconds,predicate):
        path=adapter.output/'pause_ack.json';value=json.loads(path.read_text());value['ack_sequence']+=1;put(path,value)
        return predicate()
    adapter.spin=spin
    draw=install_fake_draw(monkeypatch)
    def execute_once(seconds):
        output=gr/'fit-01';output.mkdir();infer.infer(g,output);close_gpu(g)
    monkeypatch.setattr(bridge.time,'sleep',execute_once)
    query=bridge.capture_and_wait_query(adapter,binding=binding,task=task,
        task_sha256=n['paused_target_task_sha256'],gpu_spec_path=gr/'EXECUTION_SPEC.json',camera_recipe=camera)
    windows=[]
    for i in range(556):
        sent=dict(sequence=i,opening=.95,expiry=2503*.002+i*.05+.4)
        values=dict(sequence=i,opening=sent['opening'],expiry=sent['expiry'],steps=2503+25*(i+1))
        windows.append(dict(native_start_step=2503+25*i,native_end_step=2503+25*(i+1),actual_ticks=25,
            trajectory_elapsed_atom=i-48 if i>=48 else None,
            solver_policies_with_exact_installed_target=[dict(trajectory_sha256='a'*64)],
            gripper=dict(submitted=sent,command_consumed=True,
                consumed_wrapper=dict(status='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY',values=values))))
    physical=dict(schema=v3.REPORT_SCHEMA,episode=episode,actual_spec_phase=v3.PHASE,physical_recipe_phase=ancestor.PHASE,
        query_stage='NOT_YET_CALLED',native_endpoint_step=16403,complete_task_atoms=556,
        solver_target_confirmed_atoms=556,gripper_confirmed_atoms=556,all_40_knots_consumed=True,raw_knot_consumed=[True]*40,
        actual_VLA_commands=0,VLA_commands=0,VLA_inferences=0,WBC_changes=False,physics_dt_s=.002,full_task_success=False,
        operator_cancel=dict(kind='operator_cancel',mode='HOLD',native_step=16403),
        adaptation=dict(raw_knots_retained=40,adapted_knots_retained=40,clock=dict(end_atom=500,time_stretch_factor=2,
            atom_dt_s=.05,native_dt_s=.002,native_ticks_per_atom=25,knot_indices=[32+12*i for i in range(40)])),
        windows=windows,original_Processor_decodes=0,Processor_decodes=0)
    physical_path=nr/'native-01/TRAJECTORY_QUALIFICATION.private.json';physical_pin=put(physical_path,physical)
    report=dict(physical,query_stage='ONE_FROZEN_GPU_QUERY_RETURNED_AND_PAUSE_REVALIDATED',
        physical_report_path=str(physical_path),physical_report_sha256=physical_pin,paused_query=query,
        actual_VLA_inferences=1,VLA_inferences=1,VLA_queries=1,VLA_action_installations=0,native_process_VLA_inferences=0,
        native_process_Processor_decodes=0,GPU_action_decode_contract='ONE_ORIGINAL_PROCESSOR_DECODE_INSIDE_RESIDENT_GENERATE',
        additional_action_decodes=0,query_wall_s=.1,query_CPU_s=.01,camera_equivalence=False,
        full_policy_input_distribution_qualified=False,returned_action_is_live_reference=False,
        online_VLA=False,online_closed_loop=False,native_integrations_by_query=0,native_stage_RESULT_json_required=True)
    report.pop('original_Processor_decodes');report.pop('Processor_decodes')
    put(nr/'native-01/SOURCE_POSTURE_PAUSED_QUERY.private.json',report)
    native=dict(status='NATIVE_STAGE_PASSED',phase=v3.PHASE,source_sha=n['source_sha'],scene_sha256=pair['scene_sha256'],
        actual_target_integrations=16403,reached_exact_pause=True,shutdown_clean=True,failure=None,physics_stops=[],
        original_joint_or_warning_samples=0,native_outer_guard_rejection_count=0,native_outer_guard_rejections={},
        trace_audit_error=None,owned_process_exits=[dict(name=k,exit_code=0)for k in('sim','planner','controller','solver')],
        trace_audit=dict(status='RECORDED_TRACE_CHECKS_PASS',errors=[],missing=[],writer=dict(dropped=0),native_step_records=16403),
        reference_qualification=report)
    put(nr/'native-01/RESULT.json',native)
    put(nr/'WORKER_EXIT.private.json',dict(exit_code=0,failure=None))
    lease=json.loads(Path(n['lease']).read_text());lease.update(status='CLOSED',worker_exit_code=0);put(n['lease'],lease)
    capture=Path(g['recorded_observation']);result=json.loads((gr/'fit-01/RESULT.private.json').read_text())
    pins=dict(native_spec_sha256=digest(nr/'EXECUTION_SPEC.json'),gpu_spec_sha256=digest(gr/'EXECUTION_SPEC.json'),
        gpu_result_sha256=digest(gr/'fit-01/RESULT.private.json'),action_sha256=result['action_sha256'],observation_sha256=digest(capture),
        pair_sha256=digest(pair_path),native_result_sha256=digest(nr/'native-01/RESULT.json'),source_sha=g['source_sha'],
        checkpoint_manifest_sha256=archive.CHECKPOINT,checkpoint_file_inventory_sha256=archive.INVENTORY,
        task_sha256=g['paused_task_sha256'],frame_map_sha256=pair['frame_map_sha256'],query_uuid=g['query_uuid'],seed=g['seed'],
        episode_sha256=n['episode_manifest_sha256'],physical_report_sha256=physical_pin,
        query_report_sha256=digest(nr/'native-01/SOURCE_POSTURE_PAUSED_QUERY.private.json'),source_camera_identity_sha256=audit_pin,
        noise_sha256=result['noise']['sha256'])
    def forbidden(*args,**kwargs):raise AssertionError('HISTORICAL_READER_CALLED_LIVE_API')
    for module,name in ((inference,'read_payload'),(inference,'validate_spec'),(observation,'verify_still_paused'),(bridge,'preflight_pair')):
        monkeypatch.setattr(module,name,forbidden)
    return nr/'EXECUTION_SPEC.json',gr/'EXECUTION_SPEC.json',pins,native,physical,result,draw


def test_full1_v3_closed_archive_uses_actual_nested_return_and_retains_every_output(tmp_path,monkeypatch):
    n,g,pins,_,_,_,draw=make_archive(tmp_path,monkeypatch)
    value=archive.read_archive(n,g,pins=pins)
    assert value['status']=='PAIR_COMPLETED_AND_PAUSE_REVALIDATED' and value['paired_return']['established']
    assert value['physical_H40'].shape==(40,10) and value['normalized'].shape==(1,40,132)
    assert value['physical_H40'][-1,-1]==.399 and np.all(value['normalized']==1.25)
    assert len(unseal(value['observation'])['files'])==8 and len(value['physical_report']['windows'])==556
    assert value['noise_identity']['sha256']==pins['noise_sha256'] and not value['noise_bytes_retained'] and not value['noise_recomputed']
    assert value['new_queries']==value['new_integrations']==value['processor_decodes']==0 and draw.queries==1
    assert not value['current_deadline_validation_performed'] and not value['current_native_pause_checked']
    assert not value['native_execution_authorized'] and not value['target_action_qualified'] and not value['checkpoint_bytes_reread']
    for key in('physical_H40','normalized'):
        with pytest.raises(ValueError):value[key].setflags(write=True)


def test_historical_deadlines_never_become_a_fake_current_live_grant(tmp_path,monkeypatch):
    n,g,pins,*_=make_archive(tmp_path,monkeypatch)
    monkeypatch.setattr(time,'time',lambda: (_ for _ in ()).throw(AssertionError('NO_CURRENT_TIME_FOR_HISTORY')))
    value=archive.read_archive(n,g,pins=pins)
    assert value['historical_read'] and not value['native_execution_authorized']


def test_archive_never_rereads_bulk_weights_or_calls_current_admission(tmp_path,monkeypatch):
    n,g,pins,*_=make_archive(tmp_path,monkeypatch)
    checkpoint=Path(json.loads(g.read_text())['checkpoint']);read=Path.read_bytes
    def bounded(path):
        assert not path.is_relative_to(checkpoint),'NO_WEIGHT_OR_PROCESSOR_BULK_READ_IN_ARCHIVE'
        return read(path)
    monkeypatch.setattr(Path,'read_bytes',bounded)
    value=archive.read_archive(n,g,pins=pins)
    assert not value['checkpoint_bytes_reread'] and value['reuse_original_frozen_inference_verified_inventory']


@pytest.mark.parametrize('fault',['V1_schema','3703_boundary','half_geometry','time1','undeclared_camera','returned_actions'])
def test_rehashed_pair_cannot_change_the_actual_v3_family_boundary_or_sensor(tmp_path,monkeypatch,fault):
    n,g,pins,*_=make_archive(tmp_path,monkeypatch)
    ns,gs=json.loads(n.read_text()),json.loads(g.read_text());path=Path(gs['paused_query_pair']);pair=json.loads(path.read_text())
    key,value={'V1_schema':('schema',bridge.SCHEMA),'3703_boundary':('native_boundary_step',3703),
        'half_geometry':('geometry_scale',.5),'time1':('time_stretch_factor',1),
        'undeclared_camera':('camera_recipe',v3.CAMERA_RECIPE),'returned_actions':('returned_VLA_action_installations',1)}[fault]
    pair[key]=value;pins['pair_sha256']=put(path,pair)
    ns['paused_query_pair_sha256']=gs['paused_query_pair_sha256']=pins['pair_sha256']
    pins['native_spec_sha256']=put(n,ns);gs['producer_native_spec_sha256']=pins['native_spec_sha256']
    pins['gpu_spec_sha256']=put(g,gs)
    for spec,pin in((ns,pins['native_spec_sha256']),(gs,pins['gpu_spec_sha256'])):
        lease=json.loads(Path(spec['lease']).read_text());lease['spec_sha256']=pin;put(spec['lease'],lease)
    with pytest.raises(ValueError,match='EXACT_PAIR_V2_NOT_V1_OR_OLD_GEOMETRY'):
        archive.read_archive(n,g,pins=pins)


def test_resealed_processor_category_change_rejected_before_gpu_result_claims(tmp_path,monkeypatch):
    n,g,pins,*_=make_archive(tmp_path,monkeypatch)
    path=Path(json.loads(g.read_text())['recorded_observation']);body=unseal(json.loads(path.read_text()))
    shadow=unseal(body['original_shadow_ABI']);shadow['provenance']['processor']='NEW_TARGET_PROCESSOR'
    body['original_shadow_ABI']=sealed(shadow)
    body['files']['BOUND_SHADOW_ABI.private.json']=put(path.parent/'BOUND_SHADOW_ABI.private.json',body['original_shadow_ABI'])
    pins['observation_sha256']=put(path,sealed(body))
    with pytest.raises(ValueError,match='UNCHANGED_SOURCE_PROCESSOR_CATEGORY10'):
        archive.read_archive(n,g,pins=pins)


@pytest.mark.parametrize('fault',['missing_pin','wrong_seed','wrong_noise_pin','wrong_frame','old_pair_schema','old_boundary',
    'capture_png','native_png','calibration','packet','nested','action','source','open_native','open_gpu','worker_failed'])
def test_archive_rejects_missing_pins_artifact_tamper_or_unclosed_workers(tmp_path,monkeypatch,fault):
    n,g,pins,*_=make_archive(tmp_path,monkeypatch);ns,gs=json.loads(n.read_text()),json.loads(g.read_text())
    folder=Path(gs['recorded_observation']).parent
    paths=dict(capture_png=folder/'images/000000-body_rgb.png',native_png=folder/'native-camera/images/000000-body_rgb.png',
        calibration=folder/'images/000000-calibration.json',packet=folder/'STATE.private.bin',nested=folder/'BOUND_SHADOW_ABI.private.json',
        action=g.parent/'fit-01/REAL_VLA_ACTION.npz',source=Path(ns['source_root'])/archive.SOURCE_MEMBERS[0])
    if fault in paths:paths[fault].write_bytes(b'changed')
    elif fault=='missing_pin':pins.pop('physical_report_sha256')
    elif fault=='wrong_seed':pins['seed']+=1
    elif fault=='wrong_noise_pin':pins['noise_sha256']='e'*64
    elif fault=='wrong_frame':pins['frame_map_sha256']='e'*64
    elif fault in('old_pair_schema','old_boundary'):
        pair=json.loads(Path(gs['paused_query_pair']).read_text())
        pair['schema' if fault=='old_pair_schema' else 'native_boundary_step']=bridge.SCHEMA if fault=='old_pair_schema' else 3703
        put(gs['paused_query_pair'],pair)
    elif fault in('open_native','open_gpu'):
        path=ns['lease']if fault=='open_native'else gs['lease'];lease=json.loads(Path(path).read_text());lease['status']='ACTIVE';put(path,lease)
    else:put(g.parent/'WORKER_EXIT.private.json',dict(exit_code=1,failure='retained failure'))
    with pytest.raises((ValueError,KeyError)):archive.read_archive(n,g,pins=pins)


def repin_native(n,pins,native,physical=None):
    if physical is not None:
        pins['physical_report_sha256']=put(n.parent/'native-01/TRAJECTORY_QUALIFICATION.private.json',physical)
        native['reference_qualification']['physical_report_sha256']=pins['physical_report_sha256']
    pins['query_report_sha256']=put(n.parent/'native-01/SOURCE_POSTURE_PAUSED_QUERY.private.json',native['reference_qualification'])
    pins['native_result_sha256']=put(n.parent/'native-01/RESULT.json',native)


@pytest.mark.parametrize('fault',['native_fail','drop','trace_count','child_fail','duplicate_child','missing_knot','missing_window',
    'missing_pose','wrong_grip_sequence','missing_opening','bad_elapsed','changed_physical','query_missing','wrong_result',
    'wrong_action','wrong_capture','packet_changed','stale_ack','advance_fsm','advance_leg','advance_gait','advance_native_time',
    'false_grant','installed_returned_action'])
def test_repinning_claims_does_not_replace_full_physical_and_postquery_evidence(tmp_path,monkeypatch,fault):
    n,g,pins,native,physical,_,_=make_archive(tmp_path,monkeypatch)
    report=native['reference_qualification'];query=report['paused_query'];checked=query['query_result'];pause=checked['original_pause_revalidation']
    if fault=='native_fail':native['status']='NATIVE_STAGE_FAILED_PRESERVED'
    elif fault=='drop':native['trace_audit']['writer']['dropped']=1
    elif fault=='trace_count':native['trace_audit']['native_step_records']-=1
    elif fault=='child_fail':native['owned_process_exits'][0]['exit_code']=1
    elif fault=='duplicate_child':native['owned_process_exits'].append(copy.deepcopy(native['owned_process_exits'][0]))
    elif fault=='missing_knot':physical['raw_knot_consumed'][3]=False
    elif fault=='missing_window':physical['windows'].pop()
    elif fault=='missing_pose':physical['windows'][100]['solver_policies_with_exact_installed_target']=[]
    elif fault=='wrong_grip_sequence':physical['windows'][100]['gripper']['consumed_wrapper']['values']['sequence']-=1
    elif fault=='missing_opening':
        physical['windows'][100]['gripper']['consumed_wrapper']['values'].pop('opening');physical['windows'][100]['gripper']['submitted'].pop('opening')
    elif fault=='bad_elapsed':physical['windows'][100]['trajectory_elapsed_atom']+=1
    elif fault=='changed_physical':report['physics_dt_s']=.004
    elif fault=='query_missing':report.pop('paused_query')
    elif fault=='wrong_result':checked['result_sha256']='e'*64
    elif fault=='wrong_action':checked['action_sha256']='e'*64
    elif fault=='wrong_capture':query['capture_sha256']='e'*64
    elif fault=='packet_changed':pause['unchanged_measured_packet_sha256']='e'*64
    elif fault=='stale_ack':
        body=unseal(json.loads(Path(json.loads(g.read_text())['recorded_observation']).read_text()))
        pause['pause_after_query']['ack_sequence']=body['pause_after']['ack_sequence']
    elif fault in('advance_fsm','advance_leg','advance_gait','advance_native_time'):
        key=dict(advance_fsm='motiontime',advance_leg='leg_history_steps',advance_gait='gait_indices',advance_native_time='time_s')[fault]
        pause['pause_after_query'][key]+=1
    elif fault=='false_grant':checked['native_execution_authorized']=True
    else:report['VLA_action_installations']=1
    repin_native(n,pins,native,physical)
    with pytest.raises((ValueError,KeyError)):archive.read_archive(n,g,pins=pins)


@pytest.mark.parametrize('fault',['noise_shape','noise_dtype','noise_hash','noise_missing','wrong_precision','wrong_processor_inventory',
    'output_shape','outside_count','source_frame','grant','TTL_rebase'])
def test_externally_repinned_gpu_result_still_requires_frozen_raw_draw_contract(tmp_path,monkeypatch,fault):
    n,g,pins,_,_,result,_=make_archive(tmp_path,monkeypatch)
    if fault=='noise_shape':result['noise']['shape']=[1,8,132]
    elif fault=='noise_dtype':result['noise']['dtype']='torch.bfloat16'
    elif fault=='noise_hash':result['noise']['sha256']='unknown'
    elif fault=='noise_missing':result.pop('noise')
    elif fault=='wrong_precision':result['precision']['cuda_matmul_allow_tf32']=True
    elif fault=='wrong_processor_inventory':result['checkpoint_file_inventory_sha256']='e'*64
    elif fault=='output_shape':result['physical_H40_shape']=[8,10]
    elif fault=='outside_count':result['normalized_outside_unit_components']=0
    elif fault=='source_frame':result['source_frame']='TARGET_BASE'
    elif fault=='grant':result['native_execution_authorized']=True
    else:result['reference_TTL_rebased']=True
    pins['gpu_result_sha256']=put(g.parent/'fit-01/RESULT.private.json',result)
    with pytest.raises((ValueError,KeyError)):archive.read_archive(n,g,pins=pins)


@pytest.mark.parametrize('fault',['truncated_raw','truncated_normalized','nonfinite_raw','nonfinite_normalized','float64_normalized','extra_array'])
def test_repinned_npz_must_keep_the_complete_original_finite_array_abi(tmp_path,monkeypatch,fault):
    n,g,pins,_,_,_,_=make_archive(tmp_path,monkeypatch);path=g.parent/'fit-01/REAL_VLA_ACTION.npz'
    with np.load(path,allow_pickle=False) as arrays:values={k:arrays[k]for k in arrays.files}
    if fault=='truncated_raw':values['physical_H40']=values['physical_H40'][:8]
    elif fault=='truncated_normalized':values['normalized']=values['normalized'][:,:8]
    elif fault=='nonfinite_raw':values['physical_H40'][0,0]=np.nan
    elif fault=='nonfinite_normalized':values['normalized'][0,0,0]=np.inf
    elif fault=='float64_normalized':values['normalized']=values['normalized'].astype(np.float64)
    else:values['noise_recreated']=np.zeros((1,40,132),np.float32)
    with path.open('wb')as stream:np.savez_compressed(stream,**values)
    pins['action_sha256']=digest(path)
    result_path=g.parent/'fit-01/RESULT.private.json';result=json.loads(result_path.read_text());result['action_sha256']=pins['action_sha256']
    pins['gpu_result_sha256']=put(result_path,result)
    with pytest.raises(ValueError,match='ORIGINAL_TWO_RAW|FINITE_UNCLIPPED'):
        archive.read_archive(n,g,pins=pins)


def test_reader_preserves_finite_ood_raw_values_without_clipping_or_qualification(tmp_path,monkeypatch):
    n,g,pins,native,_,result,_=make_archive(tmp_path,monkeypatch);path=g.parent/'fit-01/REAL_VLA_ACTION.npz'
    with np.load(path,allow_pickle=False) as arrays:values={k:arrays[k]for k in arrays.files}
    values['physical_H40'][-1,9]=1.5
    with path.open('wb')as stream:np.savez_compressed(stream,**values)
    pins['action_sha256']=digest(path);result['action_sha256']=pins['action_sha256']
    pins['gpu_result_sha256']=put(g.parent/'fit-01/RESULT.private.json',result)
    checked=native['reference_qualification']['paused_query']['query_result']
    checked.update(action_sha256=pins['action_sha256'],result_sha256=pins['gpu_result_sha256'])
    repin_native(n,pins,native)
    value=archive.read_archive(n,g,pins=pins)
    assert value['physical_H40'][-1,9]==1.5 and not value['target_action_qualified']


def test_pinned_json_hash_and_parse_use_one_read_and_ignore_later_file_replacement(tmp_path,monkeypatch):
    path=tmp_path/'pin.json';path.write_text('{"original":true}');pin=digest(path);original=Path.read_bytes;calls=[]
    def once(p):
        value=original(p);calls.append(p);p.write_text('{"replacement":true}');return value
    monkeypatch.setattr(Path,'read_bytes',once)
    assert archive._read(path,pin)==dict(original=True) and calls==[path]


def test_reader_imports_no_model_renderer_or_native_runtime_and_cannot_touch_current_site():
    program="import sys;from go2arx5_vla.world_model import fusion_source_posture_query_archive_v3;assert not any(k in sys.modules for k in ('torch','gr00t','mujoco','rclpy','transformers'))"
    reply=subprocess.run([sys.executable,'-B','-c',program],capture_output=True,text=True)
    assert reply.returncode==0,reply.stderr
