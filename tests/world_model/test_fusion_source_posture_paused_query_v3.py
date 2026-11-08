"""Synthetic V3 recipe/pair/native seams; no real world, renderer, or GPU."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pi0292.relay_protocol import unseal
from scripts.wm02 import infer_fusion_paused_vla as infer
from scripts.wm02 import fusion_source_posture_paused_query_check as check_cli
from go2arx5_vla.world_model import auto1_native, target_sequential
from go2arx5_vla.world_model import fusion_source_posture_paused_query_v3 as v3
from go2arx5_vla.world_model import fusion_source_posture_qualification_contract as posture
from go2arx5_vla.world_model import fusion_paused_query_bridge as bridge
from go2arx5_vla.world_model import fusion_paused_inference as inference
from go2arx5_vla.world_model import fusion_trajectory_qualification_native as trajectory_native
from go2arx5_vla.world_model.fusion_observation import transform_hash
from go2arx5_vla.world_model.fusion_trajectory_adapter import TrajectoryAdaptation
from go2arx5_vla.world_model.fusion_short_skill import task_document, score_contract, document_hash
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_fusion_source_posture_qualification_contract import recipe as posture_recipe, episode as posture_episode
from tests.world_model.test_fusion_live_proposal_qualification_contract import prior_files, put, BASE
from tests.world_model.test_fusion_paused_query_bridge import setup_pair, close_gpu
from tests.world_model.test_fusion_paused_inference import install_fake_draw
from tests.world_model.test_fusion_live_and_posture_native import MeasuredAdapter
from tests.world_model.test_fusion_trajectory_qualification import setup_flow

PROJECT = Path(__file__).resolve().parents[2]


def new_episode(root,inputs):
    program = TrajectoryAdaptation(inputs['raw'],inputs['query_tcp'],inputs['frame_map'],
        np.eye(4),np.eye(4),BASE,1.,time_stretch_factor=2)
    return dict(v3._fixed_episode(),run=str(root),geometry_scale=1.,planned_end_atom=program.end_atom)


@pytest.fixture
def recipe(tmp_path,monkeypatch):
    folder=tmp_path/'recipe';folder.mkdir()
    native,_,_,inputs,archive,_=posture_recipe.__wrapped__(folder,monkeypatch)
    # Only synthetic artifact pins differ. Exercise every real recipe validator.
    pins=dict(posture.PINS,frame_map_sha256=transform_hash(inputs['frame_map']))
    monkeypatch.setattr(posture,'PINS',pins)
    archive['external_pins']=copy.deepcopy(pins)
    native['source_posture_query_pins']=copy.deepcopy(pins)
    gpu,_,pair,_,_=setup_pair(tmp_path/'pair',monkeypatch)
    capsule=Path(native['source_root'])
    for name in set(v3.SOURCE_MEMBERS)|set(inference.SOURCE_MEMBERS):
        path=capsule/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((PROJECT/name).read_bytes());native['source_files'][name]=digest(path)
    native.update(phase=v3.PHASE,stage_identity_sha256='a'*64,posture_query_camera_recipe=v3.CAMERA_RECIPE)
    episode=new_episode(Path(native['registered_root']),inputs)
    native.update(episode_manifest_sha256=put(native['episode_manifest'],episode),max_ticks=v3.endpoint(episode))
    half_root=tmp_path/'complete-half';half=posture_episode(half_root,inputs,.5)
    previous,path,pin,_=prior_files(half_root,half,native)
    native.update(previous_stage=str(path),previous_stage_sha256=pin)
    task=task_document(digest(native['model_path']))
    task_path=tmp_path/'TASK.json';score_path=tmp_path/'SCORE.json'
    native.update(paused_target_task=str(task_path),paused_target_task_file_sha256=put(task_path,task),
        paused_target_task_sha256=document_hash(task),paused_score_contract=str(score_path),
        paused_score_contract_file_sha256=put(score_path,score_contract()),
        paused_source_from_target=inputs['frame_map'].tolist(),paused_frame_map_sha256=pins['frame_map_sha256'])
    for key in ('owner','host','uid','registry','pilot1_binding','pilot1_binding_sha256'):
        native[key]=gpu[key]
    native.update(role='SYNTHETIC_POSTURE_QUERY_V3_NATIVE',source_sha=gpu['source_sha'],
        lease=str(Path(native['registered_root'])/'control/LEASE.json'),deadline_epoch=gpu['deadline_epoch']+1,
        paused_query_pair=gpu['paused_query_pair'],paused_query_gpu_spec=str(Path(gpu['registered_root'])/'EXECUTION_SPEC.json'))
    gpu.update(snapshot_binding_mode=v3.PAIR_MODE,source_root=native['source_root'],source_files=native['source_files'].copy(),
        campaign_root=native['campaign_root'],producer_native_spec=str(Path(native['registered_root'])/'EXECUTION_SPEC.json'),
        paused_task_sha256=native['paused_target_task_sha256'],recorded_observation=str(
            Path(native['registered_root'])/'native-01/PAUSED_OBSERVATIONS'/gpu['query_uuid']/'PAUSED_OBSERVATION.private.json'))
    pair.update(schema=v3.PAIR_SCHEMA,native_root=native['registered_root'],native_boundary_step=native['max_ticks'],
        task_sha256=gpu['paused_task_sha256'],scene_sha256=digest(native['model_path']),
        frame_map_sha256=native['paused_frame_map_sha256'],producer_phase=v3.PHASE,
        physical_recipe_phase=posture.PHASE,source_posture_episode_sha256=native['episode_manifest_sha256'],
        posture_source_observation_sha256=posture.POSTURE_OBSERVATION,geometry_scale=1.,time_stretch_factor=2,
        returned_VLA_action_installations=0,camera_recipe=v3.CAMERA_RECIPE,source_camera_identity_sha256=None)
    bind(native,gpu,pair)
    return native,gpu,pair,episode,inputs,previous


def bind(native,gpu,pair):
    pair_pin=put(gpu['paused_query_pair'],pair)
    native['paused_query_pair_sha256']=gpu['paused_query_pair_sha256']=pair_pin
    native_path=Path(native['registered_root'])/'EXECUTION_SPEC.json'
    gpu['producer_native_spec_sha256']=put(native_path,native)
    gpu_path=Path(gpu['registered_root'])/'EXECUTION_SPEC.json';put(gpu_path,gpu)
    entries=[]
    for spec,path in ((native,native_path),(gpu,gpu_path)):
        root=Path(spec['registered_root'])
        put(spec['lease'],dict(status='ACTIVE',spec_sha256=digest(path),run_uuid=root.name))
        entries.append(dict(path=str(root),role=spec['role'],source_sha=spec['source_sha'],
            owner=spec['owner'],local_lease=spec['lease']))
    put(gpu['registry'],dict(entries=entries))


def test_real_v3_recipe_keeps_actual_spec_and_manifest_with_explicit_ancestor_view(recipe):
    native,gpu,pair,episode,inputs,previous=recipe
    original=copy.deepcopy(native);raw=Path(native['episode_manifest']).read_bytes()
    actual,loaded,task=v3.episode_config(native)
    assert actual==episode and native==original and Path(native['episode_manifest']).read_bytes()==raw
    assert native['phase']==actual['phase']==v3.PHASE
    ancestor=v3.parent_recipe_view(actual)
    assert ancestor['phase']==posture.PHASE and ancestor['schema']==posture.SCHEMA
    assert actual['physical_recipe_phase']==posture.PHASE and actual['geometry_scale']==1.
    np.testing.assert_array_equal(loaded['raw'],inputs['raw'])
    assert task==task_document(digest(native['model_path']))
    assert v3.episode_limit(native,previous)==native['max_ticks']<=33000
    pair_checked,producer=bridge.validate_pair_spec(gpu)
    assert pair_checked==pair and producer==native
    proof=bridge.preflight_pair(Path(gpu['registered_root'])/'EXECUTION_SPEC.json',native)
    assert proof['schema']==v3.PAIR_SCHEMA and proof['native_boundary_step']==native['max_ticks']>3703
    assert not proof['snapshot_exists'] and not proof['native_execution_authorized']
    inference.validate_spec(gpu)
    assert not Path(gpu['recorded_observation']).exists()


@pytest.mark.parametrize('change',[
    dict(phase=posture.PHASE),dict(schema=posture.SCHEMA),dict(geometry_scale=.5),dict(geometry_scale=True),
    dict(time_stretch_factor=1),dict(physical_recipe_phase=v3.PHASE),dict(max_queries=2),
    dict(returned_VLA_action_installations=1),dict(split='TEST'),dict(full_task_success_claim=True),
    dict(capture_camera_recipe='SIM_EXTERNAL_WORLD_SOURCE_RIG_NOT_YET_INTEGRATED')])
def test_exact_v3_cannot_be_downgraded_or_turned_into_action_execution(recipe,change):
    episode=recipe[3]
    with pytest.raises(ValueError):v3.episode_contract(dict(episode,**change))


@pytest.mark.parametrize('fault',['bootstrap','same-full1','wrong-family','failed','trace-drop','grip','knots','open'])
def test_only_immediate_closed_complete_half_posture_is_a_predecessor(recipe,fault):
    native,_,_,_,_,previous=recipe
    path=Path(native['previous_stage'])
    if fault in ('bootstrap','same-full1','wrong-family'):
        prior_spec_path=path.parent.parent/'EXECUTION_SPEC.json';prior_spec=json.loads(prior_spec_path.read_text())
        ep=previous['reference_qualification']['episode']
        if fault=='same-full1':ep['geometry_scale']=1.
        else:previous['phase']=prior_spec['phase']='collect-target-fusion-trajectory-slow-qualification-v2'
        prior_spec['episode_manifest_sha256']=put(prior_spec['episode_manifest'],ep)
        put(prior_spec_path,prior_spec)
        lease=path.parent.parent/'control/LEASE.json';d=json.loads(lease.read_text())
        d['spec_sha256']=digest(prior_spec_path);put(lease,d)
    elif fault=='failed':previous['status']='NATIVE_STAGE_FAILED_PRESERVED'
    elif fault=='trace-drop':previous['trace_audit']['writer']['dropped']=1
    elif fault=='grip':previous['reference_qualification']['windows'][-1]['gripper']['command_consumed']=False
    elif fault=='knots':previous['reference_qualification']['raw_knot_consumed'][-1]=False
    else:
        lease=path.parent.parent/'control/LEASE.json';d=json.loads(lease.read_text());d['status']='ACTIVE';put(lease,d)
    native['previous_stage_sha256']=put(path,previous)
    with pytest.raises(ValueError):v3.recipe_config(native)


@pytest.mark.parametrize('fault',['max-clock','recipe-clock','task','score','frame','capsule','scope','dt','camera-mode','camera-override'])
def test_rehashed_clock_task_frame_or_source_changes_reject_before_physics(recipe,fault):
    native,_,_,episode,_,_=recipe
    if fault=='max-clock':native['max_ticks']+=25
    elif fault=='recipe-clock':
        episode['planned_end_atom']+=1
        native['episode_manifest_sha256']=put(native['episode_manifest'],episode);native['max_ticks']+=25
    elif fault=='task':
        task=json.loads(Path(native['paused_target_task']).read_text());task['language']='Unregistered task'
        native['paused_target_task_file_sha256']=put(native['paused_target_task'],task)
    elif fault=='score':
        score=score_contract();score['minimum_signed_progress_m']=0
        native['paused_score_contract_file_sha256']=put(native['paused_score_contract'],score)
    elif fault=='frame':native['paused_source_from_target'][0][3]+=.001
    elif fault=='capsule':(Path(native['source_root'])/v3.SOURCE_MEMBERS[-1]).write_text('changed')
    elif fault=='scope':native['fusion_binding']='/not-the-native-P1-route'
    elif fault=='camera-mode':native['posture_query_camera_recipe']='UNRECOGNIZED_CAMERA_RIG'
    elif fault=='camera-override':native['camera_override']=dict(fovy=55.)
    else:native['physics_dt_s']=.004
    with pytest.raises(ValueError):v3.recipe_config(native)


@pytest.mark.parametrize('fault',['boundary3703','boundary-offone','schema','mode','extra','recipe','scale','returned-action','producer-phase','producer-pin','capture-index'])
def test_pair_v2_requires_actual_phase_and_derived_endpoint_not_a_free_boundary(recipe,fault):
    native,gpu,pair,*_=recipe
    if fault=='boundary3703':pair['native_boundary_step']=3703
    elif fault=='boundary-offone':pair['native_boundary_step']+=25
    elif fault=='schema':pair['schema']=bridge.SCHEMA
    elif fault=='mode':gpu['snapshot_binding_mode']=bridge.MODE
    elif fault=='extra':pair['caller_says_paused']=True
    elif fault=='recipe':pair['physical_recipe_phase']=v3.PHASE
    elif fault=='scale':pair['geometry_scale']=.5
    elif fault=='returned-action':pair['returned_VLA_action_installations']=1
    elif fault=='producer-phase':native['phase']=posture.PHASE
    elif fault=='capture-index':pair['capture_index']=1
    else:pair['source_posture_episode_sha256']='0'*64
    bind(native,gpu,pair)
    with pytest.raises(ValueError):inference.validate_spec(gpu)


@pytest.mark.parametrize('fault',['native-closed','gpu-closed','registry','stop'])
def test_both_actual_active_registrations_required_before_physical_entry(recipe,monkeypatch,fault):
    native,gpu,_,_,_,_=recipe
    if fault in ('native-closed','gpu-closed'):
        target=native if fault=='native-closed' else gpu
        lease=json.loads(Path(target['lease']).read_text());lease['status']='CLOSED';put(target['lease'],lease)
    elif fault=='registry':put(gpu['registry'],dict(entries=[]))
    else:put(Path(gpu['registered_root'])/'STOP.json',{})
    prepared=[]
    adapter=SimpleNamespace(spec=native,prepare_dynamic=lambda:prepared.append(True))
    with pytest.raises(ValueError):v3.run(adapter)
    assert prepared==[]


def test_old_pair_v1_remains_exact_3703(tmp_path,monkeypatch):
    gpu,adapter,pair,_,_=setup_pair(tmp_path,monkeypatch)
    pair['native_boundary_step']=5853
    gpu['paused_query_pair_sha256']=put(gpu['paused_query_pair'],pair)
    with pytest.raises(ValueError):bridge.validate_pair_spec(gpu)


def test_full_program_saved_cancelled_and_sealed_before_single_query_without_spec_mutation(recipe,tmp_path,monkeypatch):
    native,gpu,_,episode,_,_=recipe
    output=Path(native['registered_root'])/'native-01';output.mkdir()
    class Adapter(MeasuredAdapter):
        def cancel_reference(self):
            self.cancelled=True
            return dict(kind='operator_cancel',mode='HOLD',native_step=self.step)
        def seal_trace(self,step):
            super().seal_trace(step);self.trace_sealed=True
    adapter=Adapter(native,output);setup_flow(monkeypatch,adapter)
    original=copy.deepcopy(native);calls=[]
    def query(a,**kw):
        physical=output/'TRAJECTORY_QUALIFICATION.private.json'
        saved=json.loads(physical.read_text())
        assert a is adapter and a.cancelled and a.trace_sealed and a.step==v3.endpoint(episode)
        assert saved['actual_spec_phase']==v3.PHASE and saved['physical_recipe_phase']==posture.PHASE
        assert saved['all_40_knots_consumed'] and saved['source_posture_diagnostic']['residual_basis'].startswith('ACTUAL_MEASURED')
        assert not saved['source_posture_diagnostic']['full_scale_source_pose_close']
        calls.append((len(a.installs),len(a.grip_packets)))
        return dict(schema=v3.PAIR_SCHEMA,native_integrations_by_bridge=0,query_result=dict(model_result_checked=True))
    monkeypatch.setattr(bridge,'capture_and_wait_query',query)
    report=v3.run(adapter)
    assert native==original and native['phase']==report['episode']['phase']==v3.PHASE
    assert len(calls)==1 and calls[0]==(len(adapter.installs),len(adapter.grip_packets))
    assert report['actual_VLA_inferences']==report['VLA_queries']==1 and report['VLA_action_installations']==0
    assert not report['full_task_success'] and not report['camera_equivalence']
    assert report['native_endpoint_step']==v3.endpoint(episode)
    assert digest(report['physical_report_path'])==report['physical_report_sha256']
    assert (output/'SOURCE_POSTURE_PAUSED_QUERY.private.json').is_file()


@pytest.mark.parametrize('fault',['query-failure','nested-spec-mutation'])
def test_query_failure_retains_complete_calibration_and_explicit_nonretryable_failure(recipe,monkeypatch,fault):
    native,_,_,episode,_,_=recipe
    output=Path(native['registered_root'])/'native-01';output.mkdir()
    class Adapter(MeasuredAdapter):
        def cancel_reference(self):return dict(kind='operator_cancel',mode='HOLD',native_step=self.step)
        def seal_trace(self,step):super().seal_trace(step);self.trace_sealed=True
    adapter=Adapter(native,output);setup_flow(monkeypatch,adapter)
    calls=[]
    def fail(a,**kwargs):
        calls.append(True)
        if fault=='query-failure':raise RuntimeError('SYNTHETIC_GPU_FAILED_NO_RETRY')
        a.spec['source_files']['nested-mutation']='0'*64
        return dict(schema=v3.PAIR_SCHEMA,native_integrations_by_bridge=0)
    monkeypatch.setattr(bridge,'capture_and_wait_query',fail)
    with pytest.raises((RuntimeError,ValueError)):
        v3.run(adapter)
    physical=output/'TRAJECTORY_QUALIFICATION.private.json'
    saved=json.loads(physical.read_text());failure=json.loads((output/'SOURCE_POSTURE_QUERY_FAILED.private.json').read_text())
    assert saved['all_40_knots_consumed'] and 'source_posture_diagnostic' in saved
    assert failure['physical_report_sha256']==digest(physical) and failure['retry_authorized'] is False
    assert failure['returned_VLA_action_installations']==0 and len(calls)==1
    assert adapter.step==v3.endpoint(episode)
    assert adapter.reference_qualification==saved
    assert not (output/'SOURCE_POSTURE_PAUSED_QUERY.private.json').exists()


def capture_pair_v2(tmp_path,monkeypatch):
    """Real pause/capture/readback seam; immutable recipe is tested separately."""
    gpu,adapter,pair,events,calls=setup_pair(tmp_path,monkeypatch)
    raw=np.repeat(np.array([0,0,0,1,0,0,0,1,0,1.])[None],40,axis=0)
    inputs=dict(raw=raw,query_tcp=np.eye(4),frame_map=np.eye(4))
    episode=new_episode(Path(adapter.spec['registered_root']),inputs)
    end=v3.endpoint(episode)
    native=adapter.spec
    native.update(phase=v3.PHASE,max_ticks=end,episode_manifest=str(Path(native['registered_root'])/'EPISODE.json'))
    native['episode_manifest_sha256']=put(native['episode_manifest'],episode)
    pair.update(schema=v3.PAIR_SCHEMA,native_boundary_step=end,producer_phase=v3.PHASE,
        physical_recipe_phase=posture.PHASE,source_posture_episode_sha256=native['episode_manifest_sha256'],
        posture_source_observation_sha256=posture.POSTURE_OBSERVATION,geometry_scale=1.,time_stretch_factor=2,
        returned_VLA_action_installations=0,camera_recipe=v3.CAMERA_RECIPE,source_camera_identity_sha256=None)
    gpu['snapshot_binding_mode']=v3.PAIR_MODE
    bind(native,gpu,pair)
    adapter.fixture_binding=replace(adapter.fixture_binding,target_runtime_sha256=digest(Path(native['registered_root'])/'EXECUTION_SPEC.json'))
    packet=bytearray((adapter.output/'state.bin').read_bytes());struct.pack_into('<Qd',packet,8,end,end*.002)
    (adapter.output/'state.bin').write_bytes(packet)
    ack=json.loads((adapter.output/'pause_ack.json').read_text());ack.update(step=end,time_s=end*.002)
    put(adapter.output/'pause_ack.json',ack);(adapter.output/'step_quota').write_text(str(end)+'\n')
    lines=(adapter.output/'quota.binding').read_text().splitlines();lines[4]=str(end)
    (adapter.output/'quota.binding').write_text('\n'.join(lines)+'\n')
    from go2arx5_vla.world_model import fusion_paused_observation
    original=fusion_paused_observation.FrameRecorder
    class Recorder(original):
        def save(self,index,packet):
            value=super().save(index,packet)
            path=self.folder/'images'/f'{index:06d}-calibration.json'
            calibration=json.loads(path.read_text())
            for camera in calibration.values():camera['timestamp_s']=packet['time_s']
            put(path,calibration)
            return value
    monkeypatch.setattr(fusion_paused_observation,'FrameRecorder',Recorder)
    monkeypatch.setattr(v3,'recipe_config',lambda spec:(episode,inputs,adapter.fixture_task))
    return gpu,adapter,pair,events,calls


def test_v2_real_capture_one_original_generate_and_pause_readback_at_new_absolute_endpoint(tmp_path,monkeypatch):
    gpu,adapter,pair,events,_=capture_pair_v2(tmp_path,monkeypatch)
    draw=install_fake_draw(monkeypatch)
    packet=(adapter.output/'state.bin').read_bytes();quota=(adapter.output/'step_quota').read_bytes()
    def execute_once(seconds):
        assert seconds==.05
        output=Path(gpu['registered_root'])/'fit-01';output.mkdir()
        infer.infer(gpu,output);close_gpu(gpu)
    monkeypatch.setattr(bridge.time,'sleep',execute_once)
    value=bridge.capture_and_wait_query(adapter,binding=adapter.fixture_binding,task=adapter.fixture_task,
        task_sha256=gpu['paused_task_sha256'],gpu_spec_path=Path(gpu['registered_root'])/'EXECUTION_SPEC.json')
    assert draw.loads==draw.queries==1 and events.saves==[(0,pair['native_boundary_step'])]
    assert pair['native_boundary_step']>3703 and value['schema']==v3.PAIR_SCHEMA
    assert value['query_result']['model_result_checked'] and not value['online_action_execution']
    assert (adapter.output/'state.bin').read_bytes()==packet and (adapter.output/'step_quota').read_bytes()==quota
    body=unseal(json.loads(Path(gpu['recorded_observation']).read_text()))
    assert body['native_step']==pair['native_boundary_step'] and body['native_time_s']==pair['native_boundary_step']*.002
    result=json.loads((Path(gpu['registered_root'])/'fit-01/RESULT.private.json').read_text())
    assert result['producer_phase']==v3.PHASE and result['physical_recipe_phase']==posture.PHASE
    assert result['observation_binding']['snapshot_binding_mode']==v3.PAIR_MODE
    assert not result['camera_equivalence'] and result['returned_VLA_action_installations']==0
    with np.load(Path(gpu['registered_root'])/'fit-01/REAL_VLA_ACTION.npz',allow_pickle=False) as arrays:
        assert arrays['physical_H40'].shape==(40,10) and arrays['physical_H40'][-1,-1]==.399
        assert np.all(arrays['normalized']==1.25)
    with pytest.raises((ValueError,FileExistsError)):
        bridge.capture_and_wait_query(adapter,binding=adapter.fixture_binding,task=adapter.fixture_task,
            task_sha256=gpu['paused_task_sha256'],gpu_spec_path=Path(gpu['registered_root'])/'EXECUTION_SPEC.json')
    assert draw.queries==1


@pytest.mark.parametrize('fault',['quota','FSM','new-ACK','GPU-failure'])
def test_v2_wait_rejects_changed_pause_or_failed_query_without_replay(tmp_path,monkeypatch,fault):
    gpu,adapter,pair,events,_=capture_pair_v2(tmp_path,monkeypatch)
    draw=install_fake_draw(monkeypatch)
    def corrupt(seconds):
        if fault=='quota':(adapter.output/'step_quota').write_text(str(pair['native_boundary_step']+25)+'\n')
        elif fault=='new-ACK':
            events.fresh_ack=False
            path=adapter.output/'pause_ack.json';ack=json.loads(path.read_text());ack['ack_sequence']=1;put(path,ack)
        elif fault=='GPU-failure':put(Path(gpu['registered_root'])/'fit-01/FAILED.private.json',dict(failure='synthetic'))
        else:
            path=adapter.output/'pause_ack.json';ack=json.loads(path.read_text());ack['leg_history_steps']+=1;put(path,ack)
    monkeypatch.setattr(bridge.time,'sleep',corrupt)
    reason={'quota':'PAUSE_ACK_AND_EXACT_CURRENT_STEP_QUOTA',
        'FSM':'PHYSICS_FSM_OR_LEG_HISTORY_ADVANCED_WHILE_PAUSED',
        'new-ACK':'FRESH_ACTUAL_NATIVE_PAUSE_ACK_REQUIRED','GPU-failure':'GPU_QUERY_FAILED_NO_RETRY'}[fault]
    with pytest.raises(ValueError,match=reason):
        bridge.capture_and_wait_query(adapter,binding=adapter.fixture_binding,task=adapter.fixture_task,
            task_sha256=gpu['paused_task_sha256'],gpu_spec_path=Path(gpu['registered_root'])/'EXECUTION_SPEC.json')
    assert events.saves==[(0,pair['native_boundary_step'])] and draw.queries==draw.loads==0
    assert (Path(gpu['recorded_observation']).parent/'PAIR_SNAPSHOT_READY.private.json').is_file()


def test_new_phase_native_routes_and_original_worker_check_cli(recipe,tmp_path,monkeypatch,capsys):
    native,_,_,episode,_,previous=recipe
    assert auto1_native.phase_limit(v3.PHASE,previous,native)==v3.endpoint(episode)
    assert target_sequential.native_task_bucket(v3.PHASE,True)=='64'
    adapter=target_sequential.TargetSequentialAdapter(native,tmp_path)
    sentinel=object();monkeypatch.setattr(v3,'run',lambda a:sentinel if a is adapter else None)
    assert adapter.dynamic_diagnostic() is sentinel
    calls=[]
    monkeypatch.setitem(sys.modules,'scripts.wm02.pilot1_native_worker',SimpleNamespace(main=lambda argv:calls.append(argv) or 75))
    path=Path(native['registered_root'])/'EXECUTION_SPEC.json'
    assert check_cli.main(['--spec',str(path),'--check-only'])==0 and calls==[]
    assert json.loads(capsys.readouterr().out)['native_execution_authorized'] is False
    assert check_cli.main(['--spec',str(path),'--execute'])==75
    assert calls==[['--spec',str(path)]]


def test_v3_actual_trace524_loader_order_and_deferred_gripper_route(recipe,tmp_path,monkeypatch):
    native,*_=recipe
    adapter=MeasuredAdapter(native,tmp_path);adapter.prepare_dynamic()
    assert trajectory_native.actual_trace_binding(adapter)['actual_trace_capacity']==524288
    from go2arx5_vla.world_model.init_trace_reader import RECORD
    (tmp_path/'init.trace').write_bytes(struct.pack('<8sII',b'WM02IT01',RECORD.size,131072))
    with pytest.raises(ValueError,match='ACTUAL_NEW_TRACE_HEADER_CAPACITY_524288'):
        trajectory_native.actual_trace_binding(adapter)
    command=dict(native,xvfb_run='/fixture/xvfb',outer_library='/fixture/outer')
    argv,environment=target_sequential.native_display_command(command,tmp_path,dict(LD_PRELOAD='unsafe'))
    assert 'LD_PRELOAD='+':'.join(command[name] for name in ('trace_library','outer_library','observer_library')) in argv
    assert 'LD_PRELOAD' not in environment
    from scripts.pi0294 import task_state
    from go2arx5_vla.world_model import fusion_trajectory_qualification_contract as geometry
    writes=[];checks=[]
    monkeypatch.setattr(task_state,'write_gripper',lambda *args:writes.append(args))
    monkeypatch.setattr(trajectory_native,'validate_adapter_install',lambda *args:checks.append(args) or {})
    owner=target_sequential.TargetSequentialAdapter(dict(phase=v3.PHASE),tmp_path)
    owner.reference_sequence=-1;owner.latest=dict(ee=SimpleNamespace(time=5.006))
    owner.wait_source=lambda:None;owner.observe=lambda:dict(step=2503,time_s=5.006)
    owner.command_event=lambda event:None;owner.refs=object();events=[]
    def publish(topic,wire):
        import hashlib
        identity=hashlib.sha256(wire.encode()).hexdigest()
        events.extend(dict(kind=kind,ref_id=identity) for kind in ('accept','installed'))
    owner.publish=publish;owner.reference_events=lambda:events;owner.spin=lambda seconds,predicate:predicate()
    owner.install_reference(geometry.hold_preview(np.eye(4)))
    assert len(checks)==1 and writes==[]


def test_import_and_cli_help_do_not_load_model_renderer_or_native_backend():
    code=('import sys;import go2arx5_vla.world_model.fusion_source_posture_paused_query_v3;'
        'import scripts.wm02.fusion_source_posture_paused_query_check;'
        'assert not {"torch","mujoco","rclpy","transformers","gr00t"}&sys.modules.keys()')
    subprocess.run([sys.executable,'-B','-c',code],check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    subprocess.run([sys.executable,'-B','-m','scripts.wm02.fusion_source_posture_paused_query_check','--help'],
        check=True,capture_output=True)


def source_camera_recipe(recipe,monkeypatch):
    """Synthetic pinned audit through the owned typed-constructor seam."""
    from go2arx5_vla.world_model import fusion_source_camera_capture as capture
    from tests.world_model.test_fusion_source_camera_rig import fixture as rig_fixture
    import hashlib
    native,gpu,pair,episode,inputs,previous=recipe
    audit,_=rig_fixture();raw=json.dumps(audit,allow_nan=False).encode();pin=hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(v3,'SOURCE_CAMERA_IDENTITY',pin)
    monkeypatch.setattr(capture,'SOURCE_IDENTITY_SHA256',pin)
    monkeypatch.setattr(capture,'FRAME_MAP_SHA256',transform_hash(inputs['frame_map']))
    path=Path(native['registered_root'])/'SYNTHETIC_SOURCE_CAMERA_IDENTITY.json';path.write_bytes(raw)
    capsule=Path(native['source_root'])
    for name in v3.SOURCE_CAMERA_MEMBERS:
        member=capsule/name;member.parent.mkdir(parents=True,exist_ok=True);member.write_bytes((PROJECT/name).read_bytes())
        native['source_files'][name]=digest(member)
    gpu['source_files']=native['source_files'].copy()
    fixed=v3._fixed_episode(v3.SOURCE_CAMERA_RECIPE)
    episode=dict(fixed,run=episode['run'],geometry_scale=1.,planned_end_atom=episode['planned_end_atom'])
    native.update(posture_query_camera_recipe=v3.SOURCE_CAMERA_RECIPE,source_camera_identity=str(path),
        source_camera_identity_sha256=pin,episode_manifest_sha256=put(native['episode_manifest'],episode))
    pair.update(camera_recipe=v3.SOURCE_CAMERA_RECIPE,source_camera_identity_sha256=pin,
        source_posture_episode_sha256=native['episode_manifest_sha256'])
    bind(native,gpu,pair)
    return native,gpu,pair,episode,inputs,previous


def test_source_camera_requires_declared_typed_pinned_recipe_without_loading_renderer(recipe,monkeypatch):
    native,gpu,pair,episode,_,_=source_camera_recipe(recipe,monkeypatch)
    actual,inputs,_=v3.episode_config(native)
    assert actual==episode and inputs['capture_camera_recipe'].name==v3.SOURCE_CAMERA_RECIPE
    assert inputs['capture_camera_recipe'].source_identity_bytes==Path(native['source_camera_identity']).read_bytes()
    assert bridge.validate_pair_spec(gpu)[0]==pair
    np.testing.assert_array_equal(inputs['capture_camera_recipe'].T_source_from_target,native['paused_source_from_target'])
    adapter=SimpleNamespace(spec=native,output=Path(native['registered_root'])/'native-01')
    from go2arx5_vla.world_model.fusion_observation import TargetShadowBinding
    binding=TargetShadowBinding('ACTUAL_TARGET_WORLD',digest(native['model_path']),
        digest(Path(native['registered_root'])/'EXECUTION_SPEC.json'),
        np.asarray(native['paused_source_from_target']),native['paused_frame_map_sha256'])
    with pytest.raises(ValueError,match='EXACT_TYPED_PREDECLARED_SOURCE_CAMERA_CAPTURE_REQUIRED'):
        bridge.capture_and_wait_query(adapter,binding=binding,task=task_document(digest(native['model_path'])),
            task_sha256=native['paused_target_task_sha256'],gpu_spec_path=Path(gpu['registered_root'])/'EXECUTION_SPEC.json')
    assert not Path(gpu['recorded_observation']).exists()


@pytest.mark.parametrize('fault',['bytes','spec-pin','pair-pin','path','source','override'])
def test_source_camera_pins_and_unknown_overrides_fail_before_native_or_model(recipe,monkeypatch,fault):
    native,gpu,pair,*_=source_camera_recipe(recipe,monkeypatch)
    if fault=='bytes':Path(native['source_camera_identity']).write_text('{}')
    elif fault=='spec-pin':native['source_camera_identity_sha256']='0'*64
    elif fault=='pair-pin':pair['source_camera_identity_sha256']='0'*64
    elif fault=='path':native['source_camera_identity']='relative-audit.json'
    elif fault=='source':(Path(native['source_root'])/v3.SOURCE_CAMERA_MEMBERS[0]).write_text('changed')
    else:native['camera_override']=dict(fovy=55.)
    bind(native,gpu,pair)
    with pytest.raises(ValueError):inference.validate_spec(gpu)


@pytest.mark.parametrize('fault',['missing','old-camera','audit','frame','rig'])
def test_declared_source_pair_rejects_missing_or_mismatched_sealed_sensor_before_model(fault):
    pair=dict(camera_recipe=v3.SOURCE_CAMERA_RECIPE,source_camera_identity_sha256=v3.SOURCE_CAMERA_IDENTITY,
              frame_map_sha256=posture.PINS['frame_map_sha256'])
    sensor=dict(schema='WM02_FUSION_SOURCE_CAMERA_CAPTURE_V1',status='ACTUAL_TARGET_PACKET_CPU_RENDERED',
        rig=v3.SOURCE_CAMERA_RECIPE,source_identity_audit_sha256=v3.SOURCE_CAMERA_IDENTITY,
        frame_map_sha256=pair['frame_map_sha256'])
    if fault=='missing':sensor=None
    elif fault=='old-camera':sensor['status']='ORIGINAL_NATIVE_CAMERA'
    elif fault=='audit':sensor['source_identity_audit_sha256']='0'*64
    elif fault=='frame':sensor['frame_map_sha256']='0'*64
    else:sensor['rig']='UNDECLARED_CAMERA'
    shadow=dict(target_observation=dict(sensor_adapter=sensor))
    with pytest.raises(ValueError,match='EXACT_DECLARED_SOURCE_RIG'):v3.validate_capture_camera(pair,shadow)


def test_original_pair_v1_rejects_typed_source_camera_override_before_capture(tmp_path,monkeypatch):
    gpu,adapter,_,events,_=setup_pair(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='PAIR_V1_ORIGINAL_CAMERA_DEFAULT_UNCHANGED'):
        bridge.capture_and_wait_query(adapter,binding=adapter.fixture_binding,task=adapter.fixture_task,
            task_sha256=gpu['paused_task_sha256'],gpu_spec_path=Path(gpu['registered_root'])/'EXECUTION_SPEC.json',
            camera_recipe=object())
    assert events.saves==[]


def test_source_rig_v2_capture_to_one_original_gpu_query_retains_eight_members(tmp_path,monkeypatch):
    """Tiny CPU FK with synthetic renderer/model doubles; every mj_step blocked."""
    from scripts.pi0294.task_state import ActualState
    from tests.world_model.test_fusion_source_camera_capture import (
        adapter_fixture,recipe as camera_recipe,forbid_step)
    from go2arx5_vla.world_model import fusion_source_camera_rig as rig
    original_update=ActualState.update
    gpu,old_adapter,pair,_,_=setup_pair(tmp_path/'base',monkeypatch)
    adapter,binding,task,_=adapter_fixture(tmp_path/'camera',monkeypatch)
    monkeypatch.setattr(ActualState,'update',original_update)
    camera=camera_recipe(monkeypatch);forbid_step(monkeypatch)
    import hashlib
    pin=hashlib.sha256(camera.source_identity_bytes).hexdigest()
    monkeypatch.setattr(v3,'SOURCE_CAMERA_IDENTITY',pin)
    native=dict(old_adapter.spec,**adapter.spec)
    raw=np.repeat(np.array([0,0,0,1,0,0,0,1,0,1.])[None],40,axis=0)
    inputs=dict(raw=raw,query_tcp=np.eye(4),frame_map=np.eye(4))
    episode=new_episode(Path(native['registered_root']),inputs)
    episode.update(capture_camera_recipe=v3.SOURCE_CAMERA_RECIPE,source_camera_identity_sha256=pin)
    end=v3.endpoint(episode)
    native.update(phase=v3.PHASE,max_ticks=end,episode_manifest=str(Path(native['registered_root'])/'EPISODE.json'),
        lease=str(Path(native['registered_root'])/'control/LEASE.json'),
        paused_target_task_sha256=document_hash(task),paused_frame_map_sha256=binding.expected_transform_sha256)
    native['episode_manifest_sha256']=put(native['episode_manifest'],episode)
    adapter.spec=native
    gpu.update(snapshot_binding_mode=v3.PAIR_MODE,producer_native_spec=str(Path(native['registered_root'])/'EXECUTION_SPEC.json'),
        paused_task_sha256=document_hash(task),recorded_observation=str(adapter.output/'PAUSED_OBSERVATIONS'/gpu['query_uuid']/'PAUSED_OBSERVATION.private.json'))
    pair.update(schema=v3.PAIR_SCHEMA,native_root=native['registered_root'],native_boundary_step=end,
        task_sha256=gpu['paused_task_sha256'],scene_sha256=digest(native['model_path']),
        frame_map_sha256=binding.expected_transform_sha256,producer_phase=v3.PHASE,
        physical_recipe_phase=posture.PHASE,source_posture_episode_sha256=native['episode_manifest_sha256'],
        posture_source_observation_sha256=posture.POSTURE_OBSERVATION,geometry_scale=1.,time_stretch_factor=2,
        returned_VLA_action_installations=0,camera_recipe=v3.SOURCE_CAMERA_RECIPE,source_camera_identity_sha256=pin)
    bind(native,gpu,pair)
    binding=replace(binding,target_runtime_sha256=digest(Path(native['registered_root'])/'EXECUTION_SPEC.json'))
    packet=adapter.observe();packet.update(step=end,time_s=end*.002)
    wire=bytearray((adapter.output/'state.bin').read_bytes());struct.pack_into('<Qd',wire,8,end,end*.002)
    (adapter.output/'state.bin').write_bytes(wire)
    ack=json.loads((adapter.output/'pause_ack.json').read_text());ack.update(step=end,time_s=end*.002)
    put(adapter.output/'pause_ack.json',ack);(adapter.output/'step_quota').write_text(str(end)+'\n')
    lines=(adapter.output/'quota.binding').read_text().splitlines();lines[4]=str(end)
    (adapter.output/'quota.binding').write_text('\n'.join(lines)+'\n')
    def spin(seconds,predicate):
        path=adapter.output/'pause_ack.json';value=json.loads(path.read_text());value['ack_sequence']+=1;put(path,value)
        return predicate()
    adapter.spin=spin
    # The full real recipe is separately covered above. Here only that pure
    # immutable-boundary seam is doubled to join actual synthetic camera FK.
    monkeypatch.setattr(v3,'recipe_config',lambda spec:(episode,inputs,task))
    draw=install_fake_draw(monkeypatch)
    def execute_once(seconds):
        output=Path(gpu['registered_root'])/'fit-01';output.mkdir()
        infer.infer(gpu,output);close_gpu(gpu)
    monkeypatch.setattr(bridge.time,'sleep',execute_once)
    result=bridge.capture_and_wait_query(adapter,binding=binding,task=task,
        task_sha256=gpu['paused_task_sha256'],gpu_spec_path=Path(gpu['registered_root'])/'EXECUTION_SPEC.json',
        camera_recipe=camera)
    assert draw.loads==draw.queries==1 and result['query_result']['model_result_checked']
    record=json.loads(Path(gpu['recorded_observation']).read_text());body=unseal(record)
    shadow=unseal(body['original_shadow_ABI']);sensor=shadow['target_observation']['sensor_adapter']
    assert len(body['files'])==8 and sensor['rig']==v3.SOURCE_CAMERA_RECIPE
    assert sensor['native_target_camera_records']['body_rgb']['fovy_deg']==75.
    assert shadow['target_observation']['cameras'][rig.OVERLAY_CAMERA]['fovy_deg']==55.
    assert (adapter.output/'state.bin').read_bytes()==wire
    folder=Path(gpu['recorded_observation']).parent
    assert (folder/'images/000000-wrist_rgb.png').read_bytes()==(folder/'native-camera/images/000000-wrist_rgb.png').read_bytes()
    identity=json.loads((Path(gpu['registered_root'])/'fit-01/RESULT.private.json').read_text())['observation_binding']
    assert identity['files']==body['files'] and identity['producer_phase']==v3.PHASE
    assert identity['capture_camera_recipe']==v3.SOURCE_CAMERA_RECIPE and identity['source_camera_identity_sha256']==pin
    assert not identity['camera_equivalence']
    retained=folder/'native-camera/images/000000-body_rgb.png';retained.write_bytes(b'changed original native view')
    with pytest.raises(ValueError):inference.read_payload(gpu)
    assert draw.queries==1
