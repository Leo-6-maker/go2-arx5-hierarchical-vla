"""Pure synthetic CLOSED V9 archive ABI; no simulator, FK, model, or query."""
import base64
import copy
import io
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from scripts.pi0292.relay_protocol import digest as dh, sealed, unseal
from scripts.pi0294.task_state import ActualState
from go2arx5_vla.world_model import fusion_graded_query_archive_v9 as archive
from go2arx5_vla.world_model import fusion_graded_prompt_control_v2 as control
from go2arx5_vla.world_model import fusion_paused_observation as observation
from go2arx5_vla.world_model import fusion_source_camera_capture as sensor_module
from go2arx5_vla.world_model.fusion_source_camera_rig import OVERLAY_CAMERA, record_hash
from go2arx5_vla.world_model.fusion_observation import TargetShadowBinding, transform_hash
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_fusion_observation import fixture as observation_fixture, png

QUERY = 'e13d8ed0-ce0a-41a9-b6bb-3117a0db9d10'
LANGUAGE = 'Pick up the red cube, place it at the target, release it, and retreat.'
ROOT = Path(__file__).resolve().parents[2]


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return digest(path)


def manifest_bytes(view):
    return json.dumps(view, sort_keys=True, allow_nan=False).encode()


@pytest.fixture
def retained(tmp_path, monkeypatch):
    archive._stream_pin.cache_clear()
    nr, gr, source = (tmp_path/name for name in ('native','gpu','capsule'))
    native, gpu = nr/'native-01', gr/'fit-01'
    native.mkdir(parents=True); gpu.mkdir(parents=True)
    files = {}
    for name in archive.SOURCE_MEMBERS:
        dest = source/name; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes()); files[name] = digest(dest)
    inventory = {'model.safetensors':'1'*64, 'processor/config.json':'2'*64}
    monkeypatch.setattr(archive, 'INVENTORY', dh(inventory))
    monkeypatch.setattr(control, 'INVENTORY', dh(inventory))
    images = [dict(camera=key,png_base64=base64.b64encode(png(color)).decode(),
                   sha256=__import__('hashlib').sha256(png(color)).hexdigest())
              for key,color in (('external_rgb',21),('wrist_rgb',81))]
    original = dict(snapshot_uuid='ca8cb3a0-3b6f-414f-9a20-c6369fc24d48', language=LANGUAGE,
        state_eef10=[.4,-.1,.5,1.,0.,0.,0.,1.,0.,.95], images=images,
        frame='PI0291_MUJOCO_WORLD',tcp='ARX5_NATIVE_SITE_x5_ee')
    source_path = tmp_path/'source71558.json'; source_pin = put(source_path,sealed(dict(payload=original)))
    monkeypatch.setattr(archive,'SOURCE_OBSERVATION',source_pin)
    _, args, measured, frame = observation_fixture(34303)
    monkeypatch.setattr(sensor_module,'FRAME_MAP_SHA256',transform_hash(frame))
    score = dict(schema='WM02_GRADED_FULL_TASK_PROPOSAL_DIAGNOSTIC_SCORE_V9', full_task_success_claim=False)
    score_path = nr/'SCORE.json'; score_pin = put(score_path,score)
    task = dict(schema=observation.TASK_SCHEMA,task_id='ORIGINAL71558_FULL_TASK_AT_GRADED_TARGET_STATE_V9',
        language=LANGUAGE,scene_sha256='c'*64,score_contract_sha256=dh(score),language_provenance='EXACT_SOURCE_TASK_PROMPT')
    task_path = nr/'TASK.json'; task_pin = put(task_path,task)
    episode = dict(schema='WM02_FUSION_TASK_PHASE_POSE_SLOW_FEEDBACK_EPISODE_V9', phase=archive.RECIPE_PHASE,
        run=str(nr),geometry_scale=.5,ramp_atoms=768,planned_end_atom=1216,
        feedback_decision_native_steps=[23103+900*i for i in range(12)])
    ep_path = nr/'EPISODE.json'; ep_pin = put(ep_path,episode)
    pair_path = nr/'QUERY_PAIR.private.json'
    pair = dict(schema=archive.PAIR_SCHEMA,pair_uuid='1f5dabca-2ffe-4058-be59-1b0b9a2839a5',query_uuid=QUERY,
        native_root=str(nr),gpu_root=str(gr),native_run_id='native-01',native_boundary_step=34303,native_endpoint_step=66503,
        capture_index=0,producer_phase=archive.PHASE,source_sha='9'*40,seed=366104,task_sha256=dh(task),
        frame_map_sha256=transform_hash(frame),scene_sha256='c'*64,checkpoint_manifest_sha256=archive.CHECKPOINT,
        max_queries=1,native_integrations_by_bridge=0,source_camera_identity_sha256=archive.SOURCE_CAMERA_IDENTITY,
        camera_recipe='SIM_EXTERNAL_WORLD_SOURCE_RIG',task_condition='SOURCE71558_ORIGINAL_FULL_TASK_NO_PARAPHRASE',deadline_epoch=1.)
    pair_pin = put(pair_path,pair)
    common = dict(source_sha='9'*40,source_root=str(source),source_files=files,owner='SYNTHETIC_OWNER')
    n = dict(common,registered_root=str(nr),lease=str(nr/'control/LEASE.json'),scope='WM02_PILOT1_SIM_ONLY',
        phase=archive.PHASE,kind='FUSION_TASK_PHASE_LIVE_SKILL_V9',physics=True,physics_dt_s=.002,WBC_changes=False,
        max_ticks=66503,live_skill_start_step=34303,episode_manifest=str(ep_path),episode_manifest_sha256=ep_pin,
        paused_query_pair=str(pair_path),paused_query_pair_sha256=pair_pin,paused_query_gpu_spec=str(gr/'EXECUTION_SPEC.json'),
        paused_source_from_target=frame.tolist(),task_phase_source_observation=str(source_path),
        paused_target_task=str(task_path),paused_target_task_file_sha256=task_pin,paused_target_task_sha256=dh(task),
        paused_score_contract=str(score_path),paused_score_contract_file_sha256=score_pin,deadline_epoch=1.)
    npth = nr/'EXECUTION_SPEC.json'; n_pin = put(npth,n)
    folder = native/'PAUSED_OBSERVATIONS'/QUERY
    capture_path = folder/'PAUSED_OBSERVATION.private.json'
    g = dict(common,registered_root=str(gr),lease=str(gr/'control/LEASE.json'),scope='WM02_FUSION_REFERENCE_SIM_ONLY',
        physics=False,kind='FROZEN_VLA_LIVE_PAUSED_QUERY',snapshot_binding_mode=archive.PAIR_MODE,max_queries=1,seed=366104,
        profile='wm02-fusion-vla-shadow',device='cuda',execution_site='CORE_FUSION_NEURAL',inference_mode='ACTUAL_LIVE_PAUSED_VLA_INFERENCE',
        query_uuid=QUERY,numeric_mode='ORIGINAL_FP32_WEIGHTS_BF16_AUTOCAST',checkpoint_manifest_sha256=archive.CHECKPOINT,
        checkpoint_files=inventory,producer_native_spec=str(npth),producer_native_spec_sha256=n_pin,
        paused_query_pair=str(pair_path),paused_query_pair_sha256=pair_pin,paused_task_sha256=dh(task),
        recorded_observation=str(capture_path),deadline_epoch=1.)
    gpth = gr/'EXECUTION_SPEC.json'; g_pin = put(gpth,g)
    binding = TargetShadowBinding('ACTUAL_TARGET_WORLD','c'*64,n_pin,frame,transform_hash(frame))
    args.update(capture=dict(index=0,native_step=34303,time_s=34303*.002),language=LANGUAGE,snapshot_uuid=QUERY)
    monkeypatch.setattr(ActualState,'update',lambda self,packet:measured)
    shadow = unseal(binding.bind(**args))
    target = shadow['target_observation']; cameras = copy.deepcopy(target['cameras'])
    external = copy.deepcopy(cameras['body_rgb'])
    external.update(name=OVERLAY_CAMERA,native_camera=False,rendering_copy_slot='body_rgb',fovy_deg=55.)
    target['cameras'] = {OVERLAY_CAMERA:external,'wrist_rgb':copy.deepcopy(cameras['wrist_rgb'])}
    sensor = dict(schema=sensor_module.SCHEMA,status='ACTUAL_TARGET_PACKET_CPU_RENDERED',rig='SIM_EXTERNAL_WORLD_SOURCE_RIG',
        source_identity_audit_sha256=archive.SOURCE_CAMERA_IDENTITY,frame_map_sha256=transform_hash(frame),
        policy_camera_role_mapping={'external_rgb':OVERLAY_CAMERA,'wrist_rgb':'wrist_rgb'},
        rendering_slot_alias={'body_rgb':OVERLAY_CAMERA,'wrist_rgb':'wrist_rgb'},original_compiled_model_sha256='c'*64,
        packet_identity=dict(native_step=34303,time_s=34303*.002),native_target_camera_records=cameras,
        native_camera_records_sha256=record_hash(cameras),native_image_sha256=copy.deepcopy(target['image_sha256']),
        rendered_external_image_sha256=target['image_sha256']['body_rgb'],
        native_calibration_sha256=target['calibration_sha256'],declared_rendering_sensors={OVERLAY_CAMERA:copy.deepcopy(external)},
        effects=dict(renders=3,render_copy_FK_calls=2,integrations=0,model_queries=0))
    sensor.update({key:False for key in ('native_camera_equivalence','image_domain_equivalence','occlusion_equivalence',
        'target_embodiment_qualified','target_task_qualified','native_execution_authorized','source_images_supplied')})
    target['sensor_adapter']=sensor
    shadow['provenance']['policy_camera_sensor_mapping']=sensor['policy_camera_role_mapping']
    shadow = sealed(shadow)
    members = {'STATE.private.bin':args['native_state_bytes'],'BOUND_SHADOW_ABI.private.json':manifest_bytes(shadow),
        'images/000000-calibration.json':args['calibration_bytes'],
        'native-camera/images/000000-calibration.json':args['calibration_bytes']}
    for camera in ('body_rgb','wrist_rgb'):
        members['images/000000-'+camera+'.png']=args['image_bytes'][camera]
        members['native-camera/images/000000-'+camera+'.png']=args['image_bytes'][camera]
    for name,raw in members.items():
        path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    (native/'state.bin').write_bytes(args['native_state_bytes'])
    before = dict(step=34303,time_s=34303*.002,running=False,clock_paused=True,ack_sequence=1,
        motiontime=25790,leg_history_steps=3431,gait_indices=.39)
    after = dict(before,ack_sequence=2)
    body = dict(schema=observation.SCHEMA,input_mode=observation.INPUT_MODE,query_uuid=QUERY,snapshot_uuid=QUERY,
        capture=args['capture'],native_step=34303,native_time_s=34303*.002,task=task,task_sha256=dh(task),
        world=dict(execution_spec_sha256=n_pin,native_output=str(native),scene_sha256='c'*64),
        renderer=dict(GL_VENDOR='Mesa',GL_RENDERER='llvmpipe SYNTHETIC',backend='egl_mesa_cpu'),
        pause_before=before,pause_after=after,files={name:digest(folder/name) for name in members},original_shadow_ABI=shadow,
        provenance=dict(native_execution_authorized=False,VLA_inferences=0,physics_integrations=0))
    record=sealed(body); capture_pin=put(capture_path,record)
    payload=observation.paused_inference_payload(record)
    pins=dict(native_spec_sha256=n_pin,gpu_spec_sha256=g_pin,observation_sha256=capture_pin,pair_sha256=pair_pin,
        source_sha='9'*40,checkpoint_manifest_sha256=archive.CHECKPOINT,checkpoint_file_inventory_sha256=archive.INVENTORY,
        task_sha256=dh(task),frame_map_sha256=transform_hash(frame),query_uuid=QUERY,seed=366104,
        source_observation_sha256=source_pin,source_camera_identity_sha256=archive.SOURCE_CAMERA_IDENTITY,
        episode_sha256=ep_pin,noise_sha256='6'*64)
    for side,spec,root,code in (('native',n,nr,1),('gpu',g,gr,0)):
        pins[side+'_lease_sha256']=put(spec['lease'],dict(status='CLOSED',spec_sha256=pins[side+'_spec_sha256'],
            run_uuid=root.name,source_sha='9'*40,owner=spec['owner'],scope=spec['scope'],worker_exit_code=code))
        pins[side+'_worker_sha256']=put(root/'WORKER_EXIT.private.json',dict(exit_code=code,failure=None))
    raw=np.tile(np.asarray(original['state_eef10']),(40,1)); normalized=np.zeros((1,40,132),dtype=np.float32)
    buffer=io.BytesIO();np.savez(buffer,physical_H40=raw,normalized=normalized)
    action_path=gpu/'REAL_VLA_ACTION.npz';action_path.write_bytes(buffer.getvalue());pins['action_sha256']=digest(action_path)
    # Actual V9 GPU binding schema, independently declared here.
    identity=dict(query_uuid=QUERY,snapshot_uuid=QUERY,observation_sha256=capture_pin,sealed_body_sha256=record['sha256'],
        task_sha256=dh(task),task=task,world=body['world'],native_step=34303,native_time_s=34303*.002,
        files=body['files'],renderer=body['renderer'],input_mode=observation.INPUT_MODE,mapped_payload_sha256=dh(payload),
        target_camera_provenance=unseal(shadow)['provenance'],
        source_policy_view={k:v for k,v in unseal(shadow)['source_policy_view'].items() if k!='payload'},
        snapshot_binding_mode=archive.PAIR_MODE,pause_before=before,pause_after_capture=after,
        native_pause_after_query='NOT_CHECKED_BY_GPU_PROCESS',native_caller_must_verify_still_paused=True,
        native_execution_authorized=False,target_embodiment_qualified=False,target_task_qualified=False,online_closed_loop=False,
        source_outcomes_supplied=False,producer_phase=archive.PHASE,experiment_role='GRADED_NONCONTACT_DIAGNOSTIC',
        physical_recipe_phase=archive.RECIPE_PHASE,initialization_recipe_phase=archive.RECIPE_PHASE,
        initialization_geometry_scale=.5,native_boundary_step=34303,native_endpoint_step=66503,
        task_condition='SOURCE71558_ORIGINAL_FULL_TASK_NO_PARAPHRASE',capture_condition='PRESEAL_WBC_ORIGINAL_QUOTA_ACK_PAUSED',
        capture_camera_recipe='SIM_EXTERNAL_WORLD_SOURCE_RIG',source_camera_identity_sha256=archive.SOURCE_CAMERA_IDENTITY,
        returned_VLA_action_installations=0,camera_equivalence=False,old_initialization_readiness_relabelled=False,
        source_task_phase_equivalence=False,native_trace_seal_checked_by_GPU=False)
    gpu_result=dict(schema='WM02_FUSION_LIVE_PAUSED_INFERENCE_RESULT_V1',status='REAL_FROZEN_VLA_LIVE_PAUSED_QUERY_DONE',
        observation_binding=identity,source_sha='9'*40,query_uuid=QUERY,seed=366104,checkpoint_manifest_sha256=archive.CHECKPOINT,
        checkpoint_file_inventory_sha256=archive.INVENTORY,pair_sha256=pair_pin,action_sha256=pins['action_sha256'],
        execution_spec_sha256=g_pin,producer_native_spec_sha256=n_pin,recorded_observation_sha256=capture_pin,
        actual_model_loads=1,actual_VLA_inferences=1,native_integrations=0,GR00T_updates=0,WM_updates=0,Actor_updates=0,
        numeric_mode=g['numeric_mode'],inference_mode=g['inference_mode'],native_execution_authorized=False,returned_VLA_action_installations=0,
        precision=dict(mode='ORIGINAL_E0_BF16_AUTOCAST_FP32_MATMUL_HIGHEST',bf16_autocast=True,cuda_matmul_allow_tf32=False),
        noise=dict(shape=[1,40,132],dtype='torch.float32',sha256='6'*64))
    pins['gpu_result_sha256']=put(gpu/'RESULT.private.json',gpu_result)
    pause=dict(schema=observation.SCHEMA,query_uuid=QUERY,native_step=34303,native_time_s=34303*.002,
        unchanged_measured_packet_sha256=body['files']['STATE.private.bin'],native_execution_authorized=False,model_result_checked=False,
        pause_after_query=dict(after,ack_sequence=3))
    checked=dict(schema=gpu_result['schema'],query_uuid=QUERY,result_sha256=pins['gpu_result_sha256'],action_sha256=pins['action_sha256'],
        original_pause_revalidation=pause,model_result_checked=True,native_execution_authorized=False,original_certificate_and_ACK_still_required=True)
    query=dict(schema=archive.PAIR_SCHEMA,query_uuid=QUERY,pair_sha256=pair_pin,gpu_spec_sha256=g_pin,capture_sha256=capture_pin,
        native_integrations_by_bridge=0,native_execution_authorized=False,online_action_execution=False,query_result=checked)
    proposal=dict(schema='WM02_ACTUAL_GENERATED_LIVE_PROPOSAL_V9',native_step=34303,pre_effect=True,
        clips=0,original_Processor_extra_decodes=0,
        action_npz_sha256=pins['action_sha256'],raw_knots=40,original_decoded_physical_H40=raw.tolist(),observation_identity=identity,query=query)
    pins['proposal_sha256']=put(native/'LIVE_SKILL_GENERATED_PROPOSAL.private.json',proposal)
    failure=dict(schema='WM02_GRADED_LIVE_SKILL_FAILURE_V9',phase=archive.PHASE,native_step=34303,proposal_saved=True,
        failure='ValueError: TASK_PHASE_LIVE_SKILL_V8_'+archive.DURATION_FAILURE,query=query)
    pins['failure_sha256']=put(native/'LIVE_SKILL_FAILED.private.json',failure)
    native_result=dict(status='NATIVE_STAGE_FAILED_PRESERVED',phase=archive.PHASE,source_sha='9'*40,
        actual_target_integrations=34303,complete_atomic_transitions=1372,shutdown_clean=True,failure=failure['failure'])
    pins['native_result_sha256']=put(native/'RESULT.json',native_result)
    pins['initialization_sha256']=put(native/'LIVE_SKILL_INITIALIZATION_STAGE.private.json',dict(
        schema='SYNTHETIC_OPAQUE_HASH_PINNED_REPORT',windows='NOT_PARSED_OR_REPLAYED_BY_ARCHIVE'))
    target=dict(native_spec=str(npth),gpu_spec=str(gpth),pins=pins)
    return dict(target=target,n=n,g=g,nr=nr,gr=gr,native=native,gpu=gpu,folder=folder,body=body,record=record,
                payload=payload,proposal=proposal,failure=failure,result=gpu_result,pins=pins)


def read(value):
    target=value['target']
    return archive.read_archive(target['native_spec'],target['gpu_spec'],pins=target['pins'])


def short_view(value):
    return control.make_view(target_archive=value['target'],prompt_variant='CURRENT_SHORT')


def spec_for(value,view):
    raw=manifest_bytes(view)
    return dict(observation_mode=control.MODE,recorded_observation_sha256=__import__('hashlib').sha256(raw).hexdigest(),
        seed=366104,checkpoint_manifest_sha256=control.CHECKPOINT,checkpoint_files=value['g']['checkpoint_files'],
        numeric_mode='ORIGINAL_FP32_WEIGHTS_BF16_AUTOCAST',max_queries=1,inference_mode='OFFLINE_REAL_VLA_INFERENCE')


def resolve(value,view,spec=None):
    raw=manifest_bytes(view); spec=spec_for(value,view) if spec is None else spec
    return control.prompt_control_payload(raw,view_sha256=__import__('hashlib').sha256(raw).hexdigest(),query_spec=spec)


def test_closed_native_failure_valid_saved_query_and_only_short_language_differs(retained):
    saved=read(retained); assert saved['provenance']['closed_workers']['native']['exit_code']==1
    assert saved['provenance']['native_status']=='NATIVE_STAGE_FAILED_PRESERVED'
    original=control.make_view(target_archive=retained['target'],prompt_variant='ORIGINAL_71558')
    short=short_view(retained); payload,sidecar=resolve(retained,short)
    assert original['policy_payload_sha256']==dh(retained['payload'])
    assert original['reuse_existing_original_query'] and not short['reuse_existing_original_query']
    assert original['non_language_payload_sha256']==short['non_language_payload_sha256']
    assert payload['language']==control.SHORT_LANGUAGE
    assert {k:v for k,v in payload.items() if k!='language'}=={k:v for k,v in retained['payload'].items() if k!='language'}
    assert sidecar['base_provenance']['original_task']['language']==LANGUAGE
    assert sidecar['base_provenance']['original_score']['full_task_success_claim'] is False
    assert sidecar['expected_noise_sha256']=='6'*64 and sidecar['noise_match_requires_new_result']
    assert not sidecar['native_execution_authorized'] and not sidecar['task_success_established']
    assert sidecar['new_queries']==sidecar['new_integrations']==sidecar['processor_decodes']==0
    with pytest.raises(ValueError,match='ONLY_CURRENT_SHORT'):
        resolve(retained,original)


@pytest.mark.parametrize('side,code',[('native',0),('gpu',1)])
def test_rehashed_closed_metadata_cannot_promote_failure_or_accept_failed_gpu(retained,side,code):
    root=retained['nr'] if side=='native' else retained['gr']
    lease=json.loads((root/'control/LEASE.json').read_text());lease['worker_exit_code']=code
    retained['pins'][side+'_lease_sha256']=put(root/'control/LEASE.json',lease)
    retained['pins'][side+'_worker_sha256']=put(root/'WORKER_EXIT.private.json',dict(exit_code=code,failure=None))
    with pytest.raises(ValueError,match='EXACT_CLOSED'):
        read(retained)


@pytest.mark.parametrize('fault',['missing-member','image','raw-state','nested-shadow','original-source','prefix','source-code'])
def test_pinned_capture_and_original_source_bytes_remain_required(retained,fault):
    if fault=='missing-member': (retained['folder']/'native-camera/images/000000-body_rgb.png').unlink()
    elif fault=='image': (retained['folder']/'images/000000-wrist_rgb.png').write_bytes(png(99))
    elif fault=='raw-state':
        path=retained['folder']/'STATE.private.bin';raw=bytearray(path.read_bytes());struct.pack_into('<Q',raw,8,34328);path.write_bytes(raw)
    elif fault=='nested-shadow': put(retained['folder']/'BOUND_SHADOW_ABI.private.json',{})
    elif fault=='original-source': Path(retained['n']['task_phase_source_observation']).write_bytes(b'{}')
    elif fault=='prefix': (retained['native']/'LIVE_SKILL_INITIALIZATION_STAGE.private.json').write_text('{}')
    else: (Path(retained['n']['source_root'])/archive.SOURCE_MEMBERS[0]).write_text('tampered')
    with pytest.raises((ValueError,FileNotFoundError)):
        read(retained)


@pytest.mark.parametrize('fault',['same-ack','leg-history','action-return','grant','pair-schema','return-schema'])
def test_rehashed_pair_return_cannot_forge_same_pause_or_execution(retained,fault):
    proposal=copy.deepcopy(retained['proposal']); checked=proposal['query']['query_result']
    if fault=='same-ack': checked['original_pause_revalidation']['pause_after_query']['ack_sequence']=2
    elif fault=='leg-history': checked['original_pause_revalidation']['pause_after_query']['leg_history_steps']+=1
    elif fault=='action-return': checked['action_sha256']='0'*64
    elif fault=='grant': checked['native_execution_authorized']=True
    elif fault=='return-schema': checked['schema']='OLD'
    else: proposal['query']['schema']='OLD'
    failure=copy.deepcopy(retained['failure']);failure['query']=proposal['query']
    retained['pins']['proposal_sha256']=put(retained['native']/'LIVE_SKILL_GENERATED_PROPOSAL.private.json',proposal)
    retained['pins']['failure_sha256']=put(retained['native']/'LIVE_SKILL_FAILED.private.json',failure)
    with pytest.raises(ValueError): read(retained)


@pytest.mark.parametrize('key,value',[('schema','OLD'),('study','COMPLETED_2X2'),('base_kind','TARGET_FULL1_V3'),
    ('policy_payload_sha256','0'*64),('non_language_payload_sha256','0'*64),('condition','ORIGINAL'),
    ('expected_noise_sha256','0'*64),('reuse_existing_original_query',True),('seed',1)])
def test_rehashed_view_cannot_change_inputs_or_study(retained,key,value):
    view=short_view(retained);view[key]=value
    with pytest.raises(ValueError): resolve(retained,view)


@pytest.mark.parametrize('key,value',[('observation_mode','FROZEN_PROMPT_CONTROL_VIEW_V1'),
    ('recorded_observation_sha256','0'*64),('seed',366103),('seed',True),('checkpoint_files',{}),
    ('checkpoint_manifest_sha256','0'*64),('numeric_mode','FP16'),('max_queries',2),('max_queries',True),
    ('inference_mode','ONLINE'),('expected_noise_sha256','0'*64)])
def test_current_job_must_match_view_frozen_model_seed_precision_and_single_short_draw(retained,key,value):
    view=short_view(retained);spec=spec_for(retained,view);spec[key]=value
    with pytest.raises(ValueError): resolve(retained,view,spec)


def test_privileged_hidden_input_and_unregistered_descriptor_keys_are_rejected(retained):
    view=short_view(retained)
    for key in ('payload','object_pose','state106','permission'):
        altered=copy.deepcopy(view);altered[key]=[1,2,3]
        with pytest.raises(ValueError): resolve(retained,altered)
    changed=copy.deepcopy(retained['target']);changed['pins']['permission']=True
    with pytest.raises(ValueError): control.make_view(target_archive=changed,prompt_variant='CURRENT_SHORT')


def test_archive_never_uses_current_clock_live_admission_or_parses_prefix_windows(retained,monkeypatch):
    import time
    def forbidden(*args,**kwargs): raise AssertionError('LIVE_OR_HEAVY_API_CALLED')
    monkeypatch.setattr(time,'time',forbidden)
    monkeypatch.setattr(observation,'verify_still_paused',forbidden)
    prefix=retained['native']/'LIVE_SKILL_INITIALIZATION_STAGE.private.json'
    original=Path.read_bytes
    def no_large_parse(path):
        if path==prefix: raise AssertionError('PREFIX_READ_INTO_MEMORY')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',no_large_parse)
    value=read(retained); again=read(retained)
    assert value['payload']==again['payload'] and archive._stream_pin.cache_info().hits>=1
    assert not value['provenance']['prefix_windows_parsed'] and not value['provenance']['prefix_math_replayed']
    assert not value['provenance']['current_deadline_validation_performed']


def test_real_recorded_payload_route_preserves_saved_images_state_and_native_failure(retained,tmp_path):
    from scripts.wm02.infer_fusion_vla import recorded_payload
    view=short_view(retained);spec=spec_for(retained,view)
    path=tmp_path/'SHORT_VIEW.json';path.write_bytes(manifest_bytes(view));spec['recorded_observation']=str(path)
    payload,provenance=recorded_payload(spec)
    assert payload['language']==control.SHORT_LANGUAGE
    assert provenance['observation_mode']==control.MODE
    assert provenance['image_sha256']==[dict(camera=x['camera'],sha256=x['sha256']) for x in retained['payload']['images']]
    assert provenance['state_eef10_sha256']==dh(retained['payload']['state_eef10'])
    assert provenance['base_provenance']['native_status']=='NATIVE_STAGE_FAILED_PRESERVED'


def test_imports_remain_cpu_only_without_simulator_ros_or_neural_model():
    code='from go2arx5_vla.world_model import fusion_graded_query_archive_v9, fusion_graded_prompt_control_v2; import sys; assert not ({"torch","mujoco","rclpy","gr00t","transformers"}&set(sys.modules))'
    q=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=20)
    assert q.returncode==0,q.stderr
