"""Language-only saved-input tests; no real model, renderer or native run."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.pi0292.relay_protocol import digest as document_hash, sealed, unseal, validate_observation
from scripts.wm02.infer_fusion_vla import recorded_payload
from go2arx5_vla.world_model import fusion_prompt_control as control
from go2arx5_vla.world_model import fusion_source_posture_query_archive_v3 as target_reader


def raw_json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def non_language(payload):
    return {k:v for k,v in payload.items() if k != 'language'}


@pytest.fixture
def source(tmp_path, monkeypatch):
    images=[]
    for camera,color in (('external_rgb',(17,31,47)),('wrist_rgb',(63,79,97))):
        stream=io.BytesIO();Image.new('RGB',(256,256),color).save(stream,format='PNG')
        raw=stream.getvalue()
        images.append(dict(camera=camera,png_base64=base64.b64encode(raw).decode(),sha256=sha(raw)))
    payload=dict(snapshot_uuid='ca8cb3a0-3b6f-414f-9a20-c6369fc24d48',
        language='Original language taken from these exact saved bytes; never from a copied document.',
        state_eef10=[.4,-.1,.54,1.,0.,0.,0.,1.,0.,.95],images=images,
        frame='PI0291_MUJOCO_WORLD',tcp='ARX5_NATIVE_SITE_x5_ee')
    record=sealed(dict(payload=payload,run_uuid='SYNTHETIC_71558_FIXTURE'))
    path=tmp_path/'SOURCE.json';path.write_bytes(raw_json(record))
    monkeypatch.setattr(control,'SOURCE_OBSERVATION',sha(path.read_bytes()))
    files={'model.safetensors':'1'*64,'processor/config.json':'2'*64}
    monkeypatch.setattr(control,'INVENTORY',document_hash(files))
    spec=dict(seed=366104,checkpoint_manifest_sha256=control.CHECKPOINT,checkpoint_files=files,
        numeric_mode='ORIGINAL_FP32_WEIGHTS_BF16_AUTOCAST',max_queries=1,
        inference_mode='OFFLINE_REAL_VLA_INFERENCE')
    return dict(path=str(path),sha256=control.SOURCE_OBSERVATION),payload,spec


def resolve(view,spec):
    raw=raw_json(view)
    return control.prompt_control_payload(raw,view_sha256=sha(raw),query_spec=spec)


@pytest.mark.parametrize('variant',['ORIGINAL_71558','CURRENT_SHORT'])
def test_source_view_uses_actual_original_prompt_and_changes_only_language(source,variant):
    descriptor,base,spec=source
    before=Path(descriptor['path']).read_bytes()
    view=control.make_view(source_observation=descriptor,base_kind='SOURCE_71558',prompt_variant=variant)
    payload,provenance=resolve(view,spec)
    expected=base['language'] if variant=='ORIGINAL_71558' else control.SHORT_LANGUAGE
    assert payload['language']==expected
    assert raw_json(non_language(payload))==raw_json(non_language(base))
    assert Path(descriptor['path']).read_bytes()==before
    assert view['non_language_payload_sha256']==document_hash(non_language(base))
    assert view['policy_payload_sha256']==document_hash(payload)
    assert provenance['language_changed']==(variant=='CURRENT_SHORT')
    assert provenance['original_observation_unmodified'] and provenance['only_language_field_may_differ']
    assert not provenance['actual_saved_target_images']
    assert not provenance['native_execution_authorized'] and not provenance['action_installations_authorized']
    assert not provenance['language_effect_established_by_this_reader']
    assert provenance['new_queries']==provenance['new_integrations']==provenance['processor_decodes']==0
    assert set(payload)=={'snapshot_uuid','language','state_eef10','images','frame','tcp'}
    json.dumps(provenance,allow_nan=False)


def fake_target(source,monkeypatch):
    descriptor,original,spec=source
    payload=copy.deepcopy(original)
    payload.update(snapshot_uuid='41a6c7f5-d0b3-4f98-a4ac-7c2fc7505638',language=control.SHORT_LANGUAGE)
    payload['state_eef10'][0]+=.037
    target=dict(native_spec='/fixture/native/EXECUTION_SPEC.json',gpu_spec='/fixture/gpu/EXECUTION_SPEC.json',
        pins=dict(seed=366104,checkpoint_manifest_sha256=control.CHECKPOINT,
                  checkpoint_file_inventory_sha256=control.INVENTORY,observation_sha256='5'*64))
    body=dict(capture=dict(index=0,native_step=16403,time_s=32.806),native_step=16403,native_time_s=32.806,
        task=dict(language=control.SHORT_LANGUAGE,score_contract_sha256='6'*64),task_sha256='7'*64,
        world=dict(scene_sha256='8'*64),renderer=dict(GL_VENDOR='Mesa'),
        provenance=dict(native_execution_authorized=False),original_shadow_ABI=sealed(dict(
            provenance=dict(camera_equivalence=False,source_camera_identity_sha256='9'*64),
            target_observation=dict(scene_sha256='8'*64,native_state_sha256='a'*64),
            source_policy_view=dict(payload=payload,transform_sha256='b'*64,T_source_from_target=[[1.]]))))
    value=dict(schema=target_reader.SCHEMA,status='PAIR_COMPLETED_AND_PAUSE_REVALIDATED',
        external_pins=copy.deepcopy(target['pins']),payload=payload,observation=sealed(body))
    calls=[]
    def read(n,g,*,pins):
        assert str(n)==target['native_spec'] and str(g)==target['gpu_spec'] and pins==target['pins']
        calls.append('read');return copy.deepcopy(value)
    monkeypatch.setattr(target_reader,'read_archive',read)
    return target,payload,value,calls


def test_four_views_pair_exact_non_language_inputs_but_do_not_claim_between_domain_equivalence(source,monkeypatch):
    descriptor,source_payload,spec=source
    target,base,_,calls=fake_target(source,monkeypatch)
    result={}
    for base_kind in control.BASE_KINDS:
        for prompt in control.PROMPT_VARIANTS:
            view=control.make_view(source_observation=descriptor,base_kind=base_kind,prompt_variant=prompt,
                target_archive=target if base_kind=='TARGET_FULL1_V3' else None)
            result[view['condition']]=resolve(view,spec)
    assert len(result)==4 and len(calls)==4
    for prefix in ('SOURCE','TARGET'):
        a,pa=result[prefix+'_ORIGINAL'];b,pb=result[prefix+'_SHORT']
        assert raw_json(non_language(a))==raw_json(non_language(b))
        assert a['language']==source_payload['language'] and b['language']==control.SHORT_LANGUAGE
        assert pa['non_language_payload_sha256']==pb['non_language_payload_sha256']
        assert not pa['source_target_input_equivalence_established']
        assert not pa['between_base_domain_effect_identified']
    assert result['SOURCE_ORIGINAL'][1]['non_language_payload_sha256']!=result['TARGET_ORIGINAL'][1]['non_language_payload_sha256']
    target_provenance=result['TARGET_ORIGINAL'][1]
    assert target_provenance['actual_saved_target_images'] and not target_provenance['target_online_images']
    retained=target_provenance['base_provenance']['target']
    assert retained['original_task']['language']==control.SHORT_LANGUAGE
    assert retained['original_shadow_provenance']['camera_equivalence'] is False
    assert retained['target_observation']['native_state_sha256']=='a'*64
    assert 'payload' not in retained['source_policy_mapping']
    assert base['language']==control.SHORT_LANGUAGE


def test_real_v3_archive_integration_retains_original_task_and_all_pixels(source,tmp_path,monkeypatch):
    from tests.world_model.test_fusion_source_posture_query_archive_v3 import make_archive
    descriptor,original,spec=source
    n,g,pins,_,_,_,draw=make_archive(tmp_path/'target',monkeypatch)
    monkeypatch.setattr(control,'INVENTORY',pins['checkpoint_file_inventory_sha256'])
    # The hermetic archive has its own preregistered fixture seed. Production
    # remains fixed at 366104; the independent mismatch tests cover that pin.
    monkeypatch.setattr(control,'SEED',pins['seed']);spec['seed']=pins['seed']
    spec['checkpoint_files']=json.loads(g.read_text())['checkpoint_files']
    target=dict(native_spec=str(n),gpu_spec=str(g),pins=pins)
    archived=target_reader.read_archive(n,g,pins=pins)
    # Fixture uses its declared real task; this study's base is specifically the
    # short-language full1 condition. The saved task is pinned, never rewritten.
    assert archived['payload']['language']==control.SHORT_LANGUAGE
    view=control.make_view(source_observation=descriptor,base_kind='TARGET_FULL1_V3',
        prompt_variant='ORIGINAL_71558',target_archive=target)
    payload,provenance=resolve(view,spec)
    assert payload['language']==original['language']
    assert raw_json(non_language(payload))==raw_json(non_language(archived['payload']))
    assert provenance['base_provenance']['target']['original_task']==unseal(archived['observation'])['task']
    assert provenance['base_provenance']['target']['external_pins']==pins
    assert not provenance['current_native_pause_checked'] and not provenance['native_execution_authorized']
    assert draw.queries==1  # Only fixture creation, zero additional calls by this reader.


@pytest.mark.parametrize('key,value',[
    ('schema','OLD'),('study','OTHER'),('condition','TARGET_ORIGINAL'),('seed',1),
    ('base_payload_sha256','0'*64),('policy_payload_sha256','0'*64),
    ('non_language_payload_sha256','0'*64),('original_language_sha256','0'*64),
    ('selected_language_sha256','0'*64),('checkpoint_manifest_sha256','0'*64),
    ('checkpoint_file_inventory_sha256','0'*64),('prompt_variant','FREE_TEXT'),('base_kind','ONLINE')])
def test_rehashed_manifest_cannot_change_declared_condition_or_derived_inputs(source,key,value):
    descriptor,_,spec=source
    view=control.make_view(source_observation=descriptor,base_kind='SOURCE_71558',prompt_variant='CURRENT_SHORT')
    view[key]=value
    with pytest.raises(ValueError):resolve(view,spec)


@pytest.mark.parametrize('key,value',[
    ('seed',366101),('seed',True),('checkpoint_manifest_sha256','0'*64),
    ('checkpoint_files',{}),('numeric_mode','FP16'),('max_queries',2),('max_queries',True),
    ('inference_mode','ONLINE')])
def test_matched_query_seed_checkpoint_precision_and_single_query_are_required(source,key,value):
    descriptor,_,spec=source
    view=control.make_view(source_observation=descriptor,base_kind='SOURCE_71558',prompt_variant='CURRENT_SHORT')
    with pytest.raises(ValueError):resolve(view,dict(spec,**{key:value}))


@pytest.mark.parametrize('fault',['changed-bytes','wrong-pin','symlink','extra-payload','extra-ground-truth','missing-key','view-hash'])
def test_original_bytes_and_strict_manifest_reject_forgery_or_privileged_inputs(source,tmp_path,fault):
    descriptor,_,spec=source
    view=control.make_view(source_observation=descriptor,base_kind='SOURCE_71558',prompt_variant='CURRENT_SHORT')
    if fault=='changed-bytes':Path(descriptor['path']).write_bytes(b'{}')
    elif fault=='wrong-pin':view['source_observation']['sha256']='0'*64
    elif fault=='symlink':
        alias=tmp_path/'alias';alias.symlink_to(descriptor['path']);view['source_observation']['path']=str(alias)
    elif fault=='extra-payload':view['payload']={'state_eef10':[0]*10}
    elif fault=='extra-ground-truth':view['object_pose']=[1,2,3]
    elif fault=='missing-key':view.pop('seed')
    if fault=='view-hash':
        with pytest.raises(ValueError):control.prompt_control_payload(raw_json(view),view_sha256='0'*64,query_spec=spec)
    else:
        with pytest.raises(ValueError):resolve(view,spec)


@pytest.mark.parametrize('fault',['source-has-target','target-missing','wrong-seed','wrong-base-language','failed-archive'])
def test_target_cannot_shed_archive_or_relabel_another_condition(source,monkeypatch,fault):
    descriptor,_,_=source
    target,_,archive,_=fake_target(source,monkeypatch)
    if fault=='source-has-target':
        with pytest.raises(ValueError):control.make_view(source_observation=descriptor,base_kind='SOURCE_71558',
            prompt_variant='CURRENT_SHORT',target_archive=target)
        return
    if fault=='target-missing':target=None
    elif fault=='wrong-seed':archive['external_pins']['seed']=366102
    elif fault=='wrong-base-language':archive['payload']['language']='A different task.'
    elif fault=='failed-archive':archive['status']='SAVED_QUERY_ONLY'
    if fault not in ('target-missing',):
        monkeypatch.setattr(target_reader,'read_archive',lambda *a,**k:copy.deepcopy(archive))
    with pytest.raises(ValueError):control.make_view(source_observation=descriptor,base_kind='TARGET_FULL1_V3',
        prompt_variant='ORIGINAL_71558',target_archive=target)


def test_new_cli_mode_delegates_and_original_source_mode_is_unchanged(source,tmp_path,monkeypatch):
    descriptor,base,spec=source
    view=control.make_view(source_observation=descriptor,base_kind='SOURCE_71558',prompt_variant='CURRENT_SHORT')
    path=tmp_path/'VIEW.json';path.write_bytes(raw_json(view))
    actual,provenance=recorded_payload(dict(spec,observation_mode=control.MODE,
        recorded_observation=str(path),recorded_observation_sha256=sha(path.read_bytes())))
    assert actual['language']==control.SHORT_LANGUAGE and provenance['condition']=='SOURCE_SHORT'
    original,original_provenance=recorded_payload(dict(recorded_observation=descriptor['path'],
        recorded_observation_sha256=descriptor['sha256']))
    assert original==base and original_provenance['observation_mode']=='ARCHIVED_SOURCE_RELAY_PACKET'
    validate_observation(actual)
    assert 'ResidentModel' not in control.__dict__ and 'torch' not in control.__dict__


def test_source_pin_and_parse_use_the_same_single_read(source,monkeypatch):
    descriptor,base,_=source
    old=Path.read_bytes;reads=[]
    def once(path):
        if str(path)==descriptor['path']:
            reads.append(str(path));assert len(reads)==1
        return old(path)
    monkeypatch.setattr(Path,'read_bytes',once)
    view=control.make_view(source_observation=descriptor,base_kind='SOURCE_71558',prompt_variant='ORIGINAL_71558')
    assert view['base_payload_sha256']==document_hash(base) and len(reads)==1
