"""Versioned slow-prefix transport and whole-skill seams; synthetic only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from go2arx5_vla.world_model import fusion_task_phase_live_skill_pair_v8 as old_pair
from go2arx5_vla.world_model import fusion_task_phase_live_skill_pair_v9 as pair9
from go2arx5_vla.world_model import fusion_task_phase_live_skill_contract_v9 as contract
from go2arx5_vla.world_model import fusion_task_phase_live_skill_native_v9 as live9
from go2arx5_vla.world_model import fusion_task_phase_pose_slow_feedback_v9 as slow
from go2arx5_vla.world_model import fusion_task_phase_pose_continuation_v7 as old_prefix
from go2arx5_vla.world_model import fusion_paused_query_bridge as bridge
from go2arx5_vla.world_model import fusion_paused_inference as inference
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_native_v4 as feedback
from go2arx5_vla.world_model import auto1_native
from go2arx5_vla.world_model.target_sequential import native_task_bucket,TargetSequentialAdapter
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.relay_protocol import digest as document_hash
from scripts.wm02 import infer_fusion_paused_vla as infer
from test_fusion_task_phase_live_skill_pair_v8 import (
    pair_fixture as original_pair_fixture,bind,write_json,capture_source_archive,PROJECT,
    install_fake_draw,close_gpu)
from test_fusion_task_phase_live_skill_native_v8 import composition,flow


@pytest.fixture
def slow_pair(tmp_path,monkeypatch):
    f=original_pair_fixture.__wrapped__(tmp_path,monkeypatch)
    for key in('SOURCE_OBSERVATION','SOURCE_PAYLOAD','FRAME_MAP','SOURCE_CAMERA_IDENTITY'):
        monkeypatch.setattr(pair9,key,getattr(old_pair,key))
    g,a,p=f['gpu'],f['adapter'],f['pair']
    episode=slow.make_episode(run=a.spec['registered_root'],geometry_scale=.5)
    write_json(a.spec['episode_manifest'],episode)
    a.spec.update(phase=pair9.PHASE,kind=pair9.KIND,snapshot_binding_mode=pair9.PAIR_MODE,
        episode_manifest_sha256=digest(a.spec['episode_manifest']))
    g['snapshot_binding_mode']=pair9.PAIR_MODE
    p.update(schema=pair9.PAIR_SCHEMA,producer_phase=pair9.PHASE)
    capsule=Path(g['source_root'])
    for name in pair9.SOURCE_MEMBERS:
        target=capsule/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((PROJECT/name).read_bytes());g['source_files'][name]=digest(target)
    a.spec['source_files']=g['source_files'].copy()
    bind(g,a,p);f['episode']=episode
    return f


def test_new_mode_has_real_v9_episode_and_same_fixed_native_bounds(slow_pair):
    f=slow_pair;g,a,p=f['gpu'],f['adapter'],f['pair']
    inference.validate_spec(g)
    assert bridge.validate_pair_spec(g)==(p,a.spec)
    proof=bridge.preflight_pair(Path(g['registered_root'])/'EXECUTION_SPEC.json',a.spec)
    assert proof['native_boundary_step']==34303 and proof['native_endpoint_step']==66503
    assert proof['schema']==pair9.PAIR_SCHEMA and not proof['native_execution_authorized']
    assert f['episode']['phase']==slow.PHASE and len(f['episode']['feedback_decision_native_steps'])==12
    assert native_task_bucket(pair9.PHASE,True)=='128'
    assert native_task_bucket(old_pair.PHASE,True)=='128'
    assert native_task_bucket(old_prefix.PHASE,True)=='96'
    assert pair9.validate_native_pause(a) is None and not a.trace_sealed
    assert not Path(g['recorded_observation']).exists()


@pytest.mark.parametrize('fault',['v7-episode','v8-pair','v8-producer','closed-native','closed-gpu',
    'wrong-boundary','wrong-endpoint','extra-key','changed-prefix-code','short-task'])
def test_rehashed_metadata_cannot_replace_new_recipe_owner_or_full_language(slow_pair,fault):
    f=slow_pair;g,a,p=f['gpu'],f['adapter'],f['pair']
    if fault=='v7-episode':
        write_json(a.spec['episode_manifest'],old_prefix.make_episode(run=a.spec['registered_root'],geometry_scale=.5))
        a.spec['episode_manifest_sha256']=digest(a.spec['episode_manifest'])
    elif fault=='v8-pair':
        p['schema']=old_pair.PAIR_SCHEMA
    elif fault=='v8-producer':
        a.spec.update(phase=old_pair.PHASE,kind=old_pair.KIND)
    elif fault=='wrong-boundary':p['native_boundary_step']+=25
    elif fault=='wrong-endpoint':p['native_endpoint_step']=a.spec['max_ticks']=66528
    elif fault=='extra-key':p['approved']=True
    elif fault=='changed-prefix-code':(Path(g['source_root'])/pair9.SOURCE_MEMBERS[0]).write_text('tampered')
    elif fault=='short-task':
        task=dict(f['task'],language='Shorter replacement task')
        write_json(a.spec['paused_target_task'],task)
        a.spec['paused_target_task_file_sha256']=digest(a.spec['paused_target_task'])
        a.spec['paused_target_task_sha256']=g['paused_task_sha256']=p['task_sha256']=document_hash(task)
    bind(g,a,p)
    if fault.startswith('closed-'):
        spec=a.spec if fault=='closed-native' else g
        lease=json.loads(Path(spec['lease']).read_bytes());lease['status']='CLOSED';write_json(spec['lease'],lease)
    with pytest.raises((ValueError,KeyError)):
        bridge.preflight_pair(Path(g['registered_root'])/'EXECUTION_SPEC.json',a.spec)
    assert not Path(g['recorded_observation']).exists() and f['events'].saves==[]


def test_payload_names_actual_slow_recipe_and_preserves_all_eight_members(slow_pair):
    f=slow_pair;record=capture_source_archive(f)
    payload,identity,returned=inference.read_payload(f['gpu'])
    assert returned==record and len(identity['files'])==8
    assert payload['language']==f['source_payload']['language']
    assert identity['physical_recipe_phase']==identity['initialization_recipe_phase']==slow.PHASE
    assert identity['producer_phase']==pair9.PHASE
    assert not identity['source_task_phase_equivalence'] and not identity['old_initialization_readiness_relabelled']


def test_fake_generate_roundtrip_uses_new_wait_mode_and_rechecks_original_pause(slow_pair,monkeypatch):
    f=slow_pair;g,a=f['gpu'],f['adapter'];draw=install_fake_draw(monkeypatch)
    monkeypatch.setattr(bridge,'capture_paused_observation',lambda *args,**kwargs:capture_source_archive(f))
    def generate_once(seconds):
        assert seconds==.05
        assert (Path(g['recorded_observation']).parent/'PAIR_SNAPSHOT_READY.private.json').is_file()
        inference.validate_spec(g);bridge.wait_for_snapshot(g)
        out=Path(g['registered_root'])/'fit-01';out.mkdir();infer.infer(g,out);close_gpu(g)
    monkeypatch.setattr(bridge.time,'sleep',generate_once)
    result=bridge.capture_and_wait_query(a,binding=a.fixture_binding,task=f['task'],
        task_sha256=g['paused_task_sha256'],gpu_spec_path=Path(g['registered_root'])/'EXECUTION_SPEC.json',camera_recipe=f['camera'])
    assert draw.loads==draw.queries==1 and result['query_result']['model_result_checked']
    assert result['native_integrations_by_bridge']==0 and not result['online_action_execution']
    out=Path(g['registered_root'])/'fit-01/RESULT.private.json'
    saved=json.loads(out.read_bytes());assert saved['physical_recipe_phase']==slow.PHASE
    a.trace_sealed=True
    with pytest.raises(ValueError,match='PRESEAL_WBC'):
        inference.verify_paused_query_return(a,g,out)


def test_slow_prefix_shared_consumer_keeps_full_draw_and_actual_v9_identity(composition):
    a,samples,windows,report,raw,action,calls=composition
    a.spec.update(phase=pair9.PHASE,kind=pair9.KIND)
    Path(a.spec['registered_root'],'EXECUTION_SPEC.json').write_text(json.dumps(a.spec))
    original=copy.deepcopy(report)
    result=live9._continue(a,report,samples,windows,task={'full':'original'},camera=object())
    assert report==original and len(calls)==1 and not a.trace_sealed
    assert result['actual_spec_phase']==pair9.PHASE and result['schema'].endswith('_V9')
    assert result['generated_segment']['phase']==pair9.PHASE
    assert result['all_40_knots_consumed'] and all(result['raw_knot_consumed'])
    assert result['episode_role']=='SLOW_V9_INITIALIZATION_PREFIX_ONLY'
    assert result['additional_action_decodes']==result['WM_updates']==result['GR00T_updates']==result['Actor_updates']==0
    saved=json.loads((a.output/'LIVE_SKILL_GENERATED_PROPOSAL.private.json').read_bytes())
    np.testing.assert_array_equal(saved['original_decoded_physical_H40'],raw)
    assert saved['schema'].endswith('_V9') and saved['geometry_scale']==1. and saved['time_stretch_factor']==2
    assert not result['full_task_success'] and not result['source_task_phase_equivalence']


def test_actual_entry_routes_only_new_recipe_and_existing_native_worker(monkeypatch):
    episode,source,task,camera=object(),object(),object(),object();calls=[]
    monkeypatch.setattr(live9,'episode_config',lambda spec:(episode,source,task,camera))
    def run(a,e,s,**kwargs):calls.append((a,e,s,kwargs));return 'same-native-world'
    monkeypatch.setattr(feedback,'run_feedback_program',run)
    a=SimpleNamespace(spec={'phase':pair9.PHASE})
    assert TargetSequentialAdapter.dynamic_diagnostic(a)=='same-native-world'
    assert calls[0][3]['recipe_module']is slow and calls[0][3]['actual_phase']==pair9.PHASE
    assert callable(calls[0][3]['continuation'])
    monkeypatch.setattr(contract,'episode_limit',lambda spec,previous:66503)
    assert auto1_native.phase_limit(pair9.PHASE,{},a.spec)==66503


def test_new_score_changes_only_declared_prefix_and_retains_full_H40_safety():
    score=contract.score_contract()
    assert score['initialization_recipe']=='12_SLOTS_MOVE28_SETTLE8_SAME_B34303'
    assert score['capture_boundary_step']==34303 and score['registered_final_step']==66503
    assert score['geometry_scale']==1. and score['time_stretch_factor']==2 and score['raw_knots_required']==40
    assert not score['native_safety_thresholds_changed'] and not score['old_prefix_query_ready_reclassified']
