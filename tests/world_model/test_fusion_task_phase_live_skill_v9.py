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
        live_skill_time_stretch_factor=2,
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
    a.spec.update(phase=pair9.PHASE,kind=pair9.KIND,live_skill_time_stretch_factor=2,
        task_condition=contract.TASK_CONDITION)
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


def test_declared_time1_is_a_distinct_score_and_task_not_a_relabelled_time2():
    old=contract.score_contract();fast=contract.score_contract(time_stretch_factor=1)
    expected=dict(old,schema=old['schema']+'_TIME1',time_stretch_factor=1)
    assert fast==expected and document_hash(fast)!=document_hash(old)
    original=contract.task_document('a'*64,'verbatim original task')
    changed=contract.task_document('a'*64,'verbatim original task',time_stretch_factor=1)
    assert changed==dict(original,task_id=original['task_id']+'_TIME1',
        score_contract_sha256=document_hash(fast))
    assert original['score_contract_sha256']==document_hash(old)


@pytest.mark.parametrize('factor',[None,True,False,0,3,1.,'1'])
def test_undeclared_or_noninteger_time_mapping_rejected_before_capture(slow_pair,factor):
    f=slow_pair;g,a,p=f['gpu'],f['adapter'],f['pair']
    a.spec['live_skill_time_stretch_factor']=factor;bind(g,a,p)
    with pytest.raises(ValueError,match='EXACT_GRADED_DIAGNOSTIC'):
        bridge.preflight_pair(Path(g['registered_root'])/'EXECUTION_SPEC.json',a.spec)
    with pytest.raises(ValueError,match='EXPLICIT_ORIGINAL_ADAPTER'):
        contract.score_contract(time_stretch_factor=factor)
    assert not Path(g['recorded_observation']).exists() and f['events'].saves==[]


def test_explicit_time1_consumes_entire_same_draw_and_install_guard_binds_its_clock(composition):
    a,samples,windows,report,raw,action,calls=composition
    a.spec.update(phase=pair9.PHASE,kind=pair9.KIND,live_skill_time_stretch_factor=1,
        task_condition=contract.TASK_CONDITION)
    Path(a.spec['registered_root'],'EXECUTION_SPEC.json').write_text(json.dumps(a.spec))
    result=live9._continue(a,report,samples,windows,task={'full':'original'},camera=object())
    saved=json.loads((a.output/'LIVE_SKILL_GENERATED_PROPOSAL.private.json').read_bytes())
    assert saved['time_stretch_factor']==1 and len(calls)==1
    np.testing.assert_array_equal(saved['original_decoded_physical_H40'],raw)
    assert result['all_40_knots_consumed'] and all(result['raw_knot_consumed'])
    assert result['full_task_success'] is False and result['additional_action_decodes']==0
    program=a.live_skill_program
    check=SimpleNamespace(spec=a.spec,observe=lambda:{'step':contract.B},live_skill_program=program,
        base_pose_hold=program.base_pose_hold,trajectory_initial_base=program.base_pose_hold)
    proof=live9.validate_adapter_install(check,program.preview(0))
    assert proof['time_stretch_factor']==1 and proof['physics_advance_authorized'] is False
    check.spec=dict(a.spec,live_skill_time_stretch_factor=2)
    with pytest.raises(ValueError,match='WHOLE_LIVE_SUFFIX'):
        live9.validate_adapter_install(check,program.preview(0))


@pytest.mark.parametrize('factor',[1,2])
def test_recipe_requires_its_actual_declared_time_task_and_score(slow_pair,monkeypatch,factor):
    from go2arx5_vla.world_model import fusion_source_posture_paused_query_v3 as sensor
    f=slow_pair;a=f['adapter'];spec=a.spec;episode=f['episode']
    source={'source_observation_bytes':Path(spec['task_phase_source_observation']).read_bytes(),
        'T_source_from_target':spec['paused_source_from_target']}
    calls=[]
    def operational(s,e,**kwargs):
        calls.append((s,e,kwargs));return e,source
    monkeypatch.setattr(contract.operational,'recipe_config',operational)
    monkeypatch.setattr(sensor,'SOURCE_CAMERA_IDENTITY',spec['source_camera_identity_sha256'])
    spec.update(live_skill_geometry_scale=1.,live_skill_time_stretch_factor=factor,
        live_skill_start_step=34303)
    task=contract.task_document(digest(spec['model_path']),f['source_payload']['language'],time_stretch_factor=factor)
    score=contract.score_contract(time_stretch_factor=factor)
    score_path=Path(spec['registered_root'])/'DECLARED_SCORE.json';write_json(score_path,score)
    write_json(spec['paused_target_task'],task)
    spec.update(paused_score_contract=str(score_path),paused_score_contract_file_sha256=digest(score_path),
        paused_target_task_file_sha256=digest(spec['paused_target_task']),paused_target_task_sha256=document_hash(task))
    got=contract.recipe_config(spec)
    assert got[:3]==(episode,source,task) and len(calls)==1
    assert calls[0][2]['actual_phase']==pair9.PHASE
    wrong=contract.score_contract(time_stretch_factor=3-factor)
    write_json(score_path,wrong);spec['paused_score_contract_file_sha256']=digest(score_path)
    with pytest.raises(ValueError,match='EXACT_SOURCE_FULL_LANGUAGE_AND_DISTINCT'):
        contract.recipe_config(spec)


@pytest.fixture
def short_pair(slow_pair):
    from go2arx5_vla.world_model import fusion_graded_short_skill_v1 as short
    f=slow_pair;g,a,p=f['gpu'],f['adapter'],f['pair']
    task=contract.task_document(digest(a.spec['model_path']),short.LANGUAGE,
        time_stretch_factor=1,task_condition=contract.SHORT_TASK_CONDITION)
    score=contract.score_contract(time_stretch_factor=1,task_condition=contract.SHORT_TASK_CONDITION)
    path=Path(a.spec['registered_root'])/'SYNTHETIC_SHORT_SCORE.json';write_json(path,score)
    write_json(a.spec['paused_target_task'],task)
    a.spec.update(task_condition=contract.SHORT_TASK_CONDITION,live_skill_time_stretch_factor=1,
        paused_target_task_file_sha256=digest(a.spec['paused_target_task']),
        paused_target_task_sha256=document_hash(task),paused_score_contract=str(path),
        paused_score_contract_file_sha256=digest(path))
    p.update(task_condition=contract.SHORT_TASK_CONDITION,task_sha256=document_hash(task))
    g['paused_task_sha256']=document_hash(task)
    capsule=Path(g['source_root'])
    for name in('src/go2arx5_vla/world_model/fusion_graded_short_skill_v1.py',
                'src/go2arx5_vla/world_model/fusion_short_skill.py'):
        dest=capsule/name;dest.write_bytes((PROJECT/name).read_bytes());g['source_files'][name]=digest(dest)
    a.spec['source_files']=g['source_files'].copy();bind(g,a,p)
    a.fixture_task=task;f['task']=task
    return f


def test_old_full_task_score_and_document_remain_identical():
    assert document_hash(contract.score_contract())=='65916035e2cb58bd1bc2f05599dd2625b4f045dee71489c77050f368b1ec78a7'
    assert document_hash(contract.task_document('a'*64,'original full task'))=='3b1aa63cba0bd25ca0824d0b1bae01e9420791130815210b092c9e5e45442ad9'


def test_new_short_language_and34303_score_are_explicit_in_pair_and_payload(short_pair):
    from go2arx5_vla.world_model import fusion_graded_short_skill_v1 as short
    f=short_pair;g,a,p=f['gpu'],f['adapter'],f['pair']
    assert bridge.validate_pair_spec(g)==(p,a.spec)
    original_source=digest(f['source_path'])
    record=capture_source_archive(f)
    payload,identity,_=inference.read_payload(g)
    assert payload['language']==short.LANGUAGE and payload['language']!=f['source_payload']['language']
    assert identity['task_condition']==contract.SHORT_TASK_CONDITION
    assert identity['physical_recipe_phase']==slow.PHASE and not identity['source_task_phase_equivalence']
    assert identity['task']['score_contract_sha256']==document_hash(
        contract.score_contract(time_stretch_factor=1,task_condition=contract.SHORT_TASK_CONDITION))
    assert digest(f['source_path'])==original_source and not identity['native_execution_authorized']


@pytest.mark.parametrize('fault',['original-language','old3703-score','full-taskID','implicit-task-condition'])
def test_rehashed_short_task_cannot_use_original_intent_or_old_clock(short_pair,fault):
    from go2arx5_vla.world_model import fusion_short_skill as old
    f=short_pair;g,a,p=f['gpu'],f['adapter'],f['pair'];task=copy.deepcopy(f['task'])
    if fault=='original-language':task['language']=f['source_payload']['language']
    elif fault=='full-taskID':task['task_id']='ORIGINAL71558_FULL_TASK_AT_GRADED_TARGET_STATE_V9_TIME1'
    elif fault=='implicit-task-condition':p['task_condition']=a.spec['task_condition']='SHORT'
    else:
        score=json.loads(Path(a.spec['paused_score_contract']).read_bytes())
        score.update(actual_short_skill_score_contract=old.score_contract(),
            actual_short_skill_score_contract_sha256=document_hash(old.score_contract()))
        write_json(a.spec['paused_score_contract'],score)
        a.spec['paused_score_contract_file_sha256']=digest(a.spec['paused_score_contract'])
        task['score_contract_sha256']=document_hash(score)
    write_json(a.spec['paused_target_task'],task)
    a.spec['paused_target_task_file_sha256']=digest(a.spec['paused_target_task'])
    a.spec['paused_target_task_sha256']=g['paused_task_sha256']=p['task_sha256']=document_hash(task)
    bind(g,a,p)
    with pytest.raises(ValueError):bridge.preflight_pair(Path(g['registered_root'])/'EXECUTION_SPEC.json',a.spec)
    assert not Path(g['recorded_observation']).exists()


@pytest.fixture
def real_clock_composition(monkeypatch,request):
    import test_fusion_live_skill_segment as segment_fixture
    import test_fusion_task_phase_live_skill_native_v8 as live_fixture
    monkeypatch.setattr(segment_fixture,'B',34303)
    monkeypatch.setattr(live_fixture,'B',34303)
    return request.getfixturevalue('composition')


def test_short_desired_intent_does_not_fake_actual_skill_success(real_clock_composition):
    from go2arx5_vla.world_model import fusion_graded_short_skill_v1 as short
    a,samples,windows,report,raw,action,calls=real_clock_composition
    actual=samples[-1]['tcp'];cube=samples[-1]['object_pose']
    direction=cube[:3,3]-actual[:3,3];direction/=np.linalg.norm(direction)
    raw[:,:3]=actual[:3,3]+.003*np.minimum(np.arange(40)/8,1.)[:,None]*direction
    raw[:,9]=1.
    with action.open('wb')as f:np.savez(f,physical_H40=raw)
    result_path=action.parent/'RESULT.private.json';gpu_result=json.loads(result_path.read_bytes())
    gpu_result['action_sha256']=digest(action);result_path.write_text(json.dumps(gpu_result))
    a.spec.update(phase=pair9.PHASE,kind=pair9.KIND,live_skill_time_stretch_factor=1,
        task_condition=contract.SHORT_TASK_CONDITION)
    Path(a.spec['registered_root'],'EXECUTION_SPEC.json').write_text(json.dumps(a.spec))
    complete=live9._continue(a,report,samples,windows,task={'language':short.LANGUAGE},camera=object())
    assert complete['short_intent_screen']['qualified'] and complete['all_40_knots_consumed']
    measured=complete['graded_short_skill_score']
    assert not measured['approach_step_success'] and measured['decision_native_step']==34303
    assert measured['metrics']['signed_progress_m']<.002
    assert not complete['full_task_success'] and not measured['native_execution_authorized']


@pytest.mark.parametrize('condition',[contract.SHORT_TASK_CONDITION,contract.TRACKING_TASK_CONDITION,contract.MECHANICAL_TASK_CONDITION])
def test_late_short_closure_preserves_complete_proposal_without_install_or_integration(real_clock_composition,condition):
    from go2arx5_vla.world_model import fusion_graded_short_skill_v1 as short
    a,samples,windows,report,raw,action,calls=real_clock_composition
    actual=samples[-1]['tcp'];cube=samples[-1]['object_pose']
    direction=cube[:3,3]-actual[:3,3];direction/=np.linalg.norm(direction)
    raw[:,:3]=actual[:3,3]+.003*np.minimum(np.arange(40)/8,1.)[:,None]*direction
    # Keep duration admissible, so the semantic gate rather than the earlier
    # duration guard must reject this gradually closing late suffix.
    raw[:,9]=1.;raw[-16:,9]=np.linspace(1.,.85,16)
    with action.open('wb')as f:np.savez(f,physical_H40=raw)
    result_path=action.parent/'RESULT.private.json';gpu_result=json.loads(result_path.read_bytes())
    gpu_result['action_sha256']=digest(action);result_path.write_text(json.dumps(gpu_result))
    a.spec.update(phase=pair9.PHASE,kind=pair9.KIND,live_skill_time_stretch_factor=1,
        task_condition=condition)
    Path(a.spec['registered_root'],'EXECUTION_SPEC.json').write_text(json.dumps(a.spec))
    reason={contract.TRACKING_TASK_CONDITION:'WHOLE_GENERATED_TRACKING_DIAGNOSTIC_REJECT',
            contract.MECHANICAL_TASK_CONDITION:'WHOLE_GENERATED_MECHANICAL_DOMAIN_REJECT',
            contract.SHORT_TASK_CONDITION:'WHOLE_GENERATED_SHORT_INTENT_REJECT'}[condition]
    with pytest.raises(ValueError,match=reason):
        live9._continue(a,report,samples,windows,task={'language':short.LANGUAGE},camera=object())
    assert len(calls)==1 and not a.installs and not a.advances
    saved=json.loads((a.output/'LIVE_SKILL_GENERATED_PROPOSAL.private.json').read_bytes())
    np.testing.assert_array_equal(saved['original_decoded_physical_H40'],raw)
    screen=json.loads((a.output/'LIVE_SKILL_SHORT_INTENT_SCREEN.private.json').read_bytes())
    assert screen['raw_knots']==40 and not screen['qualified'] and not screen['predicates']['open']
    failed=json.loads((a.output/'LIVE_SKILL_FAILED.private.json').read_bytes())
    assert failed['proposal_saved'] and not failed['automatic_retry']


def test_tracking_task_and_capture_keep_distinct_identity_and_strict_score(short_pair):
    from go2arx5_vla.world_model import fusion_graded_short_skill_v1 as short
    f=short_pair;g,a,p=f['gpu'],f['adapter'],f['pair']
    task=contract.task_document(digest(a.spec['model_path']),short.LANGUAGE,
        time_stretch_factor=1,task_condition=contract.TRACKING_TASK_CONDITION)
    score=contract.score_contract(time_stretch_factor=1,task_condition=contract.TRACKING_TASK_CONDITION)
    write_json(a.spec['paused_target_task'],task);write_json(a.spec['paused_score_contract'],score)
    a.spec.update(task_condition=contract.TRACKING_TASK_CONDITION,
        paused_target_task_file_sha256=digest(a.spec['paused_target_task']),
        paused_target_task_sha256=document_hash(task),
        paused_score_contract_file_sha256=digest(a.spec['paused_score_contract']))
    p.update(task_condition=contract.TRACKING_TASK_CONDITION,task_sha256=document_hash(task))
    g['paused_task_sha256']=document_hash(task);bind(g,a,p);f['task']=task;a.fixture_task=task
    g['wall_limit_s']=1800;bind(g,a,p)
    assert bridge.validate_pair_spec(g)==(p,a.spec)
    assert task['task_id']==short.TRACKING_TASK_ID and task['task_id']!=short.TASK_ID
    assert score['actual_short_skill_score_contract']==short.score_contract()
    capture_source_archive(f);payload,identity,_=inference.read_payload(g)
    assert payload['language']==short.LANGUAGE and identity['task_condition']==contract.TRACKING_TASK_CONDITION
    g['wall_limit_s']=1801;bind(g,a,p)
    with pytest.raises(ValueError,match='AT_MOST_1800_SECONDS'):bridge.validate_pair_spec(g)


def test_existing_strict_short_condition_cannot_use_tracking_wall_window(short_pair):
    f=short_pair;g,a,p=f['gpu'],f['adapter'],f['pair']
    g['wall_limit_s']=1800;bind(g,a,p)
    with pytest.raises(ValueError):bridge.validate_pair_spec(g)


def test_tracking_diagnostic_consumes_complete_path_but_does_not_erase_strict_failure(real_clock_composition):
    from go2arx5_vla.world_model import fusion_graded_short_skill_v1 as short
    a,samples,windows,report,raw,action,calls=real_clock_composition
    actual=samples[-1]['tcp'];direction=samples[-1]['object_pose'][:3,3]-actual[:3,3]
    direction/=np.linalg.norm(direction)
    lateral=np.cross(direction,[0.,0.,1.]);lateral/=np.linalg.norm(lateral)
    # The proposal must fit the existing fixture budget to examine tracking
    # admission, rather than the independent earlier duration rejection.
    progress=np.arange(40)/39
    raw[:,:3]=actual[:3,3]+progress[:,None]*(.003*direction+.004*lateral)
    raw[:,9]=1.
    with action.open('wb')as f:np.savez(f,physical_H40=raw)
    rp=action.parent/'RESULT.private.json';value=json.loads(rp.read_bytes())
    value['action_sha256']=digest(action);rp.write_text(json.dumps(value))
    a.spec.update(phase=pair9.PHASE,kind=pair9.KIND,live_skill_time_stretch_factor=1,
        task_condition=contract.TRACKING_TASK_CONDITION)
    Path(a.spec['registered_root'],'EXECUTION_SPEC.json').write_text(json.dumps(a.spec))
    complete=live9._continue(a,report,samples,windows,task={'language':short.LANGUAGE},camera=object())
    assert complete['exploratory_tracking_diagnostic'] and complete['all_40_knots_consumed']
    assert complete['strict_short_proposal_failure_retained']
    assert not complete['short_intent_screen']['qualified']
    assert not complete['graded_short_skill_score']['approach_step_success']
    assert complete['tracking_diagnostic_screen']['eligible_for_original_native_certificate']
    assert not complete['full_task_success'] and not complete['target_task_success']


@pytest.fixture
def mechanical_pair(short_pair):
    from go2arx5_vla.world_model import fusion_bounded_reference_tracking_v1 as mechanical
    from go2arx5_vla.world_model import fusion_graded_short_skill_v1 as short
    f=short_pair;g,a,p=f['gpu'],f['adapter'],f['pair']
    task=contract.task_document(digest(a.spec['model_path']),short.LANGUAGE,
        time_stretch_factor=1,task_condition=mechanical.TASK_CONDITION)
    score=contract.score_contract(time_stretch_factor=1,task_condition=mechanical.TASK_CONDITION)
    write_json(a.spec['paused_target_task'],task);write_json(a.spec['paused_score_contract'],score)
    a.spec.update(task_condition=mechanical.TASK_CONDITION,
        paused_target_task_file_sha256=digest(a.spec['paused_target_task']),
        paused_target_task_sha256=document_hash(task),
        paused_score_contract_file_sha256=digest(a.spec['paused_score_contract']))
    p.update(task_condition=mechanical.TASK_CONDITION,task_sha256=document_hash(task))
    g.update(paused_task_sha256=document_hash(task),task_condition=mechanical.TASK_CONDITION)
    name='src/go2arx5_vla/world_model/fusion_bounded_reference_tracking_v1.py'
    dest=Path(g['source_root'])/name;dest.write_bytes((PROJECT/name).read_bytes())
    g['source_files'][name]=digest(dest);a.spec['source_files']=g['source_files'].copy()
    bind(g,a,p);f['task']=task;a.fixture_task=task
    return f


def test_mechanical_condition_is_explicit_and_cannot_change_old_task_contracts(mechanical_pair):
    from go2arx5_vla.world_model import fusion_bounded_reference_tracking_v1 as mechanical
    from go2arx5_vla.world_model import fusion_graded_short_skill_v1 as short
    f=mechanical_pair;g,a,p=f['gpu'],f['adapter'],f['pair'];task=f['task']
    score=json.loads(Path(a.spec['paused_score_contract']).read_bytes())
    assert bridge.validate_pair_spec(g)==(p,a.spec)
    assert task['task_id']==mechanical.TASK_ID and score['mechanical_domain']==mechanical.domain_contract()
    assert score['actual_short_skill_score_contract']==short.score_contract()
    assert document_hash(short.score_contract())=='755139841bbd17949c4a93fa91369b9375ab738915d063a08b3728265cc6ff2a'
    capture_source_archive(f);_,identity,_=inference.read_payload(g)
    assert identity['task_condition']==mechanical.TASK_CONDITION
    g['wall_limit_s']=1201;bind(g,a,p)
    with pytest.raises(ValueError,match='AT_MOST_1200_SECONDS'):bridge.validate_pair_spec(g)


@pytest.mark.parametrize('capture_failure',[False,True])
def test_preloaded_model_still_needs_fresh_capture_before_one_original_generate(mechanical_pair,monkeypatch,capture_failure):
    f=mechanical_pair;g,a,p=f['gpu'],f['adapter'],f['pair']
    g['frozen_model_load_schedule']=a.spec['frozen_model_load_schedule']=inference.PRELOAD_BEFORE_SNAPSHOT
    bind(g,a,p);events=install_fake_draw(monkeypatch)
    def wait(spec):
        assert spec is g and events.loads==1 and events.queries==0
        assert not Path(g['recorded_observation']).exists()
        if capture_failure:raise RuntimeError('SNAPSHOT_MISSING_NO_DRAW')
        capture_source_archive(f)
    monkeypatch.setattr(bridge,'wait_for_snapshot',wait)
    out=Path(g['registered_root'])/'fit-01';out.mkdir()
    if capture_failure:
        with pytest.raises(RuntimeError,match='SNAPSHOT_MISSING'):infer.infer(g,out)
        failure=json.loads((out/'FAILED.private.json').read_bytes())
        assert failure['completed_model_loads']==1 and failure['generation_invocations']==0
        assert failure['stage']=='SNAPSHOT_WAIT' and not failure['retry_authorized']
        assert events.loads==1 and events.queries==0
    else:
        result=infer.infer(g,out)
        assert events.loads==events.queries==1
        assert result['frozen_model_load_schedule']==inference.PRELOAD_BEFORE_SNAPSHOT
        assert result['checkpoint_integrity_check_wall_s']>=0
        assert result['physical_H40_shape']==[40,10] and not result['native_execution_authorized']


@pytest.mark.parametrize('fault',['one-sided','strict-task','unknown-schedule'])
def test_early_load_cannot_be_silently_enabled_on_old_or_mismatched_scopes(mechanical_pair,monkeypatch,fault):
    f=mechanical_pair;g,a,p=f['gpu'],f['adapter'],f['pair']
    g['frozen_model_load_schedule']=inference.PRELOAD_BEFORE_SNAPSHOT
    if fault!='one-sided':a.spec['frozen_model_load_schedule']=inference.PRELOAD_BEFORE_SNAPSHOT
    if fault=='strict-task':p['task_condition']=a.spec['task_condition']=contract.SHORT_TASK_CONDITION
    elif fault=='unknown-schedule':g['frozen_model_load_schedule']=a.spec['frozen_model_load_schedule']='UNDECLARED_PRELOAD'
    bind(g,a,p);events=install_fake_draw(monkeypatch)
    out=Path(g['registered_root'])/'fit-01';out.mkdir()
    with pytest.raises(ValueError):infer.infer(g,out)
    assert events.loads==events.queries==0


@pytest.mark.parametrize('install_failure',[False,True])
def test_mechanical_consumption_preserves_semantic_failure_and_install_checks(real_clock_composition,monkeypatch,install_failure):
    from go2arx5_vla.world_model import fusion_bounded_reference_tracking_v1 as mechanical
    from go2arx5_vla.world_model import fusion_graded_short_skill_v1 as short
    a,samples,windows,report,raw,action,calls=real_clock_composition
    actual=samples[-1]['tcp'];direction=samples[-1]['object_pose'][:3,3]-actual[:3,3]
    direction/=np.linalg.norm(direction)
    raw[:,:3]=actual[:3,3]-.003*(np.arange(40)/39)[:,None]*direction
    raw[:,9]=1.
    with action.open('wb')as f:np.savez(f,physical_H40=raw)
    path=action.parent/'RESULT.private.json';value=json.loads(path.read_bytes())
    value['action_sha256']=digest(action);path.write_text(json.dumps(value))
    a.spec.update(phase=pair9.PHASE,kind=pair9.KIND,live_skill_time_stretch_factor=1,
        task_condition=mechanical.TASK_CONDITION)
    Path(a.spec['registered_root'],'EXECUTION_SPEC.json').write_text(json.dumps(a.spec))
    if install_failure:
        def reject(_):raise ValueError('ORIGINAL_NATIVE_CERTIFICATE_REFUSED_SYNTHETIC')
        monkeypatch.setattr(a,'install_reference',reject)
        with pytest.raises(ValueError,match='ORIGINAL_NATIVE_CERTIFICATE_REFUSED'):
            live9._continue(a,report,samples,windows,task={'language':short.LANGUAGE},camera=object())
        assert not a.advances
    else:
        complete=live9._continue(a,report,samples,windows,task={'language':short.LANGUAGE},camera=object())
        assert complete['engineering_full_H40_consumption_only'] and complete['all_40_knots_consumed']
        assert complete['strict_short_proposal_failure_retained']
        assert not complete['short_intent_screen']['qualified']
        assert not complete['graded_short_skill_score']['approach_step_success']
        assert not complete['full_task_success'] and not complete['target_task_success']
    assert len(calls)==1
