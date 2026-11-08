import copy,json
from pathlib import Path
import pytest
from go2arx5_vla.world_model import fusion_source_phase_c01 as c01
from go2arx5_vla.world_model import fusion_task_phase_live_skill_contract_v9 as contract
from go2arx5_vla.world_model import fusion_task_phase_live_skill_pair_v9 as pair
from go2arx5_vla.world_model import fusion_paused_inference as inference
from go2arx5_vla.world_model import fusion_paused_query_bridge as bridge
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.relay_protocol import digest as doc_hash
from test_fusion_task_phase_live_skill_v9 import slow_pair,real_clock_composition,composition,flow
from test_fusion_task_phase_live_skill_pair_v8 import bind,write_json,PROJECT,capture_source_archive


def registration(spec,tmp_path,monkeypatch):
    import numpy as np
    model=digest(spec['model_path']);monkeypatch.setattr(c01,'SCENE',model)
    proof=tmp_path/'STATIC.json';write_json(proof,dict(status='STATIC_ONLY_PASSED_NEW_NATIVE_ADMISSION_REQUIRED',scene_sha256=model,
        robot_contract=dict(sha256=c01.ROBOT,controller_or_safety_changes=False,original_dt_s=.002),
        dense_prefix_geometry=dict(saved_pose_FK_calls=34303,contact_free_at_saved_poses=True),new_native_admission=False))
    monkeypatch.setattr(c01,'STATIC',digest(proof));root=Path(spec['registered_root'])
    panel=tmp_path/'PANEL.json';write_json(panel,dict(assignments=[dict(native_run=root.name,parent='C01_FROZEN_QUERY',split='DIAGNOSTIC',scene_sha256=model,task_condition=c01.CONDITION)]));pin=digest(panel)
    reference=tmp_path/'NOMINAL_REFERENCE.json';write_json(reference,{'synthetic_separately_tested_reference':True})
    monkeypatch.setattr(c01,'nominal_reference',lambda value:np.eye(4))
    spec.update(task_condition=c01.CONDITION,live_skill_start_step=34303,max_ticks=34303,
        phase=pair.PHASE,kind=pair.KIND,live_skill_geometry_scale=1.,live_skill_time_stretch_factor=2,
        response_learning_panel=str(panel),response_learning_panel_sha256=pin,
        response_learning_assignment=dict(schema='WM02_PROGRAM_RESPONSE_ASSIGNMENT_V1',parent='C01_FROZEN_QUERY',split='DIAGNOSTIC',panel_sha256=pin),
        source_phase_scene_binding=dict(scene_id=c01.SCENE_ID,parent=c01.PARENT,model_sha256=model,static_result=str(proof),static_result_sha256=digest(proof),nominal_calibration_goal_binding=str(reference),nominal_calibration_goal_binding_sha256=c01.NOMINAL_REFERENCE_SHA))
    return spec


@pytest.fixture
def c01_pair(slow_pair,tmp_path,monkeypatch):
    f=slow_pair;g,a,p=f['gpu'],f['adapter'],f['pair'];s=a.spec
    Path(s['model_path']).write_bytes(b'DISTINCT_SYNTHETIC_C01_MODEL_NOT_T06')
    registration(s,tmp_path,monkeypatch);p['scene_sha256']=digest(s['model_path'])
    wrapped=c01.episode(s['registered_root']);write_json(s['episode_manifest'],wrapped);s['episode_manifest_sha256']=digest(s['episode_manifest'])
    task=contract.task_document(digest(s['model_path']),f['source_payload']['language'],task_condition=c01.CONDITION)
    score=contract.score_contract(task_condition=c01.CONDITION);scorepath=tmp_path/'SCORE.json';write_json(scorepath,score);write_json(s['paused_target_task'],task)
    s.update(paused_target_task_file_sha256=digest(s['paused_target_task']),paused_target_task_sha256=doc_hash(task),paused_score_contract=str(scorepath),paused_score_contract_file_sha256=digest(scorepath),frozen_model_load_schedule=inference.PRELOAD_BEFORE_SNAPSHOT)
    p.update(task_condition=c01.CONDITION,native_endpoint_step=34303,task_sha256=doc_hash(task))
    g.update(task_condition=c01.CONDITION,paused_task_sha256=doc_hash(task),frozen_model_load_schedule=inference.PRELOAD_BEFORE_SNAPSHOT)
    member='src/go2arx5_vla/world_model/fusion_source_phase_c01.py';dest=Path(g['source_root'])/member;dest.write_bytes((PROJECT/member).read_bytes());g['source_files'][member]=digest(dest)
    from dataclasses import replace
    a.fixture_binding=replace(a.fixture_binding,scene_sha256=digest(s['model_path']))
    s['source_files']=g['source_files'].copy();bind(g,a,p);f['c01_wrapper']=wrapped;f['task']=task;a.fixture_task=task
    quota=(a.output/'quota.binding').read_text().splitlines();quota[4]='34303';(a.output/'quota.binding').write_text('\n'.join(quota)+'\n')
    return f


def test_new_scene_pair_uses_its_actual_model_and_calibration_recipe_separately(c01_pair):
    f=c01_pair;g,a,p=f['gpu'],f['adapter'],f['pair']
    inference.validate_spec(g);assert bridge.validate_pair_spec(g)==(p,a.spec)
    assert pair.validate_native_pause(a)is None
    assert f['c01_wrapper']['actual_scene_sha256']==digest(a.spec['model_path'])
    assert f['c01_wrapper']['calibration_prefix_recipe']['scene_sha256']!=digest(a.spec['model_path'])
    assert contract.terminal_step(c01.CONDITION)==34303 and contract.terminal_step(contract.MECHANICAL_TASK_CONDITION)==66503
    capture_source_archive(f);payload,identity,_=inference.read_payload(g)
    assert payload['language']==f['source_payload']['language']
    assert identity['native_endpoint_step']==34303 and not identity['source_task_phase_equivalence']
    assert identity['actual_scene_id']==c01.SCENE_ID and not identity['generated_action_execution_allowed']


@pytest.mark.parametrize('fault',['old_model','TRAIN','different_panel','unapproved_scope','old_endpoint','false_grant'])
def test_static_proof_does_not_allow_other_scenes_actions_or_relabelled_data(c01_pair,fault):
    s=copy.deepcopy(c01_pair['adapter'].spec)
    if fault=='old_model':s['source_phase_scene_binding']['model_sha256']='f'*64
    elif fault=='TRAIN':s['response_learning_assignment']['split']='TRAIN'
    elif fault=='different_panel':Path(s['response_learning_panel']).write_text('{}')
    elif fault=='unapproved_scope':s['program_reference_selection']={'method':'WM'}
    elif fault=='old_endpoint':s['max_ticks']=66503
    else:
        path=Path(s['source_phase_scene_binding']['static_result']);value=json.loads(path.read_text());value['new_native_admission']=True;write_json(path,value)
    with pytest.raises(ValueError):c01.scene(s)


def test_query_only_saves_full_draw_without_an_install_or_native_step(real_clock_composition,tmp_path,monkeypatch):
    from go2arx5_vla.world_model.fusion_trajectory_qualification_native import complete_reference_consumption
    a,samples,windows,report,raw,action,calls=real_clock_composition;registration(a.spec,tmp_path,monkeypatch)
    samples[-1]['object_pose']=samples[-1]['tcp'].copy();samples[-1]['object_pose'][2,3]-=.08
    report['task_phase_initialization']['readiness']['predicates']['actual_cube_distance']=False
    result=c01.query_only(a,report,samples,windows,task={'original':'full'},camera=object())
    assert len(calls)==1 and not a.advances and not a.installs and a.step==34303
    assert result['initialization_all40_knots_consumed'] and not result['all_40_knots_consumed']
    assert result['actual_VLA_inferences']==1 and result['actual_VLA_commands']==0
    assert complete_reference_consumption(a.spec,result)
    saved=json.loads((a.output/'SOURCE_PHASE_QUERY_ONLY_PROPOSAL.private.json').read_bytes())
    assert saved['original_decoded_physical_H40']==raw.tolist()
    bad=copy.deepcopy(result);bad['actual_VLA_commands']=1;assert not complete_reference_consumption(a.spec,bad)
    assert result['WM_updates']==result['Actor_updates']==0
    assert result['task_phase_initialization']['readiness']['predicates']['actual_cube_distance']is False
    assert not result['strict_short_desired_intent']['qualified']
    assert result['strict_short_desired_intent']['status']=='OUTSIDE_ORIGINAL_SHORT_TASK_START_DOMAIN'


def test_c01_actual_cube_and_preserved_nominal_reference_remain_distinct(tmp_path,monkeypatch):
    import numpy as np
    from scripts.pi0292.relay_protocol import unseal,sealed
    from go2arx5_vla.world_model import fusion_task_phase_initialization as original
    from test_fusion_task_phase_initialization import setup as original_setup
    fixture=original_setup.__wrapped__(monkeypatch);old=fixture['inputs']
    artifact=tmp_path/'ORIGINAL_GOAL.json';write_json(artifact,dict(frozen_input=old['frozen'],actual_VLA_commands=0))
    monkeypatch.setattr(c01,'NOMINAL_REFERENCE_SHA',digest(artifact));monkeypatch.setattr(c01,'CALIBRATION_SCENE',original.SCENE)
    reference=dict(path=str(artifact),sha256=digest(artifact));args=copy.deepcopy(fixture['args']);actual_cube=np.array(args['measured_decision_state']['object_pose']).copy();actual_cube[:3,3]+=[.10,.05,.11]
    args['measured_decision_state']['object_pose']=actual_cube;args['scene_sha256']=c01.SCENE
    with pytest.raises(ValueError):original.load_inputs(**args)
    new=original.load_inputs(**args,nominal_calibration_reference=reference);body=original._inputs(new)
    np.testing.assert_array_equal(body['frozen_cube_pose_target'],actual_cube)
    np.testing.assert_array_equal(body['full_goal_TCP_target'],unseal(old['frozen'])['full_goal_TCP_target'])
    assert not np.array_equal(body['frozen_cube_pose_target'],body['nominal_calibration_cube_pose_target'])
    assert new['provenance']['actual_cube_is_not_replaced_by_nominal_calibration_cube']is True
    from go2arx5_vla.world_model import fusion_task_phase_pose_slow_feedback_v9 as slow
    episode=slow.make_episode(run='/fixture/distinct-c01',geometry_scale=.5)
    before=slow.make_program(old,episode,fixture['initial'],fixture['decision'],fixture['base'])
    after=slow.make_program(new,episode,fixture['initial'],fixture['decision'],fixture['base'])
    np.testing.assert_array_equal(after.desired_path,before.desired_path)
    np.testing.assert_array_equal(after.reference_path,before.reference_path)
    body['nominal_calibration_cube_pose_target'][0][3]+=.001;bad=dict(new,frozen=sealed(body))
    with pytest.raises(ValueError,match='NOMINAL_REFERENCE'):original._inputs(bad)
    def forbidden_rederivation(*args,**kwargs):raise AssertionError('Metadata reader cannot rederive runtime arithmetic')
    monkeypatch.setattr(original,'_inputs',forbidden_rederivation)
    np.testing.assert_array_equal(c01.nominal_reference(reference),unseal(old['frozen'])['frozen_cube_pose_target'])


def test_same_endpoint_continuation_reaches_runtime_prepare_only_for_c01(monkeypatch):
    from types import SimpleNamespace
    from go2arx5_vla.world_model import fusion_trajectory_qualification_native as native
    class Prepared(Exception):pass
    def prepare():raise Prepared()
    spec=dict(phase='collect-target-fusion-task-phase-live-skill-v9',kind='FUSION_TASK_PHASE_LIVE_SKILL_V9',task_condition=c01.CONDITION,live_skill_start_step=34303,max_ticks=34303)
    a=SimpleNamespace(spec=spec,prepare_dynamic=prepare,trace_sealed=False);monkeypatch.setattr(native,'endpoint',lambda episode:34303)
    with pytest.raises(Prepared):native.run_program(a,{}, {},continuation=lambda *args:None)
    spec['task_condition']=contract.MECHANICAL_TASK_CONDITION
    with pytest.raises(ValueError,match='PREREGISTERED_QUERY_BOUNDARY'):native.run_program(a,{}, {},continuation=lambda *args:None)
