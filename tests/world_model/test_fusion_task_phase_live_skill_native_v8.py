"""Hermetic query-to-consumption composition; no native/GPU qualification."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from go2arx5_vla.world_model import fusion_task_phase_live_skill_native_v8 as live
from go2arx5_vla.world_model import fusion_task_phase_live_skill_contract_v8 as contract
from go2arx5_vla.world_model import fusion_paused_query_bridge as bridge
from go2arx5_vla.world_model import fusion_paused_inference as inference
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_native_v4 as feedback
from go2arx5_vla.world_model.target_sequential import native_task_bucket
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.fusion_observation import transform_hash
from scripts.pi0292.relay_protocol import digest as document_hash
from test_fusion_live_skill_segment import flow, B, eef


@pytest.fixture
def composition(flow,tmp_path,monkeypatch):
    adapter,old_program,samples,windows=flow
    flags=dict(robot_object_contact=False,left_object_contact=False,right_object_contact=False,
        native_grasp=False,gripper_state_valid=True,object_present=True)
    for sample in samples:sample.update(flags)
    measured=adapter.measured_task_state
    def state():
        value=measured();value.update(flags);return value
    monkeypatch.setattr(adapter,'measured_task_state',state)
    gpu=tmp_path/'gpu';out=gpu/'fit-01';out.mkdir(parents=True)
    model=tmp_path/'model.mjb';model.write_bytes(b'NO_MODEL_LOADED')
    gpu_spec=dict(registered_root=str(gpu))
    (gpu/'EXECUTION_SPEC.json').write_text(json.dumps(gpu_spec))
    identity=dict(synthetic_query_identity='ONE_FIXED_DRAW')
    raw=np.asarray(old_program.decoded_H40).copy()
    action=out/'REAL_VLA_ACTION.npz';np.savez(action,physical_H40=raw)
    (out/'RESULT.private.json').write_text(json.dumps(dict(action_sha256=digest(action),observation_binding=identity)))
    adapter.spec.update(kind=live.KIND,paused_query_gpu_spec=str(gpu/'EXECUTION_SPEC.json'),
        model_path=str(model),registered_root=str(tmp_path),paused_source_from_target=np.eye(4).tolist(),
        paused_frame_map_sha256=transform_hash(np.eye(4)))
    monkeypatch.setattr(live,'B',B)
    original=live.TrajectoryAdaptation
    def construct(*args,**kwargs):
        value=original(*args,**kwargs);adapter.program=value
        return value
    monkeypatch.setattr(live,'TrajectoryAdaptation',construct)
    proposal=original(raw,samples[-1]['tcp'],np.eye(4),samples[-1]['tcp'],
        adapter.reference_history[-1]['pose'],adapter.base_pose_hold,1.,time_stretch_factor=2)
    end=B+25*(proposal.end_atom+8);adapter.spec['max_ticks']=end
    (tmp_path/'EXECUTION_SPEC.json').write_text(json.dumps(adapter.spec))
    monkeypatch.setattr(live,'E',end)
    calls=[]
    def capture(current,**kwargs):
        assert current is adapter and current.step==B and not current.trace_sealed
        assert current.latest['mode'].data=='WBC'
        assert not current.installs
        calls.append(kwargs)
        return dict(query_result=dict(action_sha256=digest(action)),synthetic=True)
    monkeypatch.setattr(bridge,'capture_and_wait_query',capture)
    monkeypatch.setattr(inference,'read_payload',lambda spec:(
        dict(state_eef10=eef(samples[-1]['tcp']).tolist()),identity,{'synthetic':True}))
    report=dict(task_phase_initialization=dict(readiness=dict(query_ready=False,
        predicates=dict(full_grade=False,position=True,lateral=True,orientation=True,
          stability=True,velocity=True,open=True,noncontact=True,object=True,base=True))),
        all_40_knots_consumed=True,windows=windows,original_Processor_decodes=0,Processor_decodes=0)
    report.update(adaptation={'initialization_prefix_only':True},actual_endpoint_TCP=samples[-1]['tcp'].tolist(),
        original_install_ACKs=[{'prefix':True}],task_native_ticks=B-2503,post_decision_atoms=(B-3703)//25)
    return adapter,samples,windows,report,raw,action,calls


def test_one_query_preserves_entire_draw_and_consumes_same_world_without_finalizing(composition):
    adapter,samples,windows,report,raw,action,calls=composition
    original=copy.deepcopy(report);spec=copy.deepcopy(adapter.spec)
    result=live._continue(adapter,report,samples,windows,task={'full':'original'},camera=object())
    assert len(calls)==1 and report==original and adapter.spec==spec
    assert not adapter.trace_sealed and adapter.cancel_calls==1  # Fixture's earlier cancel only.
    assert result['VLA_queries']==1 and result['unique_generated_action_chunks']==1
    assert result['additional_action_decodes']==result['native_process_Processor_decodes']==0
    assert result['WM_updates']==result['Actor_updates']==result['GR00T_updates']==0
    assert result['complete_task_atoms']==(adapter.spec['max_ticks']-2503)//25
    assert result['windows'][:len(windows)]==windows
    assert result['all_40_knots_consumed'] and all(result['raw_knot_consumed'])
    saved=json.loads((adapter.output/'LIVE_SKILL_GENERATED_PROPOSAL.private.json').read_text())
    np.testing.assert_array_equal(saved['original_decoded_physical_H40'],raw)
    assert saved['clips']==0 and saved['geometry_scale']==1. and saved['time_stretch_factor']==2
    assert result['source_task_phase_equivalence']is False and result['full_task_success']is False
    assert result['task_phase_initialization']['readiness']['query_ready']is False
    final=adapter.measured_task_state()['tcp']
    np.testing.assert_array_equal(result['actual_endpoint_TCP'],final)
    np.testing.assert_allclose(result['actual_minus_desired_xyz_m'],final[:3,3]-adapter.program.desired_at(adapter.program.end_atom)[:3],rtol=0,atol=1e-15)
    assert result['task_native_ticks']==adapter.spec['max_ticks']-2503
    assert result['post_decision_atoms']==(adapter.spec['max_ticks']-3703)//25
    assert result['adaptation']==adapter.program.evidence()
    assert result['initialization_adaptation']==report['adaptation']
    assert result['original_install_ACKs'][0]=={'prefix':True}
    assert len(result['original_install_ACKs'])==1+len(result['generated_segment']['original_install_ACKs'])


def test_selected_reference_is_used_in_outer_endpoint_and_full_adaptation_report(composition,monkeypatch):
    from dataclasses import replace
    from go2arx5_vla.world_model import fusion_program_reference_selection_v1 as selection
    from test_fusion_program_reference_intervention_v1 import document
    adapter,samples,windows,report,raw,action,calls=composition
    adapter.spec['program_reference_selection']={'synthetic_fixture_only':True}
    class Recorder:
        models={'FIXTURE':None};records=[];wall_s=0.
        def save(self,*args,**kwargs):pass
    def prepare(current,program,*args):
        chosen=replace(program,reference_intervention=document('Y_PLUS_050'))
        adapter.program=chosen
        return chosen,Recorder(),dict(proposed_reference_changes=1)
    monkeypatch.setattr(selection,'prepare',prepare)
    result=live._continue(adapter,report,samples,windows,task={},camera=object())
    assert result['adaptation']==adapter.program.evidence()==result['generated_segment']['adaptation']
    assert result['reference_intervention']['candidate']=='Y_PLUS_050'
    assert result['generated_segment']['model_reference_changes']==1
    final=np.asarray(result['actual_endpoint_TCP'])[:3,3]
    np.testing.assert_allclose(result['actual_minus_reference_xyz_m'],
        final-adapter.program.preview(adapter.program.end_atom)[0,:3],rtol=0,atol=1e-15)


def test_full_proposal_outside_noncontact_domain_is_saved_rejected_without_install(composition):
    adapter,samples,windows,report,raw,action,calls=composition
    cube=np.asarray(samples[-1]['tcp']).copy();cube[2,3]-=.13
    samples[-1]['object_pose']=cube
    with pytest.raises(ValueError,match='FULL_PROPOSAL_OUTSIDE_PREDECLARED_NONCONTACT'):
        live._continue(adapter,report,samples,windows,task={'full':'original'},camera=object())
    assert len(calls)==1 and not adapter.installs and not adapter.advances
    assert (adapter.output/'LIVE_SKILL_GENERATED_PROPOSAL.private.json').exists()
    failed=json.loads((adapter.output/'LIVE_SKILL_FAILED.private.json').read_text())
    assert failed['proposal_saved']is True and failed['automatic_retry']is False


def test_query_cannot_sneak_full_grade_reclassification(composition):
    adapter,samples,windows,report,_,_,calls=composition
    report['task_phase_initialization']['readiness']['query_ready']=True
    with pytest.raises(ValueError,match='NOT_RELABELLED_FULL_QUERY_READY'):
        live._continue(adapter,report,samples,windows,task={},camera=None)
    assert calls==[] and not adapter.installs and not adapter.advances


def test_declared_live_scope_uses_original128_bucket_and_old_v8_stays96():
    assert native_task_bucket(contract.PHASE,True)=='128'
    assert native_task_bucket('collect-target-fusion-task-phase-bias-fade-v8',True)=='96'
    assert native_task_bucket('collect-target-fusion-task-phase-pose-continuation-v7',True)=='96'


def test_new_score_keeps_old_readiness_and_original_full_language():
    score=contract.score_contract();task=contract.task_document('b'*64,'unchanged original language')
    assert task['language']=='unchanged original language'
    assert task['score_contract_sha256']==document_hash(score)
    assert score['raw_knots_required']==40 and not score['source_task_phase_equivalence']
    assert not score['old_prefix_query_ready_reclassified']
    assert score['capture_boundary_step']==34303 and score['registered_final_step']==66503


@pytest.mark.parametrize('field',['robot_object_contact','left_object_contact','right_object_contact','native_grasp'])
def test_noncontact_observer_rejects_actual_contact_even_with_safety_valid(composition,field):
    _,samples,*_=composition
    sample=copy.deepcopy(samples[-1]);sample[field]=True
    with pytest.raises(ValueError,match='NONCONTACT_MEASUREMENT'):
        live._noncontact(sample,sample['object_pose'])


def test_new_wrapper_forwards_the_actual_v7_half_and_preseal_continuation(monkeypatch):
    episode={'fixture':'prefix'};source={'fixture':'source'};task={'fixture':'full language'};camera=object()
    monkeypatch.setattr(live,'episode_config',lambda spec:(episode,source,task,camera))
    calls=[]
    def run(adapter,e,s,**kwargs):calls.append((e,s,kwargs));return 'prefix-run'
    monkeypatch.setattr(feedback,'run_feedback_program',run)
    assert live.run(SimpleNamespace(spec={}))=='prefix-run'
    e,s,k=calls[0]
    assert e is episode and s is source and k['recipe_module']is live.prefix
    assert k['actual_phase']==contract.PHASE and callable(k['continuation'])
