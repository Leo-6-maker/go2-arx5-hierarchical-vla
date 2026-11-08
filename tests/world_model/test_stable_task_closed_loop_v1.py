"""Synthetic same-world loop tests; no model learning or physics qualification."""
import copy
from pathlib import Path
import numpy as np
import pytest
from go2arx5_vla.world_model import stable_task_closed_loop_v1 as loop
from go2arx5_vla.world_model import stable_task_prefix_v1 as prefix
from tests.world_model import test_stable_task_prefix_v1 as partial
from tests.world_model import test_fusion_live_skill_segment as old


@pytest.fixture
def flow(tmp_path,monkeypatch):
    adapter,program,samples,windows=partial.flow.__wrapped__(tmp_path,monkeypatch)
    adapter.spec.update(task_command='APPROACH_RED_4MM_OPEN',data_role='DEV',data_parent='SYNTHETIC_DEV',model_path='/synthetic/model')
    measured=adapter.measured_task_state
    def safe_fixture():
        value=measured();value.update(robot_object_contact=False,left_object_contact=False,right_object_contact=False,native_grasp=False,gripper_state_valid=True,object_present=True)
        return value
    adapter.measured_task_state=safe_fixture
    for value in samples:value.update(robot_object_contact=False,left_object_contact=False,right_object_contact=False,native_grasp=False,gripper_state_valid=True,object_present=True)
    install=adapter.install_reference
    def current_install(action):
        adapter.program=adapter.stable_task_active_cycle['program']
        before=old.B;old.B=adapter.stable_task_active_cycle['start_step']
        try:return install(action)
        finally:old.B=before
    adapter.install_reference=current_install
    calls=[]
    monkeypatch.setattr(loop,'validate',lambda spec:dict(export={'synthetic':True}))
    monkeypatch.setattr(loop.policy,'load_export',lambda d:{'synthetic':True})
    def capture(a):
        return dict(path='/synthetic/observation',sha256='synthetic',native_step=a.observe()['step'])
    monkeypatch.setattr(loop,'capture_input',capture)
    def predict(spec,observation,arrays,*,expected_step):
        calls.append((observation['native_step'],expected_step))
        assert observation['native_step']==expected_step==adapter.step
        return np.array([.003,0.,0.]),dict(synthetic=True)
    monkeypatch.setattr(loop.policy,'predict_from_observation',predict)
    monkeypatch.setattr(loop.dataset,'load',lambda *a,**k:({'synthetic':True},{}))
    monkeypatch.setattr(loop,'digest',lambda path:'synthetic-scene')
    from go2arx5_vla.world_model import fusion_paused_observation,fusion_task_reference_diagnostic_v1
    monkeypatch.setattr(fusion_paused_observation,'verify_still_paused',lambda a,r:dict(step=a.observe()['step']))
    monkeypatch.setattr(fusion_task_reference_diagnostic_v1,'require_standing',lambda *a:None)
    monkeypatch.setattr(loop.common,'build_for_config',lambda *a,**k:pytest.fail('Teacher used in policy loop'))
    initial=dict(all_40_knots_consumed=True,raw_knot_consumed=[True]*40,windows=copy.deepcopy(windows),
        task_phase_initialization=dict(readiness=dict(predicates=dict(full_grade=False,fixture_valid=True))))
    return adapter,samples,windows,initial,calls


def test_three_genuinely_new_observations_and_explicit_final_hold(flow):
    adapter,samples,windows,initial,calls=flow;before=copy.deepcopy(initial)
    value=loop.continue_native(adapter,initial,tuple(samples),tuple(windows))
    assert calls==[(34303,34303),(37303,37303),(40303,40303)]
    assert adapter.step==43503 and len(adapter.advances)==368
    assert [c['complete_suffix_atoms']for c in value['generated_cycles']]==[120,120,120,8]
    assert value['TASK_ADAPTER_inferences']==value['policy_queries']==3
    assert value['teacher_fallbacks']==value['TASK_ADAPTER_updates']==0
    assert value['all_40_knots_consumed']is False and value['GR00T_raw_H40_consumed']is False
    assert value['real_time_deployment']is False
    assert initial==before and value['windows'][:len(windows)]==windows
    assert loop.complete_consumption(adapter.spec,value)
    assert adapter.cancel_calls==1 and not adapter.trace_sealed # outer native worker owns these
    for i,c in enumerate(value['generated_cycles'][:3]):
        assert c['cancellation_status']=='SUPERSEDED_BY_ORIGINAL_NEW_REFERENCE_ACK'
        replacement=c['cancellation'];next_cycle=value['generated_cycles'][i+1]
        assert replacement['native_step']==next_cycle['cycle']['start_step']
        assert replacement['old_reference_sha256']==c['final_reference_sha256']
        assert replacement['future_may_have_influenced_previous_MPC']is True
        assert (adapter.output/'STABLE_TASK_CYCLES'/f'{i:02d}'/'REPLACEMENT.private.json').exists()
    bad=copy.deepcopy(value);bad['generated_cycles'][1]['cancellation_status']='UNCONFIRMED'
    assert not loop.complete_consumption(adapter.spec,bad)
    bad=copy.deepcopy(value);bad['policy_observations'][1]['native_step']=34303
    assert not loop.complete_consumption(adapter.spec,bad)


def test_original_full_h40_consumer_does_not_accept_partial_protocol(flow):
    from go2arx5_vla.world_model import fusion_live_skill_segment as segment
    adapter,samples,windows,_,_=flow
    adapter.spec['max_ticks']=prefix.B+3000
    with pytest.raises(ValueError,match='TYPED_PHASE_REQUIRED'):
        segment.run_consumed_segment(adapter,adapter.program,start_step=prefix.B,
            end_step=prefix.B+3000,past_samples=samples,past_windows=windows)
    assert not adapter.advances


def test_closed_loop_dispatch_keeps_actual_phase_and_identity(monkeypatch):
    from go2arx5_vla.world_model.auto1_native import phase_limit
    seen=[];spec=dict(phase=loop.PHASE,kind=loop.KIND)
    monkeypatch.setattr(loop,'episode_limit',lambda s,p:seen.append(s)or prefix.E)
    assert phase_limit(loop.PHASE,None,spec)==43503
    assert seen==[spec]


def test_frozen_closed_loop_scene_and_gate_bind_one_candidate(tmp_path,monkeypatch):
    import json
    from go2arx5_vla.world_model import stable_task_prefix_qualification_v1 as qualification
    from go2arx5_vla.world_model.storage import digest
    from tests.world_model.test_stable_task_evaluation_v1 import configuration,put
    spec,_,_=configuration(tmp_path,monkeypatch)
    candidate={k:spec['stable_task_evaluation'][k]for k in ('training_result','export')}
    proof=dict(schema=qualification.SCHEMA,
        status='REAL_TRAIN_PREFIX_AND_SEALED_CANDIDATE_DEV_CHECKED_NOT_DYNAMIC_PERMISSION',
        candidate=candidate,proposed_protocol=prefix.protocol(),qualified_TRAIN_prefix_native_ticks=3000,
        actual_numpy_inference_calls=200,native_integrations=0,optimizer_updates=0,teacher_fallbacks=0,
        native_execution_authorized=False,qualification_module_sha256=digest(qualification.__file__),
        TRAIN_native_result={'synthetic':'TRAIN'},DEV_native_result={'synthetic':'DEV'})
    descriptor=put(tmp_path/'QUALIFICATION.json',proof)
    panel=json.loads(Path(spec['stable_task_panel']).read_text())
    panel.update(schema='WM02_STABLE_TASK_CLOSED_LOOP_PANEL_V1',candidate=candidate,
        closed_loop_protocol=prefix.protocol(),prefix_qualification=descriptor)
    panel_descriptor=put(tmp_path/'LOOP_PANEL.json',panel)
    spec.update(phase=loop.PHASE,kind=loop.KIND,max_ticks=prefix.E,stable_task_closed_loop=prefix.protocol(),
        stable_task_candidate=candidate,stable_task_prefix_qualification=descriptor,
        snapshot_binding_mode='PRE_REGISTERED_STABLE_TASK_CLOSED_LOOP_V1',
        stable_task_panel=panel_descriptor['path'],stable_task_panel_sha256=panel_descriptor['sha256'],
        physics_dt_s=.002,WBC_changes=False)
    seen=[]
    monkeypatch.setattr(qualification,'verify_inputs',lambda *a:seen.append(a))
    assert loop.validate(spec)==candidate and len(seen)==1
    bad=copy.deepcopy(spec);bad['stable_task_candidate']['export']['sha256']='0'*64
    with pytest.raises(ValueError,match='REAL_PINNED_TRAIN_TIMING'):loop.validate(bad)
    bad=copy.deepcopy(spec);bad['physics_dt_s']=.001
    with pytest.raises(ValueError,match='EXACT_NEW_TYPED'):loop.validate(bad)
