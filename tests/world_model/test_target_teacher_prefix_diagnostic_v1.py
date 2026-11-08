"""CPU consumer/role tests; no native qualification or expert data produced."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from go2arx5_vla.world_model import target_teacher_prefix_diagnostic_v1 as teacher
from go2arx5_vla.world_model import stable_task_prefix_v1 as consumer
from tests.world_model import test_target_gr00t_native_loop_v1 as prior


@pytest.fixture
def flow(tmp_path,monkeypatch):
    a,s,w,r,calls=prior.flow.__wrapped__(tmp_path,monkeypatch)
    for k in ('target_gr00t_policy','target_gr00t_gpu_root','target_gr00t_pair_id',
              'target_gr00t_query_seeds','target_gr00t_prefix_protocol'):
        a.spec.pop(k,None)
    a.spec.update(phase=teacher.PHASE,kind=teacher.KIND,max_ticks=teacher.E,data_role='TRAIN',
        target_teacher_prefix_diagnostic=teacher.protocol(),
        experiment_role='TARGET_TEACHER_TWO_PREFIX_DIAGNOSTIC_NOT_BC')
    monkeypatch.setattr(teacher,'validate',lambda spec:None)
    from go2arx5_vla.world_model.stable_task_policy_v1 import build_for_delta
    decisions=[]
    def build(actual,reference,base,cube,config):
        decisions.append((a.observe()['step'],np.asarray(actual).copy(),config.copy()))
        return build_for_delta(actual,reference,base,np.array([.003,0.,0.]))
    monkeypatch.setattr(teacher.tasks,'build',build)
    return a,s,w,r,decisions,calls


def test_two_fresh_privileged_decisions_and_original_complete_consumer(flow):
    a,s,w,r,decisions,model_calls=flow;before=copy.deepcopy(r)
    out=teacher.continue_native(a,r,tuple(s),tuple(w))
    assert r==before and [d[0]for d in decisions]==[34303,37303]
    assert all(d[2]['method']=='GEOMETRIC_TEACHER_4MM'and d[2]['step_m']==.004 for d in decisions)
    assert not np.array_equal(decisions[0][1],decisions[1][1])
    assert not model_calls and a.step==40503 and len(a.advances)==248
    assert [c['complete_suffix_atoms']for c in out['generated_cycles']]==[120,120,8]
    assert out['GR00T_queries']==out['GR00T_updates']==out['WM_updates']==out['Actor_updates']==0
    assert not out['expert_training_eligible']and not out['GR00T_native_interface_qualified']
    assert teacher.complete_consumption(a.spec,out)
    from go2arx5_vla.world_model.fusion_trajectory_qualification_native import complete_reference_consumption
    assert complete_reference_consumption(a.spec,out)
    altered=copy.deepcopy(out);altered['generated_cycles'][1]['windows'][0]['solver_policies_with_exact_installed_target']=[]
    assert not teacher.complete_consumption(a.spec,altered)
    altered=copy.deepcopy(out);altered['generated_cycles'][0]['cancellation']['new_reference_sha256']='wrong'
    assert not teacher.complete_consumption(a.spec,altered)
    altered=copy.deepcopy(out);altered['expert_training_eligible']=True
    assert not teacher.complete_consumption(a.spec,altered)


def test_original_standing_rejection_keeps_partial_and_does_not_produce_qualified_result(flow,monkeypatch):
    a,s,w,r,decisions,model_calls=flow
    from go2arx5_vla.world_model import fusion_task_reference_diagnostic_v1 as standing
    def reject(sample,initial):
        if sample['native_step']>=37428:raise ValueError('ORIGINAL_STANDING_BOUND_SYNTHETIC_TEST')
    monkeypatch.setattr(standing,'require_standing',reject)
    with pytest.raises(ValueError,match='ORIGINAL_STANDING_BOUND'):
        teacher.continue_native(a,r,tuple(s),tuple(w))
    failed=json.loads((a.output/'TEACHER_PREFIX_FAILED.private.json').read_bytes())
    assert failed['last_native_step']==37428 and failed['teacher_decisions']==2
    assert failed['GR00T_queries']==0 and failed['expert_training_eligible']is False
    assert len(decisions)==2 and not model_calls
    actual=json.loads((a.output/'STABLE_TASK_CYCLES/01/RESULT.private.json').read_bytes())
    assert actual['status']=='FAILED_PREFIX_RETAINED_NO_RETRY'and actual['complete_suffix_atoms']==4


def spec_fixture(tmp_path,monkeypatch):
    from tests.world_model.test_stable_task_data_v1 import fixture,put
    spec,_=fixture(tmp_path,monkeypatch,np.eye(4))
    spec.update(phase=teacher.PHASE,kind=teacher.KIND,max_ticks=teacher.E,
        task_command='APPROACH_RED_4MM_OPEN',target_teacher_prefix_diagnostic=teacher.protocol(),
        target_gr00t_start_readiness=teacher.tasks.start_readiness_contract(),
        physics_dt_s=.002,WBC_changes=False,live_skill_geometry_scale=1.,live_skill_time_stretch_factor=2,
        experiment_role='TARGET_TEACHER_TWO_PREFIX_DIAGNOSTIC_NOT_BC',
        snapshot_binding_mode='PRE_REGISTERED_TARGET_TEACHER_TWO_PREFIX_V1')
    panel=json.loads(Path(spec['stable_task_panel']).read_bytes());panel.update(
        schema='WM02_TARGET_TEACHER_PREFIX_DIAGNOSTIC_PANEL_V1',teacher_prefix_protocol=teacher.protocol())
    panel['assignments'][0]['command']=spec['task_command']
    d=put(tmp_path/'TEACHER_PANEL.json',panel);spec.update(stable_task_panel=d['path'],stable_task_panel_sha256=d['sha256'])
    return spec


def test_separate_diagnostic_role_uses_same_two_cycle_clocks_without_model_or_demo(tmp_path,monkeypatch):
    spec=spec_fixture(tmp_path,monkeypatch);teacher.validate(spec)
    assert consumer.observation_steps(spec)==(34303,37303)
    assert [consumer.cycle(spec,i)['atoms']for i in range(3)]==[120,120,8]
    assert consumer.cycle(spec,2)['end_step']==40503
    for k in ('target_gr00t_collection','target_gr00t_policy','target_gr00t_gpu_root','stable_task_candidate'):
        with pytest.raises(ValueError):teacher.validate(dict(spec,**{k:{'injected':True}}))
    with pytest.raises(ValueError):teacher.validate(dict(spec,max_ticks=38703))
    with pytest.raises(ValueError):teacher.validate(dict(spec,data_role='DEV'))
    with pytest.raises(ValueError):teacher.validate(dict(spec,experiment_role='DEMONSTRATION'))


def test_phase_dispatch_does_not_substitute_other_native_recipe(monkeypatch):
    from go2arx5_vla.world_model.auto1_native import phase_limit
    observed=[]
    monkeypatch.setattr(teacher,'episode_limit',lambda spec,previous:observed.append((spec,previous)) or 40503)
    assert phase_limit(teacher.PHASE,{'prior':1},{'new':2})==40503
    assert observed==[({'new':2},{'prior':1})]


def test_real_teacher_entry_crosses_shared_feedback_and_initialization_typed_guards(tmp_path,monkeypatch):
    """Exercise the actual pre-integration route that the first native run rejected."""
    from types import SimpleNamespace
    from go2arx5_vla.world_model import fusion_trajectory_qualification_native as lower
    episode=teacher.common.prefix.make_episode(run=str(tmp_path),geometry_scale=.5)
    source=dict(source_observation_bytes=b'CPU_FIXTURE_NOT_NATIVE_INPUT',T_source_from_target=np.eye(4),
        scene_sha256='f'*64,stable_task_scene_reference={'fixture_only':True})
    adapter=SimpleNamespace(spec=dict(phase=teacher.PHASE,kind=teacher.KIND,
        max_ticks=teacher.E,live_skill_start_step=teacher.B),output=tmp_path)
    monkeypatch.setattr(teacher,'episode_config',lambda spec:(episode,source))
    seen=[]
    def downstream(actual,ep,inputs,**kwargs):
        seen.append((actual,ep,kwargs));return {'reached_original_consumer':True}
    monkeypatch.setattr(lower,'run_program',downstream)
    assert teacher.run(adapter)=={'reached_original_consumer':True}
    assert seen[0][0]is adapter and seen[0][1]is episode
    assert seen[0][2]['continuation']is teacher.continue_native
    assert adapter.spec['phase']==teacher.PHASE
