"""Synthetic full-loop consumption tests, not real VLA or native qualification."""
import copy,json
from pathlib import Path
import numpy as np
import pytest

from go2arx5_vla.world_model import target_gr00t_native_loop_v1 as loop
from go2arx5_vla.world_model import target_gr00t_prefix_v1 as prefix
from go2arx5_vla.world_model import target_gr00t_profile_v1 as profile
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.relay_protocol import sealed,unseal
from tests.world_model import test_stable_task_closed_loop_v1 as prior


@pytest.fixture
def flow(tmp_path,monkeypatch):
    a,s,w,report,_=prior.flow.__wrapped__(tmp_path,monkeypatch)
    a.spec.pop('stable_task_closed_loop')
    a.spec.update(phase=prefix.PHASE,kind=prefix.KIND,max_ticks=prefix.E,
        target_gr00t_prefix_protocol=prefix.protocol(),
        target_gr00t_start_readiness=loop.tasks.start_readiness_contract(),
        target_gr00t_gpu_root=str(tmp_path/'gpu'),target_gr00t_pair_id='mock-pair',
        target_gr00t_query_seeds=[10,11],target_gr00t_policy=dict(arm='B0',
            parent_manifest_sha256=profile.PARENT,package=None,Core_qualification=None))
    # The base-consumer fixture supplies no real native velocity: make its
    # synthetic measurement complete, with no changed control or actual physics.
    for value in s:value.update(qvel=np.zeros(32),time_s=value['native_step']*.002)
    monkeypatch.setattr(loop,'validate',lambda spec:spec['target_gr00t_policy'])
    from go2arx5_vla.world_model import stable_task_data_v1 as data,fusion_observation
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from go2arx5_vla.world_model.stable_task_policy_v1 import build_for_delta
    gpu=Path(a.spec['target_gr00t_gpu_root']);(gpu/'control').mkdir(parents=True)
    (gpu/'EXECUTION_SPEC.json').write_text('{}')
    (gpu/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED',worker_exit_code=0)))
    from go2arx5_vla.world_model import fusion_paused_query_bridge as bridge
    monkeypatch.setattr(bridge,'_registered',lambda spec,path,**kw:kw)
    monkeypatch.setattr(loop,'digest',lambda path:digest(path)if Path(path).exists()else'synthetic-scene')
    calls=[]
    def capture(adapter,**kw):
        step=adapter.observe()['step'];calls.append(('capture',step,kw['language']))
        tcp=adapter.measured_task_state()['tcp'];state=pose10(tcp,1.).tolist()
        shadow=sealed(dict(source_policy_view=dict(T_source_from_target=np.eye(4).tolist()),
            synthetic_payload=dict(state_eef10=state)))
        path=tmp_path/f'observation-{step}.json';path.write_text(json.dumps(sealed(dict(original_shadow_ABI=shadow))))
        return dict(path=str(path),sha256=digest(path),native_step=step,native_time_s=step*.002)
    monkeypatch.setattr(data,'capture_input',capture)
    monkeypatch.setattr(fusion_observation,'shadow_inference_payload',lambda record:unseal(record)['synthetic_payload'])
    def query(adapter,observation,index):
        calls.append(('query',index,observation['native_step']))
        program,_=build_for_delta(adapter.measured_task_state()['tcp'],adapter.reference_history[-1]['pose'],
            adapter.base_pose_hold,np.array([.003,0.,0.]))
        folder=gpu/'fit-01/queries'/f'{index:02d}';folder.mkdir(parents=True)
        file=folder/'ACTION.npz';np.savez(file,physical_H40=program.decoded_H40,normalized=np.zeros((1,40,132)))
        returned=dict(index=index,actual_model_inferences=1,schema=loop.pair.SCHEMA+'_RESULT')
        (folder/'RESULT.json').write_text(json.dumps(returned))
        return file,digest(file),dict(result=returned,result_sha256=digest(folder/'RESULT.json'))
    monkeypatch.setattr(loop.pair,'query_from_native',query)
    return a,s,w,report,calls


def test_two_actual_new_observation_cycles_and_only_explicit_final_hold(flow):
    a,s,w,initial,calls=flow
    result=loop.continue_native(a,initial,tuple(s),tuple(w))
    assert [v[1]for v in calls if v[0]=='capture']==[34303,37303]
    assert [v[2]for v in calls if v[0]=='query']==[34303,37303]
    assert a.step==40503 and [v['complete_suffix_atoms']for v in result['generated_cycles']]==[120,120,8]
    assert result['actual_VLA_inferences']==2 and result['TASK_ADAPTER_inferences']==0
    assert result['teacher_fallbacks']==result['GR00T_updates']==0
    assert result['raw_reference_is_GR00T']is True and result['GR00T_raw_H40_consumed']is False
    assert loop.complete_consumption(a.spec,result)
    assert (a.output/'TARGET_GR00T_QUERY_RETURN_01.private.json').is_file()
    bad=copy.deepcopy(result);bad['policy_observations'][1]['native_step']=34303
    assert not loop.complete_consumption(a.spec,bad)
    bad=copy.deepcopy(result);bad['generated_cycles'][0]['cancellation_status']='UNCONFIRMED'
    assert not loop.complete_consumption(a.spec,bad)
    bad=copy.deepcopy(result);bad['generated_cycles'][1]['windows'][0]['gripper']['command_consumed']=False
    assert not loop.complete_consumption(a.spec,bad)


def test_unsafe_first_effect_keeps_completed_query_without_another_draw(flow):
    a,s,w,initial,calls=flow;a.effect_fault='unsafe'
    with pytest.raises(ValueError):loop.continue_native(a,initial,tuple(s),tuple(w))
    assert len([v for v in calls if v[0]=='query'])==1 and len(a.advances)==1
    evidence=json.loads((a.output/'TARGET_GR00T_LOOP_FAILED.private.json').read_bytes())
    assert evidence['actual_paired_GPU_query_records']==1 and evidence['retry_authorized']is False
    assert (a.output/'TARGET_GR00T_QUERY_RETURN_00.private.json').is_file()


def test_new_native_route_never_delegates_to_old_MLP_or_Teacher(monkeypatch):
    from go2arx5_vla.world_model.auto1_native import phase_limit
    from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
    seen=[];spec=dict(phase=loop.PHASE,kind=loop.KIND)
    monkeypatch.setattr(loop,'episode_limit',lambda s,p:seen.append(s)or loop.E)
    assert phase_limit(loop.PHASE,None,spec)==40503 and seen==[spec]
    monkeypatch.setattr(loop,'run',lambda a:seen.append(a)or'actual-new-route')
    class Adapter:pass
    a=Adapter();a.spec=spec
    assert TargetSequentialAdapter.dynamic_diagnostic(a)=='actual-new-route'


def test_DEV_needs_a_safe_two_query_native_interface_and_unchanged_frozen_control(monkeypatch):
    from go2arx5_vla.world_model import stable_task_prefix_qualification_v1 as qualified
    with pytest.raises(ValueError,match='REAL_NATIVE_INTERFACE_QUALIFICATION_REQUIRED'):
        loop.require_interface_qualification({})
    files={n:'a'*64 for n in loop.SOURCE_MEMBERS}
    q=dict(phase=loop.PHASE,kind=loop.KIND,data_role='TRAIN',
        experiment_role='TARGET_GR00T_NATIVE_INTERFACE_QUALIFICATION_NOT_BC',
        source_files=files,frozen_files={'frozen-WBC':'b'*64,'TRAIN-scene':'c'*64},model_path='TRAIN-scene')
    result=dict(reference_qualification={'synthetic-consumption':True})
    monkeypatch.setattr(qualified,'sealed_native',lambda descriptor:('/synthetic',q,result))
    monkeypatch.setattr(loop,'complete_consumption',lambda s,r:r=={'synthetic-consumption':True})
    s=dict(source_files=dict(files),frozen_files={'frozen-WBC':'b'*64,'DEV-scene':'d'*64},
        target_gr00t_native_interface_qualification=dict(path='/synthetic/qualified',sha256='f'*64))
    loop.require_interface_qualification(s)
    s['frozen_files']['frozen-WBC']='e'*64
    with pytest.raises(ValueError,match='SAME_FROZEN_NATIVE_CONTROL'):loop.require_interface_qualification(s)
    s['frozen_files']['frozen-WBC']='b'*64;s['source_files'][next(iter(files))]='f'*64
    with pytest.raises(ValueError,match='SAME_IMPLEMENTATION'):loop.require_interface_qualification(s)


def test_new_scene_panel_retains_actual_task_role_policy_and_original_layout(tmp_path,monkeypatch):
    from tests.world_model.test_stable_task_data_v1 import fixture,put
    spec,_=fixture(tmp_path,monkeypatch,np.eye(4))
    spec.update(phase=loop.PHASE,kind=loop.KIND,max_ticks=loop.E,
        task_command='RETREAT_RED_4MM_OPEN',target_gr00t_prefix_protocol=prefix.protocol(),
        target_gr00t_start_readiness=loop.tasks.start_readiness_contract(),
        target_gr00t_policy=dict(arm='B0',parent_manifest_sha256=profile.PARENT,package=None,Core_qualification=None),
        target_gr00t_query_seeds=[10,11],physics_dt_s=.002,WBC_changes=False,
        experiment_role='TARGET_GR00T_NATIVE_INTERFACE_QUALIFICATION_NOT_BC',
        snapshot_binding_mode='PRE_REGISTERED_TARGET_GR00T_PREFIX_PAIR_V1')
    panel=json.loads(Path(spec['stable_task_panel']).read_bytes());panel.update(
        schema='WM02_TARGET_GR00T_NATIVE_PREFIX_PANEL_V1',target_gr00t_prefix_protocol=prefix.protocol(),
        policies=dict(B0=spec['target_gr00t_policy']))
    panel['assignments'][0].update(command=spec['task_command'],arm='B0',seeds=[10,11])
    d=put(tmp_path/'GR00T_PANEL.json',panel);spec.update(stable_task_panel=d['path'],stable_task_panel_sha256=d['sha256'])
    assert loop.validate(spec)==spec['target_gr00t_policy']
    with pytest.raises(ValueError,match='TRAIN_INTERFACE'):loop.validate(dict(spec,experiment_role='FORMAL_DEV'))
    with pytest.raises(ValueError,match='NEW_GR00T_LOOP'):loop.validate(dict(spec,program_reference_selection={'model':'WM'}))
