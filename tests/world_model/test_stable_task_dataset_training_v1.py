"""Synthetic data-contract/trainer tests, never reported as real GPU learning."""
import copy
import numpy as np
import pytest
from go2arx5_vla.world_model import stable_task_dataset_v1 as dataset
from go2arx5_vla.world_model import stable_task_training_v1 as training
from go2arx5_vla.world_model import stable_task_adapter_v1 as adapter


def qualification():
    rows=[]
    for i in range(176):
        ref=np.zeros(17);ref[0]=.004*min(i,160)/160
        rows.append(dict(native_start_step=34303+25*i,native_end_step=34303+25*(i+1),actual_ticks=25,
            actual_installed_consumed_EEF10_BASE7=ref.tolist(),actual_TCP=np.eye(4).tolist(),
            solver_policies_with_exact_installed_target=[{'confirmed':True}],
            gripper=dict(command_consumed=True,submitted=dict(sequence=1272+i))))
    return dict(capability_reference_all40_consumed=True,generated_segment=dict(windows=rows,
        all_40_knots_consumed=True,raw_knot_consumed=[True]*40,complete_suffix_atoms=176))


def test_supervision_uses_consumed_command_not_future_TCP():
    q=qualification();label=dataset.consumed_label(q)
    for row in q['generated_segment']['windows']:row['actual_TCP'][0][3]=999.
    np.testing.assert_array_equal(dataset.consumed_label(q),label)
    np.testing.assert_allclose(label,[.004,0,0],atol=1e-15)
    q['generated_segment']['windows'][-1]['gripper']['command_consumed']=False
    with pytest.raises(ValueError,match='SOLVER_GRIPPER'):
        dataset.consumed_label(q)


def test_unexecuted_suffix_never_becomes_a_full_horizon_label():
    q=qualification();q['generated_segment']['raw_knot_consumed'][-1]=False
    with pytest.raises(ValueError,match='FULL_REAL_REFERENCE'):
        dataset.consumed_label(q)


def spec():
    return dict(kind=training.KIND,profile=training.PROFILE,execution_site='CORE_PILOT1_NEURAL',
        candidate_recipe=training.RECIPE,start_step=0,endpoint=2000,TASK_ADAPTER_updates=2000,
        GR00T_updates=0,Actor_updates=0,WM_updates=0,frozen_GR00T_manifest_sha256=training.FROZEN_GR00T,
        seed=110801,learning_rate=.001,weight_decay=.001,batch_size=16,WBC_changes=False,physics=False)


@pytest.mark.parametrize('fault',['zero','fusion','GR00T','formal_actor','unknown_seed'])
def test_candidate_contract_cannot_hide_learning_or_unfreeze_baseline(fault):
    value=spec();training.contract(value)
    if fault=='zero':value['TASK_ADAPTER_updates']=0
    if fault=='fusion':value['fusion_binding']='/old/zero-update/window'
    if fault=='GR00T':value['GR00T_updates']=1
    if fault=='formal_actor':value['Actor_updates']=1
    if fault=='unknown_seed':value['seed']=999
    with pytest.raises(PermissionError,match='NONZERO_TASK_ADAPTER'):
        training.contract(value)


def numeric_fixture():
    rng=np.random.default_rng(1);x=rng.normal(size=(10,adapter.INPUT_DIM)).astype(np.float32)
    rows=[];y=[]
    for i in range(10):
        command=adapter.COMMANDS[i%2];role='TRAIN'if i<6 else'DEV'
        rows.append(dict(parent_id=role+str(i//2),data_role=role,task_command=command))
        y.append([0.,0.,0.]if i%2==0 else[.002,.0001,0.])
    train,dev=adapter.split_rows(rows)
    return x,np.asarray(y,np.float32),rows,train,dev


def test_finite_trainer_exports_and_restores_optimizer_sampler_and_selection(tmp_path):
    x,y,rows,train,dev=numeric_fixture();out=tmp_path/'fit';out.mkdir()
    provenance=dict(origin='SYNTHETIC_SOFTWARE_ONLY_NOT_REAL_NATIVE_TRAIN')
    result=training.fit(x,y,rows,train,dev,out,seed=110801,endpoint=4,
        provenance=provenance,device='cpu')
    assert result['optimizer_updates']==result['TASK_ADAPTER_updates']==4
    assert result['GR00T_updates']==result['Actor_updates']==0
    assert result['optimizer_restored']and result['sampler_restored']and result['RNG_restored']
    assert result['DEV_native_task_success']=='NOT_RUN'and result['export_max_error_m']<1e-7
    resumed=tmp_path/'resumed';resumed.mkdir()
    continued=training.fit(x,y,rows,train,dev,resumed,seed=110801,endpoint=6,start_step=4,
        resume=out/'CHECKPOINT_00000004.pt',provenance=provenance,device='cpu')
    assert continued['optimizer_updates']==2 and continued['curve'][0]['step']==0
    assert continued['curve'][-1]['step']==6


def test_hardware_scope_route_still_checks_actual_pilot1_binding(monkeypatch):
    from go2arx5_vla.world_model import pilot1_scope
    value=spec();value.update(GPU_uuid=next(iter(pilot1_scope.GPU_UUIDS)),allocator_limit_bytes=512*1024**2)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',value['GPU_uuid']);called=[]
    monkeypatch.setattr(pilot1_scope,'validate',lambda spec:called.append(spec))
    pilot1_scope.validate_gpu(value,dict(uuid=value['GPU_uuid'],free_bytes=4*1024**3))
    assert called==[value]
