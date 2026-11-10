"""Admission and comparison seams; these tests do not qualify a real A40."""
import json
import pytest
from go2arx5_vla.world_model import task_learning_vanda_restore_v3 as v
from go2arx5_vla.world_model import target_gr00t_profile_v1 as p
from go2arx5_vla.world_model.target_gr00t_training_v1 import dataset_type,semantic_recipe
from go2arx5_vla.world_model.task_learning_native_dataset_v1 import TaskLearningNativeDataset


def spec(start=0,end=4):
    return dict(profile=v.PROFILE,kind=v.KIND,recipe=v.recipe(v.DATA,v.STATS,endpoint=end),
        execution_site='VANDA_A40',physics=False,WBC_changes=False,start_step=start,
        GR00T_updates=end-start,WM_updates=0,Actor_updates=0,TASK_ADAPTER_updates=0,
        parent_manifest_sha256=p.PARENT,initial_package=dict(path='/private/package',sha256=v.PACKAGE),
        resume=dict(path='/private/cp',sha256='a'*64)if start else None,deadline_epoch=1794400000.)


@pytest.mark.parametrize('start,end',[(0,2),(0,4),(2,4),(4,5)])
def test_independent_full_scope_from_sft2000_with_original_formal_schedule(start,end):
    s=spec(start,end);r=p.validate(s)
    assert p.expected_recipe(r)==r
    assert r['mode']=='RESTORE4'and r['variant']=='FULL_ACTION_SIDE'
    assert r['scheduler_horizon']==8000 and r['warmup_updates']==200
    assert r['effective_batch']==32 and r['microbatch']==1 and r['accumulation']==32
    assert r['memory_policy']=='CUDA_RESIDENT'and r['initial_package_sha256']==v.PACKAGE
    assert dataset_type(r)is TaskLearningNativeDataset
    assert semantic_recipe(r)==semantic_recipe(v.recipe(v.DATA,v.STATS,endpoint=2))


@pytest.mark.parametrize('key,value',[
    ('execution_site','HOPPER_SINGLE_GPU'),('execution_site','CORELAB_4080S'),
    ('initial_package',dict(path='/old/FIT4',sha256='b'*64)),('WBC_changes',True),
    ('physics',True),('resume',dict(path='/old/SFT2000',sha256='b'*64)),
    ('GR00T_updates',500),('start_step',2000)])
def test_cannot_alias_old_fit4_full_continuation_or_unsafe_site(key,value):
    with pytest.raises(PermissionError):p.validate(dict(spec(),**{key:value}))


@pytest.mark.parametrize('endpoint',[0,1,200,500,2000])
def test_no_automatic500_or_old_task_data(endpoint):
    with pytest.raises(PermissionError):v.recipe(v.DATA,v.STATS,endpoint=endpoint)
    with pytest.raises(PermissionError):v.recipe('a'*64,v.STATS,endpoint=4)


def test_actual_batch_and_clipped_gradient_disagreement_rejects(tmp_path):
    one=tmp_path/'one';two=tmp_path/'two'
    for root in(one,two):
        (root/'evidence').mkdir(parents=True)
        (root/'evidence/MICROBATCHES.jsonl').write_text(''.join(json.dumps(dict(cursor=c,noise='original',native_loss=.1))+ '\n'for c in range(32)))
        (root/'GRADIENTS_ACTUAL.jsonl').write_text(json.dumps(dict(step=1,all_declared_gradients={'p':'sha'}))+'\n')
    assert v.compare_trace(one,two,start=0,stop=1)['consumed_microbatches']==32
    (two/'GRADIENTS_ACTUAL.jsonl').write_text(json.dumps(dict(step=1,all_declared_gradients={'p':'different'}))+'\n')
    with pytest.raises(PermissionError,match='CLIPPED_GRADIENTS'):v.compare_trace(one,two,start=0,stop=1)
