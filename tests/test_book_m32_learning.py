from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from scripts.pi0294.multitask.book_m32_learning import plan,is_plan,block_parameter_names,LimitedContract,KIND
from scripts.pi0294.multitask.book_fit import FitSampler,optimization_rows,validate_manifest
from scripts.pi0294.multitask.book_checkpoint import validate_delta_header
from scripts.pi0294.multitask.book_m32_export import COUNTS


def recipe(blocks=2,steps=512):return plan('1'*64,'2'*64,'3'*64,blocks=blocks,steps=steps)


def rows():
    names=[*[f'BOOK_SOURCE_SEED{i:03}'for i in range(15)],'BOOK_RECOVERY_TRAIN000_EXPERT_SUFFIX',
           *[f'BOOK_M32_RECOVERY{i:03}_EXPERT_SUFFIX'for i in range(5)]]
    return [(dict(id=n,split='TRAIN'),a)for n,c in zip(names,COUNTS)for a in range(c)]


def test_same_data_parent_noise_schedule_and_visits_for_a_and_b():
    a,b=recipe(0),recipe(2)
    allowed={'limited_DiT_blocks','shared_DiT_frozen'}
    assert {k for k in a if a[k]!=b[k]}==allowed
    population=rows();assert optimization_rows(population,a)==population
    # Every visit and an interrupted/resumed suffix match across both arms.
    sa=FitSampler(population,fit_recipe=a);sb=FitSampler(population,fit_recipe=b)
    for cursor in range(0,16384,13):assert sa.index_at(cursor)==sb.index_at(cursor)
    assert list(FitSampler(population,start=16320,fit_recipe=a))==list(FitSampler(population,start=16320,fit_recipe=b))
    assert sorted(sa.index_at(i)for i in range(12533))==list(range(12533))
    assert len(population)==12533 and b['student_prefix_expert_labels']==0


@pytest.mark.parametrize('alteration',[('max_steps',513),('learning_rate',2e-4),('statistics_sha256','bad'),
    ('initial_delta_checkpoint_manifest_sha256','4'*64),('DEV_or_TEST_used',True),('scheduler_horizon_steps',10)])
def test_changed_science_is_rejected(alteration):
    p=recipe();p[alteration[0]]=alteration[1];assert not is_plan(p)


def test_actual_numeric_last_blocks_no_lexical_sort_or_other_action_parameters():
    names=tuple(f'action_head.model.transformer_blocks.{i}.weight'for i in range(32))+(
        'action_head.model.proj_out_2.weight','action_head.vl_self_attention.weight','backbone.weight')
    assert block_parameter_names(names,2)==names[30:32]
    assert block_parameter_names(names,1)==names[31:32]
    with pytest.raises(ValueError):block_parameter_names(names[:31],2)
    with pytest.raises(ValueError):block_parameter_names(names,3)


def test_limited_contract_cannot_train_vlm_other_blocks_or_changed_category():
    class Parameter:
        def __init__(self,enabled=False):self.requires_grad=enabled;self.shape=(2,3)
        def requires_grad_(self,v):self.requires_grad=v;return self
        def numel(self):return 6
    target=tuple(f'category_leaf{i}.target_W'for i in range(14))
    blocks=tuple(f'action_head.model.transformer_blocks.{i}.weight'for i in range(32))
    params={n:Parameter(n in target)for n in(*target,*blocks,'backbone.weight','action_head.vlln.weight')}
    leaf=SimpleNamespace(target=10)
    model=SimpleNamespace(named_parameters=lambda:iter(params.items()),get_parameter=lambda n:params[n],
        get_submodule=lambda n:leaf,action_head=SimpleNamespace(state_dropout_prob=0))
    base=SimpleNamespace(arm='E1_ADAPTER_ONLY',target=10,trainable_names=target,model=model,
        validate=lambda:None,enforce_modes=lambda:None,
        describe=lambda:dict(trainable=[dict(name=n,shape=[2,3])for n in target],upstream_commit='pinned',module_mode='eval_with_autograd'))
    contract=LimitedContract(base,2)
    assert set(contract.trainable_names)==set(target)|set(blocks[-2:])
    assert not params['backbone.weight'].requires_grad and not params[blocks[29]].requires_grad
    params['backbone.weight'].requires_grad_(True)
    with pytest.raises(ValueError):contract.validate()
    params['backbone.weight'].requires_grad_(False);leaf.target=11
    with pytest.raises(ValueError):contract.validate()


def test_admission_population_cannot_use_dev_failed_or_unrun_labels():
    p=recipe();population=rows()
    for bad in(population[:-1],population[:-1]+population[:1],population+[(dict(id='BOOK_DEV000',split='TRAIN'),0)]):
        with pytest.raises(PermissionError):optimization_rows(bad,p)
    bad=deepcopy(population);bad[-1][0]['split']='DEV'
    with pytest.raises(PermissionError):optimization_rows(bad,p)


def test_capacity_ten_updates_is_finite_and_checkpoint_bound():
    p=recipe(2,10);assert is_plan(p)and p['TRAIN_visits']==320 and p['scheduler_horizon_steps']==512
    m=dict(kind=KIND,checkpoint_manifest_sha256=p['checkpoint_manifest_sha256'],
        delta_checkpoint_manifest_sha256=p['initial_delta_checkpoint_manifest_sha256'],
        limits=dict(optimizer_updates=10,physics_integrations=0,generation_queries=32,model_loads=1))
    validate_manifest(m,p)
    m['limits']['optimizer_updates']=512
    with pytest.raises(PermissionError):validate_manifest(m,p)
    needed=['target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json',
            'statistics.json','SAMPLER_STATE.json','FROZEN_PARENT_TENSORS.json','ADAPTER_PARAMETER_CONTRACT.json',
            'TRAIN_DEV_SEAL.json','TRAIN_POPULATION.json']
    cp=dict(schema='BOOK_M32_PARENT_BOUND_TRAINING_DELTA_V1',recipe=p,global_step=10,cursor=320,
        standalone_model=False,full_resume_with_required_parent=True,
        required_parent_checkpoint_manifest_sha256=p['checkpoint_manifest_sha256'],files={k:'pin'for k in needed})
    validate_delta_header(cp,p['checkpoint_manifest_sha256'])
    cp['cursor']=321
    with pytest.raises(PermissionError):validate_delta_header(cp,p['checkpoint_manifest_sha256'])
    with pytest.raises(ValueError):recipe(0,10)
