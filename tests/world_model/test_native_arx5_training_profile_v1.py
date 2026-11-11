import hashlib
import numpy as np
import pytest
from go2arx5_vla.world_model import native_arx5_training_profile_v1 as p


def test_exact_used_lr_schedule_and_final_zero():
    values=p.learning_rates()
    assert len(values)==500 and all(v>0 for v in values)
    assert values[0]==4.000000000000001e-6
    assert values[-1]==1.0935809887702154e-9
    assert values[24]==values[25]==1e-4
    assert p.lr_multiplier(500)==0
    assert hashlib.sha256((''.join(v.hex()+'\n' for v in values)).encode()).hexdigest()==p.LR_SHA256
    for bad in (-1,501,0.,True):
        with pytest.raises(ValueError):p.lr_multiplier(bad)


def test_source_sampling_supports_variable_population_without_extra_task_reweighting():
    for count in (1,13,1450,3200):
        samples=p.source_visit_indices(count)
        assert samples.shape==(16000,) and samples.dtype==np.int64
        assert set(samples[:count])==set(range(count))
        assert np.array_equal(samples,p.source_visit_indices(count))
    with pytest.raises(ValueError):p.source_visit_indices(3201)


def test_matched_target_visits_preserve_all1450_and_equal_episode_exposure_without_global_rng():
    episodes=[]
    for index in range(100):
        anchors=list(range(14 if index<50 else 15))
        episodes.append(dict(id=f'TRAIN{index:03d}',split='TRAIN',supervised_anchors=anchors,
                             optimization_anchors=(anchors*3)[:32]))
    before=np.random.get_state()
    visits=p.target_visit_indices(dict(episodes=episodes))
    assert np.array_equal(visits,p.target_visit_indices(dict(episodes=episodes)))
    assert visits.shape==(16000,) and set(visits)==set(range(1450))
    row_episode=np.repeat(np.arange(100),[14]*50+[15]*50)
    assert np.array_equal(np.bincount(row_episode[visits]),np.full(100,160))
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]


def test_stage_identity_resets_target_state_without_resetting_source_weights():
    a=p.recipe('A_TARGET','a'*64,'b'*64)
    b=p.recipe('B_TARGET','a'*64,'b'*64)
    assert a['RNG_namespace']==b['RNG_namespace']=='MATCHED_TARGET'
    assert a['fresh_optimizer_scheduler_RNG_cursor'] and b['fresh_optimizer_scheduler_RNG_cursor']
    assert a['initial_weights']=='OFFICIAL_RAW_PARENT'
    assert b['initial_weights']=='B_SOURCE500_WHITELIST_ONLY'
    assert {k:v for k,v in a.items() if k not in ('stage','initial_weights')}=={
        k:v for k,v in b.items() if k not in ('stage','initial_weights')}
    assert not a['CPU_training_activation_offload'] and a['endpoint']==500


def test_torch_scheduler_uses_all500_registered_positive_lrs():
    torch=pytest.importorskip('torch')
    x=torch.nn.Parameter(torch.ones(()))
    opt=torch.optim.AdamW([x],lr=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,p.lr_multiplier)
    observed=[]
    for i in range(500):
        observed.append(opt.param_groups[0]['lr'])
        opt.step();scheduler.step()
    assert observed==p.learning_rates()
    assert opt.param_groups[0]['lr']==0 and scheduler.last_epoch==500
