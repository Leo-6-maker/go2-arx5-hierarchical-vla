from copy import deepcopy
import numpy as np
import pytest
from scripts.pi0294.multitask.book_fit import plan,full_plan,validate_manifest,select_rows,optimization_rows,FitSampler,validate_delta


def test_fit_train_population_and_every_committed_sampler_visit():
    p=plan();rows=[(dict(id=f'BOOK_SOURCE_SEED{s:03}',split='TRAIN'),a)for s in range(5)for a in range(601)]
    chosen=select_rows(rows)
    assert [(r['id'],a)for r,a in chosen]==[(r['id'],r['anchor'])for r in p['windows']]
    sampler=FitSampler(chosen);visits=list(sampler)
    assert len(visits)==8192 and all(i==sampler.index_at(c)for c,i in visits)
    assert all(sorted(i for c,i in visits[k:k+16])==list(range(16))for k in range(0,8192,16))
    resumed=FitSampler(chosen,start=4096)
    assert list(resumed)==visits[4096:]
    bad=deepcopy(rows);bad[0][0]['split']='DEV'
    with pytest.raises(PermissionError):select_rows(bad)


def test_fit_rejects_changed_science_and_undeclared_delta_tensors():
    p=plan();m=dict(checkpoint_manifest_sha256=p['checkpoint_manifest_sha256'],limits=dict(optimizer_updates=256,physics_integrations=0,generation_queries=32,model_loads=1))
    validate_manifest(m,p)
    for key,value in [('max_steps',257),('DEV_or_TEST_used',True),('shared_DiT_frozen',False)]:
        bad=deepcopy(p);bad[key]=value
        with pytest.raises(PermissionError):validate_manifest(m,bad)
    names=[f'action_head.target{i}'for i in range(14)]
    tensors={n:np.zeros((2,3),dtype=np.float32)for n in names}
    validate_delta(tensors,tensors,names)
    with pytest.raises(ValueError):validate_delta(dict(tensors,backbone=np.zeros(1)),tensors,names)
    bad=dict(tensors);bad[names[0]]=np.zeros((3,2),dtype=np.float32)
    with pytest.raises(ValueError):validate_delta(bad,tensors,names)
    bad=dict(tensors);bad[names[0]]=np.zeros((2,3),dtype=np.float64)
    with pytest.raises(ValueError):validate_delta(bad,tensors,names)


def test_full_five_episode_population_excludes_dev_and_visits_every_window():
    p=full_plan();rows=[(dict(id=f'BOOK_SOURCE_SEED{s:03}',split='TRAIN'),a)for s in range(5)for a in range(601)]
    assert optimization_rows(rows,p)==rows
    s=FitSampler(rows,fit_recipe=p);visits=list(s)
    assert len(visits)==32768 and sorted(i for c,i in visits[:3005])==list(range(3005))
    assert visits[0][1]==s.index_at(0)and visits[-1][1]==s.index_at(32767)
    assert list(FitSampler(rows,start=32000,fit_recipe=p))==visits[32000:]
    assert p['max_steps']==1024 and p['warmup_updates']==32 and not p['DEV_or_TEST_used']
    m=dict(kind='V7_BOOK5_FINITE_TARGET_LEARNING',checkpoint_manifest_sha256=p['checkpoint_manifest_sha256'],limits=dict(optimizer_updates=1024,physics_integrations=0,generation_queries=32,model_loads=1))
    validate_manifest(m,p)
    bad=deepcopy(rows);bad[-1][0]['split']='DEV'
    with pytest.raises(PermissionError):optimization_rows(bad,p)
    with pytest.raises(PermissionError):optimization_rows(rows[:-1],p)
    with pytest.raises(PermissionError):optimization_rows(rows[:-1]+rows[:1],p)
    m['kind']='V7_BOOK_TRAIN16_FIT_DIAGNOSTIC'
    with pytest.raises(PermissionError):validate_manifest(m,p)
