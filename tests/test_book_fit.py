from copy import deepcopy
import numpy as np
import pytest
from scripts.pi0294.multitask.book_fit import plan,validate_manifest,select_rows,FitSampler,validate_delta


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
