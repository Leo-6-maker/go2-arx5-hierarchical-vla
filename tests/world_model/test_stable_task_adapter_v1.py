"""Synthetic software fixtures only; no real training or native result here."""
import numpy as np
import pytest
from go2arx5_vla.world_model import stable_task_adapter_v1 as a


def test_pixels_and_declared_goal_affect_features_without_privileged_fields():
    rgb=np.zeros((256,256,3),np.uint8);rgb[30:40,60:70,0]=255
    x=a.features(body_rgb=rgb,wrist_rgb=rgb,actual_target_eef10=[0,0,0,1,0,0,0,1,0,1],
                 task_command='APPROACH_RED_4MM_OPEN')
    y=a.features(body_rgb=np.zeros_like(rgb),wrist_rgb=rgb,
                 actual_target_eef10=[0,0,0,1,0,0,0,1,0,1],task_command='HOLD_OPEN')
    assert x.shape==(a.INPUT_DIM,) and np.isfinite(x).all()
    assert not np.array_equal(x,y)
    np.testing.assert_array_equal(x[-2:],[0,1]);np.testing.assert_array_equal(y[-2:],[1,0])
    with pytest.raises(TypeError):
        a.features(body_rgb=rgb,wrist_rgb=rgb,actual_target_eef10=np.zeros(10),
                   task_command='HOLD_OPEN',object_pose=np.eye(4))


def test_diagnostics_and_cross_split_physical_parents_are_rejected():
    with pytest.raises(ValueError,match='DIAGNOSTIC_RELABEL'):
        a.split_rows([dict(data_role='DIAGNOSTIC',parent_id='old')])
    with pytest.raises(ValueError,match='PARENT_CROSSES'):
        a.split_rows([dict(data_role=role,parent_id='same') for role in ('TRAIN','DEV')])
    train,dev=a.split_rows([dict(data_role='TRAIN',parent_id='new1'),dict(data_role='DEV',parent_id='new2')])
    np.testing.assert_array_equal(train,[0]);np.testing.assert_array_equal(dev,[1])


def test_numpy_export_matches_torch_and_candidate_output_has_explicit_bound():
    import torch
    torch.manual_seed(110801)
    x=np.random.default_rng(1).normal(size=(4,a.INPUT_DIM)).astype(np.float32)
    mean,scale=a.normalizer(x);model=a.make_model(mean,scale)
    actual=model(torch.from_numpy(x)).detach().numpy()
    exported=a.export_arrays(model)
    np.testing.assert_allclose(a.numpy_predict(exported,x),actual,atol=1e-9,rtol=1e-6)
    assert (np.linalg.norm(actual,axis=1)<a.STEP_M).all()
    # Candidate training never needs a GR00T parameter or object-state feature.
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    previous=exported['output.weight'].copy()
    loss=model(torch.from_numpy(x)).square().mean()/a.STEP_M**2
    loss.backward();optimizer.step()
    assert not np.array_equal(previous,a.export_arrays(model)['output.weight'])


def test_checkpoint_restores_exact_next_batch_and_optimizer_continuation(tmp_path):
    import torch
    x=np.random.default_rng(1).normal(size=(5,a.INPUT_DIM)).astype(np.float32)
    mean,scale=a.normalizer(x);torch.manual_seed(110801)
    model=a.make_model(mean,scale);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    sampler=a.Sampler(len(x),110801)
    def update(m,o,indices):
        o.zero_grad();loss=m(torch.from_numpy(x[indices])).square().mean()/a.STEP_M**2
        loss.backward();o.step()
    update(model,opt,sampler.take(3))
    path=tmp_path/'candidate.pt';provenance=dict(origin='SYNTHETIC_SOFTWARE_FIXTURE_NOT_REAL_TRAIN')
    a.save_checkpoint(path,model,opt,sampler,step=1,provenance=provenance)
    batch=sampler.take(4);update(model,opt,batch);expected=a.export_arrays(model)
    loaded=a.make_model(mean,scale);optimizer=torch.optim.AdamW(loaded.parameters(),lr=.001)
    restored=a.Sampler(len(x),9)
    assert a.restore_checkpoint(path,loaded,optimizer,restored,expected_provenance=provenance)==1
    np.testing.assert_array_equal(restored.take(4),batch)
    update(loaded,optimizer,batch)
    for key,value in a.export_arrays(loaded).items():
        np.testing.assert_array_equal(value,expected[key])
