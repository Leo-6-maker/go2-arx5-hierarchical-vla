"""Real CPU optimization on explicit synthetic software fixtures, not WM evidence."""
import copy
import numpy as np
import pytest
import torch
from go2arx5_vla.world_model import fusion_program_response_data_v1 as data_contract
from go2arx5_vla.world_model import fusion_program_response_v1 as core
from go2arx5_vla.world_model.fusion_program_response_numpy_v1 import predict
from go2arx5_vla.world_model.actor_sampling import ExactSampler
from go2arx5_vla.world_model.neural_training import tree_error,rng_state


def fixture():
    rng=np.random.default_rng(367401);n=32
    x=rng.normal(size=(n,data_contract.FEATURE_DIM)).astype('float32')
    future=x[:,data_contract.FUTURE_SLICE.start:data_contract.FUTURE_SLICE.start+8]
    y=np.arange(1,17,dtype='float32')[None,:,None]*future[:,None,:]*np.array([.0001]*3+[.001]*3+[.00001]*2)
    return dict(X=x,Y=y.astype('float32'),valid=np.ones((n,16),bool),
        split=np.array(['TRAIN']*24+['DEV']*8),parent=np.array(['P%d'%(i//8)for i in range(n)]),
        run_id=np.array(['RUN%d'%(i//8)for i in range(n)]),program_atom=np.arange(n)*7)


def tensors(data):
    return [torch.as_tensor(data[k])for k in ['X','Y','valid']]


def test_normalizers_never_read_dev_outcomes_and_parent_roles_are_fixed():
    d=fixture();splits=data_contract.validate(d);a=data_contract.normalizers(d,splits['TRAIN'])
    d['X'][24:]=1e6;d['Y'][24:]=-1e9;b=data_contract.normalizers(d,splits['TRAIN'])
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    with pytest.raises(ValueError,match='ONLY_REAL_TRAIN'):data_contract.normalizers(d,splits['DEV'])
    d['parent'][24]='P0'
    with pytest.raises(ValueError,match='CROSS_SPLITS'):data_contract.validate(d)


def test_unobserved_targets_are_masked_in_loss_and_normalization():
    d=fixture();d['valid'][:,4:]=False;d['Y'][:,4:]=1e10
    ids=data_contract.validate(d)['TRAIN'];stats=data_contract.normalizers(d,ids)
    assert not stats['supported_horizons'][4:].any()
    m=core.ProgramResponse('SCALED');m.set_normalizers(stats);x,y,valid=tensors(d)
    a=m.loss(x,y,valid);y=y.clone();y[:,4:]=-1e10;b=m.loss(x,y,valid)
    assert torch.equal(a,b)
    d['valid'][0,7]=True
    with pytest.raises(ValueError,match='PREFIX_MASKS'):data_contract.validate(d)


def test_tail_hold_cannot_hide_program_error_in_phase_metrics_or_change_inputs():
    d=fixture();before=d['X'].copy();d['Y'][:]=0.
    d['record_phase']=np.array(['PROGRAM']*4+['POST_PROGRAM_HOLD']*20+['POST_QUERY_HOLD_CONTROL']*8)
    d['Y'][:4,:,0]=.01
    ids=data_contract.validate(d)['TRAIN']
    result=data_contract.metrics_by_phase(np.zeros_like(d['Y']),d,ids)
    assert result['PROGRAM']['16']['XYZ_vector_RMSE_mm']==pytest.approx(10.)
    assert result['POST_PROGRAM_HOLD']['16']['XYZ_vector_RMSE_mm']==0.
    assert result['POST_QUERY_HOLD_CONTROL']['16']['anchors']==0
    np.testing.assert_array_equal(before,d['X'])


@pytest.mark.parametrize('arm',data_contract.ARMS)
def test_optimization_exact_checkpoint_restore_and_portable_forecast(tmp_path,arm):
    torch.set_num_threads(1);torch.manual_seed(367401);np.random.seed(367401)
    d=fixture();ids=data_contract.validate(d)['TRAIN'];stats=data_contract.normalizers(d,ids)
    m=core.ProgramResponse(arm);m.set_normalizers(stats);opt=torch.optim.AdamW(m.parameters(),lr=.001)
    sampler=ExactSampler(len(ids),'SEEDED_GLOBAL_SHUFFLE',367401);x,y,valid=tensors(d)
    with torch.no_grad():np.testing.assert_array_equal(m(x).numpy(),np.broadcast_to(stats['y_mean'],(32,16,8)))
    for step in range(1,16):core.update(m,opt,x,y,valid,ids[sampler.take(8)])
    provenance=dict(purpose='SYNTHETIC_SOFTWARE_TEST_ONLY',data_sha256='a'*64,source_sha='b'*40,arm=arm)
    path=tmp_path/'checkpoint15.pt';core.save_checkpoint(path,m,opt,sampler,15,provenance)
    core.update(m,opt,x,y,valid,ids[sampler.take(8)]);end_rng=rng_state()
    clone,co,cs,c=core.load_checkpoint(path,'cpu',provenance)
    core.update(clone,co,x,y,valid,ids[cs.take(8)])
    assert tree_error(m.state_dict(),clone.state_dict())==0.
    assert tree_error(opt.state_dict(),co.state_dict())==0.
    assert tree_error(sampler.state_dict(),cs.state_dict())==0. and tree_error(end_rng,rng_state())==0.
    with torch.no_grad():expected=m(x).numpy()
    actual=predict(m.arrays(),d['X'])
    np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=0)
    changed=d['X'].copy();changed[:,data_contract.FUTURE_SLICE]+=5.
    if arm=='NO_FUTURE_ACTION':np.testing.assert_array_equal(predict(m.arrays(),changed),actual)
    else:assert np.max(np.abs(predict(m.arrays(),changed)-actual))>0.
    with pytest.raises(ValueError,match='SOURCE_DATA_OR_CONFIG'):
        core.load_checkpoint(path,'cpu',dict(provenance,data_sha256='c'*64))
