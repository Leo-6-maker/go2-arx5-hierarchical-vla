import numpy as np
import pytest
from go2arx5_vla.sla_adapter.full_h40_v6 import acceleration_samples,loss,aux_slots


def test_all40_accelerations_include_heads_and_interior_and_rotation_invariance():
    rng=np.random.default_rng(32);p=rng.normal(size=(4,40,3))*.001
    v=np.diff(p,axis=1)/.05
    expected=np.linalg.norm(np.diff(np.concatenate([np.zeros((4,1,3)),v,np.zeros((4,1,3))],axis=1),axis=1)/.05,axis=-1)
    np.testing.assert_allclose(acceleration_samples(p,xp=np),expected,rtol=1e-15)
    q,_=np.linalg.qr(rng.normal(size=(3,3)));rotated=p@q+np.array([.1,-.4,.2])
    np.testing.assert_allclose(acceleration_samples(rotated,xp=np),expected,atol=2e-13)
    assert aux_slots(0)==(0,1,2,3) and aux_slots(7)==(28,29,30,31) and aux_slots(8)==aux_slots(0)


def test_loss_sees_internal_spike_even_when_both_endpoints_are_stationary():
    a=np.zeros((1,40,32));a[0,20,0]=.01
    value,detail=loss(a,np.zeros(7),np.ones(7),xp=np)
    assert detail[0]==detail[1]==0 and detail[2]>.25 and value>0
    assert loss(np.zeros((1,40,32)),np.zeros(7),np.ones(7),xp=np)[0]==0


def test_static_ten_step_sampler_matches_original_and_differentiates_every_step():
    jax=pytest.importorskip('jax');jnp=pytest.importorskip('jax.numpy');nnx=pytest.importorskip('flax.nnx')
    from go2arx5_vla.sla_adapter.full_h40_v6 import make_sampler
    from scripts.sla.close_goal_g1_v1 import make_sampler as original
    class Toy(nnx.Module):
        def __init__(self):self.gain=nnx.Param(jnp.asarray(.2,jnp.float32))
    graph,active,frozen=nnx.split(Toy(),nnx.Param,...)
    fields=dict(prefix_field=lambda m,o:(o,),cache_field=lambda m,c,x,t:m.gain.value*x+c)
    obs=jnp.asarray(0.,jnp.float32);noise=jnp.ones((1,40,32),jnp.float32)
    static=make_sampler(graph,fields,rematerialize=True);old=original(graph,fields,cached=True,precision=None)
    np.testing.assert_array_equal(np.asarray(static(active,frozen,obs,noise)),np.asarray(old(active,frozen,obs,noise)[0]))
    gradient=jax.grad(lambda a:jnp.mean(static(a,frozen,obs,noise)))(active)
    actual=float(jax.tree.leaves(gradient)[0]);expected=-(1-.1*.2)**9
    assert actual==pytest.approx(expected,rel=1e-5)
    zero=jnp.zeros((1,40,32));derivative=jax.grad(lambda x:loss(x,jnp.zeros(7),jnp.ones(7),xp=jnp)[0])(zero)
    assert np.isfinite(np.asarray(derivative)).all() and not np.asarray(derivative).any()
