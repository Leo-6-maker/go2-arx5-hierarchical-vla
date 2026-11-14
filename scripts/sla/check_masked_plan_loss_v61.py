"""Finite CPU array/attention/autodiff assay; never SLA model qualification.

Uses the pinned upstream preprocessing/attention helper and an explicit small
synthetic attention network. Loads no parent/adapter/teacher/optimizer and
performs no optimization. A separate real-SLA gradient seam is still needed.
"""
import json,time
import numpy as np


def run():
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation
    from go2arx5_vla.sla_adapter import batch_v2,masked_plan_loss_v61 as masked
    assert all(d.platform=='cpu'for d in jax.devices()),'CPU_ONLY_NO_NEURAL_GPU_OWNER'
    start=time.time();cpu=time.process_time();B=2
    observation=Observation.from_dict(dict(image={k:jnp.zeros((B,224,224,3))for k in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},
        image_mask={k:jnp.ones(B,bool)for k in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},
        image_padding_mask={k:jnp.ones((B,224,224),bool)for k in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},
        state=jnp.reshape(jnp.arange(B*32,dtype=jnp.float32),(B,32))*.001,
        tokenized_prompt=jnp.zeros((B,2),jnp.int32),tokenized_prompt_mask=jnp.ones((B,2),bool)))
    captured=[]
    class Model:
        action_horizon=40
        def __init__(self,p,capture=False):
            self.p=p;self.capture=capture;self.PaliGemma=self
        def embed_prefix(self,obs):
            return jnp.repeat(obs.state[:,None,:],2,axis=1),jnp.ones((B,2),bool),jnp.zeros(2,bool)
        def embed_suffix(self,obs,x,t):
            tokens=jnp.concatenate([obs.state[:,None,:],x*self.p['in']],axis=1)
            return tokens,jnp.ones((B,41),bool),jnp.array([True,True]+[False]*39),None
        def llm(self,tokens,*,mask,positions,adarms_cond):
            if self.capture:captured.append((np.asarray(mask),np.asarray(positions)))
            x=jnp.concatenate(tokens,axis=1)
            scores=jnp.einsum('bqd,bkd->bqk',x,x)*.001
            weights=jax.nn.softmax(jnp.where(mask,scores,-1e30),axis=-1)
            result=jnp.tanh(x+jnp.einsum('bqk,bkd->bqd',weights,x)+positions[...,None]*.0001)
            return (result[:,:2],result[:,2:]),None
        def action_out_proj(self,x):return x*self.p['out']
    p={'in':jnp.array(.7),'out':jnp.array(.4)}
    actions=jnp.sin(jnp.arange(B*40*32,dtype=jnp.float32).reshape(B,40,32)*.03)*.1
    noise=jnp.cos(jnp.arange(B*40*32,dtype=jnp.float32).reshape(B,40,32)*.07)
    flow=jnp.array([.2,.7]);all_true=jnp.ones((B,40),bool)
    def original(params,targets):
        error=batch_v2.error32(Model(params),observation,targets,noise,flow)
        return jnp.mean(jnp.mean(error,axis=-1))
    def proposed(params,targets,mask):return masked.loss(Model(params),observation,targets,noise,flow,mask)[0]
    old_value,old_grad=jax.jit(jax.value_and_grad(original))(p,actions)
    new_value,new_grad=jax.jit(jax.value_and_grad(proposed))(p,actions,all_true)
    gradient_max=max(float(jnp.max(jnp.abs(old_grad[k]-new_grad[k])))for k in p)
    np.testing.assert_allclose(new_value,old_value,rtol=1e-6,atol=1e-7)
    for k in p:np.testing.assert_allclose(new_grad[k],old_grad[k],rtol=1e-6,atol=1e-7)
    mask=np.ones((B,40),bool);mask[0,7:]=False;mask[1,3:12]=False;mask[1,::3]=False
    masked.validate_host_batch(np.asarray(actions),mask)
    function=jax.jit(jax.value_and_grad(proposed))
    baseline,baseline_grad=function(p,actions,mask)
    variants=[]
    for replacement in (1e6,np.nan,-np.inf):
        changed=jnp.where(jnp.asarray(mask)[...,None],actions,replacement)
        value,grad=function(p,changed,mask)
        np.testing.assert_array_equal(value,baseline)
        for k in p:np.testing.assert_array_equal(grad[k],baseline_grad[k])
        variants.append(dict(invalid_target=str(replacement),loss_and_parameter_gradient_bit_exact=True))
    target_grad=jax.jit(jax.grad(lambda a:proposed(p,a,jnp.asarray(mask))))(actions)
    assert np.isfinite(target_grad).all()and np.count_nonzero(np.asarray(target_grad)[~mask])==0
    masked.error32(Model(p,True),observation,actions,noise,flow,mask)
    attention,positions=captured[0]
    for row in range(B):
        assert not attention[row,:,3:][:,~mask[row]].any()
        assert attention[row,:,2].any(),'STATE_TOKEN_IS_NOT_A_SUPERVISION_SLOT'
    np.testing.assert_array_equal(positions,np.tile(np.arange(43),(B,1)))
    # Deliberately unequal per-row valid lengths: the answer must be5, not a
    # global point-weighted average. Active7/padding25 keep their old means.
    error=jnp.stack([jnp.ones((40,32)),jnp.ones((40,32))*9]);unequal=np.ones((B,40),bool);unequal[0,1:]=False
    value,parts=masked.reduce32(error,unequal)
    np.testing.assert_array_equal(np.asarray([value,*parts]),[5.,5.,5.])
    pads=jnp.broadcast_to(jnp.concatenate([jnp.ones(7),jnp.ones(25)*9]),(B,40,32))
    value,parts=masked.reduce32(pads,unequal)
    np.testing.assert_array_equal(np.asarray([value,*parts]),[7.25,1.,9.])
    assert not np.isfinite(masked.reduce32(error,jnp.zeros((B,40),bool))[0])
    return dict(schema='SLA_MASKED_PLAN_SYNTHETIC_CPU_AUTODIFF_ASSAY_V6_1',passes=True,
        real_SLA_model_qualified=False,training_engine_bound=False,parent_model_loads=0,teacher_loads=0,
        optimizer_updates=0,training_visits=0,physical_requests=0,physics_integrations=0,
        all_true_loss_absolute_error=float(abs(new_value-old_value)),all_true_parameter_gradient_max_error=gradient_max,
        invalid_target_variants=variants,invalid_target_gradient_exact_zero=True,invalid_attention_keys_absent=True,
        original_state_token_and_H40_positions_preserved=True,unequal_row_normalization_pass=True,
        original_FM32_active7_padding25_pass=True,all_false_device_rejected_by_nonfinite=True,
        wall_s=time.time()-start,CPU_s=time.process_time()-cpu)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
