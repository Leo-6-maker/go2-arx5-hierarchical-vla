import time
import numpy as np
import pytest


def toy(tmp_path, monkeypatch, micro, bad=False):
 import jax
 import jax.numpy as jnp
 from go2arx5_vla.sla_adapter import runtime_v3 as module
 e=module.DeviceResidentEngine.__new__(module.DeviceResidentEngine)
 e.active={'w':jnp.asarray(1.)};e.frozen={};e.opt_state={'m':jnp.asarray(0.)}
 e.microbatch=micro;e.accumulation_steps=32//micro;e.rng=jax.random.PRNGKey(1)
 e.dataset=e.tokenizer=e.visit_map=e.loader=None;e.step=e.cursor=0
 e.output=tmp_path;e.spec={'deadline_epoch':time.time()+60}
 e.counts=dict(forward_backward_attempts=0,attempted_training_visits=0,completed_forward_backward=0,completed_training_visits=0,optimizer_updates=0)
 e.draw_bank=lambda *args:(jnp.zeros((32,40,32)),jnp.zeros(32))
 def batches(*args,**kwargs):
  for offset in range(0,32,micro):
   yield dict(cursors=np.arange(offset,offset+micro),indices=[0]*micro,state_token_dropped=[False]*micro,prepare_wall_s=0.,host_wait_s=0.)
 monkeypatch.setattr(module,'prepared_batches',batches)
 monkeypatch.setattr(module,'device_batch',lambda host:(jnp.ones(micro)*(jnp.inf if bad else 2.),jnp.ones(micro)))
 def objective(active,frozen,obs,actions,noise,t):
  loss=jnp.mean((active['w']*obs-actions)**2)
  return loss,(loss,loss*0)
 e.grad=jax.jit(jax.value_and_grad(objective,has_aux=True))
 e.accumulate=jax.jit(lambda a,g:jax.tree.map(lambda x,y:x+y/e.accumulation_steps,a,g))
 e.add_metrics=jax.jit(lambda a,v:a+jnp.stack(v)/e.accumulation_steps)
 e.summarize=jax.jit(lambda m,g:jnp.concatenate([m,jnp.reshape(jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(g))),(1,))]))
 def apply(active,opt,grad):return jax.tree.map(lambda x,g:x-.1*g,active,grad),opt,jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(grad)))
 e.apply=jax.jit(apply)
 return e


@pytest.mark.parametrize('micro',[1,8,16])
def test_commits_same_effective32_with_one_terminal_metric_read(tmp_path,monkeypatch,micro):
 e=toy(tmp_path,monkeypatch,micro);row=e.update()
 assert e.cursor==32 and e.step==1 and e.counts['optimizer_updates']==1
 assert e.counts['completed_training_visits']==32
 assert e.counts['update_metric_D2H_calls']==1 and e.counts['update_metric_D2H_bytes']==16
 assert row['mean_FM32_loss']==1 and row['grad_norm_before_clip']==4
 np.testing.assert_allclose(np.asarray(e.active['w']),.6,rtol=1e-6)


def test_nonfinite_is_rejected_before_optimizer_or_cursor_change(tmp_path,monkeypatch):
 e=toy(tmp_path,monkeypatch,8,bad=True)
 with pytest.raises(ValueError,match='NO_UPDATE'):e.update()
 assert e.step==e.cursor==e.counts['optimizer_updates']==0
 assert float(e.active['w'])==1
