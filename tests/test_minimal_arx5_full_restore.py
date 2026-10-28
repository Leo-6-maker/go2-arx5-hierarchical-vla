"""Real CPU AdamW/scheduler/RNG/safetensors; framework calls are fixtures, not GR00T PASS."""
import json,base64,random,os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from safetensors.torch import load_file,save_file
from test_minimal_arx5_adapter_contract import fixture_pair
from test_minimal_arx5_training import recipe,data_fixture
from scripts.pi0294.generalization import minimal_resume as n,minimal_training as t,minimal_study as s
from scripts.pi0294.generalization.minimal_dataset import Sampler
from scripts.pi0294.generalization.minimal_adapter import parameter_state_fingerprint,frozen_fingerprint
from scripts.pi0294.e1.trainer import learning_rate_multiplier
from scripts.pi0294.native_dataset import sha


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

class State:
 def __init__(self,global_step=0):self.global_step=global_step
 @classmethod
 def load_from_json(cls,path):return cls(**json.loads(Path(path).read_bytes()))

class FrameworkFixture:
 def __init__(self,root,arm=0):
  _,a,b,ca,cb=fixture_pair();self.model,self.parameter_contract=(a,ca)if arm==0 else(b,cb)
  self.recipe=recipe(self.parameter_contract.arm);self.args=SimpleNamespace(output_dir=str(root),data_seed=20261019,max_steps=1024,gradient_accumulation_steps=16,per_device_train_batch_size=2,world_size=1)
  self.train_dataset=data_fixture();self.train_dataset.make_sampler=Sampler;self.sampler=Sampler(self.train_dataset);self.state=State();self.consumed=0;self.start_cursor=0;self.qualification_only=True;self.session_stop=2;self.frozen_baseline=frozen_fingerprint(self.model);self.calls=[];self.create_optimizer_and_scheduler(1024)
 def known_effects(self):return dict(optimizer_updates=2,pending_updates=[],unknown_effects=[])
 def create_optimizer_and_scheduler(self,num_training_steps):
  assert num_training_steps==1024;self.calls.append('create');self.optimizer=torch.optim.AdamW(self.parameter_contract.parameters(),lr=1e-5,weight_decay=.01);self.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda i:learning_rate_multiplier(i,1024,64))
 def _load_from_checkpoint(self,path):
  self.calls.append('model');self.model.load_state_dict(load_file(Path(path)/'model.safetensors'),strict=True)
 def _load_optimizer_and_scheduler(self,path):
  self.calls.append('optimizer_scheduler');self.optimizer.load_state_dict(torch.load(Path(path)/'optimizer.pt',weights_only=True));self.lr_scheduler.load_state_dict(torch.load(Path(path)/'scheduler.pt',weights_only=True))
 def _load_rng_state(self,path):
  self.calls.append('rng');v=torch.load(Path(path)/'rng_state.pth',weights_only=False);random.setstate(v['python']);np.random.set_state(v['numpy']);torch.random.set_rng_state(v['cpu'])

@pytest.fixture
def saved(tmp_path):
 previous=torch.get_num_threads();rng_before=(random.getstate(),np.random.get_state(),torch.random.get_rng_state())
 name=os.environ.get('ARX5_ORIGINAL_METADATA_CAPTURE')
 if not name:pytest.skip('explicit private actual original R0 metadata required')
 p=Path(name);cap=json.loads(json.loads(p.read_bytes())['stdout']);stats=base64.b64decode(cap['files']['statistics.json']['base64']);assert __import__('hashlib').sha256(stats).hexdigest()==s.R0
 torch.set_num_threads(2)
 def save(arm=0):
  trainer=FrameworkFixture(tmp_path,arm);cp=tmp_path/'checkpoint-2';cp.mkdir();random.seed(71);np.random.seed(72);torch.manual_seed(73)
  for i in range(2):
   trainer.optimizer.zero_grad(set_to_none=True);trainer.model(torch.randn(2,1,4),torch.tensor([10,10])).square().mean().backward();trainer.optimizer.step();trainer.lr_scheduler.step()
  trainer.state.global_step=2;trainer.consumed=64
  save_file(trainer.model.state_dict(),cp/'model.safetensors');torch.save(trainer.optimizer.state_dict(),cp/'optimizer.pt');torch.save(trainer.lr_scheduler.state_dict(),cp/'scheduler.pt');torch.save(dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.random.get_rng_state()),cp/'rng_state.pth')
  (cp/'processor').mkdir();write(cp/'processor/fixture.json',dict(CPU_ONLY=True));write(cp/'config.json',dict(CPU_ONLY=True));(cp/'statistics.json').write_bytes(stats);write(cp/'trainer_state.json',dict(global_step=2));write(cp/'SAMPLER_STATE.json',dict(recipe=trainer.recipe,sampler=trainer.sampler.state_dict(64),manifest_sha256=trainer.train_dataset.manifest_sha,statistics_sha256=s.R0));write(cp/'ADAPTER_PARAMETER_CONTRACT.json',trainer.parameter_contract.describe());write(cp/'FROZEN_PARAMETER_IDENTITY.json',trainer.frozen_baseline);write(cp/'MODEL_PARAMETER_STATE_IDENTITY.json',parameter_state_fingerprint(trainer.model));write(cp/n.FILENAME,n.save_identity(trainer,cp));files={str(p.relative_to(cp)):sha(p)for p in cp.rglob('*')if p.is_file()};write(cp/'CHECKPOINT_COMPLETE.json',dict(global_step=2,qualification_only=True,full_resume=True,scheduler_horizon=1024,training_identity=trainer.recipe['arm'],files=files));return trainer,cp
 yield save
 random.setstate(rng_before[0]);np.random.set_state(rng_before[1]);torch.random.set_rng_state(rng_before[2]);torch.set_num_threads(previous)

@pytest.mark.parametrize('arm',[0,1])
def test_real_cpu_full_state_and_fixed_output_restored_without_update(saved,arm):
 original,cp=saved(arm);x=torch.arange(8,dtype=torch.float32).reshape(2,1,4)/10;ids=torch.tensor([10,10]);before=original.model(x,ids).detach();fresh=FrameworkFixture(cp.parent,arm);calls=[];out=n.restore_qualification(fresh,cp,guard=lambda:calls.append('guard'))
 assert torch.equal(fresh.model(x,ids),before)
 assert fresh.calls[-4:]==['create','model','optimizer_scheduler','rng']
 assert fresh.sampler.state_dict(64)==original.sampler.state_dict(64)
 assert next(iter(fresh.sampler))==list(original.sampler)[64]
 assert n.fingerprint(n.current_rng(n.saved_rng(cp)))==n.fingerprint(n.saved_rng(cp))
 assert out['new_optimizer_updates']==out['new_queries']==0 and out['native_model_qualification_complete']is False
 with pytest.raises(ValueError,match='ONLY_FRESH_OWN'):n.restore_qualification(fresh,cp,guard=lambda:None)

@pytest.mark.parametrize('kind',['optimizer','optimizer-order','scheduler','rng','model','pending','horizon','formal','wrong-arm','different-cursor'])
def test_permissive_or_wrong_restore_is_caught(saved,kind):
 original,cp=saved();fresh=FrameworkFixture(cp.parent)
 if kind=='optimizer':
  actual=fresh._load_optimizer_and_scheduler
  def wrong(path):actual(path);next(iter(fresh.optimizer.state.values()))['exp_avg'].add_(1)
  fresh._load_optimizer_and_scheduler=wrong
 elif kind=='optimizer-order':
  actual=fresh.create_optimizer_and_scheduler
  def wrong(num_training_steps):
   actual(num_training_steps);p=fresh.optimizer.param_groups[0]['params'];p[0],p[2]=p[2],p[0]
  fresh.create_optimizer_and_scheduler=wrong
 elif kind=='scheduler':
  actual=fresh._load_optimizer_and_scheduler
  def wrong(path):actual(path);fresh.lr_scheduler.last_epoch=0
  fresh._load_optimizer_and_scheduler=wrong
 elif kind=='rng':fresh._load_rng_state=lambda path:torch.manual_seed(999)
 elif kind=='model':fresh._load_from_checkpoint=lambda path:None
 elif kind=='pending':fresh.known_effects=lambda:dict(optimizer_updates=2,pending_updates=[1],unknown_effects=[])
 elif kind=='horizon':fresh.args.max_steps=512
 elif kind=='formal':fresh.qualification_only=False;fresh.session_stop=512
 elif kind=='wrong-arm':fresh=FrameworkFixture(cp.parent,1)
 else:
  path=cp/'SAMPLER_STATE.json';v=json.loads(path.read_bytes());v['sampler']['cursor']=32;write(path,v);complete=json.loads((cp/'CHECKPOINT_COMPLETE.json').read_bytes());complete['files']['SAMPLER_STATE.json']=sha(path);write(cp/'CHECKPOINT_COMPLETE.json',complete)
 with pytest.raises((ValueError,RuntimeError)):n.restore_qualification(fresh,cp,guard=lambda:None)


def test_member_tamper_rejected_before_framework(saved,tmp_path):
 original,cp=saved();fresh=FrameworkFixture(cp.parent);(cp/'scheduler.pt').write_bytes(b'bad')
 with pytest.raises(ValueError,match='COMPLETE_OWN'):n.restore_qualification(fresh,cp,guard=lambda:None)
 assert fresh.calls==['create']


def test_device_dtype_and_nested_optimizer_changes_have_distinct_identities():
 a=dict(state={0:dict(step=torch.tensor(2.),exp_avg=torch.zeros(2,dtype=torch.float32))},param_groups=[dict(lr=1e-5,params=[0])]);b=deepcopy(a);b['state'][0]['exp_avg']=b['state'][0]['exp_avg'].double();assert n.fingerprint(a)!=n.fingerprint(b)
