"""Cold explicit-identity contracts; no native optimizer or restore claims."""
from copy import deepcopy
import pytest
from scripts.pi0294.generalization import amd_learning as g,amd_bank as b,amd_dataset as d,amd_pilot_scope as scope
from scripts.pi0294.pv1 import mix1024 as numerical
from scripts.pi0294.test_pv1_recipe import old_recipe
from scripts.pi0294.test_gen1_amd_bank import fixture_reader

def test_frozen_numerical_recipe_and_arguments_match_C50_but_identity_is_GEN():
 old=old_recipe();v=g.recipe(old);ref=numerical.recipe(old,'C50')
 changed={'schema','route','arm','stage','sampling','initial_optimizer'}
 assert {k for k in v if v[k]!=ref[k]}==changed
 assert v['initial_parent_checkpoint_manifest_sha256']==g.PARENT and v['new_expert_fraction']==.5 and v['entire_branch_new_fraction']==.375
 assert g.trainer_arguments(v,old)==numerical.trainer_arguments(ref,old)
 with pytest.raises(PermissionError):numerical.validate(v,old)
 for key,value in [('arm','C50'),('seed',20261020),('max_steps',4096),('workers',4),('statistics_fit',True),('new_expert_fraction',.25)]:
  bad=dict(v,**{key:value})
  with pytest.raises(PermissionError):g.validate(bad,old)

def test_all_1024_intents_and_original_commits_keep_explicit_identity():
 recipe=g.recipe(old_recipe());value=dict(optimizer_updates=0,pending_updates=[],unknown_effects=[],training_runs={})
 for step in range(1024):
  pending=g.begin_update(value,local_step=step,request_id='fixture-request',recipe_value=recipe)
  assert value['pending_updates']==[]and pending['pending_updates'][0]['arm']==b.IDENTITY
  with pytest.raises(PermissionError):g.begin_update(pending,local_step=step,request_id='fixture-request',recipe_value=recipe)
  with pytest.raises(PermissionError):g.commit_update(pending,local_step_after=step+1,request_id='foreign')
  value=g.commit_update(pending,local_step_after=step+1,request_id='fixture-request')
 assert value['optimizer_updates']==1024 and set(value['training_runs'])=={b.IDENTITY}
 with pytest.raises(PermissionError):g.begin_update(value,local_step=1024,request_id='fixture-request',recipe_value=recipe)
 value['unknown_effects']=['unknown']
 with pytest.raises(PermissionError):g.begin_update(value,local_step=1023,request_id='fixture-request',recipe_value=recipe)

def test_exact_full_state_resume_rejects_changed_members_recipe_or_cursor():
 reader=fixture_reader();sampler=d.Sampler(reader);recipe=g.recipe(old_recipe());step=512
 files={name:'fixture-only-not-native'for name in ['config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','SAMPLER_STATE.json','processor/config.json','model.safetensors']};files['statistics.json']=scope.a.STATS
 complete=dict(global_step=step,files=files,full_resume=True);saved=dict(recipe=recipe,sampler=sampler.state_dict(step*32),manifest_sha256=reader.manifest_sha,statistics_sha256=scope.a.STATS)
 assert g.resume_state(saved,complete,dict(global_step=step),recipe,sampler,files)==512
 for member in ('optimizer.pt','scheduler.pt','rng_state.pth','statistics.json'):
  changed=dict(files);changed[member]='different'
  with pytest.raises(PermissionError):g.resume_state(saved,complete,dict(global_step=step),recipe,sampler,changed)
 changed=deepcopy(saved);changed['sampler']['cursor']-=32
 with pytest.raises(PermissionError):g.resume_state(changed,complete,dict(global_step=step),recipe,sampler,files)
 with pytest.raises(PermissionError):g.resume_state(saved,dict(complete,full_resume=False),dict(global_step=step),recipe,sampler,files)
