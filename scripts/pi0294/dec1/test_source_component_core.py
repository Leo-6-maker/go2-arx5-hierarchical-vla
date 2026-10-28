"""Existing core dispatch/counter rejection fixtures; never native admission."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from scripts.pi0292.durable import write
from scripts.pi0294.pv1 import core
from scripts.pi0294.pv1.scope import SOURCE_COMPONENT_ROLE
from scripts.pi0294.test_pv1_core_ledger import fixture as existing_ledger
from scripts.pi0294.dec1.test_source_single_pose_prefix import harness,prepare
from scripts.pi0294.dec1.source_component_factory import source_component_adapter
from scripts.pi0294.dec1.source_single_pose_prefix import SinglePosePrefixExecutor


def stage(tmp_path):
 root,l=existing_ledger(tmp_path)
 m=json.loads((root/'RUN_MANIFEST.json').read_text());m.update(scope=SOURCE_COMPONENT_ROLE,code=dict(vla_commit='f'*40))
 write(root/'RUN_MANIFEST.json',m);(root/'plans').mkdir();(root/'evidence').mkdir()
 write(root/'INPUT_BINDINGS.private.json',dict(asset_root='CPU_FIXTURE_NO_NATIVE_ASSETS',runtime_python='CPU_FIXTURE'))
 card=dict(id='CPU_COMPONENT',task='T1');p=root/'plans/card.json';write(p,card)
 write(root/'plans/DEC1_SOURCE_CPU_COMPONENT_PLAN.json',dict(conditions=[dict(card=card,condition_id=card['id'])]))
 return root,l,m,p,card

@pytest.mark.parametrize('fault',['missing-counter','bad-counter','bad-binding','wrong-card','relay-card'])
def test_new_role_rejects_before_native_import_model_or_episode_start(tmp_path,fault):
 root,l,m,p,card=stage(tmp_path)
 q=dict(status='PASS_ORIGINAL_VS_COUNTED_NATIVE_MODEL_STATE_XML_AND_TWO_RGB',integrations=0,model_constructor_attempts=2)
 if fault!='missing-counter':
  if fault=='bad-counter':q['integrations']=1
  write(root/'evidence/COUNTER_NATIVE_QUALIFICATION.json',q)
  write(root/'evidence/SOURCE_COMPONENT_COUNTER_BINDING.json',dict(schema='WRONG'))
 if fault=='wrong-card':card['id']='OTHER';write(p,card)
 if fault=='relay-card':
  card['relay_request_relative']='MUST_NOT_USE_FAKE_READY';write(p,card)
  write(root/'plans/DEC1_SOURCE_CPU_COMPONENT_PLAN.json',dict(conditions=[dict(card=card,condition_id=card['id'])]))
 with patch.object(core,'validate_runtime',return_value=m):
  with pytest.raises((PermissionError,FileNotFoundError)):core.run(root,p)
 v=json.loads(l.path.read_text());assert v['main_starts']==v['engine_starts']==0 and not v['reservations']


def test_component_counter_has_own_once_intent_and_cannot_borrow_legacy_session(tmp_path):
 root,l,m,p,card=stage(tmp_path)
 with patch.object(core,'validate_runtime',return_value=m):
  with pytest.raises(PermissionError,match='EXACT_SOURCE_COUNTER'):core.qualify(root,'B0_CORR')
  write(root/'evidence/SOURCE_COMPONENT_COUNTER_INTENT.json',dict(status='UNKNOWN_PRIOR_ATTEMPT'))
  with pytest.raises(FileExistsError,match='ALREADY_ATTEMPTED'):core.qualify(root,'DEC1_SOURCE_COMPONENT_COUNTER')
 assert json.loads(l.path.read_text())['engine_starts']==0

@pytest.mark.parametrize('mode',['I_TIME','I_PROGRESS'])
def test_native_source_profile_uses_original_point_four_limit_and_unbound_gates_reject(harness,mode):
 h=harness(mode)
 before=source_component_adapter(mode,h.executor.episode_id)
 assert before.reference.stale_timeout_s==before.config.new_chunk_max_age_s==.4
 with pytest.raises(PermissionError,match='NATIVE_PREFIX_GATES_NOT_BOUND'):before._validate(0,None)
 adapter=source_component_adapter(mode,h.executor.episode_id)
 h.executor=SinglePosePrefixExecutor(h.engine,adapter,executor_id='CPU_SINGLE_POSE',guard=lambda:None,
  reserve=h.reserve,charge=h.charge,finish=h.finish,record=h.events.append,stop=h.stop)
 proof=prepare(h);h.executor.execute_prefix(proof['nonce'])
 assert h.engine.steps==200 and h.executor.counts==dict(shadow=200,live=200) and not h.stops
