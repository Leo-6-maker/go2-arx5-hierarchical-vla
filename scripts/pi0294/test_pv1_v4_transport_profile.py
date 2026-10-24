"""Typed physical transport identity, no models, native job or CPU fixtures GO."""
from copy import deepcopy
import re
import pytest
from scripts.pi0294.pv1 import evaluation_platform as e,evaluation_stage as s,evaluation_scope as scope


def core():
 return dict(evaluation_peer_platform=e.VERSION,scope='PI0294_PV1_CORE_POLICY_EVALUATION_V1',execution=dict(profile='4080-simulation',sealed_inputs={s.PLAN:s.PLAN_SHA,s.CARD:s.CARD_SHA,'plans/V4_S2_GLOBAL_RESOURCE_AUTHORIZATION.json':scope.GRANT_SHA}))

def test_Vanda_file_transport_keeps_Core_profile_and_legacy_H200_identity():
 gpu=dict(platform_evaluation_version=e.VERSION,scope=e.ROLE,execution=dict(profile=e.PROFILE))
 assert e.file_RPC_profile(gpu,'hopper')==e.PROFILE and e.file_RPC_profile(core(),'core')=='4080-simulation'
 for m,role in [(gpu,'hopper'),(core(),'core')]:
  pattern=e.fixture_job_pattern(m,role)
  assert re.fullmatch(pattern,'1482000.stdct-mgmt-02') and not re.fullmatch(pattern,'1482000.hopper-m-02')
 assert e.file_RPC_profile({},'hopper')=='hopper-neural'
 assert re.fullmatch(e.fixture_job_pattern({},'hopper'),'1482000.hopper-m-02')
 assert not re.fullmatch(e.fixture_job_pattern({},'core'),'1482000.stdct-mgmt-02')

@pytest.mark.parametrize('field',['version','role','profile','plan','card','grant'])
def test_Core_cannot_infer_Vanda_fixture_permission_from_job_suffix(field):
 m=core()
 if field=='version':m['evaluation_peer_platform']='TRAINING'
 if field=='role':m['scope']='DEC1_SOURCE_CPU_COMPONENT_COMMISSIONING_V1'
 if field=='profile':m['execution']['profile']='4080-cpu-data'
 if field in ('plan','card','grant'):
  key={'plan':s.PLAN,'card':s.CARD,'grant':'plans/V4_S2_GLOBAL_RESOURCE_AUTHORIZATION.json'}[field]
  m['execution']['sealed_inputs'][key]='0'*64
 with pytest.raises(PermissionError):e.fixture_job_pattern(m,'core')

@pytest.mark.parametrize('profile',['hopper-neural','vanda-neural-training','4080-simulation'])
def test_partial_GPU_evaluation_marker_never_falls_back_to_legacy(profile):
 m=dict(platform_evaluation_version=e.VERSION,scope=e.ROLE,execution=dict(profile=profile))
 with pytest.raises(PermissionError):e.file_RPC_profile(m,'hopper')

def test_typed_DEV_owner_publication_matches_original_neural_reader(tmp_path):
 from scripts.pi0294.dec1.owner_heartbeat_publish import publication_path
 from scripts.pi0294.dec1.test_a40_training_runtime import put
 assert publication_path(tmp_path)==tmp_path/'control/OWNER_HEARTBEAT.json'
 put(tmp_path/'RUN_MANIFEST.json',dict(platform_evaluation_version=e.VERSION,scope=e.ROLE,execution=dict(profile=e.PROFILE)))
 assert publication_path(tmp_path)==tmp_path/'OWNER_HEARTBEAT.json'
