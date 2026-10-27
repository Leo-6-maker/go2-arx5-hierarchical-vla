"""Actual saved Core manifest identity, checked by the production wire parser."""
import base64,json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_core_assembly as assembly,mix_core_accounting as c
from scripts.pi0294.pv1.evaluation_platform import fixture_job_pattern

@pytest.fixture
def peers():
    p=Path(__file__).resolve().parents[2].parent/'private/minimal-adaptation-20261027/CORE_RETRY07_C50_PEERS_PREPARED.private.json'
    if not p.exists():pytest.skip('private actual C07 packet absent')
    return {role:json.loads(base64.b64decode(row['manifest_bytes_base64']))for role,row in json.loads(p.read_bytes())['assembly']['rows'].items()}

def test_exact_saved_wrong_legacy_platform_is_rejected_before_gpu(peers):
    m=peers['core'];assert m['evaluation_peer_platform']!=c.VERSION
    with pytest.raises(PermissionError,match='CORELAB_PEER_ROLE'):assembly.validate_peer_contract(m,'core')

def test_only_current_physical_platform_identity_repairs_saved_packet(peers):
    m=peers['core'];before=deepcopy(m);m['evaluation_peer_platform']=c.VERSION
    p=assembly.validate_peer_contract(m,'core');assert p['wire_job_pattern'].startswith('coregpu:')
    assert not p['native_job_claim']and not p['model_or_physics_permission']
    before['evaluation_peer_platform']=c.VERSION;assert before==m
    assert assembly.validate_peer_contract(peers['hopper'],'hopper')['wire_job_pattern']==p['wire_job_pattern']

@pytest.mark.parametrize('kind',range(5))
def test_wrong_role_scope_authority_or_panel_remains_rejected(peers,kind):
    m=peers['core'];m['evaluation_peer_platform']=c.VERSION
    if kind==0:m['candidate']='A_CONTROL'
    if kind==1:m['scope']='PI0294_PV1_CORE_POLICY_EVALUATION_V1'
    if kind==2:m['authority']['protocol_ref']='0'*40
    if kind==3:m['execution']['sealed_inputs'].pop('V5_CD_SCOPE_BINDING.json')
    if kind==4:m['execution']['profile']='4080-neural-evaluation'
    with pytest.raises(PermissionError):assembly.validate_peer_contract(m,'core')
