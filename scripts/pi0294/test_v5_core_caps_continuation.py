"""Real closed predecessor capture; replacement identities remain CPU fixtures."""
import base64,gzip,json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_core_caps_continuation as n,mix_accounting as a,mix_core_accounting as c

@pytest.fixture
def case():
    root=Path(__file__).resolve().parents[2].parent/'private/minimal-adaptation-20261027';files=sorted(root.glob('CORE_FAILED05_INDEPENDENT_THREE_CLOSE_READONLY_*private.json'))
    if not files:pytest.skip('private actual three-closed predecessor capture absent')
    p=json.loads(files[-1].read_bytes())['value'];return dict(global_bytes=gzip.decompress(base64.b64decode(p['GLOBAL_gzip_base64'])),local_bytes=gzip.decompress(base64.b64decode(p['LOCAL_gzip_base64'])),closed_readback=p,new_request_id='bace25e7-df52-4b4a-92a6-f58edceca9c2',new_nonce='e3684408-a222-4948-bcf4-516755cbfefd',now_epoch=p['observed_epoch']+.1)

def test_full_actual_zero_model_costs_retained_and_future_fits(case):
    p=n.propose(**case);old=json.loads(case['global_bytes']);g=p['proposed_global'];l=p['proposed_LOCAL'];assert not p['full_future_proof']['cap_changes_required'];assert g['used']==old['used']and g['PBS_submissions']==old['PBS_submissions'];assert g['CPU_s']==old['CPU_s'];assert g['new_storage_peak_bytes']==old['new_storage_peak_bytes'];assert g[c.FIELD]==old[c.FIELD];assert g[a.PACKAGE]['completed_stages']==old[a.PACKAGE]['completed_stages'];c.paired(g,l)
    from scripts.pi0294.pv1.mix_vanda_first import budget_LOCAL
    root=Path(__file__).resolve().parents[2].parent/'private/minimal-adaptation-20261027';hist=json.loads((root/'FIRST512_HISTORICAL_LOCAL_BYTES.private.json').read_bytes());view=budget_LOCAL(g,hist);assert view['CPU_s']['CoreLab']==l['CPU_s']['CoreLab']

@pytest.mark.parametrize('kind',range(9))
def test_nonzero_unknown_live_stale_or_different_identity_rejected(case,kind):
    p=case['closed_readback']
    if kind==0:p['leases']['GPU']['status']='ACTIVE'
    if kind==1:p['effects']['query_attempts']=1
    if kind==2:p['physical']['engine_starts']=1
    if kind==3:p['native_proof']['recorded_process_terminal']=False
    if kind==4:p['LOCAL_sha256']='0'*64
    if kind==5:case['now_epoch']+=121
    if kind==6:case['new_request_id']=n.FAILED
    if kind==7:p['leases']['control']['run_uuid']='another-owner'
    if kind==8:p['effects']['unknown_query_intents']=[1]
    with pytest.raises((PermissionError,KeyError)):n.propose(**case)

def test_adoption_once_and_prior_row_immutable(case):
    p=n.propose(**case);case['global_bytes']=a.raw(p['proposed_global']);case['local_bytes']=a.raw(p['proposed_LOCAL'])
    with pytest.raises(PermissionError):n.propose(**case)

@pytest.fixture
def rpc_case():
    root=Path(__file__).resolve().parents[2].parent/'private/minimal-adaptation-20261027';files=sorted(root.glob('CORE_FAILED06_INDEPENDENT_THREE_CLOSE_READONLY_*private.json'))
    if not files:pytest.skip('private actual three-closed predecessor capture absent')
    p=json.loads(files[-1].read_bytes())['value'];return dict(global_bytes=gzip.decompress(base64.b64decode(p['GLOBAL_gzip_base64'])),local_bytes=gzip.decompress(base64.b64decode(p['LOCAL_gzip_base64'])),closed_readback=p,new_request_id='a730080e-1717-4f5a-aaab-3cda7fb335b6',new_nonce='fbd529c8-31f2-42d6-9dd8-f01aa4c2ad32',now_epoch=p['observed_epoch']+.1)

def test_rpc_full_actual_zero_model_costs_retained_and_future_fits(rpc_case):
    p=n.propose_RPC(**rpc_case);old=json.loads(rpc_case['global_bytes']);g=p['proposed_global'];l=p['proposed_LOCAL'];assert not p['full_future_proof']['cap_changes_required'];assert g['used']==old['used']and g['PBS_submissions']==old['PBS_submissions'];assert g['CPU_s']==old['CPU_s'];assert g['new_storage_peak_bytes']==old['new_storage_peak_bytes'];assert g[c.FIELD]==old[c.FIELD];assert g[a.PACKAGE]['completed_stages']==old[a.PACKAGE]['completed_stages'];c.paired(g,l)
    from scripts.pi0294.pv1.mix_vanda_first import budget_LOCAL
    root=Path(__file__).resolve().parents[2].parent/'private/minimal-adaptation-20261027';hist=json.loads((root/'FIRST512_HISTORICAL_LOCAL_BYTES.private.json').read_bytes());view=budget_LOCAL(g,hist);assert view['CPU_s']['CoreLab']==l['CPU_s']['CoreLab']

@pytest.mark.parametrize('kind',range(9))
def test_rpc_nonzero_unknown_live_stale_or_different_identity_rejected(rpc_case,kind):
    p=rpc_case['closed_readback']
    if kind==0:p['leases']['GPU']['status']='ACTIVE'
    if kind==1:p['effects']['query_attempts']=1
    if kind==2:p['physical']['engine_starts']=1
    if kind==3:p['native_proof']['recorded_process_terminal']=False
    if kind==4:p['LOCAL_sha256']='0'*64
    if kind==5:rpc_case['now_epoch']+=121
    if kind==6:rpc_case['new_request_id']=n.RPC_FAILED
    if kind==7:p['leases']['control']['run_uuid']='another-owner'
    if kind==8:p['effects']['unknown_query_intents']=[1]
    with pytest.raises((PermissionError,KeyError)):n.propose_RPC(**rpc_case)

def test_rpc_adoption_once_and_prior_row_immutable(rpc_case):
    p=n.propose_RPC(**rpc_case);rpc_case['global_bytes']=a.raw(p['proposed_global']);rpc_case['local_bytes']=a.raw(p['proposed_LOCAL'])
    with pytest.raises(PermissionError):n.propose_RPC(**rpc_case)
