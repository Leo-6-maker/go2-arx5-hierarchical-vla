"""Real kernel CPU limits + captured one-model/zero-step engineering close."""
import base64,gzip,json,os,subprocess,sys
from pathlib import Path
from copy import deepcopy
import pytest
from scripts.pi0294.pv1 import mix_core_caps_continuation as cont,mix_core_accounting as c,mix_accounting as a,mix_core_premodel_repair as repair

@pytest.mark.parametrize('inherited,expected',[(6880,6880),(7200,7200),(8000,7200),(600,600)])
def test_original_RPC_child_keeps_real_kernel_stricter_limit_without_raising(inherited,expected):
    code='from scripts.pi0294.relay_file_rpc import bounded_CPU_limit;import resource,json;resource.setrlimit(resource.RLIMIT_CPU,('+str(inherited)+','+str(inherited)+'));value=bounded_CPU_limit();print(json.dumps(dict(value=value,limits=resource.getrlimit(resource.RLIMIT_CPU))))'
    q=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=10)
    assert q.returncode==0,q.stderr
    assert json.loads(q.stdout)==dict(value=expected,limits=[expected,expected])

@pytest.fixture
def actual():
    path=os.environ.get('ARX5_CORE_RPC_LIMIT_CAPTURE')
    if not path:pytest.skip('explicit original closed C10 captured bytes required')
    p=json.loads(Path(path).read_bytes())['value'];gb=gzip.decompress(base64.b64decode(p['GLOBAL_gzip_base64']));lb=gzip.decompress(base64.b64decode(p['LOCAL_gzip_base64']))
    return p,gb,lb

def proposed(actual):
    p,gb,lb=actual
    return cont.propose_rpc_limit(gb,lb,closed_readback=p,new_request_id='12345678-1234-4234-8234-123456789abc',new_nonce='12345678-1234-4234-8234-123456789abd',now_epoch=p['observed_epoch']+.1)

def test_full_actual_costs_model_constructor_retained_original_pool_debited_once(actual):
    p,gb,lb=actual;before=json.loads(gb);v=proposed(actual);g,l=v['proposed_global'],v['proposed_LOCAL'];m=cont.validate_rpc_limit(g)
    assert before[c.FIELD][cont.LIMIT_FAILED]['model_load_attempts']==1
    assert g[c.FIELD]==before[c.FIELD]and g['used']==before['used']and g['PBS_submissions']==before['PBS_submissions']
    assert p['physical']['main_starts']==p['physical']['engine_starts']==1 and p['physical']['live_and_recovery_integrations']==0
    assert g[a.PACKAGE]['engineering_replacements_used']==before[a.PACKAGE]['engineering_replacements_used']+1
    allocation=repair.allocation(g,'C50');assert allocation['episode_starts']==25 and allocation['engine_starts']==2107
    assert p['physical']['main_starts']+allocation['episode_starts']==26
    assert not v['full_future_proof']['cap_changes_required']and m['new_PBS']==0
    c.paired(g,l)

@pytest.mark.parametrize('change',['query','integrated','valid','new-model','log','lease','pool','stale'])
def test_no_valid_outcome_query_unknown_integrated_replay_or_unbounded_pool(actual,change):
    p,gb,lb=deepcopy(actual)
    if change=='query':p['effects']['query_attempts']=1
    elif change=='integrated':p['physical']['live_and_recovery_integrations']=1
    elif change=='valid':p['engineering']['adjudication']['base64']=base64.b64encode(b'{"status":"VALID_FAILURE"}').decode()
    elif change=='new-model':p['effects']['model_load_attempts']=2
    elif change=='log':p['engineering']['RPC_log']['base64']=base64.b64encode(b'not original error').decode()
    elif change=='lease':p['leases']['control']['status']='ACTIVE'
    elif change=='stale':p['observed_epoch']-=121
    else:
        g=json.loads(gb);l=json.loads(lb);g[a.PACKAGE]['engineering_replacements_used']=l[a.PACKAGE]['engineering_replacements_used']=4;gb=a.raw(g);lb=a.raw(l);p.update(GLOBAL_sha256=a.sha(gb),LOCAL_sha256=a.sha(lb))
    with pytest.raises((PermissionError,KeyError)):
        cont.propose_rpc_limit(gb,lb,closed_readback=p,new_request_id='12345678-1234-4234-8234-123456789abc',new_nonce='12345678-1234-4234-8234-123456789abd',now_epoch=actual[0]['observed_epoch']+.1)
