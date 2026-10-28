"""Production startup, captured C11 close, and immutable owner-only contracts.

Endpoint native admission/ledger IO are fixtures in the CPU child. The actual
production serve branch, registry/visibility/control/mode/frame/EOF code runs.
No model, query, constructor, scheduler or physics is used by these tests.
"""
import base64,gzip,json,os,subprocess,sys
from pathlib import Path
from copy import deepcopy
import pytest
from scripts.pi0294.pv1 import mix_evaluation as evaluation,control_heartbeat as heartbeat
from scripts.pi0294.pv1 import mix_core_caps_continuation as cont,mix_core_accounting as c,mix_accounting as a,mix_core_premodel_repair as repair
from scripts.pi0294 import relay_file_rpc as rpc


def test_explicit_core_selector_preserves_legacy_and_rejects_mixed_platform():
    legacy=evaluation.stage('C50');new=evaluation.stage('C50',original_core_control=True)
    assert 'owner_heartbeat_policy' not in legacy and 'NFS_lookup_cache_assistance' not in legacy
    assert new['production_RPC_control']==evaluation.CORE_CONTROL
    assert new['owner_heartbeat_policy']==heartbeat.POLICY
    assert {k:v for k,v in new.items()if k not in ('production_RPC_control','owner_heartbeat_policy','NFS_lookup_cache_assistance')}==legacy
    for kwargs in (dict(original_core_control=1),dict(colocated=True,original_core_control=True)):
        with pytest.raises(PermissionError):evaluation.stage('C50',**kwargs)
    with pytest.raises(PermissionError):rpc.production_contract('/not-read','core','PV1-PREMODEL-x')

@pytest.fixture
def actual():
    p=os.environ.get('ARX5_CORE_PRODUCTION_CONTROL_CAPTURE')
    if not p:pytest.skip('explicit real original-owner closed C11 capture required')
    p=json.loads(Path(p).read_bytes())['value']
    return p,gzip.decompress(base64.b64decode(p['GLOBAL_gzip_base64'])),gzip.decompress(base64.b64decode(p['LOCAL_gzip_base64']))

def proposal(actual):
    p,gb,lb=actual
    return cont.propose_production_control(gb,lb,closed_readback=p,new_request_id='12345678-1234-4234-8234-123456789abc',new_nonce='12345678-1234-4234-8234-123456789abd',now_epoch=p['observed_epoch']+.1)

def test_real_closed_costs_two_spent_constructors_and_all_future_preserved(actual):
    p,gb,lb=actual;g=json.loads(gb);v=proposal(actual);new=v['proposed_global'];m=cont.validate_production_control(new)
    assert new[c.FIELD]==g[c.FIELD]and new['used']==g['used']and new['PBS_submissions']==g['PBS_submissions']
    assert m['engineering_replacements_used_before']==g[a.PACKAGE]['engineering_replacements_used'] and m['engineering_replacements_used_after']==g[a.PACKAGE]['engineering_replacements_used']+1
    assert repair.allocation(new,'C50')['episode_starts']==24 and repair.allocation(new,'C50')['engine_starts']==2106
    assert p['effects']['query_attempts']==0 and p['effects']['model_load_attempts']==1
    assert not v['full_future_proof']['cap_changes_required'];c.paired(new,v['proposed_LOCAL'])

@pytest.mark.parametrize('change',['query','integration','log','lease','pool','stale'])
def test_cannot_restore_valid_behavior_or_hide_unknown_or_reuse_exhausted_pool(actual,change):
    p,gb,lb=deepcopy(actual)
    if change=='query':p['effects']['query_attempts']=1
    elif change=='integration':p['physical']['live_and_recovery_integrations']=1
    elif change=='log':p['engineering']['RPC_log']['base64']=base64.b64encode(b'not saved trace').decode()
    elif change=='lease':p['leases']['control']['status']='ACTIVE'
    elif change=='stale':p['observed_epoch']-=121
    else:
        g=json.loads(gb);l=json.loads(lb)
        for x in(g,l):x[a.PACKAGE]['engineering_replacements_used']=4
        gb=a.raw(g);lb=a.raw(l);p.update(GLOBAL_sha256=a.sha(gb),LOCAL_sha256=a.sha(lb))
    with pytest.raises((PermissionError,KeyError)):
        cont.propose_production_control(gb,lb,closed_readback=p,new_request_id='12345678-1234-4234-8234-123456789abc',new_nonce='12345678-1234-4234-8234-123456789abd',now_epoch=actual[0]['observed_epoch']+.1)

@pytest.mark.parametrize('role',['core','hopper'])
@pytest.mark.parametrize('sealed_control',[True,False])
def test_actual_saved_packet_production_serve_to_EOF_under_inherited6880(tmp_path,role,sealed_control):
    path=os.environ.get('ARX5_CORE_PRODUCTION_PACKET')
    if not path:pytest.skip('explicit real saved C11 packet required')
    row=json.loads(Path(path).read_bytes())['assembly']['rows'][role]
    root=tmp_path/'captured-C11';root.mkdir()
    def put(p,b):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    for name,b in row['files'].items():put(root/name,base64.b64decode(b))
    m=deepcopy(row['manifest']);m['limits']['rpc_cpu']=min(os.sched_getaffinity(0))
    m['code'].update(capsule_relative='capsules/fixture/code',source_seal_relative='capsules/fixture/SOURCE_SEAL.json')
    m['execution'].update(owner_heartbeat_policy=heartbeat.POLICY,production_RPC_control=evaluation.CORE_CONTROL)
    stage=evaluation.stage('C50',original_core_control=sealed_control);put(root/'PV1_STAGE_BINDING.json',a.raw(stage))
    m['execution']['sealed_inputs']['PV1_STAGE_BINDING.json']=a.sha(a.raw(stage))
    put(root/'RUN_MANIFEST.json',a.raw(m));put(root/'OUTPUT_BINDING.private.json',a.raw(dict(lease=str(root/'lease.json'))))
    put(root/'capsules/fixture/code/control.py',b'# CPU fixture source, no model\n')
    put(root/'capsules/fixture/SOURCE_SEAL.json',a.raw(dict(files=['control.py'])))
    import time
    registry=dict(run_uuid=root.name,deadline_epoch=time.time()+600,control_budget_profile=stage['control_budget_profile'],cases={})
    for r in ('core','hopper'):registry.update({r+'_source_sha':m['code']['vla_commit'],r+'_manifest_sha256':a.sha(a.raw(m))})
    put(root/'plans/C50-DEV-N294101/REGISTRY.json',a.raw(registry))
    program=r'''
import sys,json,resource
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager
from scripts.pi0294 import relay_file_rpc as rpc,relay_endpoint
from scripts.pi0294.pv1 import core_ledger,evaluation_core_ledger
root=Path(sys.argv[1]);role=sys.argv[2];m=json.loads((root/'RUN_MANIFEST.json').read_bytes())
class CPULedger:
 def __init__(self,r):self.role=role;self.value={}
 @contextmanager
 def transaction(self):yield self.value
resource.setrlimit(resource.RLIMIT_CPU,(6880,6880))
with patch.object(relay_endpoint,'validate',return_value=m),patch.object(core_ledger,'Ledger',CPULedger),patch.object(evaluation_core_ledger,'Ledger',CPULedger):
 rpc.serve(root,role,'C50-DEV-N294101')
p=root/'evidence/file-rpc-sessions'/('file-rpc-'+role+'-C50-DEV-N294101')/'TERMINAL.json';v=json.loads(p.read_bytes());assert v['operations']==0 and v['new_model_generations_by_this_process']==v['new_physical_integrations']==0;assert resource.getrlimit(resource.RLIMIT_CPU)==(6880,6880)
print(json.dumps(dict(status=v['status'],operations=v['operations'],CPU_limits=resource.getrlimit(resource.RLIMIT_CPU))))
'''
    q=subprocess.run([sys.executable,'-c',program,str(root),role],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=15)
    if sealed_control:
        assert q.returncode==0,q.stderr
        assert json.loads(q.stdout)['status']=='SESSION_EOF_OR_DEADLINE'
    else:
        assert q.returncode!=0 and 'PV1_PRESEALED_DIRECTORY_ASSISTANCE_REQUIRED' in q.stderr
        assert not list(root.glob('capsules/**/STDIO_MANIFEST*')) and not (root/'aux_worker_processes').exists()

@pytest.mark.parametrize('role',['core','hopper'])
def test_core_successor_owner_only_requires_real_scope_grant(tmp_path,role):
    from scripts.pi0294.pv1 import mix_core_identity as cd
    from scripts.pi0294.pv1.successor_core_context import EVAL_ROLE
    stage=evaluation.stage('C50',original_core_control=True)
    (tmp_path/'PV1_STAGE_BINDING.json').write_bytes(a.raw(stage))
    m=dict(scope=cd.c.ROLE if role=='hopper'else EVAL_ROLE,owner='EXISTING_LOCAL_PC',authority=dict(protocol_ref=cd.s.TASK_SOURCE,scope_grant_sha256=cd.s.GRANT_SHA),execution=dict(owner_heartbeat_policy=heartbeat.POLICY))
    assert heartbeat.owner_only(tmp_path,m)
    m['authority']['scope_grant_sha256']='f'*64
    with pytest.raises(PermissionError):heartbeat.owner_only(tmp_path,m)
