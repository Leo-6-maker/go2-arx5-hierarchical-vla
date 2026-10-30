"""Complete only the unwritten tail of an independently reconciled cold stage.

The registry write is verified, never repeated. The original runtime capsule
stays byte identical. No process or physics is started by this completion.
"""
from . import night_source_stage

PREFIX=r'''
import base64,gzip,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'and not os.environ.get('PBS_JOBID')
resource.setrlimit(resource.RLIMIT_CPU,(160,160));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});started=time.process_time()
raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();sha=lambda b:hashlib.sha256(b).hexdigest()
root=Path(P['root']);registry=Path(P['registry']);local_path=Path(P['LOCAL_path'])
assert root.is_absolute()and root.parent.name=='PI0294'and root.name.endswith('-night-source-four01')and root.is_dir()
for name in ('SOURCE_SCOPE_ADOPTION.json','OUTPUT_BINDING.private.json','RUN_MANIFEST.json','RUNTIME_PATHS.private.json',
 'COLD_ADOPTION_RESULT.json','control/NATIVE_LAUNCH_INTENT.json','REGISTERED_COMPLETION_ERROR.json'):
 assert not(root/name).exists(),('REGISTERED_COMPLETION_NO_REPLAY',name)
seal_bytes=(root/'SOURCE_SEAL.json').read_bytes();assert sha(seal_bytes)==P['source_seal_sha256']
seal=json.loads(seal_bytes);assert seal['source_sha']==P['source_sha'];capsule=root/'entry/code/vla'
code={n:(capsule/n).read_bytes()for n in seal['files']}
assert all(sha(code[n])==h for n,h in seal['files'].items())
sys.path[:0]=[str(capsule),str(capsule/'src')]
from scripts.pi0294.multitask import night_scope as scope
from scripts.pi0294.multitask.source_stage_bundle import plan,CARD_SHA
from scripts.pi0294.task_path_entry import registered_snapshot,ENTRY_FILES,node_local_paths
from scripts.pi0294.dec1.source_cpu_once_io import _exclusive as put,commit,reconcile
scope=scope.for_root(root)
approval=(root/'NIGHT_HUMAN_APPROVAL.json').read_bytes();scope.authority(approval,now=time.time())
pc=(root/'CLOSED_PC_PREPARATION_ACCOUNT.json').read_bytes();assert sha(pc)==P['PC_closed_account_sha256']
assert(root/'control/NIGHT_SOURCE_ACCOUNT.json').read_bytes()==pc
account=scope.validate_account(json.loads(pc))
assert account['custody']=='CLOSED_CARRIED_TO_NATIVE_OWNER_PENDING_INDEPENDENT_READBACK'and account['native_adopted']is False
assert account['native_destination']==str(root)and account['charged_upper']['attempts']==0
lease_path=root/'control/NATIVE_SOURCE_LEASE.json';lease=json.loads(lease_path.read_bytes())
assert lease['status']=='ACTIVE'and lease['run_uuid']==root.name and lease['deadline_epoch']==scope.DEADLINE
assert not(root/'STOP.json').exists()
snapshot=registry.read_bytes();assert sha(snapshot)==P['registry_after_sha256']
assert snapshot==(root/'control/CANONICAL_REGISTRY_READBACK.json').read_bytes()
check=reconcile(registry,intent_path=registry.with_name('NIGHT30-REGISTER-ONE-SOURCE-OWNER.INTENT.json'),
 receipt_path=registry.with_name('NIGHT30-REGISTER-ONE-SOURCE-OWNER.RECEIPT.json'))
assert check['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'and check['receipt_binding_valid']
registration=json.loads(registry.with_name('NIGHT30-REGISTER-ONE-SOURCE-OWNER.RECEIPT.json').read_bytes())
assert raw(registration)==(root/'control/ORIGINAL_REGISTRATION_RECEIPT.json').read_bytes()
registered_snapshot(snapshot,expected_sha256=sha(snapshot),run_root=str(root),role=P['role'])
reg=json.loads(snapshot);own=[r for r in reg['entries']if r.get('role')==P['role']or r.get('path')==str(root)]
assert len(own)==1 and own[0]['source_sha']==P['source_sha']and own[0]['run_uuid']==root.name
reg['entries'].remove(own[0]);rb=raw(reg);assert sha(rb)==registration['before_sha256']==P['registry_before_sha256']
lb=local_path.read_bytes();assert sha(lb)==P['LOCAL_sha256'];local=json.loads(lb)
assert local['pending_queries']==local['pending_updates']==local['unknown_effects']==[]
assert 'NIGHT_SOURCE30_BOUNDED_AUTONOMY'not in local
for target,key in ((local_path,'NIGHT30-ADOPT-DISTINCT-SOURCE-ALLOWANCE'),
 (root/'control/NIGHT_SOURCE_ACCOUNT.json','NIGHT30-ADOPT-ONE-NATIVE-SOURCE-ACCOUNT')):
 for f in (target.with_name(key+'.INTENT.json'),target.with_name(key+'.RECEIPT.json'),target.with_name(target.name+'.source-once-'+key)):
  assert not f.exists(),('UNWRITTEN_TAIL_REQUIRED',str(f))
for unit in P['idle_D25_units']:
 q=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','LoadState'],capture_output=True,text=True,timeout=4)
 assert dict(x.split('=',1)for x in q.stdout.splitlines()if '='in x)==dict(MainPID='0',LoadState='not-found')
q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
assert q.returncode==0 and not q.stdout.strip()
gb=gzip.decompress(base64.b64decode(P['GLOBAL_gzip_base64']));assert sha(gb)==P['observation']['GLOBAL']['sha256']
budget=scope.forecast(approval,gb,lb,observation=P['observation'],now=time.time())
assert budget==json.loads((root/'FULL14_PROOF.json').read_bytes())
disk=os.statvfs(root);required=P['total_required_Core_bytes'];assert disk.f_bavail*disk.f_frsize>=required
assets=json.loads((root/'ASSET_BINDINGS.private.json').read_bytes());assert sha(raw(assets))==P['asset_bindings_sha256']
for pin in assets['files'].values():
 b=Path(pin['path']).read_bytes();assert sha(b)==pin['sha256']and len(b)==pin['bytes']
for name,pin in ENTRY_FILES.items():assert sha((root/'entry/infrastructure'/name).read_bytes())==pin
assert sha((root/'FIXED_TASK_CARDS.json').read_bytes())==CARD_SHA['ALL_PINNED']
assert sha((root/'SOURCE_QUALIFICATION.json').read_bytes())==sha(raw(P['qualification']))
bundle=P['bundle'];assert bundle['task_kind']=='ALL_PINNED'and bundle['plan']==plan('ALL_PINNED')and bundle['total_binary_bytes']==18345876
assert set(bundle['definitions'])=={n.removeprefix('definition/')for n in assets['files']if n.startswith('definition/')}
def cas(path,before,after,key):
 return commit(path,expected_before_sha256=before,proposed_bytes=raw(after),request_id=key,
  intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),
  lock_path=path.with_name(path.name+'.owner.lock'))
put(root/'control/REGISTERED_COMPLETION_PROVENANCE.json',raw(dict(
 schema='SOURCE_REGISTERED_COLD_COMPLETION_V1',original_source_sha=P['source_sha'],
 completion_source_sha=P['completion_source_sha'],original_registry_receipt=registration,
 original_LOCAL_before_sha256=sha(lb),original_closed_PC_account_sha256=sha(pc),
 original_runtime_capsule_unchanged=True,registry_write_repeated=False,
 fresh_full14_proof_sha256=sha(raw(budget)),observed_epoch=time.time())))
'''


def program():
    """Use the reviewed stage tail, without replaying its registry prefix."""
    boundary="account.update(custody='ONE_NATIVE_CORE_SOURCE_OWNER'"
    assert night_source_stage.PROGRAM.count(boundary)==1
    tail=boundary+night_source_stage.PROGRAM.split(boundary,1)[1]
    assert "new_local['revision']"not in tail
    assert "cas(registry"not in tail and "root.mkdir"not in tail
    before="sealed_names=['NIGHT_HUMAN_APPROVAL.json'"
    assert tail.count(before)==1
    tail=tail.replace(before,"sealed_names=['control/REGISTERED_COMPLETION_PROVENANCE.json','NIGHT_HUMAN_APPROVAL.json'")
    body=PREFIX+'\n'+tail
    compile(body,'registered-cold-completion','exec')
    return body
