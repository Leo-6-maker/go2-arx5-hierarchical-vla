"""Cold management-only programs for the original finite D25 startup.

No action on import. No new owner, scheduler or cross-site runtime route.
The original bind/admission/delegation/onceIO functions remain authoritative.
"""
COMMON=r'''
import base64,gzip,hashlib,json,lzma,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'and not os.environ.get('PBS_JOBID');sys.dont_write_bytecode=True
raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();SHA=sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
BASE=Path(P['root']);mb=(BASE/'RUN_MANIFEST.json').read_bytes();m=json.loads(mb);assert sha(mb)==P['owner_manifest_sha256']
cb=(BASE/'EXECUTION_CONFIG.private.json').read_bytes();cfg=json.loads(cb);assert sha(cb)==m['execution']['configuration_sha256']
CODE=Path(cfg['source_code']);sys.path[:0]=[str(CODE),str(CODE/'src')]
from scripts.pi0294.pv1 import mix_accounting as a,mix_core_accounting as c,mix_core_delegation as d,mix_core_identity as identity
from scripts.pi0294.dec1 import source_cpu_once_io as once
peers=read(BASE/'REGISTERED_PEERS.private.json');N=Path(peers['hopper']['root']);F=Path(peers['core']['root']);Q=N/'pbs/alloc-02';rb=(Q/'REQUEST.json').read_bytes();r=json.loads(rb);assert sha(rb)==P['request_sha256']
LOCAL=Path(cfg['canonical_LOCAL_ledger']);S=BASE/'root-management-finite-startup-01'
def put(path,value,mode=0o400):
 body=value if type(value)is bytes else raw(value);path=Path(path)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode)
 with os.fdopen(fd,'wb')as f:f.write(body);f.flush();os.fsync(f.fileno())
 assert path.read_bytes()==body
def cas(path,before,after,key):
 return once.commit(path,expected_before_sha256=sha(before),proposed_bytes=after,request_id=key,intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),lock_path=path.with_suffix('.owner.lock'))
def receipts(path,key):
 return {kind:base64.b64encode(path.with_name(key+'.'+suffix+'.json').read_bytes()).decode()for kind,suffix in [('intent','INTENT'),('receipt','RECEIPT')]}
def process():
 owner=identity.current_owner(N,r,request_sha256=sha(rb))
 helper=BASE/'control-code/ORIGINAL_RESIDENT_KERNEL_HELPERS.py';body=helper.read_bytes();assert sha(body)==m['execution']['owner_code_sha256'][helper.name]
 exec(compile(body,str(helper),'exec'),globals())
 p=fresh_process(SimpleNamespace(request=r,directory=Q,request_sha=sha(rb)))
 return owner,p
'''

PRESTART=COMMON+r'''
assert not S.exists();S.mkdir(mode=0o700)
for unit in(BASE.name+'.service',r['native_unit']):
 q=subprocess.run(['systemctl','--user','show',unit,'-p','LoadState','-p','MainPID'],capture_output=True,text=True,timeout=5)
 props=dict(x.split('=',1)for x in q.stdout.splitlines()if '='in x);assert props==dict(MainPID='0',LoadState='not-found')
assert 0<=time.time()-P['GLOBAL_observed_epoch']<=120
packed=Path(P['cache']['path']).read_bytes();assert sha(packed)==P['cache']['packed_sha256'];gb=lzma.decompress(packed);assert sha(gb)==P['GLOBAL_sha256']==P['cache']['sha256']
lb=LOCAL.read_bytes();assert sha(lb)==P['LOCAL_sha256'];g,l=map(json.loads,(gb,lb));c.paired(g,l);row=g[c.FIELD][r['request_id']]
assert row['status']==c.STATES[0]and row['native_handle']is None and c.progress(row)==(0,0,0)
assert (BASE/'ASSEMBLY.private.json').is_file()and sha((BASE/'ASSEMBLY.private.json').read_bytes())==P['assembly_sha256']
for role,peer in [('core',F),('hopper',N)]:
 pm=(peer/'RUN_MANIFEST.json').read_bytes();assert sha(pm)==peers[role]['manifest_sha256'];pb=read(peer/'OUTPUT_BINDING.private.json');assert read(pb['lease'])['status']=='ACTIVE'
rawGPU={}
for name,args in [('devices',['--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu']),('processes',['--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory'])]:
 q=subprocess.run(['nvidia-smi',*args,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8);assert q.returncode==0;rawGPU[name]=dict(stdout=q.stdout,stderr=q.stderr,returncode=q.returncode)
from scripts.pi0294.pv1.evaluation_core_identity import parse_GPU_rows
device=parse_GPU_rows(rawGPU['devices']['stdout'],rawGPU['processes']['stdout'],GPU_uuid=r['GPU_uuid'],desktop_request=r if 'Core_desktop_coexistence_grant_sha256'in r else None)
assert device['total_bytes']>=30*1024**3 and {0,1,2,3,4,5,6,7}<=os.sched_getaffinity(0)
put(S/'GLOBAL_PRESTART.json.xz',packed);put(S/'FULL_QUEUE_AND_DEVICE_READBACK.private.json',dict(full_queue=P['full_queue'],GLOBAL_observed_epoch=P['GLOBAL_observed_epoch'],device=device,GPU_raw=rawGPU,Core_observed_epoch=time.time()))
put(BASE/'CORE_FINITE_PRESTART_CAPTURE.json',dict(GLOBAL_base64=base64.b64encode(gb).decode(),GLOBAL_sha256=sha(gb),observed_epoch=P['GLOBAL_observed_epoch'],canonical_GLOBAL_is_fresh_at_capture=True,management_connection_only=True))
print(json.dumps(dict(status='ACTUAL_ORIGINAL_D25_FINITE_PRESTART_INPUTS_FIXED',GLOBAL_sha256=sha(gb),LOCAL_sha256=sha(lb),model_permission=False,physics_permission=False,observed_epoch=time.time())))
'''

PREPARE_BIND=COMMON+r'''
owner,p=process();gb=lzma.decompress((S/'GLOBAL_PRESTART.json.xz').read_bytes());lb=LOCAL.read_bytes()
assert sha(gb)==P['GLOBAL_before_sha256']and sha(lb)==P['LOCAL_before_sha256']
g,l=map(json.loads,(gb,lb));b=c.bind(g,l,request_id=r['request_id'],process=p,now_epoch=time.time());b['proposed_global']['revision']+=1;c.checked(b['proposed_global']);c.paired(b['proposed_global'],b['proposed_LOCAL'])
vg,vl=map(raw,(b['proposed_global'],b['proposed_LOCAL']));put(S/'GLOBAL_BOUND.json.xz',lzma.compress(vg));put(S/'LOCAL_BOUND.json',vl);put(S/'ACTUAL_PREBIND_PROCESS.json',p)
out=dict(status='ACTUAL_NATIVE_D25_BIND_PROPOSAL_ONLY',row=b['proposed_global'][c.FIELD][r['request_id']],GLOBAL_after_sha256=sha(vg),LOCAL_after_sha256=sha(vl),owner_pid=owner['pid'],owner_proc_start=owner['proc_start'],observed_epoch=time.time(),model_permission=False)
put(S/'BIND_PROPOSAL.private.json',out);print(json.dumps(out))
'''

APPLY_BOUND_LOCAL=COMMON+r'''
owner,p=process();plan=read(S/'BIND_PROPOSAL.private.json');assert p['native_handle']==plan['row']['native_handle']
g=lzma.decompress((S/'GLOBAL_BOUND.json.xz').read_bytes());after=(S/'LOCAL_BOUND.json').read_bytes();before=LOCAL.read_bytes()
assert sha(g)==P['GLOBAL_after_sha256']==plan['GLOBAL_after_sha256']and sha(after)==plan['LOCAL_after_sha256']and sha(before)==P['LOCAL_before_sha256']
proof=P['GLOBAL_readback'];assert 0<=time.time()-proof['observed_epoch']<=120 and proof['sha256']==sha(g)
d._receipt(base64.b64decode(proof['intent']),base64.b64decode(proof['receipt']),before=P['GLOBAL_before_sha256'],after=sha(g))
c.paired(json.loads(g),json.loads(after));cas(LOCAL,before,after,P['key'])
print(json.dumps(dict(status='ACTUAL_D25_NATIVE_BIND_LOCAL_CAS',sha256=sha(after),proof=receipts(LOCAL,P['key']),observed_epoch=time.time())))
'''

ISSUE=COMMON+r'''
owner,p=process();gb=lzma.decompress((S/'GLOBAL_BOUND.json.xz').read_bytes());lb=LOCAL.read_bytes();g,l=map(json.loads,(gb,lb));c.paired(g,l)
assert sha(gb)==P['GLOBAL_sha256']and sha(lb)==P['LOCAL_sha256']and 0<=time.time()-P['GLOBAL_observed_epoch']<=120
from scripts.pi0294.pv1.evaluation_admission import resource_floor
from scripts.pi0294.pv1 import mix_core_admission as admission,successor_cd as successor
used,held=resource_floor(g,l);physical=read(F/'RESOURCE_LEDGER.json');assert physical['reservations']=={}and physical['unknown_effects']==[]
used['Core_storage_bytes']=max(used['Core_storage_bytes'],cfg['Core_storage_upper_bytes']);used['local_storage_bytes']=max(used['local_storage_bytes'],cfg['LOCAL_storage_upper_bytes'])
snap=dict(schema=admission.SNAPSHOT,task=successor.TASK,scope_grant_sha256=successor.GRANT_SHA,T_CD=r['T_CD'],deadline_epoch=r['deadline_epoch'],GLOBAL_sha256=sha(gb),LOCAL_sha256=sha(lb),observed_epoch=min(time.time(),P['GLOBAL_observed_epoch']),heartbeat_only_raw_ledger=False,used=used,reserved=held,Core_storage_upper_bytes=used['Core_storage_bytes'],LOCAL_storage_upper_bytes=used['local_storage_bytes'],actual_current_Core_pending=physical['reservations'],historical_storage_witness_sha256=cfg['storage_witness_sha256'])
inputs=dict(manifest_bytes=(N/'RUN_MANIFEST.json').read_bytes(),request_bytes=rb,global_bytes=gb,local_bytes=lb,snapshot_bytes=raw(snap),native_facts_bytes=raw(p),now_epoch=time.time())
for key,name in [('training_capture_bytes','plans/V5_CD_CLOSED_TRAINING_CAPTURE.private.json'),('population_bytes','plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'),('plan_bytes','plans/SOURCE_EVALUATION_PLAN.json'),('prior_bytes','plans/PRIOR_EVALUATION_OUTCOMES.json')]:inputs[key]=(N/name).read_bytes()
proposal=d.issue(**inputs);grant=raw(proposal['grant']);vg,vl=map(raw,(proposal['proposed_global'],proposal['proposed_LOCAL']))
put(S/'GRANT.json',grant);put(S/'GLOBAL_DELEGATED.json.xz',lzma.compress(vg));put(S/'LOCAL_DELEGATED.json',vl);put(S/'ISSUE_SNAPSHOT.json',snap)
out=dict(status='ACTUAL_NATIVE_FULL_REMAINING_ADMISSION_FINITE_PROPOSAL',grant_sha256=sha(grant),marker=d._marker(grant),GLOBAL_after_sha256=sha(vg),LOCAL_after_sha256=sha(vl),shortfalls=proposal['grant']['body']['admission']['shortfalls'],issued_epoch=proposal['grant']['body']['issued_epoch'],native_handle=p['native_handle'],owner_pid=owner['pid'],owner_proc_start=owner['proc_start'],model_permission=False,physics_permission=False)
put(S/'ISSUE_PROPOSAL.private.json',out);print(json.dumps(out))
'''

APPLY_DELEGATED_LOCAL=COMMON+r'''
owner,p=process();plan=read(S/'ISSUE_PROPOSAL.private.json');assert p['native_handle']==plan['native_handle']and 0<=time.time()-plan['issued_epoch']<=120
g=lzma.decompress((S/'GLOBAL_DELEGATED.json.xz').read_bytes());after=(S/'LOCAL_DELEGATED.json').read_bytes();before=LOCAL.read_bytes()
assert sha(g)==P['GLOBAL_after_sha256']==plan['GLOBAL_after_sha256']and sha(after)==plan['LOCAL_after_sha256']and sha(before)==P['LOCAL_before_sha256']
proof=P['GLOBAL_readback'];assert proof['sha256']==sha(g)and 0<=time.time()-proof['observed_epoch']<=120
d._receipt(base64.b64decode(proof['intent']),base64.b64decode(proof['receipt']),before=P['GLOBAL_before_sha256'],after=sha(g))
c.paired(json.loads(g),json.loads(after));cas(LOCAL,before,after,P['key'])
print(json.dumps(dict(status='ACTUAL_D25_LOCAL_FINITE_DELEGATION_CAS',sha256=sha(after),proof=receipts(LOCAL,P['key']),observed_epoch=time.time())))
'''

ACTIVATE=COMMON+r'''
owner,p=process();plan=read(S/'ISSUE_PROPOSAL.private.json');assert p['native_handle']==plan['native_handle']and (owner['pid'],owner['proc_start'])==(plan['owner_pid'],plan['owner_proc_start'])
gb=lzma.decompress((S/'GLOBAL_DELEGATED.json.xz').read_bytes());lb=LOCAL.read_bytes();grant=(S/'GRANT.json').read_bytes();proof=P['GLOBAL_readback']
assert sha(gb)==proof['sha256']==plan['GLOBAL_after_sha256']and sha(lb)==plan['LOCAL_after_sha256']and sha(grant)==plan['grant_sha256']and 0<=time.time()-proof['observed_epoch']<=120
args=dict(global_bytes=gb,local_bytes=lb,global_intent_bytes=base64.b64decode(proof['intent']),global_receipt_bytes=base64.b64decode(proof['receipt']),local_intent_bytes=LOCAL.with_name(P['LOCAL_key']+'.INTENT.json').read_bytes(),local_receipt_bytes=LOCAL.with_name(P['LOCAL_key']+'.RECEIPT.json').read_bytes())
now=time.time();state=d.activate(grant,now_epoch=now,**args)
from scripts.pi0294.pv1 import mix_core_delegation_store as store,mix_core_delegated_control as control,mix_core_delegated_ledger as ledger
bundle=ledger.activation_bundle(grant,now_epoch=now,**args)
for path,value,mode in [(N/'evidence'/store.GRANT,grant,0o400),(N/'evidence'/store.STATE,state,0o600),(N/'evidence'/control.FILE,control.initial(grant,adopted_global_bytes=gb),0o600),(N/'evidence'/ledger.ACTIVATION,bundle,0o400),(N/'NATIVE_CORE_PROCESS_BOUND.json',dict(native_handle=p['native_handle'],request_id=r['request_id'],request_sha256=sha(rb),GLOBAL_after_sha256=sha(gb),actual_process_sha256=c.digest(p),finite_delegation_grant_sha256=sha(grant)),0o400)]:put(path,value,mode)
confirmed=store.read_confirmed(N,expected_grant_sha256=sha(grant));control.read_confirmed(N,grant,adopted_global_bytes=gb);ledger.verify_activation(grant,(N/'evidence'/ledger.ACTIVATION).read_bytes(),confirmed['state'])
ready=dict(request_sha256=sha(rb),native_handle=p['native_handle'],owner_pid=owner['pid'],owner_proc_start=owner['proc_start'],grant_sha256=sha(grant),actual_dual_once_receipts_confirmed=True,observed_epoch=time.time(),model_permission=False,physics_permission=False)
put(BASE/'CORE_FINITE_ACTIVATION_READY.json',ready)
print(json.dumps(dict(status='ACTUAL_EXACT_DUAL_FINITE_ACTIVATION_READY_ORIGINAL_GATES_REMAIN',**ready)))
'''

GLOBAL=r'''
import base64,hashlib,json,os,sys,time
from pathlib import Path
assert os.getuid()==22182 and os.uname().nodename.startswith('stdct-login-')and not os.environ.get('PBS_JOBID');sys.dont_write_bytecode=True
sha=lambda b:hashlib.sha256(b).hexdigest();raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
sys.path[:0]=[P['code'],P['code']+'/src']
from scripts.pi0294.pv1 import mix_core_accounting as c,mix_core_delegation as d
from scripts.pi0294.dec1.source_cpu_once_io import commit
path=Path(P['path']);before=path.read_bytes();assert sha(before)==P['before'];v=json.loads(before)
assert 0<=time.time()-P['observed_epoch']<=120
if P['kind']=='bind':
 old=v[c.FIELD][P['request_id']];assert old['status']==c.STATES[0]and old['native_handle']is None
 v[c.FIELD][P['request_id']]=P['row']
else:
 assert P['kind']=='delegate'and P['request_id']not in v.get(d.FIELD,{})
 v.setdefault(d.FIELD,{})[P['request_id']]=P['marker']
v['revision']+=1;after=raw(v);assert sha(after)==P['after'];c.checked(v);key=P['key']
commit(path,expected_before_sha256=sha(before),proposed_bytes=after,request_id=key,intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),lock_path=path.with_suffix('.owner.lock'))
print(json.dumps(dict(sha256=sha(after),observed_epoch=time.time())))
'''

READ_GLOBAL=r'''
import base64,json,hashlib,os,time
from pathlib import Path
assert os.getuid()==22182 and os.uname().nodename.startswith('stdct-login-')and not os.environ.get('PBS_JOBID')
sha=lambda b:hashlib.sha256(b).hexdigest();path=Path(P['path']);b=path.read_bytes();assert sha(b)==P['after'];key=P['key']
ib=path.with_name(key+'.INTENT.json').read_bytes();rb=path.with_name(key+'.RECEIPT.json').read_bytes();i,r=map(json.loads,(ib,rb))
assert r['intent_sha256']==sha(ib)and r['after_sha256']==i['proposed_sha256']==sha(b)and r['before_sha256']==i['before_sha256']==P['before']and not Path(i['temporary']).exists()
print(json.dumps(dict(sha256=sha(b),intent=base64.b64encode(ib).decode(),receipt=base64.b64encode(rb).decode(),native_receipt_binding_valid=True,observed_epoch=time.time(),new_mutations=0)))
'''

PROGRAMS=dict(prestart=PRESTART,prepare_bind=PREPARE_BIND,bind_LOCAL=APPLY_BOUND_LOCAL,issue=ISSUE,delegate_LOCAL=APPLY_DELEGATED_LOCAL,activate=ACTIVATE,GLOBAL=GLOBAL,read_GLOBAL=READ_GLOBAL)
for _name,_body in PROGRAMS.items():compile(_body,'<D25-local04-root-management-'+_name+'>','exec')
