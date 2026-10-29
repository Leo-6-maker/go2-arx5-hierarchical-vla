"""Exact bounded management programs for the original D25 cold-input stage.

Payloads contain the existing registered owner, fixed package/entry pins and
original request. Run only through the existing finite ControlIO and Journal;
these strings are not standalone launch entries or experiment permission.
No model, environment installation, physics, registry or lease mutation."""

STAGE = r'''
import base64,hashlib,json,os,stat,subprocess,time
from pathlib import Path
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'and not os.environ.get('PBS_JOBID')
sha=lambda b:hashlib.sha256(b).hexdigest();raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
root=Path(P['root']);mb=(root/'RUN_MANIFEST.json').read_bytes();assert sha(mb)==P['manifest']
m=json.loads(mb);cfgb=(root/'EXECUTION_CONFIG.private.json').read_bytes();cfg=json.loads(cfgb)
assert sha(cfgb)==m['execution']['configuration_sha256']and cfg['source_sha']==P['original_source']
binding=json.loads((root/'OUTPUT_BINDING.private.json').read_bytes());lease=json.loads(Path(binding['lease']).read_bytes())
assert lease['status']=='ACTIVE'and lease['run_uuid']==root.name and time.time()<lease['deadline_epoch']
lb=Path(cfg['canonical_LOCAL_ledger']).read_bytes();assert sha(lb)==P['LOCAL_sha256'];local=json.loads(lb)
row=local['V5_CD_Core_allocations'][P['request_id']]
assert row['status']=='PRESTART_INTENT_UNKNOWN_NO_RETRY'and row['native_handle']is None
assert row['request_sha256']==P['request_sha256']and local['pending_queries']==local['pending_updates']==local['unknown_effects']==[]
units={}
for unit in P['units']:
 q=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','LoadState'],capture_output=True,text=True,timeout=3)
 props=dict(x.split('=',1)for x in q.stdout.splitlines()if '='in x);assert props==dict(MainPID='0',LoadState='not-found');units[unit]=props
assert not(root/'resident').exists()
helper=base64.b64decode(P['helper_base64'],validate=True);assert sha(helper)==P['helper_sha256']
ns={'__name__':'_pinned_cold_input_byte_verifier'};exec(compile(helper,'<published-cold-input-byte-verifier>','exec'),ns)
inventory=P['inventory'];total=ns['package_inventory'](inventory,official_commit=P['official_commit']);assert total<=1024*1024
response=subprocess.run(['git','-C',P['official_source'],'cat-file','--batch'],input=('\n'.join(x['git_blob_sha']for x in inventory.values())+'\n').encode(),capture_output=True,timeout=12)
assert response.returncode==0
members=ns['decode_git_batch'](inventory,response.stdout,official_commit=P['official_commit'])
witness=ns['byte_witness'](members,inventory,official_commit=P['official_commit'])
entry={}
for name,pin in P['entry_files'].items():
 b=(Path(cfg['entry']).parent/name).read_bytes();assert len(b)==pin['bytes']and sha(b)==pin['sha256'];entry[name]=b
assert set(entry)=={'enter.sh','doctor.py','ENVIRONMENTS.json'}and sum(map(len,entry.values()))<=32768
destination=root/'cold-inputs-20261030';intent=root/'D25_NODE_LOCAL_INPUTS_30.INTENT.json'
assert not destination.exists()and not destination.is_symlink()and not intent.exists()
assert all(str(destination).startswith(p+'/')for p in [str(root)])
mounts=[]
for line in Path('/proc/self/mountinfo').read_text().splitlines():
 before,after=line.split(' - ',1);mounts.append((before.split()[4],after.split()[0]))
resolved=str(root.resolve());matches=[x for x in mounts if resolved==x[0]or resolved.startswith(x[0].rstrip('/')+'/')]
assert max(matches,key=lambda x:len(x[0]))[1]in('ext4','xfs','btrfs','zfs')
def put(path,body,mode=0o400):
 path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode)
 with os.fdopen(fd,'wb')as f:f.write(body);f.flush();os.fsync(f.fileno())
put(intent,raw(dict(schema='D25_NODE_LOCAL_INPUTS_COLD_STAGE_INTENT_V1',owner_manifest_sha256=P['manifest'],request_id=P['request_id'],request_sha256=P['request_sha256'],LOCAL_before_sha256=sha(lb),publication=P['publication'],helper_sha256=sha(helper),package_file_bytes=total,package_files=len(members),deadline_epoch=lease['deadline_epoch'],automatic_retry=False,models_queries_physics=0)))
destination.mkdir(mode=0o700)
for name,body in members.items():put(destination/'official'/name,body)
for name,body in entry.items():put(destination/'entry/infrastructure'/name,body,0o500 if name=='enter.sh'else 0o400)
put(destination/'BYTE_WITNESS.json',raw(witness))
manifest_files={str((destination/'official'/n).relative_to(destination)):v for n,v in witness['files'].items()}
manifest_files.update({str(Path('entry/infrastructure')/n):dict(bytes=len(b),sha256=sha(b))for n,b in entry.items()})
result=dict(schema='D25_NODE_LOCAL_ORIGINAL_INPUTS_STAGED_NOT_ADMITTED_V1',status='ACTUAL_ORIGINAL_OFFICIAL_PACKAGE_AND_ENTRY_STAGED_COLD',owner_manifest_sha256=P['manifest'],original_request_sha256=P['request_sha256'],original_runtime_source=cfg['source_sha'],official_commit=P['official_commit'],publication=P['publication'],files=manifest_files,package_bytes=total,package_files=len(members),entry_files=len(entry),intent_sha256=sha(intent.read_bytes()),byte_witness_sha256=sha((destination/'BYTE_WITNESS.json').read_bytes()),LOCAL_unchanged_sha256=sha(Path(cfg['canonical_LOCAL_ledger']).read_bytes()),units=units,old_runtime_config_unchanged=sha((root/'EXECUTION_CONFIG.private.json').read_bytes())==sha(cfgb),models_queries_physics=0,runtime_binding_adopted=False,launch_allowed=False,observed_epoch=time.time())
assert result['LOCAL_unchanged_sha256']==P['LOCAL_sha256']and result['old_runtime_config_unchanged']
put(destination/'STAGE_RESULT.json',raw(result))
print(json.dumps(dict(status=result['status'],package_files=len(members),package_bytes=total,entry_files=len(entry),stage_result_sha256=sha(raw(result)),intent_sha256=result['intent_sha256'],models_queries_physics=0,runtime_binding_adopted=False,launch_allowed=False,observed_epoch=time.time())))
'''

READBACK = r'''
import hashlib,json,os,stat,subprocess,time
from pathlib import Path
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
sha=lambda b:hashlib.sha256(b).hexdigest();root=Path(P['root']);destination=root/'cold-inputs-20261030'
rb=(destination/'STAGE_RESULT.json').read_bytes();assert sha(rb)==P['stage_result_sha256'];r=json.loads(rb)
assert sha((root/'RUN_MANIFEST.json').read_bytes())==r['owner_manifest_sha256']
cfg=json.loads((root/'EXECUTION_CONFIG.private.json').read_bytes());assert sha(Path(cfg['canonical_LOCAL_ledger']).read_bytes())==r['LOCAL_unchanged_sha256']
assert sha((root/'D25_NODE_LOCAL_INPUTS_30.INTENT.json').read_bytes())==r['intent_sha256']
assert sha((destination/'BYTE_WITNESS.json').read_bytes())==r['byte_witness_sha256']
for name,pin in r['files'].items():
 p=destination/name;s=p.lstat();assert stat.S_ISREG(s.st_mode)and s.st_uid==2009 and not s.st_mode&0o222
 b=p.read_bytes();assert len(b)==pin['bytes']and sha(b)==pin['sha256']
units={}
for unit in r['units']:
 q=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','LoadState'],capture_output=True,text=True,timeout=3)
 props=dict(x.split('=',1)for x in q.stdout.splitlines()if '='in x);assert props==r['units'][unit];units[unit]=props
assert not(root/'resident').exists()
print(json.dumps(dict(status='INDEPENDENT_ALL83_STAGED_INPUT_BYTES_AND_ORIGINAL_IDLE_STATE_VERIFIED',stage_result_sha256=sha(rb),stage_result=r,full_byte_readbacks=len(r['files']),units=units,models_queries_physics=0,runtime_binding_adopted=False,launch_allowed=False,observed_epoch=time.time())))
'''
