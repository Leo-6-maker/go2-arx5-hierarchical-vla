"""Finite management programs for the proved unsubmitted E2 preparation.

No action on import. Supply private paths/pins through the original paid
ControlIO/Journal. Closing this lease neither submits nor cancels a PBS job.
"""

VERIFY = r'''
import base64,gzip,hashlib,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
assert os.getuid()==22182 and os.uname().nodename.startswith('stdct-login-')and not os.environ.get('PBS_JOBID')
sha=lambda b:hashlib.sha256(b).hexdigest();raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
root=Path(P['root']);mb=(root/'RUN_MANIFEST.json').read_bytes();assert sha(mb)==P['manifest_sha256'];m=json.loads(mb)
assert m['code']['vla_commit']==P['source_sha']and root.name=='arx5-minimal-e2-only-raw-fixed512-02'
binding=json.loads((root/'OUTPUT_BINDING.private.json').read_bytes());lease_path=Path(binding['lease']);lease_bytes=lease_path.read_bytes();lease=json.loads(lease_bytes)
assert lease['run_uuid']==root.name and lease['manifest_sha256']==sha(mb)
eb=(root/'control/RESIDENT_OWNER_OS_EXIT.json').read_bytes();sb=(root/'control/resident-owner/PRESERVED_STOP.private.json').read_bytes()
assert sha(eb)==P['owner_OS_sha256']and sha(sb)==P['owner_stop_sha256']and json.loads(eb)['returncode']==1
stop=json.loads(sb);assert stop['error_type']=='FileNotFoundError'and '/__pycache__/heartbeat_recovery.cpython-39.pyc.'in stop['message']
for name,pk,sk in [('control/RESIDENT_WRAPPER_PROCESS.private.json','wrapper_pid','wrapper_proc_start_ticks'),('control/resident-owner/SERVER_OWNER_PROCESS.private.json','pid','proc_start_ticks')]:
 row=json.loads((root/name).read_bytes());p=Path('/proc',str(row[pk]),'stat');f=p.read_text().rsplit(')',1)[1].split()if p.exists()else None
 assert not(f and f[0]not in('Z','X')and int(f[19])==int(row[sk]))
for name in ['SOURCE_REQUEST_INTENT','SOURCE_GLOBAL_RESERVED','SOURCE_QSUB_INTENT','SOURCE_QSUB_REPLY','SOURCE_JOB_BOUND']:
 assert not(root/'control/resident-owner'/(name+'.private.json')).exists()
assert not(root/'A40_NATIVE_EFFECT_LEDGER.json').exists()
for name in ['PBS_READY.json','A40_PREMODEL_QUALIFICATION.json','MODEL_START_RELEASE.json','FIRST_UPDATE_RELEASE.json','PBS_PROCESS_FINAL.json','PBS_WRAPPER_FINAL.json']:
 assert not list((root/'pbs').glob('alloc-*/'+name))
gp=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';gb=gp.read_bytes();g=json.loads(gb)
assert sha(gb)==P['GLOBAL_sha256']and g['pending_updates']==g['unknown_effects']==[]
assert all(r['status']=='NATIVE_TERMINAL'for r in g['PBS_submissions'])
assert all(r.get('request_id')!=P['request_id']and r.get('run_uuid')!=root.name for r in g['PBS_submissions'])
assert g[P['prepaid_field']][P['request_id']]['run_manifest_sha256']==sha(mb)
code=root/m['code']['capsule_relative'];seal=json.loads((root/'SOURCE_SEAL.json').read_bytes())
assert sha((root/'SOURCE_SEAL.json').read_bytes())==m['code']['source_seal_sha256']
for name in ['scripts/pi0294/dec1/source_cpu_once_io.py','scripts/pi0294/generalization/minimal_remaining_physical.py']:
 assert sha((code/name).read_bytes())==seal['files'][name]
sys.path[:0]=[str(code),str(code/'src')]
'''

CLOSE = VERIFY + r'''
assert lease['status']=='ACTIVE'and sha(lease_bytes)==P['before_lease_sha256']
from scripts.pi0294.dec1.source_cpu_once_io import commit
key='E2-PRE-NATIVE30-CLOSE-UNSUBMITTED-OWN-LEASE'
closed=dict(lease,status='CLOSED',closed_epoch=time.time(),closure_reason='OWNER_OS1_BEFORE_NATIVE_REGISTRATION_OR_DISPATCH',pre_native_owner_OS_sha256=sha(eb),PBS_submitted=False)
receipt=commit(lease_path,expected_before_sha256=sha(lease_bytes),proposed_bytes=raw(closed),request_id=key,
 intent_path=lease_path.with_name(key+'.INTENT.json'),receipt_path=lease_path.with_name(key+'.RECEIPT.json'),lock_path=lease_path.with_suffix('.owner.lock'))
assert lease_path.read_bytes()==raw(closed)and gp.read_bytes()==gb
print(json.dumps(dict(status='OWN_UNSUBMITTED_PREPARATION_LEASE_CLOSED_ONCE',closed_lease_sha256=sha(raw(closed)),receipt=receipt,GLOBAL_unchanged_sha256=sha(gb),native_requests=0,new_PBS_models_queries_updates=0,observed_epoch=time.time())))
'''

READ_CLOSED = VERIFY + r'''
assert lease['status']=='CLOSED'and sha(lease_bytes)==P['closed_lease_sha256']and lease.get('PBS_submitted')is False
key='E2-PRE-NATIVE30-CLOSE-UNSUBMITTED-OWN-LEASE';ib=lease_path.with_name(key+'.INTENT.json').read_bytes();rb=lease_path.with_name(key+'.RECEIPT.json').read_bytes();intent,receipt=map(json.loads,(ib,rb))
assert receipt['intent_sha256']==sha(ib)and receipt['after_sha256']==intent['proposed_sha256']==sha(lease_bytes)
assert not Path(intent['temporary']).exists()
from scripts.pi0294.generalization.minimal_remaining_physical import prefix_inventory,identity
snapshot=prefix_inventory(root);assert 0<snapshot['allocated_bytes']<=256*1024**2
record=dict(schema='ARX5_E2_CLOSED_UNREGISTERED_PREPARATION_V1',request_id=P['request_id'],run_uuid=root.name,
 source_sha=P['source_sha'],manifest_sha256=sha(mb),owner_OS_sha256=sha(eb),owner_stop_sha256=sha(sb),
 closed_lease_sha256=sha(lease_bytes),snapshot_sha256=sha(raw(snapshot)),snapshot_identity_sha256=identity(snapshot),
 allocated_bytes=snapshot['allocated_bytes'],native_requests=0,all_recorded_actors_terminal=True,lease_status='CLOSED')
assert gp.read_bytes()==gb
print(json.dumps(dict(status='INDEPENDENT_UNSUBMITTED_OWNER_CLOSURE_AND_RETAINED_SNAPSHOT',record=record,
 snapshot_gzip_base64=base64.b64encode(gzip.compress(raw(snapshot))).decode(),closed_lease_intent_sha256=sha(ib),
 closed_lease_receipt_sha256=sha(rb),GLOBAL_sha256=sha(gb),observed_epoch=time.time(),new_writes=0)))
'''

LEDGER_CAS = r'''
import base64,gzip,hashlib,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
assert os.getuid()==22182 and not os.environ.get('PBS_JOBID')
sha=lambda b:hashlib.sha256(b).hexdigest();raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
root=Path(P['root']);mb=(root/'RUN_MANIFEST.json').read_bytes();m=json.loads(mb);assert sha(mb)==P['manifest_sha256']
assert 0<=time.time()-P['observed_epoch']<=120
gp=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';before=gp.read_bytes();assert sha(before)==P['before_GLOBAL_sha256'];g=json.loads(before)
assert g['pending_updates']==g['unknown_effects']==[]and all(r['status']=='NATIVE_TERMINAL'for r in g['PBS_submissions'])
delta=P['delta'];assert set(delta)==set(P['allowed_fields'])and delta['revision']==g['revision']+1
after=dict(g,**delta);body=raw(after);assert sha(body)==P['after_GLOBAL_sha256']
assert after['PBS_submissions']==g['PBS_submissions']and after['reserved']==g['reserved']and after['used']==g['used']
if 'CPU_s'in delta:assert delta['CPU_s']==g['CPU_s']+3600
code=root/m['code']['capsule_relative'];seal=json.loads((root/'SOURCE_SEAL.json').read_bytes());name='scripts/pi0294/dec1/source_cpu_once_io.py';assert sha((code/name).read_bytes())==seal['files'][name];sys.path[:0]=[str(code),str(code/'src')]
from scripts.pi0294.dec1.source_cpu_once_io import commit
key=P['key'];receipt=commit(gp,expected_before_sha256=sha(before),proposed_bytes=body,request_id=key,
 intent_path=gp.with_name(key+'.INTENT.json'),receipt_path=gp.with_name(key+'.RECEIPT.json'),lock_path=gp.with_suffix('.owner.lock'))
assert gp.read_bytes()==body
print(json.dumps(dict(status='ORIGINAL_GLOBAL_FINITE_COLD_BOOKKEEPING_ONCE',GLOBAL_sha256=sha(body),receipt=receipt,new_PBS_models_queries_updates=0,observed_epoch=time.time())))
'''

LEDGER_READ = r'''
import hashlib,json,os,time
from pathlib import Path
assert os.getuid()==22182 and not os.environ.get('PBS_JOBID')
sha=lambda b:hashlib.sha256(b).hexdigest();root=Path(P['root']);gp=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json'
b=gp.read_bytes();assert sha(b)==P['GLOBAL_sha256'];g=json.loads(b)
key=P['key'];ib=gp.with_name(key+'.INTENT.json').read_bytes();rb=gp.with_name(key+'.RECEIPT.json').read_bytes();i,r=map(json.loads,(ib,rb))
assert r['intent_sha256']==sha(ib)and r['after_sha256']==i['proposed_sha256']==sha(b)and not Path(i['temporary']).exists()
assert g['pending_updates']==g['unknown_effects']==[]and all(x['status']=='NATIVE_TERMINAL'for x in g['PBS_submissions'])
binding=json.loads((root/'OUTPUT_BINDING.private.json').read_bytes());lease=Path(binding['lease']).read_bytes()
assert sha(lease)==P['closed_lease_sha256']and json.loads(lease)['status']=='CLOSED'
if P.get('new_root'):
 p=Path(P['new_root']);assert p.parent==root.parent and p.name=='arx5-minimal-e2-only-raw-fixed512-02-cold01'and not p.exists()and not p.is_symlink()
cfg=json.loads((root/'MIX_OWNER_CONFIG.private.json').read_bytes())
print(json.dumps(dict(status='INDEPENDENT_CANONICAL_GLOBAL_AND_ONCE_RECEIPT_FULL_BYTES',GLOBAL_sha256=sha(b),GLOBAL_bytes=len(b),
 intent_sha256=sha(ib),receipt_sha256=sha(rb),used_PBS=g['used']['PBS_submissions'],pending_updates=[],unknown_effects=[],
 no_existing_new_owner_or_dispatch=True,old_config_programs_have_literal_root=any(str(root)in body for body in cfg['programs'].values()),retired_preparation_sha256=P['retired_preparation_sha256'],observed_epoch=time.time(),new_writes=0)))
'''
