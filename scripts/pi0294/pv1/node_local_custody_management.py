"""Close only the known unsubmitted D25-04 preparation roles, once.

No financial refund, native terminal fabrication, source modification or new
runtime admission. The original task escrow and scientific window stay open.
"""

CLOSE=r'''
import hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
sha=lambda b:hashlib.sha256(b).hexdigest();raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
assert P['request_id']=='3e774da8-1f75-4930-b732-3f0409f3d5a4'and P['source_sha']=='2a23eeb64e7f769fa5842eb72d371ac12ee99664'
lb=Path(P['LOCAL_path']).read_bytes();assert sha(lb)==P['LOCAL_sha256']
local=json.loads(lb);row=local['V5_CD_Core_allocations'][P['request_id']]
assert local['pending_queries']==local['pending_updates']==local['unknown_effects']==[]
assert row['source_sha']==P['source_sha']and row['status']=='PRESTART_INTENT_UNKNOWN_NO_RETRY'and row['native_handle']is None
assert not row.get('charged_effects')and not row.get('query_attempts',0)and not row.get('model_load_attempts',0)
assert set(P['roles'])=={'owner','physical','neural'}
assert len({v['root']for v in P['roles'].values()})==3
units={}
for unit in P['units']:
 q=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','LoadState'],capture_output=True,text=True,timeout=4)
 units[unit]=dict(x.split('=',1)for x in q.stdout.splitlines()if '='in x)
 assert units[unit]==dict(MainPID='0',LoadState='not-found')
prepared={}
for role,pin in P['roles'].items():
 root=Path(pin['root']);assert root.name==('v5-cd-d25-eval-owner-04'if role=='owner'else'v5-cd-d25-n294101-04')
 mb=(root/'RUN_MANIFEST.json').read_bytes();assert sha(mb)==pin['manifest_sha256'];m=json.loads(mb)
 assert m['code']['vla_commit']==P['source_sha']
 assert m['scope']=={'owner':'PI0294_PV1_V5_CD_CORE_RESIDENT_OWNER_V1','physical':'V5_CD_CORE_POLICY_EVALUATION_V1','neural':'V5_CD_CORE_NEURAL_EVALUATION_V1'}[role]
 b=json.loads((root/'OUTPUT_BINDING.private.json').read_bytes());lease_path=Path(b['lease']);leaseb=lease_path.read_bytes();lease=json.loads(leaseb)
 assert str(lease_path)==pin['lease_path']and sha(leaseb)==pin['lease_sha256']
 assert lease['status']=='ACTIVE'and lease['run_uuid']==root.name and lease['deadline_epoch']==row['deadline_epoch']
 assert time.time()<lease['deadline_epoch']and not(root/'resident').exists()
 traces={name:sum(1 for _ in root.glob(name))for name in('**/GPU_START_INTENT.json','**/NATIVE_CORE_PROCESS_STARTED.json','**/MODEL_LOAD_INTENTS/*.json','**/QUERY_STATE.json')}
 assert not any(traces.values())and not(root/'D25_COLD04_UNUSED_CLOSED_RECORD.json').exists()
 prepared[role]=dict(root=root,manifest_sha256=sha(mb),lease_path=lease_path,lease_bytes=leaseb,lease=lease,native_traces=traces)
assert len({str(v['lease_path'])for v in prepared.values()})==3
physical=prepared['physical']['root'];pb=(physical/'RESOURCE_LEDGER.json').read_bytes();physical_ledger=json.loads(pb)
assert physical_ledger['run_uuid']=='v5-cd-d25-n294101-04'and physical_ledger['schema']=='V5_CD_CORE_POLICY_EFFECT_LEDGER_V1'
assert physical_ledger['reservations']=={}and physical_ledger['unknown_effects']==[]
assert all(physical_ledger[k]==0 for k in('main_starts','engine_starts','live_and_recovery_integrations','shadow_integrations'))
assert physical_ledger['Core_CPU_s']==physical_ledger['metadata_control_financial_upper_s']==300
assert physical_ledger['CPU_reserved_s']==93600
actors=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=2009:continue
  if any(str(v['root']).encode()in(proc/'cmdline').read_bytes()for v in prepared.values()):actors.append(int(proc.name))
 except (OSError,ProcessLookupError):pass
assert actors==[]
owner=prepared['owner']['root'];om=json.loads((owner/'RUN_MANIFEST.json').read_bytes())
seal=json.loads((owner/om['code']['source_seal_relative']).read_bytes());assert seal['source_sha']==P['source_sha']
once_file=owner/om['code']['capsule_relative']/'scripts/pi0294/dec1/source_cpu_once_io.py'
assert sha(once_file.read_bytes())==seal['files']['scripts/pi0294/dec1/source_cpu_once_io.py']
spec=importlib.util.spec_from_file_location('_original_D25_04_onceIO',once_file);once=importlib.util.module_from_spec(spec);spec.loader.exec_module(once)
records={}
for role,item in prepared.items():
 root=item['root'];path=item['lease_path'];lease=dict(item['lease'])
 lease.update(status='CLOSED',terminal_reason='UNSUBMITTED_COLD04_PREPARATION_CLOSED_FOR_NODE_LOCAL_SUCCESSOR',
  all_actual_workers_terminal=True,models_queries_physics=0,scientific_stage_or_scope_closed=False,
  original_financial_escrow_and_metadata_costs_retained=True,closed_epoch=time.time())
 key='D25-CLOSE-UNSUBMITTED04-'+role.upper()
 receipt=once.commit(path,expected_before_sha256=sha(item['lease_bytes']),proposed_bytes=raw(lease),request_id=key,
  intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),lock_path=path.with_name('D25_UNUSED_ROLE_CLOSE.owner.lock'))
 record=dict(schema='D25_04_UNSUBMITTED_COLD_ROLE_CLOSED_V1',status='ACTUAL_UNUSED_COLD_ROLE_CLOSED_WITH_NO_NATIVE_ACTOR',
  role=role,root=str(root),source_sha=P['source_sha'],request_id=P['request_id'],manifest_sha256=item['manifest_sha256'],
  lease_path=str(path),before_lease_sha256=sha(item['lease_bytes']),closed_lease_sha256=receipt['after_sha256'],
  original_once_intent_sha256=receipt['intent_sha256'],original_once_receipt_sha256=sha(raw(receipt)),
  native_effect_traces=item['native_traces'],live_actors=[],models_queries_physics=0,
  original_physical_ledger_sha256=sha(pb),physical_metadata_CPU_upper_s_carried=300,
  physical_unused_CPU_reserved_s_retained=93600,old_NATIVE_TERMINAL_row_fabricated=False,
  old_ledger_rows_or_costs_rewritten=False,observed_epoch=time.time())
 once._exclusive(root/'D25_COLD04_UNUSED_CLOSED_RECORD.json',raw(record));records[role]=record
assert Path(P['LOCAL_path']).read_bytes()==lb
print(json.dumps(dict(status='ACTUAL_THREE_UNSUBMITTED_COLD04_OWN_LEASES_CLOSED_ONCE',records=records,
 units=units,LOCAL_unchanged_sha256=sha(lb),native_rows_or_financial_escrow_changed=False,
 new_models_queries_physics=0,scientific_scope_closed=False,automatic_relaunch=False,observed_epoch=time.time())))
'''

READ_CLOSED=r'''
import hashlib,json,os,time
from pathlib import Path
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
sha=lambda b:hashlib.sha256(b).hexdigest();out={}
for role,pin in P['roles'].items():
 root=Path(pin['root']);rb=(root/'D25_COLD04_UNUSED_CLOSED_RECORD.json').read_bytes();r=json.loads(rb)
 assert r['source_sha']==P['source_sha']and r['request_id']==P['request_id']and r['role']==role
 assert sha((root/'RUN_MANIFEST.json').read_bytes())==r['manifest_sha256']
 path=Path(r['lease_path']);lb=path.read_bytes();lease=json.loads(lb)
 assert lease['status']=='CLOSED'and lease['run_uuid']==root.name and sha(lb)==r['closed_lease_sha256']
 key='D25-CLOSE-UNSUBMITTED04-'+role.upper();ib=path.with_name(key+'.INTENT.json').read_bytes();cb=path.with_name(key+'.RECEIPT.json').read_bytes();i,c=map(json.loads,(ib,cb))
 assert sha(ib)==r['original_once_intent_sha256']==c['intent_sha256']and sha(cb)==r['original_once_receipt_sha256']
 assert i['target']==c['target']==str(path)and i['proposed_sha256']==c['after_sha256']==sha(lb)
 assert i['before_sha256']==c['before_sha256']==r['before_lease_sha256']
 out[role]=dict(record=r,record_sha256=sha(rb),closed_lease_sha256=sha(lb),once_receipt_binding_verified=True)
physical=Path(P['roles']['physical']['root'])/'RESOURCE_LEDGER.json'
assert sha(physical.read_bytes())==out['physical']['record']['original_physical_ledger_sha256']
assert sha(Path(P['LOCAL_path']).read_bytes())==P['LOCAL_sha256']
print(json.dumps(dict(status='INDEPENDENT_THREE_D25_04_CLOSED_OWN_LEASES_AND_ORIGINAL_COST_CARRY_VERIFIED',
 roles=out,LOCAL_unchanged_sha256=P['LOCAL_sha256'],new_writes=0,new_model_queries_physics=0,
 financial_escrow_changed=False,scientific_stage_closed=False,observed_epoch=time.time())))
'''
