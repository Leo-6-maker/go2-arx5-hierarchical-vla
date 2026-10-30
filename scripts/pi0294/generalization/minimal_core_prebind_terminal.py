"""Settle the exact shared E1 unit exit before canonical process binding.

The real process is terminal evidence only. No dead PID is adopted as running,
no delegation is invented and no continuation is granted. Uses original holds.
"""
import base64,json,math,re
from copy import deepcopy
from . import minimal_core_accounting as c,minimal_core_sharing as sharing

FIELD='ARX5_E1_shared03_known_zero_prebind_terminal'
KIND='ARX5_E1_SHARED03_ACTUAL_PREBIND_RELEASE_TIMEOUT_TERMINAL_V1'
SOURCE='cf62052564a838c5b99ec13b4c80b14f987403e4'
REQUEST='45667d2cd24ecba02758f18f5872176b48d01b16c77e106b66bc367bef8928b8'
need=c.need


def row(r):
    need(r.get('request_id')==sharing.RID and r.get('source_sha')==SOURCE
        and r.get('request_sha256')==REQUEST and r.get('run_uuid')==sharing.RUN
        and r.get('status')==c.STATES[3] and r.get('native_handle')is None
        and r.get('terminal_identity_kind')==KIND and r.get('native_exit_status')==1
        and r.get('query_attempts')==r.get('model_load_attempts')==0
        and r.get('unknown_query_intents')==[] and r.get('reserved_neural_queries')==0
        and not any(k in r for k in('pid','proc_start','boot_id','process_started_epoch')),
        'EXACT_E1_TERMINAL_WITNESS_NOT_INVENTED_CANONICAL_LIVE_BIND')
    return r


def validate(g):
    if FIELD not in g:
        need(not any(r.get('terminal_identity_kind')==KIND for r in g.get(c.FIELD,{}).values()),
            'NO_UNADOPTED_PREBIND_TERMINAL')
        return None
    m=g[FIELD];r=row(g[c.FIELD][sharing.RID])
    need(m['schema']==KIND and m['request_id']==sharing.RID and m['source_sha']==SOURCE
        and m['terminal_native_sha256']==r['terminal_native_sha256']
        and m['GPU_s']==r['conservative_GPU_s'] and m['neural_CPU_s']==r['conservative_CPU_s']
        and m['native_handle']is None and m['actual_start_is_terminal_witness_only']is True
        and type(m['GPU_s'])is int and 240<=m['GPU_s']<=245
        and type(m['neural_CPU_s'])is int and 0<m['neural_CPU_s']<=10
        and m['physical_CPU_s']==300 and m['new_runtime_transfer_bytes']==0
        and type(m['Core_output_bytes'])is int and 0<m['Core_output_bytes']<=256*1024**2
        and m['query_attempts']==m['model_load_attempts']==m['new_physics']==m['new_episodes']==0
        and m['stage_completed']is False and m['new_native_permission']is False,
        'EXACT_KNOWN_ZERO_FINANCIAL_TERMINAL_ONLY')
    return m


def observed(p,r,*,global_sha256,local_sha256,now_epoch):
    core,global_read,failed=p['core'],p['global'],p['failed_bind']
    need(r['request_id']==sharing.RID and r['source_sha']==SOURCE and r['request_sha256']==REQUEST
        and r['native_handle']is None and r['status']==c.STATES[0] and c.progress(r)==(0,0,0),
        'EXACT_UNBOUND_UNSPENT_REQUEST')
    need(core['status']=='ACTUAL_SHARED03_PREMODEL_TIMEOUT_UNIT_EXIT_FULL_ZERO_RECORDS_READONLY'
        and core['host']=='corelab-4080s' and core['uid']==2009
        and core['LOCAL_sha256']==local_sha256
        and global_read['GLOBAL_sha256']==global_sha256
        and global_read['all_three_paths_absent']is True
        and global_read['unknown_effects']==global_read['pending_updates']==[]
        and failed['sha256']==failed['expected_before']==global_sha256
        and failed['owned_live_actor_birth_window']==[]
        and all(v['exists']is False for v in failed['files'].values())
        and set(failed['files'])=={'INTENT.json','RECEIPT.json'}
        and all(0<=now_epoch-v['observed_epoch']<=120 for v in(core,global_read)),
        'FRESH_BOTH_CANONICAL_BYTES_AND_EXACT_PRECOMMIT_ZERO_NO_REPLAY')
    need(core['qualification_model_query_release_or_effect_records']==[],
        'NO_QUALIFICATION_RELEASE_MODEL_QUERY_OR_PHYSICAL_WORKER')
    def decode(name):
        f=core['files'][name];b=base64.b64decode(f['base64'],validate=True)
        need(c.sha(b)==f['sha256'],'ACTUAL_CAPTURED_FILE_BYTES:'+name)
        return json.loads(b)
    for name in core['files']:decode(name)
    start=decode('hopper/pbs/alloc-01/NATIVE_CORE_PROCESS_STARTED.json')
    final=decode('hopper/pbs/alloc-01/NATIVE_PROCESS_FINAL.json')
    physical=decode('core/RESOURCE_LEDGER.json')
    owner=decode('owner/V5_CD_NATIVE_OWNER_STARTED.private.json')
    exception=decode('owner/OWNER_PRESERVED_EXCEPTION.json')
    need(core['files']['hopper/RUN_MANIFEST.json']['sha256']==r['run_manifest_sha256']
        and core['files']['core/RUN_MANIFEST.json']['sha256']==r['physical_manifest_sha256'],
        'SAME_IMMUTABLE_REGISTERED_PEERS')
    need(start['source_sha']==SOURCE and start['request_id']==sharing.RID
        and start['request_sha256']==REQUEST and start['run_manifest_sha256']==r['run_manifest_sha256']
        and start['native_model_permission']is start['physics_authorized']is False
        and start['native_handle']==c.process_handle(start['boot_id'],start['pid'],start['proc_start'])
        and owner['source_sha']==SOURCE and owner['request_id']==sharing.RID
        and owner['request_sha256']==REQUEST and owner['run_uuid']=='arx5-e1-fixed512-eval-owner-03'
        and final['job_id']==start['native_handle'] and final['status']=='FAILED_PRESERVED'
        and final['phase']=='WAIT_ACTUAL_GLOBAL_BIND_AND_PREMODEL_RELEASE'
        and final['error']=='TimeoutError: PV1_CORE_NATIVE_RELEASE_NOT_OBSERVED_NO_RESTART'
        and final['records']==[] and final['new_optimizer_updates']==0
        and exception['error']=='EXACT_NATIVE_PROCESS_DURING_FINITE_ADOPTION',
        'EXACT_ORDERED_ZERO_MODEL_FAILURE_NOT_A_GENERIC_TIMEOUT')
    units=core['units'];neural=r['native_unit'];owner_unit=owner['run_uuid']+'.service'
    need(set(units)=={neural,owner_unit} and all(v['MainPID']=='0' and v['ActiveState']=='failed'
        and v['ExecMainStatus']==v['ExecMainCode']=='1' and v['InvocationID'] for v in units.values())
        and int(units[neural]['ExecMainPID'])==start['pid']
        and int(units[owner_unit]['ExecMainPID'])==owner['pid'],
        'BOTH_ACTUAL_EXACT_NATIVE_UNIT_EXITS')
    u=units[neural];begin=core['boot_epoch']+int(u['ExecMainStartTimestampMonotonic'])/1e6
    end=core['boot_epoch']+int(u['ExecMainExitTimestampMonotonic'])/1e6
    # /proc/stat btime is whole seconds; do not compare its reconstructed
    # epoch to time.time() as if both had microsecond precision. Independently
    # bound realtime using the actual unit's recorded Unix timestamps, retaining
    # the full uncertain last second in the conservative financial charge.
    need(all(type(u.get(k))is str and re.fullmatch(r'@[0-9]+',u[k])for k in
        ('ExecMainStartTimestamp','ExecMainExitTimestamp')),'ACTUAL_UNIT_REALTIME_FIELDS')
    real_begin=int(u['ExecMainStartTimestamp'][1:]);real_end=int(u['ExecMainExitTimestamp'][1:])
    need(abs(begin-start['started_epoch'])<.1 and r['created_epoch']<=begin<end<=core['observed_epoch']
        and abs((real_end-real_begin)-(end-begin))<1.1
        and start['observed_epoch']<=final['completed_epoch']
        and real_end<=final['completed_epoch']<real_end+1<=core['observed_epoch'],
        'ACTUAL_NATIVE_MONOTONIC_AND_REALTIME_EXIT_BOUNDARIES')
    gpu=math.ceil(max(end,real_end+1)-start['started_epoch'])+60
    cpu=math.ceil(int(u['CPUUsageNSec'])/1e9)
    need(240<=gpu<=245 and 0<cpu<=10,'COMPLETE_PADDED_NATIVE_WALL_AND_CPU')
    need(physical['unknown_effects']==[] and physical['reservations']=={}
        and physical['Core_CPU_s']==300 and all(physical[k]==0 for k in
        ('main_starts','engine_starts','live_and_recovery_integrations','shadow_integrations')),
        'COMPLETE_ORIGINAL_ZERO_PHYSICS_COUNTERS')
    storage=core['terminal_storage']
    need(storage['both_native_units_terminal']is True and storage['followed_external_symlinks']is False
        and storage['deleted_files']==0 and type(storage['allocated_bytes_upper'])is int
        and 0<storage['allocated_bytes_upper']<=storage['charged_upper_bytes']<=256*1024**2,
        'ALL_THREE_RETAINED_TERMINAL_ROOTS_CONSERVATIVE_SMALL_STORAGE')
    return dict(GPU_s=gpu,neural_CPU_s=cpu,physical_CPU_s=300,
        Core_output_bytes=storage['charged_upper_bytes'],new_runtime_transfer_bytes=0)


def propose(global_bytes,local_bytes,*,proof,now_epoch):
    g,l=map(json.loads,(global_bytes,local_bytes))
    need(FIELD not in g and FIELD not in l,'ONCE_PREBIND_TERMINAL')
    r=g[c.FIELD][sharing.RID]
    amounts=observed(proof,r,global_sha256=c.sha(global_bytes),local_sha256=c.sha(local_bytes),now_epoch=now_epoch)
    c.paired(g,l)
    v,w=deepcopy(g),deepcopy(l);rr=v[c.FIELD][sharing.RID];digest=c.digest(proof)
    effects=dict(schema=KIND,request_id=sharing.RID,model_load_attempts=0,query_attempts=0,
        new_physics=0,new_episodes=0,actual_start_is_terminal_witness_only=True)
    rr.update(status=c.STATES[3],terminal_identity_kind=KIND,native_exit_status=1,
        conservative_GPU_s=amounts['GPU_s'],conservative_CPU_s=amounts['neural_CPU_s'],
        model_load_attempts=0,query_attempts=0,unknown_query_intents=[],reserved_neural_queries=0,
        terminal_native_sha256=digest,terminal_effect_summary_sha256=c.digest(effects),settled_epoch=now_epoch)
    for x in(v,w):x[c.FIELD][sharing.RID]=deepcopy(rr)
    v['used']['H200_s']+=amounts['GPU_s'];v['reserved']['H200_s']-=c.HOLD['H200_s']
    v['CPU_s']+=amounts['neural_CPU_s'];v['CPU_reserved_s']-=c.HOLD['Hopper_CPU_s']
    v['reserved']['neural_queries']-=c.HOLD['neural_queries']
    v['storage_reserved_bytes']-=c.HOLD['Hopper_storage_bytes']
    v['new_storage_current_bytes']+=amounts['Core_output_bytes']
    v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    w['GPU_reserved_s']-=c.HOLD['H200_s'];w['GPU_allocated_s']+=amounts['GPU_s']
    for site,hold,spent in [('Hopper',c.HOLD['Hopper_CPU_s'],amounts['neural_CPU_s']),
        ('CoreLab',c.HOLD['Core_CPU_s'],amounts['neural_CPU_s']+amounts['physical_CPU_s'])]:
        w['CPU_reserved_s'][site]-=hold;w['CPU_s'][site]+=spent
    for key,n in dict(episode_starts=26,physics_ticks=832000,constructors=2108,
        CoreLab_output_bytes=c.HOLD['Core_storage_bytes'],transfer_bytes=c.HOLD['transfer_bytes']).items():
        need(w['Core_native_GPU_reserved'][key]>=n,'OWN_UNSPENT_HOLD:'+key)
        w['Core_native_GPU_reserved'][key]-=n
    w['Core_native_GPU_current_storage_bytes']=w.get('Core_native_GPU_current_storage_bytes',0)+amounts['Core_output_bytes']
    receipt=dict(schema=KIND,request_id=sharing.RID,source_sha=SOURCE,terminal_native_sha256=digest,
        native_handle=None,actual_start_is_terminal_witness_only=True,query_attempts=0,
        model_load_attempts=0,new_physics=0,new_episodes=0,stage_completed=False,
        new_native_permission=False,old_control_costs_retained=True,leases_closed=False,
        observed_epoch=now_epoch,**amounts)
    for x in(v,w):x[FIELD]=deepcopy(receipt)
    v['revision']+=1;c.paired(v,w)
    need(all(v[k]==g[k] for k in('PBS_submissions',c.old.CORE_ROWS,
        'ARX5_minimal_first512_execution_package','generation_queries','model_loads')),
        'LIVE_E2_D25_OLD_PACKAGE_AND_OLD_NEURAL_COSTS_PRESERVED')
    return dict(proposed_global=v,proposed_LOCAL=w,receipt=receipt,effects=effects,
        original_onceIO_required=True,native_permission=False,leases_closed=False)
