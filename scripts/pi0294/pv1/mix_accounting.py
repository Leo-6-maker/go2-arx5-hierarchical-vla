"""Typed C/D stage records inside the existing PV1 global ledger and onceIO.

No second budget, transport or scheduler. All stage resources are parts of the
one adopted successor package. Legacy rows remain validated by legacy code.
"""
from copy import deepcopy
import hashlib,json,math,re,uuid
from . import successor_cd as scope
from .future_budget import _vector
from .mix1024 import ARMS
from .recipe import PARENT
from scripts.pi0294.e1.scope import STATS

VERSION='V5_CD_VANDA_A40_MATCHED1024_V1'
CORE_VERSION='V5_CD_CORE_MATCHED_DEV24_V1'
PACKAGE='V5_CD_execution_package'
CORE_ROWS='V5_CD_Core_allocations'
TERMINAL=('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB')
STATUSES=('QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY','SUBMITTED_PENDING_NATIVE','NATIVE_RUNNING_QUALIFICATION',*TERMINAL)

def raw(x):return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def need(ok,msg):
    if not ok:raise PermissionError('V5_CD_ACCOUNTING_'+msg)
def number(v,integer=False):
    need(type(v)in(int,float) and math.isfinite(v) and v>=0 and (not integer or type(v)is int),'FINITE_COUNTER');return v

def stage_vector(kind,*,Core_GPU_projection=False):
    from .evaluation_accounting import output_upper
    from .evaluation_core_accounting import MODEL_BYTES
    if kind=='TRAIN':return _vector(dict(H200_s=22560,PBS_submissions=1,optimizer_updates=1024,neural_queries=29,Hopper_CPU_s=55800,local_CPU_s=15,transfer_bytes=256*1024**2,Hopper_storage_bytes=70000000000+64*1024**2))
    need(kind=='EVAL','ONLY_TRAIN_OR_EVAL')
    output=output_upper(24)+2*output_upper(1)+MODEL_BYTES
    need(type(Core_GPU_projection)is bool,'EXPLICIT_PHYSICAL_CORE_CPU_CLASSIFICATION')
    return _vector(dict(H200_s=28860,episode_starts=26,neural_queries=26*80+17,physics_ticks=26*32000,engine_starts=26*81+2,Hopper_CPU_s=87200,Core_CPU_s=93900+(80000 if Core_GPU_projection else 0),local_CPU_s=7200,transfer_bytes=output+MODEL_BYTES+1024**3,Hopper_storage_bytes=output+64*1024**2,Core_storage_bytes=26*768*1024**2+512*1024**2+MODEL_BYTES+128*1024**2))

def package_view(g):
    scope.validated(g);p=g.get(PACKAGE)
    need(type(p)is dict and p.get('schema')=='V5_CD_EXISTING_LEDGER_PACKAGE_ESCROW_V1','PACKAGE_ACTUALLY_RESERVED')
    need(p['order']==list(scope.execution_order(g)) and p['scope_grant_sha256']==scope.GRANT_SHA,'FROZEN_ORDER_AND_AUTHORITY')
    _vector(p['remaining_resources']);need(p['completed_stages']==list(scope.execution_order(g)[:len(p['completed_stages'])]),'NO_SKIP_REORDER_OR_NEW_CANDIDATE')
    need(p['engineering_replacements_used']in range(5),'TOTAL_FOUR_ENGINEERING_REPLACEMENTS')
    return p

def validate_item(r):
    need(r.get('platform_training_version')==VERSION and r.get('native_platform')=='VANDA_A40','REAL_A40_NEW_TYPED_VERSION')
    arm=r.get('arm');need(arm in ARMS and r.get('stage')=='V5_MATCHED_TRAIN1024','ONLY_DISTINCT_C_D_TRAINING')
    for k,v in dict(parent_checkpoint_manifest_sha256=PARENT,statistics_sha256=STATS,optimizer_update_limit=1024,query_limit=29,CPU_upper_s=54000,reserved_H200_s=22500,accounting_pad_s=60,physical_GPU_count=1,physical_GPU_kind='NVIDIA_A40',project='CFP04-CF-007',job_name='pi0294_dec1_a40_train',additional_output_upper_bytes=70000000000).items():need(r.get(k)==v,'EXACT_STAGE_RESOURCE:'+k)
    for k in ('request_id','submission_nonce'):
        try:need(str(uuid.UUID(r[k]))==r[k],'CANONICAL_UUID')
        except (KeyError,ValueError,TypeError,AttributeError):raise PermissionError('V5_CD_CANONICAL_UUID')
    for k,n in [('source_sha',40),('request_sha256',64),('run_manifest_sha256',64),('cold_input_audit_sha256',64)]:need(type(r.get(k))is str and re.fullmatch('[0-9a-f]{'+str(n)+'}',r[k]),'SEALED_IDENTITY:'+k)
    need(r.get('status')in STATUSES,'KNOWN_NATIVE_STATE')
    if r.get('job_id')is None:need(r['status']=='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY','UNBOUND_IS_UNKNOWN')
    else:need(re.fullmatch(r'[0-9]+\.stdct-mgmt-02',r['job_id']),'ACTUAL_VANDA_JOB')
    if r['status']in TERMINAL:
        for k in ('terminal_native_sha256','terminal_effect_summary_sha256'):need(re.fullmatch('[0-9a-f]{64}',r.get(k,'')),'ACTUAL_TERMINAL_PROOF')
    return r

def validate_global(g):
    m=scope.validated(g);rows=[r for r in g['PBS_submissions'] if r.get('platform_training_version')==VERSION]
    # New GEN1 jobs still use this original ledger. Its two corrected caps
    # come only from the exact adopted four-cap grant, never a manifest wish.
    caps=dict(PBS_submissions=34,Hopper_storage_bytes=scope.NEW_STORAGE['Hopper_storage_bytes'])
    from scripts.pi0294.generalization import amd_adoption as gen1
    if gen1.FIELD in g:caps.update(gen1.validate(g)['caps'])
    from .mix_vanda_cohost import validated as cohost_marker
    cohost_marker(g)
    need(len(rows)<=6,'TWO_ARMS_MAX_THREE_ENGINEERING_ATTEMPTS_EACH')
    for r in rows:
        validate_item(r);need(r['T_CD']==m['T_CD'] and r['deadline_epoch']==m['deadline_epoch'] and r['scope_grant_sha256']==scope.GRANT_SHA,'ONE_ACTUAL_SCOPE_CLOCK')
    need(all(sum(r['arm']==arm for r in rows)<=3 for arm in ARMS),'THREE_PER_ARM')
    active=[r for r in g['PBS_submissions'] if r['status']not in TERMINAL]
    old_core=[r for r in g.get('Core_native_GPU_allocations',{}).values() if r['status'] not in ('NATIVE_TERMINAL','NATIVE_QUEUED_NOT_RUNNING','STAGE_QUEUE_INTENT_UNKNOWN_NO_RETRY')]
    new_core=[r for r in g.get(CORE_ROWS,{}).values() if r['status']!='NATIVE_TERMINAL']
    need(len(active)+len(old_core)+len(new_core)<=1,'ONE_TASK_GPU_ACROSS_BOTH_SITES')
    if PACKAGE in g:
        p=package_view(g);remaining=p['remaining_resources']
        from .evaluation_work import MATRIX
        from .future_budget import FINAL
        need(g['new_storage_current_bytes']+g['storage_reserved_bytes']+MATRIX['Hopper_storage_bytes']+FINAL['Hopper_storage_bytes']<=caps['Hopper_storage_bytes'],'OUTPUT_MATRIX_AND_FINAL_PROTECTED')
        need(g['used']['H200_s']+g['reserved']['H200_s']+MATRIX['H200_s']+FINAL['H200_s']<=121*3600,'GPU_MATRIX_AND_FINAL_PROTECTED')
        need(g['used']['PBS_submissions']+remaining['PBS_submissions']+MATRIX['PBS_submissions']<=caps['PBS_submissions'],'MATRIX_THREE_PBS_PROTECTED')
        need(g['CPU_s']+g['CPU_reserved_s']+MATRIX['Hopper_CPU_s']<=213*3600,'SITE_CPU_MATRIX_PROTECTED')
        need(g['reserved']['H200_s']>=remaining['H200_s'] and g['reserved']['optimizer_updates']>=remaining['optimizer_updates'] and g['reserved']['neural_queries']>=remaining['neural_queries'],'ALL_PACKAGE_NEURAL_ESCROW_RETAINED')
        need(g['CPU_reserved_s']>=remaining['Hopper_CPU_s'] and g['storage_reserved_bytes']>=remaining['Hopper_storage_bytes'],'ALL_PACKAGE_CPU_AND_OUTPUT_ESCROW_RETAINED')
    if CORE_ROWS in g:
        from .mix_core_accounting import validate_rows
        validate_rows(g)
    return g

def reserve_package(global_raw,local_raw,*,now_epoch,Core_storage_upper_bytes,local_storage_upper_bytes):
    g,l=json.loads(global_raw),json.loads(local_raw);m=scope.validated(g)
    need(l.get(scope.FIELD)==m and m['T_CD']<=now_epoch<m['stop_new_training_epoch'],'BOTH_ADOPTED_AND_TRAIN_WINDOW')
    need(PACKAGE not in g and PACKAGE not in l,'ONCE_WHOLE_PACKAGE_RESERVATION')
    need(not any(g['reserved'].values()) and g.get('CPU_reserved_s',0)==g.get('storage_reserved_bytes',0)==0,'NO_OTHER_GPU_HOLD')
    proof=scope.recompute(g,l,Core_storage_upper_bytes=Core_storage_upper_bytes,local_storage_upper_bytes=local_storage_upper_bytes)
    need(not proof['shortfalls'],'WHOLE_PACKAGE_AND_MATRIX_FINAL_PAYABLE')
    v,w=deepcopy(g),deepcopy(l);resources=scope.whole_package();p=dict(schema='V5_CD_EXISTING_LEDGER_PACKAGE_ESCROW_V1',scope_grant_sha256=scope.GRANT_SHA,order=list(scope.ORDER),created_epoch=now_epoch,original_resources=resources,remaining_resources=deepcopy(resources),completed_stages=[],engineering_replacements_used=0)
    v[PACKAGE]=deepcopy(p);w[PACKAGE]=deepcopy(p)
    for k in ('H200_s','optimizer_updates','neural_queries'):v['reserved'][k]+=resources[k]
    v['CPU_reserved_s']+=resources['Hopper_CPU_s'];v['storage_reserved_bytes']+=resources['Hopper_storage_bytes']
    w['GPU_reserved_s']+=resources['H200_s']
    for k,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:w['CPU_reserved_s'][site]+=resources[k]
    validate_global(v)
    return dict(proposed_global=v,proposed_LOCAL=w,proof=proof,not_native_GPU_admission=True)

def prepay_training_control(g,*,arm,now_epoch):
    v=deepcopy(g);p=package_view(v);need(p['completed_stages']==list(scope.execution_order(g)[:scope.execution_order(g).index('TRAIN:'+arm+':1024')]),'PRIOR_CANDIDATE_AND_EVAL_BEFORE_NEXT_ARM')
    key='TRAIN:'+arm+':1024';charges=p.setdefault('prepaid_control',{});need(key not in charges,'CONTROL_PAID_ONCE')
    costs=dict(Hopper_CPU_s=1800,Hopper_storage_bytes=64*1024**2,transfer_bytes=256*1024**2,local_CPU_s=15)
    charges[key]=dict(costs=costs,epoch=now_epoch,financial_upper_not_actual_process_CPU=True)
    for k,x in costs.items():p['remaining_resources'][k]-=x
    v['CPU_reserved_s']-=costs['Hopper_CPU_s'];v['CPU_s']+=costs['Hopper_CPU_s'];v['storage_reserved_bytes']-=costs['Hopper_storage_bytes'];v['new_storage_current_bytes']+=costs['Hopper_storage_bytes'];v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes']);validate_global(v);return v

def register_request(g,request,*,now_epoch):
    v=deepcopy(g);p=package_view(v);r=deepcopy(request);validate_item(r)
    key='TRAIN:'+r['arm']+':1024';need(key in p.get('prepaid_control',{}),'REAL_ONCE_CONTROL_PAYMENT_FIRST')
    need(not any(x['status']not in TERMINAL for x in v['PBS_submissions']),'NO_ACTIVE_SECOND_GPU')
    need(r['status']=='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY' and r['job_id']is None and not any(x.get('request_id')==r['request_id'] for x in v['PBS_submissions']),'UNIQUE_REQUEST_NEVER_RESUBMIT')
    need(r['T_CD']==scope.validated(v)['T_CD'] and now_epoch<scope.validated(v)['stop_new_training_epoch'],'FRESH_SUCCESSOR_TRAINING_WINDOW')
    need(p['remaining_resources']['PBS_submissions']>=1 and p['remaining_resources']['optimizer_updates']>=1024,'BOTH_ARM_UPDATE_AND_PBS_ESCROW')
    v['PBS_submissions'].append(r);v['used']['PBS_submissions']+=1;p['remaining_resources']['PBS_submissions']-=1;validate_global(v);return v

def bind(g,request_id,job_id,*,now_epoch):
    v=deepcopy(g);package_view(v);r=next(x for x in v['PBS_submissions'] if x.get('request_id')==request_id);need(r['status']=='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY' and r['job_id']is None,'ONE_BIND_NO_QSUB_REPLAY');r.update(job_id=job_id,status='SUBMITTED_PENDING_NATIVE',native_reply_epoch=now_epoch);validate_global(v);return v

def settle_training(g,request_id,*,native,effects,audit,now_epoch):
    v=deepcopy(g);p=package_view(v);r=next(x for x in v['PBS_submissions'] if x.get('request_id')==request_id);validate_item(r)
    need(r['status']not in TERMINAL,'NO_REPEAT_SETTLEMENT')
    need(native['job_state']=='F' and native['job_id']==r['job_id'] and native['request_id']==request_id and native['submission_nonce']==r['submission_nonce'],'REAL_NATIVE_FINAL_NOT_OBSERVATION_TIMEOUT')
    need(effects['schema']=='V5_CD_ACTUAL_NATIVE_EFFECT_SUMMARY_V1' and effects['request_id']==request_id and effects['request_sha256']==r['request_sha256'] and effects['job_id']==r['job_id'] and effects['arm']==r['arm'],'ACTUAL_SAME_REQUEST_EFFECTS')
    steps=number(effects['optimizer_updates_committed'],True);queries=number(effects['query_attempts'],True);loads=number(effects['model_load_attempts'],True);gpu=number(native['conservative_GPU_s']);cpu=number(native['conservative_CPU_s'])
    need(steps<=1024 and queries<=29 and loads<=4 and gpu<=22560 and cpu<=54000,'ALLOCATION_COST_AND_EFFECT_BOUNDS')
    need(effects['pending_updates']==effects['unknown_query_intents']==[],'UNRESOLVED_EFFECTS_KEPT_NOT_SETTLED_AS_SUCCESS')
    success=native['Exit_status']==0 and steps==1024 and audit.get('status')=='PASS_INDEPENDENT_V5_CD_1024_COMPLETE_STATE'
    amounts=dict(H200_s=22560,optimizer_updates=1024,neural_queries=29,Hopper_CPU_s=54000,Hopper_storage_bytes=70000000000)
    for k,x in amounts.items():p['remaining_resources'][k]-=x;need(p['remaining_resources'][k]>=0,'NO_PACKAGE_RESOURCE_UNDERFLOW')
    for k in ('H200_s','optimizer_updates','neural_queries'):v['reserved'][k]-=amounts[k]
    v['CPU_reserved_s']-=54000;v['storage_reserved_bytes']-=70000000000
    v['used']['H200_s']+=gpu;v['optimizer_updates']+=steps;v['used']['optimizer_updates']=v['optimizer_updates'];v['generation_queries']+=queries;v['used']['neural_queries']=v['generation_queries'];v['model_loads']+=loads;v['CPU_s']+=cpu
    # Preserve the complete conservative allocation output charge, including
    # failures. No previous checkpoint or paid bound is deleted/refunded.
    v['new_storage_current_bytes']+=70000000000;v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    r.update(status='NATIVE_TERMINAL',native_exit_status=native['Exit_status'],conservative_H200_s=gpu,conservative_CPU_s=cpu,committed_optimizer_updates=steps,query_attempts=queries,model_load_attempts=loads,reserved_neural_queries=0,reserved_optimizer_updates=0,settled_epoch=now_epoch,terminal_native_sha256=sha(raw(native)),terminal_effect_summary_sha256=sha(raw(effects)),full_checkpoint_audit_sha256=sha(raw(audit)),scientific_endpoint_complete=success)
    if success:
        key='TRAIN:'+r['arm']+':1024';need(scope.execution_order(g)[len(p['completed_stages'])]==key,'FIXED_STAGE_ORDER');p['completed_stages'].append(key)
        p.setdefault('endpoints',{})[r['arm']]=dict(checkpoint_manifest_sha256=audit['checkpoint_manifest_sha256'],checkpoint_root=audit['checkpoint_root'],global_step=1024,request_id=request_id,job_id=r['job_id'],source_sha=r['source_sha'],audit_sha256=sha(raw(audit)))
    else:p['blocked_stage']=dict(request_id=request_id,arm=r['arm'],reason='FAILED_OR_INCOMPLETE_NOT_A_VALID_SCIENTIFIC_ENDPOINT',autonomous_cold_repair_required=True)
    validate_global(v);return v

def validate_request(r):
    need('request_sha256' not in r,'REQUEST_HASH_LIVES_IN_CENTRAL_ROW_NOT_SELF_REFERENTIAL_BYTES')
    validate_item(dict(r,request_sha256=sha(raw(r))))
    need(r.get('schema')=='V5_CD_ACTUAL_NATIVE_TRAINING_REQUEST_V1' and r.get('expected_native_start_step')==0
        and r.get('training_block')==dict(arm=r['arm'],replica=False,stop_after_step=1024,resume_checkpoint_relative=None), 'FRESH_SAME_PARENT_C_D_1024_BLOCK')
    return r
