"""Single raw-E2 continuation in the original ledger; E1 is never rerun."""
import base64,json,math,re
from copy import deepcopy
from . import minimal_execution as e,minimal_e2only_contract as c,minimal_e2only_scope as authority
from . import minimal_saved512_execution as previous
from .minimal_remaining_execution import TERMINAL_KEYS
from .minimal_validation_cache import exact_ledger_validation
from .pbs_count_authority import history_upper

FIELD='ARX5_minimal_E2_only_raw512_execution'
VERSION=c.VERSION;PROFILE=c.PROFILE;RUN=c.RUN
SCHEMA='ARX5_E2_ONLY_RAW512_ORIGINAL_GLOBAL_INDEX_V1'
PHYSICAL='plans/MINIMAL_E2_ONLY_PHYSICAL_CUSTODY.private.json'
PHYSICAL_SCHEMA='ARX5_E2_ONLY_ALL_RETAINED_ROOTS_AND_NEW70GB_CUSTODY_V1'
PAID_OUTPUT=144294967296
need=e.need
CONTROL=authority.CONTROL
LOCAL_CONTROL=FIELD+'_additional_control_projection'
PREPAID=FIELD+'_cold_retry_controls'
RETIRED=FIELD+'_closed_unregistered_preparations'


def native_resources():return deepcopy(authority.NATIVE)


def fixed_request_fields(attempt=1,cold_generation=0):
    v=e.fixed_request_fields();v.update(schema='ARX5_E2_ONLY_RAW512_REQUEST_V1',
        platform_training_version=VERSION,profile=PROFILE,run_uuid=c.run_name(attempt,cold_generation),walltime_s=22440,accounting_pad_s=120,
        reserved_H200_s=22560,CPU_upper_s=54000,prior_readonly_preparation_CPU_upper_s=0,
        additional_output_upper_bytes=0,paid_output_allowance_bytes=PAID_OUTPUT,new_output_bound_bytes=70000000000,
        optimizer_update_limit=512,query_limit=37,model_load_limit=2,session_plan=c.sessions(),
        native_session_plan_version='E2_ONLY_ORIGINAL_RAW512_V1',retained_predecessor_updates=122,
        physical_guard_policy=PHYSICAL_SCHEMA,old_sessions_never_reexecuted=True,
        recovery_checkpoint_policy=c.recovery_policy(),**c.binding())
    if attempt>1:v['attempt_index']=attempt
    if cold_generation:v['pre_native_cold_generation']=cold_generation
    return v


def request(v,*,central=False):
    attempt=v.get('attempt_index',1);cold_generation=v.get('pre_native_cold_generation',0);c.run_name(attempt,cold_generation)
    need(attempt>1 or 'attempt_index'not in v,'E2ONLY_ORIGINAL_REQUEST_BYTES_NO_RESTAMP')
    need(all(v.get(k)==x and type(v.get(k))is type(x)for k,x in fixed_request_fields(attempt,cold_generation).items()),
        'E2ONLY_EXACT_FIXED512_37_2_REQUEST')
    old=deepcopy(v);old.update(e.fixed_request_fields());e.request(old,central=central)
    need(v['source_sha']!='769c59ff7912ff16bdeb6f99bfde37f759a6fd37' and v['request_id']!=authority.REQUEST,
        'E2ONLY_DISTINCT_COLD_SOURCE_AND_NEW_REQUEST_NOT_REPLAY')
    if attempt>1:
        b=v.get('zero_effect_predecessor',{})
        need(set(b)=={'request_id','row_sha256','marker_sha256','lease_sha256','settled_sha256','finite_OS0_sha256'}
            and e._uuid(b['request_id'])and all(type(x)is str and re.fullmatch('[0-9a-f]{64}',x)for k,x in b.items()if k!='request_id'),
            'E2ONLY_EXACT_CLOSED_ZERO_PREDECESSOR_PINS')
    if cold_generation:
        from .minimal_e2only_pre_native import binding
        binding(v.get('pre_native_predecessor',{}))
    else:need('pre_native_predecessor'not in v and 'pre_native_cold_generation'not in v,
        'E2ONLY_OLD_PRE_NATIVE_REQUEST_BYTES_UNCHANGED')
    for k in('physical_custody_sha256','effective_queue_proof_sha256'):
        need(type(v.get(k))is str and re.fullmatch('[0-9a-f]{64}',v[k]),'E2ONLY_EXACT_SEALED_'+k)
    return v


def select_profile(limits,r):
    request(r);need(limits.get('walltime')=='06:14:00','E2ONLY_EXACT6H14_NATIVE_PROFILE')
    old=deepcopy(r);old.update(e.fixed_request_fields());view=dict(limits,walltime='16:30:00')
    # All capacity/chunk/device checks are the original strict native parser.
    return e.select_profile(view,old)


def project_envelope(r):
    request(r);return dict(GPU_hours=r['reserved_H200_s']/3600,
        CPU_hours=36*r['reserved_H200_s']/3600,reserved_CPU_including_prior_preparation_s=r['CPU_upper_s'],
        actual_process_CPU_upper_s=r['CPU_upper_s'],active_CPU_limit=8,project_hold_is_not_experiment_CPU_spend=True)


def process_CPU_limits(r):
    request(r);return dict(parent_RLIMIT_CPU_s=53160,isolated_child_CPU_upper_s=600,
        terminal_observer_CPU_upper_s=240,prior_readonly_preparation_CPU_upper_s=0,whole_reserved_CPU_upper_s=54000,new_CPU_budget=0)


def _basic_index(g,m,row):
    need(m.get('schema')==SCHEMA and m.get('version')==VERSION,'E2ONLY_EXACT_EXISTING_INDEX_SCHEMA')
    request(row,central=True);rb=base64.b64decode(m['request_base64'],validate=True);saved=json.loads(rb);request(saved)
    need(e.sha(rb)==m['request_sha256']==row['request_sha256']
        and all(row.get(k)==x and type(row.get(k))is type(x)for k,x in saved.items())
        and set(row)-set(saved)<=TERMINAL_KEYS|{'request_sha256','status','job_id','submitted_epoch'},
        'E2ONLY_ORIGINAL_REQUEST_BYTES_NOT_RESTAMPED')
    need(m['predecessor_binding']==c.binding() and m['request_id']==row['request_id'] and m['source_sha']==row['source_sha']
        and m['run_manifest_sha256']==row['run_manifest_sha256'] and m['scope_grant_sha256']==authority.GRANT_SHA
        and m['physical_custody_sha256']==row['physical_custody_sha256']==e.sha(e.raw(m['physical_custody']))
        and m['native_hold']==native_resources() and m['control_paid_here']==(CONTROL if row.get('attempt_index',1)>1 else e._vector({}))
        and m['old_paid_gross_and_history_unchanged']is True,'E2ONLY_EXACT_ADOPTED_IDENTITY_HOLD_AND_PAID_HISTORY')
    position=next(i+1 for i,v in enumerate(g['PBS_submissions'])if v['request_id']==row['request_id'])
    attempt=row.get('attempt_index',1)
    need(m['before']['PBS_submissions']==position-1 and row['submit_count']==position
        and (attempt>1 or position==41) and g['used']['PBS_submissions']==len(g['PBS_submissions'])
        and position<=len(g['PBS_submissions'])<=history_upper(g,44)
        and (row['status']==e.STATES[2]or position==len(g['PBS_submissions'])),
        'E2ONLY_KEEP_EVERY_ORIGINAL_TYPED_ROW_AND_ACTUAL_COUNT')
    clock=e.authority.validate(g)
    need(all(row[k]==clock[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch'))
        and clock['T_CD']<=row['created_epoch']<=m['registered_epoch']==row['submitted_epoch']
        and m['registered_epoch']+row['reserved_H200_s']<clock['stop_new_training_epoch'],
        'E2ONLY_ORIGINAL_CLOCK_AND_FULL_SAVE_WINDOW')
    return m,row


def indices(g):
    m=g.get(FIELD);rows=[r for r in g['PBS_submissions']if r.get('platform_training_version')==VERSION]
    if m is None:
        need(not rows,'E2ONLY_ROW_REQUIRES_EXPLICIT_ORIGINAL_GLOBAL_INDEX');return []
    prior=m.get('prior_attempts',[])
    need(type(prior)is list and all(type(x)is dict and 'prior_attempts'not in x for x in prior),
        'E2ONLY_FLAT_PRESERVED_ORIGINAL_ATTEMPT_INDICES')
    sequence=prior+[m];need(len(sequence)==len(rows),'E2ONLY_ALL_OLD_INDICES_AND_ROWS_RETAINED')
    selected=[]
    for attempt,(index,row)in enumerate(zip(sequence,rows),1):
        _basic_index(g,index,row)
        need(row.get('attempt_index',1)==attempt and row['run_uuid']==c.run_name(attempt,row.get('pre_native_cold_generation',0)),
            'E2ONLY_ORDERED_SAME_RECIPE_DISTINCT_ATTEMPTS')
        if attempt>1:
            link=index.get('zero_effect_predecessor',{});previous,oldrow=selected[-1]
            need(link.get('request_id')==oldrow['request_id']and link.get('row_sha256')==e.sha(e.raw(oldrow))
                and link.get('marker_sha256')==e.sha(e.raw({k:v for k,v in previous.items()if k!='prior_attempts'}))
                and oldrow['status']==e.STATES[2]and oldrow['native_exit_status']!=0
                and oldrow['committed_optimizer_updates']==oldrow['query_attempts']==oldrow['model_load_attempts']==0
                and row.get('zero_effect_predecessor')==link
                and index['physical_custody'].get('closed_zero_predecessor')==link
                and previous['status']=='INCOMPLETE'and previous['completed_sessions']==[],
                'E2ONLY_RETRY_ONLY_REAL_CLOSED_ZERO_MODEL_QUERY_UPDATE_NOT_VALID_SCIENCE')
        selected.append((index,row))
    return selected


def basic(g,request_id=None):
    sequence=indices(g)
    if not sequence:return None
    if request_id is None:return sequence[-1]
    matches=[p for p in sequence if p[1]['request_id']==request_id]
    need(len(matches)==1,'E2ONLY_EXPLICIT_OWN_ATTEMPT_NOT_CURRENT_ALIAS');return matches[0]


@exact_ledger_validation
def validate(g,*,generation=None):
    need(generation is None,'E2ONLY_EXPLICIT_RECOVERY_NOT_ZERO_EFFECT_GENERATION')
    selected=basic(g)
    if selected is None:return None
    authority.validate(g);authority.predecessor(g)
    for m,row in indices(g):_validate_index(g,m,row)
    m,row=selected
    held=forecast_without_validate(g,m,row);old=g[e.original.PACKAGE]['remaining_resources']
    need(all(g['reserved'][k]>=old[k]+held[k]for k in e.package.TRACKED)
        and g['CPU_reserved_s']>=old['Hopper_CPU_s']+held['Hopper_CPU_s']
        and g['storage_reserved_bytes']>=old['Hopper_storage_bytes']+held['Hopper_storage_bytes'],
        'E2ONLY_RETAIN_ALL_OLD_EVAL_ESCROW_AND_NEW_COMPUTE')
    return m


def _validate_index(g,m,row):
    need(m['scope_adoption_sha256']==e.sha(e.raw(g[authority.FIELD]))
        and m['original_package_sha256']==e.sha(e.raw(g[e.package.FIELD])),
        'E2ONLY_EXACT_ADOPTED_CONTROL_AND_OLD_PACKAGE_BYTES')
    from . import minimal_e2only_physical as physical
    physical.validate_custody(m['physical_custody'],row,predecessor_binding=m['predecessor_binding'])
    physical.check_proof(m['physical_preflight'],m['physical_custody'],row,
        now_epoch=m['registered_epoch'],observed_epoch=m['physical_observed_epoch'])
    need(all(g[k]>=m['before'][k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes','optimizer_updates','generation_queries','model_loads'))
        and g['used']['H200_s']>=m['before']['H200_s'],'E2ONLY_NO_COST_REFUND_OR_ENDPOINT_ERASURE')
    if row['status']==e.STATES[2]:
        from .minimal_e2only_terminal import validate_terminal
        validate_terminal(g,row,m)
    else:need(m['status']=='REGISTERED'and m['completed_sessions']==[]and 'terminal'not in m,'E2ONLY_PREPARED_IS_NOT_COMPLETE')
    if row.get('attempt_index',1)>1:
        need(g['CPU_s']>=m['before']['CPU_s']+CONTROL['Hopper_CPU_s'],'E2ONLY_NEW_CONTROL_PAID_WITHOUT_REFUND')
    return m


def forecast_without_validate(g,m,row):
    v=previous.forecast_remaining(g)
    if row['status']!=e.STATES[2]:
        for k,x in native_resources().items():
            if k!='PBS_submissions':v[k]+=x
    return v


def forecast_remaining(g):
    return previous.forecast_remaining(g)if validate(g)is None else forecast_without_validate(g,*basic(g))


def rows(g):return []if validate(g)is None else[p[1]for p in indices(g)]
def current_id(g):return basic(g)[1]['request_id']
def terminal_CPU_prior(g,row):
    need(basic(g,row['request_id'])[1]==row,'E2ONLY_ONLY_OWN_NEW_CPU');return 0


def budget_LOCAL(g,local):return authority.budget_LOCAL(g,local)


def paid_retry_controls(g):
    items=g.get(PREPAID,{})
    need(type(items)is dict,'E2ONLY_EXPLICIT_PREPAID_RETRY_CONTROLS')
    from .pbs_count_authority import validate as unlimited
    if items:need(unlimited(g)is not None,'E2ONLY_RETRY_CONTROL_DIRECT_COUNT_AUTHORITY')
    for rid,m in items.items():
        if m.get('preparation_only')is True:
            from .minimal_e2only_pre_native import validate_payment
            validate_payment(g,m);continue
        if 'preparation_payment'in m:
            from .minimal_e2only_pre_native import validate_payment
            p=validate_payment(g,m['preparation_payment'],allow_registered=True)
            need(all(p[k]==m[k]for k in('request_id','submission_nonce','run_uuid','before_CPU_s')),
                'E2ONLY_FINAL_CONTROL_RETAINS_SAME_ORIGINAL_PREPARATION_PAYMENT')
            if p['source_sha']!=m['source_sha']:
                repair=m.get('pre_native_software_repair',{})
                need(repair.get('original_source_sha')==p['source_sha']
                    and repair.get('new_source_sha')==m['source_sha']
                    and repair.get('old_owner_CPU_upper_s')==400
                    and repair.get('new_owner_CPU_upper_s')==2600
                    and repair.get('new_native_requests')==0
                    and all(type(repair.get(k))is str and re.fullmatch('[0-9a-f]{64}',repair[k])for k in
                        ('original_paid_marker_sha256','actual_closed_owner_readback_sha256','old_manifest_sha256','old_owner_OS_sha256')),
                    'E2ONLY_EXPLICIT_UNSUBMITTED_COLD_SOFTWARE_REPAIR_NO_FEE_OR_IDENTITY_ALIAS')
                old=repair.get('original_paid_marker',{})
                need(e.sha(e.raw(old))==repair['original_paid_marker_sha256']
                    and old.get('source_sha')==repair['original_source_sha']
                    and old.get('run_manifest_sha256')==repair['old_manifest_sha256']
                    and 'pre_native_software_repair'not in old
                    and all(old.get(k)==m[k]for k in m if k not in
                        ('source_sha','run_manifest_sha256','pre_native_software_repair')),
                    'E2ONLY_ORIGINAL_BOUND_PAID_MARKER_ALL_FIELDS_PRESERVED')
        need(e._uuid(rid)and m['request_id']==rid and m['paid_control']==CONTROL
            and m['native_upper']==native_resources()and m['new_storage_charge_bytes']==0
            and m['new_GPU_permission']is False and m['Core_LOCAL_written']is False
            and g['CPU_s']>=m['before_CPU_s']+CONTROL['Hopper_CPU_s']
            and all(type(m[k])is str and re.fullmatch('[0-9a-f]{64}',m[k])for k in
                ('run_manifest_sha256','physical_custody_sha256','before_GLOBAL_sha256')),
            'E2ONLY_PAID_RETRY_EXACT_COSTS_IDENTITY_AND_HISTORY')
    return items


def require_paid_retry(g,r):
    if r.get('attempt_index',1)==1:return
    paid=paid_retry_controls(g).get(r['request_id'])
    need(paid is not None and not paid.get('preparation_only',False) and all(paid[k]==r[k]for k in
        ('request_id','submission_nonce','run_uuid','source_sha','physical_custody_sha256','zero_effect_predecessor')),
        'E2ONLY_NEW_OWNER_REQUIRES_OWN_ALREADY_PAID_CONTROL_SLICE')
    if r.get('pre_native_cold_generation'):
        from .minimal_e2only_pre_native import retired
        records=retired(g);p=r['pre_native_predecessor']
        need(records.get(p['request_id'])==p,'E2ONLY_ACTUAL_PRE_NATIVE_CLOSE_ADOPTED')
    if 'pre_native_software_repair'in paid:
        need(r.get('pre_native_owner_CPU_upper_s')==2600,'E2ONLY_REPAIRED_OWNER_CANNOT_REUSE_OLD400_CPU')


def planned_retry(g):
    held=e._vector({});registered={r['request_id']for r in g['PBS_submissions']}
    from .minimal_e2only_pre_native import retired
    closed=retired(g)
    for rid,m in paid_retry_controls(g).items():
        if rid not in registered and rid not in closed:
            for k,v in native_resources().items():held[k]+=v
    return held


def propose_retry_control(global_bytes,local_bytes,*,request_bytes,closed_zero_files,closed_readback,now_epoch):
    from scripts.pi0294.pv1.neural_ledger import check
    g=json.loads(global_bytes);r=json.loads(request_bytes);request(r);check(g)
    old,row=basic(g);zero=closed_zero_binding(g,**closed_zero_files)
    if r.get('pre_native_cold_generation'):
        from .minimal_e2only_pre_native import retired
        p=r['pre_native_predecessor']
        need(retired(g).get(p['request_id'])==p,'E2ONLY_COLD_REPLACEMENT_REQUIRES_ACTUAL_RETIRED_PREPARATION')
    prep=paid_retry_controls(g).get(r['request_id'])
    if prep is not None:
        need(prep.get('preparation_only')is True and r.get('pre_native_cold_generation')==1
            and all(prep[k]==r[k]for k in('request_id','submission_nonce','source_sha','run_uuid')),
            'E2ONLY_BIND_OWN_PAID_PREPARATION_TO_REAL_COLD_MANIFEST')
    need(r.get('attempt_index')==len(indices(g))+1 and r['zero_effect_predecessor']==zero
        and (prep is None or prep.get('preparation_only')is True)
        and all(v.get('request_id')!=r['request_id']and v.get('submission_nonce')!=r['submission_nonce']for v in g['PBS_submissions'])
        and all(v['status']==e.STATES[2]for v in g['PBS_submissions'])
        and g['pending_updates']==g['unknown_effects']==[]
        and closed_readback['GLOBAL_sha256']==e.sha(global_bytes)
        and closed_readback['closed_zero_predecessor']==zero
        and closed_readback['no_existing_new_owner_or_dispatch']is True
        and 0<=now_epoch-closed_readback['observed_epoch']<=120,
        'E2ONLY_ONLY_FRESH_COLD_ZERO_CLOSED_NO_OWNER_BEFORE_CONTROL')
    v=deepcopy(g);v['CPU_s']+=0 if prep else CONTROL['Hopper_CPU_s'];v['revision']+=1
    v.setdefault(PREPAID,{})[r['request_id']]=dict(request_id=r['request_id'],submission_nonce=r['submission_nonce'],
        run_uuid=r['run_uuid'],source_sha=r['source_sha'],run_manifest_sha256=r['run_manifest_sha256'],
        physical_custody_sha256=r['physical_custody_sha256'],zero_effect_predecessor=zero,
        paid_control=deepcopy(CONTROL),native_upper=native_resources(),before_CPU_s=prep['before_CPU_s']if prep else g['CPU_s'],
        before_GLOBAL_sha256=e.sha(global_bytes),paid_epoch=now_epoch,new_storage_charge_bytes=0,
        new_GPU_permission=False,Core_LOCAL_written=False)
    if prep:v[PREPAID][r['request_id']]['preparation_payment']=deepcopy(prep)
    check(v);future=e.scope.first512_forecast(e.raw(v),local_bytes)
    need(future['cap_changes_required']=={},'E2ONLY_ALL14_FULL_FUTURE_BEFORE_CONTROL_PAID')
    return dict(proposed_global=v,full_future_proof=future,original_onceIO_required=True,native_admission=False)


def closed_zero_binding(g,*,lease_bytes,settled_bytes,finite_OS_bytes):
    """Consume real closed files, never infer retry authority from Exit1 alone."""
    validate(g);m,row=basic(g)
    need(row['status']==e.STATES[2]and row['native_exit_status']!=0
        and row['committed_optimizer_updates']==row['query_attempts']==row['model_load_attempts']==0
        and m['status']=='INCOMPLETE'and m['completed_sessions']==[],
        'E2ONLY_ONLY_FAILED_ZERO_EFFECT_ATTEMPT_MAY_BE_REPLACED')
    lease,settled,final=map(json.loads,(lease_bytes,settled_bytes,finite_OS_bytes))
    t=m['terminal']
    need(lease['status']=='CLOSED'and lease['run_uuid']==row['run_uuid']
        and lease['manifest_sha256']==row['run_manifest_sha256']
        and lease['terminal_receipt_sha256']==e.sha(settled_bytes)
        and all(settled[k]==t[k]for k in('native','effects','audit'))
        and final['returncode']==0 and final['timeout']is False
        and final['finished_epoch']>=lease['closed_epoch']>=row['settled_epoch']
        and t['native']['all_native_workers_terminal']is True
        and t['audit']['new_physics']==0,'E2ONLY_ACTUAL_SETTLED_CLOSED_FINITE_OS0_AND_ZERO_PHYSICS')
    return dict(request_id=row['request_id'],row_sha256=e.sha(e.raw(row)),
        marker_sha256=e.sha(e.raw({k:v for k,v in m.items()if k!='prior_attempts'})),
        lease_sha256=e.sha(lease_bytes),settled_sha256=e.sha(settled_bytes),finite_OS0_sha256=e.sha(finite_OS_bytes))


def register(global_bytes,local_bytes,*,request_bytes,physical_custody,
             physical_proof,physical_observed_epoch,fresh_readback,now_epoch,closed_zero_files=None):
    from scripts.pi0294.pv1.neural_ledger import check
    from scripts.pi0294.dec1 import vanda_campaign as queue
    from scripts.pi0294.pv1.evaluation_account_jobs import checked_outside
    from .minimal_remaining_concurrency import cold_Core_rows
    from . import minimal_e2only_physical as physical
    g=json.loads(global_bytes);r=json.loads(request_bytes);request(r);check(g);authority.predecessor(g)
    from .pbs_count_authority import validate as unlimited
    earlier=indices(g);attempt=r.get('attempt_index',1)
    need(authority.validate(g)is not None and attempt==len(earlier)+1
        and (not earlier or unlimited(g)is not None)
        and g['pending_updates']==g['unknown_effects']==[]
        and all(x['status']in e.original.TERMINAL for x in g['PBS_submissions']),
        'E2ONLY_ONLY_AFTER_TRUE_CLOSED_PREDECESSOR_NO_UNKNOWN_CLEAR')
    b=c.binding();cold_Core_rows(g);zero=None
    if earlier:
        need(type(closed_zero_files)is dict and set(closed_zero_files)=={'lease_bytes','settled_bytes','finite_OS_bytes'},
            'E2ONLY_ACTUAL_CLOSED_PREDECESSOR_FILES_REQUIRED')
        zero=closed_zero_binding(g,**closed_zero_files)
        need(r.get('zero_effect_predecessor')==zero and fresh_readback.get('closed_zero_predecessor')==zero
            and physical_custody.get('closed_zero_predecessor')==zero,'E2ONLY_EXACT_PRIOR_ZERO_BINDING_IN_ALL_INPUTS')
    else:need(closed_zero_files is None and 'zero_effect_predecessor'not in r,'E2ONLY_ORIGINAL_V1_BYTES_UNCHANGED')
    actual=queue.queue_rows(fresh_readback['complete_queue'],now_epoch=now_epoch);own,outside,unknown=queue.classify(g,actual)
    checked_outside(actual,outside,base64.b64decode(fresh_readback['independent_confirmation_base64'],validate=True))
    need(own==unknown==[]and 0<=now_epoch-fresh_readback['observed_epoch']<=120
        and fresh_readback['GLOBAL_sha256']==e.sha(global_bytes)
        and fresh_readback['complete_queue_verified']is True and fresh_readback['old_workers_terminal']is True
        and fresh_readback['old_closed_science_lease_sha256']==authority.LEASE,
        'E2ONLY_FRESH_COMPLETE_NATIVE_QUEUE_CANONICAL_AND_REAL_OLD_CLOSE')
    need(all(x.get('request_id')!=r['request_id']and x.get('submission_nonce')!=r['submission_nonce']
        for x in[*g['PBS_submissions'],*g.get(e.original.CORE_ROWS,{}).values()]),'E2ONLY_NEW_UNIQUE_BOTH_SITE_IDENTITY')
    physical.validate_custody(physical_custody,r,predecessor_binding=b)
    if r.get('pre_native_cold_generation'):
        from .minimal_e2only_pre_native import retired
        p=r['pre_native_predecessor']
        need(retired(g).get(p['request_id'])==p,'E2ONLY_REGISTER_ONLY_AFTER_REAL_PRE_NATIVE_CLOSE')
    physical.check_proof(physical_proof,physical_custody,r,now_epoch=now_epoch,observed_epoch=physical_observed_epoch)
    v=deepcopy(g);v['PBS_submissions'].append(dict(r,request_sha256=e.sha(request_bytes),status=e.STATES[0],job_id=None,submitted_epoch=now_epoch));v['used']['PBS_submissions']+=1
    for k in e.package.TRACKED:v['reserved'][k]+=native_resources()[k]
    v['CPU_reserved_s']+=54000
    paid=paid_retry_controls(g).get(r['request_id'])if earlier else None
    if paid:
        need(all(paid[k]==r[k]for k in('request_id','submission_nonce','run_uuid','source_sha','run_manifest_sha256','physical_custody_sha256','zero_effect_predecessor')),
            'E2ONLY_OWN_PREPAID_CONTROL_NOT_ANOTHER_REQUEST')
    elif earlier:v['CPU_s']+=CONTROL['Hopper_CPU_s']
    before={k:g[k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes','optimizer_updates','generation_queries','model_loads')}
    before.update(H200_s=g['used']['H200_s'],PBS_submissions=g['used']['PBS_submissions'])
    if paid:before['CPU_s']=paid['before_CPU_s']
    v[FIELD]=dict(schema=SCHEMA,version=VERSION,request_id=r['request_id'],request_sha256=e.sha(request_bytes),request_base64=base64.b64encode(request_bytes).decode(),
        source_sha=r['source_sha'],run_manifest_sha256=r['run_manifest_sha256'],predecessor_binding=b,
        scope_grant_sha256=authority.GRANT_SHA,scope_adoption_sha256=e.sha(e.raw(g[authority.FIELD])),original_package_sha256=e.sha(e.raw(g[e.package.FIELD])),
        native_hold=native_resources(),control_paid_here=deepcopy(CONTROL)if earlier else e._vector({}),physical_custody=deepcopy(physical_custody),physical_custody_sha256=r['physical_custody_sha256'],
        physical_preflight=deepcopy(physical_proof),physical_observed_epoch=physical_observed_epoch,before=before,registered_epoch=now_epoch,status='REGISTERED',completed_sessions=[],old_paid_gross_and_history_unchanged=True)
    if earlier:
        v[FIELD]['prior_attempts']=[deepcopy({k:v for k,v in m.items()if k!='prior_attempts'})for m,_ in earlier]
        v[FIELD]['zero_effect_predecessor']=deepcopy(zero)
    v['revision']+=1;validate(v);check(v);future=e.scope.first512_forecast(e.raw(v),local_bytes)
    need(future['cap_changes_required']=={},'E2ONLY_COMPLETE_ALL14_MATRIX_FINAL_BEFORE_REGISTER')
    return dict(proposed_global=v,full_future_proof=future,original_onceIO_required=True,native_admission=False)


def bind(g,request_id,job_id,*,now_epoch):
    validate(g);v=deepcopy(g);_,r=basic(v)
    need(r['request_id']==request_id and r['status']==e.STATES[0]and r['job_id']is None,'E2ONLY_BIND_ONCE_ONLY')
    r.update(status=e.STATES[1],job_id=job_id,native_reply_epoch=now_epoch);v['revision']+=1
    from scripts.pi0294.pv1.neural_ledger import check
    validate(v);check(v);return v


def settle(g,request_id,**kwargs):
    from .minimal_e2only_terminal import settle as once
    return once(g,request_id,**kwargs)
