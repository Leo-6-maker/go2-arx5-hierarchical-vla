"""Full byte-bound C/D premodel resource join, using the existing dimensions.

Pure validation. Native collection/persistence belongs to the original owner,
process reader and onceIO; this result grants neither model nor physics GO.
"""
import base64,json
from . import mix_accounting as a, mix_core_accounting as c, mix_core_identity as n
from . import mix_core_context as context, mix_evaluation as plan, successor_cd as s
from .evaluation_admission import resource_floor, adopted_caps
from .future_budget import _vector, FINAL
from .evaluation_work import MATRIX
from .dataset import POPULATION

SNAPSHOT='V5_CD_CORE_FRESH_DUAL_RESOURCE_SNAPSHOT_V1'

def need(ok,why):
    if not ok:raise PermissionError('V5_CD_CORE_PREMODEL_'+why)

def load(b):
    need(type(b)is bytes,'CAPTURED_BYTES_NOT_A_PASS_SUMMARY');return json.loads(b)

def adjusted_full_dimension(global_ledger,key,full,used,reserved):
    """Keep actual floors and every finite cap when joining the full forecast."""
    total=full['projected']+max(0,used-full['used'])+max(0,reserved-full['reserved'])
    cap=full['cap']
    if cap is None:
        from scripts.pi0294.generalization.pbs_count_authority import validate
        need(key=='PBS_submissions'and validate(global_ledger)is not None
             and full.get('PBS_submission_count_limit','MISSING')is None
             and full.get('count_still_accounted')is True,
             'ONLY_EXACT_HUMAN_ADOPTED_PBS_COUNT_IS_UNLIMITED')
        shortfall=0
    else:shortfall=max(0,total-cap)
    return dict(full,projected=total,shortfall=shortfall)

def recompute(*,manifest_bytes,request_bytes,global_bytes,local_bytes,snapshot_bytes,
              training_capture_bytes,population_bytes,plan_bytes,prior_bytes,
              native_facts_bytes,now_epoch):
    m,r,g,l,snap=map(load,(manifest_bytes,request_bytes,global_bytes,local_bytes,snapshot_bytes))
    n.request_identity(m,r,a.sha(manifest_bytes));c.checked(g);c.paired(g,l)
    marker=s.validated(g);now=a.number(now_epoch);arm=r['evaluation_candidate']
    need(marker['T_CD']==r['T_CD'] and marker['deadline_epoch']==r['deadline_epoch']
         and marker['T_CD']<=now<marker['deadline_epoch'],'ONE_ACTUAL_SUCCESSOR_CLOCK')
    need(snap.get('schema')==SNAPSHOT and snap.get('task')==s.TASK
         and snap.get('scope_grant_sha256')==s.GRANT_SHA
         and snap.get('T_CD')==marker['T_CD'] and snap.get('deadline_epoch')==marker['deadline_epoch']
         and snap.get('GLOBAL_sha256')==a.sha(global_bytes)
         and snap.get('LOCAL_sha256')==a.sha(local_bytes)
         and type(snap.get('observed_epoch'))in(int,float)
         and 0<=now-snap['observed_epoch']<=120
         and snap.get('heartbeat_only_raw_ledger')is False,'FRESH_FULL_RAW_DUAL_SNAPSHOT_NOT_OLD_HEARTBEAT')
    from scripts.pi0294.generalization.minimal_concurrency import permit_Core,validate_LOCAL
    permit_Core(g,r,m)
    validate_LOCAL(g,l)
    row=g.get(c.FIELD,{}).get(r['request_id'])
    need(type(row)is dict and row['status']in c.STATES[1:3]
         and row['request_sha256']==a.sha(request_bytes)
         and row['run_manifest_sha256']==a.sha(manifest_bytes)
         and row['child_run_uuid']==r['run_uuid']
         and all(row.get(k)==r.get(k)for k in ('source_sha','submission_nonce','evaluation_candidate',
             'checkpoint_manifest_sha256','statistics_sha256','GPU_uuid','planned_panel_sha256',
             'training_handoff_sha256','cold_input_audit_sha256')),'ACTUAL_BOUND_SAME_CURRENT_REQUEST')
    need(sum(x['status']!=c.STATES[3]for x in g[c.FIELD].values())==1,'ONE_TASK_NATIVE_GPU_OWNER')
    sealed=m['execution']['sealed_inputs']
    for name,b in [(context.CAPTURE,training_capture_bytes),('plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json',population_bytes),
                   ('plans/SOURCE_EVALUATION_PLAN.json',plan_bytes),('plans/PRIOR_EVALUATION_OUTCOMES.json',prior_bytes)]:
        need(sealed.get(name)==a.sha(b),'SEALED_ACTUAL_MEMBER:'+name)
    captured=load(training_capture_bytes);proof=context.captured_endpoint(captured)
    need(proof['arm']==arm and proof['checkpoint_manifest_sha256']==r['checkpoint_manifest_sha256']
         and a.sha(a.raw(proof))==r['training_handoff_sha256'],'SAME_CLOSED_1024_NOT_AN_OLD_AB_ENDPOINT')
    terminal={k:base64.b64decode(v,validate=True)for k,v in captured['terminal_capture_base64'].items()}
    c.closed_training(g,l,terminal,arm)
    need(a.sha(population_bytes)==POPULATION,'EXACT_ORIGINAL_POPULATION_BYTES')
    expected=plan.candidate_plan(load(population_bytes)['groups'],arm)
    need(load(plan_bytes)==expected and r['planned_panel_sha256']==a.sha(plan_bytes),
         'ALL_TWENTY_FOUR_ORIGINAL_CONDITIONS_NOT_A_SELECTED_SUBSET')
    from .evaluation_plan import remaining_panel
    from .mix_core_partial_resume import envelope,KEY,PRIOR_SHA,record_binding
    limits=envelope(r);record_binding(g,row)
    if KEY in r:need(a.sha(prior_bytes)==PRIOR_SHA,'EXACT_ORIGINAL_AUDITED_PRIOR_ELEVEN')
    need(len(remaining_panel(load(prior_bytes),expected['panels'][arm+'-DEV-N294101']))==limits['remaining_conditions'],
         'VALID_POLICY_FAILURES_CANNOT_BE_REPEATED')
    process=load(native_facts_bytes);start=a.number(process['started_epoch'])
    allocation={k:r[k]for k in ('source_sha','request_id','submission_nonce','stage','evaluation_candidate',
        'checkpoint_manifest_sha256','statistics_sha256','native_platform','allocation_number')}
    allocation.update(status=n.STATUS,platform_evaluation_version=c.VERSION,run_uuid=r['run_uuid'],
        manifest_sha256=a.sha(manifest_bytes),request_sha256=a.sha(request_bytes),job_id=process['native_handle'],
        native_handle=process['native_handle'],native_process=process,native_process_sha256=c.digest(process),
        hostname=process['hostname'],allocation_start_epoch=start,
        allocation_end_epoch=min(start+r['walltime_s'],r['deadline_epoch']),model_loaded=False,new_optimizer_updates=0)
    n.sealed_identity(m,r,allocation,run_uuid=r['run_uuid'],manifest_sha256=a.sha(manifest_bytes),
        request_sha256=a.sha(request_bytes),same_bound_process=row)
    need(0<=now-process['observed_epoch']<=120 and process['observed_epoch']>=start
         and all(process.get(k)==row.get(v)for k,v in [('native_handle','native_handle'),('boot_id','boot_id'),
             ('pid','pid'),('proc_start','proc_start'),('started_epoch','process_started_epoch')]),
         'FRESH_RAW_PROCESS_AND_ACTUAL_GLOBAL_BIND_NO_PREPARED_PID')
    floor,hold=resource_floor(g,l);used,reserved=_vector(snap['used']),_vector(snap['reserved'])
    need(all(used[k]>=floor[k]and reserved[k]>=hold[k]for k in floor),'NO_COUNTER_HOLD_OR_HISTORICAL_COST_UNDERSTATEMENT')
    need(type(snap.get('Core_storage_upper_bytes'))is int and snap['Core_storage_upper_bytes']>0
         and type(snap.get('LOCAL_storage_upper_bytes'))is int and snap['LOCAL_storage_upper_bytes']>0
         and used['Core_storage_bytes']>=snap['Core_storage_upper_bytes']
         and used['local_storage_bytes']>=snap['LOCAL_storage_upper_bytes'],
         'COMPLETE_ORIGINAL_NATIVE_STORAGE_CENSUS_UPPER')
    future=_vector(a.package_view(g)['remaining_resources']);caps=adopted_caps(g)
    credit={k:0 for k in future}
    for k in ('H200_s','optimizer_updates','neural_queries','Hopper_CPU_s','Hopper_storage_bytes','Core_CPU_s','local_CPU_s'):
        credit[k]=min(reserved[k],future[k])
    dimensions={}
    for k in caps:
        total=used[k]+reserved[k]+future[k]-credit[k]+MATRIX[k]+FINAL.get(k,0)
        dimensions[k]=dict(cap=caps[k],used=used[k],reserved=reserved[k],remaining_package=future[k],
            exact_existing_package_credit=credit[k],matrix=MATRIX[k],final=FINAL.get(k,0),
            projected=total,shortfall=max(0,total-caps[k]))
    from . import mix_core_recovery as recovered
    recovery_future=None
    if recovered.validate(g) is not None:
        from scripts.pi0294.generalization.minimal_scope import first512_forecast
        recovery_future=first512_forecast(global_bytes,local_bytes)
        # The complete forecast retains GEN1 and both minimal-adaptation arms.
        # A larger actual snapshot can only increase its paid/reserved floors.
        for k,full in recovery_future['dimensions'].items():
            dimensions[k]=adjusted_full_dimension(g,k,full,used[k],reserved[k])
    shortfalls={k:v['shortfall']for k,v in dimensions.items()if v['shortfall']}
    need(not shortfalls,'ALL_ORIGINAL_REMAINING_DIMENSIONS_AND_FINAL_PROTECTED:'+str(shortfalls))
    current_end=start+limits['reserved_GPU_s']
    if arm=='C50':
        need(current_end<marker['stop_new_training_epoch'] and current_end+22560+28860<marker['deadline_epoch'],
             'REMAINING_D_TRAIN_AND_BOTH_PANELS_INSIDE_ORIGINAL_SCOPE')
    else:need(current_end<marker['deadline_epoch'],'LAST_PANEL_CLOSE_INSIDE_ORIGINAL_SCOPE')
    need(now<allocation['allocation_end_epoch']-2700,'ORIGINAL_FORTY_FIVE_MINUTE_NATIVE_CLOSEOUT_RESERVE')
    return dict(schema='V5_CD_CORE_FULL_REMAINING_NATIVE_RESOURCE_RECOMPUTATION_V1',
        request_id=r['request_id'],request_sha256=a.sha(request_bytes),GLOBAL_sha256=a.sha(global_bytes),
        LOCAL_sha256=a.sha(local_bytes),snapshot_sha256=a.sha(snapshot_bytes),native_process_sha256=c.digest(process),
        training_handoff_sha256=r['training_handoff_sha256'],dimensions=dimensions,shortfalls={},
        recovered_full_campaign_future=recovery_future,
        all_remaining_costs_fit=True,allocation=allocation,original_matrix_and_final_protected=True,
        native_or_scientific_admission=False,model_permission=False,physics_authorized=False,new_native_writes=0)
