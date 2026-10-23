"""Complete cold CPU reservation view for original matrix SDK preparation.

The old TRAIN-only forecaster is unchanged. SDK CPU holds are displayed as
committed uppers in its arithmetic view, then credited against the SAME matrix
CPU work, once. Actual spent/held vectors are retained separately, never
reported as measured use or as a legacy native-view PASS. No GPU/physics/
queries, constructor or IO credit. Full matrix rows/ceilings remain unchanged.
"""
from copy import deepcopy
from . import target_zero_counter_contract as common
from . import future_budget as original
from .target_sdk_client import SDK_ROLE, CAMPAIGN, T0, HARD_CLOSE, STOP_NEW

ROLE=SDK_ROLE
ROWS='target_sdk_client_reservations'
SETTLEMENTS='target_sdk_client_settlements'
ROW_SCHEMA='DEC1_TARGET_SDK_CPU_RESERVATION_V1'
WORK='DEC1_CORE_MATRIX_AND_COMMISSIONING'
COMPONENT='ORIGINAL_TARGET_SDK_COLD_BINDING_V1'
# Three finite engineering attempts inside the SAME original200000/3600 matrix
# CPU work; never an additive global cap or a new physics/episode allowance.
MAX_ATTEMPTS=3
CPU_CEILING=dict(CoreLab_CPU_s=10800,local_CPU_s=360)
REQUESTED=dict(CoreLab_CPU_s=3600,local_CPU_s=120,episode_starts=0,physics_ticks=0,
               constructors=2,CoreLab_output_bytes=512*1024**2,
               local_output_bytes=64*1024**2,transfer_bytes=64*1024**2)
NAMES=('pv1-20261019','dec1-20261021','vap1-20261021','science-next',
       'dec1-source-component-01','dec1-source-component-02',
       'dec1-target-zero-counter-01','dec1-target-zero-counter-02')
MAP=dict(CoreLab_CPU_s='Core_CPU_s',local_CPU_s='local_CPU_s',constructors='engine_starts',
         CoreLab_output_bytes='Core_storage_bytes',transfer_bytes='transfer_bytes')
def need(ok,why):
    if not ok:raise PermissionError('DEC1_SDK_BUDGET_'+why)


def rows(parent,core):
    """Only actual paired rows or an explicit proposed pair; never one side."""
    a=parent.get(ROWS,{});b=core.get(ROWS,{})
    need(common.same(a,b),'SDK_EXACT_PAIRED_ROWS')
    need(len(a)<=MAX_ATTEMPTS,'SDK_THREE_FINITE_SAME_WORK_ENGINEERING_ATTEMPTS')
    spent=dict.fromkeys(common.DIMS,0);held=dict.fromkeys(common.DIMS,0)
    for rid,row in a.items():
        wanted=common.vector(row['requested']);left=common.vector(row['remaining'])
        need(all(wanted[k]<=REQUESTED[k] for k in common.DIMS),'SDK_ORIGINAL_EIGHT_DIMENSION_COMPONENT_UPPER')
        need(row['schema']==ROW_SCHEMA and row['scope']==ROLE and row['request_id']==rid
             and row['funded_work_id']==WORK and row['component_id']==COMPONENT
             and row['unknown_effects']==[] and row['status'] in ('RESERVED','SETTLED')
             and wanted['episode_starts']==wanted['physics_ticks']==0,
             'SDK_TYPED_KNOWN_ZERO_PHYSICS_COMPONENT')
        if row['status']=='RESERVED':
            need(left==wanted,'SDK_FULL_HOLD_NOT_PARTIALLY_DROPPED')
            amount=left;destination=held
        else:
            need(not any(left.values()),'SDK_SETTLED_HOLDS_ZERO')
            s=parent[SETTLEMENTS][rid]
            need(common.same(s,core[SETTLEMENTS][rid]) and common.sha(common.encode(s))==row['settlement_sha256']
                 and s['schema']=='DEC1_TARGET_SDK_CPU_SETTLEMENT_V1' and s['request_id']==rid
                 and s['scope']==ROLE and s['manifest_sha256']==row['manifest_sha256']
                 and s['target_task_native_qualified'] is False,'SDK_EXACT_PAIRED_SETTLEMENT')
            amount=common.vector(s['financial_charged']);destination=spent
            need(all(amount[k]<=wanted[k] for k in common.DIMS),'SDK_NO_UNRESERVED_COST')
        for k in common.DIMS:destination[k]+=amount[k]
    need(sum(x['status']=='RESERVED' for x in a.values())<=1,'SDK_ONE_CURRENT_CPU_OWNER')
    for k,cap in CPU_CEILING.items():
        need(spent[k]+held[k]<=cap,'SDK_SINGLE_MATRIX_CPU_COMPONENT_CEILING_'+k)
    for ledger in (parent,core):
        need(common.same(ledger.get('target_sdk_client_reserved',dict.fromkeys(common.DIMS,0)),held),
             'SDK_FULL_EIGHT_DIMENSION_HOLD_AGGREGATE')
    need(core['CPU_s']>=spent['CoreLab_CPU_s'] and parent['CPU_s']['local_PC']>=spent['local_CPU_s']
         and core['constructors']>=spent['constructors']
         and parent['network_transfer_accounted_bytes']>=spent['transfer_bytes'],
         'SDK_CREDIT_MUST_ALREADY_BE_IN_ACTUAL_TOTALS')
    return a,spent,held


def namespace(proof,*,requested_local_bytes,now_epoch):
    need(proof['schema']=='DEC1_SDK_ACTUAL_INCLUSIVE_LOCAL_NAMESPACES_V1'
         and proof['complete_existing_namespace_set']==list(NAMES)
         and type(proof['SDK_namespace']) is str and proof['SDK_namespace'] not in NAMES
         and proof['SDK_namespace'] and '/' not in proof['SDK_namespace'] and '\\' not in proof['SDK_namespace']
         and set(proof['allocated_bytes_by_namespace'])==set(NAMES)|{proof['SDK_namespace']}
         and 0<=now_epoch-proof['observed_epoch']<120
         and proof['observed_epoch']<=now_epoch,'SDK_FRESH_COMPLETE_LOCAL_POOL_NOT_FAKE_ZERO_OUTPUT')
    common.sha(common.encode(proof))
    for x in proof['allocated_bytes_by_namespace'].values():
        need(type(x) is int and x>=0,'SDK_ACTUAL_ALLOCATED_BYTES')
    # Preserve the ORIGINAL 1GiB matrix local output plus all three64MiB
    # controller output ceilings. They are not erased to fit this component.
    future=proof['protected_future_additional_upper_bytes']
    need(type(future) is int and future>=1024**3+3*64*1024**2,'SDK_ALL_ORIGINAL_FUTURE_LOCAL_OUTPUT_PROTECTED')
    need(sum(proof['allocated_bytes_by_namespace'].values())+future+requested_local_bytes+512*1024**2
         <8*1024**3,'SDK_NEW_NAMESPACE_INSIDE_INCLUSIVE_EIGHT_GIB_POOL')


def prepare(snapshot,plan,parent_bytes,Core_bytes,*,namespace_proof,now_epoch):
    """Preserve the full actual view and construct ONLY an arithmetic view."""
    p,c=map(common.obj,(parent_bytes,Core_bytes));a,spent,held=rows(p,c)
    need(p['run_uuid']==c['campaign_run_uuid']==snapshot['campaign_run_uuid']==CAMPAIGN
         and c['authority_release']==snapshot['authority_release']==common.AUTHORITY
         and c['VAP1_authority_release']==common.VAP1_AUTHORITY
         and p['T0_epoch']==c['original_T0_epoch']==snapshot['T0_epoch']==T0
         and p['hard_closeout_epoch']==c['hard_close_epoch']==HARD_CLOSE
         and p['unknown_effects']==p.get('Hopper_unknown_effects',[])==p['pending_updates']==c['unknown_effects']==[],
         'SDK_ORIGINAL_CAMPAIGN_AND_NO_UNKNOWN_WRITE')
    need(not snapshot.get('target_sdk_component_arithmetic_view') and 'SDK_matrix_funding_component' not in plan,
         'SDK_ONE_COMPONENT_OVERLAY_NO_SECOND_CREDIT')
    need(not any(common.vector(c.get('source_cpu_reserved',dict.fromkeys(common.DIMS,0))).values())
         and not any(common.vector(c.get('target_zero_counter_reserved',dict.fromkeys(common.DIMS,0))).values()),
         'SDK_OTHER_CORE_PENDING_NEEDS_COMPLETE_SEPARATE_ADAPTER')
    need(p['CPU_reserved_s']['CoreLab']==held['CoreLab_CPU_s']
         and p['CPU_reserved_s']['local_PC']==held['local_CPU_s'],'SDK_NO_HIDDEN_OTHER_CPU_HOLD')
    need(c['CPU_s']<=p['CPU_s']['CoreLab'] and c['constructors']<=p['engine_starts'],
         'SDK_ACTUAL_NATIVE_COSTS_NOT_OMITTED')
    namespace(namespace_proof,requested_local_bytes=held['local_output_bytes'],now_epoch=now_epoch)
    s=deepcopy(snapshot);q=deepcopy(plan)
    matrix=q['remaining_matrix_envelope']
    need(matrix['core_cells']==64 and matrix['commissioning_upper']==8
         and matrix['engineering_replacements_upper']==16 and matrix['valid_failure_replays']==0
         and matrix['resources']['Core_CPU_s']>=200000 and matrix['resources']['local_CPU_s']>=3600,
         'SDK_ORIGINAL_MATRIX_COMPONENT_NOT_A_NEW_TRAINING_OR_DATA_SCOPE')
    need(snapshot['reserved'].get('Core_CPU_s',0)==snapshot['reserved'].get('local_CPU_s',0)==0
         and snapshot['dec1_reserved'].get('Core_CPU_s',0)==snapshot['dec1_reserved'].get('local_CPU_s',0)==0,
         'SDK_LEGACY_TRAIN_CREDIT_VIEW_NOT_DROPPING_OTHER_CORE_HOLDS')
    for label in ('used','dec1_used'):
        need(s[label]['Core_CPU_s']<=p['CPU_s']['CoreLab'] and s[label]['local_CPU_s']<=p['CPU_s']['local_PC'],
             'SDK_NO_RESET_OF_CONSERVATIVE_OLD_CPU_CLASSIFICATION')
        s[label]['Core_CPU_s']=p['CPU_s']['CoreLab']+held['CoreLab_CPU_s']
        s[label]['local_CPU_s']=p['CPU_s']['local_PC']+held['local_CPU_s']
        s[label]['Core_storage_bytes']+=spent['CoreLab_output_bytes']+held['CoreLab_output_bytes']
    s['used']['engine_starts']=p['engine_starts']+held['constructors']
    s['dec1_used']['engine_starts']+=spent['constructors']+held['constructors']
    s['used']['transfer_bytes']=p['network_transfer_accounted_bytes']+held['transfer_bytes']
    s['dec1_used']['transfer_bytes']+=spent['transfer_bytes']+held['transfer_bytes']
    for label in ('used','dec1_used'):
        need(s[label]['local_storage_bytes']==8*1024**3,'SDK_EXISTING_INCLUSIVE_POOL_UNCHANGED')
    # Increasing used Core output must NOT silently shrink the old future
    # lifecycle. Keep the whole original future Core output envelope, too.
    extra_output=spent['CoreLab_output_bytes']+held['CoreLab_output_bytes']
    q['storage_lifecycle']['new_output_peak_upper_bytes']['Core_storage_bytes']+=extra_output
    for label,peaks in (('used','storage_peaks'),('dec1_used','dec1_storage_peaks')):
        s[peaks]['Core_storage_bytes']=max(s[peaks]['Core_storage_bytes'],s[label]['Core_storage_bytes'])
    credit={MAP[k]:spent[k]+held[k] for k in CPU_CEILING}
    q['SDK_matrix_funding_component']=dict(work_id=WORK,component_id=COMPONENT,CPU_credit=credit)
    s['target_sdk_component_arithmetic_view']=dict(
        schema='DEC1_SDK_COMPLETE_CPU_COMMITMENT_ARITHMETIC_VIEW_V1',
        parent_sha256=common.sha(parent_bytes),Core_sha256=common.sha(Core_bytes),
        actual_spent=spent,actual_reserved=held,requests=list(a),CPU_credit=credit,
        pending_shown_as_committed_uppers_not_measured_use=True,
        original_actual_snapshot=deepcopy(snapshot),native_dispatch_admitted=False)
    return s,q


def forecast(*,snapshot,plan,parent_bytes,Core_bytes,namespace_proof,now_epoch,**original_inputs):
    s,q=prepare(snapshot,plan,parent_bytes,Core_bytes,namespace_proof=namespace_proof,now_epoch=now_epoch)
    value=original.forecast(snapshot=s,plan=q,now_epoch=now_epoch,**original_inputs)
    # Preserve the original forecast, including STOPs and UNKNOWN task-cost
    # feasibility. Recompute only CPU budget overages for the once-funded CPU
    # component. Neither native legacy view nor target science becomes PASS.
    before=deepcopy(value);credit=s['target_sdk_component_arithmetic_view']['CPU_credit']
    for key,amount in credit.items():
        need(value['future_required_additional'][key]>=amount,'SDK_CPU_COMPONENT_INSIDE_REMAINING_MATRIX')
        value['future_required_additional'][key]-=amount
    for key in credit:
        g=s['used'][key]+s['reserved'].get(key,0)+value['future_required_additional'][key]+value['original_final_protection'].get(key,0)-value['effective_global_caps'][key]
        d=s['dec1_used'][key]+s['dec1_reserved'].get(key,0)+value['future_required_additional'][key]-value['effective_DEC1_VAP1_caps'][key]
        for output,amount in (('global_shortfalls',g),('DEC1_shortfalls',d)):
            value[output].pop(key,None)
            if amount>0:value[output][key]=amount
    # The original sequential wall bound remains, plus an additional finite
    # CPU preparation interval: no optimistic overlap or shrinking job walls.
    value['projected_finish_epoch']+=1800
    if value['projected_finish_epoch']>=value['clock']['hard_closeout_epoch']:
        value['clock_failures'].append('SDK_CPU_PREPARATION_HARD_CLOSE')
    if value['projected_finish_epoch']-q['package_closeout_wall_s']>=value['clock']['stop_physics_epoch']:
        value['clock_failures'].append('SDK_CPU_PREPARATION_PHYSICS_CUTOFF')
    if now_epoch+1800>=STOP_NEW:value['clock_failures'].append('SDK_CPU_PREPARATION_NEW_JOB_CUTOFF')
    value['admissible_forecast']=not value['global_shortfalls'] and not value['DEC1_shortfalls'] and not value['clock_failures']
    value['status']='PASS_SDK_SAME_MATRIX_CPU_FUNDING_NOT_NATIVE_ADMISSION' if value['admissible_forecast'] else 'STOP_RESOURCE_OR_CLOCK'
    value['SDK_funding']=dict(component_id=COMPONENT,work_id=WORK,CPU_credit=credit,
        exact_parent_sha256=common.sha(parent_bytes),exact_Core_sha256=common.sha(Core_bytes),
        full_original_forecast_before_component_credit=before,
        actual_spent_and_held=s['target_sdk_component_arithmetic_view'],
        matrix_rows_and_all_original_controller_ceilings_unchanged=True,
        constructor_IO_GPU_physics_query_or_episode_credit=False,native_dispatch_admitted=False)
    return value
