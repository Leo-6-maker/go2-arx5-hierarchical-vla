"""Cold resource contract for a zero-integration frozen-target scene counter.

No IO, process, renderer, model, registration or authority producer. This role
cannot run WBC, a policy, an episode, or physics. Existing global/DEC1 costs,
source settlements and final protection are copied without replacement.
"""
from copy import deepcopy
import hashlib
import json
import math
from .authority import AUTHORITY,PARENT_AUTHORITY,HARD_CLOSE

ROLE='DEC1_FROZEN_TARGET_ZERO_INTEGRATION_COUNTER_V1'
REQUEST_SCHEMA='DEC1_FROZEN_TARGET_ZERO_COUNTER_REQUEST_V1'
ROW_SCHEMA='DEC1_TARGET_ZERO_COUNTER_RESOURCE_RESERVATION_V1'
DIMS=('CoreLab_CPU_s','local_CPU_s','episode_starts','physics_ticks','constructors',
      'CoreLab_output_bytes','local_output_bytes','transfer_bytes')
REQUESTED=dict(zip(DIMS,(1200,300,0,0,2,64*1024**2,128*1024**2,256*1024**2)))
CAPS=dict(zip(DIMS,(160*3600,16*3600,136,5000000,12000,128*1024**3,8*1024**3,128*1024**3)))
VAP1_AUTHORITY='43f7fe6f86c683f2e38e5c8802eb8cde208e575c'
VAP1_SHA='a26aeb5076e075791e1dfea4e5da0ccaae05ccbb7fa578a1583dea1c4764ed89'
PROJECTION='plans/TARGET_ZERO_COUNTER_NATIVE_RESOURCE_PROJECTION.private.json'
PROTECTED='plans/EXISTING_FULL_REMAINING_BUDGET_PROTECTION.private.json'
RECOVERY_RELEASE='871094ee31b18c47183c7403459ebd5fdad88b1f'
RECOVERY_GRANT='f32922b15138816517de3e9bd63d4c413698f74b49c20908bb1d4dfea9a279a8'


def need(ok,why):
    if not ok:raise PermissionError('TARGET_ZERO_COUNTER_'+why)


def encode(v):return (json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def same(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def obj(raw):
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'DUPLICATE_JSON_KEY');d[k]=v
        return d
    need(type(raw) is bytes,'EXACT_IMMUTABLE_BYTES')
    v=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:need(False,'FINITE_JSON'))
    need(type(v) is dict,'OBJECT');return v


def vector(v):
    need(type(v) is dict and set(v)==set(DIMS),'ALL_EIGHT_RESOURCE_DIMENSIONS')
    for k,x in v.items():
        need(type(x) in (int,float) and math.isfinite(x) and x>=0
             and (k.endswith('_CPU_s') or type(x) is int),'FINITE_NONNEGATIVE_TYPED_COST')
    return v


def totals(ledger):
    total=dict.fromkeys(DIMS,0)
    for rid,r in ledger.get('target_zero_counter_reservations',{}).items():
        need(r['schema']==ROW_SCHEMA and r['scope']==ROLE and r['request_id']==rid
             and r['unknown_effects']==[] and r['status'] in ('RESERVED','SETTLED'),
             'PRIOR_TARGET_COUNTER_ROWS_KNOWN_AND_TYPED')
        left=vector(r['remaining']);wanted=vector(r['requested'])
        need(all(left[k]<=wanted[k] for k in DIMS),'OLD_HOLDS_NEVER_EXPANDED')
        need(r['status']!='SETTLED' or not any(left.values()),'SETTLED_HOLDS_ZERO')
        for k,x in left.items():total[k]+=x
    need(same(total,ledger.get('target_zero_counter_reserved',dict.fromkeys(DIMS,0))),
         'EXACT_PRIOR_TARGET_COUNTER_AGGREGATE')
    return total


def reserve(parent_bytes,Core_bytes,request_bytes,manifest_bytes,projection_bytes,VAP1_bytes,*,protected_budget_bytes,now_epoch):
    """Return proposed LOCAL/Core ledgers; existing once writer owns each CAS."""
    p,c,q,m,x=map(obj,(parent_bytes,Core_bytes,request_bytes,manifest_bytes,projection_bytes))
    need(sha(VAP1_bytes)==VAP1_SHA and c['VAP1_authority_release']==VAP1_AUTHORITY,'ACTUAL_ADOPTED_VAP1_CAPS')
    campaign='20261018T182956Z-localpc-pv1';t0=1792348196
    need(p['run_uuid']==c['campaign_run_uuid']==q['campaign_run_uuid']==campaign
         and p['T0_epoch']==c['original_T0_epoch']==q['original_T0_epoch']==t0
         and p['hard_closeout_epoch']==c['hard_close_epoch']==HARD_CLOSE,
         'ORIGINAL_CAMPAIGN_CLOCK_NOT_RESET')
    need(p['unknown_effects']==p.get('Hopper_unknown_effects',[])==p['pending_updates']==c['unknown_effects']==[],
         'NO_UNRECONCILED_ACTUAL_WRITE')
    need(c['authority_release']==q['authority_release']==AUTHORITY and q['schema']==REQUEST_SCHEMA
         and q['scope']==m['scope']==ROLE and m['schema']=='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1'
         and m['authority']['protocol_ref']==PARENT_AUTHORITY
         and m['authority']['amendments']==[AUTHORITY,VAP1_AUTHORITY],
         'EXPLICIT_NEW_ZERO_COUNTER_ROLE_UNDER_EXISTING_AUTHORITY')
    need(q['observed_epoch']<=now_epoch<q['deadline_epoch']<=t0+144*3600
         and 0<=now_epoch-q['observed_epoch']<=120
         and m['limits']['operation_deadline_epoch']==q['deadline_epoch']
         and m['limits']['PV1_T0_epoch']==t0 and m['limits']['PV1_hard_close_epoch']==HARD_CLOSE,
         'FRESH_NATIVE_RESOURCE_READBACK_AND_ORIGINAL_DEADLINE')
    vector(q['requested'])
    need(q['requested']==REQUESTED and all(type(q[k]) is int and q[k]==0 for k in
         ('GPU_submissions','training_updates','GR00T_queries','policy_episodes','physics_ticks')),
         'EXACT_TWO_CONSTRUCTORS_AND_ZERO_POLICY_PHYSICS_GPU')
    need(m['limits']['engine_starts']==2 and m['limits']['policy_episodes']==m['limits']['physics_ticks']==0,
         'NO_PHYSICS_OR_EPISODE_PERMISSION_IN_MANIFEST')
    need(sha(parent_bytes)==q['parent_before_sha256']==x['parent_before_sha256']
         and sha(Core_bytes)==q['Core_before_sha256']==x['Core_before_sha256']
         and x['schema']=='DEC1_TARGET_ZERO_COUNTER_ACTUAL_NATIVE_RESOURCE_PROJECTION_V1'
         and x['observed_epoch']==q['observed_epoch'] and x['origin']=='EXISTING_LOCAL_PC'
         and x['unknown_effects']==[],'EXACT_ACTUAL_PARENT_CORE_AND_OWNER_PROJECTION')
    need(m['execution']['sealed_inputs'][PROJECTION]==sha(projection_bytes)
         and m['execution']['sealed_inputs']['REQUEST.json']==sha(request_bytes),
         'FINAL_MANIFEST_SEALS_REQUEST_AND_FULL_PROJECTION')
    rid=q['request_id'];run=m['run_uuid'];need(rid not in p.get('target_zero_counter_reservations',{})
        and rid not in c.get('target_zero_counter_reservations',{}) and run==q['child_run_uuid']
        and run and '/' not in run and '\\' not in run,'NEW_ONCE_REQUEST_AND_CHILD')
    old=totals(c);need(same(p.get('target_zero_counter_reservations',{}),c.get('target_zero_counter_reservations',{}))
        and same(totals(p),old),'DUAL_OLD_ROWS_AND_HOLDS_IDENTICAL')
    used=vector(x['DEC1_used']);reserved=vector(x['DEC1_reserved'])
    for name,key in (('CoreLab_CPU_s','CPU_s'),('episode_starts','episode_starts'),('physics_ticks','physics_ticks'),('constructors','constructors')):
        need(used[name]==c[key],'NATIVE_CORE_COSTS_NOT_RESET_'+name)
    need(used['local_CPU_s']>=p['CPU_s']['local_PC'],'NO_STALE_LOCAL_CPU_CREDIT')
    for prior in ('source_cpu_current_classification','target_zero_counter_current_classification'):
        if prior in c:
            baseline=vector(c[prior]);need(all(used[k]>=baseline[k] for k in DIMS),'PRIOR_FULL_CLASSIFICATION_PRESERVED')
    source=vector(c.get('source_cpu_reserved',dict.fromkeys(DIMS,0)))
    need(all(reserved[k]>=source[k]+old[k] for k in DIMS),'ALL_EXISTING_SOURCE_AND_TARGET_HOLDS_PRESERVED')
    protected=obj(protected_budget_bytes)['budget_proof']
    need(protected['full_remaining_work_admissible'] is True
         and protected['all_old_spent_and_reserved_preserved'] is True
         and protected['original_T0_epoch']==t0 and protected['hard_close_epoch']==HARD_CLOSE
         and protected['zero_update_resume_release']==RECOVERY_RELEASE
         and protected['zero_update_resume_grant_sha256']==RECOVERY_GRANT
         and protected['forecast']['admissible_forecast'] is True
         and protected['forecast']['original_final_protection']['H200_s']==72000
         and protected['forecast']['original_final_protection']['Hopper_storage_bytes']==40*1024**3,
         'ORIGINAL_FUNDED_FULL_REMAINING_WORK_AND_FINAL_PROTECTION')
    need(now_epoch+300<protected['forecast']['projected_finish_epoch']
         and protected['forecast']['projected_finish_epoch']+300<=protected['forecast']['clock']['stop_new_jobs_epoch']==t0+144*3600,
         'ORIGINAL_FULL_PLAN_AND_COUNTER_WALL_HEADROOM_NO_CLOCK_RESET')
    forecast=protected['forecast']['future_required_additional']
    mapping=dict(CoreLab_CPU_s='Core_CPU_s',local_CPU_s='local_CPU_s',episode_starts='episode_starts',
        physics_ticks='physics_ticks',constructors='engine_starts',CoreLab_output_bytes='Core_storage_bytes',
        local_output_bytes='local_storage_bytes',transfer_bytes='transfer_bytes')
    remaining=vector({k:forecast[v] for k,v in mapping.items()})
    need(x['protected_remaining']==remaining and x['protected_budget_sha256']==sha(protected_budget_bytes)
         and m['execution']['sealed_inputs'][PROTECTED]==sha(protected_budget_bytes),
         'SEALED_UNREDUCED_EXISTING_FUTURE_FORECAST_NO_FREED_B1_CREDIT')
    need(all(used[k]+reserved[k]+REQUESTED[k]+remaining[k]<=CAPS[k] for k in DIMS),'DEC1_NONADDITIVE_CAPS')
    floor=p['final_reserve_inside_total'];need(floor['H200_s']==72000 and floor['episode_starts']==100
         and floor['live_ticks']==floor['shadow_ticks']==1600000,'ORIGINAL_FINAL_RESERVE_INTACT')
    parent_used=dict(CoreLab_CPU_s=p['CPU_s']['CoreLab'],local_CPU_s=p['CPU_s']['local_PC'],
        episode_starts=p['episode_starts'],physics_ticks=sum(p['physics_ticks'].values()),constructors=p['engine_starts'])
    children=p['children'];pending=dict(CoreLab_CPU_s=p['CPU_reserved_s']['CoreLab'],local_CPU_s=p['CPU_reserved_s']['local_PC'],
        episode_starts=sum(a.get('episode_starts_reserved',0) for a in children),
        physics_ticks=sum(a.get('physics_ticks_reserved',0) for a in children),
        constructors=sum(a.get('engine_starts_reserved',0) for a in children))
    original=dict(CoreLab_CPU_s=320*3600,local_CPU_s=48*3600,episode_starts=900,physics_ticks=36800000,constructors=100000)
    for k in original:need(parent_used[k]+pending[k]+REQUESTED[k]+remaining[k]<=original[k],'PARENT_AND_PROTECTED_FINAL_CAP_'+k)
    need(all(REQUESTED[k]<=min(vector(x['site_remaining'])[k],vector(x['parent_remaining'])[k]) for k in DIMS),
         'ALL_NATIVE_SITE_AND_PARENT_DIMENSIONS')
    need(x['site_permission']=='EXISTING_REGISTERED_CORELAB_4080_SIMULATION'
         and type(x['concurrent_CoreLab_cores']) is int and 1<=x['concurrent_CoreLab_cores']<=2,
         'EXISTING_SITE_AND_TWO_CORE_BOUND')
    row=dict(schema=ROW_SCHEMA,scope=ROLE,request_id=rid,authority_release=AUTHORITY,
        campaign_run_uuid=campaign,child_run_uuid=run,source_sha=m['code']['vla_commit'],manifest_sha256=sha(manifest_bytes),
        request_sha256=sha(request_bytes),projection_sha256=sha(projection_bytes),
        requested=deepcopy(REQUESTED),remaining=deepcopy(REQUESTED),status='RESERVED',
        unknown_effects=[],deadline_epoch=q['deadline_epoch'])
    outputs=[]
    for ledger in (p,c):
        out=deepcopy(ledger);out.setdefault('target_zero_counter_reservations',{})[rid]=deepcopy(row)
        out['target_zero_counter_reserved']={k:old[k]+REQUESTED[k] for k in DIMS};outputs.append(out)
    after,core=outputs
    after['CPU_reserved_s']['CoreLab']+=REQUESTED['CoreLab_CPU_s'];after['CPU_reserved_s']['local_PC']+=REQUESTED['local_CPU_s']
    after['children'].append(dict(id=run,request_id=rid,kind=ROLE,status='RESERVED',
        CPU_reserved_s=REQUESTED['CoreLab_CPU_s'],local_CPU_reserved_s=REQUESTED['local_CPU_s'],
        episode_starts_reserved=0,physics_ticks_reserved=0,engine_starts_reserved=2,unknown_effects=[]))
    core['target_zero_counter_current_classification']=deepcopy(used)
    return dict(status='PURE_TARGET_COUNTER_PROPOSAL_NOT_NATIVE_ADMISSION',reservation=row,
        parent_before_sha256=sha(parent_bytes),Core_before_sha256=sha(Core_bytes),
        proposed_parent=after,proposed_Core=core,native_dispatch_admitted=False)

POPULATION_SHA='288be5e878baf1b112190a8ac69c1b507336ca03f550ae9b4f4672f13ef90db1'
FROZEN_SOURCES={
 'scripts/pi0294/task_plant.py':'558732a93d9cf6dae5146bf82876e62fb4efd7d7a26e9fe3d9c132586a7b39f4',
 'scripts/pi0294/task_contacts.py':'522dff09d62ed52145e947fa86a65a8dc21b4382a455b7c2bed46185b3a33888',
 'scripts/pi0294/task_gripper_contract.py':'ccb3eb07388586b43c9deb8068fd42204648163ec1a523c8590dfac4b23220e8'}


def plan(population_bytes):
    need(sha(population_bytes)==POPULATION_SHA,'ORIGINAL_SEALED_TRAIN_POPULATION')
    p=obj(population_bytes);groups=[g for g in p['groups'] if g['group_id']=='PI0294/PV1/TRAIN/F0/000']
    need(len(groups)==1 and groups[0]['split']=='TRAIN'
         and p['status']=='SEALED_STATIC_QUALIFIED_BEFORE_TASK_RESULTS'
         and p['evaluation_success_screening'] is False,'EXACT_EXISTING_TRAIN_GEOMETRY_NOT_SUCCESS_SELECTION')
    g=groups[0]
    card=dict(id='DEC1-TARGET-ZERO-COUNTER-01',task='T1',split='TRAIN',
        source_xy=deepcopy(g['source_xy']),goal_xy=deepcopy(g['goal_xy']),cube_yaw=g['cube_yaw'],
        source_group_id=g['group_id'],source_geometry_sha256=sha(encode(g)),
        domain='FROZEN_FULLBODY_WBC_TASK_SIM',purpose='ZERO_TIME_ASSET_COUNTER_ONLY_NO_POLICY_OR_EPISODE')
    return dict(schema='DEC1_FROZEN_TARGET_ZERO_COUNTER_PLAN_V1',scope=ROLE,population_sha256=POPULATION_SHA,
        card=card,model_constructor_attempts=2,episodes=0,physics_ticks=0,GR00T_queries=0,
        WBC_or_leg_policy_forwards=0,target_task_native_qualified=False,new_training_data=False)


def settle(parent_bytes,Core_bytes,request_id,*,result_bytes,actor_proof_bytes):
    """Actual terminal actor proof required; no process observation or IO here.

    Missing result keeps the full financial upper and an UNKNOWN scientific
    verdict. It never creates a counter PASS or an episode/physics permission.
    """
    p,c=map(obj,(parent_bytes,Core_bytes));proof=obj(actor_proof_bytes)
    rows=p['target_zero_counter_reservations'];need(same(rows,c['target_zero_counter_reservations']), 'DUAL_TARGET_ROWS_BEFORE_SETTLEMENT')
    row=rows[request_id];need(row['status']=='RESERVED' and row['remaining']==row['requested']==REQUESTED
        and row['unknown_effects']==[],'ONE_UNSETTLED_COUNTER_REQUEST')
    need(proof['schema']=='DEC1_TARGET_ZERO_COUNTER_ACTUAL_TERMINAL_ACTOR_PROOF_V1'
        and proof['request_id']==request_id and proof['manifest_sha256']==row['manifest_sha256']
        and proof['actor_terminal'] is True and proof['current_matching_actors']==0
        and type(proof['pid']) is int and proof['pid']>0 and str(proof['proc_start']).isdigit(),
        'INDEPENDENT_ACTUAL_PID_START_EXIT_AND_NO_LIVE_ACTOR')
    need(p['unknown_effects']==p.get('Hopper_unknown_effects',[])==c['unknown_effects']==[], 'NO_UNKNOWN_METADATA_WRITE')
    total=totals(c);need(same(totals(p),total),'DUAL_FULL_PENDING_IDENTICAL')
    amount=deepcopy(REQUESTED)
    if result_bytes is None:
        need(proof.get('result_sha256') is None and proof.get('result_present') is False,'ACTUAL_RESULT_ABSENCE_NOT_DROPPED_KNOWN_EVIDENCE')
        verdict='COUNTER_ENGINEERING_INCOMPLETE_UNKNOWN_FULL_UPPER_CHARGED';result_hash=None
    else:
        result=obj(result_bytes);result_hash=sha(result_bytes)
        need(proof.get('result_present') is True,'ACTUAL_RESULT_PRESENT')
        need(result['request_id']==request_id and result['manifest_sha256']==row['manifest_sha256']
            and proof['result_sha256']==result_hash
            and result['scope']==ROLE and result['target_task_native_qualified'] is False
            and all(type(result[k]) is int and result[k]==0 for k in
                ('physics_ticks','policy_episodes','WBC_or_leg_policy_forwards','GR00T_queries')),
            'EXACT_ORIGINAL_ZERO_TIME_RESULT_NOT_A_TASK_SUCCESS')
        n=result['constructor_attempts'];cpu=result['actual_worker_CPU_s']
        need(type(n) is int and 0<=n<=2 and type(cpu) in (int,float) and math.isfinite(cpu) and 0<=cpu<=600,
            'BOUNDED_ACTUAL_CONSTRUCTOR_ATTEMPTS_AND_CPU')
        if result['status']=='PASS_FROZEN_FULLBODY_TASK_ZERO_INTEGRATION_ASSET_COUNTER_ONLY':
            need(n==2 and result['integration_attempts']==[] and result['nq']==34 and result['nv']==32 and result['nu']==20
                 and set(result['images'])=={'body_rgb','wrist_rgb'}
                 and result['frozen_robot']['status']=='COMPILED_NATIVE_PROPERTIES_EXACTLY_PRESERVED',
                 'COMPLETE_ORIGINAL_ZERO_TIME_COUNTER_PASS_EVIDENCE')
        amount['constructors']=n;amount['CoreLab_CPU_s']=600+cpu
        verdict=('ZERO_TIME_ASSET_COUNTER_PASS_NOT_NATIVE_TASK_QUALIFICATION'
            if result['status']=='PASS_FROZEN_FULLBODY_TASK_ZERO_INTEGRATION_ASSET_COUNTER_ONLY'
            else 'KNOWN_ZERO_TIME_COUNTER_ENGINEERING_FAILURE')
    sid=request_id+'-target-zero-counter-terminal'
    record=dict(schema='DEC1_TARGET_ZERO_COUNTER_TERMINAL_SETTLEMENT_V1',settlement_id=sid,
        scope=ROLE,request_id=request_id,manifest_sha256=row['manifest_sha256'],source_sha=row['source_sha'],
        result_sha256=result_hash,actor_proof_sha256=sha(actor_proof_bytes),scientific_verdict=verdict,
        financial_charged=amount,target_task_native_qualified=False,core64_cells=0,
        no_new_training_data=True,original_source_failures_and_costs_preserved=True)
    out=[]
    for ledger in (p,c):
        v=deepcopy(ledger);v['target_zero_counter_reservations'][request_id].update(
            status='SETTLED',remaining=dict.fromkeys(DIMS,0),settlement_sha256=sha(encode(record)))
        v['target_zero_counter_reserved']={k:total[k]-REQUESTED[k] for k in DIMS}
        v.setdefault('target_zero_counter_settlements',{})[request_id]=deepcopy(record);out.append(v)
    after,core=out
    after['CPU_reserved_s']['CoreLab']-=REQUESTED['CoreLab_CPU_s'];after['CPU_reserved_s']['local_PC']-=REQUESTED['local_CPU_s']
    need(min(after['CPU_reserved_s'].values())>=0,'NO_RELEASE_OF_OTHER_CPU_HOLDS')
    after['CPU_s']['CoreLab']+=amount['CoreLab_CPU_s'];after['CPU_s']['local_PC']+=amount['local_CPU_s']
    after['engine_starts']+=amount['constructors'];after['network_transfer_accounted_bytes']+=amount['transfer_bytes']
    matches=[x for x in after['children'] if x.get('request_id')==request_id and x.get('kind')==ROLE]
    need(len(matches)==1 and matches[0]['status']=='RESERVED','EXACT_ORIGINAL_GLOBAL_CHILD')
    matches[0].update(status='CLOSED_TARGET_ZERO_COUNTER_EFFECTS_SETTLED',CPU_reserved_s=0,
        local_CPU_reserved_s=0,episode_starts_reserved=0,physics_ticks_reserved=0,engine_starts_reserved=0)
    core['CPU_s']+=amount['CoreLab_CPU_s'];core['constructors']+=amount['constructors']
    baseline=vector(core['target_zero_counter_current_classification'])
    core['target_zero_counter_current_classification']={k:baseline[k]+amount[k] for k in DIMS}
    return dict(status='PURE_TARGET_ZERO_COUNTER_SETTLEMENT_PROPOSAL_NOT_NATIVE_CLOSURE',
        settlement=record,proposed_parent=after,proposed_Core=core,
        actual_once_Core_LOCAL_and_child_lease_closure_still_required=True,native_dispatch_admitted=False)
