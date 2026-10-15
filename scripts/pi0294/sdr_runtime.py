"""Adopted source-demo scope; immutable prior failures and cumulative budgets."""
import contextlib
from pathlib import Path
import time
from .task_capture_runtime import read,sha,validate_entry,cpu_stage as original_cpu_stage

FOLDER='source-demo-retarget-v1'
VERSION='FULLBODY_SOURCE_DEMO_RETARGET_SEED_V1'
AUTHORITY='fca7e07d9eac87c9d27ff549199d7aa8494b7c63e0744ef4ff0be0bb10262059'

def cards():
    from .task_plan import pilot_cards
    return [dict(next(c for c in pilot_cards() if c['scene_class']=='W0' and c['task']==t),
                 id='SDR_SEED_W0_'+t,teacher=VERSION,split='SEED_DEVELOPMENT',
                 demonstration_source='DEMO_RETARGETED_SOURCE') for t in ['T2','T1']]

def require_adoption(root,manifest=None):
    root=Path(root);p=root/FOLDER
    if manifest is None:manifest,parent=validate_entry(root,read_only=True)
    a=read(p/'HUMAN_ADOPTION.json');s=read(p/'SERIES_LEDGER.json')
    if sha(p/'HUMAN_ADOPTION.json')!=AUTHORITY or not a['authorizes_execution'] or a['budget_reset']:
        raise PermissionError('SDR_EXPLICIT_ADOPTION_REQUIRED')
    if time.time()>=a['window']['execution_work_deadline_epoch']:raise PermissionError('SDR_ORIGINAL_DEADLINE')
    for local,source in [('TASK_CARD.json','TASK_CARD'),('SOURCE_PLAN.json','SOURCE_PLAN'),('SOURCE_LOCK.json','SOURCE_LOCK'),('CONDITIONS.json','CONDITIONS')]:
        if sha(p/local)!=a['source_bound_files']['SOURCE_DEMO_RETARGET_'+source+'_V1.json']:raise PermissionError('SDR_FIXED_CONTRACT_CHANGED')
    if (p/'SERIES_CLOSED.json').exists():raise PermissionError('SDR_TERMINAL_PRESERVED')
    if (p/'NATIVE_SAFETY_FAIL.json').exists():raise PermissionError('SDR_NATIVE_SAFETY_FAIL_PRESERVED')
    for path,digest in s['old_evidence'].items():
        if sha(root/path)!=digest:raise PermissionError('SDR_OLD_EVIDENCE_CHANGED:'+path)
    from .task_capture_runtime import require_task_asset
    require_task_asset(root,dict(manifest,code=s['baseline_code']),'FULLBODY_TASK_OVERLAY_V2_WORKING_STROKE',_adopted_teacher=True)
    return a,s

def require_card(root,manifest,episode):
    require_adoption(root,manifest);p=Path(root)/FOLDER
    card=next((r for r in cards() if r['id']==episode),None)
    if card is None:raise PermissionError('SDR_UNKNOWN_EPISODE')
    proof=read(p/'CPU_ADMISSION.json')
    if proof['checks_passed']!=16 or proof['source_sha']!=manifest['code']['vla_commit']:raise PermissionError('SDR_ACTUAL_SOURCE_BOUND_CPU_REQUIRED')
    result=next(r for r in proof['task_results'] if r['task']==card['task'])
    if result['status']!='PASS_SOURCE_ROUTE_AND_TIME':raise PermissionError('SDR_OWN_SOURCE_ROUTE_REQUIRED')
    return dict(card,locked_source_id=result['source_id'])

@contextlib.contextmanager
def cpu_scope(root,stage,*,cpu_s_cap=1200,closeout=False):
    root=Path(root);manifest,parent=validate_entry(root,read_only=True);a,s=require_adoption(root,manifest)
    ledger=read(parent/'R4_RESOURCE_LEDGER.json');used=ledger['Core_CPU_s']-s['initial_counters']['Core_CPU_s']
    reserve=0 if closeout else 1800
    if used+cpu_s_cap+reserve>14400 or ledger['Core_CPU_s']+cpu_s_cap>36844.79211945514:raise PermissionError('SDR_CPU_RESERVE')
    with original_cpu_stage(root,'SDR_'+stage,cores=1,cpu_s_cap=cpu_s_cap):yield manifest

def reserve_bundle(root,condition,index,input_sha):
    from scripts.pi0292.durable import OwnerLock,write
    root=Path(root);m,parent=validate_entry(root,read_only=True);a,s=require_adoption(root,m)
    condition_row=next(c for c in read(root/FOLDER/'CONDITIONS.json')['conditions'] if c['id']==condition)
    if not 0<=index<condition_row['decision_slots']:raise PermissionError('SDR_DECISION_CAP')
    owner=OwnerLock(parent/'R4_RESOURCE_OWNER.lock')
    try:
        path=root/FOLDER/'SEARCH_LEDGER.json';v=read(path);key=condition+'/'+str(index)
        if key in v['bundles']:
            if v['bundles'][key]['input_sha256']!=input_sha:raise ValueError('SDR_FIRST_INPUT_CHANGED')
        else:
            if len(v['bundles'])>=168:raise PermissionError('SDR_BUNDLE_CAP')
            if read(parent/'R4_RESOURCE_LEDGER.json')['Core_CPU_s']-s['initial_counters']['Core_CPU_s']+1800>=14400:raise PermissionError('SDR_CPU_CLOSEOUT_RESERVE')
            v['bundles'][key]=dict(input_sha256=input_sha,started_epoch=time.time(),slots={});write(path,v)
    finally:owner.close()
    return condition_row['random_streams'][index]

def slot_cost(kind,slot,counters,existing=()):
    if kind+'/'+str(slot) in existing:raise PermissionError('SDR_FIRST_INTERNAL_SLOT_PRESERVED')
    definitions={'IK':(12,dict(goal_IK_calls=1,CLIK_reserved=16),dict(goal_IK_calls=2016,CLIK_reserved=32256)),
        'RRT':(8,dict(RRT_calls=1),dict(RRT_calls=1344)),
        'DENSE':(2,dict(dense_reserved_points=1001),dict(dense_reserved_points=174174))}
    n,cost,limits=definitions[kind]
    if not 0<=slot<n:raise PermissionError('SDR_INTERNAL_SLOT_CAP')
    for k,value in cost.items():
        if counters.get(k,0)+value>limits[k]:raise PermissionError('SDR_GLOBAL_'+k)
    return cost

def reserve_internal(root,key,kind,slot):
    from scripts.pi0292.durable import OwnerLock,write
    root=Path(root);parent=Path(read(root/'OUTPUT_BINDING.private.json')['parent_root']);owner=OwnerLock(parent/'R4_RESOURCE_OWNER.lock')
    try:
        if time.time()>=1792083759:raise PermissionError('SDR_ORIGINAL_DEADLINE')
        path=root/FOLDER/'SEARCH_LEDGER.json';v=read(path);b=v['bundles'][key]
        if kind=='DENSE' and not key.startswith('SDR_CPU_') and slot>=1:raise PermissionError('SDR_ONE_DENSE_WINDOW_PER_LIVE_BUNDLE')
        cost=slot_cost(kind,slot,v['counters'],b['slots'])
        for name,n in cost.items():v['counters'][name]=v['counters'].get(name,0)+n
        b['slots'][kind+'/'+str(slot)]=dict(state='RESERVED_NO_REDRAW',reserved_epoch=time.time(),capacity=cost);write(path,v)
    finally:owner.close()

def finish_internal(root,key,kind,slot,actual):
    from scripts.pi0292.durable import OwnerLock,write
    root=Path(root);parent=Path(read(root/'OUTPUT_BINDING.private.json')['parent_root']);owner=OwnerLock(parent/'R4_RESOURCE_OWNER.lock')
    try:
        path=root/FOLDER/'SEARCH_LEDGER.json';v=read(path);row=v['bundles'][key]['slots'][kind+'/'+str(slot)]
        if row['state']=='COMPLETE':
            if row['actual']!=actual:raise ValueError('SDR_COMPLETED_RESULT_CHANGED')
            return
        row.update(state='COMPLETE',actual=actual,finished_epoch=time.time());write(path,v)
    finally:owner.close()
