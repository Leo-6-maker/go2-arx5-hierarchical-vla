"""Adopted V2 scope, persistent internal slots and unchanged parent budgets."""
import contextlib
from pathlib import Path
import time
from .task_capture_runtime import read,sha,validate_entry,cpu_stage

FOLDER='source-demo-retime-join-v2'
VERSION='FULLBODY_SOURCE_LOCAL_RETIME_DIRECT_JOIN_V2'
AUTHORITY='ad84276c2aa1161fe8d82b1fdd7a2c067f786d0ab99a234ff086d9d068549ff7'

def authority_values(a,digest,now):
    if digest!=AUTHORITY or not a['authorizes_execution'] or a['budget_reset']:raise PermissionError('SDR2_EXPLICIT_ADOPTION_REQUIRED')
    if now>=a['window']['execution_work_deadline_epoch']:raise PermissionError('SDR2_ORIGINAL_WORK_DEADLINE')

def goal_consumer(request,goal):
    keys=['binding','joint_qpos','root_pose','world_objects','allowed_world_pairs','pad_links','attached_cube','actual_packet_arrays_sha256']
    return dict(common={k:request.get(k) for k in keys},pose=goal['pose'],stage=goal['stage'],closed=goal.get('closed',False),seed=goal['IK_seed'],
        n_init_qpos=16,threshold=1e-3,return_closest=False)

def cards():
    from .task_plan import pilot_cards
    return [dict(next(c for c in pilot_cards() if c['scene_class']=='W0' and c['task']==t),
        id='SDR2_SEED_W0_'+t,teacher=VERSION,split='SEED_DEVELOPMENT',demonstration_source='DEMO_RETARGETED_LOCAL_RETIME_DIRECT_JOIN') for t in ('T2','T1')]

def require_adoption(root,manifest=None):
    root=Path(root);p=root/FOLDER
    if manifest is None:manifest,_=validate_entry(root,read_only=True)
    a=read(p/'HUMAN_ADOPTION.json');s=read(p/'SERIES_LEDGER.json')
    authority_values(a,sha(p/'HUMAN_ADOPTION.json'),time.time())
    for local,key in [('TASK_CARD.json','TASK_CARD'),('CONDITIONS.json','CONDITIONS'),('SOURCE_LOCK.json','SOURCE_LOCK')]:
        if sha(p/local)!=a['source_bound_files']['SOURCE_RETIME_JOIN_V2_'+key+'.json']:raise PermissionError('SDR2_FIXED_CONTRACT_CHANGED')
    if (p/'SERIES_CLOSED.json').exists():raise PermissionError('SDR2_TERMINAL_PRESERVED')
    if (p/'NATIVE_SAFETY_FAIL.json').exists():raise PermissionError('SDR2_NATIVE_SAFETY_FAIL_PRESERVED')
    for name,digest in s['old_evidence'].items():
        if sha(root/name)!=digest:raise PermissionError('SDR2_OLD_EVIDENCE_CHANGED:'+name)
    from .task_capture_runtime import require_task_asset
    require_task_asset(root,dict(manifest,code=s['baseline_code']),'FULLBODY_TASK_OVERLAY_V2_WORKING_STROKE',_adopted_teacher=True)
    return a,s

def require_card(root,manifest,episode):
    require_adoption(root,manifest);p=Path(root)/FOLDER
    card=next(c for c in cards() if c['id']==episode);proof=read(p/'CPU_ADMISSION.json')
    if proof['checks_passed']!=20:raise PermissionError('SDR2_TWENTY_ACTUAL_CPU_CHECKS_REQUIRED')
    if proof['source_sha']!=manifest['code']['vla_commit']:
        binding=read(p/'CPU_EXECUTION_BINDING.json')
        if binding['source_sha']!=manifest['code']['vla_commit'] or binding['original_CPU_sha256']!=sha(p/'CPU_ADMISSION.json') or binding['status']!='PASS_SAME_INPUT_AND_FIXED_CORE_NO_REQUERY':raise PermissionError('SDR2_CPU_SOURCE_BINDING')
    r=next(t for t in proof['task_results'] if t['task']==card['task'])
    if r['status']!='PASS_SOURCE_ROUTE_AND_TIME':raise PermissionError('SDR2_OWN_SOURCE_ROUTE_REQUIRED')
    return dict(card,locked_source_id=r['source_id'])

@contextlib.contextmanager
def cpu_scope(root,stage,*,cpu_s_cap=1200,closeout=False):
    root=Path(root);m,parent=validate_entry(root,read_only=True);a,s=require_adoption(root,m)
    ledger=read(parent/'R4_RESOURCE_LEDGER.json');used=ledger['Core_CPU_s']-s['initial_counters']['Core_CPU_s']
    if used+cpu_s_cap+(0 if closeout else 1800)>14390 or ledger['Core_CPU_s']+cpu_s_cap>36844.79211945514:raise PermissionError('SDR2_CPU_RESERVE')
    with cpu_stage(root,'SDR2_'+stage,cores=1,cpu_s_cap=cpu_s_cap):yield m

def condition(root,name):return next(c for c in read(Path(root)/FOLDER/'CONDITIONS.json')['conditions'] if c['id']==name)

def reserve_bundle(root,name,index,identity):
    from scripts.pi0292.durable import OwnerLock,write
    root=Path(root);m,parent=validate_entry(root,read_only=True);a,s=require_adoption(root,m);c=condition(root,name)
    if not 0<=index<c['decision_slots']:raise PermissionError('SDR2_CONDITION_QUERY_CAP')
    lock=OwnerLock(parent/'R4_RESOURCE_OWNER.lock')
    try:
        file=root/FOLDER/'SEARCH_LEDGER.json';v=read(file);key=name+'/'+str(index)
        if key in v['bundles']:
            if v['bundles'][key]['input_sha256']!=identity:raise ValueError('SDR2_FIRST_INPUT_CHANGED')
        else:
            if len(v['bundles'])>=164:raise PermissionError('SDR2_BUNDLE_CAP')
            if read(parent/'R4_RESOURCE_LEDGER.json')['Core_CPU_s']-s['initial_counters']['Core_CPU_s']+1800>=14390:raise PermissionError('SDR2_CPU_CLOSEOUT_RESERVE')
            v['bundles'][key]=dict(input_sha256=identity,slots={},started_epoch=time.time());write(file,v)
    finally:lock.close()
    return c

def slot_cost(kind,slot,counters,existing,c):
    if kind+'/'+str(slot) in existing:raise PermissionError('SDR2_FIRST_INTERNAL_SLOT_PRESERVED')
    ik=c.get('max_new_goal_IK_calls',c.get('max_goal_IK_calls',c.get('max_goal_IK_calls_per_decision',0)))
    rr=c.get('max_RRT_calls',c.get('max_RRT_calls_per_decision',0));dense=c.get('max_dense_windows',1)
    definitions={'IK':(ik,{'goal_IK_calls':1,'CLIK_reserved':16},{'goal_IK_calls':169,'CLIK_reserved':2704}),
        'RRT':(rr,{'RRT_calls':1},{'RRT_calls':666}),
        'DENSE':(dense,{'dense_reserved_points':1001,'local_IK_reserved':5005},{'dense_reserved_points':166166,'local_IK_reserved':830830})}
    n,cost,limits=definitions[kind]
    if not 0<=slot<n:raise PermissionError('SDR2_INTERNAL_'+kind+'_CAP')
    for k,value in cost.items():
        if counters.get(k,0)+value>limits[k]:raise PermissionError('SDR2_GLOBAL_'+k)
    return cost

def reserve_internal(root,key,kind,slot):
    from scripts.pi0292.durable import OwnerLock,write
    root=Path(root);parent=Path(read(root/'OUTPUT_BINDING.private.json')['parent_root']);lock=OwnerLock(parent/'R4_RESOURCE_OWNER.lock')
    try:
        if time.time()>=1792083759:raise PermissionError('SDR2_ORIGINAL_WORK_DEADLINE')
        file=root/FOLDER/'SEARCH_LEDGER.json';v=read(file);b=v['bundles'][key];cost=slot_cost(kind,slot,v['counters'],b['slots'],condition(root,key.split('/')[0]))
        for k,n in cost.items():v['counters'][k]=v['counters'].get(k,0)+n
        b['slots'][kind+'/'+str(slot)]=dict(state='RESERVED_NO_REDRAW',capacity=cost,reserved_epoch=time.time());write(file,v)
    finally:lock.close()

def finish_internal(root,key,kind,slot,actual):
    from scripts.pi0292.durable import OwnerLock,write
    root=Path(root);parent=Path(read(root/'OUTPUT_BINDING.private.json')['parent_root']);lock=OwnerLock(parent/'R4_RESOURCE_OWNER.lock')
    try:
        file=root/FOLDER/'SEARCH_LEDGER.json';v=read(file);r=v['bundles'][key]['slots'][kind+'/'+str(slot)]
        if r['state']=='COMPLETE':
            if r['actual']!=actual:raise ValueError('SDR2_COMPLETED_RESULT_CHANGED')
        else:r.update(state='COMPLETE',actual=actual,finished_epoch=time.time());write(file,v)
    finally:lock.close()
