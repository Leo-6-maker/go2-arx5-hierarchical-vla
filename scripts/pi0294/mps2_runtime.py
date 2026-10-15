"""MP-S2 adopted scope, immutable internal slots and cumulative hard budgets."""
import time
from pathlib import Path
from .task_capture_runtime import read,sha

VERSION='FULLBODY_MANISKILL_SEARCH_AND_ASSISTED_SEED_V2'
FOLDER='maniskill-search-v2'
AUTHORITY='61059c0e4488cae907c18379ab2f2bd7bbd89f9e18ec986557eac3df9d8bc43f'

def cards():
    from .task_plan import pilot_cards
    result=[]
    for phase,task in [('DEV','T2'),('DEV','T1'),('SEED','T2'),('SEED','T1')]:
        old=next(c for c in pilot_cards() if c['scene_class']=='W0' and c['task']==task)
        result.append(dict(old,id=f'MPS2_{phase}_W0_{task}',duration_s=12. if phase=='DEV' else 32.,
            teacher=VERSION,split='SEED_DEVELOPMENT',demonstration_source='AUTOMATIC_PLANNING'))
    return result

def require_adoption(root,manifest):
    root=Path(root);p=root/FOLDER;a=read(p/'HUMAN_ADOPTION.json');s=read(p/'SERIES_LEDGER.json')
    if sha(p/'HUMAN_ADOPTION.json')!=AUTHORITY or not a['authorizes_execution'] or a['budget_reset']:
        raise PermissionError('MPS2_EXPLICIT_ADOPTION_REQUIRED')
    if time.time()>=a['window']['execution_work_deadline_epoch']:raise PermissionError('MPS2_ORIGINAL_WORK_DEADLINE')
    for local,source in [('TASK_CARD.json','TASK_CARD'),('POLICY.json','SEARCH_POLICY'),('CONDITIONS.json','CONDITIONS'),('SOURCE_LOCK.json','SOURCE_LOCK')]:
        if sha(p/local)!=a['source_bound_files']['MP_SEARCH_V2_'+source+'.json']:raise PermissionError('MPS2_FIXED_CONTRACT_CHANGED')
    if (p/'SERIES_CLOSED.json').exists():raise PermissionError('MPS2_TERMINAL_PRESERVED')
    if (p/'NATIVE_SAFETY_FAIL.json').exists():raise PermissionError('MPS2_NATIVE_SAFETY_FAIL_PRESERVED')
    for rel,digest in s['old_evidence'].items():
        if sha(root/rel)!=digest:raise PermissionError('MPS2_OLD_EVIDENCE_CHANGED:'+rel)
    from .task_capture_runtime import require_task_asset
    require_task_asset(root,dict(manifest,code=s['baseline_code']),'FULLBODY_TASK_OVERLAY_V2_WORKING_STROKE',_adopted_teacher=True)
    return a,s

def require_card(root,manifest,episode):
    a,s=require_adoption(root,manifest);p=Path(root)/FOLDER
    c=next((c for c in cards() if c['id']==episode),None)
    if c is None:raise PermissionError('MPS2_UNKNOWN_CONDITION')
    cpu=read(p/'CPU_ADMISSION.json')
    if cpu['source_sha']!=manifest['code']['vla_commit'] or cpu['checks_passed']!=16:raise PermissionError('MPS2_ACTUAL_CPU_CONTRACT_REQUIRED')
    if episode.startswith('MPS2_DEV_'):
        row=next(x for x in cpu['start_results'] if x['task']==c['task'])
        if row['status']!='PASS_VALID_PLANNED_H40':raise PermissionError('MPS2_OWN_START_PATH_REQUIRED')
    else:
        mode=read(p/'MODES.json')['modes'][episode]
        if mode=='AUTO':
            q=read(p/'episodes'/('MPS2_DEV_W0_'+c['task'])/'INDEPENDENT_CPU_AUDIT.json')
            if q['status']!='PASS_SHORT_MILESTONES_AND_INTEGRITY':raise PermissionError('MPS2_OWN_SHORT_PASS_REQUIRED')
        elif mode=='ASSISTED':
            ready=read(p/'ASSISTED_READY.json')
            if not ready['actual_human_connected'] or not ready['ghost_CPU_pass']:raise PermissionError('MPS2_REAL_OPERATOR_REQUIRED')
            c=dict(c,demonstration_source='HUMAN_ASSISTED_SIM')
        else:raise PermissionError('MPS2_MODE_NOT_ADMITTED')
    return c

def reserve_bundle(root,condition,index,input_sha):
    from scripts.pi0292.durable import OwnerLock,write
    from .task_capture_runtime import validate_entry
    root=Path(root);m,parent=validate_entry(root,read_only=True);a,s=require_adoption(root,m)
    c=next(c for c in read(root/FOLDER/'CONDITIONS.json')['conditions'] if c['id']==condition)
    if not 0<=index<c['decision_slots']:raise PermissionError('MPS2_CONDITION_QUERY_CAP')
    owner=OwnerLock(parent/'R4_RESOURCE_OWNER.lock')
    try:
        path=root/FOLDER/'SEARCH_LEDGER.json';ledger=read(path);key=condition+'/'+str(index)
        if key in ledger['bundles']:
            if ledger['bundles'][key]['input_sha256']!=input_sha:raise ValueError('MPS2_FIRST_INPUT_IDENTITY')
        else:
            if len(ledger['bundles'])>=222:raise PermissionError('MPS2_BUNDLE_CAP')
            used=read(parent/'R4_RESOURCE_LEDGER.json')['Core_CPU_s']-s['initial_counters']['Core_CPU_s']
            if used+1800>=21574:raise PermissionError('MPS2_CPU_CLOSEOUT_RESERVE')
            ledger['bundles'][key]=dict(input_sha256=input_sha,started_epoch=time.time(),slots={})
            write(path,ledger)
    finally:owner.close()
    return c['random_streams'][index]

def reserve_internal(root,key,kind,slot):
    """Reserve before a native library call; incomplete slots are never redrawn."""
    from scripts.pi0292.durable import OwnerLock,write
    root=Path(root);parent=Path(read(root/'OUTPUT_BINDING.private.json')['parent_root'])
    owner=OwnerLock(parent/'R4_RESOURCE_OWNER.lock')
    try:
        path=root/FOLDER/'SEARCH_LEDGER.json';v=read(path);b=v['bundles'][key];token=kind+'/'+str(slot)
        if token in b['slots']:raise PermissionError('MPS2_FIRST_INTERNAL_SLOT_PRESERVED')
        if time.time()>=1792083759:raise PermissionError('MPS2_ORIGINAL_WORK_DEADLINE')
        from .mps2_contract import slot_cost
        cost=slot_cost(kind,slot,v['counters'],b['slots'])
        for name,n in cost.items():v['counters'][name]=v['counters'].get(name,0)+n
        b['slots'][token]=dict(state='RESERVED_NO_REDRAW',reserved_epoch=time.time(),capacity=cost)
        write(path,v)
    finally:owner.close()

def finish_internal(root,key,kind,slot,actual):
    from scripts.pi0292.durable import OwnerLock,write
    root=Path(root);parent=Path(read(root/'OUTPUT_BINDING.private.json')['parent_root'])
    owner=OwnerLock(parent/'R4_RESOURCE_OWNER.lock')
    try:
        path=root/FOLDER/'SEARCH_LEDGER.json';v=read(path);row=v['bundles'][key]['slots'][kind+'/'+str(slot)]
        if row['state']=='COMPLETE':
            if row['actual']!=actual:raise ValueError('MPS2_COMPLETED_SLOT_CHANGED')
            return
        row.update(state='COMPLETE',actual=actual,finished_epoch=time.time());write(path,v)
    finally:owner.close()
