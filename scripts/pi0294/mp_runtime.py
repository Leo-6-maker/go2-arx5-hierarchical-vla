"""Explicitly adopted, bounded ManiSkill component scope; old series stay closed."""
from pathlib import Path
import time
from .task_capture_runtime import read,sha
from .task_plan import pilot_cards

VERSION='FULLBODY_MANISKILL_COMPONENT_SEED_V1'
FOLDER='maniskill-component-v1'
AUTHORITY='8f40a73e594e2340c835b4699fd2c090b4c4b00b4c7e5e1c610c72af20053344'


def cards():
    result=[]
    for kind,task in (('DEV','T2'),('DEV','T1'),('FULL','T2'),('FULL','T1')):
        old=next(c for c in pilot_cards() if c['scene_class']=='W0' and c['task']==task)
        result.append(dict(old,id='MP_'+kind+'_W0_'+task,duration_s=12. if kind=='DEV' else 32.,
            split='TEACHER_DEVELOPMENT' if kind=='DEV' else 'TEACHER_CONFIRMATION',teacher=VERSION))
    return result


def require_adoption(root,manifest):
    root=Path(root);p=root/FOLDER;a=read(p/'HUMAN_ADOPTION.json');s=read(p/'SERIES_LEDGER.json')
    if sha(p/'HUMAN_ADOPTION.json')!=AUTHORITY or not a['authorizes_execution'] or a['budget_reset']:
        raise PermissionError('EXPLICIT_FIXED_MANISKILL_ADOPTION_REQUIRED')
    for local,source in (('TASK_CARD.json','MANISKILL_PLANNING_COMPONENT_TASK_CARD_V1.json'),
            ('FIXED_CONDITIONS.json','MANISKILL_PLANNER_FIXED_CONDITIONS_V1.json'),('SOURCE_LOCK.json','MANISKILL_PLANNER_SOURCE_LOCK_V1.json')):
        if sha(p/local)!=a['source_bound_files'][source]:raise PermissionError('ADOPTED_MANISKILL_CARD_CHANGED')
    if time.time()>=a['execution_work_deadline_epoch']:raise PermissionError('MANISKILL_ORIGINAL_WORK_DEADLINE')
    if (p/'SERIES_CLOSED.json').exists():raise PermissionError('MANISKILL_SERIES_TERMINAL_PRESERVED')
    if (p/'NATIVE_SAFETY_FAIL.json').exists():raise PermissionError('MANISKILL_NATIVE_SAFETY_FAIL_PRESERVED')
    for path,key in (('TASK_PILOT_CLOSED.json','old_terminal_sha256'),('NATIVE_SAFETY_FAIL.json','old_native_safety_sha256'),
            ('teacher-qualification-v1/SERIES_CLOSED.json','old_V1_terminal_sha256'),
            ('teacher-qualification-v2/SERIES_CLOSED.json','old_V2_terminal_sha256')):
        if sha(root/path)!=s[key]:raise PermissionError('MANISKILL_OLD_EVIDENCE_CHANGED')
    from .task_capture_runtime import require_task_asset
    baseline=dict(manifest,code=s['baseline_code'])
    require_task_asset(root,baseline,'FULLBODY_TASK_OVERLAY_V2_WORKING_STROKE',_adopted_teacher=True)
    return a,s


def require_card(root,manifest,episode):
    a,s=require_adoption(root,manifest);p=Path(root)/FOLDER
    card=next((c for c in cards() if c['id']==episode),None)
    if card is None:raise PermissionError('UNKNOWN_FIXED_MANISKILL_CONDITION')
    proof=read(p/'CPU_ADMISSION.json')
    if proof['status']!='PASS_CPU_CONTRACT_AND_START_PATHS' or proof['source_sha']!=manifest['code']['vla_commit']:
        raise PermissionError('ACTUAL_MANISKILL_SOURCE_CPU_ADMISSION_REQUIRED')
    if episode.startswith('MP_FULL_'):
        q=read(p/'episodes'/('MP_DEV_W0_'+card['task'])/'INDEPENDENT_CPU_AUDIT.json')
        if q['status']!='PASS_SHORT_MILESTONES_AND_INTEGRITY':raise PermissionError('OWN_TASK_SHORT_PASS_REQUIRED')
    return card


def reserve_request(root,condition,index,*,cpu=False):
    from scripts.pi0292.durable import OwnerLock,write
    from .task_capture_runtime import validate_entry
    root=Path(root);m,parent=validate_entry(root,read_only=True);a,s=require_adoption(root,m)
    plan=read(root/FOLDER/'FIXED_CONDITIONS.json')
    entries=plan['CPU_saved_input_conditions'] if cpu else plan['conditions']
    c=next(c for c in entries if c.get('id',c.get('condition_id'))==condition)
    item=next(v for v in c['planning_seed_table'] if v['query_index']==index)
    with OwnerLock(parent/'R4_RESOURCE_OWNER.lock'):
        p=root/FOLDER/'PLANNING_REQUESTS.json';r=read(p) if p.exists() else dict(requests=[])
        key=condition+'/'+str(index)
        if any(v['key']==key for v in r['requests']):raise PermissionError('FIRST_PLANNING_REQUEST_PRESERVED')
        if len(r['requests'])>=232:raise PermissionError('MANISKILL_REQUEST_BUDGET')
        r['requests'].append(dict(key=key,cpu=cpu,seed=item['seed'],started_epoch=time.time()))
        write(p,r)
    return item['seed']
