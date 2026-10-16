"""R1 task/search admission using the existing registered successor and ledger."""
from pathlib import Path
from contextlib import closing
import time
from .tf_r3_runtime import validate_entry,resource_lock,cpu_scope,read,sha,CUTOFF,DEADLINE,FOLDER,VERSION
from scripts.pi0292.durable import write


def require_adoption(root,manifest=None,*,read_only=False):
    validate_entry(root,read_only=read_only)
    return read(Path(root)/'R3_CONTINUATION_ADOPTION.json'),read(Path(root)/FOLDER/'SERIES_LEDGER.json')


def condition(root,name):return read(Path(root)/FOLDER/'CONDITIONS.json')['conditions'][name]
def cards(root):return read(Path(root)/FOLDER/'CARDS.json')['cards']


def require_card(root,manifest,episode):
    validate_entry(root,physical_asset=VERSION)
    root=Path(root);card=next(c for c in cards(root) if c['id']==episode)
    release=read(root/FOLDER/'versions'/card['teacher_version']/'TEACHER_RELEASE.json')
    if (release['CPU_admission']!='PASS' or release.get('quarantined') or release['source_sha']!=manifest['code']['vla_commit']
        or card['table_top_m']!=.47 or card['initial_finger_q_m']!=.042):raise PermissionError('R1_EXACT_ADMITTED_RELEASE_AND_STARTUP_DOMAIN_REQUIRED')
    if card['role']=='PRODUCTION' and not release['tasks'][card['task']]['repeatability_PASS']:raise PermissionError('R1_TWO_OF_THREE_TASK_REPEATABILITY')
    startup=read(root/'evidence/STARTUP_QUALIFICATION.json')
    if any(r['module_sha256']!=sha(Path(__file__).with_name('night_r1_startup.py')) for r in startup['rows']):raise PermissionError('ACTUAL_STARTUP_QUALIFICATION_SOURCE_CHANGED')
    if card['scientific_condition']!='MIDPOINT_PUSH':raise PermissionError('R3_ONLY_DECLARED_MIDPOINT_PUSH')
    return card


def reserve_static(root,name,points):
    root=Path(root);validate_entry(root,read_only=True)
    with closing(resource_lock(root)):
        path=root/FOLDER/'SEARCH_LEDGER.json';v=read(path);rs=v.setdefault('static_reservations',{})
        if name in rs:raise PermissionError('R1_STATIC_FIRST_RESULT_PRESERVED')
        if v['caps']['static_or_dense_points'] is not None and sum(r['reserved_points'] for r in rs.values())+v['counters'].get('dense_reserved_points',0)+points>v['caps']['static_or_dense_points']:raise PermissionError('R1_STATIC_POINT_BUDGET')
        rs[name]=dict(reserved_points=points,status='RESERVED_NO_REDRAW');write(path,v)


def finish_static(root,name,actual,status):
    root=Path(root)
    with closing(resource_lock(root)):
        path=root/FOLDER/'SEARCH_LEDGER.json';v=read(path);r=v['static_reservations'][name]
        if r['status']!='RESERVED_NO_REDRAW' or actual>r['reserved_points']:raise ValueError('R1_STATIC_SETTLEMENT')
        r.update(status=status,actual_points=actual);write(path,v)


def reserve_bundle(root,name,index,identity):
    root=Path(root);validate_entry(root,read_only=True);c=condition(root,name)
    if time.time()>=CUTOFF or not 0<=index<c['decision_slots']:raise PermissionError('R1_FIXED_CONDITION_DECISION_OR_CUTOFF')
    with closing(resource_lock(root)):
        path=root/FOLDER/'SEARCH_LEDGER.json';v=read(path);key=name+'/'+str(index)
        if key in v['bundles']:
            if v['bundles'][key]['input_sha256']!=identity:raise ValueError('R1_FIRST_BUNDLE_INPUT_CHANGED')
        else:
            if v['caps']['bundles'] is not None and len(v['bundles'])>=v['caps']['bundles']:raise PermissionError('R1_PLANNING_BUNDLE_BUDGET')
            v['bundles'][key]=dict(input_sha256=identity,slots={},started_epoch=time.time());write(path,v)
    return c


def reserve_internal(root,key,kind,slot):
    root=Path(root);c=condition(root,key.split('/')[0])
    if time.time()>=CUTOFF:raise PermissionError('R1_WORK_CUTOFF')
    with closing(resource_lock(root)):
        path=root/FOLDER/'SEARCH_LEDGER.json';v=read(path);b=v['bundles'][key]
        if kind+'/'+str(slot) in b['slots']:raise PermissionError('R1_INTERNAL_SLOT_NO_REDRAW')
        limits={'IK':c.get('max_new_goal_IK_calls',12),'RRT':8,'DENSE':c.get('max_dense_windows',4)}
        costs={'IK':{'goal_IK_calls':1,'CLIK_reserved':16},'RRT':{'RRT_calls':1},'DENSE':{'dense_reserved_points':1101,'local_IK_reserved':5505}}
        caps=dict(goal_IK_calls=None,CLIK_reserved=v['caps']['CLIK'],RRT_calls=v['caps']['RRT'],dense_reserved_points=v['caps']['static_or_dense_points'],local_IK_reserved=None if v['caps']['static_or_dense_points'] is None else v['caps']['static_or_dense_points']*5)
        if not 0<=slot<limits[kind]:raise PermissionError('R1_PER_BUNDLE_'+kind+'_CAP')
        for k,n in costs[kind].items():
            extra=sum(r['reserved_points'] for r in v.get('static_reservations',{}).values()) if k=='dense_reserved_points' else 0
            if caps[k] is not None and v['counters'].get(k,0)+n+extra>caps[k]:raise PermissionError('R1_GLOBAL_'+k+'_CAP')
        for k,n in costs[kind].items():v['counters'][k]=v['counters'].get(k,0)+n
        b['slots'][kind+'/'+str(slot)]=dict(state='RESERVED_NO_REDRAW',capacity=costs[kind],reserved_epoch=time.time());write(path,v)


def finish_internal(root,key,kind,slot,actual):
    root=Path(root)
    with closing(resource_lock(root)):
        path=root/FOLDER/'SEARCH_LEDGER.json';v=read(path);r=v['bundles'][key]['slots'][kind+'/'+str(slot)]
        if kind=='DENSE' and (actual.get('points',0)>r['capacity']['dense_reserved_points'] or actual.get('local_IK_iterations',0)>r['capacity']['local_IK_reserved']):raise ValueError('R1_ACTUAL_DENSE_COST_EXCEEDS_RESERVATION')
        if r['state']=='COMPLETE':
            if r['actual']!=actual:raise ValueError('R1_FIRST_INTERNAL_RESULT_CHANGED')
        else:r.update(state='COMPLETE',actual=actual,finished_epoch=time.time());write(path,v)
