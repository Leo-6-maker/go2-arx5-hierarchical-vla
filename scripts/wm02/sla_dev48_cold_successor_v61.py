"""Finish only the original unstarted DEV48 suffix at a verified cold boundary.

Waits for the existing immutable observer,retains all original outcomes,uses
existing finite authority renewal and the original group registration/runtime.
No model retraining,condition retry,checkpoint or safety change is allowed.
"""
import argparse,copy,json,os,time
from pathlib import Path
from scripts.pi0292.durable import write
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.stable_task_scene_v1 import pinned
from go2arx5_vla.world_model.task_learning_student_registration_v2 import descriptor
from scripts.wm02.sla_queue_native_v1 import unit_state,wait_capacity,never_dispatched,CAPACITY_REASON
from scripts.wm02.sla_n20_endpoint_followup_v5 import group_terminal

SCHEMA='SLA_ORIGINAL_DEV48_UNSTARTED_COLD_SUFFIX_COORDINATOR_V6_1'


def capacity_dispatch(registered,group,journal,dispatch_group):
    try:return dispatch_group(registered['input']['path'])
    except(BlockingIOError,ValueError)as error:
        if(not isinstance(error,BlockingIOError)and str(error)!=CAPACITY_REASON)or not never_dispatched(group,journal):
            raise RuntimeError('UNKNOWN_OR_EFFECTFUL_DISPATCH_NEVER_REPLAY')from error
        raise


def continuation_reason(result):
    if result.get('status')=='COMPLETE_FIXED_TRAIN8_AND_NEW_DEV48_NOT_GUARANTEED_SUCCESS':return 'ALREADY_COMPLETE'
    if (result.get('status')=='FOLLOWUP_STOPPED_RECONCILE_NO_REPLAY'
            and 'N20_FOLLOWUP_NO_LATE_NEW_GROUP'in result.get('error','')):
        return 'FINITE_WINDOW_ONLY'
    if result.get('status')in('FINITE_PRE_DISPATCH_CAPACITY_WAIT_EXPIRED','REGISTERED_UNDISPATCHED_QUEUE_EXPIRED'):
        return 'CAPACITY_EXPIRED_REQUIRE_ABSENT_NEXT_GROUP'
    raise ValueError('PRIOR_TECHNICAL_SAFETY_OR_UNKNOWN_FAILURE_REQUIRES_RECONCILIATION')


def closed_prefix(panel_root):
    root=Path(panel_root);panel=descriptor(root/'PANEL.private.json');cursor=0;previous=None;last=None
    groups=[]
    for g in root.glob('*-group-*'):
        p=g/'INPUT.private.json'
        if not p.exists():raise ValueError('UNKNOWN_PARTIAL_GROUP_NO_RECREATE')
        inp=json.loads(p.read_bytes());groups.append((inp['assignment_slice'],g))
    for span,g in sorted(groups):
        if span[0]!=cursor or span[1]!=cursor+8:raise ValueError('ORIGINAL_CONTIGUOUS_EIGHT_CONDITION_GROUPS_REQUIRED')
        if unit_state(g)['MainPID']!='0':raise ValueError('EXISTING_GROUP_STILL_LIVE_NO_RESTART')
        good,rows=group_terminal(g,panel,span[1])
        if not good:raise ValueError('RETAIN_TECHNICAL_OR_SAFETY_NEGATIVE_NO_AUTOMATIC_SUFFIX')
        cursor=span[1];previous=descriptor(g/'RESULT.private.json');last=(g,rows[span[0]:span[1]])
    if cursor>48:raise ValueError('FIXED_DEV48_ONLY')
    return panel,cursor,previous,last


def renewal_request(spec,last):
    from scripts.wm02.renew_fusion_scope import load
    campaign=Path(spec['campaign_root']);parent,pin=load(campaign/'control/LEASE.json')
    if parent['deadline_epoch']!=spec['old_parent_deadline_epoch']:raise ValueError('AUTHORITY_CHANGED_DO_NOT_REPLACE_ANOTHER_OWNER')
    template,_=load(spec['renewal_template']['path'],spec['renewal_template']['sha256']);request=copy.deepcopy(template)
    end=spec['new_parent_deadline_epoch'];out=Path(spec['renewal_root'])
    if out.exists():raise ValueError('EXISTING_RENEWAL_IDENTITY_NO_REPLAY')
    request.update(request_id=Path(spec['renewal_root']).name.upper(),user_authorization=spec['user_authorization'],
        user_authorized_deadline_epoch=end,parent_deadline_epoch=end,fusion_deadline_epoch=end-1200,
        old_parent_lease_sha256=pin,new_pilot_root=spec['new_pilot_root'],new_fusion_root=spec['new_fusion_root'],
        old_pilot_binding=spec['old_pilot_binding'],old_fusion_binding=spec['old_fusion_binding'],
        runtime_source_root=spec['execution_source_root'],runtime_source_sha=spec['execution_source_sha'],
        evidence_files=pinned(spec['old_pilot'])['evidence_files'],registration_request_epoch=time.time(),
        successor_purpose='FINISH_ORIGINAL_UNSTARTED_DEV48_SUFFIX_NO_VALID_CONDITION_RETRY',
        renewal_module_sha256=spec['renewal_module_sha256'],latest_terminal_runs=[])
    for name in('pilot','fusion'):
        p=Path(request['old_'+name+'_binding']);request['old_'+name+'_binding_sha256']=digest(p);request['old_'+name+'_lease_sha256']=digest(p.parent/'LEASE.json')
    g,rows=last
    for row in rows:
        for tail in('','-gpu'):
            r=g/'runs'/(row['native_run']+tail);request['latest_terminal_runs'].append(dict(root=str(r),
                spec_sha256=digest(r/'EXECUTION_SPEC.json'),lease_sha256=digest(r/'control/LEASE.json'),worker_sha256=digest(r/'WORKER_EXIT.private.json')))
    out.mkdir();write(out/'REQUEST.private.json',request);return out/'REQUEST.private.json'


def run(spec):
    from scripts.wm02.renew_fusion_scope import renew
    from scripts.pi0294.e1.journal import Journal
    from go2arx5_vla.world_model.task_learning_student_registration_v2 import register_group,dispatch_group
    root=Path(spec['registered_root']);prior=Path(spec['prior_observer']['path']).parent;campaign=Path(spec['campaign_root']);start=time.time()
    if spec['schema']!=SCHEMA or os.getuid()!=2009 or os.uname().nodename!='corelab-4080s':raise ValueError('EXISTING_CORE_FINITE_SUFFIX_ONLY')
    old=pinned(spec['prior_observer']);runtime=Path(spec['execution_source_root'])
    if old['source_sha']!=spec['execution_source_sha']or digest(runtime/'scripts/wm02/renew_fusion_scope.py')!=spec['renewal_module_sha256']:
        raise ValueError('IMMUTABLE_ORIGINAL_RUNTIME_AND_RENEWAL_REQUIRED')
    def live():
        if time.time()>=spec['deadline_epoch']or(root/'STOP.json').exists()or(campaign/'STOP.json').exists():raise TimeoutError('FINITE_SUCCESSOR_STOP_NO_RESTART')
    def state(status,**fields):write(root/'PROGRESS.private.json',dict(state=status,epoch=time.time(),new_optimizer_updates=0,**fields))
    while True:
        live();handle=unit_state(prior)
        if handle['LoadState']!='loaded':raise ValueError('PRIOR_OBSERVER_HANDLE_UNKNOWN_NO_RESTART')
        lease=json.loads((prior/'control/LEASE.json').read_bytes())
        if handle['MainPID']=='0'and lease['status']=='CLOSED':break
        if handle['MainPID']=='0':raise ValueError('PRIOR_OBSERVER_EXIT_UNRECONCILED')
        state('WAIT_EXISTING_OBSERVER_ACTUAL_CLOSEOUT',new_physical_requests=0);time.sleep(120)
    result=json.loads((prior/'RESULT.private.json').read_bytes());reason=continuation_reason(result)
    panel,cursor,previous,last=closed_prefix(old['dev_panel_root'])
    if cursor==48:return dict(status='ORIGINAL_DEV48_ALREADY_COMPLETE_NO_NEW_DISPATCH',exit_code=0,DEV48_completed=48,new_physical_requests=0)
    if reason=='ALREADY_COMPLETE'or cursor==0 or last is None:raise ValueError('CONSISTENT_ACTUAL_CLOSED_DEVELOPMENT_PREFIX_REQUIRED')
    next_group=Path(old['dev_panel_root'])/(Path(old['dev_panel_root']).name+f'-group-{cursor:03d}-{cursor+8:03d}')
    if next_group.exists():raise ValueError('NEXT_GROUP_EXISTS_RECONCILE_NOT_RECREATE')
    state('CLOSED_PREFIX_VERIFIED_WAIT_OLD_AUTHORITY_EXPIRY',DEV48_retained=cursor,new_physical_requests=0,
        original_observer_result_sha256=digest(prior/'RESULT.private.json'))
    while time.time()<=spec['old_parent_deadline_epoch']:
        live();time.sleep(min(120,max(1,spec['old_parent_deadline_epoch']-time.time()+1)))
    # The existing renewal code refuses live/unclosed roles and requires all
    # original locks. There is exactly one attempt;any unknown effect stops.
    request=renewal_request(spec,last)
    preflight=renew(request,execute=False);write(root/'RENEWAL_PREFLIGHT.private.json',preflight)
    actual=renew(request,execute=True);write(root/'RENEWAL_ACTUAL.private.json',actual)
    pilot=descriptor(Path(actual['bindings']['pilot_binding']));fusion=descriptor(Path(actual['bindings']['fusion_binding']))
    journal=Journal(campaign/'control/journal',campaign.name);retained=cursor
    while cursor<48:
        live();end=cursor+8;remaining=int(spec['deadline_epoch']-time.time()-120)
        if remaining<=3900:raise TimeoutError('NO_LATE_NEW_GROUP_IN_FINITE_SUCCESSOR')
        group=Path(old['dev_panel_root'])/(Path(old['dev_panel_root']).name+f'-group-{cursor:03d}-{end:03d}')
        if group.exists():raise ValueError('ORIGINAL_NEXT_GROUP_UNKNOWN_NEVER_REPLAY')
        request=dict(root=str(group),panel=panel,assignment_slice=[cursor,end],population_predecessor=previous,
            pilot1_binding=pilot,fusion_binding=fusion,max_wall_s=min(3*3600,remaining))
        registered=register_group(request);inp=pinned(registered['input']);write(root/f'GROUP_{cursor:02d}_{end:02d}_REGISTERED.private.json',registered)
        def dispatch():
            return capacity_dispatch(registered,group,journal,dispatch_group)
        ready=wait_capacity(dispatch,min(spec['deadline_epoch'],inp['deadline_epoch']-3900),
            lambda v:state('WAIT_ORIGINAL_RESOURCE_CAPACITY',DEV48_retained=cursor,actual=v),
            stopped=lambda:(root/'STOP.json').exists()or(campaign/'STOP.json').exists())
        if not ready:raise TimeoutError('ORIGINAL_CAPACITY_WAIT_EXPIRED_NO_REPLAY')
        state('ACTUAL_UNSTARTED_SUFFIX_GROUP_DISPATCHED',begin=cursor,end=end,group=str(group))
        while unit_state(group)['MainPID']!='0':
            live();time.sleep(60)
        good,rows=group_terminal(group,panel,end)
        write(root/f'GROUP_{end:02d}_TERMINAL.private.json',dict(result=descriptor(group/'RESULT.private.json'),good=good,conditions=end))
        if not good:return dict(status='ORIGINAL_TECHNICAL_OR_SAFETY_STOP_RETAINED',exit_code=1,DEV48_completed=len(rows),new_physical_requests=len(rows)-retained)
        cursor=end;previous=descriptor(group/'RESULT.private.json')
    return dict(status='COMPLETE_ORIGINAL_DEV48_UNSTARTED_SUFFIX_NOT_TASK_SUCCESS',exit_code=0,DEV48_completed=48,
        retained_conditions=retained,new_physical_requests=48-retained,optimizer_updates=0,wall_s=time.time()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path);a=p.parse_args();print(json.dumps(run(json.loads(a.spec.read_bytes()))))
