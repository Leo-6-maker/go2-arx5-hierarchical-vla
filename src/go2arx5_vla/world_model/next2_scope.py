"""NEXT2 finite scope through the original AUTO1 registry/Journal/locks."""
from pathlib import Path
import json,time,os
from .storage import digest
from .next1_scope import C08
PLAN_SHA='4c1e4cb83a1cd5e83d4849e2c0e84d31daef0910'


def validate_limits(b,*,now):
    expected=dict(schema='WM02_NEXT2_FINITE_WINDOW_V1',plan_sha=PLAN_SHA,owner='EXISTING_LOCAL_PC',host='corelab-4080s',uid=2009,scope='WM02_NEXT2_SIM_ONLY',C08_native_binary_sha256=C08,max_native_attempts=96,max_native_ticks=600000,max_native_slots=2,max_task_atoms=400,max_main_updates=1600,max_restore_probes=4,max_neural_recipes=2)
    if any(b.get(k)!=v for k,v in expected.items())or not b['activation_epoch']<=now<b['dispatch_cutoff_epoch']<b['deadline_epoch']or b['deadline_epoch']>min(b['activation_epoch']+21600,b['parent_campaign_deadline_epoch'])or b['deadline_epoch']-b['dispatch_cutoff_epoch']<1200:
        raise PermissionError('EXACT_LIVE_NEXT2_FINITE_SCOPE_REQUIRED')


def usage(campaign,binding):
    n=t=m=p=0
    for root in(Path(campaign)/'runs').glob('next2-*'):
        f=root/'EXECUTION_SPEC.json'
        if not f.exists():continue
        s=json.loads(f.read_text())
        if s.get('next2_binding')!=str(binding):continue
        if s.get('profile')=='wm02-auto1-target':
            n+=1;f=root/'native-01/RESULT.json';t+=json.loads(f.read_text())['actual_target_integrations']if f.exists()else s['max_ticks']
        elif s.get('profile')=='wm02-auto1-target-neural':
            f=root/'fit-01/RESULT.json'
            if f.exists():v=json.loads(f.read_text());m+=v['main_updates'];p+=v['continuation_probe_updates']
            else:m+=s['endpoint'];p+=1
    return dict(attempts=n,native_ticks_or_reservations=t,main_updates_or_reservations=m,restore_probes_or_reservations=p)


def validate(spec,registry,*,now=None):
    path=Path(spec['next2_binding']);b=json.loads(path.read_text());validate_limits(b,now=time.time()if now is None else now);lease=json.loads((path.parent/'LEASE.json').read_text())
    if digest(path)!=spec['next2_binding_sha256']or b['campaign_root']!=spec['campaign_root']or lease.get('status')!='ACTIVE'or lease.get('binding_sha256')!=digest(path)or lease.get('owner')!=b['owner']or lease.get('deadline_epoch')!=b['deadline_epoch']or spec['deadline_epoch']>b['deadline_epoch']or spec.get('next1_binding')or spec.get('night1_binding')or(path.parent/'STOP.json').exists():raise PermissionError('NEW_NEXT2_ACTIVE_LEASE_REQUIRED_NO_CLOSED_SCOPE_REUSE')
    if not any(e.get('scope')==b['scope']and e.get('path')==str(path.parent.parent)and e.get('local_lease')==str(path.parent/'LEASE.json')and e.get('owner')==b['owner']for e in registry['entries']):raise PermissionError('ACTUAL_NEXT2_WINDOW_REGISTRATION_REQUIRED')
    if not any(e.get('path')==spec['registered_root']and e.get('source_sha')==spec['source_sha']and e.get('next2_binding')==str(path)for e in registry['entries']):raise PermissionError('ACTUAL_NEXT2_WORKER_REGISTRATION_REQUIRED')
    u=usage(spec['campaign_root'],path)
    if u['attempts']>96 or u['native_ticks_or_reservations']>600000 or u['main_updates_or_reservations']>1600 or u['restore_probes_or_reservations']>4:raise PermissionError('NEXT2_FINITE_ENDPOINTS_EXHAUSTED')
    if digest(spec['native_binary'])!=C08:raise PermissionError('NEXT2_FROZEN_C08_BINARY_REQUIRED')
    return b


def register(binding):
    from contextlib import closing
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.night10.native_relay import snapshot,once
    path=Path(binding);b=json.loads(path.read_text());validate_limits(b,now=time.time());c=Path(b['campaign_root']);w=path.parent.parent
    if os.uname().nodename!=b['host']or os.getuid()!=b['uid']:raise PermissionError('ACTUAL_NEXT2_HOST_OWNER_REQUIRED')
    with closing(OwnerLock(c/'OWNER.lock')),closing(OwnerLock(b['physical_owner_lock'])):
        parent=json.loads((c/'control/LEASE.json').read_text())
        if parent.get('status')!='ACTIVE'or parent.get('scope')!='WM02_AUTO1_SIM_ONLY'or parent.get('owner')!=b['owner']or parent['deadline_epoch']!=b['parent_campaign_deadline_epoch']:raise PermissionError('ACTUAL_ORIGINAL_PARENT_SCOPE_REQUIRED')
        if(path.parent/'LEASE.json').exists():raise FileExistsError('EXISTING_OR_UNKNOWN_NEXT2_WINDOW_MUST_BE_RECONCILED')
        registry,pin=snapshot(b['registry'])
        for entry in registry['entries']:
            if entry.get('scope')in('WM02_NEXT1_SIM_ONLY','WM02_NEXT2_SIM_ONLY'):
                f=Path(entry['local_lease'])
                if not f.exists()or json.loads(f.read_text()).get('status')!='CLOSED':raise PermissionError('PRIOR_RESEARCH_WINDOW_NOT_CLOSED')
        for root in(c/'runs').iterdir():
            if root.name.startswith(('night1-','next1-','next2-'))and(root/'control/LEASE.json').exists()and json.loads((root/'control/LEASE.json').read_text())['status']!='CLOSED':raise PermissionError('PRIOR_RESEARCH_WORKER_ACTIVE_OR_UNKNOWN')
        if not any(e.get('path')==str(c)and e.get('scope')=='WM02_AUTO1_SIM_ONLY'and e.get('owner')==b['owner']for e in registry['entries']):raise PermissionError('ORIGINAL_OWNER_NOT_REGISTERED')
        profiles=json.loads(Path(b['environments']).read_text())['profiles']
        if not any(p['name']=='4080-simulation'and b['host']in p['hostnames']and p['python']==b['runtime_python']for p in profiles):raise PermissionError('EXISTING_SITE_PROFILE_REQUIRED')
        for name,pin_expected in b['evidence_files'].items():
            if digest(name)!=pin_expected:raise PermissionError('SEALED_NEXT2_INPUT_CHANGED:'+name)
        registry['entries'].append(dict(role='WM02_NEXT2_WINDOW_20261105_01',path=str(w),run_uuid=w.name,executor_node=b['host'],owner=b['owner'],scope=b['scope'],source_sha=b['source_sha'],local_lease=str(path.parent/'LEASE.json'),plan_sha=PLAN_SHA));once(Path(b['registry']),registry,'WM02-NEXT2-WINDOW-20261105-01-REGISTER',expected_before_sha256=pin)
        write(path.parent/'LEASE.json',dict(status='ACTIVE',owner=b['owner'],scope=b['scope'],run_uuid=w.name,binding_sha256=digest(path),activated_epoch=b['activation_epoch'],deadline_epoch=b['deadline_epoch']))
        state,statepin=snapshot(c/'AUTO1_STATE.json')
        if state['owner']!=b['owner']:raise PermissionError('CANONICAL_OWNER_CHANGED')
        state.update(current_phase='NEXT2_REGISTERED_OFFLINE_AUDITS',updated_epoch=time.time());state['next2']=dict(status='REGISTERED_NOT_NEW_NATIVE_QUALIFIED',plan_sha=PLAN_SHA,binding=str(path),deadline_epoch=b['deadline_epoch'],native_attempts=0,native_ticks=0,main_updates=0,restore_probes=0,Actor_updates=0,next='M0 fairness/ARX and C0 first exit audit');once(c/'AUTO1_STATE.json',state,'WM02-NEXT2-WINDOW-20261105-01-INDEX',expected_before_sha256=statepin)
        v=dict(status='NEXT2_ACTUAL_SCOPE_REGISTERED',binding_sha256=digest(path),owner=b['owner'],deadline_epoch=b['deadline_epoch'],dispatch_cutoff_epoch=b['dispatch_cutoff_epoch'],old_closed_leases_reopened=False,new_integrations=0,new_optimizer_updates=0);write(w/'REGISTERED.private.json',v);return v
