"""Finite NEXT1 window under the existing AUTO1 owner/registry/Journal."""
from pathlib import Path
import json,time,os
from .storage import digest

PLAN_SHA='a673d4b0d361c6fea9fcd5d5b9cbd28f58461b4f'
C08='29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2'


def validate_limits(b,*,now):
    if (b.get('schema')!='WM02_NEXT1_FINITE_WINDOW_V1' or b.get('plan_sha')!=PLAN_SHA
            or b.get('owner')!='EXISTING_LOCAL_PC' or b.get('host')!='corelab-4080s' or b.get('uid')!=2009
            or b.get('scope')!='WM02_NEXT1_SIM_ONLY' or b.get('C08_native_binary_sha256')!=C08
            or not b['activation_epoch']<=now<b['dispatch_cutoff_epoch']<b['deadline_epoch']
            or b['deadline_epoch']>min(b['activation_epoch']+6*3600,b['parent_campaign_deadline_epoch'])
            or b['deadline_epoch']-b['dispatch_cutoff_epoch']<1200
            or b.get('max_native_attempts')!=64 or b.get('max_native_ticks')!=400000
            or b.get('max_native_slots')!=2 or b.get('max_task_atoms')!=160
            or b.get('max_main_updates')!=600 or b.get('max_restore_probes')!=3):
        raise PermissionError('EXACT_LIVE_NEXT1_FINITE_WINDOW_REQUIRED')


def usage(campaign,binding):
    attempts=ticks=0
    for root in (Path(campaign)/'runs').glob('next1-*'):
        p=root/'EXECUTION_SPEC.json'
        if not p.exists():continue
        s=json.loads(p.read_text())
        if s.get('next1_binding')!=str(binding) or s.get('profile')!='wm02-auto1-target':continue
        attempts+=1
        result=root/'native-01/RESULT.json'
        # Reserve full finite endpoints for outstanding/unknown attempts.
        ticks+=json.loads(result.read_text())['actual_target_integrations'] if result.exists() else s['max_ticks']
    return dict(attempts=attempts,native_ticks_or_outstanding_reservations=ticks)


def validate(spec,registry,*,now=None):
    now=time.time() if now is None else now
    path=Path(spec['next1_binding']);b=json.loads(path.read_text());validate_limits(b,now=now)
    lease_path=path.parent/'LEASE.json';lease=json.loads(lease_path.read_text())
    if (digest(path)!=spec['next1_binding_sha256'] or b['campaign_root']!=spec['campaign_root']
            or lease.get('status')!='ACTIVE' or lease.get('binding_sha256')!=digest(path)
            or lease.get('owner')!=b['owner'] or lease.get('deadline_epoch')!=b['deadline_epoch']
            or spec['deadline_epoch']>b['deadline_epoch'] or spec.get('night1_binding')
            or (path.parent/'STOP.json').exists()):raise PermissionError('ACTUAL_NEW_NEXT1_LEASE_REQUIRED_OLD_NIGHT1_NOT_REUSED')
    window_root=str(path.parent.parent)
    if not any(e.get('scope')=='WM02_NEXT1_SIM_ONLY'and e.get('path')==window_root and e.get('local_lease')==str(lease_path) and e.get('owner')==b['owner']for e in registry['entries']):raise PermissionError('ACTUAL_NEXT1_SCOPE_REGISTRATION_REQUIRED')
    if not any(e.get('path')==spec['registered_root']and e.get('source_sha')==spec['source_sha']and e.get('next1_binding')==str(path)for e in registry['entries']):raise PermissionError('ACTUAL_NEXT1_WORKER_REGISTRATION_REQUIRED')
    u=usage(spec['campaign_root'],path)
    if u['attempts']>b['max_native_attempts']or u['native_ticks_or_outstanding_reservations']>b['max_native_ticks']:raise PermissionError('NEXT1_FINITE_ATTEMPTS_OR_TICKS_EXHAUSTED')
    if digest(spec['native_binary'])!=C08:raise PermissionError('NEXT1_FROZEN_C08_EXECUTOR_REQUIRED')
    return b


def register(binding_path):
    """Metadata effect only, under real canonical locks; no physics imports."""
    from contextlib import closing
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.night10.native_relay import snapshot,once
    path=Path(binding_path);b=json.loads(path.read_text());validate_limits(b,now=time.time())
    campaign=Path(b['campaign_root']);parent_lease=campaign/'control/LEASE.json';window=path.parent.parent
    if os.uname().nodename!=b['host']or os.getuid()!=b['uid']:raise PermissionError('ACTUAL_NEXT1_OWNER_HOST_UID_REQUIRED')
    with closing(OwnerLock(campaign/'OWNER.lock')),closing(OwnerLock(b['physical_owner_lock'])):
        parent=json.loads(parent_lease.read_text())
        if parent.get('status')!='ACTIVE'or parent.get('owner')!=b['owner']or parent.get('scope')!='WM02_AUTO1_SIM_ONLY'or parent['deadline_epoch']!=b['parent_campaign_deadline_epoch']:raise PermissionError('EXISTING_PARENT_OWNER_SCOPE_NOT_ACTIVE')
        for root in(campaign/'runs').glob('night1-*'):
            lease=root/'control/LEASE.json'
            if lease.exists()and json.loads(lease.read_text())['status']!='CLOSED':raise PermissionError('NIGHT1_WORKER_NOT_CLOSED')
        if (path.parent/'LEASE.json').exists():raise FileExistsError('EXISTING_OR_UNKNOWN_NEXT1_WINDOW_REQUIRES_RECONCILIATION')
        registry,pin=snapshot(b['registry'])
        for e in registry['entries']:
            if e.get('scope')=='WM02_NEXT1_SIM_ONLY':
                lease=Path(e['local_lease'])
                if not lease.exists()or json.loads(lease.read_text()).get('status')!='CLOSED':raise PermissionError('ANOTHER_NEXT1_WINDOW_ACTIVE_OR_UNKNOWN')
        if not any(e.get('path')==str(campaign)and e.get('scope')=='WM02_AUTO1_SIM_ONLY'and e.get('owner')==b['owner']for e in registry['entries']):raise PermissionError('EXISTING_OWNER_NOT_REGISTERED')
        profiles=json.loads(Path(b['environments']).read_text())['profiles']
        if not any(p['name']=='4080-simulation'and b['host']in p['hostnames']and p['python']==b['runtime_python']for p in profiles):raise PermissionError('ACTUAL_EXISTING_SITE_PROFILE_REQUIRED')
        for name,pin_expected in b['evidence_files'].items():
            if digest(name)!=pin_expected:raise PermissionError('SEALED_NEXT1_PARENT_EVIDENCE_CHANGED:'+name)
        registry['entries'].append(dict(role='WM02_NEXT1_WINDOW_20261105_01',path=str(window),run_uuid=window.name,executor_node=b['host'],owner=b['owner'],scope=b['scope'],source_sha=b['source_sha'],local_lease=str(path.parent/'LEASE.json'),plan_sha=PLAN_SHA))
        once(Path(b['registry']),registry,'WM02-NEXT1-WINDOW-20261105-01-REGISTER',expected_before_sha256=pin)
        write(path.parent/'LEASE.json',dict(status='ACTIVE',owner=b['owner'],scope=b['scope'],run_uuid=window.name,binding_sha256=digest(path),activated_epoch=b['activation_epoch'],deadline_epoch=b['deadline_epoch']))
        state_path=campaign/'AUTO1_STATE.json';state,state_pin=snapshot(state_path)
        if state.get('owner')!=b['owner']:raise PermissionError('CANONICAL_OWNER_CHANGED')
        state.update(current_phase='NEXT1_R0_REGISTERED_S0_PENDING',updated_epoch=time.time())
        state['next1']=dict(status='REGISTERED_NOT_NATIVE_QUALIFIED',plan_sha=PLAN_SHA,binding=str(path),deadline_epoch=b['deadline_epoch'],native_attempts=0,native_ticks=0,valid_atoms=0,main_updates=0,restore_probes=0,Actor_updates=0,next='S0 independent finite HOLD prefix qualification')
        once(state_path,state,'WM02-NEXT1-WINDOW-20261105-01-INDEX',expected_before_sha256=state_pin)
        result=dict(status='NEXT1_ACTUAL_SCOPE_REGISTERED',owner=b['owner'],plan_sha=PLAN_SHA,binding_sha256=digest(path),deadline_epoch=b['deadline_epoch'],dispatch_cutoff_epoch=b['dispatch_cutoff_epoch'],old_NIGHT1_reopened=False,physics_integrations=0,WM_updates=0)
        write(window/'REGISTERED.private.json',result);return result
