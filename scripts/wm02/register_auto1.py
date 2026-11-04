"""Register the directly authorized AUTO1 campaign through the existing Core owner.

Registration is metadata only. Native construction/integration requires the
controlled commissioning entry, original physical owner lock and phase checks.
"""
import argparse,json,os,time
from pathlib import Path

def validate(binding,*,host,uid,now,profile):
    if (binding['scope']!='WM02_AUTO1_SIM_ONLY'or binding['owner']!='EXISTING_LOCAL_PC'
            or binding['host']!=host or host!='corelab-4080s'or binding['uid']!=uid or uid!=2009
            or binding['profile']!='4080-simulation'or profile['name']!='4080-simulation'
            or host not in profile['hostnames']or binding['runtime_python']!=profile['python']
            or binding['authority_commit']!='930bb2ef0ff178771e48fb5f03ea3e37fa717582'
            or not now<binding['activation_deadline_epoch']<=now+3600
            or binding['campaign_hours']!=72 or binding['max_target_instances']!=1
            or binding['max_init_candidates']!=6):raise PermissionError('AUTO1_OWNER_PROFILE_OR_FINITE_SCOPE_DIFFERS')

def register(path):
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.night10.native_relay import snapshot,once
    from scripts.pi0294.e1.smoke import node_local
    from go2arx5_vla.world_model.storage import digest
    b=json.loads(Path(path).read_text());root=Path(b['root']);source=Path(b['source_root'])
    profiles=json.loads(Path(b['environments']).read_text())['profiles'];matches=[p for p in profiles if p['name']==b['profile']]
    if len(matches)!=1:raise PermissionError('EXACT_EXISTING_SITE_PROFILE_REQUIRED')
    validate(b,host=os.uname().nodename,uid=os.getuid(),now=time.time(),profile=matches[0])
    if not Path(__file__).resolve().is_relative_to(source.resolve()):raise PermissionError('WRONG_SOURCE_CAPSULE')
    for p in(root,source,Path(b['runtime_python'])):node_local(p)
    for name,pin in b['source_files'].items():
        if digest(source/name)!=pin:raise PermissionError('AUTO1_SOURCE_CHANGED:'+name)
    old=Path(b['historical_target_lease']);old_pin=digest(old)
    if old_pin!=b['historical_target_lease_sha256']or json.loads(old.read_text())['status']!='CLOSED':raise PermissionError('HISTORICAL_LEASE_MUST_STAY_CLOSED')
    if (root/'control/LEASE.json').exists():raise FileExistsError('EXISTING_CAMPAIGN_REQUIRES_READONLY_RECONCILIATION')
    registry=Path(b['registry'])
    with_lock=OwnerLock(b['physical_owner_lock'])
    try:
        v,before=snapshot(registry)
        if not any(e.get('path')==b['historical_target_root']and e.get('owner')==b['owner']for e in v['entries']):raise PermissionError('EXISTING_TARGET_OWNER_BINDING_NOT_FOUND')
        for e in v['entries']:
            if e.get('scope')=='WM02_AUTO1_SIM_ONLY':
                lease=Path(e['local_lease'])
                if not lease.exists()or json.loads(lease.read_text()).get('status')!='CLOSED':raise PermissionError('ANOTHER_AUTO1_OWNER_IS_REGISTERED_OR_UNKNOWN')
        if any(e.get('path')==str(root)for e in v['entries']):raise FileExistsError('AUTO1_RUN_ID_ALREADY_REGISTERED')
        v['entries'].append(dict(role=b['role'],path=str(root),run_uuid=root.name,executor_node=b['host'],owner=b['owner'],scope=b['scope'],profile=b['profile'],source_sha=b['source_sha'],authority_commit=b['authority_commit'],local_lease=str(root/'control/LEASE.json')))
        key=root.name.upper()+'-REGISTER'
        receipt=once(registry,v,key,expected_before_sha256=before)
        epoch=time.time();lease=dict(status='ACTIVE',run_uuid=root.name,owner=b['owner'],scope=b['scope'],host=b['host'],uid=b['uid'],source_sha=b['source_sha'],binding_sha256=digest(path),activated_epoch=epoch,deadline_epoch=epoch+72*3600,old_closed_lease_reopened=False)
        write(root/'control/LEASE.json',lease)
        result=dict(schema='WM02_AUTO1_RESOLVED_V1',user_authorized=True,registration='REGISTERED',native_qualification='NOT_RUN',owner=b['owner'],host=b['host'],uid=b['uid'],profile=b['profile'],source_sha=b['source_sha'],authority_commit=b['authority_commit'],run_uuid=root.name,activated_epoch=epoch,deadline_epoch=lease['deadline_epoch'],registry_receipt=receipt,registry_sha256=digest(registry),historical_lease_sha256=old_pin,target_integrations=0,WM_updates=0,Actor_updates=0)
        write(root/'AUTO1_RESOLVED.json',result)
        if digest(old)!=old_pin:raise RuntimeError('Historical lease changed during registration')
        return result
    finally:with_lock.close()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binding',type=Path,required=True);a=p.parse_args(argv);v=register(a.binding);print(json.dumps(v));return 0
if __name__=='__main__':raise SystemExit(main())
