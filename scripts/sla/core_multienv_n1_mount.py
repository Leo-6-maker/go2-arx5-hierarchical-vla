"""Cold handoff for the unstarted suffix of the original N1 engineering matrix."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from go2arx5_vla.world_model.mfree_recovery_v2 import launch_service, register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs, unit_state
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.sla.core_multienv_n1 import fixed_cards, make_case, retained_records
from scripts.sla.core_multienv_n1_storage import capacity
from scripts.wm02.multitask_stage_cards_v6 import pin


def preview(config):
    old = read_pin(config['original']['spec'])
    keep = fixed_cards(read_pin(old['sim_only_project_authorization']))
    prefix = retained_records(config['original'],keep,closed=False)
    capacities = capacity(Path(old['registered_root']),config['storage'],old['multienv'])
    return old, keep, prefix, capacities


def drain(original, old, cards, records):
    root = Path(old['registered_root'])
    unit = 'wm02-'+root.name
    terminal = root/'RESULT.private.json'
    stop = root/'STOP.json'
    marker = dict(reason='PR425_N1_LOCAL_DISK_COLD_HANDOFF',original=original)
    if stop.exists():
        if read(stop) != marker:
            raise PermissionError('A different stop owns the original N1 schedule')
    else:
        state = unit_state(unit)
        progress = read(root/'PROGRESS.private.json')
        cid = list(cards)[len(records)]
        if (state['ActiveState'] != 'active' or int(state['MainPID']) != progress['pid']
                or progress['status'] not in ('WAIT_CAPACITY','WAIT_WORLD') or time.time()-progress['epoch'] > 12
                or progress['current'] != cid or progress['paid_requests'] != len(records)
                or progress['verified_ticks'] != sum(r['ticks'] for r in records)
                or (root/'cases'/cid).exists() or (root/'control'/'journal'/(cid+'.json')).exists()):
            raise PermissionError('Expected the original N1 owner waiting with no live condition')
        write(stop,marker)
    until = time.monotonic()+30
    while not terminal.is_file() or unit_state(unit)['MainPID'] != '0':
        if time.monotonic() > until:
            raise TimeoutError('Wait for the original N1 terminal before any replacement')
        time.sleep(1)
    references = dict(original,terminal=pin(terminal))
    retained_records(references,cards,closed=True)
    return references


def prepare(config):
    src = Path(__file__).resolve().parents[2]
    root = Path(config['root'])
    old,cards,records,storage = preview(config)
    old_root = Path(old['registered_root'])
    source = read_pin(config['source_members'])
    end = min(config['deadline'],old['deadline_epoch'])
    if (os.getuid()!=old['uid'] or os.uname().nodename!=old['host'] or root.exists()
            or root.parent!=old_root.parent or not root.name.startswith('sla-core-multienv-n1-engineering-')
            or time.time()+3600>=end or not storage['passed'] or set(old['source_files'])-source.keys()
            or any(digest(src/path)!=sha for path,sha in source.items())):
        raise PermissionError('Expected a new finite N1 owner and verified local plus NFS reserves')
    if subprocess.check_output(['loginctl','show-user',str(old['uid']),'-p','Linger'],text=True).strip()!='Linger=yes':
        raise PermissionError('Core user manager must persist across disconnects')
    pilot = read_pin(config['pilot'])
    pilot_root = Path(pilot['registered_root'])
    read_pin(config['pilot_terminal'])
    pilot_owner = unit_state('wm02-'+pilot_root.name)
    if (Path(config['pilot_terminal']['path'])!=pilot_root/'RESULT.private.json'
            or pilot_owner['MainPID']!='0' or pilot_owner['ActiveState'] not in ('inactive','failed')
            or read(pilot_root/'control/LEASE.json')['status']!='CLOSED' or pilot['mobile_world_lock']!=old['mobile_world_lock']):
        raise PermissionError('The original slot 26 repair still owns its turn')
    todo = list(cards)[len(records):]
    with closing(OwnerLock(old['mobile_world_lock'])):
        prior = drain(config['original'],old,cards,records)
        dirs(root)
        (root/'cases').mkdir()
        auth = dict(read_pin(old['sim_only_project_authorization']),source_sha=config['source_sha'],deadline_epoch=end)
        approval = root/'AUTHORITY.private.json'
        write(approval,auth)
        spec = dict(old, source_sha=config['source_sha'],source_root=str(src),source_files=source,
            source_seal=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
            registered_root=str(root),campaign_root=str(root),mobile_group=str(root),unattended_control=str(root/'control'),
            deadline_epoch=end,campaign_deadline_epoch=end,wall_limit_s=int(end-time.time()),
            sim_only_project_authorization=pin(approval),new_physical_requests_budget=len(todo),
            retained_n1=prior,n1_archive_storage=config['storage'],
            cases=[str(root/'cases'/cid/'EXECUTION_SPEC.json') for cid in todo])
        spec.update({name:str(root/name) for name in ('tmp','cache','logs','data')})
        spec['native_environment']=dict(old['native_environment'],TMPDIR=spec['tmp'],XDG_CACHE_HOME=spec['cache'],
            ROS_LOG_DIR=str(root/'logs/ros'),PYTHONPATH=os.pathsep.join((str(src),str(src/'src'),old['native_environment']['PYTHONPATH'])))
        for cid in todo:
            _, member=make_case(spec,cards[cid],None,min(end,time.time()+2160))
            validate(member)
        register(spec,[(root,spec)])
    write(root/'PREFLIGHT.private.json',dict(retained=prior,paid_before=len(records),
        admitted_train=0,conditions=todo,new_native_at_registration=0,resources=storage))
    return launch_service(spec,'wm02-'+root.name,'--core-multienv-n1',bootstrap='v6_bootstrap.py')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha')
    parser.add_argument('--inspect',action='store_true')
    options=parser.parse_args()
    os.umask(0o077)
    binding=read_pin(dict(path=options.config,sha256=options.sha))
    if options.inspect:
        _, cards, prefix, volumes=preview(binding)
        print(json.dumps(dict(retained=len(prefix),paid_ticks=sum(r['ticks'] for r in prefix),
                              next=list(cards)[len(prefix):],resources=volumes)))
    else:
        print(json.dumps(prepare(binding)))
