"""Wait for the current MIX500 owner to close, then release its staged continuation once."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import time

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import PLAN
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.scale20k_core_prefix import GIB, closed_predecessor, resources


def headroom(card, protected_units):
    station = resources(card['GPU_uuid'])
    unused = 0
    for unit in protected_units:
        values = subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState',
            '-p','MemoryMax','-p','MemoryCurrent'],text=True,timeout=15)
        state = dict(line.split('=',1) for line in values.splitlines())
        if state['ActiveState'] == 'active':
            if state['MemoryMax'] == 'infinity':
                return False
            unused += max(0,int(state['MemoryMax'])-int(state['MemoryCurrent']))
    return (station['free_bytes'] > math.ceil(station['total_bytes']*card['JAX_memory_fraction'])+2*GIB
            and station['host_bytes'] > unused+28*GIB)


def run(spec):
    root = Path(spec['root']).resolve(strict=True)
    old = Path(spec['predecessor']['path']).resolve(strict=True).parent
    new = Path(spec['staged']['path']).resolve(strict=True).parent
    parent, stage, draft = map(read_pin,(spec['predecessor'],spec['staged'],spec['draft']))
    script = Path(spec['launcher']['path']).resolve(strict=True)
    unit = unit_state(spec['unit'])
    old_unit, new_unit = parent['unit']+'.service', stage['unit']
    if (not old.parent == new.parent == root.parent or len({old,new,root}) != 3
            or os.uname().nodename != spec['host'] or os.getuid() != spec['uid'] or root.stat().st_uid != os.getuid()
            or unit['pid'] != os.getpid() or sha(script) != spec['launcher']['sha256']
            or sha(Path(__file__)) != spec['watch_sha256']
            or stage['draft_sha256'] != spec['draft']['sha256'] or draft['root'] != str(new)
            or parent['plan'] != PLAN or parent['arm'] != 'SCALE-MIX20K' or parent['global_stop_step'] != 500
            or draft['source_sha'] != parent['source_sha'] or draft['global_stop_step'] != 2000
            or parent['deadline_epoch'] != spec['deadline_epoch']
            or not time.time() < spec['deadline_epoch']):
        raise PermissionError('MIX500_WATCH_REQUIRES_THE_ORIGINAL_FINITE_SEGMENT_AND_STAGED_SUCCESSOR')

    lock = OwnerLock(root/'OWNER.lock')
    result = dict(source_sha=spec['source_sha'],old_unit=old_unit,next_unit=new_unit,
                  launcher_invocations=0,start_pid=None,new_optimizer_updates_by_watcher=0,new_native_requests=0)
    try:
        while time.time() < spec['deadline_epoch']:
            if (root/'STOP.json').exists() or (old/'STOP.json').exists():
                result['status'] = 'STOPPED_WITHOUT_HANDOFF'
                return 0
            try:
                current = unit_state(old_unit)
            except FileNotFoundError:
                time.sleep(1)
                continue
            successor = unit_state(new_unit)
            if successor['pid'] or successor['cgroup_pids'] or (new/'SCHEDULE.private.json').exists():
                raise RuntimeError('MIX500_SUCCESSOR_PRESENT_REQUIRES_READBACK')
            lease = read(old/'control/LEASE.json')
            if lease['unit'] != old_unit or lease['deadline_epoch'] != spec['deadline_epoch']:
                raise RuntimeError('MIX500_PREDECESSOR_LEASE_IDENTITY_CHANGED')
            if current['pid'] or current['cgroup_pids']:
                if current['pid'] and (current['pid'] != spec['predecessor_pid']
                        or current['proc_start'] != spec['predecessor_proc_start']):
                    raise RuntimeError('MIX500_PREDECESSOR_PROCESS_IDENTITY_CHANGED')
                progress = read(old/'train/PROGRESS.private.json')
                phase = dict(status='WAIT_ORIGINAL_OWNER_COLD',old_pid=current['pid'],step=progress.get('step'))
            else:
                if lease['status'] != 'CLOSED' or current['ActiveState'] not in ('inactive','failed'):
                    raise RuntimeError('MIX500_PREDECESSOR_OUTCOME_REQUIRES_RECONCILIATION')
                terminal = read(old/'RESULT.private.json')
                prior, _, step = closed_predecessor(dict(draft,previous_train_result=terminal['stages']['train']),root)
                if step != 500 or read_pin(prior)['exit_code'] != 0:
                    result.update(status='PREDECESSOR_CLOSED_BEFORE_FORMAL_500',last_step=step)
                    return 0
                phase = dict(status='WAIT_ACTUAL_GPU_AND_HOST_HEADROOM',step=step)
                if headroom(draft,spec['protected_units']):
                    if time.time() >= spec['deadline_epoch'] or (root/'STOP.json').exists() or (old/'STOP.json').exists():
                        break

                    def launch():
                        with (root/'launch.log').open('x') as log:
                            child = subprocess.run(['/usr/bin/python3','-B',str(script),'start',stage['launcher_sha']],
                                                  stdout=log,stderr=subprocess.STDOUT,timeout=60)
                        return dict(exit_code=child.returncode,unit=unit_state(new_unit))

                    result['launcher_invocations'] = 1
                    effect = Journal(root/'handoff',root.name).once('START_MIX_AFTER500',
                        dict(prior=prior,launcher=spec['launcher'],draft=spec['draft']),launch)
                    if effect['exit_code'] or not effect['unit']['pid']:
                        raise RuntimeError('MIX500_HANDOFF_REQUIRES_RECONCILIATION')
                    result.update(status='STAGED_SUCCESSOR_RUNNING',start_pid=effect['unit']['pid'])
                    return 0
            write(root/'PROGRESS.private.json',dict(phase,epoch=time.time(),PID=os.getpid()))
            time.sleep(min(30,max(0,spec['deadline_epoch']-time.time())))
        result['status'] = 'ORIGINAL_WAIT_WINDOW_ENDED'
        return 0
    except Exception as error:
        result.update(status='HANDOFF_REQUIRES_RECONCILIATION',failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(root/'RESULT.private.json',result)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
