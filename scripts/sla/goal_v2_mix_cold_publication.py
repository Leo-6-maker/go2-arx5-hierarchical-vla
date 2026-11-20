"""Publish one actual cold Core segment under the original site window."""
import argparse
import json
import os
from pathlib import Path
import time

from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, cursors
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.sla.goal_v2_core_mix_after_native import PLAN, TRAINER, await_cold
from scripts.sla.publish_scale_checkpoint_hf import REPO, run as publish, sha


ARM = 'SCALE-MIX20K'
STOP = 2000


def verified_effect(predecessor, schedule):
    parent_pin = schedule['previous_train_result']
    parent = read_pin(parent_pin)
    first = parent['global_step']
    if (not 1000 <= first < STOP or parent['arm']!=ARM or parent['source_sha']!=TRAINER
            or parent['mode']!='train' or parent['global_cursors']!=cursors(ARM,first)
            or (parent['exit_code'],parent['status']) not in ((0,'FIXED_SEGMENT_SAVED'),(75,'BOUNDED_SAVED_PREFIX'))
            or not parent['checkpoint'] or parent.get('failure')):
        raise ValueError('MIX_HF_NEXT_PREFIX_REQUIRES_THE_PINNED_RESUME_PARENT')
    worker = read(predecessor/'train/SPEC.private.json')
    if worker['resume_from_closed_segment']!=parent_pin or worker['source_sha']!=TRAINER:
        raise ValueError('MIX_HF_NEXT_PREFIX_WORKER_PARENT_CHANGED')
    outer = read(predecessor/'RESULT.private.json')
    pinned = outer['stages']['train']
    effect = read_pin(pinned)
    step = effect['global_step']
    before, after = cursors(ARM,first), cursors(ARM,step)
    absolute_phase = (effect['exit_code'],effect['status'])
    if (Path(pinned['path'])!=predecessor/'train/RESULT.private.json' or outer['status']!=effect['status']
            or effect['source_sha']!=TRAINER or effect['arm']!=ARM or effect['mode']!='train'
            or not first<step<=STOP or effect['global_cursors']!=after or effect['new_updates']!=step-first
            or effect['new_visits']!={d:after[d]-before[d] for d in DOMAINS}
            or (step<STOP and absolute_phase!=(75,'BOUNDED_SAVED_PREFIX'))
            or (step==STOP and absolute_phase!=(0,'FIXED_SEGMENT_SAVED'))
            or effect['new_native_requests'] or effect['new_native_ticks'] or outer.get('first_failure')):
        raise ValueError('MIX_HF_NEXT_PREFIX_REQUIRES_THE_AUTHORITATIVE_COLD_RESULT')
    lease = read(predecessor/'control/LEASE.json')
    if (lease['status']!='CLOSED' or lease['unit']!=schedule['unit']+'.service'
            or lease['deadline_epoch']!=schedule['deadline_epoch']):
        raise PermissionError('MIX_HF_NEXT_PREFIX_REQUIRES_CLOSED_TRAINING_LEASE')
    path = predecessor/'train/UPDATES.private.jsonl'
    raw = path.read_bytes().splitlines(keepends=True)
    if len(raw)!=step-first:
        raise ValueError('MIX_HF_NEXT_PREFIX_DURABLE_LEDGER_COUNT_CHANGED')
    for index,line in enumerate(raw,first+1):
        row = json.loads(line)
        if (not line.endswith(b'\n') or row['step']!=index or row['cursors']!=cursors(ARM,index)
                or row['domains']['target']['training_full_generations']!=4):
            raise ValueError('MIX_HF_NEXT_PREFIX_DURABLE_ROW_OR_GENERATION_CHANGED')
    return pinned,step,sha(path)


def watch(config):
    root = Path(config['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    campaign, schedule = (read_pin(config[key]) for key in ('campaign','predecessor_spec'))
    predecessor = Path(schedule['root']).resolve(strict=True)
    expected = config['predecessor_owner']
    owner = unit_state(config['unit'])
    end = min(campaign['deadline_epoch'],config['site_deadline_epoch'])
    if (os.uname().nodename!='corelab-4080s' or os.getuid()!=2009 or root.stat().st_uid!=os.getuid()
            or owner['pid']!=os.getpid() or owner['pid'] not in owner['cgroup_pids']
            or config['source_sha']!=(source/'SOURCE_SHA.private').read_text().strip()
            or campaign['plan_sha']!=PLAN or campaign['deadline_epoch']!=campaign['t0_epoch']+14*86400
            or schedule['unit']+'.service'!=expected['unit'] or schedule['source_sha']!=TRAINER
            or schedule['arm']!=ARM or schedule['global_stop_step']!=STOP or schedule['deadline_epoch']>=end
            or (root/'READY.private.json').exists() or (root/'RESULT.private.json').exists()):
        raise PermissionError('MIX_HF_WATCH_REQUIRES_OWNED_ORIGINAL_FINITE_SEGMENT')
    stops = root/'STOP.json',Path(campaign['stop'])
    stopped = lambda: any(path.exists() for path in stops)
    lock = OwnerLock(root/'OWNER.lock')
    freshness = dict(status='WAITING_KNOWN_MIX_PREFIX_TO_CLOSE',optimizer_updates=0,optimization_visits=0,
        model_generations=0,native_requests=0,native_ticks=0,source_sha=config['source_sha'])
    try:
        observed = unit_state(expected['unit'])
        if observed['pid'] and any(observed[k]!=expected[k] for k in ('pid','ControlGroup','proc_start','InvocationID')):
            raise PermissionError('MIX_HF_WATCHED_ORIGINAL_TRAINING_OWNER_CHANGED')
        write(root/'READY.private.json',dict(freshness,owner=owner,predecessor=observed,deadline_epoch=end))
        await_cold(expected['unit'],observed,deadline=min(end-900,schedule['deadline_epoch']+240),stopped=stopped)
        result, step, ledger = verified_effect(predecessor,schedule)
        write(root/'COLD_PREFIX.private.json',dict(verified_step=step,result=result,verified_update_ledger_sha256=ledger))
        while True:
            available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))
            write(root/'RESOURCES.private.json',dict(epoch=time.time(),available_host_bytes=available,
                                                    required_host_bytes=40*1024**3))
            if stopped() or end-time.time()<900:
                raise PermissionError('MIX_HF_PUBLICATION_CURRENT_WINDOW_OR_HOST_HEADROOM_UNAVAILABLE')
            if available>40*1024**3:
                break
            time.sleep(30)
        return publish(dict(root=str(root),repo=REPO,source_sha=config['source_sha'],arm=ARM,step=step,train=result,
            allow_bounded_Core_prefix=True,deadline_epoch=min(end,time.time()+4500),
            credential_path=config['credential_path']))
    except Exception as error:
        if not (root/'RESULT.private.json').exists():
            write(root/'RESULT.private.json',dict(freshness,status='MIX_HF_COLD_PREFIX_NOT_PUBLISHED',
                first_failure=f'{type(error).__name__}: {error}'))
        return 1
    finally:
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args=parser.parse_args()
    os.umask(0o077)
    raise SystemExit(watch(read_pin(dict(path=args.spec,sha256=args.sha256))))
