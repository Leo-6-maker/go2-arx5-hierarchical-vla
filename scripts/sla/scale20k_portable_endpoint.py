"""Transport a formal SCALE checkpoint and the original training evidence."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile

from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_checkpoint import read_complete
from go2arx5_vla.sla_adapter.scale20k_contract import CHECKPOINTS, DOMAINS, PLAN, RECIPE, cursors
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from scripts.pi0292.durable import read, write
from scripts.wm02.d416_endpoint_transfer import byte_record, pin, scheduler_terminal


SCHEMA = 'SLA_SCALE20K_VANDA_PORTABLE_COMMITTED_STATE_V1'
ARM = 'SCALE-TGT5K'
CORE_SCHEMA = 'SLA_SCALE20K_CORE_MIX500_PORTABLE_COMMITTED_STATE_V1'
CORE_ARM = 'SCALE-MIX20K'
MANIFEST = 'TRANSFER.private.json'
MAX_BYTES = 8*1024**3
CARDS = ('EXECUTION.private.json','RESULT.private.json','train/SPEC.private.json',
         'train/RESULT.private.json','train/MODEL_AT_START.private.json')
CORE_CARDS = ('control/LEASE.json','train/UPDATES.private.jsonl')


def layout(schema):
    if schema == SCHEMA:
        return ARM, CARDS, CHECKPOINTS[ARM]
    if schema == CORE_SCHEMA:
        return CORE_ARM, (*CARDS,*CORE_CARDS), (500,)
    raise ValueError('SCALE_PORTABLE_UNKNOWN_SOURCE_SITE')


def archive_names(arrays):
    names = [f'{i:06d}.npy' for i in range(140)]
    if [row['file'] for row in arrays] != names:
        raise ValueError('SCALE_PORTABLE_EXACT_140_ARRAY_INDEX_REQUIRED')
    return {'checkpoint/'+name:row['sha256'] for name,row in zip(names,arrays,strict=True)}


def closed_training(execution, parent, spec, result, site, step):
    training = parent.get('training_source_sha', parent['source_sha'])
    if (execution['plan_sha'] != PLAN or execution['arm'] != ARM or execution['job_id'] != site['id']
            or parent['PBS_JOBID'] != site['id'] or site['state'] != 'F' or site['exit_code'] != 0
            or parent['source_sha'] != execution['source_sha'] or parent['status'] != 'FIXED_SEGMENT_SAVED'
            or spec['source_sha'] != training or result['source_sha'] != training
            or spec['plan_sha'] != PLAN or spec['recipe'] != RECIPE or spec['arm'] != ARM
            or result['arm'] != ARM or result['scope'] != RECIPE or result['mode'] != 'train'
            or step not in CHECKPOINTS[ARM] or spec['global_stop_step'] != step or result['global_step'] != step
            or result['global_cursors'] != cursors(ARM,step) or result['status'] != 'FIXED_SEGMENT_SAVED'
            or result['exit_code'] or result['new_native_requests'] or result['new_native_ticks']
            or spec['physics'] is not False or spec['hardware_allowed'] is not False):
        raise ValueError('SCALE_PORTABLE_REQUIRES_SUCCESSFULLY_CLOSED_FORMAL_TRAINING')


def closed_mix500(execution, parent, spec, result, site, step, lease, ledger, train_terminal):
    end, before = cursors(CORE_ARM,500), cursors(CORE_ARM,200)
    if (step != 500 or execution['plan'] != PLAN or execution['arm'] != CORE_ARM
            or execution['global_stop_step'] != 500 or execution['unit']+'.service' != site['unit']
            or lease['unit'] != site['unit'] or lease['status'] != 'CLOSED'
            or lease['deadline_epoch'] != execution['deadline_epoch']
            or site['ActiveState'] != 'inactive' or site['pid'] or site['cgroup_pids']
            or parent['status'] != 'FIXED_SEGMENT_SAVED' or parent['stages']['train']['sha256'] != train_terminal
            or spec['source_sha'] != execution['source_sha'] or result['source_sha'] != execution['source_sha']
            or spec['plan_sha'] != PLAN or spec['recipe'] != RECIPE or spec['arm'] != CORE_ARM
            or spec['global_stop_step'] != 500 or spec['physics'] is not False or spec['hardware_allowed'] is not False
            or result['scope'] != RECIPE or result['mode'] != 'train' or result['arm'] != CORE_ARM
            or result['global_step'] != 500 or result['global_cursors'] != end
            or result['status'] != 'FIXED_SEGMENT_SAVED' or result['exit_code']
            or result['new_updates'] != 300 or result['new_model_generations'] != 1200
            or result['new_native_requests'] or result['new_native_ticks']
            or result['new_visits'] != {d:end[d]-before[d] for d in DOMAINS}):
        raise ValueError('SCALE_PORTABLE_REQUIRES_COLD_FORMAL_CORE_MIX500')
    lines = ledger.splitlines(keepends=True)
    if len(lines) != 300 or any(not line.endswith(b'\n') for line in lines):
        raise ValueError('SCALE_PORTABLE_MIX500_EXPECTS_ALL_DURABLE_POST_200_UPDATES')
    for step, line in enumerate(lines,201):
        row = json.loads(line)
        if (row['step'] != step or row['cursors'] != cursors(CORE_ARM,step)
                or set(row['domains']) != set(DOMAINS) or row['domains']['target']['training_full_generations'] != 4):
            raise ValueError('SCALE_PORTABLE_MIX500_LEDGER_EFFECTS_CHANGED')


def export_bundle(root, stream, *, core=False):
    root = root.resolve(strict=True)
    schema = CORE_SCHEMA if core else SCHEMA
    arm, names, _ = layout(schema)
    files = {name:root/name for name in names}
    if core:
        files[CARDS[0]] = root/'SCHEDULE.private.json'
    execution = read(files[CARDS[0]])
    if execution['root'] != str(root) or root.stat().st_uid != os.getuid():
        raise PermissionError('SCALE_PORTABLE_REQUIRES_OWNED_TRAINING_ROOT')
    # Check the actual scheduler before opening a checkpoint or writing any archive bytes.
    if core:
        if execution['host'] != os.uname().nodename or execution['uid'] != os.getuid():
            raise PermissionError('SCALE_PORTABLE_REQUIRES_ORIGINAL_CORE_HOST_AND_USER')
        observed = unit_state(execution['unit']+'.service')
        site = {k:observed[k] for k in ('unit','ActiveState','pid','cgroup_pids')}
        if site['ActiveState'] != 'inactive' or site['pid'] or site['cgroup_pids']:
            raise PermissionError('SCALE_PORTABLE_CORE_MIX500_TRAINING_STILL_OWNED')
    else:
        site = scheduler_terminal(execution)
    parent, spec, result, model = (read(root/name) for name in CARDS[1:])
    step = result['global_step']
    terminal = pin(root/CARDS[3])
    if parent['stages']['train'] != terminal:
        raise ValueError('SCALE_PORTABLE_TRAINING_RESULT_NOT_BOUND_TO_PARENT')
    if core:
        closed_mix500(execution,parent,spec,result,site,step,read(files[CORE_CARDS[0]]),
                      files[CORE_CARDS[1]].read_bytes(),terminal['sha256'])
    else:
        closed_training(execution,parent,spec,result,site,step)
    checkpoint, complete = read_complete(result['checkpoint'],arm,model['binding'])
    if checkpoint != root/'train'/f'step{step:05d}' or complete['step'] != step:
        raise ValueError('SCALE_PORTABLE_REQUIRES_ORIGINAL_SEALED_CHECKPOINT_DIRECTORY')
    files['checkpoint/COMPLETE.json'] = checkpoint/'COMPLETE.json'
    wanted = archive_names(complete['arrays'])
    files.update({name:checkpoint/Path(name).name for name in wanted})
    members = {name:byte_record(file,wanted.get(name)) for name,file in files.items()}
    if sum(record['bytes'] for record in members.values()) > MAX_BYTES:
        raise ValueError('SCALE_PORTABLE_TRANSFER_BYTE_CAP_EXCEEDED')
    manifest = dict(schema=schema,plan_sha=PLAN,arm=arm,step=step,job=site,members=members,
                    source_sha=spec['source_sha'],optimizer_updates=0,native_requests=0,model_generations=0)
    data = (json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    with tarfile.open(fileobj=stream,mode='w|') as archive:
        entry = tarfile.TarInfo(MANIFEST)
        entry.size = len(data)
        entry.mode = 0o600
        archive.addfile(entry,io.BytesIO(data))
        for name,path in files.items():
            archive.add(path,arcname=name,recursive=False)


def local_custody(destination, transfer):
    dest = destination.resolve(strict=True)
    manifest = read_pin(transfer)
    arm, _, _ = layout(manifest['schema'])
    members = manifest['members']
    execution,parent,spec,result,model = (read_pin(dict(path=str(dest/name),sha256=members[name]['sha256'])) for name in CARDS)
    step = manifest['step']
    if arm == CORE_ARM:
        lease, ledger = (dest/name for name in CORE_CARDS)
        if (pin(lease)['sha256'] != members[CORE_CARDS[0]]['sha256']
                or pin(ledger)['sha256'] != members[CORE_CARDS[1]]['sha256']):
            raise ValueError('SCALE_PORTABLE_CORE_LEASE_OR_LEDGER_CHANGED')
        closed_mix500(execution,parent,spec,result,manifest['job'],step,read(lease),ledger.read_bytes(),
                      members[CARDS[3]]['sha256'])
    else:
        closed_training(execution,parent,spec,result,manifest['job'],step)
    checkpoint = dict(path=str(dest/'checkpoint'),sha256=members['checkpoint/COMPLETE.json']['sha256'])
    _,complete = read_complete(checkpoint,arm,model['binding'])
    wanted = archive_names(complete['arrays'])
    if (complete['step'] != step or checkpoint['sha256'] != result['checkpoint']['sha256']
            or manifest['source_sha'] != spec['source_sha']
            or parent['stages']['train']['sha256'] != members['train/RESULT.private.json']['sha256']
            or any(members[key]['sha256'] != digest for key,digest in wanted.items())):
        raise ValueError('SCALE_PORTABLE_ARCHIVE_BREAKS_ORIGINAL_CHECKPOINT_HASH_CONTRACT')
    return dict(transfer=transfer,arm=arm,step=step,training_source_sha=spec['source_sha'],
                training_spec=pin(dest/'train/SPEC.private.json'),checkpoint=checkpoint,
                new_optimizer_updates=0,new_native_requests=0)


def receive_bundle(destination, stream):
    with tarfile.open(fileobj=stream,mode='r|') as archive:
        first = archive.next()
        if first is None or first.name != MANIFEST or not first.isfile() or not 0 < first.size <= 131072:
            raise ValueError('SCALE_PORTABLE_REQUIRES_BOUNDED_ARCHIVE_MANIFEST')
        data = archive.extractfile(first).read()
        manifest = json.loads(data)
        arm, names, expected_steps = layout(manifest['schema'])
        required = {*names,'checkpoint/COMPLETE.json',*(f'checkpoint/{i:06d}.npy' for i in range(140))}
        members = manifest['members']
        if (manifest['plan_sha'] != PLAN or manifest['arm'] != arm
                or manifest['step'] not in expected_steps or set(members) != required
                or any(not 0 < r['bytes'] <= MAX_BYTES for r in members.values())
                or sum(r['bytes'] for r in members.values()) > MAX_BYTES
                or any(manifest[k] != 0 for k in ('optimizer_updates','model_generations','native_requests'))):
            raise ValueError('SCALE_PORTABLE_INVALID_ARCHIVE_MEMBERS_OR_PLAN')
        destination.mkdir(mode=0o700)
        seen = set()
        for member in iter(archive.next,None):
            name = member.name
            if name not in required or name in seen or not member.isfile() or member.size != members[name]['bytes']:
                raise ValueError('SCALE_PORTABLE_UNEXPECTED_ARCHIVE_MEMBER:'+name)
            path = destination/name
            path.parent.mkdir(exist_ok=True)
            digest = hashlib.sha256()
            with archive.extractfile(member) as source, path.open('xb') as out:
                for block in iter(lambda:source.read(4*1024**2),b''):
                    out.write(block)
                    digest.update(block)
                out.flush()
                os.fsync(out.fileno())
            if digest.hexdigest() != members[name]['sha256']:
                raise ValueError('SCALE_PORTABLE_RECEIVED_FILE_HASH_CHANGED:'+name)
            seen.add(name)
        if seen != required:
            raise ValueError('SCALE_PORTABLE_TRANSFER_MISSING_FILES')
    (destination/MANIFEST).write_bytes(data)
    custody = local_custody(destination,pin(destination/MANIFEST))
    write(destination/'CUSTODY.private.json',custody)
    return custody


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('export','receive'))
    parser.add_argument('root',type=Path)
    parser.add_argument('--from-core',action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    if args.action=='export':
        export_bundle(args.root,sys.stdout.buffer,core=args.from_core)
    else:
        print(json.dumps(receive_bundle(args.root,sys.stdin.buffer),allow_nan=False))
