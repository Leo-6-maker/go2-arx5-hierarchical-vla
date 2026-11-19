"""Transport a closed VANDA SCALE checkpoint without changing its training card."""
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
from go2arx5_vla.sla_adapter.scale20k_contract import CHECKPOINTS, PLAN, RECIPE, cursors
from scripts.pi0292.durable import read, write
from scripts.wm02.d416_endpoint_transfer import byte_record, pin, scheduler_terminal


SCHEMA = 'SLA_SCALE20K_VANDA_PORTABLE_COMMITTED_STATE_V1'
ARM = 'SCALE-TGT5K'
MANIFEST = 'TRANSFER.private.json'
MAX_BYTES = 8*1024**3
CARDS = ('EXECUTION.private.json','RESULT.private.json','train/SPEC.private.json',
         'train/RESULT.private.json','train/MODEL_AT_START.private.json')


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


def export_bundle(root, stream):
    root = root.resolve(strict=True)
    execution = read(root/CARDS[0])
    # Check the actual scheduler before opening a checkpoint or writing any archive bytes.
    site = scheduler_terminal(execution)
    if execution['root'] != str(root) or root.stat().st_uid != os.getuid():
        raise PermissionError('SCALE_PORTABLE_REQUIRES_OWNED_VANDA_ROOT')
    parent, spec, result, model = (read(root/name) for name in CARDS[1:])
    step = result['global_step']
    closed_training(execution, parent, spec, result, site, step)
    if parent['stages']['train'] != pin(root/CARDS[3]):
        raise ValueError('SCALE_PORTABLE_TRAINING_RESULT_NOT_BOUND_TO_PBS_PARENT')
    checkpoint, complete = read_complete(result['checkpoint'],ARM,model['binding'])
    if checkpoint != root/'train'/f'step{step:05d}' or complete['step'] != step:
        raise ValueError('SCALE_PORTABLE_REQUIRES_ORIGINAL_SEALED_CHECKPOINT_DIRECTORY')
    files = {name:root/name for name in CARDS}
    files['checkpoint/COMPLETE.json'] = checkpoint/'COMPLETE.json'
    wanted = archive_names(complete['arrays'])
    files.update({name:checkpoint/Path(name).name for name in wanted})
    members = {name:byte_record(file,wanted.get(name)) for name,file in files.items()}
    if sum(record['bytes'] for record in members.values()) > MAX_BYTES:
        raise ValueError('SCALE_PORTABLE_TRANSFER_BYTE_CAP_EXCEEDED')
    manifest = dict(schema=SCHEMA,plan_sha=PLAN,arm=ARM,step=step,job=site,members=members,
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
    members = manifest['members']
    execution,parent,spec,result,model = (read_pin(dict(path=str(dest/name),sha256=members[name]['sha256'])) for name in CARDS)
    step = manifest['step']
    closed_training(execution,parent,spec,result,manifest['job'],step)
    checkpoint = dict(path=str(dest/'checkpoint'),sha256=members['checkpoint/COMPLETE.json']['sha256'])
    _,complete = read_complete(checkpoint,ARM,model['binding'])
    wanted = archive_names(complete['arrays'])
    if (complete['step'] != step or checkpoint['sha256'] != result['checkpoint']['sha256']
            or manifest['source_sha'] != spec['source_sha']
            or parent['stages']['train']['sha256'] != members['train/RESULT.private.json']['sha256']
            or any(members[key]['sha256'] != digest for key,digest in wanted.items())):
        raise ValueError('SCALE_PORTABLE_ARCHIVE_BREAKS_ORIGINAL_CHECKPOINT_HASH_CONTRACT')
    return dict(transfer=transfer,arm=ARM,step=step,training_source_sha=spec['source_sha'],
                training_spec=pin(dest/'train/SPEC.private.json'),checkpoint=checkpoint,
                new_optimizer_updates=0,new_native_requests=0)


def receive_bundle(destination, stream):
    required = {*CARDS,'checkpoint/COMPLETE.json',*(f'checkpoint/{i:06d}.npy' for i in range(140))}
    with tarfile.open(fileobj=stream,mode='r|') as archive:
        first = archive.next()
        if first is None or first.name != MANIFEST or not first.isfile() or not 0 < first.size <= 131072:
            raise ValueError('SCALE_PORTABLE_REQUIRES_BOUNDED_ARCHIVE_MANIFEST')
        data = archive.extractfile(first).read()
        manifest = json.loads(data)
        members = manifest['members']
        if (manifest['schema'] != SCHEMA or manifest['plan_sha'] != PLAN or manifest['arm'] != ARM
                or manifest['step'] not in CHECKPOINTS[ARM] or set(members) != required
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
    args = parser.parse_args()
    os.umask(0o077)
    if args.action=='export':
        export_bundle(args.root,sys.stdout.buffer)
    else:
        print(json.dumps(receive_bundle(args.root,sys.stdin.buffer),allow_nan=False))
