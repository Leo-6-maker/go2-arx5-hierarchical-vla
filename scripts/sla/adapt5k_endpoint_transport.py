"""Move one complete goal-bound ADAPT inference archive with exact member checks."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile

from scripts.pi0292.durable import write


GOAL = '1b2a25b7c1720051a2627d30831fb18130462374'
PHASES = (2000, 5000)
EXPORT_SCHEMA = 'SLA_ADAPT_GO2_ENDPOINT_EXPORT_ACTUAL_V1'
PACKAGE_SCHEMA = 'SLA_ADAPT_TRAINED_TARGET_RGB_PACKAGE_V1'
RECEIPT_SCHEMA = 'SLA_ADAPT_GO2_CORE_CUSTODY_V1'
RGB_SCHEMA = 'SLA_ADAPT_TRAINED_CORE_RGB16_STRICT_V1'
CLOCK_PHASE = 2500
CLOCK_ARMS = ('CLOCK-A-original-width','CLOCK-B-event-weighted-width')
CLOCK_EXPORT = 'SLA_CLOCK500_GO2_CUSTODY_ACTUAL_V1'
CLOCK_PACKAGE = 'SLA_CLOCK500_GO2_TARGET_PACKAGE_V1'
CLOCK_RECEIPT = 'SLA_CLOCK500_GO2_CORE_CUSTODY_V1'
CLOCK_RGB = 'SLA_CLOCK500_CORE_RGB16_STRICT_V1'
CUSTODY = 'CUSTODY.private.json'
MANIFEST = 'package/MANIFEST.private.json'
MAX_BYTES = 3 * 1024**3
MAX_CUSTODY_BYTES = 64 * 1024
NAMES = {'package/D1_STATE_STATISTICS.json':'state_statistics_sha256',
         'package/C20_ACTION_STATISTICS.json':'action_statistics_sha256',
         'package/paligemma_tokenizer.model':'tokenizer_sha256',
         'archive/FORMAL.private.json':'evaluation_sha256',
         'archive/RESTORE_PROOF.private.json':'restore_sha256','archive/PANEL.private.json':'panel_sha256'}


def sha(path):
    with path.open('rb') as src:
        digest = hashlib.sha256()
        for data in iter(lambda:src.read(4*1024**2),b''):
            digest.update(data)
        return digest.hexdigest()


def pin(path):
    return dict(path=str(path),sha256=sha(path))


def read_custody(data, expected, phase, *, arm=None):
    if len(data) > MAX_CUSTODY_BYTES or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('ADAPT_TRANSPORT_REQUIRES_PINNED_EXPORT_CUSTODY')
    manifest = json.loads(data)
    members = manifest['members']
    ordinals = [6*i+i%6 for i in range(16)]
    files = {*NAMES,MANIFEST,*(f'archive/{i:03d}.npz' for i in ordinals)}
    arrays = set(members)-files
    clock = phase==CLOCK_PHASE and arm in CLOCK_ARMS
    valid = clock or (phase in PHASES and arm is None)
    if (not valid or manifest['schema']!=(CLOCK_EXPORT if clock else EXPORT_SCHEMA)
            or manifest.get('arm')!=arm or manifest['goal'] != GOAL
            or manifest['phase'] != phase or manifest['archived_rgb_ordinals'] != ordinals
            or not files <= members.keys() or len(arrays) != 29
            or any(not re.fullmatch(r'package/parameters/[0-9]{6}\.npy',name) for name in arrays)
            or any(type(row['bytes']) is not int or not 0 < row['bytes'] <= MAX_BYTES
                   or not re.fullmatch('[0-9a-f]{64}',row['sha256']) for row in members.values())
            or sum(row['bytes'] for row in members.values()) > MAX_BYTES
            or any(manifest[k] for k in ('optimizer_updates','model_generations','native_requests','native_ticks'))):
        raise ValueError('ADAPT_TRANSPORT_UNEXPECTED_ARCHIVE_MEMBERS_OR_PHASE')
    return manifest


def validate_package(root, custody):
    members = custody['members']
    package = json.loads((root/MANIFEST).read_bytes())
    formal,restore,panel = (
        json.loads((root/'archive'/name).read_bytes())
        for name in ('FORMAL.private.json','RESTORE_PROOF.private.json','PANEL.private.json')
    )
    phase, checkpoint = custody['phase'],custody['checkpoint']
    arm = custody.get('arm')
    clock = phase==CLOCK_PHASE
    rows = package['arrays']
    exported = {f"package/{row['file']}":row['sha256'] for row in rows}
    landed = {name:meta['sha256'] for name,meta in members.items() if name.startswith('package/parameters/')}
    if (package['schema']!=(CLOCK_PACKAGE if clock else PACKAGE_SCHEMA) or package['phase']!=phase
            or any(package[k] is not False for k in ('physics','hardware_allowed','native_admission'))
            or package['checkpoint_sha256'] != checkpoint or len(rows) != 29 or exported != landed
            or custody['package_sha256'] != members[MANIFEST]['sha256']
            or any(members[name]['sha256'] != package[k] for name,k in NAMES.items())
            or formal['phase'] != phase or formal['checkpoint']['sha256'] != checkpoint
            or formal['status'] != 'ADAPT_OFFLINE_RESULTS_SEALED' or formal['role'] != 'formal'
            or formal['exit_code'] or formal['model_generations'] != 376 or formal['old_TEST_access']
            or restore['phase'] != phase or restore['checkpoint'] != formal['checkpoint']
            or restore['panel'] != formal['panel'] or formal['panel']['sha256'] != package['panel_sha256']
            or restore['ordinary_predictions'] != 288 or restore['independent_restore_exact'] != 48
            or restore['phase_train_predictions'] != 88 or panel['old_TEST_access']
            or panel['restore_ordinals'] != custody['archived_rgb_ordinals']):
        raise ValueError('ADAPT_TRANSPORT_FORMAL_ENDPOINT_AND_DEPLOYMENT_PACKAGE_DISAGREE')
    if clock and (package['arm']!=arm or arm not in CLOCK_ARMS or package['plan']!=GOAL
            or package['formal_ADAPT_parent_eligible'] is not False or package['parent_phase']!=2000
            or package['branch_updates']!=500 or package['comparison_sha256']!=custody['comparison']
            or formal['scope']!='SLA_CLOCK500_BRANCH_OFFLINE_V1'
            or restore['schema']!='SLA_CLOCK500_BRANCH_OUTPUT_RESTORE_PROOF_V1'
            or any(r['arm']!=arm or r['comparison']['sha256']!=custody['comparison'] for r in (formal,restore))):
        raise ValueError('CLOCK_TRANSPORT_PACKAGE_AND_INDEPENDENT_EVAL_ARM_CHANGED')
    index = {}
    for ordinal in custody['archived_rgb_ordinals']:
        row = formal['rows']['target'][ordinal]
        rel = f'archive/{ordinal:03d}.npz'
        if row['output']['sha256'] != members[rel]['sha256'] or row['ordinal'] != ordinal:
            raise ValueError('ADAPT_TRANSPORT_PREDICTION_NOT_FROM_THE_FIXED_FORMAL_RGB_PANEL')
        index[str(ordinal)] = dict(path=rel,sha256=members[rel]['sha256'])
    return index


def send(root, stream, phase, expected, *, arm=None):
    data = (root/CUSTODY).read_bytes()
    custody = read_custody(data,expected,phase,arm=arm)
    with tarfile.open(fileobj=stream,mode='w|',bufsize=1024**2) as archive:
        record = tarfile.TarInfo(CUSTODY)
        record.size,record.mode = len(data),0o600
        archive.addfile(record,io.BytesIO(data))
        for name,row in custody['members'].items():
            src = root/name
            member = archive.gettarinfo(str(src),name)
            if not member.isfile() or member.size != row['bytes']:
                raise ValueError('ADAPT_TRANSPORT_SOURCE_MEMBER_CHANGED')
            with src.open('rb') as handle:
                archive.addfile(member,handle)


def receive(destination, stream, phase, expected, *, arm=None):
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix='.adapt-incoming-',dir=destination.parent) as tmp:
        root = Path(tmp)/'bundle'
        root.mkdir()
        with tarfile.open(fileobj=stream,mode='r|',bufsize=1024**2) as archive:
            meta = archive.next()
            if meta is None or meta.name != CUSTODY or not meta.isfile() or not 0 < meta.size <= MAX_CUSTODY_BYTES:
                raise ValueError('ADAPT_TRANSPORT_CUSTODY_MUST_BE_FIRST_BOUNDED_MEMBER')
            data = archive.extractfile(meta).read()
            custody = read_custody(data,expected,phase,arm=arm)
            seen = set()
            for member in iter(archive.next,None):
                name = member.name
                row = custody['members'].get(name)
                if row is None or name in seen or not member.isfile() or member.size != row['bytes']:
                    raise ValueError('ADAPT_TRANSPORT_UNEXPECTED_OR_DUPLICATE_TAR_MEMBER')
                dst = root/name
                dst.parent.mkdir(exist_ok=True,parents=True)
                digest = hashlib.sha256()
                with archive.extractfile(member) as src, dst.open('xb') as out:
                    for block in iter(lambda:src.read(4*1024**2),b''):
                        out.write(block)
                        digest.update(block)
                    out.flush()
                    os.fsync(out.fileno())
                if digest.hexdigest() != row['sha256']:
                    raise ValueError('ADAPT_TRANSPORT_RECEIVED_MEMBER_HASH_CHANGED')
                seen.add(name)
            if seen != custody['members'].keys():
                raise ValueError('ADAPT_TRANSPORT_MISSING_MEMBERS')
        with (root/CUSTODY).open('xb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        archived = validate_package(root,custody)
        receipt = dict(schema=RECEIPT_SCHEMA,goal=GOAL,phase=phase,export_custody_sha256=expected,
            checkpoint_sha256=custody['checkpoint'],files=len(seen),bytes=sum(r['bytes'] for r in custody['members'].values()),
            package_sha256=custody['package_sha256'],archived=archived,
            optimizer_updates=0,model_generations=0,native_requests=0,native_ticks=0)
        if arm is not None:
            receipt.update(schema=CLOCK_RECEIPT,arm=arm,comparison_sha256=custody['comparison'])
        write(root/'CORE_CUSTODY.private.json',receipt)
        for directory in (root/'package/parameters',root/'package',root/'archive',root):
            sync(directory)
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(root,destination)
        sync(destination.parent)
    return pin(destination/'CORE_CUSTODY.private.json')


def sync(path):
    fd = os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('send','receive'))
    parser.add_argument('path',type=Path)
    parser.add_argument('phase',type=int,choices=(*PHASES,CLOCK_PHASE))
    parser.add_argument('--arm',choices=CLOCK_ARMS)
    parser.add_argument('custody_sha256')
    args = parser.parse_args()
    os.umask(0o077)
    if args.operation == 'send':
        send(args.path,sys.stdout.buffer,args.phase,args.custody_sha256,arm=args.arm)
    else:
        print(json.dumps(receive(args.path,sys.stdin.buffer,args.phase,args.custody_sha256,arm=args.arm)))
