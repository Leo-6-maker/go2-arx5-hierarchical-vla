"""Seal new lossless copies of terminal Core traces and restore them for original readers."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from scripts.pi0292.durable import OwnerLock, read
from scripts.sla.core_multienv_trace_storage import FILES, GIB, compress
from scripts.sla.hopper_parent_custody import digest, sync_dir, write_once

SCHEMA = 'SLA_CORE_MULTIENV_TRACE_ARCHIVE_V1'
PLAN = '4fe2c8cf1b69c168b2e3079e061bb7c702089cd3'
MEMBERS = (*FILES, 'init.trace.summary', 'MOBILE_SAFETY_TRACE.private.jsonl')
MAX_BYTES = 2*GIB


def owned_scratch(path, forbidden):
    parent = path.parent.resolve(strict=True)
    if (os.uname().nodename != 'corelab-4080s' or os.getuid() != 2009
            or parent.stat().st_uid != os.getuid() or path.is_symlink()
            or parent.is_relative_to(forbidden)):
        raise PermissionError('Use a separate, project-owned Core scratch directory')
    usage = shutil.disk_usage(parent)
    floor = max(50*GIB, int(.15*usage.total))+8*GIB
    if usage.free < floor+5*GIB:
        raise PermissionError('Keep the original disk floor and room for materialization')
    return parent/path.name


def pack(case, expected, destination):
    case = case.resolve(strict=True)
    parent = case.parent.parent
    spec = read(case/'EXECUTION_SPEC.json')
    lease = read(case/'control/LEASE.json')
    terminal = case/'RESULT.private.json'
    result = read(terminal)
    registered = next(r for r in read(parent/'RECORDS.private.json') if r['condition']==case.name)
    if (case.parent.name != 'cases' or spec['goal_plan_sha'] != PLAN or spec['registered_root'] != str(case)
            or time.time() >= spec['campaign_deadline_epoch'] or case.stat().st_uid != os.getuid()
            or lease['status'] != 'CLOSED' or lease['spec_sha256'] != digest(case/'EXECUTION_SPEC.json')
            or digest(terminal) != expected or registered['result'] != dict(path=str(terminal),sha256=expected)
            or result['physical_requests'] != 1 or registered['ticks'] != result['actual_native_integrations']
            or result['condition']['collection_episode'] != case.name):
        raise PermissionError('Expected a closed original Core collection condition')
    destination = owned_scratch(destination, parent)
    paths = [case/'native-01'/name for name in MEMBERS]
    if (any(x.is_symlink() or not x.is_file() for x in paths)
            or sum(x.stat().st_size for x in paths)>MAX_BYTES):
        raise ValueError('Unexpected or oversized trace members')
    with closing(OwnerLock(destination.parent/'archive.lock')):
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(prefix='.archive-', dir=destination.parent) as stage:
            temp = Path(stage)/'ready'
            temp.mkdir(mode=0o700)
            files = {}
            for original in paths:
                compressed = temp/(original.name+'.zst')
                row = compress(original, compressed)
                row['packed_sha256'] = digest(compressed, flush=True)
                files[original.name] = row
            manifest = dict(schema=SCHEMA, plan=PLAN, condition=case.name, terminal_sha256=expected,
                            source=spec['source_sha'], state='sealed', native_ticks=registered['ticks'],
                            codec='zstd -1 -T1', original_files_rewritten=0, files=files)
            write_once(temp/'MANIFEST.json',manifest)
            temp.rename(destination)
            sync_dir(destination.parent)
    return manifest


def materialize(archive, expected, destination):
    archive = archive.resolve(strict=True)
    destination = owned_scratch(destination, archive)
    manifest = read(archive/'MANIFEST.json')
    if (destination.parent!=archive.parent or digest(archive/'MANIFEST.json')!=expected or manifest['schema']!=SCHEMA
            or manifest['plan']!=PLAN or manifest['state']!='sealed'
            or set(manifest['files'])!=set(MEMBERS)
            or {p.name for p in archive.iterdir()}!={'MANIFEST.json',*(m+'.zst' for m in MEMBERS)}
            or not 0 < sum(x['original_bytes'] for x in manifest['files'].values()) <= MAX_BYTES):
        raise ValueError('Unexpected archive manifest or member set')
    with closing(OwnerLock(destination.parent/'archive.lock')):
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(prefix='.materialize-', dir=destination.parent) as stage:
            temp = Path(stage)/'ready'
            temp.mkdir(mode=0o700)
            for name, row in manifest['files'].items():
                member = archive/(name+'.zst')
                if (member.stat().st_size!=row['packed_bytes'] or digest(member)!=row['packed_sha256']
                        or row['name']!=name or row['original_sha256']!=row['restored_sha256']):
                    raise ValueError('Changed compressed member: '+name)
                target = temp/name
                with target.open('xb') as out:
                    subprocess.run(['zstd','-q','-T1','-d','-c','--',str(member)], stdout=out,
                                   check=True,timeout=45)
                if target.stat().st_size!=row['original_bytes'] or digest(target,flush=True)!=row['original_sha256']:
                    raise ValueError('Changed restored trace: '+name)
            sync_dir(temp)
            temp.rename(destination)
            sync_dir(destination.parent)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('pack','materialize'))
    p.add_argument('source',type=Path)
    p.add_argument('sha256')
    p.add_argument('destination',type=Path)
    a = p.parse_args()
    os.umask(0o077)
    operation = pack if a.mode=='pack' else materialize
    operation(a.source,a.sha256,a.destination)
    receipt = a.destination/'MANIFEST.json' if a.mode=='pack' else a.source/'MANIFEST.json'
    print(digest(receipt))
