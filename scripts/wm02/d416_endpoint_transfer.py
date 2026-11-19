"""Move sealed endpoint evidence and RGB16 references between authorized sites."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.world_model.sla_followup_native_v1 import CANDIDATES, JOINT_ARMS, PLAN, trained_endpoint
from scripts.pi0292.durable import read, write


SCHEMA = 'SLA_D416_PORTABLE_ENDPOINT_RGB_V1'
ARM_SET = {'D5': CANDIDATES, 'J': JOINT_ARMS}
MAX_ARCHIVE_BYTES = 2*1024**3
ARCHIVE_MANIFEST = 'TRANSFER.private.json'


def pin(path):
    return dict(path=str(path), sha256=sha(path))


def byte_record(path, checksum=None):
    if not path.is_file() or path.is_symlink():
        raise ValueError('D416_FILE_REQUIRED:'+str(path))
    actual = sha(path)
    if checksum is not None and checksum != actual:
        raise ValueError('D416_TRANSFER_INPUT_HASH_MISMATCH:'+str(path))
    return dict(sha256=actual, bytes=path.stat().st_size)


def scheduler_terminal(config):
    job = config['job_id']
    process = subprocess.run(['/opt/pbs/bin/qstat','-xf','-F','json',job],
                             check=True,capture_output=True,text=True,timeout=30)
    current = json.loads(process.stdout)['Jobs'][job]
    if (current['job_state'] != 'F' or current['Exit_status'] != 0
            or current['Job_Owner'].split('@',1)[0] != config['user']
            or current['project'] != config['project'] or os.getuid() != config['uid']
            or int(current['Resource_List']['ngpus']) != 1):
        raise ValueError('FIXED_SUCCESSFUL_SITE_JOB_TERMINAL_REQUIRED_BEFORE_TRANSFER')
    return dict(id=job, state='F', exit_code=0, project=config['project'],
                resources=current.get('resources_used',{}))


def export_bundle(mode, source, stream):
    source = source.resolve(strict=True)
    config_path = source/'EXECUTION.private.json'
    config = read(config_path)
    parent = source/'RESULT.private.json'
    terminal = read(parent)
    wanted = ARM_SET[mode]
    if (config['plan_commit'] != PLAN or config['root'] != str(source)
            or terminal['exit_code'] or terminal['new_native_requests'] or terminal['new_integrations']
            or terminal['source_sha'] != config['source_sha'] or terminal['job_id'] != config['job_id']
            or tuple(row['arm'] for row in terminal['records']) != wanted):
        raise ValueError('D416_SEALED_MATCHED_PAIR_REQUIRED_BEFORE_TRANSFER')
    site = scheduler_terminal(config)
    files = {}
    records = {}

    def add(relative, file, checksum=None):
        if relative in records:
            raise ValueError('DUPLICATE_D416_TRANSFER_MEMBER:'+relative)
        if not file.is_absolute():
            raise ValueError('ACTUAL_D416_SITE_ARTIFACT_PATH_REQUIRED')
        records[relative] = byte_record(file, checksum)
        files[relative] = file

    add('VANDA_PAIR_RESULT.private.json', parent)
    context = read(source/wanted[0]/'EXECUTION_SPEC.json')['portable_evaluation']
    add('CONTEXTS.private.json',Path(context['path']),context['sha256'])
    for row in terminal['records']:
        arm = row['arm']
        path = source/arm
        card = read(path/'EXECUTION_SPEC.json')
        if context != card['portable_evaluation'] or card['source_sha'] != config['source_sha']:
            raise ValueError('ACTUAL_TRAINING_CARD_IDENTITY_REQUIRED')
        stages = [('train','RESULT.private.json'), ('restore','RESTORE.private.json')]
        if mode == 'J':
            stages.append(('restore_source','SOURCE_RESTORE.private.json'))
        for role, name in stages:
            child = row[role]['result']
            if row[role]['exit_code'] or Path(child['path']) != path/name:
                raise ValueError('D416_COLD_SUCCESSFUL_EXACT_CHILD_RECEIPT_REQUIRED')
            add(f'endpoints/{arm}/{name}',path/name,child['sha256'])
        result = row['train']['result']
        actual = read_pin(result)
        package = actual['package']
        asset = path/'RGB_PACKAGE'
        if Path(package['path']) != asset:
            raise ValueError('FROZEN_ENDPOINT_MUST_BE_OWNED_BY_EXACT_ARM_DIRECTORY')
        restored = row.get('restore_source',{}).get('result')
        manifest, _ = trained_endpoint(arm,package,result,row['restore']['result'],source_restore=restored)
        members = {str(x['file']):x['sha256'] for x in manifest['adapter_arrays']}
        if len(members) != 28 or any(Path(name).parts != ('adapters',Path(name).name) for name in members):
            raise ValueError('TWENTY_EIGHT_PINNED_ADAPTER_MEMBERS_REQUIRED')
        members.update({'MANIFEST.private.json':package['sha256'],'STATISTICS.json':manifest['statistics_sha256'],
                        'paligemma_tokenizer.model':manifest['tokenizer_sha256']})
        for name,digest in members.items():
            file = asset/name
            if not file.resolve().is_relative_to(asset):
                raise ValueError('D416_OUTSIDE_REGISTERED_MODEL_ASSET')
            add(f'endpoints/{arm}/RGB_PACKAGE/{name}',file,digest)
        target = actual['RGB16_list']
        endpoint_cases = Path(target['path'])
        if endpoint_cases != path/'ENDPOINT_RGB16.private.json':
            raise ValueError('D416_EXACT_TARGET_RGB16_LIST_LOCATION_REQUIRED')
        add(f'reference/{arm}/CASES.private.json',endpoint_cases,target['sha256'])
        cases = read_pin(target)
        if len(cases) != 16 or len({(r['index'],r['noise']) for r in cases}) != 16:
            raise ValueError('D416_EXACT_TARGET_RGB16_REFERENCE_BANK_REQUIRED')
        for case in cases:
            prediction = case['output']
            file = Path(prediction['path'])
            name = f"{case['index']:04d}-{case['noise']}.npz"
            if file != path/'data'/name:
                raise ValueError('D416_EXACT_FROZEN_TARGET_PREDICTION_REQUIRED')
            add(f'reference/{arm}/{name}',file,prediction['sha256'])

    transfer = dict(schema=SCHEMA, plan_commit=PLAN, mode=mode, arms=list(wanted), job=site,
                    vanda_source_sha=config['source_sha'], vanda_execution_sha256=sha(config_path),
                    members=records, transfer_model_generations=0, transfer_native_requests=0)
    total = sum(r['bytes'] for r in records.values())
    if total >= MAX_ARCHIVE_BYTES:
        raise ValueError('BOUNDED_PORTABLE_ENDPOINT_PAYLOAD_REQUIRED')
    contents = (json.dumps(transfer,indent=2,allow_nan=False)+'\n').encode()
    with tarfile.open(fileobj=stream,mode='w|') as archive:
        entry = tarfile.TarInfo(ARCHIVE_MANIFEST)
        entry.size = len(contents)
        entry.mode = 0o600
        archive.addfile(entry,io.BytesIO(contents))
        for name,path in files.items():
            archive.add(path,arcname=name,recursive=False)


def receive_bundle(mode, destination, stream):
    """A partial transfer stays in its own directory without a custody receipt."""
    destination.mkdir(mode=0o700)
    with tarfile.open(fileobj=stream,mode='r|') as archive:
        first = archive.next()
        if first is None or first.name != ARCHIVE_MANIFEST or not first.isfile() or first.size > 65536:
            raise ValueError('D416_ARCHIVE_MUST_START_WITH_BOUNDED_MANIFEST')
        contents = archive.extractfile(first).read()
        manifest = json.loads(contents)
        expected = manifest['members']
        if (manifest['schema'] != SCHEMA or manifest['plan_commit'] != PLAN or manifest['mode'] != mode
                or manifest['arms'] != list(ARM_SET[mode]) or manifest['job']['state'] != 'F'
                or manifest['job']['exit_code'] or manifest['transfer_model_generations']
                or manifest['transfer_native_requests'] or sum(r['bytes'] for r in expected.values()) >= MAX_ARCHIVE_BYTES):
            raise ValueError('D416_TRANSFER_MANIFEST_WRONG_IDENTITY_OR_RESOURCE_BOUND')
        seen = set()
        for member in iter(archive.next,None):
            name = member.name
            relative = Path(name)
            if (name not in expected or name in seen or relative.is_absolute() or '..' in relative.parts
                    or not member.isfile() or member.size != expected[name]['bytes']):
                raise ValueError('D416_UNDECLARED_OR_UNSAFE_ARCHIVE_MEMBER:'+name)
            file = destination/relative
            file.parent.mkdir(parents=True,exist_ok=True)
            with file.open('xb') as handle:
                shutil.copyfileobj(archive.extractfile(member),handle,1024**2)
            if byte_record(file) != expected[name]:
                raise ValueError('D416_RECEIVED_MEMBER_HASH_MISMATCH:'+name)
            seen.add(name)
        if seen != set(expected):
            raise ValueError('D416_TRANSFER_MISSING_MEMBERS')
    with (destination/ARCHIVE_MANIFEST).open('xb') as f:
        f.write(contents)
    binding = local_bindings(destination)
    write(destination/'CUSTODY.private.json',binding)
    return binding


def local_bindings(destination):
    destination = destination.resolve(strict=True)
    transfer = pin(destination/ARCHIVE_MANIFEST)
    manifest = read_pin(transfer)
    out = dict(mode=manifest['mode'],plan_commit=PLAN,transfer=transfer,arms={},
               portable_evaluation=pin(destination/'CONTEXTS.private.json'), new_model_generations=0,
               new_native_requests=0,source_job=manifest['job'])
    for arm in ARM_SET[out['mode']]:
        path = destination/'endpoints'/arm
        bundle = path/'RGB_PACKAGE'
        package = dict(path=str(bundle),sha256=sha(bundle/'MANIFEST.private.json'))
        result = pin(path/'RESULT.private.json')
        restore = pin(path/'RESTORE.private.json')
        source = pin(path/'SOURCE_RESTORE.private.json') if out['mode'] == 'J' else None
        trained_endpoint(arm,package,result,restore,source_restore=source)
        out['arms'][arm] = dict(package=package,result=result,restore=restore,source_restore=source,
                               cases=pin(destination/'reference'/arm/'CASES.private.json'))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('export','receive'))
    parser.add_argument('mode',choices=tuple(ARM_SET))
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.action == 'export':
        export_bundle(args.mode,args.root,sys.stdout.buffer)
    else:
        print(json.dumps(receive_bundle(args.mode,args.root,sys.stdin.buffer),allow_nan=False))


if __name__ == '__main__':
    main()
