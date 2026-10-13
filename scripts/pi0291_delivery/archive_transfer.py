"""Private PI0291 data delivery, independent of the frozen physical runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import tarfile
import time

from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_url
import requests
from deadline_guard import atomic_write, install_deadline, EXTERNAL_OUTPUT_UPPER_BYTES, resource_bounds

DEADLINE = 1791851837.0
LIMIT_BYTES = 40 * 1024**3
LIMIT_CPU_S = 48 * 3600
NATIVE_METADATA = {'DATASET_MANIFEST.json', 'STATISTICS_PROVENANCE.json', 'statistics.json',
                   'modality.json', 'SPLIT.json', 'task_map.json', 'EXPORT_RECEIPT.json'}
EPISODE_METADATA = {'CARD.json', 'RESULT.json', 'SCORE.json', 'RENDER_PROCESS.json',
                    'samples.json', 'references.json', 'physics.npz', 'highrate.npz',
                    'scene.xml.json', 'TEACHER_EVENTS.json', 'ERROR.json',
                    'ACTUAL_DUAL_RGB.mp4', 'VIDEO_RECEIPT.json', 'FILE_MANIFEST.json'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    atomic_write(path, value)


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 << 20), b''):
        h.update(block)
    return h.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return digest(stream)


def contained_file(base, relative):
    base, relative = Path(base), Path(relative)
    assert not relative.is_absolute() and '..' not in relative.parts
    path = base / relative
    assert path.resolve().is_relative_to(base.resolve()) and path.is_file()
    cursor = path
    while cursor != base:
        assert not cursor.is_symlink()
        cursor = cursor.parent
    return path


def guard(root, began_cpu, planned_extra=0):
    deadline, remaining = resource_bounds(root)
    assert time.time() < deadline, 'Bound delivery deadline expired'
    ledger = read(root/'RESOURCE_ACCOUNTING.json')
    own = resource.getrusage(resource.RUSAGE_SELF)
    used = own.ru_utime+own.ru_stime-began_cpu
    assert used < remaining, 'Bound delivery CPU allowance exhausted'
    assert ledger['CPU_s']+ledger['unmetered_dependency_setup_CPU_s_upper']+used < LIMIT_CPU_S
    size = sum(path.stat().st_size for path in root.rglob('*') if path.is_file() and not path.is_symlink())
    assert size+planned_extra+EXTERNAL_OUTPUT_UPPER_BYTES < LIMIT_BYTES
    free = __import__('shutil').disk_usage(root).free
    assert free > planned_extra+10*1024**3


def verified_tar(output, members):
    if not output.exists():
        with tarfile.open(output, 'w:gz', compresslevel=3) as archive:
            for name, path in sorted(members.items()):
                assert path.is_file() and not path.is_symlink()
                archive.add(path, arcname=name, recursive=False)
    actual = {}
    with tarfile.open(output, 'r|gz') as archive:
        for member in archive:
            assert member.isfile() and member.name in members and member.name not in actual
            actual[member.name] = digest(archive.extractfile(member))
    expected = {name: sha(path) for name, path in members.items()}
    assert actual == expected
    return dict(files=len(actual), member_sha256=actual, sha256=sha(output), bytes=output.stat().st_size)


def main(args):
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[-2:]))
    begin = time.monotonic()
    root = Path(args.root)
    install_deadline(root, 'archive_transfer')
    usage = resource.getrusage(resource.RUSAGE_SELF)
    began_cpu = usage.ru_utime+usage.ru_stime
    guard(root,began_cpu)
    archive = root / 'delivery'
    archive.mkdir(exist_ok=True)
    seal = read(root / 'TEACHER_AND_LAYOUT_SEAL.json')
    admission = read(root / 'DATASET_ADMISSION.json')
    smoke = read(root / 'evidence/DATALOADER_SMOKE.json')
    assert smoke['status'] == 'NATIVE_DATALOADER_PASS'
    assert admission['source_sha'] == seal['source_sha']
    native = Path(args.dataset_root)
    native_manifest = read(native/'DATASET_MANIFEST.json')
    assert smoke['dataset_manifest_sha256'] == sha(native/'DATASET_MANIFEST.json')
    assert smoke['statistics_sha256'] == sha(native/'statistics.json')
    assert smoke['statistics_provenance_sha256'] == sha(native/'STATISTICS_PROVENANCE.json')
    assert native_manifest['admission_sha256'] == sha(root/'DATASET_ADMISSION.json')
    assert native_manifest['domain'] == 'FIXED_BASE_ARX5_TASK_SIM'
    assert native_manifest['official_source_sha'] == smoke['official_source_sha'] == '51d4c89f72fda44cbf77285c6a8114b52676b8a1'
    assert smoke['codec']['TRAIN']['full_H40_windows_actually_checked'] == native_manifest['TRAIN_windows']
    assert all(row['source_sha'] == seal['source_sha'] for row in native_manifest['episodes'])
    assert sum(row['split']=='TRAIN' for row in native_manifest['episodes']) >= 2
    token = Path(args.existing_token_file).read_text().strip()
    api = HfApi(token=token)
    assert api.whoami()['name'] == args.repo.split('/')[0]
    api.create_repo(args.repo, repo_type='dataset', private=True, exist_ok=True)
    info = api.repo_info(args.repo, repo_type='dataset')
    assert info.private
    prefix = 'PI0291/'+root.name+'/v1-'+seal['source_sha'][:8]
    assert not any(name.startswith(prefix+'/') for name in api.list_repo_files(args.repo, repo_type='dataset')), 'Existing prefix: preserve immutable transfer'
    payload, proofs = {}, {}
    assert all(name.endswith('.py') and name.startswith(('scripts/pi0291_arx5_data/', 'src/go2arx5_vla/')) for name in seal['code_files'])
    code_members = {name:contained_file(root/'code',name) for name in seal['code_files']}
    assert {name:sha(path) for name, path in code_members.items()} == seal['code_files']
    frozen = archive / 'frozen_source.tar.gz'
    proofs['source'] = verified_tar(frozen, code_members)
    payload['frozen_source.tar.gz'] = frozen
    ledger = read(root / 'RESOURCE_ACCOUNTING.json')
    assert not any(row['status']=='STARTED' for row in ledger['starts'])
    estimated_archive_bytes = sum(path.stat().st_size for path in (root/'episodes').rglob('*') if path.is_file())
    estimated_archive_bytes += sum(path.stat().st_size for path in native.rglob('*') if path.is_file())
    guard(root,began_cpu,int(estimated_archive_bytes*1.1)+10*1024**2)
    for row in ledger['starts']:
        guard(root,began_cpu)
        folder = root / 'episodes' / row['id']
        assert row['status'] != 'STARTED', 'Never package a live episode'
        expected = read(folder/'FILE_MANIFEST.json')
        assert all(r['path'] in EPISODE_METADATA or re.fullmatch(r'(?:body_rgb|wrist_rgb)/[0-9]{4}\.png',r['path']) for r in expected)
        members = {folder.name+'/'+r['path']:contained_file(folder,r['path']) for r in expected}
        for record in expected:
            assert sha(folder/record['path']) == record['sha256']
        members[folder.name+'/FILE_MANIFEST.json'] = folder/'FILE_MANIFEST.json'
        out = archive / (folder.name+'.tar.gz')
        proofs[folder.name] = verified_tar(out, members)
        payload['episodes/'+out.name] = out
    allowed_native = NATIVE_METADATA | {'episodes/'+row['id']+'.npz' for row in native_manifest['episodes']}
    actual_native = {path.relative_to(native).as_posix() for path in native.rglob('*') if path.is_file()}
    assert actual_native == allowed_native, 'Unexpected export file: never silently upload'
    for row in native_manifest['episodes']:
        assert row['file'] == 'episodes/'+row['id']+'.npz'
        assert sha(contained_file(native,row['file'])) == row['sha256']
    members = {name:contained_file(native,name) for name in allowed_native}
    native_tar = archive/'native_dataset.tar.gz'
    proofs['native_dataset'] = verified_tar(native_tar, members)
    payload['native_dataset.tar.gz'] = native_tar
    delivery_members = {name:Path(__file__).with_name(name) for name in
                        ('archive_transfer.py','bind_consumer.py','readback_transfer.py','deadline_guard.py')}
    proofs['delivery_source'] = verified_tar(archive/'delivery_source.tar.gz',delivery_members)
    payload['delivery_source.tar.gz'] = archive/'delivery_source.tar.gz'
    for name in ('TEACHER_AND_LAYOUT_SEAL.json', 'DATASET_ADMISSION.json', 'RESOURCE_ACCOUNTING.json'):
        payload['receipts/'+name] = root/name
    if os.environ.get('PI0291_CLOSEOUT_AUTHORIZATION_SHA256'):
        payload['receipts/DATA_CLOSEOUT_AUTHORIZATION.json'] = root/'evidence/DATA_CLOSEOUT_AUTHORIZATION.json'
    for name in ('DATALOADER_SMOKE.json', 'PILOT_SCORE.json', 'DATASET_CARD.md', 'TRAINING_HANDOFF.json'):
        payload['receipts/'+name] = root/'evidence'/name
    proof_file = archive/'ARCHIVE_MEMBER_MANIFEST.json'
    write(proof_file, proofs)
    payload['ARCHIVE_MEMBER_MANIFEST.json'] = proof_file
    files = [dict(path=prefix+'/'+name, bytes=path.stat().st_size, sha256=sha(path)) for name,path in payload.items()]
    guard(root,began_cpu)
    license_scope = dict(visibility='private', assets='No robot mesh, private WBC tree, weights, tokenizer or authentication included',
        physics_replay='Requires existing privately bound native asset identity; external asset-free replay NOT_CLAIMED',
        data='Original simulation RGB/physics and independently generated code; authorized PI0291 research handoff')
    data_revision = api.create_commit(args.repo, repo_type='dataset', parent_commit=info.sha,
        commit_message='PI0291 actual ARX5 evidence and native data before READY',
        operations=[CommitOperationAdd(path_in_repo=prefix+'/'+name,path_or_fileobj=str(path)) for name,path in payload.items()]).oid
    journal = dict(status='DATA_COMMIT_CREATED_VERIFICATION_PENDING', repo_id=args.repo,
                   prefix=prefix, data_revision=data_revision, files=files,
                   source_sha=seal['source_sha'], READY_claim=False)
    write(root/'evidence/TRANSFER_PROGRESS.json',journal)

    def verify(revision, rows):
        remote = {entry.path:entry for entry in api.list_repo_tree(args.repo, repo_type='dataset', revision=revision,
                  path_in_repo=prefix, recursive=True) if hasattr(entry,'size')}
        assert set(remote) == {row['path'] for row in rows}
        for row in rows:
            item = remote[row['path']]
            assert item.size == row['bytes']
            if item.lfs:
                actual = item.lfs.sha256 if hasattr(item.lfs,'sha256') else item.lfs['sha256']
                assert actual == row['sha256']
            else:
                response = requests.get(hf_hub_url(args.repo,row['path'],repo_type='dataset',revision=revision),
                    headers={'Authorization':'Bearer '+token}, timeout=60)
                response.raise_for_status()
                assert hashlib.sha256(response.content).hexdigest() == row['sha256']

    verify(data_revision, files)
    journal['status'] = 'DATA_COMMIT_ALL_SHA_VERIFIED_READY_PENDING'
    write(root/'evidence/TRANSFER_PROGRESS.json',journal)
    guard(root,began_cpu)
    actual_counts = {split:sum(row['admitted'] and row['split']==split for row in admission['episodes']) for split in ('TRAIN','DEV')}
    if admission['dataset_status'] == 'ARX5_DATASET_READY_FOR_FIRST_TRAINING':
        assert actual_counts['TRAIN'] >= 56 and actual_counts['DEV'] >= 14
        assert sum(row['id'].startswith('T') for row in ledger['starts']) == 64
        assert sum(row['id'].startswith('D') for row in ledger['starts']) == 16
        assert read(root/'PILOT_GATE.json')['status'] == 'PASS'
    ready = dict(status='READY_FOR_FIXED_REVISION_READBACK', dataset_component_status=admission['dataset_status'],
        complete_episode_download='NOT_RUN', Hopper_small_READY='NOT_RUN',
        repo_id=args.repo, repo_type='dataset', private=True,
        prefix=prefix, data_revision=data_revision, source_sha=seal['source_sha'], files=files,
        dataset_counts=admission['counts'], admitted_episode_counts=actual_counts,
        NATIVE_DATALOADER_PASS=True, license_scope=license_scope,
        raw_attempts=len(ledger['starts']), READY_uploaded_last=True, model_loads=0, neural_forward=0,
        training=0, WBC=0, hardware=0, selected='NONE', deployment='NONE')
    if os.environ.get('PI0291_CLOSEOUT_AUTHORIZATION_SHA256'):
        ready['CPU_only_closeout_authority_sha256'] = os.environ['PI0291_CLOSEOUT_AUTHORIZATION_SHA256']
    ready_file = archive/'READY.json'
    write(ready_file,ready)
    final = api.create_commit(args.repo,repo_type='dataset',parent_commit=data_revision,
        commit_message='PI0291 READY last after complete archive member and remote SHA checks',
        operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(ready_file))]).oid
    journal.update(status='READY_COMMIT_CREATED_FINAL_VERIFICATION_PENDING',revision=final)
    write(root/'evidence/TRANSFER_PROGRESS.json',journal)
    files.append(dict(path=prefix+'/READY.json',bytes=ready_file.stat().st_size,sha256=sha(ready_file)))
    verify(final,files)
    assert api.repo_info(args.repo,repo_type='dataset',revision=final).private
    own = resource.getrusage(resource.RUSAGE_SELF)
    receipt = dict(status='HF_UPLOAD_ALL_FILES_SHA_VERIFIED_COMPLETE_EPISODE_DOWNLOAD_STILL_PENDING',
        repo_id=args.repo,repo_type='dataset',private=True,prefix=prefix,revision=final,data_revision=data_revision,
        files=files,source_sha=seal['source_sha'],READY_uploaded_last=True,license_scope=license_scope,
        CPU_s=own.ru_utime+own.ru_stime,elapsed_s=time.monotonic()-begin,
        delivery_code_sha256=sha(__file__), complete_episode_download='NOT_RUN', Hopper_small_READY='NOT_RUN')
    write(root/'evidence/TRANSFER_MANIFEST.json',receipt)
    journal.update(status='ALL_REMOTE_SHA_VERIFIED_DOWNLOAD_READBACK_PENDING',READY_claim=True)
    write(root/'evidence/TRANSFER_PROGRESS.json',journal)
    print(json.dumps(dict(repo=args.repo,revision=final,prefix=prefix,files=len(files),CPU_s=receipt['CPU_s'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--dataset-root',required=True)
    parser.add_argument('--repo',required=True)
    parser.add_argument('--existing-token-file',required=True)
    args = parser.parse_args()
    main(args)
