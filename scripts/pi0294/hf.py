"""Incremental private HF delivery with fixed revisions and READY uploaded last."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import time

from scripts.iteration.queue import digest
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0292.hf_bank import contained, verified_tar

OLD_REVISION = '6075aac0cdc44b8d252d7540efbac208701c30b3'
OLD_PREFIX = 'PI0292/20261013T025824Z-localpc-pi0292/bank-v2-c187d957'
V3_0_REVISION = '6aa94f4f40d791bedd6b4e75489e4ede6504d10d'


def previous_bank(root, admission):
    """v3.1 references the actual v3.0 delivery, including rejected attempts."""
    root=Path(root)
    transfer=read(root/'evidence/HF_v3_0.json')
    folder=root/'delivery/v3_0'
    ready=read(folder/'READY.json')
    prefix='PI0294/'+root.name+'/bank-v3_0-2fd28e00'
    if (transfer['status']!='PI0294_PRIVATE_FIXED_REVISION_READY_LAST_FULL_SHA_PASS'
            or not transfer['READY_claim'] or transfer['revision']!=V3_0_REVISION
            or transfer['prefix']!=prefix or ready['version']!='v3.0'
            or ready['status']!='READY_FIXED_REVISION_CPU_QUALIFIED'
            or ready['repo_id']!=transfer['repo_id'] or ready['prefix']!=prefix
            or ready['data_revision']!=transfer['data_revision']):
        raise ValueError('ACTUAL_FIXED_V3_0_DEPENDENCY_REQUIRED')
    files={row['path']:row for row in transfer['files']}
    for name in ('READY.json','BANK_ADMISSION.json','ARCHIVE_MEMBER_MANIFEST.json'):
        path=folder/name;bound=files[prefix+'/'+name]
        if path.stat().st_size!=bound['bytes'] or digest(path)!=bound['sha256']:
            raise ValueError('PREVIOUS_BANK_RECEIPT_CHANGED')
    old=read(folder/'BANK_ADMISSION.json')
    proofs=read(folder/'ARCHIVE_MEMBER_MANIFEST.json')
    current={row['id']:row for row in admission['episodes']}
    original={row['id']:row for row in old['episodes'] if row['origin']=='new'}
    if len(original)!=80:
        raise ValueError('EXACT_V3_0_EIGHTY_FIRST_ATTEMPTS_REQUIRED')
    for identifier,row in original.items():
        if current.get(identifier)!=row or not row['first_attempt']:
            raise ValueError('REUSED_V3_0_ID_SPLIT_OR_OUTCOME_CHANGED')
        if prefix+'/episodes/'+identifier+'.tar.gz' not in files or identifier not in proofs:
            raise ValueError('REUSED_RAW_ARCHIVE_AND_MEMBER_PROOF_REQUIRED')
    return dict(repo_id=transfer['repo_id'],revision=transfer['revision'],prefix=prefix,
        manifest_sha256=transfer['manifest_sha256'],statistics_sha256=transfer['statistics_sha256'],
        files=transfer['files'],episodes=original,raw_uploaded_again=False),proofs


def reused_raw(row, previous, native, bindings, proof):
    """Hash every existing archive member before binding it without a download."""
    native=Path(native)
    if (digest(native/'DATASET_MANIFEST.json')!=previous['manifest_sha256']
            or digest(native/'statistics.json')!=previous['statistics_sha256']):
        raise ValueError('PREEXISTING_V3_0_NATIVE_IDENTITY_CHANGED')
    prior=next((r for r in read(native/'DATASET_MANIFEST.json')['episodes'] if r['id']==row['id']),None)
    if prior is None or any(prior[k]!=row[k] for k in ('id','seed','group','split','task','family','origin','samples_sha256')):
        raise ValueError('PREEXISTING_V3_0_EPISODE_IDENTITY_CHANGED')
    folder=Path(read(bindings)[row['id']]).resolve(strict=True)
    if digest(folder/'samples.json')!=row['samples_sha256']:
        raise ValueError('PREEXISTING_V3_0_RAW_CHANGED')
    members=proof['member_sha256']
    if not members or any(not name.startswith(row['id']+'/') for name in members):
        raise ValueError('EXACT_REUSED_EPISODE_MEMBER_PREFIX_REQUIRED')
    for name,expected in members.items():
        relative=name[len(row['id'])+1:]
        if digest(contained(folder,relative))!=expected:
            raise ValueError('PREEXISTING_V3_0_ARCHIVE_MEMBER_CHANGED')
    return folder


def package(root, native, old_transfer, output, *, output_budget_bytes=None):
    root, native, output = Path(root), Path(native), Path(output)
    if output.exists():
        raise FileExistsError('PRESERVE_PRIOR_INCREMENTAL_PACKAGE')
    manifest = read(native / 'DATASET_MANIFEST.json')
    seal = read(native / 'BANK_SEAL.json')
    if seal['manifest_sha256'] != digest(native / 'DATASET_MANIFEST.json') or seal['statistics_sha256'] != digest(native / 'statistics.json'):
        raise ValueError('FROZEN_BANK_SEAL_CHANGED')
    if manifest.get('schema') != 'PI0294_FROZEN_MULTITASK_NATIVE_V1' or not manifest['frozen']:
        raise ValueError('FROZEN_MULTITASK_BANK_REQUIRED')
    original = read(old_transfer)
    if original['revision'] != OLD_REVISION or original['prefix'] != OLD_PREFIX:
        raise ValueError('EXACT_QUALIFIED_PI0292_HF_DEPENDENCY_REQUIRED')
    output.mkdir(parents=True, exist_ok=False)
    payload, proofs = {}, {}
    members = {p.relative_to(native).as_posix(): contained(native, p.relative_to(native).as_posix())
               for p in native.rglob('*') if p.is_file()}
    for row in manifest['episodes']:
        if digest(native / row['file']) != row['sha256']:
            raise ValueError('SEALED_NUMERIC_DATA_CHANGED')
    for name, expected in seal['file_sha256'].items():
        if name not in members or digest(members[name]) != expected:
            raise ValueError('SEALED_NATIVE_MEMBER_CHANGED')
    native_archive = output / 'native_dataset.tar.gz'
    proofs['native_dataset'] = verified_tar(native_archive, members)
    payload['native_dataset.tar.gz'] = native_archive
    admission = read(root / 'evidence' / ('BANK_' + manifest['version'].replace('.', '_') + '_ADMISSION.json'))
    previous,previous_proofs=previous_bank(root,admission) if manifest['version']=='v3.1' else (None,{})
    for row in admission['episodes']:
        if row['origin'] != 'new':
            continue
        if not row['first_attempt']:
            continue
        if previous and row['id'] in previous['episodes']:
            proofs[row['id']]=previous_proofs[row['id']]
            continue
        folder = Path(row['raw_root'])
        worker = read(folder / 'WORKER_RECEIPT.json')
        if worker['fixture'] is not False:
            raise ValueError('SYNTHETIC_EPISODE_CANNOT_BE_ARCHIVED_AS_LIVE')
        raw_manifest = read(folder / 'FILE_MANIFEST.json')
        names = {entry['path'] for entry in raw_manifest}
        for entry in raw_manifest:
            if digest(contained(folder, entry['path'])) != entry['sha256']:
                raise ValueError('ARCHIVE_RAW_MEMBER_CHANGED')
        extra = {'FILE_MANIFEST.json', 'WORKER_RECEIPT.json', 'AUDIT.json'}
        if (folder / 'READY.json').exists():
            extra.add('READY.json')
        # Worker streams/auditor logs are receipts outside the scientific payload.
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
        if actual - names - extra - {'auditor.stdout'}:
            raise ValueError('UNMANIFESTED_EPISODE_MEMBER')
        archive = output / (row['id'] + '.tar.gz')
        if output_budget_bytes is not None:
            upper = sum(contained(folder, name).stat().st_size for name in names | extra) + 1024**2
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) + upper > output_budget_bytes:
                raise ValueError('INCREMENTAL_PACKAGE_SPACE_RESERVATION_EXCEEDED')
        proofs[row['id']] = verified_tar(archive,
            {row['id'] + '/' + name: contained(folder, name) for name in names | extra})
        payload['episodes/' + archive.name] = archive
    dependency = dict(repo_id=original['repo_id'], revision=OLD_REVISION, prefix=OLD_PREFIX,
        files=original['files'], original_dependency=original.get('original_dependency'),
        admitted_TRAIN=64, admitted_DEV=16, raw_originals_uploaded_again=False,
        original_manifest_sha256=digest(old_transfer), statistics_sha256=manifest['original_statistics_sha256'])
    write(output / 'ORIGINAL_DEPENDENCY.json', dependency)
    write(output / 'ARCHIVE_MEMBER_MANIFEST.json', proofs)
    write(output / 'BANK_ADMISSION.json', admission)
    for name in ('ORIGINAL_DEPENDENCY.json', 'ARCHIVE_MEMBER_MANIFEST.json', 'BANK_ADMISSION.json'):
        payload[name] = output / name
    if previous:
        write(output/'PREVIOUS_BANK_DEPENDENCY.json',previous)
        payload['PREVIOUS_BANK_DEPENDENCY.json']=output/'PREVIOUS_BANK_DEPENDENCY.json'
    return payload, proofs, dependency


def publish(root, native, old_transfer, output, repo, *, qualification, output_budget_bytes=None,
            source_sha=None,existing_token_file=None):
    os.environ['HF_HUB_OFFLINE'] = '0'
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    root, native, output = Path(root), Path(native), Path(output)
    version = read(native / 'DATASET_MANIFEST.json')['version']
    journal_path = root / 'evidence' / ('HF_' + version.replace('.', '_') + '.json')
    owner = OwnerLock(root / ('HF_' + version.replace('.', '_') + '.lock'))
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    started = time.monotonic()
    journal = dict(status='PACKAGING_INCREMENTAL', fixture=False, READY_claim=False,
                   model_loads=0, integrations=0, version=version)
    journal_owned = False
    try:
        if journal_path.exists():
            raise FileExistsError('PRESERVE_PRIOR_HF_JOURNAL_NO_AUTOMATIC_RETRY')
        gate = read(qualification)
        if gate.get('status') not in ('PASS_PI0294_TRAIN_DEV_AND_TRAINER_LOADER_CPU',
                                      'PASS_BANK_V3_TRAIN_DEV_AND_TRAINER_LOADER_CPU'):
            raise ValueError('ACTUAL_MULTITASK_PROCESSOR_AND_LOADER_CPU_GATE_REQUIRED')
        if gate['manifest_sha256'] != digest(native / 'DATASET_MANIFEST.json') or gate['statistics_sha256'] != digest(native / 'statistics.json'):
            raise ValueError('QUALIFIED_BANK_IDENTITIES_CHANGED')
        write(journal_path, journal)
        journal_owned = True
        payload, proofs, dependency = package(root, native, old_transfer, output,
                                             output_budget_bytes=output_budget_bytes)
        payload['receipts/CPU_QUALIFICATION.json'] = Path(qualification)
        # Reuse configured authentication without logging credential contents.
        token=Path(existing_token_file).read_text().strip() if existing_token_file else None
        api = HfApi(token=token)
        info = api.repo_info(repo, repo_type='dataset')
        if not info.private:
            raise PermissionError('EXISTING_PRIVATE_DATASET_REPOSITORY_REQUIRED')
        source = source_sha or read(root / 'RUN_MANIFEST.json')['code']['vla_commit']
        prefix = 'PI0294/' + root.name + '/bank-' + version.replace('.', '_') + '-' + source[:8]
        if any(name.startswith(prefix + '/') for name in api.list_repo_files(repo, repo_type='dataset')):
            raise FileExistsError('FRESH_FIXED_HF_PREFIX_REQUIRED')
        files = [dict(path=prefix + '/' + name, bytes=p.stat().st_size, sha256=digest(p))
                 for name, p in sorted(payload.items())]
        revision = api.create_commit(repo, repo_type='dataset', parent_commit=info.sha,
            commit_message='PI0294 immutable multitask bank payload before READY',
            operations=[CommitOperationAdd(path_in_repo=prefix + '/' + name, path_or_fileobj=str(path))
                        for name, path in payload.items()]).oid
        journal.update(status='DATA_COMMIT_CREATED', repo_id=repo, prefix=prefix,
                       data_revision=revision, files=files)
        write(journal_path, journal)

        def verify(fixed_revision, expected, path_prefix=prefix):
            remote = {r.path: r for r in api.list_repo_tree(repo, repo_type='dataset',
                revision=fixed_revision, path_in_repo=path_prefix, recursive=True) if hasattr(r, 'size')}
            if set(remote) != {r['path'] for r in expected}:
                raise ValueError('EXACT_FIXED_REVISION_FILE_SET_REQUIRED')
            for row in expected:
                item = remote[row['path']]
                if item.size != row['bytes']:
                    raise ValueError('FIXED_REVISION_REMOTE_SIZE_CHANGED')
                if item.lfs:
                    actual = item.lfs.sha256 if hasattr(item.lfs, 'sha256') else item.lfs['sha256']
                else:
                    actual = digest(hf_hub_download(repo, row['path'], repo_type='dataset', revision=fixed_revision,token=token))
                if actual != row['sha256']:
                    raise ValueError('FIXED_REVISION_REMOTE_SHA_CHANGED')
        verify(revision, files)
        previous=read(output/'PREVIOUS_BANK_DEPENDENCY.json') if (output/'PREVIOUS_BANK_DEPENDENCY.json').exists() else None
        if previous:
            if previous['repo_id']!=repo:raise ValueError('SAME_PRIVATE_PREVIOUS_BANK_REPOSITORY_REQUIRED')
            verify(previous['revision'],previous['files'],previous['prefix'])
        ready = dict(schema='PI0294_HF_READY_V1', status='READY_FIXED_REVISION_CPU_QUALIFIED',
            repo_id=repo, private=True, prefix=prefix, data_revision=revision, files=files,
            original_dependency=dependency, version=version, fixture=False,
            manifest_sha256=digest(native / 'DATASET_MANIFEST.json'), statistics_sha256=digest(native / 'statistics.json'),
            qualification_sha256=digest(qualification), READY_uploaded_last=True,
            active_training_hot_update=False, actual_hopper_loader_readback='PENDING',
            selected='NONE', deployment='NONE')
        if previous:ready['previous_bank_dependency']=previous
        write(output / 'READY.json', ready)
        final = api.create_commit(repo, repo_type='dataset', parent_commit=revision,
            commit_message='PI0294 READY after full fixed-revision payload verification',
            operations=[CommitOperationAdd(path_in_repo=prefix + '/READY.json', path_or_fileobj=str(output / 'READY.json'))]).oid
        files.append(dict(path=prefix + '/READY.json', bytes=(output / 'READY.json').stat().st_size,
                          sha256=digest(output / 'READY.json')))
        verify(final, files)
        journal.update(status='PI0294_PRIVATE_FIXED_REVISION_READY_LAST_FULL_SHA_PASS',
            revision=final, READY_claim=True, files=files, manifest_sha256=ready['manifest_sha256'],
            statistics_sha256=ready['statistics_sha256'], old_raw_reuploaded=False,
            actual_hopper_loader_readback='PENDING')
        return journal
    except Exception as error:
        journal.update(status='FAILED_PRESERVED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        own = resource.getrusage(resource.RUSAGE_SELF)
        journal.update(CPU_s=own.ru_utime + own.ru_stime - cpu0.ru_utime - cpu0.ru_stime,
                       wall_s=time.monotonic() - started)
        if journal_owned:
            write(journal_path, journal)
        owner.close()


def download(delivery_manifest, output, old_native, old_bindings,*,existing_token_file=None,
             previous_native=None,previous_bindings=None):
    """Fetch only this new bank; bind preserved local old RGB explicitly."""
    os.environ['HF_HUB_OFFLINE'] = '0'
    transfer = read(delivery_manifest)
    if transfer['status'] != 'PI0294_PRIVATE_FIXED_REVISION_READY_LAST_FULL_SHA_PASS' or not transfer['READY_claim']:
        raise ValueError('VERIFIED_FIXED_REVISION_DELIVERY_REQUIRED')
    if transfer['version']=='v3.1' and (previous_native is None or previous_bindings is None):
        raise ValueError('EXISTING_LOCAL_V3_0_BINDINGS_REQUIRED_NO_REPEAT_DOWNLOAD')
    if transfer['version']=='v3.1' and not all(p.is_file() for p in (
            Path(previous_native)/'DATASET_MANIFEST.json',Path(previous_native)/'statistics.json',Path(previous_bindings))):
        raise ValueError('EXISTING_LOCAL_V3_0_FILES_REQUIRED_BEFORE_DOWNLOAD')
    from huggingface_hub import HfApi, hf_hub_download
    from scripts.pi0292.data_consumer import extract_checked
    output, old_native = Path(output), Path(old_native)
    if output.exists():
        raise FileExistsError('PRESERVE_FIRST_IMMUTABLE_DATA_DOWNLOAD')
    output.mkdir(parents=True)
    token=Path(existing_token_file).read_text().strip() if existing_token_file else None
    api = HfApi(token=token)
    repo, revision, prefix = transfer['repo_id'], transfer['revision'], transfer['prefix']
    info = api.repo_info(repo, repo_type='dataset', revision=revision)
    if not info.private or info.sha != revision:
        raise ValueError('EXACT_PRIVATE_REVISION_REQUIRED')
    bypath = {row['path']: row for row in transfer['files']}
    def guard(extra=0):
        if time.time() >= 1791939600:
            raise TimeoutError('INCREMENTAL_DOWNLOAD_CUTOFF')
        if shutil.disk_usage(output).free < extra + 10 * 1024**3:
            raise ValueError('INCREMENTAL_DOWNLOAD_SPACE_MARGIN')
    def get(name):
        guard()
        path = Path(hf_hub_download(repo, prefix + '/' + name, repo_type='dataset',
            revision=revision, local_dir=output / 'blobs',token=token))
        bound = bypath[prefix + '/' + name]
        if path.stat().st_size != bound['bytes'] or digest(path) != bound['sha256']:
            raise ValueError('FIXED_REVISION_DOWNLOAD_HASH_CHANGED')
        return path
    ready = read(get('READY.json'))
    previous=None
    if transfer['version']=='v3.1':
        previous=read(get('PREVIOUS_BANK_DEPENDENCY.json'))
        if (previous!=ready.get('previous_bank_dependency') or previous['revision']!=V3_0_REVISION
                or previous['repo_id']!=repo
                or previous['prefix']!='/'.join(prefix.split('/')[:2])+'/bank-v3_0-2fd28e00'
                or len(previous['episodes'])!=80):
            raise ValueError('EXACT_FIXED_PREVIOUS_BANK_DEPENDENCY_REQUIRED')
    proofs = read(get('ARCHIVE_MEMBER_MANIFEST.json'))
    dependency = read(get('ORIGINAL_DEPENDENCY.json'))
    bank_admission = get('BANK_ADMISSION.json')
    qualification = get('receipts/CPU_QUALIFICATION.json')
    if digest(qualification) != ready['qualification_sha256']:
        raise ValueError('FIXED_REVISION_CPU_QUALIFICATION_CHANGED')
    if dependency['revision'] != OLD_REVISION or dependency['prefix'] != OLD_PREFIX:
        raise ValueError('LEGACY_FIXED_DEPENDENCY_CHANGED')
    extract_checked(get('native_dataset.tar.gz'), output / 'native', proofs['native_dataset']['member_sha256'], guard)
    manifest = read(output / 'native' / 'DATASET_MANIFEST.json')
    if digest(output / 'native' / 'DATASET_MANIFEST.json') != ready['manifest_sha256'] or digest(output / 'native' / 'statistics.json') != ready['statistics_sha256']:
        raise ValueError('DOWNLOADED_FROZEN_MANIFEST_OR_STATISTICS_CHANGED')
    if digest(bank_admission) != manifest['admission_sha256']:
        raise ValueError('FROZEN_ADMISSION_DENOMINATOR_CHANGED')
    old = read(old_native / 'DATASET_MANIFEST.json')
    if digest(old_native / 'DATASET_MANIFEST.json') != manifest['original_native_manifest_sha256']:
        raise ValueError('PREEXISTING_LEGACY_NATIVE_BANK_IDENTITY_CHANGED')
    if digest(old_native / 'statistics.json') != manifest['original_statistics_sha256']:
        raise ValueError('PREEXISTING_LEGACY_STATS_CHANGED')
    old_byid = {row['id']: row for row in old['episodes']}
    explicit = read(old_bindings)
    bindings = {}
    for row in manifest['episodes']:
        identifier = row['id']
        if row['origin'] == 'legacy':
            prior = old_byid[identifier]
            if prior['split'] != row['split'] or prior['samples_sha256'] != row['samples_sha256']:
                raise ValueError('LEGACY_IDENTITY_OR_SPLIT_CHANGED')
            folder = Path(explicit[identifier]).resolve(strict=True)
            if digest(folder / 'samples.json') != prior['samples_sha256']:
                raise ValueError('PREEXISTING_LOCAL_LEGACY_RAW_CHANGED')
        elif previous and identifier in previous['episodes']:
            folder=reused_raw(row,previous,previous_native,previous_bindings,proofs[identifier])
        else:
            extract_checked(get('episodes/' + identifier + '.tar.gz'), output / 'raw',
                            proofs[identifier]['member_sha256'], guard)
            folder = output / 'raw' / identifier
        bindings[identifier] = str(folder.resolve())
        if digest(output / 'native' / row['file']) != row['sha256']:
            raise ValueError('DOWNLOADED_NUMERIC_EPISODE_CHANGED')
    write(output / 'RAW_BINDINGS.private.json', bindings)
    receipt = dict(status='PI0294_FIXED_REVISION_DATA_MATERIALIZED_PROCESSOR_PENDING',
        fixture=False, repo_id=repo, revision=revision, prefix=prefix,
        manifest_sha256=ready['manifest_sha256'], statistics_sha256=ready['statistics_sha256'],
        old_raw_downloaded=0, legacy_local_bindings=sum(r['origin'] == 'legacy' for r in manifest['episodes']),
        new_raw_downloaded=sum(r['origin']=='new' and not (previous and r['id'] in previous['episodes']) for r in manifest['episodes']),
        previous_bank_local_bindings=sum(r['origin']=='new' and bool(previous and r['id'] in previous['episodes']) for r in manifest['episodes']),
        model_loads=0, integrations=0, actual_loader_qualification='PENDING')
    write(output / 'DOWNLOAD_RECEIPT.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'output', 'reservation'):
        p.add_argument('--' + name, required=True)
    for name in ('native', 'old-transfer', 'repo', 'qualification', 'delivery-manifest', 'old-native', 'old-bindings'):
        p.add_argument('--' + name)
    p.add_argument('--mode', choices=('publish', 'download'), default='publish')
    p.add_argument('--manifest')
    p.add_argument('--existing-token-file')
    p.add_argument('--previous-native')
    p.add_argument('--previous-bindings')
    args = p.parse_args()
    from .bank import validate_cpu_ticket
    ticket = validate_cpu_ticket(args.root, args.reservation, args.manifest)
    if args.mode == 'publish':
        if any(getattr(args, name) is None for name in ('native', 'old_transfer', 'repo', 'qualification')):
            p.error('publish requires --native --old-transfer --repo --qualification')
        result = publish(args.root, args.native, args.old_transfer, args.output, args.repo,
                         qualification=args.qualification,
                         output_budget_bytes=ticket['reservation']['output_bytes'], source_sha=ticket['source_sha'],
                         existing_token_file=args.existing_token_file)
    else:
        if any(getattr(args, name) is None for name in ('delivery_manifest', 'old_native', 'old_bindings')):
            p.error('download requires --delivery-manifest --old-native --old-bindings')
        result = download(args.delivery_manifest, args.output, args.old_native, args.old_bindings,
                          existing_token_file=args.existing_token_file,previous_native=args.previous_native,
                          previous_bindings=args.previous_bindings)
    print(json.dumps({k: result[k] for k in ('status', 'revision', 'prefix')}))


if __name__ == '__main__':
    main()
