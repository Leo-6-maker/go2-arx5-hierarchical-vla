"""Stage the fixed CORR2048 public inference assets under a fresh FUSION CPU role.

No model, processor, optimizer or tensor is loaded. This reuses the original
inference consumer, while keeping its historical guard and scope inactive.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import time

PROFILE = 'wm02-fusion-corr-asset-cpu'
SITE = 'CORE_FUSION_ASSET_CPU'
KIND = 'FUSION_CORR_ASSET'
REPO_ID = 'leoyuer/pi0294-public-artifacts'
DATA_REVISION = 'b37a3d959335b548abb45f87ad48a5186ec662d9'
PREFIX = 'PI0294/20261018T182956Z-localpc-pv1/DEC1_CORR2048_INFERENCE_DELTA_V1'
MANIFEST_SHA256 = 'ec46067a1179405d6c75081cfa333f0daa994f8e1c54bd3658eac0b5fb4eff61'
CHECKPOINT_SHA256 = '94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38'
FIRST = 'model-00001-of-00003.safetensors'
FIRST_SHA256 = 'cbcaea5ee88f1e0f1465043920a2647c67e7de17d24adfd1c477742a6168edec'
FIRST_BYTES = 4986649584
CHUNK_BYTES = 8 * 1024**2


def boundary(spec):
    if (time.time() >= spec['deadline_epoch']
            or any((Path(path) / 'STOP.json').exists()
                   for path in (spec['registered_root'], spec['campaign_root'],
                                str(Path(spec['fusion_binding']).parent),
                                str(Path(spec['pilot1_binding']).parent)))):
        raise TimeoutError('FUSION_CORR_ASSET_FINITE_DEADLINE_OR_STOP')


def stream_hash(path, *, check=lambda: None, accounting=None):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            check()
            block = stream.read(CHUNK_BYTES)
            if not block:
                break
            value.update(block)
            if accounting is not None:
                accounting['wrapper_final_readback_hashed_bytes'] += len(block)
    check()
    return value.hexdigest()


def identity(path):
    value = Path(path).stat()
    return dict(device=value.st_dev, inode=value.st_ino, mode=stat.S_IMODE(value.st_mode),
                bytes=value.st_size, mtime_ns=value.st_mtime_ns, uid=value.st_uid)


def retained_file_bytes(directory):
    """Account even partial download/cache files without reading their contents."""
    directory = Path(directory)
    return sum(path.stat().st_size for path in directory.rglob('*')
               if path.is_file() and not path.is_symlink()) if directory.is_dir() else 0


def copy_frozen(source, destination, *, expected_bytes, expected_sha256,
                check=lambda: None, accounting=None, chunk_bytes=CHUNK_BYTES):
    """Copy bytes exclusively, preserving source inode/mode and failed partials."""
    source, destination = Path(source), Path(destination)
    before = identity(source)
    if source.is_symlink() or not source.is_file() or before['bytes'] != expected_bytes:
        raise ValueError('EXACT_REGULAR_FROZEN_FIRST_SHARD_REQUIRED')
    value, copied = hashlib.sha256(), 0
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    check()
    with source.open('rb') as original, os.fdopen(os.open(destination, flags, 0o600), 'wb') as target:
        while True:
            check()
            block = original.read(chunk_bytes)
            if not block:
                break
            target.write(block)
            value.update(block)
            copied += len(block)
            if accounting is not None:
                accounting['copied_first_shard_bytes'] += len(block)
        target.flush()
        os.fsync(target.fileno())
        if copied != expected_bytes or value.hexdigest() != expected_sha256 or identity(source) != before:
            raise ValueError('FROZEN_FIRST_COPY_BYTES_HASH_OR_SOURCE_IDENTITY_CHANGED')
        # Only the new inode is made read-only. The original is never chmod'ed.
        os.fchmod(target.fileno(), 0o400)
        os.fsync(target.fileno())
    check()
    if (identity(destination)['device'], identity(destination)['inode']) == (before['device'], before['inode']):
        raise ValueError('NEW_PRIVATE_FIRST_SHARD_MUST_HAVE_DISTINCT_INODE')
    return dict(bytes=copied, sha256=value.hexdigest(), source_identity=before)


def qualified_root(spec, output):
    from go2arx5_vla.world_model.pilot1_storage import qualified_block_path
    root, output = Path(spec['registered_root']), Path(output)
    if (root.is_symlink() or output.is_symlink() or not root.is_dir() or not output.is_dir()
            or root.stat().st_uid != os.getuid() or output.stat().st_uid != os.getuid()
            or output.parent.resolve() != root.resolve()):
        raise PermissionError('FRESH_OWNED_ASSET_ROOT_AND_ADMITTED_OUTPUT_CHILD_REQUIRED')
    qualified_block_path(root, spec['output_storage_proof'], spec['output_storage_proof_sha256'])
    if root.stat().st_dev != output.stat().st_dev:
        raise PermissionError('ASSET_OUTPUT_MUST_STAY_ON_QUALIFIED_HDD')
    cache = root/'cache'
    if (cache.is_symlink() or cache.exists() and (
            not cache.is_dir() or cache.stat().st_uid != os.getuid() or any(cache.iterdir()))):
        raise PermissionError('FRESH_EMPTY_OWNED_HF_CACHE_REQUIRED_NO_EXTERNAL_SYMLINKS')
    return root


def validate_inputs(spec, output):
    expected = dict(profile=PROFILE, execution_site=SITE, kind=KIND, device='cpu',
                    physics=False, GR00T_updates=0, WBC_changes=False,
                    repo_id=REPO_ID, data_revision=DATA_REVISION, prefix=PREFIX,
                    manifest_sha256=MANIFEST_SHA256,
                    source_first_shard_sha256=FIRST_SHA256, source_first_shard_bytes=FIRST_BYTES)
    if (any(spec.get(key) != value for key, value in expected.items())
            or spec.get('physics') is not False or spec.get('WBC_changes') is not False
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise PermissionError('EXACT_FROZEN_CORR_CPU_ANONYMOUS_INFERENCE_ASSET_RECIPE_REQUIRED')
    boundary(spec)
    root = qualified_root(spec, output)
    manifest_path = Path(spec['manifest'])
    if manifest_path.is_symlink() or stream_hash(manifest_path, check=lambda: boundary(spec)) != MANIFEST_SHA256:
        raise ValueError('PINNED_LOCAL_CORR_MANIFEST_CHANGED')
    # Import functions only: never call the old DEC1 guard/run/ready/publish.
    from scripts.pi0294.dec1 import inference_assets as assets
    manifest = assets.validate_manifest(json.loads(manifest_path.read_text()))
    if manifest['original_checkpoint_manifest_sha256'] != CHECKPOINT_SHA256 or len(manifest['files']) != 13:
        raise ValueError('ORIGINAL_CORR2048_CHECKPOINT_AND_THIRTEEN_DELTA_MEMBERS_REQUIRED')
    for name in ('frozen-base', 'download', 'inference', 'CONSUMPTION.json', 'ASSET_STAGE_INTENT.private.json'):
        path = root/name
        if path.exists() or path.is_symlink():
            raise FileExistsError('RECONCILE_EXISTING_ASSET_STAGE_NO_RETRY:' + name)
    source = Path(spec['source_first_shard'])
    if source.is_symlink() or not source.is_file() or source.stat().st_size != FIRST_BYTES:
        raise ValueError('PINNED_EXISTING_E1_FIRST_SHARD_SIZE_REQUIRED')
    files = dict(manifest['files'], **{'INFERENCE_MANIFEST.json':
        dict(bytes=manifest_path.stat().st_size, sha256=MANIFEST_SHA256)})
    needed = FIRST_BYTES + sum(row['bytes'] for row in files.values()) + 1024**3
    space = os.statvfs(root)
    if space.f_bavail*space.f_frsize < needed:
        raise PermissionError('QUALIFIED_HDD_SPACE_FOR_FIRST_COPY_FIXED_DELTA_AND_MARGIN_REQUIRED')
    return root, assets, manifest, files


@contextmanager
def owned_hf_environment(root):
    """Keep Hub metadata/caches inside this worker, without persistent settings."""
    values = dict(HF_HOME=str(root/'cache/hf-home'), HF_HUB_CACHE=str(root/'cache/hub'),
                  HF_XET_CACHE=str(root/'cache/xet'), XDG_CACHE_HOME=str(root/'cache/xdg'),
                  HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
                  HF_HUB_DISABLE_PROGRESS_BARS='1', HF_HUB_DISABLE_XET='1')
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run(spec, output, *, download=None):
    """Read public fixed bytes once; preserve failures and every partial file."""
    from go2arx5_vla.world_model.storage import write_json
    from scripts.pi0292.durable import write
    output = Path(output)
    start, cpu = time.monotonic(), time.process_time()
    root, assets, manifest, files = validate_inputs(spec, output)
    check = lambda: boundary(spec)
    accounting = dict(copied_first_shard_bytes=0, wrapper_final_readback_hashed_bytes=0,
                      requested_file_bytes=0, returned_download_file_bytes=0,
                      download_file_calls_attempted=0, download_file_calls_completed=0)
    requests, failure, result = [], None, None
    copied = None
    write_json(root/'ASSET_STAGE_INTENT.private.json', dict(
        source_sha=spec['source_sha'], manifest_sha256=MANIFEST_SHA256,
        data_revision=DATA_REVISION, files=files, automatic_retry=False,
        asset_kind='INFERENCE_ONLY_NOT_FULL_RESUME', candidate='ORIGINAL_TRAIN_DEV_FIXED_S_CORR2048'))
    try:
        (root/'cache').mkdir(exist_ok=True)
        (root/'frozen-base').mkdir()
        frozen = root/'frozen-base'/FIRST
        copied = copy_frozen(spec['source_first_shard'], frozen, expected_bytes=FIRST_BYTES,
                             expected_sha256=FIRST_SHA256, check=check, accounting=accounting)
        check()
        transfer = dict(schema=assets.SCHEMA, repo_id=REPO_ID, prefix=PREFIX,
            status='PAYLOAD_COMMITTED_PENDING_INDEPENDENT_CONSUMER_READBACK',
            dependency=manifest['dependency'], files=files,
            data_revision=DATA_REVISION, manifest_sha256=MANIFEST_SHA256)
        bindings = dict(transfer=transfer, frozen_local_member=str(frozen),
                        public_anonymous_read=True, existing_token_file=None)
        requested = set()

        def owned_download(repo_id, filename, **kwargs):
            check()
            name = filename[len(PREFIX)+1:] if filename.startswith(PREFIX+'/') else None
            if (repo_id != REPO_ID or name not in files or name in requested
                    or kwargs != dict(repo_type='dataset', revision=DATA_REVISION, local_dir=root/'download')):
                raise PermissionError('ONLY_EXACT_FOURTEEN_FIXED_ANONYMOUS_MEMBER_REQUESTS_ALLOWED')
            requested.add(name)
            row = files[name]
            entry = dict(name=name, requested_bytes=row['bytes'], sha256=row['sha256'],
                         status='REQUEST_INTENT_NO_RETRY')
            requests.append(entry)
            accounting['download_file_calls_attempted'] += 1
            accounting['requested_file_bytes'] += row['bytes']
            write(output/'PROGRESS.private.json', dict(phase='DOWNLOAD_INTENT', accounting=accounting,
                                                       requests=requests, automatic_retry=False))
            path = Path(download(repo_id, filename, **kwargs, cache_dir=root/'cache', token=False))
            check()
            expected = root/'download'/PREFIX/name
            if (path.is_symlink() or path.resolve() != expected.resolve()
                    or not path.resolve().is_relative_to(root.resolve())
                    or not path.is_file() or path.stat().st_size != row['bytes']):
                raise ValueError('EXACT_PRIVATE_DOWNLOADED_MEMBER_PATH_AND_SIZE_REQUIRED')
            accounting['download_file_calls_completed'] += 1
            accounting['returned_download_file_bytes'] += path.stat().st_size
            entry['status'] = 'RETURNED_AWAITING_ORIGINAL_CONSUMER_HASH'
            return str(path)

        with owned_hf_environment(root):
            if download is None:
                from huggingface_hub import hf_hub_download
                download = hf_hub_download
            consumed = assets.consume(root, bindings, download=owned_download)
        if requested != set(files) or accounting['download_file_calls_completed'] != 14:
            raise ValueError('EXACT_THIRTEEN_DELTA_MEMBERS_PLUS_MANIFEST_REQUIRED')
        expected = {name: FIRST_SHA256 if name == FIRST else manifest['files'][name]['sha256']
                    for name in manifest['original_inference_sha256']}
        actual = {name: stream_hash(root/'inference'/name, check=check, accounting=accounting)
                  for name in expected}
        source_hash = stream_hash(spec['source_first_shard'], check=check, accounting=accounting)
        if (actual != expected or source_hash != FIRST_SHA256
                or identity(spec['source_first_shard']) != copied['source_identity']):
            raise ValueError('FINAL_NINE_INFERENCE_FILES_OR_ORIGINAL_FIRST_SHARD_CHANGED')
        for entry in requests:
            entry['status'] = 'ORIGINAL_CONSUMER_AND_FINAL_INFERENCE_READBACK_VERIFIED'
        result = dict(status='CORR2048_INFERENCE_ASSETS_STAGED_AND_REHASHED_NO_MODEL',
                      inference_root=str(root/'inference'), inference_files_sha256=actual,
                      source_first_shard_final_sha256=source_hash,
                      original_first_shard_inode_mode_and_bytes_retained=True,
                      original_consumer_receipt=consumed)
    except BaseException as error:
        failure = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        report = dict(result or {}, schema='WM02_FUSION_CORR_ASSET_RESULT_V1',
            status=(result['status'] if failure is None and result is not None
                    else 'FAILED_OR_PARTIAL_PRESERVED_NO_AUTOMATIC_RETRY'), failure=failure,
            source_sha=spec['source_sha'], repo_id=REPO_ID, prefix=PREFIX, data_revision=DATA_REVISION,
            manifest_sha256=MANIFEST_SHA256, original_checkpoint_manifest_sha256=CHECKPOINT_SHA256,
            source_first_shard=spec['source_first_shard'], first_shard_copy=copied,
            accounting=accounting, member_requests=requests,
            retained_download_tree_logical_bytes=retained_file_bytes(root/'download'),
            retained_cache_tree_logical_bytes=retained_file_bytes(root/'cache'),
            retained_frozen_base_tree_logical_bytes=retained_file_bytes(root/'frozen-base'),
            logical_file_calls_are_not_HTTP_requests=True, network_wire_bytes=None,
            returned_bytes_are_materialized_file_bytes_not_measured_wire_bytes=True,
            hash_accounting_scope='WRAPPER_FINAL_NINE_MEMBER_AND_ORIGINAL_SOURCE_ONLY_EXCLUDES_CONSUMER_AND_COPY_HASHES',
            eligibility='REQUIRES_SUCCESSFUL_NEW_FUSION_STAGE_RESULT_AND_CLOSED_WORKER_NOT_CONSUMPTION_ALONE',
            metadata_changes=manifest['metadata_changes'], full_resume=False,
            model_loads=0, processor_loads=0, VLA_inferences=0, native_integrations=0,
            optimizer_updates=0, GR00T_updates=0, GPU_calls=0, automatic_retry=False,
            preserved_failures_and_partial_files=True, old_E1_frozen_condition_not_replaced=True,
            model_and_processor='OWN_CORR_MANIFEST_WITH_EXPLICIT_PUBLIC_COSMOS_NAME_RELOCATION',
            task_transfer_gain=False, CPU_s=time.process_time()-cpu, wall_s=time.monotonic()-start,
            CPU_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(output/'RESULT.private.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(args.spec, args.output.name, PROFILE, args.output, gpu=False, physics=False) as spec:
        result = run(spec, args.output)
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
