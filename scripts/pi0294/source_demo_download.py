"""Download exactly the authorized fixed source-demo small pack; no simulation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-lock', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = json.loads(args.source_lock.read_text(encoding='utf-8'))
    wanted = [
        ('haosulab/ManiSkill_Demonstrations', 'demos/PickCube-v1/motionplanning/trajectory.h5'),
        ('haosulab/ManiSkill_Demonstrations', 'demos/PushCube-v1/motionplanning/trajectory.h5'),
        ('TianxingChen/RoboTwin2.0', 'dataset/place_a2b_left/arx-x5_clean_50.zip'),
        ('amandlek/mimicgen_datasets', 'source/pick_place.hdf5'),
    ]
    repositories = {row['repo']: row for row in lock['repositories']}
    selected = []
    for repo, path in wanted:
        row = repositories[repo]
        item = next(f for f in row['selected'] if f['path'] == path)
        selected.append(dict(repo=repo, revision=row['revision'], path=path,
                             bytes=item['size'], expected_sha256=item['lfs']['oid']))
    assert sum(row['bytes'] for row in selected) == 338566923
    receipt_path = root / 'DOWNLOAD_RECEIPT.json'
    manifest = dict(schema='SOURCE_DEMO_DOWNLOAD_RUN_V1',
                    authority='USER_EXPLICIT_FOUR_FILE_DOWNLOAD_AND_PLAN_PREPARATION_20261015',
                    source_lock_sha256=sha(args.source_lock), script_sha256=sha(__file__),
                    source_files=selected, started_epoch=time.time(),
                    physical_launch_allowed=False, new_model_queries=0, new_physics=0,
                    selected='NONE', deployment='NONE')
    if (root / 'RUN_MANIFEST.json').exists():
        old = json.loads((root / 'RUN_MANIFEST.json').read_text(encoding='utf-8'))
        if old['source_files'] != selected:
            raise ValueError('EXISTING_DOWNLOAD_RUN_IDENTITY_MISMATCH')
        if old['script_sha256'] != sha(__file__):
            write(root / 'ENGINEERING_SOURCE_BINDING.json', dict(
                original_manifest_sha256=sha(root / 'RUN_MANIFEST.json'),
                original_script_sha256=old['script_sha256'], current_script_sha256=sha(__file__),
                reason='CLI_LOCAL_DIR_AND_CACHE_DIR_EXCLUSION_REPAIRED_SOURCE_FILES_UNCHANGED',
                source_files_changed=False, physics=0, model_queries=0))
        manifest = old
    else:
        write(root / 'RUN_MANIFEST.json', manifest)
    env = dict(os.environ, HF_HOME=str(root / 'hf-home'),
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_XET='1',
               HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DOWNLOAD_TIMEOUT='120',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    hf = Path(sys.executable).with_name('hf.exe' if os.name == 'nt' else 'hf')
    rows = []
    for item in selected:
        local_dir = root / 'payloads' / item['repo'].replace('/', '__')
        target = local_dir / item['path']
        if not target.exists() or target.stat().st_size != item['bytes'] or sha(target) != item['expected_sha256']:
            log = root / (item['repo'].replace('/', '__') + '_' + target.suffix.lstrip('.') + '.download.log')
            with log.open('ab') as output:
                result = subprocess.run([str(hf), 'download', item['repo'], item['path'],
                    '--type', 'dataset', '--revision', item['revision'], '--local-dir', str(local_dir),
                    '--max-workers', '2', '--quiet'],
                    stdout=output, stderr=subprocess.STDOUT, env=env, timeout=1800)
            if result.returncode:
                write(receipt_path, dict(status='PARTIAL_DOWNLOAD_ENGINEERING_FAILURE',
                    verified_files=rows, failed=item, returncode=result.returncode))
                raise RuntimeError('FIXED_SMALLPACK_DOWNLOAD_FAILED_SEE_LOCAL_LOG')
        digest = sha(target)
        if target.stat().st_size != item['bytes'] or digest != item['expected_sha256']:
            raise ValueError('DOWNLOADED_PAYLOAD_SHA_OR_SIZE_MISMATCH')
        row = dict(item, relative_local_path=target.relative_to(root).as_posix(),
                   actual_sha256=digest, verified_bytes=target.stat().st_size)
        rows.append(row)
        write(receipt_path, dict(status='DOWNLOADING_FIXED_PACK', verified_files=rows))
        print(json.dumps(dict(status='FIXED_PAYLOAD_FULL_HASH_PASS', file=item['path'], bytes=item['bytes'])), flush=True)
    value = dict(schema='FIXED_SOURCE_DEMO_DOWNLOAD_RECEIPT_V1', status='PASS_FOUR_FIXED_PAYLOAD_SHA256',
                 files=rows, bytes=sum(row['bytes'] for row in rows), finished_epoch=time.time(),
                 source_lock_sha256=sha(args.source_lock), downloader_script_sha256=sha(__file__),
                 source_env='ISOLATED_LOCAL_PC_PREPARATION', new_physics=0, new_model_queries=0,
                 new_training_updates=0, raw_publication=False, selected='NONE', deployment='NONE')
    write(receipt_path, value)
    print(json.dumps(dict(status=value['status'], files=len(rows), bytes=value['bytes'])), flush=True)


if __name__ == '__main__':
    main()
