"""Download one complete episode plus native shard, then exercise frozen CPU loader."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import sys
import tarfile
import time

from huggingface_hub import hf_hub_download

from bind_consumer import bind
from deadline_guard import atomic_write, install_deadline, EXTERNAL_OUTPUT_UPPER_BYTES


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def extract_checked(path, target, expected):
    target.mkdir(parents=True)
    actual = {}
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
            assert member.name in expected and member.name not in actual
            output = target / member.name
            assert output.resolve().is_relative_to(target.resolve())
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open('xb') as handle:
                stream = archive.extractfile(member)
                for block in iter(lambda:stream.read(1 << 20),b''):
                    handle.write(block)
            actual[member.name] = sha(output)
            assert actual[member.name] == expected[member.name]
    assert actual == expected
    return len(actual)


def main(args):
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-3:]))
    begin = time.monotonic()
    receipt = read(args.transfer_manifest)
    root = Path(args.run_root)
    install_deadline(root,'readback_transfer')
    seal = read(root/'TEACHER_AND_LAYOUT_SEAL.json')
    assert seal['source_sha'] == receipt['source_sha']
    code = Path(args.frozen_code_root)
    assert all(sha(code/name)==expected for name,expected in seal['code_files'].items())
    ledger = read(root/'RESOURCE_ACCOUNTING.json')
    assert ledger['CPU_s']+ledger['unmetered_dependency_setup_CPU_s_upper']+600 < 48*3600
    output = root/'delivery'/'actual_download_readback'
    assert not output.exists(), 'Preserve prior downloaded evidence'
    output.mkdir(parents=True)
    token = Path(args.existing_token_file).read_text().strip()
    files = {row['path']:row for row in receipt['files']}
    planned = sum(row['bytes'] for row in files.values()
                  if row['path'].endswith(('READY.json','ARCHIVE_MEMBER_MANIFEST.json','native_dataset.tar.gz')))
    planned += max(row['bytes'] for row in files.values() if '/episodes/' in row['path'])
    current = sum(path.stat().st_size for path in root.rglob('*') if path.is_file() and not path.is_symlink())
    assert current+planned*8+EXTERNAL_OUTPUT_UPPER_BYTES < 40*1024**3
    assert __import__('shutil').disk_usage(root).free > planned*8+10*1024**3
    prefix, revision, repo = receipt['prefix'],receipt['revision'],receipt['repo_id']
    downloaded = {}

    def download(relative):
        name = prefix+'/'+relative
        row = files[name]
        path = Path(hf_hub_download(repo,filename=name,repo_type='dataset',revision=revision,
            token=token,local_dir=output/'blobs'))
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        downloaded[name] = dict(bytes=row['bytes'],sha256=row['sha256'])
        return path

    ready = read(download('READY.json'))
    assert ready['source_sha'] == receipt['source_sha'] and ready['data_revision'] == receipt['data_revision']
    assert ready['private'] and ready['NATIVE_DATALOADER_PASS']
    proofs = read(download('ARCHIVE_MEMBER_MANIFEST.json'))
    producer = root/'native_dataset'
    manifest = read(producer/'DATASET_MANIFEST.json')
    candidates = [row['id'] for row in manifest['episodes'] if row['split']=='TRAIN']
    assert candidates
    identifier = candidates[0]
    episode_archive = download('episodes/'+identifier+'.tar.gz')
    native_archive = download('native_dataset.tar.gz')
    episode_files = extract_checked(episode_archive,output/'raw',proofs[identifier]['member_sha256'])
    native_files = extract_checked(native_archive,output/'native',proofs['native_dataset']['member_sha256'])
    assert sha(output/'native/DATASET_MANIFEST.json') == sha(producer/'DATASET_MANIFEST.json')
    bind(output/'native',output/'raw',output/'consumer',[identifier])
    sys.path.insert(0,str(Path(args.frozen_code_root)/'scripts/pi0291_arx5_data'))
    from native_dataset import DatasetLoader
    from qualify_loader import check_batch, codec_all_windows
    dataset = DatasetLoader(output/'consumer',args.official_source,args.cosmos_processor_cache,'TRAIN')
    anchors = (0,80,160,320,480,600)
    batches = []
    for anchor in anchors:
        value = dataset[anchor]
        batches.append(dict(identity=value['identity'],actual_batch=check_batch(dataset.processor.collator([value['processed']]),1)))
    codec = codec_all_windows(output/'consumer',dataset.processor,('TRAIN',))
    assert codec['TRAIN']['full_H40_windows_actually_checked'] == 601
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = dict(status='PASS_ACTUAL_FIXED_REVISION_COMPLETE_EPISODE_AND_NATIVE_SHARD_READBACK',
        repo_id=repo,repo_type='dataset',private=True,revision=revision,prefix=prefix,
        source_sha=receipt['source_sha'],downloaded_files=downloaded,
        actual_complete_episode=identifier,raw_members=episode_files,native_members=native_files,
        consumer_rebinding=True,producer_manifest_sha256=sha(producer/'DATASET_MANIFEST.json'),
        original_statistics_sha256=sha(producer/'statistics.json'),statistics_refit=False,
        actual_frozen_project_loader='native_dataset.DatasetLoader',actual_original_RGB=True,
        actual_single_episode_batches=batches,actual_601_H40_codec=codec,
        readback_scope='One complete raw episode and full native export shard, not full raw dataset download; producer full-bank smoke remains separate',
        model_loads=0,neural_forward=0,training=0,physics_steps=0,new_render=0,
        CPU_s=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime,
        elapsed_s=time.monotonic()-begin,delivery_code_sha256=sha(__file__),Hopper='NOT_RUN_BY_THIS_SCRIPT')
    atomic_write(root/'evidence/HF_READBACK_RECEIPT.json',result)
    print(json.dumps(dict(status=result['status'],revision=revision,episode=identifier,CPU_s=result['CPU_s'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--transfer-manifest',required=True)
    parser.add_argument('--existing-token-file',required=True)
    parser.add_argument('--frozen-code-root',required=True)
    parser.add_argument('--official-source',required=True)
    parser.add_argument('--cosmos-processor-cache',required=True)
    args = parser.parse_args()
    main(args)
