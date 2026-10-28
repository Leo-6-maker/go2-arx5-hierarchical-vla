"""Public full-Trainer upload and fixed-revision streamed readback for PI0261."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_url

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0261')
REPO = 'leoyuer/pi0246-gate-pass-campaign'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, result):
    rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if path.exists():
        assert path.read_text() == rendered
    else:
        temp = path.with_suffix(path.suffix + '.partial')
        temp.write_text(rendered)
        os.replace(temp, path)


def main(arm, step):
    checkpoint = ROOT / arm / 'main/training' / f'checkpoint-{step}'
    assert checkpoint.resolve().is_relative_to(ROOT.resolve())
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert (state['global_step'], state['max_steps']) == (step, 2000)
    required = ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json',
                'model.safetensors.index.json')
    assert all((checkpoint / name).is_file() for name in required)
    entries = []
    for path in sorted(checkpoint.rglob('*')):
        assert not path.is_symlink()
        if path.is_file():
            entries.append(dict(path=path.relative_to(checkpoint).as_posix(),
                                bytes=path.stat().st_size, sha256=sha(path)))
    assert len(entries) >= 16
    tree = hashlib.sha256()
    for row in entries:
        tree.update(f"PI0261/{arm}/checkpoint-{step}/{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    prefix = f'arms/PI0261/{arm}/checkpoint-{step}'
    evidence = ROOT / 'evidence' / arm
    evidence.mkdir(parents=True, exist_ok=True)
    receipt = evidence / f'checkpoint-{step}.archive_receipt.json'
    api = HfApi()
    assert api.model_info(REPO).private is False
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        assert previous['tree_sha256'] == tree.hexdigest() and previous['entries'] == entries
        revision = previous['revision']
    else:
        revision = api.upload_folder(repo_id=REPO, folder_path=str(checkpoint),
                                     path_in_repo=prefix, commit_message=f'PI0261 {arm}{step} full Trainer').oid
        save(receipt, dict(schema='PI0261_FULL_TRAINER_ARCHIVE_V1', status='UPLOADED',
                           arm=arm, step=step, repo_id=REPO, prefix=prefix, revision=revision,
                           tree_sha256=tree.hexdigest(), entries=entries,
                           total_bytes=sum(row['bytes'] for row in entries)))
    readback = evidence / f'checkpoint-{step}.full_readback.json'
    if readback.exists():
        prior = json.loads(readback.read_text())
        assert prior['status'] == 'PASS_FULL_FILE_READBACK' and prior['revision'] == revision
        print(json.dumps(dict(status=prior['status'], arm=arm, step=step, revision=revision)))
        return
    checked = []
    for row in entries:
        url = hf_hub_url(REPO, filename=prefix + '/' + row['path'], revision=revision)
        digest, size = hashlib.sha256(), 0
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            for block in response.iter_content(8 << 20):
                digest.update(block)
                size += len(block)
        assert size == row['bytes'] and digest.hexdigest() == row['sha256'], row['path']
        checked.append(row)
        print(json.dumps(dict(status='PI0261_HF_MEMBER_READBACK', arm=arm, step=step,
                              path=row['path'], bytes=size)), flush=True)
    result = dict(schema='PI0261_FULL_TRAINER_HF_READBACK_V1', status='PASS_FULL_FILE_READBACK',
                  arm=arm, step=step, repo_id=REPO, prefix=prefix, revision=revision,
                  tree_sha256=tree.hexdigest(), files=checked, complete_trainer=True)
    save(readback, result)
    print(json.dumps(dict(status=result['status'], arm=arm, step=step,
                          revision=revision, files=len(checked), tree=tree.hexdigest())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('C', 'P'), required=True)
    parser.add_argument('--step', type=int, choices=(1000, 2000), required=True)
    args = parser.parse_args()
    main(args.arm, args.step)
