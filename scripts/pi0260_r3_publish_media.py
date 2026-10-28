"""Publish the fixed R3 video and report as directly viewable public files."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_url


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + '\n')


def run(root, token_file):
    video = root / 'eval/media/PI0260_R3_Base_G_before_after.mp4'
    report = root / 'MODEL_QUALITY_REPORT.md'
    receipt = json.loads((root / 'archive/R3_PUBLIC_RECEIPT.json').read_text())
    assert receipt['status'] == 'PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK'
    out = root / 'archive/R3_media_public'
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for source in (video, report):
        target = out / source.name
        if target.exists():
            assert sha(target) == sha(source)
        else:
            shutil.copy2(source, target)
        files.append(dict(path=target.name, bytes=target.stat().st_size, sha256=sha(target)))
    repo = 'leoyuer/pi0246-gate-pass-campaign'
    prefix = 'evidence/PI0260/R3-direct-media-20260930'
    api = HfApi(token=token_file.read_text().strip())
    assert api.model_info(repo).private is False
    commit_record = root / 'archive/R3_media_upload_commit.json'
    if commit_record.exists():
        prior = json.loads(commit_record.read_text())
        assert prior['files'] == files
        revision = prior['revision']
    else:
        revision = api.upload_folder(repo_id=repo, folder_path=str(out), path_in_repo=prefix,
                                     commit_message='PI0260 R3 fixed before-after video and quality report').oid
        save(commit_record, dict(repo_id=repo, prefix=prefix, revision=revision, files=files))
    session = requests.Session()
    session.trust_env = False
    for row in files:
        target = root / 'archive' / ('anonymous-media-' + row['path'])
        with session.get(hf_hub_url(repo, filename=prefix + '/' + row['path'], revision=revision),
                         stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            assert 'Authorization' not in response.request.headers
            with target.open('wb') as output:
                for block in response.iter_content(8 << 20):
                    output.write(block)
        assert target.stat().st_size == row['bytes'] and sha(target) == row['sha256']
    result = dict(status='PASS_PUBLIC_DIRECT_MEDIA_READBACK', repo_id=repo, prefix=prefix,
                  revision=revision, files=files, full_evidence_revision=receipt['revision'],
                  model_queries=0, physics_steps=0)
    save(root / 'archive/R3_MEDIA_RECEIPT.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.token_file)
