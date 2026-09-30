"""Publish the fixed R3 offline evaluation and verify its public bytes."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_url


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def run(root, token_file):
    root = root.resolve()
    inputs = [root / name for name in (
        'H1_R3_ADMITTED.json', 'S4_FIXED_EVAL_CARD.json',
        'S4_FIXED_DISPLAY_SELECTION.json', 'motion2_eval_cases.json',
        'O64_eval_cases.json', 'noise_manifest.json')]
    allowed = ('.json', '.jsonl', '.npz', '.mp4')
    for folder in ('eval', 'noise', 'model_bindings'):
        inputs.extend(path for path in (root / folder).rglob('*')
                      if path.is_file() and path.suffix in allowed)
    inputs = sorted(set(inputs), key=lambda path: path.relative_to(root).as_posix())
    assert all(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
               for path in inputs)
    assert all(path.suffix in allowed for path in inputs)
    for point in ('Base_0', 'Base_6000', 'G_0', 'G_6000'):
        assert (root / 'eval' / point / 'cpu_replay.json').exists()
        assert (root / 'eval' / point / 'semantic_replay.json').exists()
        assert len(list((root / 'eval' / point / 'cells').glob('*/record.json'))) == 440
    assert (root / 'eval' / 'media' / 'video_receipt.json').exists()
    rows = [dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
                 sha256=digest(path)) for path in inputs]
    tree = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    archive_root = root / 'archive' / 'R3_public'
    archive_root.mkdir(parents=True, exist_ok=True)
    manifest = archive_root / 'MEMBER_MANIFEST.json'
    save(manifest, dict(schema='PI0260_R3_FIXED_EVAL_PUBLIC_V1', files=rows,
                        tree_sha256=tree, count=len(rows), source_bytes=sum(row['bytes'] for row in rows),
                        rights='Owned offline outputs and rendered RGB; no private WBC, meshes, tokenizer, or credentials.'))
    archive = archive_root / 'PI0260_R3_fixed_eval.tar'
    with tarfile.open(archive, 'w') as stream:
        for path, row in zip(inputs, rows):
            info = tarfile.TarInfo(row['path'])
            info.size = row['bytes']
            info.mode = 0o644
            info.mtime = 0
            with path.open('rb') as source:
                stream.addfile(info, source)
    repo = 'leoyuer/pi0246-gate-pass-campaign'
    prefix = 'evidence/PI0260/R3-fixed-eval-20260930'
    api = HfApi(token=token_file.read_text().strip())
    assert api.model_info(repo).private is False
    committed = root / 'archive' / 'R3_upload_commit.json'
    if committed.exists():
        previous = json.loads(committed.read_text())
        assert previous['tree_sha256'] == tree
        revision = previous['revision']
    else:
        revision = api.upload_folder(repo_id=repo, folder_path=str(archive_root),
                                     path_in_repo=prefix, commit_message='PI0260 R3 fixed offline evaluation and video').oid
        save(committed, dict(repo_id=repo, prefix=prefix, revision=revision, tree_sha256=tree))
    session = requests.Session()
    session.trust_env = False
    readback = []
    for source in (manifest, archive):
        target = root / 'archive' / ('anonymous-' + source.name)
        with session.get(hf_hub_url(repo, filename=prefix + '/' + source.name, revision=revision),
                         stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            assert 'Authorization' not in response.request.headers
            with target.open('wb') as output:
                for block in response.iter_content(8 << 20):
                    output.write(block)
        assert target.stat().st_size == source.stat().st_size and digest(target) == digest(source)
        readback.append(dict(path=source.name, bytes=target.stat().st_size,
                             sha256=digest(target), anonymous=True))
    remaining = {row['path']: row for row in rows}
    with tarfile.open(root / 'archive' / ('anonymous-' + archive.name), 'r') as stream:
        for member in stream:
            assert member.isfile() and member.name in remaining
            expected = remaining.pop(member.name)
            content = stream.extractfile(member)
            value = hashlib.sha256()
            size = 0
            for block in iter(lambda: content.read(8 << 20), b''):
                value.update(block)
                size += len(block)
            assert size == expected['bytes'] and value.hexdigest() == expected['sha256']
    assert not remaining
    receipt = dict(status='PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK',
                   repo_id=repo, prefix=prefix, revision=revision, tree_sha256=tree,
                   members=len(rows), source_bytes=sum(row['bytes'] for row in rows),
                   files=readback, model_queries=0, physics_steps=0)
    save(root / 'archive' / 'R3_PUBLIC_RECEIPT.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.token_file)
