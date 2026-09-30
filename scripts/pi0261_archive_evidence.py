"""Publicly seal the fixed PI0261 offline evidence without copying model weights."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_url

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0261')
REPO = 'leoyuer/pi0246-gate-pass-campaign'
PREFIX = 'evidence/PI0261/controlled-pairing-fixed-20260930'
ALLOWED = {'.json', '.jsonl', '.npz', '.mp4', '.md', '.csv'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    rendered = json.dumps(value, indent=2, sort_keys=True) + '\n'
    if path.exists():
        assert path.read_text() == rendered
    else:
        path.write_text(rendered)


def main(token_file):
    required = [ROOT / name for name in (
        'PAIRING_PLAN.json', 'FIXED_EVAL_CARD.json', 'PROCESSED_PAIR_BANK_AUDIT.json',
        'S0_EXPOSURE.json', 'SOURCE_FRAME_MANIFEST.json', 'MAIN_TRAINING_AUDIT.json',
        'MODEL_QUALITY_REPORT.md', 'SEMANTIC_BY_FAMILY.csv',
        'MATCHED_INPUT_EFFECT.csv', 'FINAL_CLOSEOUT.json')]
    for arm in ('C', 'P'):
        required += [ROOT / 'evidence' / arm / f'checkpoint-{step}.full_readback.json'
                     for step in (1000, 2000)]
        required += [ROOT / 'training_evidence' / arm / 'main/training' / f'{name}.jsonl'
                     for name in ('sample_audit', 'step_rng', 'S4_audit', 'full_loss')]
        required.append(ROOT / 'training_evidence' / arm / 'canary_compare.json')
    for point, count in (('Donor', 240), ('C2000', 680), ('P2000', 680)):
        required.append(ROOT / 'eval' / point / 'CPU_REPLAY.json')
        assert len(list((ROOT / 'eval' / point / 'cells').glob('*/record.json'))) == count
    required.append(ROOT / 'eval/media/video_receipt.json')
    assert all(path.is_file() for path in required)
    inputs = set(required)
    for folder in ('eval', 'evidence', 'model_bindings', 'training_evidence'):
        inputs.update(path for path in (ROOT / folder).rglob('*')
                      if path.is_file() and path.suffix in ALLOWED and '.partial.' not in path.name)
    inputs = sorted(inputs, key=lambda path: path.relative_to(ROOT).as_posix())
    assert all(not path.is_symlink() and path.resolve().is_relative_to(ROOT)
               and path.suffix in ALLOWED for path in inputs)
    rows = [dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                 sha256=sha(path)) for path in inputs]
    tree = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    archive_root = ROOT / 'archive/PI0261_public'
    archive_root.mkdir(parents=True, exist_ok=True)
    manifest = archive_root / 'MEMBER_MANIFEST.json'
    save(manifest, dict(schema='PI0261_FIXED_EVIDENCE_PUBLIC_V1', files=rows,
                        tree_sha256=tree, members=len(rows),
                        source_bytes=sum(row['bytes'] for row in rows),
                        rights='Owned offline outputs and rendered RGB; no private WBC, mesh, tokenizer or credential.'))
    archive = archive_root / 'PI0261_fixed_offline_evidence.tar'
    with tarfile.open(archive, 'w') as stream:
        for path, row in zip(inputs, rows):
            info = tarfile.TarInfo(row['path'])
            info.size, info.mode, info.mtime = row['bytes'], 0o644, 0
            with path.open('rb') as source:
                stream.addfile(info, source)
    api = HfApi(token=token_file.read_text().strip())
    assert api.model_info(REPO).private is False
    commit = ROOT / 'archive/PI0261_UPLOAD_COMMIT.json'
    if commit.exists():
        previous = json.loads(commit.read_text())
        assert previous['tree_sha256'] == tree
        revision = previous['revision']
    else:
        revision = api.upload_folder(repo_id=REPO, folder_path=str(archive_root),
            path_in_repo=PREFIX, commit_message='PI0261 fixed pairing evaluation and video').oid
        save(commit, dict(repo_id=REPO, prefix=PREFIX, revision=revision,
                          tree_sha256=tree))
    session = requests.Session()
    session.trust_env = False
    checked = []
    for source in (manifest, archive):
        target = ROOT / 'archive' / ('anonymous-' + source.name)
        with session.get(hf_hub_url(REPO, filename=PREFIX + '/' + source.name,
                                    revision=revision), stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            assert 'Authorization' not in response.request.headers
            with target.open('wb') as output:
                for block in response.iter_content(8 << 20):
                    output.write(block)
        assert target.stat().st_size == source.stat().st_size and sha(target) == sha(source)
        checked.append(dict(path=source.name, bytes=target.stat().st_size,
                            sha256=sha(target), anonymous=True))
    remaining = {row['path']: row for row in rows}
    with tarfile.open(ROOT / 'archive' / ('anonymous-' + archive.name)) as stream:
        for member in stream:
            assert member.isfile() and member.name in remaining
            expected = remaining.pop(member.name)
            digest, size = hashlib.sha256(), 0
            with stream.extractfile(member) as content:
                for block in iter(lambda: content.read(8 << 20), b''):
                    digest.update(block)
                    size += len(block)
            assert size == expected['bytes'] and digest.hexdigest() == expected['sha256']
    assert not remaining
    result = dict(status='PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK',
                  repo_id=REPO, prefix=PREFIX, revision=revision, tree_sha256=tree,
                  members=len(rows), source_bytes=sum(row['bytes'] for row in rows),
                  files=checked, model_queries=0, physics_steps=0)
    save(ROOT / 'archive/PI0261_PUBLIC_RECEIPT.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token-file', type=Path, required=True)
    main(parser.parse_args().token_file)
