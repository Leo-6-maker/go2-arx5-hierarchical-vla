"""Public fixed-revision PI0263 evidence seal; no model or private asset upload."""
import hashlib
import json
import tarfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_url
import sys
sys.path.insert(0, '/home/liuyu-nfs/Projects/go2-arx5/PI0261')
from pi0261_archive_evidence import sha, save

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0263')
REPO = 'leoyuer/pi0246-gate-pass-campaign'
PREFIX = 'evidence/PI0263/counterbalanced-template-fixed-20261001'


def run():
    replay = json.loads((ROOT / 'CPU_REPLAY.json').read_text())
    assert replay['status'] == 'PASS_INDEPENDENT_COMPUTATION_PATH_CPU_3456_REPLAY' and replay['rows'] == 3456
    required = ('UNION_EVAL_CARD.json','PAIRING_PLAN.json','PROCESSED_PLAN_AUDIT.json',
        'QUALIFICATION.json','MODEL_QUALITY_REPORT.md','LEGACY_PANEL_ROWS.csv','CROSSED_PANEL_ROWS.csv',
        'WORDING_FLIPS.csv','TARGET_CHOICE_DECOMPOSITION.csv','CPU_REPLAY.json','attempts.jsonl',
        'head_forwards.jsonl','PI0263_PROTOCOL.md','FINAL_CLOSEOUT.json','MAIN_TRAINING_AUDIT.json',
        'raw_training_heads.jsonl','raw_completed_updates.jsonl','DISPLAY_CARD.json')
    for arm in ('F','X'):
        for step in (1000,2000):
            receipt=ROOT/'evidence'/arm/f'checkpoint-{step}.full_readback.json'
            assert json.loads(receipt.read_text())['status']=='PASS_FULL_FILE_READBACK'
    assert (ROOT/'eval/media/video_receipt.json').is_file()
    assert all((ROOT / name).is_file() for name in required)
    assert len(list((ROOT / 'eval').glob('*/cells/*/record.json'))) == 3456
    allowed = {'.json', '.jsonl', '.npz', '.csv', '.md', '.py', '.log', '.mp4', '.pbs', '.OU', '.ER'}
    inputs={p for p in ROOT.iterdir() if p.is_file() and p.suffix in allowed}
    for folder in ('eval','evidence','model_bindings','training_evidence','code','logs'):
        inputs.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in allowed and '.partial.' not in p.name)
    files=sorted(inputs,key=lambda p:p.relative_to(ROOT).as_posix())
    assert all(not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve()) and '.partial.' not in p.name for p in files)
    assert not any(p.suffix in ('.pt', '.pth', '.safetensors') or 'token' in p.name.lower() for p in files)
    rows = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
    tree = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    archive_root = ROOT / 'archive/public'
    archive_root.mkdir(parents=True, exist_ok=True)
    manifest, archive = archive_root / 'MEMBER_MANIFEST.json', archive_root / 'PI0263_crossed_evidence.tar'
    save(manifest, dict(schema='PI0263_PUBLIC_CROSSED_EVIDENCE_V1', files=rows, tree_sha256=tree,
        members=len(rows), source_bytes=sum(r['bytes'] for r in rows),
        rights='Owned counterbalanced offline outputs, training audits and code; no model weights, private robot assets or credentials.'))
    with tarfile.open(archive, 'w') as stream:
        for p, row in zip(files, rows):
            info = tarfile.TarInfo(row['path'])
            info.size, info.mode, info.mtime = row['bytes'], 0o644, 0
            with p.open('rb') as source:
                stream.addfile(info, source)
    import shutil
    for direct in ('MODEL_QUALITY_REPORT.md','eval/media/PI0263_P_F_X_fixed_arrays.mp4'):
        shutil.copy2(ROOT/direct,archive_root/Path(direct).name)
    api = HfApi(token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip())
    assert api.model_info(REPO).private is False
    commit = ROOT / 'archive/UPLOAD_COMMIT.json'
    if commit.exists():
        previous = json.loads(commit.read_text())
        assert previous['tree_sha256'] == tree
        revision = previous['revision']
    else:
        revision = api.upload_folder(repo_id=REPO, folder_path=str(archive_root), path_in_repo=PREFIX,
            commit_message='PI0263 fixed F/X counterbalanced templates, 3456 rows, complete CPU/HF evidence').oid
        save(commit, dict(repo_id=REPO, prefix=PREFIX, revision=revision, tree_sha256=tree))
    session = requests.Session()
    session.trust_env = False
    checked = []
    for source in (manifest, archive, archive_root/'MODEL_QUALITY_REPORT.md',archive_root/'PI0263_P_F_X_fixed_arrays.mp4'):
        target = ROOT / 'archive' / ('anonymous-' + source.name)
        with session.get(hf_hub_url(REPO, filename=PREFIX + '/' + source.name, revision=revision),
                         stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            assert 'Authorization' not in response.request.headers
            with target.open('wb') as output:
                for block in response.iter_content(8 << 20):
                    output.write(block)
        assert target.stat().st_size == source.stat().st_size and sha(target) == sha(source)
        checked.append(dict(path=source.name, bytes=source.stat().st_size, sha256=sha(source), anonymous=True))
    remaining = {r['path']: r for r in rows}
    with tarfile.open(ROOT / 'archive' / ('anonymous-' + archive.name)) as stream:
        for member in stream:
            assert member.isfile() and member.name in remaining
            expected = remaining.pop(member.name)
            digest, size = hashlib.sha256(), 0
            with stream.extractfile(member) as content:
                for block in iter(lambda: content.read(8 << 20), b''):
                    digest.update(block); size += len(block)
            assert size == expected['bytes'] and digest.hexdigest() == expected['sha256']
    assert not remaining
    receipt = dict(status='PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK', repo_id=REPO,
        prefix=PREFIX, revision=revision, tree_sha256=tree, members=len(rows),
        source_bytes=sum(r['bytes'] for r in rows), files=checked, rows=3456, model_queries=0, physics_steps=0)
    save(ROOT / 'archive/PUBLIC_RECEIPT.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    run()
