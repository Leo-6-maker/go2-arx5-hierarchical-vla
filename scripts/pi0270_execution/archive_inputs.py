"""Publish only frozen collection inputs, with full public revision readback."""
import ast
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
from admission import BASE, ROOT, read, sha, write

os.environ.update(HF_HOME=str(ROOT/'runtime/hf_online'), HF_XET_CACHE=str(ROOT/'runtime/hf_online/xet'), HF_HUB_OFFLINE='0')
import requests
from huggingface_hub import HfApi, hf_hub_url

RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def main():
    seal = read(ROOT/'design/COLLECTION_FILESET.json')
    for row in seal['files']:
        assert sha(row['path']) == row['sha256'], row['path']
    for row in read(ROOT/'design/LABEL_FILESET.json')['files']:
        assert sha(row['path']) == row['sha256'], row['path']
    files = []
    # Only these immutable inputs: no active results/journals, caches or assets.
    for folder, prefix, suffixes in ((ROOT/'code', 'code', {'.py'}),
                                   (ROOT/'design', 'design', {'.json', '.md'}),
                                   (ROOT/'model_bindings', 'model-bindings', {'.json'})):
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in suffixes:
                assert not path.is_symlink()
                files.append((path, prefix+'/'+path.name))
    stats = ROOT/'contracts/ROT/source/dataset/statistics.json'
    assert sha(stats) == '0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
    files.append((stats, 'contracts/ROT/source/dataset/statistics.json'))
    for name in ('A0_PREDECESSOR_CLOSEOUT.json', 'COLLECTION_PLAN_CPU_CHECK.json',
                 'COLLECTION_CPU_ADMISSION.json', 'COLLECTION_NATIVE_CPU_CHECK.log',
                 'COLLECTION_NATIVE_CPU_CHECK.attempt01.log'):
        files.append((ROOT/'evidence'/name, 'evidence/'+name))
    card = read(ROOT/'design/COLLECTION_CARD.json')
    assert len(card['cases']) == len(card['noise']) == 192
    for case, noise in zip(card['cases'], card['noise']):
        path = RAW/'noise'/(case['case']+'.npz')
        assert str(path) == noise['file'] and sha(path) == noise['sha256']
        files.append((path, 'noise/'+path.name))
    assert len({name for _, name in files}) == len(files)
    assert all(p.is_file() and not p.is_symlink() and p.name not in ('token', 'stored_tokens') for p, _ in files)
    out = RAW/'archive/collection-inputs'; out.mkdir(parents=True, exist_ok=True)
    source = BASE/'PI0266/code/pi0266_report.py'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'archive_members')
    scope = dict(read=read, write=write, sha=sha, json=json)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    files, members, tree = scope['archive_members'](out, files)
    package = out/'collection-inputs.tar'
    if not package.exists():
        assert shutil.disk_usage(out).free >= sum(r['bytes'] for r in members)+10*2**30
        temp = package.with_suffix('.tar.partial')
        with tarfile.open(temp, 'w') as tar:
            for (path, name), row in zip(files, members):
                assert sha(path) == row['sha256']
                info = tarfile.TarInfo(name); info.size = row['bytes']; info.mode = 0o644; info.mtime = 0
                with path.open('rb') as stream: tar.addfile(info, stream)
        temp.replace(package)
    repo = 'leoyuer/pi0246-gate-pass-campaign'
    prefix = 'evidence/PI0270/learner-state-recovery-20261004/collection-inputs'
    api = HfApi(token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip())
    assert api.model_info(repo).private is False
    upload = RAW/'archive/COLLECTION_INPUT_UPLOAD.json'
    if upload.exists():
        prior = read(upload); assert prior['tree_sha256'] == tree; revision = prior['revision']
    else:
        revision = api.upload_folder(repo_id=repo, folder_path=str(out), path_in_repo=prefix,
                                     commit_message='PI0270 immutable learner collection inputs and actual noise').oid
        write(upload, dict(repo_id=repo, prefix=prefix, revision=revision, tree_sha256=tree))
    session = requests.Session(); session.trust_env = False; checked = []
    for path in sorted(p for p in out.iterdir() if p.is_file()):
        digest = hashlib.sha256(); size = 0
        with session.get(hf_hub_url(repo, prefix+'/'+path.name, revision=revision), stream=True, timeout=(30, 180)) as response:
            response.raise_for_status(); assert 'Authorization' not in response.request.headers
            for block in response.iter_content(8 << 20): digest.update(block); size += len(block)
        assert size == path.stat().st_size and digest.hexdigest() == sha(path)
        checked.append(dict(path=path.name, bytes=size, sha256=digest.hexdigest(), anonymous=True))
    remaining = {r['path']: r for r in members}
    with session.get(hf_hub_url(repo, prefix+'/'+package.name, revision=revision), stream=True, timeout=(30, 180)) as response:
        response.raise_for_status(); assert 'Authorization' not in response.request.headers
        response.raw.decode_content = True
        with tarfile.open(fileobj=response.raw, mode='r|') as tar:
            for member in tar:
                assert member.isfile() and member.name in remaining
                row = remaining.pop(member.name); digest = hashlib.sha256(); size = 0
                with tar.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(8 << 20), b''): digest.update(block); size += len(block)
                assert size == row['bytes'] and digest.hexdigest() == row['sha256'], member.name
    assert not remaining
    value = dict(status='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS', repo_id=repo, prefix=prefix,
                 revision=revision, tree_sha256=tree, members=len(members), source_bytes=sum(r['bytes'] for r in members),
                 files=checked, complete_study=False, active_evidence_included=False,
                 source_sha256=sha(__file__), selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    write(ROOT/'evidence/COLLECTION_INPUT_ARCHIVE.json', value); print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
