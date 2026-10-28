"""Publish the admitted fixed training data only; no models or active journals."""
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
    assert read(ROOT/'evidence/ALL_FIXED_SUPPORT_CPU_COMPLETE.json')['status']=='PASS_ALL24_FAMILIES_AND_ALL6784_CODEC_PROPOSALS'
    assert read(ROOT/'evidence/DATA_COVERAGE_ADMISSION.json')['status']=='PASS_ALL96_FIXED_NEW_SUPPORT_CELLS'
    assert read(ROOT/'evidence/NEWFIT_ADMISSION.json')['status']=='PASS_FIXED_NEWFIT192_AVAILABLE'
    mapping=read(ROOT/'design/RECOVERY_MAPPING.json');rows=mapping['rows'];assert len(rows)==4096
    assert mapping['original_plan_sha256']==sha(ROOT/'design/PAIRING_PLAN.json')
    assert mapping['data_admission_sha256']==sha(ROOT/'evidence/DATA_COVERAGE_ADMISSION.json')
    files=[]
    for source,name in [(ROOT/'design/PAIRING_PLAN.json','PAIRING_PLAN.json'),
                        (ROOT/'design/RECOVERY_MAPPING.json','RECOVERY_MAPPING.json'),
                        (ROOT/'design/NEW_FIT192.json','design/NEW_FIT192.json'),
                        (ROOT/'design/LABEL_PROPOSAL_CARD.json','design/LABEL_PROPOSAL_CARD.json'),
                        (ROOT/'contracts/ROT/source/dataset/statistics.json','contracts/ROT/source/dataset/statistics.json')]:
        files.append((source,name))
    assert sha(ROOT/'contracts/ROT/source/dataset/statistics.json')=='0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
    for name in ('A0_PREDECESSOR_CLOSEOUT.json','ALL_FIXED_SUPPORT_CPU_COMPLETE.json','DATA_COVERAGE_ADMISSION.json','NEWFIT_ADMISSION.json','COLLECTION_INPUT_ARCHIVE.json'):
        files.append((ROOT/'evidence'/name,'evidence/'+name))
    samples={}
    for row in rows:
        uid=row['source_uid'];p=Path(row['sample'])
        assert p.resolve().is_relative_to((RAW/'labels').resolve()) and not p.is_symlink()
        assert sha(p)==row['sample_sha256']
        if uid in samples:assert samples[uid]==(p,row['sample_sha256'])
        samples[uid]=(p,row['sample_sha256'])
    for uid,(p,digest) in sorted(samples.items()):
        assert not uid.startswith('/') and '..' not in Path(uid).parts
        files.append((p,'source/recovery/'+uid+'/sample.npz'))
    fit=read(ROOT/'design/NEW_FIT192.json');assert len(fit['rows'])==192
    assert fit['source_mapping_sha256']==sha(ROOT/'design/RECOVERY_MAPPING.json')
    for row in fit['rows']:
        p=Path(row['noise_file']);assert p.resolve().is_relative_to((RAW/'noise-newfit').resolve())
        assert sha(p)==row['noise_file_sha256'];files.append((p,'noise-newfit/'+p.name))
    assert len({name for _, name in files}) == len(files)
    assert all(p.is_file() and not p.is_symlink() and p.name not in ('token', 'stored_tokens') for p, _ in files)
    out = RAW/'archive/training-data'; out.mkdir(parents=True, exist_ok=True)
    source = BASE/'PI0266/code/pi0266_report.py'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'archive_members')
    scope = dict(read=read, write=write, sha=sha, json=json)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    files, members, tree = scope['archive_members'](out, files)
    package = out/'training-data.tar'
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
    prefix = 'evidence/PI0270/learner-state-recovery-20261004/training-data'
    api = HfApi(token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip())
    assert api.model_info(repo).private is False
    upload = RAW/'archive/TRAINING_DATA_UPLOAD.json'
    if upload.exists():
        prior = read(upload); assert prior['tree_sha256'] == tree; revision = prior['revision']
    else:
        revision = api.upload_folder(repo_id=repo, folder_path=str(out), path_in_repo=prefix,
                                     commit_message='PI0270 admitted fixed learner training data and recovery mapping').oid
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
                 unique_recovery_samples=len(samples), mapping_sha256=sha(ROOT/'design/RECOVERY_MAPPING.json'),
                 source_sha256=sha(__file__), selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    write(ROOT/'evidence/TRAINING_DATA_ARCHIVE.json', value); print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
