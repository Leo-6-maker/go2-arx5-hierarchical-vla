"""Stage only the new admitted data package on Hopper at its fixed revision."""
import fcntl
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
import requests
from huggingface_hub import hf_hub_url

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0270')


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()


def main():
    source = ROOT/'evidence/TRAINING_DATA_ARCHIVE.json'; receipt = read(source)
    assert receipt['status'] == 'PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS'
    assert receipt['repo_id'] == 'leoyuer/pi0246-gate-pass-campaign'
    assert receipt['prefix'] == 'evidence/PI0270/learner-state-recovery-20261004/training-data'
    out = ROOT/'archive/training-data'; out.mkdir(parents=True, exist_ok=True)
    with (out/'stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = requests.Session(); session.trust_env = False
        assert {r['path'] for r in receipt['files']} == {'training-data.tar', 'MEMBER_MANIFEST.json'}
        for row in receipt['files']:
            target = out/row['path']
            if not target.exists():
                assert shutil.disk_usage(out).free >= row['bytes']+receipt['source_bytes']+10*2**30
                temp = target.with_suffix(target.suffix+'.partial')
                with session.get(hf_hub_url(receipt['repo_id'], receipt['prefix']+'/'+row['path'], revision=receipt['revision']),
                                 stream=True, timeout=(30,180)) as response:
                    response.raise_for_status(); assert 'Authorization' not in response.request.headers
                    with temp.open('wb') as stream:
                        for block in response.iter_content(8 << 20): stream.write(block)
                        stream.flush(); os.fsync(stream.fileno())
                assert temp.stat().st_size == row['bytes'] and sha(temp) == row['sha256']
                temp.replace(target)
            assert target.stat().st_size == row['bytes'] and sha(target) == row['sha256']
        manifest = read(out/'MEMBER_MANIFEST.json'); members = manifest['files']
        assert len(members) == manifest['members'] == receipt['members']
        assert hashlib.sha256(json.dumps(members, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == manifest['tree_sha256'] == receipt['tree_sha256']
        expected = {r['path']: r for r in members}; assert len(expected) == len(members)
        assert shutil.disk_usage(ROOT).free >= receipt['source_bytes']+10*2**30
        with tarfile.open(out/'training-data.tar') as tar:
            entries = tar.getmembers()
            assert len(entries) == len(expected) and {m.name for m in entries} == set(expected)
            for member in entries:
                assert member.isfile() and member.size == expected[member.name]['bytes']
                relative = Path(member.name)
                assert not relative.is_absolute() and '..' not in relative.parts
                target = ROOT/relative; assert target.resolve().is_relative_to(ROOT.resolve()) and not target.is_symlink()
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    assert target.parent.resolve().is_relative_to(ROOT.resolve())
                    temp = target.with_suffix(target.suffix+'.partial')
                    with tar.extractfile(member) as src, temp.open('wb') as dst:
                        shutil.copyfileobj(src, dst); dst.flush(); os.fsync(dst.fileno())
                    assert temp.stat().st_size == member.size and sha(temp) == expected[member.name]['sha256']
                    temp.replace(target)
                assert target.stat().st_size == member.size and sha(target) == expected[member.name]['sha256']
        assert sha(ROOT/'RECOVERY_MAPPING.json') == receipt['mapping_sha256']
        assert read(ROOT/'evidence/DATA_COVERAGE_ADMISSION.json')['status'] == 'PASS_ALL96_FIXED_NEW_SUPPORT_CELLS'
        assert read(ROOT/'evidence/NEWFIT_ADMISSION.json')['status'] == 'PASS_FIXED_NEWFIT192_AVAILABLE'
        result = dict(status='PASS_NEW_FIXED_DATA_PACKAGE_AND_ALL_MEMBERS_STAGED', revision=receipt['revision'],
                      repo_id=receipt['repo_id'], prefix=receipt['prefix'], tree_sha256=receipt['tree_sha256'],
                      source_receipt_sha256=sha(source), files=len(entries), source_bytes=receipt['source_bytes'],
                      mapping_sha256=sha(ROOT/'RECOVERY_MAPPING.json'), model_downloads=0, optimizer_downloads=0,
                      script_sha256=sha(__file__), actual_CPU_and_real_canary_still_required=True)
        p = ROOT/'evidence/TRAINING_DATA_LOCAL_READBACK.json'; text = json.dumps(result, indent=2, sort_keys=True)+'\n'
        if p.exists(): assert p.read_text() == text
        else: p.write_text(text)
        print(result['status'], len(entries), flush=True)


if __name__ == '__main__': main()
