"""Materialize only the new fixed dataset, preserving the closed initializer."""
import hashlib,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
import requests
ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0274');CONTROL=ROOT.with_name('PI0273')
REV='a786d91e501d241eb8af29d9666f6b8fd3c7a6be';PREFIX='evidence/PI0274/pose-equilibrium-20261005/data'
RELEASE='495b17078cb57fdbd47d8257d33202c57beb058e'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(v,indent=2,sort_keys=True)+'\n'
    if p.exists():assert p.read_text()==text,str(p)
    else:p.write_text(text)
def main():
    session=requests.Session();session.trust_env=False
    specs={'MEMBER_MANIFEST.json':(1619538,'0578ea0ed985d0489c0695cc2623d625e2a50ec295d29b5bc72f3f24797a3c7f'),'data.tar':(114534400,'6ec47262722ba4f994a5e5c2151c7235731b161ea95d8f8df86f19cca52a0382')}
    for name,(size,digest) in specs.items():
        p=ROOT/'runtime'/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            with session.get(f'https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/resolve/{REV}/{PREFIX}/{name}',stream=True,timeout=(30,180)) as r:
                r.raise_for_status()
                with p.with_suffix('.partial').open('wb') as f:
                    for block in r.iter_content(8<<20):f.write(block)
            assert p.with_suffix('.partial').stat().st_size==size and sha(p.with_suffix('.partial'))==digest;p.with_suffix('.partial').replace(p)
        assert p.stat().st_size==size and sha(p)==digest
    manifest=read(ROOT/'runtime/MEMBER_MANIFEST.json');rows=manifest['files'];remaining={r['path']:r for r in rows}
    with tarfile.open(ROOT/'runtime/data.tar') as tar:
        for member in tar:
            assert member.isfile() and member.name in remaining;row=remaining.pop(member.name);relative=Path(member.name)
            assert not relative.is_absolute() and '..' not in relative.parts and member.size==row['bytes']
            target=ROOT/('source' if relative.parts[0] in ('data','noise-field') else '')/relative;target.parent.mkdir(parents=True,exist_ok=True)
            assert target.resolve().is_relative_to(ROOT.resolve()) and not target.is_symlink()
            if not target.exists():
                with tar.extractfile(member) as src,target.with_suffix(target.suffix+'.partial').open('wb') as dst:shutil.copyfileobj(src,dst,8<<20)
                assert sha(target.with_suffix(target.suffix+'.partial'))==row['sha256'];target.with_suffix(target.suffix+'.partial').replace(target)
            assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256']
    assert not remaining
    write(ROOT/'evidence/DATA_LOCAL_STAGE.json',dict(status='PASS_FIXED_NEW_DATA_ALL_MEMBERS_LOCAL',revision=REV,prefix=PREFIX,members=len(rows),tree_sha256=manifest['tree_sha256'],all_members_sha256=True,new_model_queries=0))
    for name in ('DONOR_PARAMETER_SCOPE.json','DONOR_LOCAL_READBACK.json'):
        p=ROOT/name
        if not p.exists():shutil.copy2(CONTROL/name,p)
        assert p.read_bytes()==(CONTROL/name).read_bytes()
    donor=read(ROOT/'DONOR_LOCAL_READBACK.json');assert donor['revision']=='f862abb26041c8e7d355470d6e9d48899f78b9ce'
    for row in donor['fresh_model_hashes']:
        p=Path(donor['path'])/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    stats=ROOT/'contracts/ROT/source/dataset/statistics.json';stats.parent.mkdir(parents=True,exist_ok=True)
    if not stats.exists():shutil.copy2(CONTROL/'contracts/ROT/source/dataset/statistics.json',stats)
    assert sha(stats)=='0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
    resolution=read(CONTROL/'evidence/PRECISION_CONTRACT_RESOLUTION.json');resolution.update(continuation_release=RELEASE,continuation_canonical=5990552027,predecessor_resolution_sha256=sha(CONTROL/'evidence/PRECISION_CONTRACT_RESOLUTION.json'))
    write(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json',resolution)
    mode=read(CONTROL/'design/BROAD_READOUT_R1.json');mode.update(schema='PI0274_NATIVE_S4_READOUT_PHASE_CONTRACT_V1',release=RELEASE,arm='EQUIL',inherited_readout_sha256=sha(CONTROL/'design/BROAD_READOUT_R1.json'),resolution_sha256=sha(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json'))
    write(ROOT/'design/EQUIL_READOUT_R1.json',mode)
    sys.path.insert(0,str(ROOT.with_name('PI0270')/'code'));from provision import quota_bytes
    quota=subprocess.run(['/usr/bin/hpc','quota'],capture_output=True,text=True,timeout=30);text=quota.stdout+quota.stderr;q=quota_bytes(text);assert q is not None,('USER_QUOTA_UNKNOWN',text)
    df=shutil.disk_usage(ROOT).free;unique={}
    for p in ROOT.rglob('*'):
        if p.is_file() and not p.is_symlink():s=p.stat();unique[s.st_dev,s.st_ino]=s.st_size
    required=4*donor['complete_trainer_bytes']+10*2**30;assert min(q,df)>=required
    write(ROOT/'STORAGE_ADMISSION.json',dict(available_bytes_at_freeze=min(q,df),owned_bytes_at_freeze=sum(unique.values()),parsed_quota_headroom_bytes=q,df_available_bytes=df,quota_report=text,required_bytes=required,retained_canary_checkpoints=3,formal_checkpoints=2,peak_checkpoint_slots=4,overlap=1,reserve_bytes=10*2**30))
    print('PASS_NEW_DATA_STAGE_AGG_IDENTITY_AND_ACTUAL_QUOTA',flush=True)
if __name__=='__main__':main()
