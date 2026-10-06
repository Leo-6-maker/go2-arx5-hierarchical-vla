"""Stage the fixed public task data and reuse existing AGG weights only."""
import importlib.util,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
import requests
from run import sha,write
ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0276');CONTROL=ROOT.with_name('PI0273')
RELEASE='57fd8a9c53ce241c05a8d5b351034bed7ba56b14'

def read(p):return json.loads(Path(p).read_text())

def main():
    receipt=read(ROOT/'evidence/RESULT_ARCHIVE_training_data.json')
    assert receipt['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS' and receipt['part']=='training_data'
    assert receipt['repo_id']=='leoyuer/pi0246-gate-pass-campaign' and receipt['prefix']=='evidence/PI0276/maniskill-stackcube-20261006/training_data'
    session=requests.Session();session.trust_env=False
    for row in receipt['files']:
        name=row['path'];assert Path(name).name==name
        p=ROOT/'runtime'/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            temp=p.with_suffix(p.suffix+'.partial')
            with session.get(f"https://huggingface.co/{receipt['repo_id']}/resolve/{receipt['revision']}/{receipt['prefix']}/{name}",stream=True,timeout=(30,180)) as response:
                response.raise_for_status()
                with temp.open('wb') as stream:
                    for block in response.iter_content(8<<20):stream.write(block)
            assert temp.stat().st_size==row['bytes'] and sha(temp)==row['sha256'];os.replace(temp,p)
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    manifest=read(ROOT/'runtime/MEMBER_MANIFEST.json');remaining={r['path']:r for r in manifest['files']}
    assert manifest['tree_sha256']==receipt['tree_sha256']
    with tarfile.open(ROOT/'runtime/training_data.tar') as tar:
        for member in tar:
            assert member.isfile() and member.name in remaining;row=remaining.pop(member.name);relative=Path(member.name)
            assert not relative.is_absolute() and '..' not in relative.parts and member.size==row['bytes']
            # The data archive includes Corelab source wrappers; keep them separate from Hopper's task driver.
            target=ROOT/('corelab' if relative.parts[0] in ('code','authority','licenses') else '')/relative
            target.parent.mkdir(parents=True,exist_ok=True);assert target.resolve().is_relative_to(ROOT.resolve()) and not target.is_symlink()
            if not target.exists():
                temp=target.with_suffix(target.suffix+'.partial')
                with tar.extractfile(member) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,8<<20)
                assert sha(temp)==row['sha256'];os.replace(temp,target)
            assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256']
    assert not remaining
    donor=read(CONTROL/'DONOR_LOCAL_READBACK.json')
    assert donor['revision']=='f862abb26041c8e7d355470d6e9d48899f78b9ce' and donor['tree_sha256']=='0427b2be852f572ab4f3c4fa343cc56104acbbf0cca9db2e61d60abad3b15f66'
    for row in donor['fresh_model_hashes']:
        p=Path(donor['path'])/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    write(ROOT/'DONOR_LOCAL_READBACK.json',dict(donor,status='PASS_PI0276_REUSED_AGG4096_WEIGHT_IDENTITY',
        source_receipt_sha256=sha(CONTROL/'DONOR_LOCAL_READBACK.json'),optimizer_inherited=False))
    write(ROOT/'DONOR_PARAMETER_SCOPE.json',read(CONTROL/'DONOR_PARAMETER_SCOPE.json'))
    resolution=read(CONTROL/'evidence/PRECISION_CONTRACT_RESOLUTION.json')
    resolution.update(continuation_release=RELEASE,continuation_canonical=6000169298,predecessor_resolution_sha256=sha(CONTROL/'evidence/PRECISION_CONTRACT_RESOLUTION.json'))
    write(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json',resolution)
    mode=read(CONTROL/'design/BROAD_READOUT_R1.json')
    mode.update(schema='PI0276_TASK_NATIVE_S4_PHASE_CONTRACT_V1',release=RELEASE,arm='TASK',
        statistics_sha256=sha(ROOT/'contracts/TASK/source/dataset/statistics.json'),policy_frame='PI0276_MANISKILL_WORLD',tcp='PI0276_PANDA_HAND_TCP',
        inherited_readout_sha256=sha(CONTROL/'design/BROAD_READOUT_R1.json'),resolution_sha256=sha(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json'))
    write(ROOT/'design/TASK_READOUT_R1.json',mode)
    write(ROOT/'design/FITTING_CARD.json',dict(schema='PI0276_TASK_SCHEDULE_PROVENANCE_V1',
        dataset_card_sha256=sha(ROOT/'design/DATASET_CARD.json'),train_plan_sha256=sha(ROOT/'design/TRAIN_PLAN.json'),
        processed_inputs_sha256=sha(ROOT/'design/PROCESSED_INPUTS.json'),statistics_sha256=mode['statistics_sha256']))
    path=ROOT.with_name('PI0270')/'code/provision.py'
    spec=importlib.util.spec_from_file_location('pi0276_quota_helper',path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    quota_bytes=helper.quota_bytes
    quota=subprocess.run(['/usr/bin/hpc','quota'],capture_output=True,text=True,timeout=30);text=quota.stdout+quota.stderr;q=quota_bytes(text)
    assert q is not None,('USER_QUOTA_UNKNOWN',text)
    unique={}
    for p in ROOT.rglob('*'):
        if p.is_file() and not p.is_symlink():s=p.stat();unique[s.st_dev,s.st_ino]=s.st_size
    free=shutil.disk_usage(ROOT).free;required=4*donor['complete_trainer_bytes']+10*2**30;assert min(q,free)>=required
    write(ROOT/'STORAGE_ADMISSION.json',dict(available_bytes_at_freeze=min(q,free),owned_bytes_at_freeze=sum(unique.values()),
        parsed_quota_headroom_bytes=q,quota_report=text,df_available_bytes=free,required_bytes=required,
        formal_checkpoints=2,retained_canary_checkpoints=3,canary_payloads_retired_before_main=True,peak_checkpoint_slots=4,reserve_bytes=10*2**30))
    write(ROOT/'evidence/DATA_LOCAL_STAGE.json',dict(status='PASS_PI0276_FIXED_DATA_ALL_MEMBERS_LOCAL',revision=receipt['revision'],
        tree_sha256=receipt['tree_sha256'],members=receipt['members'],all_members_sha256=True,model_queries=0))

if __name__=='__main__':main()
