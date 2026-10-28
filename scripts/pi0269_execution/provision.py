"""New cards and small isolated working set; mature large sources stay in place."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0269')
OLD=ROOT.with_name('PI0268')
RELEASE='e77ad092406555645fb8c68ba34cc8594ef18ee6'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);content=json.dumps(value,indent=2,sort_keys=True)+'\n'
    if p.exists():assert p.read_text()==content
    else:p.write_text(content)
def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);content=source.read_bytes()
    if target.exists():assert target.read_bytes()==content
    else:target.write_bytes(content)

def main():
    for name in ('evidence','design','runtime','logs'):(ROOT/name).mkdir(parents=True,exist_ok=True)
    copy(OLD/'PAIRING_PLAN.json',ROOT/'PAIRING_PLAN.json')
    assert sha(ROOT/'PAIRING_PLAN.json')=='fca9a29e69e4b63ec35ab995bacc60a4867e4b86a11e2d7a3c6397d4d316f5ef'
    stats=OLD/'contracts/ROT/source/dataset/statistics.json'
    copy(stats,ROOT/'contracts/ROT/source/dataset/statistics.json')
    statsha=sha(stats);assert statsha=='0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
    copy(OLD/'design/FIT112_FIRST2048.json',ROOT/'design/FIT112_FIRST2048.json')
    for p in (OLD/'design/fit_fixtures').iterdir():copy(p,ROOT/'design/fit_fixtures'/p.name)
    cp=OLD/'ROT/main/training/checkpoint-2048';prior=read(OLD/'evidence/ROT/checkpoint-2048.full_readback.json')
    assert prior['status']=='PASS_FULL_FILE_READBACK' and len(prior['files'])==18
    assert prior['revision']=='97d8bf027f07241e2206b05e07597e4719c891e4'
    assert prior['tree_sha256']=='3db80d35d5e476fe5a90d540ff8959fb5edf3da43a988fde7f88ce9f54b9b147'
    donor=ROOT/'DONOR_LOCAL_READBACK.json'
    if donor.exists():assert read(donor)['files']==prior['files']
    else:
        for row in prior['files']:
            p=cp/row['path'];assert not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
            print('DONOR_LOCAL_HASH',row['path'],flush=True)
        assert read(cp/'statistics.json')==read(stats) and read(cp/'trainer_state.json')['global_step']==2048
        write(donor,dict(status='PASS_PI0269_FIXED_ROT2048_COMPLETE_TRAINER_BINDING',path=str(cp),
           revision=prior['revision'],repo_id=prior['repo_id'],prefix=prior['prefix'],tree_sha256=prior['tree_sha256'],
           files=prior['files'],complete_trainer_bytes=sum(r['bytes'] for r in prior['files']),
           reused_HF_full_readback_sha256=sha(OLD/'evidence/ROT/checkpoint-2048.full_readback.json'),fresh_local_full_hash=True))
    units=read(ROOT/'PAIRING_PLAN.json')['units'];assert len(units)==4096
    write(ROOT/'design/APPROVED_RUN_CARD.json',dict(schema='PI0269_MATCHED_TEMPORAL_CARD_V1',release=RELEASE,canonical=5964897978,
       launch_allowed=True,arms=['UNIFORM','PREFIX'],updates_per_arm=4096,save_steps=[2048,4096],quality_steps=[4096],
       source_plan_sha256=sha(ROOT/'PAIRING_PLAN.json'),statistics_sha256=statsha,donor_revision=prior['revision'],
       prefix_points=list(range(9)),prefix_transitions=list(range(8)),prefix_mass=.8,suffix_mass=.2,
       coefficients=dict(native=1,position=.4,increment=.1,rotation=.1,speed=.1),
       optimizer=dict(name='AdamW',learning_rate=1e-5,warmup=10,cosine_steps=4096,weight_decay=1e-5,clip=1.,accumulation=2),
       raw_update_cap=8480,raw_training_head_cap=160000,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    if not (ROOT/'STORAGE_ADMISSION.json').exists():
        unique={}
        for p in ROOT.rglob('*'):
            if p.is_file() and not p.is_symlink():s=p.stat();unique[s.st_dev,s.st_ino]=s.st_size
        available=int(subprocess.check_output(['/usr/bin/df','-B1','--output=avail','/scratch/e1192548'],text=True).splitlines()[-1])
        size=read(donor)['complete_trainer_bytes'];required=5*size+10*2**30
        assert available>=required,('FOUR_TRAINERS_PLUS_WRITE_OVERLAP_RESERVE',available,required)
        write(ROOT/'STORAGE_ADMISSION.json',dict(available_bytes_at_freeze=available,owned_bytes_at_freeze=sum(unique.values()),
             quota_scope_path='/scratch/e1192548',quota_report='UNKNOWN_current_dedicated_quota_report_not_refreshed',
             full_trainer_bytes=size,required_four_plus_one_overlap_and_10GiB=required,
             canary_plan='Compare and retire discarded passed canary payloads per arm before next arm; no broad cleanup'))
    print('PASS_DONOR_CARDS_ACTUAL_FILESET_SAVE_ADMISSION',flush=True)

if __name__=='__main__':main()
