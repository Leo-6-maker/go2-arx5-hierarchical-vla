"""PI0286 new counters and exact independent C0/C1 phase admission."""
import json,os,time
from functools import lru_cache
from pathlib import Path
from contract import ROOT,guard,sha
from r2 import phase_allowed

ARMS=('C0_REPLAY1024','C1_CORRECT1024')
RELEASE='eae6dbc1f9281b204a4e28e5f7c100d705837318'
CANONICAL=6072813602


def load(root):
    assert Path(root).resolve()==ROOT.resolve();guard()
    return dict(formal_cap=2048,all_updates_cap=2112,canary_cap=32,memory_cap=0,technical_cap=32)


def stamp(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


@lru_cache(maxsize=1)
def admitted(path,receipt_stamp):
    v=json.loads(Path(path).read_text())
    assert v['status']=='PASS_PI0286_ORIGINAL_EXPERT_TRAIN_INPUTS_AND_FROZEN_CODEC_ADOPTED'
    assert v['TRAIN_layouts']==list(range(285100,285164)) and v['supervised_tasks']==['T1']
    assert v['original_full_codec_status']=='FAIL_PRESERVED' and v['old_CONFIRM_or_DEV_supervision']==0
    for name,h in v['bindings'].items():assert sha(ROOT/'data'/name)==h,name
    for name,item in v['array_members'].items():assert stamp(ROOT/'data'/name)==item['stamp'],name
    return v


def data_admission(root):
    root=Path(root);p=root/'data/EXPERT_DATA_ADMISSION.json';v=json.loads(p.read_text())
    # ponytail: immutable member readback once per process/receipt version, not320 stats per update.
    return admitted(str(p),tuple(stamp(p)))


def allowed(root,kind,reserve_s=0):
    root=Path(root);limits=load(root);assert kind=='update';phase_allowed(kind,reserve_s);data_admission(root)
    allocation=json.loads((root/'runtime/TRAINING_ALLOCATION.json').read_text())
    assert allocation['kind']=='PBS_H200' and os.environ.get('PBS_JOBID')==allocation['PBS_JOBID']
    assert time.time()+reserve_s<allocation['deadline_epoch'];return limits


def completed_kind(rows,arm,phase,step,limits):
    assert arm in ARMS and phase in ('main','continuous','split') and 1<=step<=1024
    assert len(rows)<limits['all_updates_cap']
    duplicate=any(r['arm']==arm and r['phase']==phase and r['step']==step for r in rows)
    kind='technical' if duplicate else ('formal' if phase=='main' else 'canary')
    assert sum(r['work_kind']==kind for r in rows)<limits[kind+'_cap']
    if kind=='formal':assert sum(r['work_kind']=='formal' and r['arm']==arm for r in rows)<1024
    if kind=='canary':assert step<=8 and sum(r['work_kind']=='canary' and r['arm']==arm for r in rows)<16
    return kind


def stage_card(root,path):
    root=Path(root);guard();v=json.loads(Path(path).read_text())
    assert v['release']==RELEASE and v['canonical']==CANONICAL and v['launch_allowed']
    assert v['stage'] in ARMS and v['updates']==1024 and v['warmup']==52 and v['peak_LRs']==[5e-5,5e-6]
    assert v['parent_global_update']==6144 and v['parent_phase_step']==2048
    assert v['RNG_lineage_offset']==6144 and v['paired_RNG_sha256']==sha(root/'design/PAIRED_NATIVE_RNG.json')
    data_admission(root);parent=Path(v['parent'])
    p=root/'design/PARENT_READBACK.json';proof=json.loads(p.read_text());assert sha(p)==v['parent_readback_sha256']
    assert proof['status']=='PASS_PI0286_EXACT_SEALED_DOSE6144_FULL_PARENT' and proof['parent']==str(parent)
    assert proof['HF_revision']=='75432eb91a2c602930a9e1e979aaebbeaebff02b'
    assert proof['HF_tree']=='fb17b65effc3ae8dcbefa784adc8f7e6fab916821d29e4b2539e5be685ed8469'
    assert len(v['parent_files'])==14 and v['parent_files']==proof['files']
    for name,item in proof['files'].items():assert stamp(parent/name)==item['stamp'],name
    assert sha(v['parent_export'])==v['parent_export_sha256']==proof['export_sha256']
    assert v['parent_full_master_sha256']==proof['full_master_sha256']
    assert json.loads((parent/'PI0282_RECOVERY.json').read_text())['cursor']==16384
    assert sha(v['plan_path'])==v['plan_sha256']
    rows=[json.loads(line) for line in Path(v['plan_path']).read_text().splitlines()]
    assert len(rows)==8192 and [r['index'] for r in rows]==list(range(8192))
    assert all(r['task']=='T1' and r['split']=='TRAIN' and 285100<=r['layout']<=285163 for r in rows)
    if v['stage']=='C0_REPLAY1024':assert all(r['source']=='expert' for r in rows)
    else:
        admission=Path(v['correction_admission']);bank=json.loads(admission.read_text());assert sha(admission)==v['correction_admission_sha256']
        assert bank['status']=='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC' and bank['requested_branches']==32 and bank['valid_branches']>=24
        assert bank['parent_point']=='PI0285_REPAIR_A2048-2048' and bank['layouts']==list(range(285108,285124)) and bank['branch_times_seconds']==[3,12]
        assert sum(r['source']=='corrected_suffix' for r in rows)==4096
    return v


def self_check():
    limits=dict(formal_cap=2048,all_updates_cap=2112,canary_cap=32,memory_cap=0,technical_cap=32)
    assert completed_kind([],ARMS[0],'main',1,limits)=='formal'
    assert completed_kind([dict(arm=ARMS[0],phase='main',step=1,work_kind='formal')],ARMS[0],'main',1,limits)=='technical'
    for rows,arm,phase,step in [([dict(arm=ARMS[0],phase='continuous',step=i+1,work_kind='canary') for i in range(16)],ARMS[0],'split',1),([],ARMS[0],'memory',1),([],ARMS[0],'main',1025)]:
        try:completed_kind(rows,arm,phase,step,limits)
        except AssertionError:pass
        else:raise AssertionError('New arm/canary/phase bound omitted')
    return dict(status='PASS_PI0286_NEW_COUNTERS_ONLY_CPU',optimizer_updates=0,model_queries=0)


if __name__=='__main__':print(json.dumps(self_check()))
