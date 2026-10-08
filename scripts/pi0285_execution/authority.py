"""Current GO, finite completed-work caps and actual data/confirmation gates."""
import json,os,time
from functools import lru_cache
from pathlib import Path
from contract import ROOT,guard,sha
from r2 import admission,publication,phase_allowed,RELEASE


def load(root):
    assert Path(root).resolve()==ROOT.resolve()
    caps=guard()['new_caps']
    return dict(formal_cap=caps['formal_updates'],all_updates_cap=caps['all_completed_updates'],canary_cap=caps['discarded_canary_updates'],memory_cap=caps['memory_check_updates'],technical_cap=caps['completed_technical_rollback_updates'])


@lru_cache(maxsize=1)
def admitted(root,receipt_stamp,member_stamps):
    root=Path(root);data=admission(root)
    for name,h in data['bindings'].items():assert sha(root/'data'/name)==h,name
    return data


def allowed(root,kind,reserve_s=0):
    root=Path(root);limits=load(root);assert kind=='update';phase_allowed(kind,reserve_s)
    path=root/'data/R2_SCOPED_DATA_ADMISSION.json';data=json.loads(path.read_text())
    def stamp(p):
        s=p.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
    # ponytail: full readback once per process/file version, not each optimizer update.
    admitted(str(root),stamp(path),tuple((n,h,stamp(root/'data'/n)) for n,h in sorted(data['bindings'].items())))
    allocation=json.loads((root/'runtime/TRAINING_ALLOCATION.json').read_text())
    assert allocation['kind']=='PBS_H200' and os.environ.get('PBS_JOBID')==allocation['PBS_JOBID']
    assert time.time()+reserve_s<allocation['deadline_epoch']
    return limits


def completed_kind(rows,arm,phase,step,limits):
    assert len(rows)<limits['all_updates_cap']
    assert arm in ('PLACE4096','SHARED4096','REPAIR_A2048','REPAIR_B2048')
    duplicate=any(r['arm']==arm and r['phase']==phase and r['step']==step for r in rows)
    kind='technical' if duplicate else ('memory' if phase=='memory' else ('formal' if phase=='main' else 'canary'))
    cap={k:limits[k+'_cap'] for k in ('formal','canary','memory','technical')}
    assert sum(r['work_kind']==kind for r in rows)<cap[kind],kind
    if kind=='formal':
        stage='A' if arm in ('PLACE4096','REPAIR_A2048') else 'B'
        assert sum(r['work_kind']=='formal' and (r['arm'] in ('PLACE4096','REPAIR_A2048'))==(stage=='A') for r in rows)<6144
    return kind


def stage_card(root,path):
    root=Path(root);guard();v=json.loads(Path(path).read_text())
    publication(root);assert v['release']==RELEASE and v['canonical']==6065031954 and v['launch_allowed']
    arm=v['stage'];assert arm in ('PLACE4096','SHARED4096','REPAIR_A2048','REPAIR_B2048')
    repair=arm.startswith('REPAIR_')
    assert v['updates']==(2048 if repair else 4096) and v['warmup']==(103 if repair else 205)
    assert v['peak_LRs']==([5e-5,5e-6] if repair else [1e-4,1e-5])
    assert sha(v['plan_path'])==v['plan_sha256']
    plan=[json.loads(x) for x in Path(v['plan_path']).read_text().splitlines()]
    assert len(plan)==v['updates']*8 and [r['index'] for r in plan]==list(range(len(plan)))
    assert all(r['split']=='TRAIN' for r in plan)
    if arm in ('PLACE4096','REPAIR_A2048'):assert all(r['task']=='T1' for r in plan)
    else:
        for task in ('T1','T2'):assert sum(r['task']==task for r in plan)==len(plan)//2
        confirm=json.loads((root/'qualification/PLACE_V1_PROTECTED.json').read_text())
        assert confirm['status']=='PLACE_V1_SOURCE_QUALIFIED_AND_IMMUTABLY_PROTECTED' and confirm['CONFIRM_A_success']>=26 and confirm['CONFIRM_A_requested']==32
        assert sha(confirm['full_policy_manifest'])==confirm['full_policy_manifest_sha256']
        bank=json.loads((root/'data/SOURCE_BANK.json').read_text());assert bank['tasks']==['T1','T2']
        shared=set.intersection(*[{r['layout'] for r in bank['admitted_episodes'] if r['split']=='TRAIN' and r['task']==task} for task in ('T1','T2')]);assert len(shared)>=48
    if arm=='PLACE4096':assert v['parent'] is None and v['parent_global_update']==0
    else:
        parent=Path(v['parent']);assert parent.is_dir() and (parent/'optimizer.pt').is_file()
        for name,item in v['parent_files'].items():assert (parent/name).stat().st_size==item['bytes'] and sha(parent/name)==item['sha256']
        assert sha(v['parent_export'])==v['parent_export_sha256']
        assert json.loads((parent/'PI0282_RECOVERY.json').read_text())['cursor']==v['parent_phase_step']*8
    if repair:
        if arm=='REPAIR_A2048' and 'branch' in v:
            from r3_branch_card import validate_branch
            validate_branch(root,v,plan)
        else:
            proof=json.loads((root/'recovery'/('A' if arm=='REPAIR_A2048' else 'B')/'ADMISSION.json').read_text())
            assert proof['status']=='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC' and proof['valid_branches']>=24
            if arm=='REPAIR_B2048':assert all(proof['valid_per_task'][t]>=12 for t in ('T1','T2'))
            assert sum(r['source']=='corrected_suffix' for r in plan)==len(plan)//2
    return v


def self_check():
    limits=dict(formal_cap=12288,all_updates_cap=12512,canary_cap=64,memory_cap=32,technical_cap=128)
    assert completed_kind([],'PLACE4096','main',1,limits)=='formal'
    for kind,phase,cap in [('formal','main',6144),('canary','continuous',64),('memory','memory',32),('technical','main',128)]:
        rows=[dict(arm='PLACE4096',phase=phase,step=i+1,work_kind=kind) for i in range(cap)]
        try:completed_kind(rows,'PLACE4096',phase,1 if kind=='technical' else cap+1,limits)
        except AssertionError:pass
        else:raise AssertionError('Finite stage/cumulative cap missing: '+kind)
    print('PASS_STAGE_AND_TOTAL_COMPLETED_WORK_CAPS; model/optimizer/physics=0')


if __name__=='__main__':self_check()
