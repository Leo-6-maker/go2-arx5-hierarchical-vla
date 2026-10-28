"""Finite owner closeout from actual receipts; no neural or physical work."""
import csv,hashlib,json,os,time
from pathlib import Path
from budget import CAPS,ROOT

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    out=ROOT/'final';out.mkdir(exist_ok=True)
    source=read(ROOT/'source/evidence/SOURCE_RESULT.json');assert source['status']=='FIXED_SOURCE_PANELS_EXECUTED'
    assert len(source['rows'])==14
    assert read(ROOT/'source/runtime/done.json')['returncode']==0
    assert read(ROOT/'source/evidence/POLICY_IMMUTABLE.json')['effective_weights_unchanged']
    assert (ROOT/'wbc/RECOVERY_DONE.json').exists()
    audited=[]
    for lane in ('ORIGINAL_BASE_JOINT','FK_OF_JOINT_COMMAND_REPLAY','EXPERT_DROID_BRIDGE_ROUNDTRIP'):
        for condition in range(10000,10004):
            r=read(ROOT/'audit'/('SOURCE-'+lane+'-'+str(condition)+'.json'))
            audited.append(r)
    wbc=[]
    for version in ('native','repair'):
        for condition in ('hold','monotonic','return'):
            name=version+'_'+condition+('-engineering-001' if version=='repair' else '')
            r=read(ROOT/'audit'/('WBC-'+name+'.json'))
            if version=='repair':
                binding=read(ROOT/'wbc'/name/'REPAIR_RUNTIME_BINDING.json');assert binding['bound'] and binding['calls']>0
            wbc.append(r)
    eligible={v:all(x['quality'] for x in wbc if x['fixture'].startswith(v+'_')) for v in ('native','repair')}
    assert not any(eligible.values()),'Qualification changed; follow the authorized downstream gate'
    account=read(ROOT/'evidence/ACCOUNTING_CORRECTION.json')
    assert all(account['effective_totals'][k]<=CAPS[k] for k in account['effective_totals'])
    assert account['closed_attempts']==account['reserved_attempts']==23
    old=read(ROOT/'execution/ADOPTION.json') if (ROOT/'execution/ADOPTION.json').exists() else None
    policy=read(ROOT/'source/evidence/POLICY_IMMUTABLE.json')
    sd=read(ROOT/'source/runtime/done.json')
    wd=[read(p) for p in (ROOT/'wbc').glob('*/DONE.json')]
    assert len(wd)==9 and all(x['returncode']==0 for x in wd)
    allocation_s=sd['elapsed_s']+sum(x['wall_s'] for x in wd)
    assert allocation_s<8*3600
    inventory=[];seen=set();allocated=0
    for p in sorted(ROOT.rglob('*')):
        if p.is_symlink():continue
        if not p.is_file() or 'final' in p.relative_to(ROOT).parts:continue
        st=p.stat();key=(st.st_dev,st.st_ino)
        if key not in seen:allocated+=st.st_blocks*512;seen.add(key)
        inventory.append(dict(path=str(p.relative_to(ROOT)),bytes=st.st_size,sha256=sha(p)))
    output_bytes=sum(x['bytes'] for x in inventory);assert output_bytes<40*(1<<30)
    write(out/'LOCAL_EVIDENCE_INVENTORY.json',dict(files=inventory,logical_bytes=output_bytes,allocated_unique_inode_bytes=allocated,symlinks_not_followed=True,HF_bulk='DEFERRED_NOT_ON_CRITICAL_PATH'))
    resources=dict(source_waited_CPU_s=sd['user_CPU_s']+sd['system_CPU_s'],wbc_waited_CPU_s=sum(x['child_cpu_user_s']+x['child_cpu_system_s'] for x in wd),
        conservative_GPU_allocation_h=allocation_s/3600,definition='Sum complete source wrapper and every WBC process lifetime, including initialization/IO; allocation bound, not GPU utilization.',
        CPU_audit=read(ROOT/'audit/FINAL_CPU_RESOURCES.json'),full_historical_CPU_total='UNOBSERVED_PARTIAL_RECEIPTS_ONLY',
        output_bytes=output_bytes,quota='UNKNOWN',missing_nonmodel_download_bytes=0,new_model_download_bytes=0)
    result=dict(status='COMPLETE_VALID_BASELINE_AND_WBC_QUALIFICATION_FAIL',authority='PI0281_CORELAB_NATIVE_READOUT_WBC_REPAIR_AND_ADAPTATION_GAP_GO_V1',
        release='360b412f55be8aec1568054d7d039ce7de2bc597',source=audited,wbc=wbc,wbc_eligible=eligible,
        target_expert=[dict(condition=i,status='NOT_EXECUTED_DEPENDENCY_FAIL',quality=None) for i in range(10000,10004)],
        target_recorded_model=[dict(condition=i,status='NOT_EXECUTED_DEPENDENCY_FAIL',quality=None) for i in range(10000,10004)],
        target_fresh_VLA='BLOCKED_TARGET_PRETRAINED_INPUT_CONTRACT_0_OF_8_HONEST_ROUTES',
        accounting=account,resources=resources,pristine_policy=policy,training=0,backward=0,optimizer=0,statistics_fits=0,target_policy_calls=0,new_teacher_calls=0,new_adaptation_data=0,
        selected='NONE',deployment='NONE',new_training_launch_allowed=False,next_state='STOP_FOR_MINIMUM_DECLARED_ADAPTATION_DECISION',predecessor=old)
    write(out/'FINAL_RESULT.json',result)
    with (out/'SOURCE_TASKS.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['lane','condition','status','quality'])
        for r in audited:writer.writerow([r[k] for k in ('lane','condition','status','quality')])
    with (out/'WBC_FIXTURES.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['fixture','full10s','quality','position_rms_m','position_max_m','SO3_rms_rad','SO3_max_rad','hard_penetration','last_progress_m','last_s_ref_m'])
        for r in wbc:
            p=r['position_final2s_m'];q=r['rotation_final2s_rad']
            writer.writerow([r['fixture'],r['full10s'],r['quality'],p['rms'] if p else None,p['max'] if p else None,q['rms'] if q else None,q['max'] if q else None,r['hard_penetration_max_SI'],r['progress']['last_m'],r['progress']['last_s_ref_m']])
    audit=[
        ('A0 authority/resources/pause','execution/ADOPTION.json; design/IMPLEMENTATION.json; evidence/CLOSING_RUNTIME.json'),
        ('A1 all240+32 saved outputs, ABI/no new inference','audit/SAVED_DUAL_READOUT.json; audit/OLD_OBJECT_FUNNEL.json'),
        ('A2a two actual fixtures','source/evidence/SOURCE_RESULT.json first2 rows'),
        ('A2b four fresh-feedback joint episodes; unchanged model/noise','source/evidence/POLICY_IMMUTABLE.json; audit/NEW_JOINT_READOUT-*.json'),
        ('A2c four commanded-q FK physical replays','audit/SOURCE-FK_OF_JOINT_COMMAND_REPLAY-*.json'),
        ('A2d four existing-expert bridge controls','audit/SOURCE-EXPERT_DROID_BRIDGE_ROUNDTRIP-*.json'),
        ('A2e conditional EEF rerun omitted, no semantic defect proven','audit/SAVED_DUAL_READOUT.json; original PI0280 DIRECT_EEF preserved0/4'),
        ('B1 actual private progress/actor/TCP chain','wbc/CPU_PROJECTION.json; wbc/PRIVATE_SOURCE_REVIEW.json; audit/WBC-*.json'),
        ('B2 one rule, actual consumption and six valid fixed fixtures','wbc/REPAIR_INSTALLATION_DEFECT.json; wbc/*engineering*/REPAIR_RUNTIME_BINDING.json; final/WBC_FIXTURES.csv'),
        ('B3 conditional target slots blocked; fresh targetVLA0','final/FINAL_RESULT.json target slots null; original input route audit'),
        ('Raw failed technical attempts, budgets and immutable predecessor','evidence/ACCOUNTING_CORRECTION.json; evidence/ACTUAL.jsonl; evidence/RESERVATIONS.jsonl'),
        ('Actual recordings/local integrity, no new render','source/evidence/SOURCE_VIDEOS.json12; wbc/WBC_VIDEOS.json9; final/LOCAL_EVIDENCE_INVENTORY.json'),
        ('Separate baseline/repair/gap/next proposal without training','final/REPORT.md; final/GAP_MATRIX.json')]
    write(out/'COMPLETION_AUDIT.json',dict(scope='Explicit PI0281 stages; conditional scientifically stopped stages remain null',requirements=[dict(requirement=k,evidence=v,status='VERIFIED_OR_PROTOCOL_CONDITIONAL_STOP') for k,v in audit],publication='PENDING_OWNER_CANONICAL_READBACK',selected='NONE',deployment='NONE'))
    print(json.dumps(dict(status=result['status'],wbc_eligible=eligible,counts=account['effective_totals'],resources=resources)))
if __name__=='__main__':main()
