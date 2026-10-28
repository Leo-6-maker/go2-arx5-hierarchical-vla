"""Verify existing closed receipts and write the final report index; no remote/science calls."""
import hashlib,json,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'reports/PI0286';D=B/'evaluation'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entries(rows):
    if isinstance(rows,dict):return {k:(v['bytes'],v['sha256']) for k,v in rows.items()}
    return {v['path']:(v['bytes'],v['sha256']) for v in rows}

def main():
    r=load(D/'ENDPOINT_CLOSED_HF_READBACK.json');u=load(D/'ENDPOINT_HF_UPLOAD.json');p=load(D/'ENDPOINT_PACKAGE_READY.json')
    assert r['status']=='PASS_PUBLIC_FIXED_REVISION_FULL_FILE_READBACK'
    assert r['revision']==u['revision']=='8927fcf7c6bbdae5b8f85531162298c51fd9edba'
    e=entries(r['files']);assert e==entries(u['files'])==entries(p['files']) and len(e)==55 and sum(v[0] for v in e.values())==43196338
    idx=load(D/'ENDPOINT_LOCAL_DELIVERY_READBACK.json')
    for member in idx['members']:
        f=Path(member['local_path']);assert f.resolve().is_relative_to(ROOT.resolve()) and f.stat().st_size==member['bytes'] and sha(f)==member['sha256']
    core=load(D/'ACTUAL_CORE_SMALL_RECEIPTS_READBACK.json')
    for rel,v in core['files'].items():
        f=D/'core'/rel;assert f.stat().st_size==v['bytes'] and sha(f)==v['sha256'],rel
    models=load(B/'final/FINAL_MODEL_INVENTORY.json');assert models['status'].startswith('PASS_OWNER_BOTH_PINNED_FULL_TRAINER')
    for short,arm in [('C0','C0_REPLAY1024'),('C1','C1_CORRECT1024')]:
        a=B/'archival'/short;rd=load(a/'checkpoint-1024.full_readback.json');up=load(a/'checkpoint-1024.archive_receipt.json');cp=load(a/(arm+'-1024_TRAINING_CPU.json'))
        assert rd['arm']==up['arm']==arm and cp['point']==arm+'-1024' and rd['status']=='PASS_FULL_FILE_READBACK'
        assert entries(rd['files'])==entries(up['entries'])==entries(cp['entries']) and len(rd['files'])==14
        assert rd['revision']==up['revision']==models['arms'][arm]['revision'] and rd['tree_sha256']==models['arms'][arm]['tree_sha256']
    close=load(B/'final/OWNER_ACTUAL_COMPARISON_CLOSE_AND_NO_CANDIDATE.json');assert close['status']=='NO_DEVELOPMENT_QUALIFIED_CANDIDATE'
    assert close['actual_routes']['C0_REPLAY1024-1024']['TRAIN']=={'T1':2} and close['actual_routes']['C0_REPLAY1024-1024']['DEV'] is None
    assert close['actual_routes']['C1_CORRECT1024-1024']['TRAIN']=={'T1':8} and close['actual_routes']['C1_CORRECT1024-1024']['DEV']=={'T1':5}
    storage=[load(B/'final'/('PI0286_OWNED_STORAGE_'+h+'.json')) for h in ('CORE','HOP')];assert not any(v['errors'] for v in storage)
    allocated=sum(v['owned_unique_file_allocated_bytes']+v['directory_allocated_bytes'] for v in storage);assert allocated<100<<30
    url='https://huggingface.co/'+r['repo_id']+'/blob/'+r['revision']+'/'+r['prefix']+'/'
    videos={n:dict(bytes=e[n][0],sha256=e[n][1],url=url+n,conditions=c,actual_saved_two_view_RGB=True,NON_REALTIME=True) for n,c in [('C0_TRAIN.mp4',8),('C1_TRAIN.mp4',8),('C1_DEV.mp4',16)]}
    value=dict(status='COMPLETE_EXECUTION_VALID_ORIGINAL_QUALITY_FAIL_AND_REQUIRED_ARCHIVAL_CLOSED',checked_epoch=time.time(),original_release='eae6dbc1f9281b204a4e28e5f7c100d705837318',canonical_authority=6072813602,original_scientific_recipe_and_budget_unchanged=True,quality_pass=False,source_qualified_policy=False,actual_routes=close['actual_routes'],original_selection_result=close['status'],fresh_CONFIRM_C_episodes=0,models=models['arms'],evidence_archive=dict(repo=r['repo_id'],prefix=r['prefix'],revision=r['revision'],files=55,bytes=43196338,status=r['status'],readback_receipt_sha256=sha(D/'ENDPOINT_CLOSED_HF_READBACK.json')),actual_video=videos,source_arrays_retained_on_Core=True,full_Trainer_not_copied_to_Core_or_PC=True,science_budget_receipt_sha256=sha(B/'final/PI0286_CLOSED_FINAL_BUDGET.json'),post_archive_owned_storage=dict(allocated_bytes=allocated,GiB=allocated/(1<<30),cap_bytes=100<<30,headroom_bytes=(100<<30)-allocated,host_snapshot_epochs={v['host']:v['epoch'] for v in storage},personal_quota_headroom='UNKNOWN'),owner_local_copy_checks=dict(HF_receipt_members=len(idx['members']),source_CPU_receipt_members=len(core['files'])),new_scientific_calls_for_finalization=0,selected='NONE',deployment='NONE')
    out=B/'final/FINAL_DELIVERY_MANIFEST.json'
    if out.exists():value['checked_epoch']=load(out)['checked_epoch'];assert load(out)==value
    else:out.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf8'))
    print(json.dumps(dict(status=value['status'],HF=r['revision'],files=len(e),local_receipts=len(core['files']),owned_GiB=allocated/(1<<30),manifest_sha256=sha(out))))

if __name__=='__main__':main()
