"""Exact superseded payloads only; parent/data/small evidence are protected."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path

BASE=Path('/scratch/e1192548/Projects/go2-arx5/experiments')
OLD=BASE/'PI0277'
TARGETS=[
    (OLD/'R2/GRIP/main/training/checkpoint-3072','SUPERSEDED_NONCANDIDATE_RECOVERY',True),
    (OLD/'R2_FOCUS/FOCUS/main/training/checkpoint-3072','SUPERSEDED_NONCANDIDATE_RECOVERY',True),
    (OLD/'R2/GRIP/main/training/checkpoint-4096','RETIRED_VALID_FAIL_OPTIMIZER_ONLY_MODEL_RETAINED',False),
    (OLD/'R2/snapshots/GRIP512','VALID_FAIL_HOPPER_DUPLICATE_CORE_MODEL_PRESERVED',True),
    (OLD/'R2/snapshots/GRIP2048','VALID_FAIL_HOPPER_DUPLICATE_CORE_MODEL_PRESERVED',True),
]

def quota():
    p=Path('/home/svu/ccepbs/nushpc/quota_hw_report/hpfs_quota_raw.txt')
    rows=[r.split() for r in p.read_text().splitlines() if '/e1192548 ' in r]
    if len(rows)!=1:return dict(status='UNKNOWN_TRANSIENT_ADMIN_REPORT_REFRESH',source=str(p),report_mtime_ns=p.stat().st_mtime_ns)
    r=rows[0];return dict(used_bytes=int(r[4]),hard_bytes=int(r[5]),report_mtime_ns=p.stat().st_mtime_ns,source=str(p),raw=' '.join(r))

def preview():
    assert not subprocess.run(['/cm/shared/apps/pbspro/current/bin/qstat','-u','e1192548'],capture_output=True,text=True).stdout.strip(),'Do not retire alongside any owned PBS job'
    files=[]
    for folder,reason,model in TARGETS:
        assert folder.resolve().is_relative_to(OLD.resolve()) and not folder.is_symlink()
        names=['optimizer.pt','scheduler.pt','rng_state.pth'] if 'checkpoint-' in folder.name else []
        if model:names+=['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']
        for name in names:
            p=folder/name
            if not p.exists():continue
            assert not p.is_symlink() and p.stat().st_uid==os.getuid() and p.stat().st_nlink==1,p
            result=subprocess.run(['fuser',str(p)],capture_output=True,text=True);assert result.returncode==1,(p,result.stdout,result.stderr)
            files.append(dict(path=str(p),class_=reason,bytes=p.stat().st_size,allocated_bytes=p.stat().st_blocks*512,dev=p.stat().st_dev,inode=p.stat().st_ino,links=1))
    assert files
    donor=OLD/'R2_FOCUS/FOCUS/main/training/checkpoint-4096'
    for n in ['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','optimizer.pt','scheduler.pt','rng_state.pth']:assert (donor/n).is_file()
    return dict(status='EXACT_BOUNDED_RETIREMENT_PREVIEW',authority='Existing explicit user permission to retire scientifically discarded or superseded checkpoints; no broad cleanup',files=files,quota_before=quota(),logical_bytes=sum(x['bytes'] for x in files),allocated_bytes=sum(x['allocated_bytes'] for x in files),protected_parent=str(donor),small_evidence_deleted=False)

def main(execute):
    out=BASE/'PI0278/evidence';out.mkdir(parents=True,exist_ok=True);p=out/'STORAGE_RETIREMENT_PREVIEW.json';done=out/'STORAGE_RETIREMENT_DONE.json'
    if done.exists():print(done.read_text());return
    if execute and p.exists():
        v=json.loads(p.read_text())
        if all(not Path(r['path']).exists() for r in v['files']):
            assert all((folder/'PI0278_RETIRED_PAYLOAD.json').is_file() for folder,_,_ in TARGETS)
            v.update(status='DONE_METADATA_FINALIZATION_AFTER_TRANSIENT_QUOTA_REPORT_REFRESH',removed=v['files'],quota_after=quota(),quota_report_may_lag=True,actual_unlinked_unique_inode_allocated_bytes=v['allocated_bytes'],shared_free_before='UNOBSERVED_REPORT_FAILURE',shared_free_after=os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize)
            done.write_text(json.dumps(v,sort_keys=True)+'\n');print(json.dumps(v));return
    v=preview()
    if not execute:p.write_text(json.dumps(v,sort_keys=True)+'\n');print(json.dumps(v));return
    expected=json.loads(p.read_text());assert expected['files']==v['files'],'Inventory changed after preview'
    before=os.statvfs(BASE);removed=[]
    for row in v['files']:
        f=Path(row['path']);s=f.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_nlink)==(row['dev'],row['inode'],row['bytes'],1)
        f.unlink();removed.append(row)
    for folder,reason,model in TARGETS:
        marker=folder/'PI0278_RETIRED_PAYLOAD.json';marker.write_text(json.dumps(dict(status='MODEL_ONLY_NOT_RESUMABLE' if not model else 'METADATA_ONLY_PAYLOAD_PERMANENTLY_RETIRED',reason=reason,retirement_receipt=str(done)))+'\n')
    after=os.statvfs(BASE);v.update(status='DONE_EXACT_SUPERSEDED_PAYLOADS_PARENT_AND_EVIDENCE_PRESERVED',removed=removed,shared_free_before=before.f_bavail*before.f_frsize,shared_free_after=after.f_bavail*after.f_frsize,shared_free_delta=(after.f_bavail*after.f_frsize-before.f_bavail*before.f_frsize),quota_after=quota(),quota_report_may_lag=True,actual_unlinked_unique_inode_allocated_bytes=sum(r['allocated_bytes'] for r in removed))
    done.write_text(json.dumps(v,sort_keys=True)+'\n');print(json.dumps(v))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');main(p.parse_args().execute)
